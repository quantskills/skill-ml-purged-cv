"""Fixed estimator adapters for the trainable temporal-model benchmark."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from typing import Any

import numpy as np

from .domain import ModelSpec
from .errors import StrategyBenchmarkError
from .temporal_model_benchmark import TemporalModelCase


def _matrix(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise StrategyBenchmarkError("temporal model features must be a finite matrix")
    return values


class NumpyRidgeTemporalEstimator:
    """Fold-local standardized ridge over a fixed lag sequence."""

    def __init__(self, alpha: float = 1.0) -> None:
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise StrategyBenchmarkError("ridge alpha must be positive and finite")
        self.alpha = float(alpha)
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._coefficients: np.ndarray | None = None

    def fit(
        self, features: np.ndarray, targets: np.ndarray
    ) -> NumpyRidgeTemporalEstimator:
        values = _matrix(features)
        target = np.asarray(targets, dtype=np.float64)
        if target.shape != (len(values),) or not np.all(np.isfinite(target)):
            raise StrategyBenchmarkError("ridge targets must be finite and row-aligned")
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (values - mean) / scale
        design = np.column_stack((np.ones(len(values)), standardized))
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        self._mean = mean
        self._scale = scale
        self._coefficients = coefficients
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._mean is None or self._scale is None or self._coefficients is None:
            raise StrategyBenchmarkError("ridge estimator is not fitted")
        values = _matrix(features)
        standardized = (values - self._mean) / self._scale
        design = np.column_stack((np.ones(len(values)), standardized))
        return np.asarray(design @ self._coefficients, dtype=np.float64)


class LightGBMTemporalEstimator:
    """Deterministic fixed-tree LightGBM adapter loaded only when requested."""

    def __init__(
        self,
        *,
        n_estimators: int = 40,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        random_seed: int = 20260804,
    ) -> None:
        if n_estimators < 1 or max_depth < 1 or learning_rate <= 0.0:
            raise StrategyBenchmarkError("LightGBM parameters must be positive")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_seed = random_seed
        self._model: Any | None = None

    def fit(
        self, features: np.ndarray, targets: np.ndarray
    ) -> LightGBMTemporalEstimator:
        if find_spec("lightgbm") is None:
            raise StrategyBenchmarkError(
                "LightGBM temporal model requires the optional temporal-models dependency"
            )
        lightgbm: Any = import_module("lightgbm")
        model = lightgbm.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=7,
            random_state=self.random_seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(_matrix(features), np.asarray(targets, dtype=np.float64))
        self._model = model
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise StrategyBenchmarkError("LightGBM estimator is not fitted")
        return np.asarray(self._model.predict(_matrix(features)), dtype=np.float64)


class TorchLSTMTemporalEstimator:
    """Small deterministic CPU LSTM over the ordered lag sequence."""

    def __init__(
        self,
        *,
        hidden_size: int = 8,
        epochs: int = 2,
        batch_size: int = 1024,
        learning_rate: float = 0.005,
        random_seed: int = 20260804,
    ) -> None:
        if hidden_size < 1 or epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
            raise StrategyBenchmarkError("LSTM parameters must be positive")
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._lstm: Any | None = None
        self._head: Any | None = None

    def fit(
        self, features: np.ndarray, targets: np.ndarray
    ) -> TorchLSTMTemporalEstimator:
        if find_spec("torch") is None:
            raise StrategyBenchmarkError(
                "Torch LSTM temporal model requires the optional temporal-models dependency"
            )
        torch: Any = import_module("torch")
        nn: Any = import_module("torch.nn")
        values = _matrix(features)
        target = np.asarray(targets, dtype=np.float32)
        if target.shape != (len(values),) or not np.all(np.isfinite(target)):
            raise StrategyBenchmarkError("LSTM targets must be finite and row-aligned")
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        normalized = ((values - mean) / scale).astype(np.float32)
        torch.manual_seed(self.random_seed)
        torch.use_deterministic_algorithms(True)
        lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, batch_first=True)
        head = nn.Linear(self.hidden_size, 1)
        optimizer = torch.optim.Adam(
            list(lstm.parameters()) + list(head.parameters()), lr=self.learning_rate
        )
        loss_function = nn.MSELoss()
        sequence = torch.from_numpy(normalized[:, :, None])
        target_tensor = torch.from_numpy(target)
        lstm.train()
        head.train()
        for _ in range(self.epochs):
            for start in range(0, len(sequence), self.batch_size):
                stop = min(len(sequence), start + self.batch_size)
                optimizer.zero_grad(set_to_none=True)
                output, _ = lstm(sequence[start:stop])
                prediction = head(output[:, -1, :]).squeeze(-1)
                loss = loss_function(prediction, target_tensor[start:stop])
                loss.backward()
                optimizer.step()
        self._mean = mean
        self._scale = scale
        self._lstm = lstm
        self._head = head
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if (
            self._mean is None
            or self._scale is None
            or self._lstm is None
            or self._head is None
        ):
            raise StrategyBenchmarkError("LSTM estimator is not fitted")
        torch: Any = import_module("torch")
        values = ((_matrix(features) - self._mean) / self._scale).astype(np.float32)
        sequence = torch.from_numpy(values[:, :, None])
        self._lstm.eval()
        self._head.eval()
        with torch.no_grad():
            output, _ = self._lstm(sequence)
            prediction = self._head(output[:, -1, :]).squeeze(-1)
        return np.asarray(prediction.cpu().numpy(), dtype=np.float64)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def registered_temporal_model_cases(
    *,
    ridge_alpha: float = 1.0,
    lightgbm_estimators: int = 40,
    lstm_epochs: int = 2,
    random_seed: int = 20260804,
    require_optional: bool = True,
) -> tuple[TemporalModelCase, ...]:
    """Return the frozen linear, tree, and sequence model family."""

    if lightgbm_estimators < 1 or lstm_epochs < 1:
        raise StrategyBenchmarkError(
            "registered temporal model training counts must be positive"
        )
    if not np.isfinite(ridge_alpha) or ridge_alpha <= 0.0:
        raise StrategyBenchmarkError("registered ridge alpha must be positive")
    missing = [name for name in ("lightgbm", "torch") if find_spec(name) is None]
    if require_optional and missing:
        raise StrategyBenchmarkError(
            "temporal model family requires optional dependencies: "
            + ", ".join(missing)
            + "; install purged-kfold-validation[temporal-models]"
        )
    cases = [
        TemporalModelCase(
            estimator_factory=lambda: NumpyRidgeTemporalEstimator(ridge_alpha),
            model_spec=ModelSpec(
                name="numpy-ridge-lag-sequence",
                version="1",
                parameters={"alpha": ridge_alpha, "numpy": np.__version__},
            ),
        )
    ]
    if find_spec("lightgbm") is not None:
        cases.append(
            TemporalModelCase(
                estimator_factory=lambda: LightGBMTemporalEstimator(
                    n_estimators=lightgbm_estimators, random_seed=random_seed
                ),
                model_spec=ModelSpec(
                    name="lightgbm-lag-sequence",
                    version="1",
                    parameters={
                        "n_estimators": lightgbm_estimators,
                        "learning_rate": 0.05,
                        "max_depth": 3,
                        "random_seed": random_seed,
                        "lightgbm": _package_version("lightgbm"),
                    },
                ),
            )
        )
    if find_spec("torch") is not None:
        cases.append(
            TemporalModelCase(
                estimator_factory=lambda: TorchLSTMTemporalEstimator(
                    epochs=lstm_epochs, random_seed=random_seed
                ),
                model_spec=ModelSpec(
                    name="torch-lstm-lag-sequence",
                    version="1",
                    parameters={
                        "hidden_size": 8,
                        "epochs": lstm_epochs,
                        "batch_size": 1024,
                        "learning_rate": 0.005,
                        "random_seed": random_seed,
                        "torch": _package_version("torch"),
                    },
                ),
            )
        )
    return tuple(cases)
