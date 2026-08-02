from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from purged_kfold_validation import (
    EvaluationProtocol,
    EvidenceChannel,
    HoldoutEvaluationError,
    HoldoutProtocolError,
    InformationInterval,
    LocalHoldoutStore,
    MetricSpec,
    ModelSpec,
    PITSnapshot,
    ReusedHoldoutError,
    TransformerSpec,
    ValidationDataset,
)


def _dataset(prefix: str, start: str, targets: tuple[float, ...]) -> ValidationDataset:
    sessions = np.arange(
        np.datetime64(start, "D"),
        np.datetime64(start, "D") + len(targets),
        dtype="datetime64[D]",
    )
    return ValidationDataset(
        sample_ids=tuple(f"{prefix}-{index}" for index in range(len(targets))),
        session_axis=sessions,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        decision_times=sessions.astype("datetime64[ns]") + np.timedelta64(12, "h"),
        feature_availability=sessions,
        pit_snapshot=PITSnapshot(
            snapshot_id=f"{prefix}-pit",
            source_digest=f"{prefix}-source",
        ),
        features=np.arange(len(targets), dtype=float).reshape(-1, 1),
        targets=np.asarray(targets),
    )


class RecordingCenterer:
    def __init__(self, fits: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self._fits = fits
        self._mean: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> RecordingCenterer:
        self._fits.append((features.copy(), targets.copy()))
        self._mean = features.mean(axis=0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        assert self._mean is not None
        return np.asarray(features - self._mean)


class RecordingMean:
    def __init__(self, fits: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self._fits = fits
        self._mean: float | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> RecordingMean:
        self._fits.append((features.copy(), targets.copy()))
        self._mean = float(targets.mean())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self._mean is not None
        return np.full(len(features), self._mean)


def _metric() -> MetricSpec:
    return MetricSpec(
        name="mse",
        version="1",
        function=lambda actual, predicted: float(np.mean((actual - predicted) ** 2)),
    )


def _protocol(
    training: ValidationDataset,
    holdout: ValidationDataset,
    model: ModelSpec,
    metric: MetricSpec,
    *,
    transformer_specs: tuple[TransformerSpec, ...] = (),
) -> EvaluationProtocol:
    return EvaluationProtocol.freeze(
        protocol_id="release-candidate-1",
        training_dataset=training,
        holdout_dataset=holdout,
        model_spec=model,
        transformer_specs=transformer_specs,
        metrics=(metric,),
        search_policy={"kind": "none", "frozen": True},
        split_spec_digest="a" * 64,
    )


def test_holdout_fits_only_on_training_and_persists_redacted_receipt(
    tmp_path: Path,
) -> None:
    training = _dataset("train", "2025-01-01", (0.0, 0.0, 1.0, 1.0))
    holdout = _dataset("holdout", "2025-02-01", (0.0, 1.0))
    model = ModelSpec(name="mean", version="1")
    metric = _metric()
    transformer_spec = TransformerSpec(name="center", version="1", code_digest="b" * 64)
    protocol = _protocol(
        training,
        holdout,
        model,
        metric,
        transformer_specs=(transformer_spec,),
    )
    transformer_fits: list[tuple[np.ndarray, np.ndarray]] = []
    estimator_fits: list[tuple[np.ndarray, np.ndarray]] = []

    evaluation = LocalHoldoutStore(tmp_path / "store").evaluate_once(
        protocol,
        training_dataset=training,
        holdout_dataset=holdout,
        estimator_factory=lambda: RecordingMean(estimator_fits),
        model_spec=model,
        transformer_factories=(lambda: RecordingCenterer(transformer_fits),),
        transformer_specs=(transformer_spec,),
        metrics=(metric,),
    )

    assert len(transformer_fits) == 1
    assert len(estimator_fits) == 1
    assert transformer_fits[0][1].tolist() == training.targets.tolist()
    assert estimator_fits[0][1].tolist() == training.targets.tolist()
    assert evaluation.result.evidence_channel is EvidenceChannel.HOLDOUT_CONFIRMATION
    assert evaluation.result.ledger.sample_ids == holdout.sample_ids
    assert evaluation.receipt.metric_values == {"mse@1": 0.25}

    receipt_path = tmp_path / "store" / "receipts" / f"{evaluation.receipt.digest}.json"
    persisted = receipt_path.read_text(encoding="utf-8")
    assert json.loads(persisted)["holdout_dataset_digest"] == holdout.digest
    assert "prediction" not in persisted
    assert "target" not in persisted
    assert "features" not in persisted


def test_same_holdout_is_consumed_even_under_a_new_protocol(tmp_path: Path) -> None:
    training = _dataset("train", "2025-01-01", (0.0, 1.0))
    holdout = _dataset("holdout", "2025-02-01", (0.0, 1.0))
    model = ModelSpec(name="mean", version="1")
    metric = _metric()
    store = LocalHoldoutStore(tmp_path / "store")
    first = _protocol(training, holdout, model, metric)
    store.evaluate_once(
        first,
        training_dataset=training,
        holdout_dataset=holdout,
        estimator_factory=lambda: RecordingMean([]),
        model_spec=model,
        metrics=(metric,),
    )
    second = EvaluationProtocol.freeze(
        protocol_id="release-candidate-2",
        training_dataset=training,
        holdout_dataset=holdout,
        model_spec=model,
        metrics=(metric,),
        search_policy={"kind": "none"},
        split_spec_digest="c" * 64,
    )

    with pytest.raises(ReusedHoldoutError, match="already been consumed"):
        store.evaluate_once(
            second,
            training_dataset=training,
            holdout_dataset=holdout,
            estimator_factory=lambda: RecordingMean([]),
            model_spec=model,
            metrics=(metric,),
        )


def test_failed_attempt_consumes_holdout_before_estimator_fit(tmp_path: Path) -> None:
    training = _dataset("train", "2025-01-01", (0.0, 1.0))
    holdout = _dataset("holdout", "2025-02-01", (0.0, 1.0))
    model = ModelSpec(name="broken", version="1")
    metric = _metric()
    protocol = _protocol(training, holdout, model, metric)
    store = LocalHoldoutStore(tmp_path / "store")

    with pytest.raises(HoldoutEvaluationError, match="after consuming"):
        store.evaluate_once(
            protocol,
            training_dataset=training,
            holdout_dataset=holdout,
            estimator_factory=lambda: (_ for _ in ()).throw(RuntimeError("fit failed")),
            model_spec=model,
            metrics=(metric,),
        )
    with pytest.raises(ReusedHoldoutError):
        store.evaluate_once(
            protocol,
            training_dataset=training,
            holdout_dataset=holdout,
            estimator_factory=lambda: RecordingMean([]),
            model_spec=model,
            metrics=(metric,),
        )


def test_protocol_rejects_non_future_or_mismatched_holdout(tmp_path: Path) -> None:
    training = _dataset("train", "2025-02-01", (0.0, 1.0))
    earlier = _dataset("holdout", "2025-01-01", (0.0, 1.0))
    model = ModelSpec(name="mean", version="1")
    metric = _metric()

    with pytest.raises(HoldoutProtocolError, match="strictly after"):
        EvaluationProtocol.freeze(
            protocol_id="invalid",
            training_dataset=training,
            holdout_dataset=earlier,
            model_spec=model,
            metrics=(metric,),
            search_policy={"kind": "none"},
            split_spec_digest="a" * 64,
        )

    valid_holdout = _dataset("holdout", "2025-03-01", (0.0, 1.0))
    protocol = _protocol(training, valid_holdout, model, metric)
    other_holdout = _dataset("other", "2025-04-01", (0.0, 1.0))
    with pytest.raises(HoldoutProtocolError, match="not match"):
        LocalHoldoutStore(tmp_path / "store").evaluate_once(
            protocol,
            training_dataset=training,
            holdout_dataset=other_holdout,
            estimator_factory=lambda: RecordingMean([]),
            model_spec=model,
            metrics=(metric,),
        )


def test_protocol_rejects_training_information_reaching_the_holdout() -> None:
    training_sessions = np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[D]")
    training = ValidationDataset(
        sample_ids=("train-0", "train-1"),
        session_axis=training_sessions,
        sessions=training_sessions,
        information_intervals=(
            InformationInterval("2025-01-01", "2025-01-01"),
            InformationInterval("2025-01-02", "2025-03-01"),
        ),
        decision_times=training_sessions.astype("datetime64[ns]")
        + np.timedelta64(12, "h"),
        feature_availability=training_sessions,
        pit_snapshot=PITSnapshot(snapshot_id="train", source_digest="train-source"),
        features=np.array([[0.0], [1.0]]),
        targets=np.array([0.0, 1.0]),
    )
    holdout = _dataset("holdout", "2025-03-01", (0.0, 1.0))

    with pytest.raises(HoldoutProtocolError, match="information intervals"):
        _protocol(training, holdout, ModelSpec(name="mean", version="1"), _metric())


def test_protocol_deeply_freezes_the_search_policy() -> None:
    training = _dataset("train", "2025-01-01", (0.0, 1.0))
    holdout = _dataset("holdout", "2025-02-01", (0.0, 1.0))
    model = ModelSpec(name="mean", version="1")
    metric = _metric()
    policy = {"grid": {"alpha": [0.1, 1.0]}}
    protocol = EvaluationProtocol.freeze(
        protocol_id="immutable",
        training_dataset=training,
        holdout_dataset=holdout,
        model_spec=model,
        metrics=(metric,),
        search_policy=policy,
        split_spec_digest="a" * 64,
    )
    policy["grid"]["alpha"].append(10.0)

    assert protocol.search_policy["grid"]["alpha"] == (0.1, 1.0)
