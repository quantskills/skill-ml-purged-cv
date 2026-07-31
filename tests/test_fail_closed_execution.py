from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from purged_kfold_validation import (
    EvaluationError,
    FactoryLifecycleError,
    InformationInterval,
    LeakageSafeEvaluator,
    MetricEvaluationError,
    MetricSpec,
    ModelSpec,
    PITSnapshot,
    PredictionShapeError,
    PurgedKFold,
    ValidationDataset,
)


class Estimator:
    def __init__(self, failure: str | None = None) -> None:
        self.failure = failure

    def fit(self, features: np.ndarray, targets: np.ndarray) -> Estimator:
        if self.failure == "fit":
            raise RuntimeError("raw training details")
        self.mean = float(targets.mean())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.failure == "predict":
            raise RuntimeError("raw prediction details")
        if self.failure == "shape":
            return np.zeros(max(0, features.shape[0] - 1))
        return np.full(features.shape[0], self.mean)


class Transformer:
    def __init__(self, failure: str) -> None:
        self.failure = failure

    def fit(self, features: np.ndarray, targets: np.ndarray) -> Transformer:
        if self.failure == "transformer-fit":
            raise RuntimeError("raw transformer fit details")
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.failure == "transform":
            raise RuntimeError("raw transform details")
        return features


def _dataset() -> ValidationDataset:
    sessions = np.array(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
        dtype="datetime64[D]",
    )
    return ValidationDataset(
        sample_ids=("s0", "s1", "s2", "s3"),
        session_axis=sessions,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        decision_times=sessions.astype("datetime64[ns]") + np.timedelta64(12, "h"),
        feature_availability=sessions,
        pit_snapshot=PITSnapshot(
            snapshot_id="fail-closed-fixture",
            source_digest="fail-closed-fixture-digest",
        ),
        features=np.arange(4, dtype=float).reshape(4, 1),
        targets=np.array([0.0, 0.0, 1.0, 1.0]),
    )


def _raising_factory() -> Estimator:
    raise RuntimeError("raw factory details")


def _raising_metric(actual: np.ndarray, predicted: np.ndarray) -> float:
    raise RuntimeError("raw metric details")


@pytest.mark.parametrize(
    ("stage", "error_type", "message"),
    [
        ("factory", FactoryLifecycleError, "estimator factory failed"),
        ("fit", EvaluationError, "estimator execution failed"),
        ("transformer-fit", EvaluationError, "transformation failed"),
        ("transform", EvaluationError, "transformation failed"),
        ("predict", EvaluationError, "estimator execution failed"),
        ("shape", PredictionShapeError, "predictions must be"),
        ("metric", MetricEvaluationError, "metric 'failing' failed"),
    ],
)
def test_every_execution_stage_fails_without_partial_evidence(
    stage: str, error_type: type[Exception], message: str
) -> None:
    evaluator_arguments: dict[str, Any] = {
        "splitter": PurgedKFold(n_splits=2),
        "estimator_factory": (
            _raising_factory if stage == "factory" else lambda: Estimator(stage)
        ),
        "model_spec": ModelSpec(name="failure-injection", version="1"),
    }
    if stage in {"transformer-fit", "transform"}:
        evaluator_arguments["transformer_factories"] = (lambda: Transformer(stage),)
    if stage == "metric":
        evaluator_arguments["metrics"] = (
            MetricSpec(name="failing", version="1", function=_raising_metric),
        )

    with pytest.raises(error_type, match=message) as error:
        LeakageSafeEvaluator(**evaluator_arguments).evaluate(_dataset())

    assert "raw" not in str(error.value)
