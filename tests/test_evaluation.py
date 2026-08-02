from __future__ import annotations

import numpy as np
import pytest

from purged_kfold_validation import (
    EvaluationError,
    FactoryLifecycleError,
    InformationInterval,
    LeakageSafeEvaluator,
    MetricSpec,
    ModelSpec,
    PITSnapshot,
    PurgedKFold,
    TransformerSpec,
    ValidationDataset,
)


class RecordingCenterer:
    def __init__(self) -> None:
        self.fit_values: np.ndarray | None = None
        self.center: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> RecordingCenterer:
        self.fit_values = features.copy()
        self.center = features.mean(axis=0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        assert self.center is not None
        return np.asarray(features - self.center)


class RecordingMeanEstimator:
    def __init__(self) -> None:
        self.fit_targets: np.ndarray | None = None
        self.mean: float | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> RecordingMeanEstimator:
        self.fit_targets = targets.copy()
        self.mean = float(targets.mean())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.mean is not None
        return np.full(features.shape[0], self.mean)


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
            snapshot_id="evaluation-fixture",
            source_digest="evaluation-fixture-digest",
        ),
        features=np.array([[0.0], [2.0], [4.0], [6.0]]),
        targets=np.array([0.0, 0.0, 1.0, 1.0]),
    )


def test_user_gets_fold_local_oos_evidence_and_a_versioned_metric() -> None:
    transformers: list[RecordingCenterer] = []
    estimators: list[RecordingMeanEstimator] = []

    def transformer_factory() -> RecordingCenterer:
        transformer = RecordingCenterer()
        transformers.append(transformer)
        return transformer

    def estimator_factory() -> RecordingMeanEstimator:
        estimator = RecordingMeanEstimator()
        estimators.append(estimator)
        return estimator

    dataset = _dataset()
    evaluator = LeakageSafeEvaluator(
        splitter=PurgedKFold(n_splits=2),
        estimator_factory=estimator_factory,
        transformer_factories=(transformer_factory,),
        transformer_specs=(
            TransformerSpec(
                name="recording-centerer",
                version="1",
                code_digest="a" * 64,
            ),
        ),
        model_spec=ModelSpec(name="toy-mean", version="1"),
        metrics=(
            MetricSpec(
                name="mean-squared-error",
                version="1",
                function=lambda actual, predicted: float(
                    np.mean((actual - predicted) ** 2)
                ),
            ),
        ),
    )

    result = evaluator.evaluate(dataset)

    assert result.ledger.sample_ids == ("s0", "s1", "s2", "s3")
    assert result.ledger.predictions.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert result.metrics[0].name == "mean-squared-error"
    assert result.metrics[0].version == "1"
    assert result.metrics[0].overall == 1.0
    assert result.metrics[0].per_fold == (1.0, 1.0)
    assert result.metrics[0].observation_count == 4
    assert result.metrics[0].coverage == 1.0

    assert len({id(value) for value in transformers}) == 2
    assert len({id(value) for value in estimators}) == 2
    assert transformers[0].fit_values is not None
    assert transformers[0].fit_values[:, 0].tolist() == [4.0, 6.0]
    assert transformers[1].fit_values is not None
    assert transformers[1].fit_values[:, 0].tolist() == [0.0, 2.0]
    assert all(
        observation.dataset_digest == dataset.digest
        for observation in result.ledger.observations
    )


def test_formal_evaluation_rejects_a_factory_that_reuses_fitted_state() -> None:
    shared = RecordingMeanEstimator()
    evaluator = LeakageSafeEvaluator(
        splitter=PurgedKFold(n_splits=2),
        estimator_factory=lambda: shared,
        model_spec=ModelSpec(name="unsafe-shared-model", version="1"),
    )

    with pytest.raises(FactoryLifecycleError, match="reused an object"):
        evaluator.evaluate(_dataset())


def test_execution_failure_aborts_without_returning_partial_evidence() -> None:
    class FailingEstimator(RecordingMeanEstimator):
        def predict(self, features: np.ndarray) -> np.ndarray:
            raise RuntimeError("sensitive raw value must not escape")

    evaluator = LeakageSafeEvaluator(
        splitter=PurgedKFold(n_splits=2),
        estimator_factory=FailingEstimator,
        model_spec=ModelSpec(name="failing-model", version="1"),
    )

    with pytest.raises(
        EvaluationError, match="fold 0 estimator execution failed"
    ) as error:
        evaluator.evaluate(_dataset())

    assert "sensitive raw value" not in str(error.value)


def test_deterministic_components_produce_repeatable_oos_evidence() -> None:
    evaluator = LeakageSafeEvaluator(
        splitter=PurgedKFold(n_splits=2),
        estimator_factory=RecordingMeanEstimator,
        model_spec=ModelSpec(name="toy-mean", version="1"),
        metrics=(
            MetricSpec(
                name="mean-squared-error",
                version="1",
                function=lambda actual, predicted: float(
                    np.mean((actual - predicted) ** 2)
                ),
            ),
        ),
    )

    first = evaluator.evaluate(_dataset())
    repeated = evaluator.evaluate(_dataset())

    assert first.run_id == repeated.run_id
    assert first.ledger.digest == repeated.ledger.digest
    assert first.metrics == repeated.metrics
