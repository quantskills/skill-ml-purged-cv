from __future__ import annotations

import numpy as np
import pytest
from uuid import UUID

from purged_kfold_validation import (
    DatasetValidationError,
    EvidenceChannel,
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


class MeanEstimator:
    def fit(self, features: np.ndarray, targets: np.ndarray) -> MeanEstimator:
        self.mean = float(targets.mean())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.mean)


def _valid_dataset() -> ValidationDataset:
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
            snapshot_id="review-fixture",
            source_digest="review-fixture-digest",
        ),
        features=np.arange(4, dtype=float).reshape(4, 1),
        targets=np.array([0.0, 0.0, 1.0, 1.0]),
    )


def test_numpy_conversion_failures_remain_typed_and_redacted() -> None:
    dataset = _valid_dataset()
    with pytest.raises(DatasetValidationError, match="decision_times") as error:
        ValidationDataset(
            sample_ids=dataset.sample_ids,
            session_axis=dataset.session_axis,
            sessions=dataset.sessions,
            information_intervals=dataset.information_intervals,
            decision_times=("not-a-time",) * 4,
            feature_availability=dataset.feature_availability,
            pit_snapshot=dataset.pit_snapshot,
            features=dataset.features,
            targets=dataset.targets,
        )
    assert "not-a-time" not in str(error.value)


def test_unstable_identities_and_non_finite_values_are_rejected() -> None:
    class ProcessLocalIdentity:
        pass

    dataset = _valid_dataset()
    with pytest.raises(DatasetValidationError, match="supported stable identity"):
        ValidationDataset(
            sample_ids=(ProcessLocalIdentity(), "s1", "s2", "s3"),
            session_axis=dataset.session_axis,
            sessions=dataset.sessions,
            information_intervals=dataset.information_intervals,
            features=dataset.features,
            targets=dataset.targets,
        )

    stable = ValidationDataset(
        sample_ids=(
            UUID("12345678-1234-5678-1234-567812345678"),
            ("asset", 1),
            "s2",
            3,
        ),
        session_axis=dataset.session_axis,
        sessions=dataset.sessions,
        information_intervals=dataset.information_intervals,
        features=dataset.features,
        targets=dataset.targets,
    )
    assert (
        stable.digest
        == ValidationDataset(
            sample_ids=(
                UUID("12345678-1234-5678-1234-567812345678"),
                ("asset", 1),
                "s2",
                3,
            ),
            session_axis=dataset.session_axis,
            sessions=dataset.sessions,
            information_intervals=dataset.information_intervals,
            features=dataset.features,
            targets=dataset.targets,
        ).digest
    )

    features = np.array(dataset.features, copy=True)
    features[0, 0] = np.nan
    with pytest.raises(DatasetValidationError, match="finite numeric values"):
        ValidationDataset(
            sample_ids=dataset.sample_ids,
            session_axis=dataset.session_axis,
            sessions=dataset.sessions,
            information_intervals=dataset.information_intervals,
            features=features,
            targets=dataset.targets,
        )


def test_nested_model_configuration_is_deeply_immutable() -> None:
    spec = ModelSpec(
        name="nested",
        version="1",
        parameters={"optimizer": {"schedule": [0.1, 0.01]}},
    )

    with pytest.raises(TypeError):
        spec.parameters["optimizer"]["extra"] = True
    assert spec.parameters["optimizer"]["schedule"] == (0.1, 0.01)


def test_non_finite_predictions_and_metrics_fail_closed() -> None:
    class NonFiniteEstimator(MeanEstimator):
        def predict(self, features: np.ndarray) -> np.ndarray:
            return np.full(features.shape[0], np.nan)

    with pytest.raises(PredictionShapeError, match="finite"):
        LeakageSafeEvaluator(
            splitter=PurgedKFold(n_splits=2),
            estimator_factory=NonFiniteEstimator,
            model_spec=ModelSpec(name="non-finite", version="1"),
        ).evaluate(_valid_dataset())

    with pytest.raises(MetricEvaluationError, match="finite"):
        LeakageSafeEvaluator(
            splitter=PurgedKFold(n_splits=2),
            estimator_factory=MeanEstimator,
            model_spec=ModelSpec(name="mean", version="1"),
            metrics=(
                MetricSpec(
                    name="non-finite",
                    version="1",
                    function=lambda actual, predicted: float("nan"),
                ),
            ),
        ).evaluate(_valid_dataset())


def test_metrics_report_observation_and_fold_coverage() -> None:
    result = LeakageSafeEvaluator(
        splitter=PurgedKFold(n_splits=2),
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="mean", version="1"),
        metrics=(
            MetricSpec(
                name="mse",
                version="1",
                function=lambda actual, predicted: float(
                    np.mean((actual - predicted) ** 2)
                ),
            ),
        ),
    ).evaluate(_valid_dataset())

    metric = result.metrics[0]
    assert metric.observation_count == 4
    assert metric.observation_coverage == 1.0
    assert metric.fold_count == 2
    assert metric.fold_coverage == 1.0
    assert result.evidence_channel is EvidenceChannel.MODEL_SELECTION
