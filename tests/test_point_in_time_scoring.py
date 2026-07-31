from __future__ import annotations

import numpy as np
import pytest

from purged_kfold_validation import (
    InformationInterval,
    LeakageSafeEvaluator,
    ModelSpec,
    PITSnapshot,
    PointInTimeValidationError,
    PurgedKFold,
    ValidationDataset,
)


class MeanEstimator:
    def fit(self, features: np.ndarray, targets: np.ndarray) -> MeanEstimator:
        self.mean = float(targets.mean())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.mean)


def _dataset(
    *,
    snapshot: PITSnapshot | None,
    availability: np.ndarray | None,
) -> ValidationDataset:
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
        feature_availability=availability,
        pit_snapshot=snapshot,
        features=np.array([[10_001.0], [10_002.0], [10_003.0], [10_004.0]]),
        targets=np.array([0.0, 0.0, 1.0, 1.0]),
    )


def _evaluator() -> LeakageSafeEvaluator:
    return LeakageSafeEvaluator(
        splitter=PurgedKFold(n_splits=2),
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="mean", version="1"),
    )


def test_latest_revision_data_can_be_diagnosed_but_not_formally_scored() -> None:
    sessions = np.array(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
        dtype="datetime64[ns]",
    )
    dataset = _dataset(
        snapshot=PITSnapshot(
            snapshot_id="latest-export",
            source_digest="abc123",
            revision_policy="latest",
        ),
        availability=sessions,
    )

    plan = PurgedKFold(n_splits=2).plan(dataset)
    assert len(plan.folds) == 2
    assert dataset.formal_scoring_issues == (
        "PIT Snapshot revision policy is not point-in-time",
    )

    with pytest.raises(PointInTimeValidationError, match="revision policy") as error:
        _evaluator().evaluate(dataset)

    assert "10_001" not in str(error.value)


def test_point_in_time_evidence_reaches_the_oos_ledger() -> None:
    availability = np.array(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
        dtype="datetime64[ns]",
    )
    snapshot = PITSnapshot(
        snapshot_id="vendor-vintage-2025-q1",
        source_digest="source-sha256-001",
    )
    dataset = _dataset(snapshot=snapshot, availability=availability)

    result = _evaluator().evaluate(dataset)

    assert dataset.formal_scoring_issues == ()
    assert len(result.ledger.observations) == 4
    assert {
        observation.pit_snapshot_digest for observation in result.ledger.observations
    } == {snapshot.provenance_digest}


def test_missing_or_late_feature_evidence_fails_closed() -> None:
    snapshot = PITSnapshot(
        snapshot_id="vendor-vintage-2025-q1",
        source_digest="source-sha256-001",
    )
    missing = _dataset(snapshot=None, availability=None)
    assert missing.formal_scoring_issues == (
        "Feature Availability evidence is missing",
        "PIT Snapshot provenance is missing",
    )
    with pytest.raises(PointInTimeValidationError, match="provenance is missing"):
        _evaluator().evaluate(missing)

    per_feature = np.array(
        [
            ["2025-01-02T09:00", "2025-01-02T10:00"],
            ["2025-01-03T09:00", "2025-01-03T10:00"],
            ["2025-01-06T09:00", "2025-01-06T13:00"],
            ["2025-01-07T09:00", "2025-01-07T10:00"],
        ],
        dtype="datetime64[ns]",
    )
    late = ValidationDataset(
        sample_ids=("s0", "s1", "s2", "s3"),
        session_axis=np.array(
            ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
            dtype="datetime64[D]",
        ),
        sessions=np.array(
            ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
            dtype="datetime64[D]",
        ),
        information_intervals=tuple(
            InformationInterval(value, value)
            for value in ("2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07")
        ),
        decision_times=np.array(
            [
                "2025-01-02T12:00",
                "2025-01-03T12:00",
                "2025-01-06T12:00",
                "2025-01-07T12:00",
            ],
            dtype="datetime64[ns]",
        ),
        feature_availability=per_feature,
        pit_snapshot=snapshot,
        features=np.arange(8, dtype=float).reshape(4, 2),
        targets=np.arange(4, dtype=float),
    )

    assert late.formal_scoring_issues == (
        "Feature Availability is later than Decision Time at sample positions 2",
    )
    with pytest.raises(PointInTimeValidationError, match="sample positions 2"):
        _evaluator().evaluate(late)
