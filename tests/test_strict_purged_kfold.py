from __future__ import annotations

import numpy as np
import pytest

from purged_kfold_validation import (
    FoldAssignment,
    InformationInterval,
    InvalidFoldError,
    LeakageSafeEvaluator,
    ModelSpec,
    PITSnapshot,
    PurgedKFold,
    ValidationDataset,
)


class MeanEstimator:
    def fit(self, features: np.ndarray, targets: np.ndarray) -> MeanEstimator:
        self.mean = float(targets.mean())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.mean)


def _six_session_dataset() -> ValidationDataset:
    sessions = np.array(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
        ],
        dtype="datetime64[D]",
    )
    return ValidationDataset(
        sample_ids=tuple(f"s{index}" for index in range(6)),
        session_axis=sessions,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        decision_times=sessions.astype("datetime64[ns]") + np.timedelta64(12, "h"),
        feature_availability=sessions,
        pit_snapshot=PITSnapshot(
            snapshot_id="strict-fixture",
            source_digest="strict-fixture-digest",
        ),
        features=np.arange(12, dtype=float).reshape(6, 2),
        targets=np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
    )


def test_invalid_candidates_remain_diagnostic_but_cannot_be_scored() -> None:
    dataset = _six_session_dataset()
    splitter = PurgedKFold(n_splits=3, min_train_samples=5)

    plan = splitter.plan(dataset)

    assert len(plan.invalid_folds) == 3
    assert plan.invalid_folds[0].reasons == ("minimum training samples not met: 4 < 5",)
    with pytest.raises(InvalidFoldError, match="fold 0"):
        plan.require_assignments()

    evaluator = LeakageSafeEvaluator(
        splitter=splitter,
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="mean", version="1"),
    )
    with pytest.raises(InvalidFoldError, match="fold 0"):
        evaluator.evaluate(dataset)


def test_complete_purged_kfold_has_contiguous_unique_test_coverage() -> None:
    dataset = _six_session_dataset()
    splitter = PurgedKFold(n_splits=3)

    plan = splitter.plan(dataset)
    assignments = plan.require_assignments()

    assert all(isinstance(fold, FoldAssignment) for fold in assignments)
    assert (
        tuple(sample_id for fold in assignments for sample_id in fold.test_sample_ids)
        == dataset.sample_ids
    )
    assert [
        (
            str(fold.test_blocks[0].start_session.astype("datetime64[D]")),
            str(fold.test_blocks[0].end_session.astype("datetime64[D]")),
        )
        for fold in assignments
    ] == [
        ("2025-01-02", "2025-01-03"),
        ("2025-01-06", "2025-01-07"),
        ("2025-01-08", "2025-01-09"),
    ]
    assert len({fold.split_id for fold in assignments}) == 3
    assert all(fold.evidence_channel == "model-selection" for fold in assignments)

    result = LeakageSafeEvaluator(
        splitter=splitter,
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="mean", version="1"),
    ).evaluate(dataset)
    assert result.ledger.sample_ids == dataset.sample_ids
    assert result.evidence_channel == "model-selection"


@pytest.mark.parametrize(
    "arguments",
    [
        {"n_splits": 1},
        {"n_splits": 2, "min_train_sessions": 0},
        {"n_splits": 2, "min_train_samples": 0},
        {"n_splits": 2, "min_test_sessions": 0},
    ],
)
def test_split_constraints_reject_unsupported_sizes(arguments: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="integer of at least"):
        PurgedKFold(**arguments)
