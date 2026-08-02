from __future__ import annotations

import numpy as np
import pytest

from purged_kfold_validation import (
    CausalWalkForward,
    EvidenceChannel,
    FoldAssignment,
    InformationInterval,
    InvalidFoldError,
    LeakageSafeEvaluator,
    ModelSpec,
    PITSnapshot,
    ValidationDataset,
)


def _dataset() -> ValidationDataset:
    sessions = np.array(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
            "2025-01-10",
            "2025-01-13",
        ],
        dtype="datetime64[D]",
    )
    return ValidationDataset(
        sample_ids=tuple(f"s{index}" for index in range(len(sessions))),
        session_axis=sessions,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        decision_times=sessions.astype("datetime64[ns]") + np.timedelta64(12, "h"),
        feature_availability=sessions,
        pit_snapshot=PITSnapshot(
            snapshot_id="walk-forward-fixture",
            source_digest="walk-forward-fixture-digest",
        ),
        features=np.arange(len(sessions), dtype=float).reshape(-1, 1),
        targets=np.arange(len(sessions), dtype=float),
    )


class MeanEstimator:
    def fit(self, features: np.ndarray, targets: np.ndarray) -> MeanEstimator:
        self.mean = float(targets.mean())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.mean)


def test_user_gets_deterministic_past_only_assignments_with_pre_test_gap() -> None:
    plan = CausalWalkForward(
        n_splits=2,
        test_sessions=2,
        pre_test_gap_sessions=1,
        include_exclusion_trace=True,
    ).plan(_dataset())

    first, second = plan.require_assignments()
    assert isinstance(first, FoldAssignment)
    assert first.test_sample_ids == ("s4", "s5")
    assert first.train_sample_ids == ("s0", "s1", "s2")
    assert first.evidence_channel is EvidenceChannel.CAUSAL_WALK_FORWARD
    assert first.exclusion_summary.pre_test_gapped == 1
    assert first.exclusion_trace is not None
    assert [(item.sample_id, item.reason) for item in first.exclusion_trace] == [
        ("s3", "pre-test-gap")
    ]

    assert second.test_sample_ids == ("s6", "s7")
    assert second.train_sample_ids == ("s0", "s1", "s2", "s3", "s4")
    assert all(
        _dataset().information_intervals[int(train)].end
        < second.test_blocks[0].start_session
        for train in second.train_positions
    )


def test_formal_evaluation_preserves_the_causal_evidence_channel() -> None:
    result = LeakageSafeEvaluator(
        splitter=CausalWalkForward(n_splits=2, test_sessions=2),
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="walk-forward-mean", version="1"),
    ).evaluate(_dataset())

    assert result.evidence_channel is EvidenceChannel.CAUSAL_WALK_FORWARD
    assert {
        observation.evidence_channel for observation in result.ledger.observations
    } == {EvidenceChannel.CAUSAL_WALK_FORWARD}
    assert result.ledger.sample_ids == ("s4", "s5", "s6", "s7")


def test_causal_fold_excludes_training_information_that_occurs_after_test_start() -> (
    None
):
    base = _dataset()
    intervals = list(base.information_intervals)
    intervals[3] = InformationInterval("2025-01-10", "2025-01-10")
    dataset = ValidationDataset(
        sample_ids=base.sample_ids,
        session_axis=base.session_axis,
        sessions=base.sessions,
        information_intervals=intervals,
        decision_times=base.decision_times,
        feature_availability=base.feature_availability,
        pit_snapshot=base.pit_snapshot,
        features=base.features,
        targets=base.targets,
    )

    first = (
        CausalWalkForward(
            n_splits=2,
            test_sessions=2,
            include_exclusion_trace=True,
        )
        .plan(dataset)
        .require_assignments()[0]
    )

    assert first.train_sample_ids == ("s0", "s1", "s2")
    assert first.exclusion_summary.noncausal == 1
    assert first.exclusion_trace is not None
    assert [(item.sample_id, item.reason) for item in first.exclusion_trace] == [
        ("s3", "causal-future-information")
    ]


def test_sliding_history_is_bounded_in_trading_sessions() -> None:
    assignments = (
        CausalWalkForward(
            n_splits=2,
            test_sessions=1,
            max_train_sessions=2,
        )
        .plan(_dataset())
        .require_assignments()
    )

    assert assignments[0].train_sample_ids == ("s4", "s5")
    assert assignments[1].train_sample_ids == ("s5", "s6")


def test_panel_rows_from_one_session_remain_indivisible() -> None:
    base = _dataset()
    sessions = np.repeat(np.asarray(base.session_axis), 2)
    dataset = ValidationDataset(
        sample_ids=tuple(
            (asset, np.datetime_as_string(session, unit="D"))
            for session in base.session_axis
            for asset in ("A", "B")
        ),
        asset_ids=("A", "B") * len(base.session_axis),
        session_axis=base.session_axis,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        decision_times=sessions + np.timedelta64(12, "h"),
        feature_availability=sessions,
        pit_snapshot=base.pit_snapshot,
        features=np.arange(len(sessions), dtype=float).reshape(-1, 1),
        targets=np.arange(len(sessions), dtype=float),
    )

    assignments = (
        CausalWalkForward(n_splits=2, test_sessions=1)
        .plan(dataset)
        .require_assignments()
    )

    assert assignments[0].test_sample_ids == (
        ("A", "2025-01-10"),
        ("B", "2025-01-10"),
    )
    assert assignments[1].test_sample_ids == (
        ("A", "2025-01-13"),
        ("B", "2025-01-13"),
    )


def test_invalid_causal_fold_aborts_formal_evaluation() -> None:
    evaluator = LeakageSafeEvaluator(
        splitter=CausalWalkForward(
            n_splits=2,
            test_sessions=1,
            pre_test_gap_sessions=5,
            min_train_sessions=2,
        ),
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="invalid-walk-forward", version="1"),
    )

    with pytest.raises(InvalidFoldError, match="minimum training sessions not met"):
        evaluator.evaluate(_dataset())


def test_pre_test_gap_advances_on_sessions_across_a_weekend() -> None:
    sessions = np.array(
        ["2025-01-02", "2025-01-03", "2025-01-06"], dtype="datetime64[D]"
    )
    dataset = ValidationDataset(
        sample_ids=("thursday", "friday", "monday"),
        session_axis=sessions,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        features=np.arange(3, dtype=float).reshape(-1, 1),
        targets=np.arange(3, dtype=float),
    )

    assignment = (
        CausalWalkForward(
            n_splits=1,
            test_sessions=1,
            pre_test_gap_sessions=1,
            include_exclusion_trace=True,
        )
        .plan(dataset)
        .require_assignments()[0]
    )

    assert assignment.train_sample_ids == ("thursday",)
    assert assignment.test_sample_ids == ("monday",)
    assert assignment.exclusion_trace is not None
    assert [(item.sample_id, item.reason) for item in assignment.exclusion_trace] == [
        ("friday", "pre-test-gap")
    ]
