from __future__ import annotations

import numpy as np

from purged_kfold_validation import (
    FoldAssignment,
    InformationInterval,
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


def test_embargo_advances_on_the_session_axis_across_closed_market_days() -> None:
    sessions = np.array(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-08",
            "2025-01-09",
            "2025-01-10",
        ],
        dtype="datetime64[D]",
    )
    dataset = ValidationDataset(
        sample_ids=tuple(f"s{index}" for index in range(6)),
        session_axis=sessions,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        features=np.arange(6, dtype=float).reshape(6, 1),
        targets=np.arange(6, dtype=float),
    )

    plan = PurgedKFold(
        n_splits=3,
        embargo_sessions=1,
        include_exclusion_trace=True,
    ).plan(dataset)

    first = plan.folds[0]
    assert isinstance(first, FoldAssignment)
    assert first.test_sample_ids == ("s0", "s1")
    assert first.train_sample_ids == ("s3", "s4", "s5")
    assert first.exclusion_summary.purged == 0
    assert first.exclusion_summary.embargoed == 1
    assert first.exclusion_trace is not None
    assert [(record.sample_id, record.reason) for record in first.exclusion_trace] == [
        ("s2", "embargo")
    ]


def test_disjoint_protected_intervals_do_not_purge_the_safe_middle() -> None:
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
    dataset = ValidationDataset(
        sample_ids=tuple(f"s{index}" for index in range(6)),
        session_axis=sessions,
        sessions=sessions,
        information_intervals=(
            InformationInterval("2025-01-01", "2025-01-01"),
            InformationInterval("2025-01-10", "2025-01-10"),
            InformationInterval("2025-01-05", "2025-01-05"),
            InformationInterval("2025-01-10", "2025-01-10"),
            InformationInterval("2025-01-11", "2025-01-11"),
            InformationInterval("2025-01-12", "2025-01-12"),
        ),
        features=np.arange(6, dtype=float).reshape(6, 1),
        targets=np.arange(6, dtype=float),
    )

    first = (
        PurgedKFold(
            n_splits=3,
            include_exclusion_trace=True,
        )
        .plan(dataset)
        .folds[0]
    )

    assert isinstance(first, FoldAssignment)
    assert first.train_sample_ids == ("s2", "s4", "s5")
    assert first.exclusion_summary.purged == 1
    assert first.exclusion_trace is not None
    assert [(item.sample_id, item.reason) for item in first.exclusion_trace] == [
        ("s3", "purge-overlap")
    ]


def test_panel_session_groups_never_cross_assignment_sides() -> None:
    axis = np.array(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
        dtype="datetime64[D]",
    )
    sessions = np.repeat(axis, 2)
    dataset = ValidationDataset(
        sample_ids=tuple(
            f"{asset}-{str(session)}" for session in axis for asset in ("A", "B")
        ),
        asset_ids=("A", "B") * 4,
        session_axis=axis,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        decision_times=sessions.astype("datetime64[ns]") + np.timedelta64(12, "h"),
        feature_availability=sessions,
        pit_snapshot=PITSnapshot(
            snapshot_id="panel-fixture",
            source_digest="panel-fixture-digest",
        ),
        features=np.arange(16, dtype=float).reshape(8, 2),
        targets=np.arange(8, dtype=float),
    )
    splitter = PurgedKFold(n_splits=2)

    assignments = splitter.plan(dataset).require_assignments()

    for session in axis:
        positions = {
            index for index, value in enumerate(dataset.sessions) if value == session
        }
        test_folds = {
            fold.fold_index
            for fold in assignments
            if positions.intersection(fold.test_positions.tolist())
        }
        assert len(test_folds) == 1
        selected = assignments[next(iter(test_folds))]
        assert positions.issubset(set(selected.test_positions.tolist()))

    result = LeakageSafeEvaluator(
        splitter=splitter,
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="panel-mean", version="1"),
    ).evaluate(dataset)
    assert result.ledger.sample_ids == dataset.sample_ids
    assert (
        tuple(item.asset_id for item in result.ledger.observations) == dataset.asset_ids
    )
