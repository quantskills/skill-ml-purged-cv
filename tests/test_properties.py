from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from purged_kfold_validation import (
    InformationInterval,
    PurgedKFold,
    ValidationDataset,
)


@settings(max_examples=50, deadline=None)
@given(
    data=st.data(),
    session_count=st.integers(min_value=4, max_value=12),
)
def test_purged_plans_preserve_coverage_determinism_and_no_overlap(
    data: st.DataObject, session_count: int
) -> None:
    n_splits = data.draw(st.integers(min_value=2, max_value=min(5, session_count)))
    horizons = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=4),
            min_size=session_count,
            max_size=session_count,
        )
    )
    axis = np.datetime64("2025-01-01") + np.arange(session_count)
    dataset = ValidationDataset(
        sample_ids=tuple(f"s{index}" for index in range(session_count)),
        session_axis=axis,
        sessions=axis,
        information_intervals=tuple(
            InformationInterval(session, session + np.timedelta64(horizon, "D"))
            for session, horizon in zip(axis, horizons)
        ),
        features=np.arange(session_count, dtype=float).reshape(session_count, 1),
        targets=np.arange(session_count, dtype=float),
    )
    splitter = PurgedKFold(n_splits=n_splits)

    plan = splitter.plan(dataset)

    all_test_positions = sorted(
        int(position) for fold in plan.folds for position in fold.test_positions
    )
    assert all_test_positions == list(range(session_count))
    for fold in plan.assignments:
        protected = tuple(
            dataset.information_intervals[int(position)]
            for position in fold.test_positions
        )
        assert all(
            not any(
                dataset.information_intervals[int(position)].overlaps(interval)
                for interval in protected
            )
            for position in fold.train_positions
        )
    assert splitter.plan(dataset) == plan


@settings(max_examples=30, deadline=None)
@given(
    session_count=st.integers(min_value=4, max_value=9),
    asset_count=st.integers(min_value=1, max_value=4),
)
def test_panel_session_groups_are_indivisible(
    session_count: int, asset_count: int
) -> None:
    axis = np.datetime64("2025-02-01") + np.arange(session_count)
    sessions = np.repeat(axis, asset_count)
    row_count = len(sessions)
    dataset = ValidationDataset(
        sample_ids=tuple(f"s{index}" for index in range(row_count)),
        asset_ids=tuple(
            f"a{asset}" for _ in range(session_count) for asset in range(asset_count)
        ),
        session_axis=axis,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        features=np.arange(row_count, dtype=float).reshape(row_count, 1),
        targets=np.arange(row_count, dtype=float),
    )

    assignments = PurgedKFold(n_splits=2).plan(dataset).require_assignments()

    for session in axis:
        group = {
            index
            for index, observed in enumerate(dataset.sessions)
            if observed == session
        }
        matching_folds = [
            fold
            for fold in assignments
            if group.intersection(fold.test_positions.tolist())
        ]
        assert len(matching_folds) == 1
        assert group.issubset(set(matching_folds[0].test_positions.tolist()))


@settings(max_examples=30, deadline=None)
@given(
    gaps=st.lists(
        st.integers(min_value=1, max_value=5),
        min_size=7,
        max_size=7,
    ),
    embargo_sessions=st.integers(min_value=0, max_value=2),
)
def test_non_session_calendar_gaps_do_not_change_embargo_counts(
    gaps: list[int], embargo_sessions: int
) -> None:
    offsets = np.concatenate(([0], np.cumsum(gaps)))
    sparse_axis = np.datetime64("2025-03-01") + offsets
    dense_axis = np.datetime64("2025-03-01") + np.arange(8)

    def dataset_for(axis: np.ndarray) -> ValidationDataset:
        return ValidationDataset(
            sample_ids=tuple(f"s{index}" for index in range(8)),
            session_axis=axis,
            sessions=axis,
            information_intervals=tuple(
                InformationInterval(session, session) for session in axis
            ),
            features=np.arange(8, dtype=float).reshape(8, 1),
            targets=np.arange(8, dtype=float),
        )

    splitter = PurgedKFold(n_splits=2, embargo_sessions=embargo_sessions)
    sparse = splitter.plan(dataset_for(sparse_axis)).require_assignments()
    dense = splitter.plan(dataset_for(dense_axis)).require_assignments()

    assert [fold.train_sample_ids for fold in sparse] == [
        fold.train_sample_ids for fold in dense
    ]
    assert [fold.exclusion_summary for fold in sparse] == [
        fold.exclusion_summary for fold in dense
    ]


@settings(max_examples=30, deadline=None)
@given(changed_value=st.floats(allow_nan=False, allow_infinity=False))
def test_relevant_feature_changes_alter_dataset_digests(changed_value: float) -> None:
    if changed_value == 1.0:
        changed_value = 2.0
    axis = np.array(["2025-01-02", "2025-01-03"], dtype="datetime64[D]")

    def dataset_with(value: float) -> ValidationDataset:
        return ValidationDataset(
            sample_ids=("s0", "s1"),
            session_axis=axis,
            sessions=axis,
            information_intervals=(
                InformationInterval(axis[0], axis[0]),
                InformationInterval(axis[1], axis[1]),
            ),
            features=np.array([[1.0], [value]]),
            targets=np.array([0.0, 1.0]),
        )

    assert dataset_with(1.0).digest != dataset_with(changed_value).digest
