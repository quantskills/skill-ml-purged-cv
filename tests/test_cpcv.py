from __future__ import annotations

from math import comb

import numpy as np
import pytest

from purged_kfold_validation import (
    CombinatorialPurgedCV,
    EvidenceChannel,
    InformationInterval,
    InvalidFoldError,
    LeakageSafeEvaluator,
    MetricSpec,
    ModelSpec,
    PITSnapshot,
    SplitPlanError,
    ValidationDataset,
)


def _dataset(session_count: int = 12) -> ValidationDataset:
    sessions = np.arange(
        np.datetime64("2025-01-02"),
        np.datetime64("2025-01-02") + np.timedelta64(session_count, "D"),
        dtype="datetime64[D]",
    )
    return ValidationDataset(
        sample_ids=tuple(f"s{index}" for index in range(session_count)),
        session_axis=sessions,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        features=np.arange(session_count, dtype=float).reshape(-1, 1),
        targets=np.arange(session_count, dtype=float),
    )


def test_user_gets_every_general_cpcv_combination_once() -> None:
    plan = CombinatorialPurgedCV(n_groups=6, n_test_groups=2).plan(_dataset())

    assignments = plan.require_assignments()
    assert len(assignments) == 15
    assert tuple(assignment.combination_index for assignment in assignments) == tuple(
        range(15)
    )
    assert assignments[0].test_group_indices == (0, 1)
    assert assignments[-1].test_group_indices == (4, 5)
    assert all(
        assignment.evidence_channel is EvidenceChannel.CPCV_ROBUSTNESS
        for assignment in assignments
    )


def test_six_by_two_cpcv_has_five_complete_deterministic_paths() -> None:
    plan = CombinatorialPurgedCV(n_groups=6, n_test_groups=2).plan(_dataset())

    decomposition = plan.path_decomposition
    assert decomposition is not None
    assert decomposition.path_count == 5
    assert len(decomposition.paths) == 5
    assert all(
        tuple(occurrence.group_index for occurrence in path.occurrences)
        == (0, 1, 2, 3, 4, 5)
        for path in decomposition.paths
    )

    assigned_occurrences = {
        (occurrence.combination_index, occurrence.group_index)
        for path in decomposition.paths
        for occurrence in path.occurrences
    }
    expected_occurrences = {
        (assignment.combination_index, group_index)
        for assignment in plan.require_assignments()
        for group_index in assignment.test_group_indices
    }
    assert assigned_occurrences == expected_occurrences

    paths_by_combination: dict[int, set[int]] = {}
    for path in decomposition.paths:
        for occurrence in path.occurrences:
            paths_by_combination.setdefault(occurrence.combination_index, set()).add(
                path.path_index
            )
    assert all(len(path_indices) == 2 for path_indices in paths_by_combination.values())

    repeated = CombinatorialPurgedCV(n_groups=6, n_test_groups=2).plan(_dataset())
    assert repeated.path_decomposition == decomposition
    assert repeated.path_decomposition.digest == decomposition.digest


def test_path_decomposition_is_general_beyond_the_six_by_two_example() -> None:
    plan = CombinatorialPurgedCV(n_groups=5, n_test_groups=3).plan(_dataset(15))

    decomposition = plan.path_decomposition
    assert decomposition is not None
    assert len(plan.require_assignments()) == 10
    assert decomposition.path_count == 6
    assert all(len(path.occurrences) == 5 for path in decomposition.paths)
    assert all(
        tuple(item.group_index for item in path.occurrences) == (0, 1, 2, 3, 4)
        for path in decomposition.paths
    )


@pytest.mark.parametrize(
    ("n_groups", "n_test_groups"),
    tuple(
        (n_groups, n_test_groups)
        for n_groups in range(3, 9)
        for n_test_groups in range(2, n_groups)
    ),
)
def test_path_invariants_hold_across_general_small_configurations(
    n_groups: int, n_test_groups: int
) -> None:
    plan = CombinatorialPurgedCV(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
    ).plan(_dataset(n_groups * 2))

    decomposition = plan.path_decomposition
    assert decomposition is not None
    assert len(plan.require_assignments()) == comb(n_groups, n_test_groups)
    assert decomposition.path_count == comb(n_groups - 1, n_test_groups - 1)
    assert all(len(path.occurrences) == n_groups for path in decomposition.paths)


def test_evaluator_preserves_repeated_predictions_as_complete_path_evidence() -> None:
    dataset = _dataset(8)
    sessions = np.asarray(dataset.sessions)
    formal_dataset = ValidationDataset(
        sample_ids=dataset.sample_ids,
        session_axis=dataset.session_axis,
        sessions=sessions,
        information_intervals=dataset.information_intervals,
        decision_times=sessions + np.timedelta64(12, "h"),
        feature_availability=sessions,
        pit_snapshot=PITSnapshot(
            snapshot_id="cpcv-evaluator-fixture",
            source_digest="cpcv-evaluator-source",
        ),
        features=dataset.features,
        targets=dataset.targets,
    )

    class MeanEstimator:
        def fit(self, features: np.ndarray, targets: np.ndarray) -> MeanEstimator:
            self.mean = float(targets.mean())
            return self

        def predict(self, features: np.ndarray) -> np.ndarray:
            return np.full(len(features), self.mean)

    result = LeakageSafeEvaluator(
        splitter=CombinatorialPurgedCV(n_groups=4, n_test_groups=2),
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="cpcv-mean", version="1"),
        metrics=(
            MetricSpec(
                name="mse",
                version="1",
                function=lambda actual, predicted: float(
                    np.mean((actual - predicted) ** 2)
                ),
            ),
        ),
    ).evaluate(formal_dataset)

    assert result.evidence_channel is EvidenceChannel.CPCV_ROBUSTNESS
    assert len(result.ledger.observations) == 24
    assert all(
        observation.combination_index is not None
        and observation.group_index is not None
        and observation.path_index is not None
        for observation in result.ledger.observations
    )
    for path_index in range(3):
        path_sample_ids = {
            observation.sample_id
            for observation in result.ledger.observations
            if observation.path_index == path_index
        }
        assert path_sample_ids == set(formal_dataset.sample_ids)

    metric = result.metrics[0]
    assert metric.observation_count == 24
    assert metric.observation_coverage == 1.0
    assert metric.path_count == 3
    assert metric.path_coverage == 1.0
    assert len(metric.per_path) == 3


def test_cpcv_purges_across_disjoint_blocks_and_embargoes_each_block() -> None:
    dataset = _dataset(12)
    intervals = list(dataset.information_intervals)
    intervals[1] = InformationInterval(dataset.sessions[1], dataset.sessions[2])
    crossing_dataset = ValidationDataset(
        sample_ids=dataset.sample_ids,
        session_axis=dataset.session_axis,
        sessions=dataset.sessions,
        information_intervals=tuple(intervals),
        features=dataset.features,
        targets=dataset.targets,
    )

    plan = CombinatorialPurgedCV(
        n_groups=6,
        n_test_groups=2,
        embargo_sessions=1,
        include_exclusion_trace=True,
    ).plan(crossing_dataset)
    disjoint = next(
        assignment
        for assignment in plan.require_assignments()
        if assignment.test_group_indices == (1, 3)
    )

    assert len(disjoint.test_blocks) == 2
    assert disjoint.exclusion_summary.purged == 1
    assert disjoint.exclusion_summary.embargoed == 2
    assert disjoint.exclusion_trace is not None
    assert {
        (record.position, record.reason) for record in disjoint.exclusion_trace
    } == {(1, "purge-overlap"), (4, "embargo"), (8, "embargo")}


def test_adjacent_cpcv_groups_form_one_contiguous_test_block() -> None:
    plan = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_sessions=1).plan(
        _dataset(12)
    )
    adjacent = next(
        assignment
        for assignment in plan.require_assignments()
        if assignment.test_group_indices == (1, 2)
    )

    assert len(adjacent.test_blocks) == 1
    assert adjacent.test_blocks[0].session_count == 4
    assert adjacent.exclusion_summary.embargoed == 1


def test_invalid_cpcv_combinations_abort_formal_assignment_access() -> None:
    plan = CombinatorialPurgedCV(
        n_groups=4,
        n_test_groups=2,
        min_train_sessions=5,
    ).plan(_dataset(8))

    assert len(plan.invalid_folds) == 6
    with pytest.raises(InvalidFoldError, match="formal assignments rejected"):
        plan.require_assignments()


def test_cpcv_rejects_combinatorial_work_above_the_declared_budget() -> None:
    with pytest.raises(SplitPlanError, match="exceeds max_combinations"):
        CombinatorialPurgedCV(
            n_groups=10,
            n_test_groups=5,
            max_combinations=100,
        )


def test_panel_rows_from_one_session_never_cross_cpcv_assignment_sides() -> None:
    sessions = np.arange(
        np.datetime64("2025-01-02"),
        np.datetime64("2025-01-08"),
        dtype="datetime64[D]",
    )
    panel_sessions = np.repeat(sessions, 2)
    dataset = ValidationDataset(
        sample_ids=tuple(
            (asset, str(session)) for session in sessions for asset in ("A", "B")
        ),
        session_axis=sessions,
        sessions=panel_sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in panel_sessions
        ),
        asset_ids=tuple(asset for _ in sessions for asset in ("A", "B")),
        features=np.arange(12, dtype=float).reshape(-1, 1),
        targets=np.arange(12, dtype=float),
    )

    for assignment in (
        CombinatorialPurgedCV(n_groups=3, n_test_groups=2)
        .plan(dataset)
        .require_assignments()
    ):
        normalized_sessions = np.asarray(dataset.sessions)
        train_sessions = set(normalized_sessions[assignment.train_positions])
        test_sessions = set(normalized_sessions[assignment.test_positions])
        assert train_sessions.isdisjoint(test_sessions)
