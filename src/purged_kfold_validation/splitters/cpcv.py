"""Deterministic combinatorial purged cross-validation planning."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

from ..domain import (
    EvidenceChannel,
    FoldAssignment,
    InvalidFold,
    SplitPlan,
    TestBlock,
    ValidationDataset,
    canonical_digest,
    canonical_identity,
)
from ..errors import SplitPlanError
from ..leakage import apply_leakage_exclusions
from ..paths import build_cpcv_path_decomposition


@dataclass(frozen=True, slots=True)
class CombinatorialPurgedCV:
    """Plan every bounded chronological test-group combination."""

    n_groups: int = 6
    n_test_groups: int = 2
    min_train_sessions: int = 1
    min_train_samples: int = 1
    min_test_sessions: int = 1
    embargo_sessions: int = 0
    max_combinations: int = 10_000
    include_exclusion_trace: bool = False

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("n_groups", self.n_groups, 3),
            ("n_test_groups", self.n_test_groups, 2),
            ("min_train_sessions", self.min_train_sessions, 1),
            ("min_train_samples", self.min_train_samples, 1),
            ("min_test_sessions", self.min_test_sessions, 1),
            ("max_combinations", self.max_combinations, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise SplitPlanError(f"{name} must be an integer of at least {minimum}")
        if self.n_test_groups >= self.n_groups:
            raise SplitPlanError("n_test_groups must be less than n_groups")
        if (
            isinstance(self.embargo_sessions, bool)
            or not isinstance(self.embargo_sessions, int)
            or self.embargo_sessions < 0
        ):
            raise SplitPlanError("embargo_sessions must be a non-negative integer")
        if not isinstance(self.include_exclusion_trace, bool):
            raise SplitPlanError("include_exclusion_trace must be a boolean")
        if self.combination_count > self.max_combinations:
            raise SplitPlanError(
                "CPCV combination count exceeds max_combinations: "
                f"{self.combination_count} > {self.max_combinations}"
            )

    @property
    def combination_count(self) -> int:
        return comb(self.n_groups, self.n_test_groups)

    @property
    def path_count(self) -> int:
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "kind": "combinatorial-purged-cv",
                "n_groups": self.n_groups,
                "n_test_groups": self.n_test_groups,
                "min_train_sessions": self.min_train_sessions,
                "min_train_samples": self.min_train_samples,
                "min_test_sessions": self.min_test_sessions,
                "embargo_sessions": self.embargo_sessions,
                "max_combinations": self.max_combinations,
                "include_exclusion_trace": self.include_exclusion_trace,
                "schema_version": "1",
            }
        )

    def plan(self, dataset: ValidationDataset) -> SplitPlan:
        """Return every deterministic CPCV combination as diagnostic evidence."""

        observed_sessions = set(dataset.sessions)
        active_sessions = tuple(
            session for session in dataset.session_axis if session in observed_sessions
        )
        if self.n_groups > len(active_sessions):
            raise SplitPlanError("n_groups cannot exceed the number of active sessions")

        chunks = tuple(
            tuple(active_sessions[int(position)] for position in chunk)
            for chunk in np.array_split(np.arange(len(active_sessions)), self.n_groups)
        )
        group_combinations = tuple(
            combinations(range(self.n_groups), self.n_test_groups)
        )
        group_blocks = tuple(
            TestBlock(
                start_session=group[0],
                end_session=group[-1],
                session_count=len(group),
            )
            for group in chunks
        )
        path_decomposition = build_cpcv_path_decomposition(
            groups=group_blocks,
            combinations=group_combinations,
        )
        folds = tuple(
            self._plan_combination(dataset, chunks, combination_index, group_indices)
            for combination_index, group_indices in enumerate(group_combinations)
        )
        plan_digest = canonical_digest(
            {
                "dataset_digest": dataset.digest,
                "split_spec_digest": self.digest,
                "folds": [
                    fold.split_id
                    if isinstance(fold, FoldAssignment)
                    else {
                        "fold_index": fold.fold_index,
                        "reasons": fold.reasons,
                    }
                    for fold in folds
                ],
                "path_decomposition_digest": path_decomposition.digest,
            }
        )
        return SplitPlan(
            folds=folds,
            dataset_digest=dataset.digest,
            split_spec_digest=self.digest,
            digest=plan_digest,
            path_decomposition=path_decomposition,
        )

    def _plan_combination(
        self,
        dataset: ValidationDataset,
        groups: tuple[tuple[np.datetime64, ...], ...],
        combination_index: int,
        group_indices: tuple[int, ...],
    ) -> FoldAssignment | InvalidFold:
        test_sessions = tuple(
            session for group_index in group_indices for session in groups[group_index]
        )
        test_session_set = set(test_sessions)
        test_positions = tuple(
            index
            for index, session in enumerate(dataset.sessions)
            if session in test_session_set
        )
        candidate_positions = tuple(
            index
            for index, session in enumerate(dataset.sessions)
            if session not in test_session_set
        )
        test_blocks = self._test_blocks(groups, group_indices)
        exclusions = apply_leakage_exclusions(
            dataset,
            candidate_positions=candidate_positions,
            test_positions=test_positions,
            test_blocks=test_blocks,
            embargo_sessions=self.embargo_sessions,
            include_trace=self.include_exclusion_trace,
        )
        train_positions = exclusions.train_positions
        train_session_count = len(
            {dataset.sessions[position] for position in train_positions}
        )
        reasons: list[str] = []
        if len(test_sessions) < self.min_test_sessions:
            reasons.append(
                "minimum test sessions not met: "
                f"{len(test_sessions)} < {self.min_test_sessions}"
            )
        if train_session_count < self.min_train_sessions:
            reasons.append(
                "minimum training sessions not met: "
                f"{train_session_count} < {self.min_train_sessions}"
            )
        if len(train_positions) < self.min_train_samples:
            reasons.append(
                "minimum training samples not met: "
                f"{len(train_positions)} < {self.min_train_samples}"
            )
        test_sample_ids = tuple(
            dataset.sample_ids[position] for position in test_positions
        )
        if reasons:
            return InvalidFold(
                fold_index=combination_index,
                reasons=tuple(reasons),
                test_sample_ids=test_sample_ids,
                test_positions=test_positions,
                exclusion_summary=exclusions.summary,
                exclusion_trace=exclusions.trace,
            )

        split_id = canonical_digest(
            {
                "dataset_digest": dataset.digest,
                "split_spec_digest": self.digest,
                "combination_index": combination_index,
                "test_group_indices": list(group_indices),
                "train_sample_ids": [
                    canonical_identity(dataset.sample_ids[position])
                    for position in train_positions
                ],
                "test_sample_ids": [
                    canonical_identity(dataset.sample_ids[position])
                    for position in test_positions
                ],
            }
        )
        return FoldAssignment(
            fold_index=combination_index,
            train_sample_ids=tuple(
                dataset.sample_ids[position] for position in train_positions
            ),
            test_sample_ids=test_sample_ids,
            train_positions=np.asarray(train_positions, dtype=np.int64),
            test_positions=np.asarray(test_positions, dtype=np.int64),
            test_blocks=test_blocks,
            exclusion_summary=exclusions.summary,
            dataset_digest=dataset.digest,
            split_spec_digest=self.digest,
            split_id=split_id,
            evidence_channel=EvidenceChannel.CPCV_ROBUSTNESS,
            combination_index=combination_index,
            test_group_indices=group_indices,
            exclusion_trace=exclusions.trace,
        )

    @staticmethod
    def _test_blocks(
        groups: tuple[tuple[np.datetime64, ...], ...],
        group_indices: tuple[int, ...],
    ) -> tuple[TestBlock, ...]:
        runs: list[list[int]] = []
        for group_index in group_indices:
            if not runs or group_index != runs[-1][-1] + 1:
                runs.append([group_index])
            else:
                runs[-1].append(group_index)
        return tuple(
            TestBlock(
                start_session=groups[run[0]][0],
                end_session=groups[run[-1]][-1],
                session_count=sum(len(groups[group_index]) for group_index in run),
            )
            for run in runs
        )
