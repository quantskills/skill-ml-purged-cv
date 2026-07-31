"""Deterministic Purged K-Fold diagnostic planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain import (
    FoldAssignment,
    InvalidFold,
    SplitPlan,
    TestBlock,
    ValidationDataset,
    canonical_identity,
    canonical_digest,
)
from ..errors import SplitPlanError
from ..leakage import apply_leakage_exclusions


@dataclass(frozen=True, slots=True)
class PurgedKFold:
    """Plan deterministic contiguous folds and purge overlapping information."""

    n_splits: int = 5
    min_train_sessions: int = 1
    min_train_samples: int = 1
    min_test_sessions: int = 1
    embargo_sessions: int = 0
    include_exclusion_trace: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_splits, bool)
            or not isinstance(self.n_splits, int)
            or self.n_splits < 2
        ):
            raise SplitPlanError("n_splits must be an integer of at least 2")
        for name, value in (
            ("min_train_sessions", self.min_train_sessions),
            ("min_train_samples", self.min_train_samples),
            ("min_test_sessions", self.min_test_sessions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SplitPlanError(f"{name} must be an integer of at least 1")
        if (
            isinstance(self.embargo_sessions, bool)
            or not isinstance(self.embargo_sessions, int)
            or self.embargo_sessions < 0
        ):
            raise SplitPlanError("embargo_sessions must be a non-negative integer")
        if not isinstance(self.include_exclusion_trace, bool):
            raise SplitPlanError("include_exclusion_trace must be a boolean")

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "kind": "purged-kfold",
                "n_splits": self.n_splits,
                "min_train_sessions": self.min_train_sessions,
                "min_train_samples": self.min_train_samples,
                "min_test_sessions": self.min_test_sessions,
                "embargo_sessions": self.embargo_sessions,
                "include_exclusion_trace": self.include_exclusion_trace,
                "schema_version": "1",
            }
        )

    def plan(self, dataset: ValidationDataset) -> SplitPlan:
        """Return diagnostic fold candidates without creating a score."""

        active_sessions = tuple(
            session
            for session in dataset.session_axis
            if session in set(dataset.sessions)
        )
        if self.n_splits > len(active_sessions):
            raise SplitPlanError("n_splits cannot exceed the number of active sessions")

        chunks = np.array_split(np.arange(len(active_sessions)), self.n_splits)
        folds: list[FoldAssignment | InvalidFold] = []
        for fold_index, chunk in enumerate(chunks):
            test_sessions = tuple(active_sessions[int(position)] for position in chunk)
            folds.append(self._plan_fold(dataset, fold_index, test_sessions))

        plan_digest = canonical_digest(
            {
                "dataset_digest": dataset.digest,
                "split_spec_digest": self.digest,
                "folds": [
                    fold.split_id
                    if isinstance(fold, FoldAssignment)
                    else {"fold_index": fold.fold_index, "reasons": fold.reasons}
                    for fold in folds
                ],
            }
        )
        return SplitPlan(
            folds=tuple(folds),
            dataset_digest=dataset.digest,
            split_spec_digest=self.digest,
            digest=plan_digest,
        )

    def _plan_fold(
        self,
        dataset: ValidationDataset,
        fold_index: int,
        test_sessions: tuple[np.datetime64, ...],
    ) -> FoldAssignment | InvalidFold:
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
        test_block = TestBlock(
            start_session=test_sessions[0],
            end_session=test_sessions[-1],
            session_count=len(test_sessions),
        )
        exclusions = apply_leakage_exclusions(
            dataset,
            candidate_positions=candidate_positions,
            test_positions=test_positions,
            test_blocks=(test_block,),
            embargo_sessions=self.embargo_sessions,
            include_trace=self.include_exclusion_trace,
        )
        train_positions = exclusions.train_positions
        train_session_count = len(
            {dataset.sessions[index] for index in train_positions}
        )
        summary = exclusions.summary
        trace = exclusions.trace
        test_sample_ids = tuple(dataset.sample_ids[index] for index in test_positions)
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
        if reasons:
            return InvalidFold(
                fold_index=fold_index,
                reasons=tuple(reasons),
                test_sample_ids=test_sample_ids,
                test_positions=test_positions,
                exclusion_summary=summary,
                exclusion_trace=trace,
            )

        split_id = canonical_digest(
            {
                "dataset_digest": dataset.digest,
                "split_spec_digest": self.digest,
                "fold_index": fold_index,
                "train_sample_ids": [
                    canonical_identity(dataset.sample_ids[index])
                    for index in train_positions
                ],
                "test_sample_ids": [
                    canonical_identity(dataset.sample_ids[index])
                    for index in test_positions
                ],
            }
        )
        return FoldAssignment(
            fold_index=fold_index,
            train_sample_ids=tuple(
                dataset.sample_ids[index] for index in train_positions
            ),
            test_sample_ids=test_sample_ids,
            train_positions=np.asarray(train_positions, dtype=np.int64),
            test_positions=np.asarray(test_positions, dtype=np.int64),
            test_blocks=(test_block,),
            exclusion_summary=summary,
            dataset_digest=dataset.digest,
            split_spec_digest=self.digest,
            split_id=split_id,
            exclusion_trace=trace,
        )
