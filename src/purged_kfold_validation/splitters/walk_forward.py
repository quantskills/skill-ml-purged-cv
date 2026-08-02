"""Deterministic causal Walk-Forward diagnostic planning."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class CausalWalkForward:
    """Plan past-only expanding or sliding causal validation assignments."""

    n_splits: int = 5
    test_sessions: int = 1
    min_train_sessions: int = 1
    min_train_samples: int = 1
    min_test_sessions: int = 1
    pre_test_gap_sessions: int = 0
    max_train_sessions: int | None = None
    include_exclusion_trace: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("n_splits", self.n_splits),
            ("test_sessions", self.test_sessions),
            ("min_train_sessions", self.min_train_sessions),
            ("min_train_samples", self.min_train_samples),
            ("min_test_sessions", self.min_test_sessions),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise SplitPlanError(f"{name} must be an integer of at least 1")
        if (
            isinstance(self.pre_test_gap_sessions, bool)
            or not isinstance(self.pre_test_gap_sessions, int)
            or self.pre_test_gap_sessions < 0
        ):
            raise SplitPlanError("pre_test_gap_sessions must be a non-negative integer")
        if self.max_train_sessions is not None and (
            isinstance(self.max_train_sessions, bool)
            or not isinstance(self.max_train_sessions, int)
            or self.max_train_sessions < 1
        ):
            raise SplitPlanError(
                "max_train_sessions must be None or an integer of at least 1"
            )
        if not isinstance(self.include_exclusion_trace, bool):
            raise SplitPlanError("include_exclusion_trace must be a boolean")

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "kind": "causal-walk-forward",
                "n_splits": self.n_splits,
                "test_sessions": self.test_sessions,
                "min_train_sessions": self.min_train_sessions,
                "min_train_samples": self.min_train_samples,
                "min_test_sessions": self.min_test_sessions,
                "pre_test_gap_sessions": self.pre_test_gap_sessions,
                "max_train_sessions": self.max_train_sessions,
                "include_exclusion_trace": self.include_exclusion_trace,
                "schema_version": "1",
            }
        )

    def plan(self, dataset: ValidationDataset) -> SplitPlan:
        """Return chronological past-only fold candidates without scoring."""

        session_values = set(dataset.sessions)
        active_sessions = tuple(
            session for session in dataset.session_axis if session in session_values
        )
        required_test_sessions = self.n_splits * self.test_sessions
        if required_test_sessions >= len(active_sessions):
            raise SplitPlanError(
                "walk-forward requires at least one active training session before "
                "the requested test blocks"
            )
        first_test = len(active_sessions) - required_test_sessions
        folds: list[FoldAssignment | InvalidFold] = []
        for fold_index in range(self.n_splits):
            start = first_test + fold_index * self.test_sessions
            stop = start + self.test_sessions
            test_sessions = active_sessions[start:stop]
            candidate_sessions = active_sessions[:start]
            if self.max_train_sessions is not None:
                candidate_sessions = candidate_sessions[-self.max_train_sessions :]
            folds.append(
                self._plan_fold(
                    dataset,
                    fold_index=fold_index,
                    test_sessions=test_sessions,
                    candidate_sessions=candidate_sessions,
                )
            )

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
        *,
        fold_index: int,
        test_sessions: tuple[np.datetime64, ...],
        candidate_sessions: tuple[np.datetime64, ...],
    ) -> FoldAssignment | InvalidFold:
        test_values = set(test_sessions)
        candidate_values = set(candidate_sessions)
        test_positions = tuple(
            index
            for index, session in enumerate(dataset.sessions)
            if session in test_values
        )
        candidate_positions = tuple(
            index
            for index, session in enumerate(dataset.sessions)
            if session in candidate_values
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
            embargo_sessions=0,
            pre_test_gap_sessions=self.pre_test_gap_sessions,
            require_information_before=test_block.start_session,
            include_trace=self.include_exclusion_trace,
        )
        train_positions = exclusions.train_positions
        train_session_count = len(
            {dataset.sessions[index] for index in train_positions}
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
        test_sample_ids = tuple(dataset.sample_ids[index] for index in test_positions)
        if reasons:
            return InvalidFold(
                fold_index=fold_index,
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
            exclusion_summary=exclusions.summary,
            dataset_digest=dataset.digest,
            split_spec_digest=self.digest,
            split_id=split_id,
            evidence_channel=EvidenceChannel.CAUSAL_WALK_FORWARD,
            exclusion_trace=exclusions.trace,
        )
