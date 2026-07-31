"""Deterministic Purged K-Fold diagnostic planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np

from ..domain import (
    ExclusionSummary,
    ExclusionRecord,
    FoldAssignment,
    InformationInterval,
    InvalidFold,
    SplitPlan,
    TestBlock,
    ValidationDataset,
    canonical_digest,
)
from ..errors import SplitPlanError


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
        protected = tuple(
            dataset.information_intervals[index] for index in test_positions
        )
        purged_positions = {
            index
            for index in candidate_positions
            if _overlaps_any(dataset.information_intervals[index], protected)
        }
        test_block = TestBlock(
            start_session=test_sessions[0],
            end_session=test_sessions[-1],
            session_count=len(test_sessions),
        )
        embargoed_sessions = self._embargoed_sessions(
            dataset, test_positions, (test_block,)
        )
        embargoed_positions = {
            index
            for index in candidate_positions
            if index not in purged_positions
            and dataset.sessions[index] in embargoed_sessions
        }
        train_positions = tuple(
            index
            for index in candidate_positions
            if index not in purged_positions and index not in embargoed_positions
        )
        train_session_count = len(
            {dataset.sessions[index] for index in train_positions}
        )
        summary = ExclusionSummary(
            candidates=len(candidate_positions),
            purged=len(purged_positions),
            embargoed=len(embargoed_positions),
            retained=len(train_positions),
        )
        trace = self._exclusion_trace(
            dataset, candidate_positions, purged_positions, embargoed_positions
        )
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
                    _identity_text(dataset.sample_ids[index])
                    for index in train_positions
                ],
                "test_sample_ids": [
                    _identity_text(dataset.sample_ids[index])
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

    def _embargoed_sessions(
        self,
        dataset: ValidationDataset,
        test_positions: tuple[int, ...],
        test_blocks: tuple[TestBlock, ...],
    ) -> set[np.datetime64]:
        if self.embargo_sessions == 0:
            return set()
        embargoed: set[np.datetime64] = set()
        axis = np.asarray(dataset.session_axis, dtype="datetime64[ns]")
        for block in test_blocks:
            block_positions = tuple(
                position
                for position in test_positions
                if block.start_session
                <= dataset.sessions[position]
                <= block.end_session
            )
            if not block_positions:
                continue
            latest_information_end = max(
                dataset.information_intervals[position].end
                for position in block_positions
            )
            start = int(np.searchsorted(axis, latest_information_end, side="right"))
            stop = min(start + self.embargo_sessions, len(axis))
            embargoed.update(np.datetime64(value, "ns") for value in axis[start:stop])
        return embargoed

    def _exclusion_trace(
        self,
        dataset: ValidationDataset,
        candidate_positions: tuple[int, ...],
        purged_positions: set[int],
        embargoed_positions: set[int],
    ) -> tuple[ExclusionRecord, ...] | None:
        if not self.include_exclusion_trace:
            return None
        records: list[ExclusionRecord] = []
        for position in candidate_positions:
            reason: str | None = None
            if position in purged_positions:
                reason = "purge-overlap"
            elif position in embargoed_positions:
                reason = "embargo"
            if reason is not None:
                records.append(
                    ExclusionRecord(
                        sample_id=dataset.sample_ids[position],
                        position=position,
                        reason=reason,
                    )
                )
        return tuple(records)


def _overlaps_any(
    candidate: InformationInterval, protected: tuple[InformationInterval, ...]
) -> bool:
    return any(candidate.overlaps(interval) for interval in protected)


def _identity_text(value: Hashable) -> dict[str, str]:
    return {"type": type(value).__qualname__, "value": str(value)}
