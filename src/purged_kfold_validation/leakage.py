"""Information Interval Purge and Session Axis Embargo semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import (
    ExclusionRecord,
    ExclusionSummary,
    InformationInterval,
    TestBlock,
    ValidationDataset,
)


@dataclass(frozen=True, slots=True)
class ExclusionResult:
    """Deterministic output of applying leakage exclusions to one candidate fold."""

    train_positions: tuple[int, ...]
    summary: ExclusionSummary
    trace: tuple[ExclusionRecord, ...] | None


def apply_leakage_exclusions(
    dataset: ValidationDataset,
    *,
    candidate_positions: tuple[int, ...],
    test_positions: tuple[int, ...],
    test_blocks: tuple[TestBlock, ...],
    embargo_sessions: int,
    include_trace: bool,
    pre_test_gap_sessions: int = 0,
    require_information_before: np.datetime64 | None = None,
) -> ExclusionResult:
    """Purge exact overlaps, then apply declared session-axis gap policies."""

    purged_positions = overlapping_candidate_positions(
        dataset,
        candidate_positions=candidate_positions,
        protected_positions=test_positions,
    )
    embargoed_session_values = _embargoed_sessions(
        dataset,
        test_positions=test_positions,
        test_blocks=test_blocks,
        embargo_sessions=embargo_sessions,
    )
    embargoed_positions = {
        index
        for index in candidate_positions
        if index not in purged_positions
        and dataset.sessions[index] in embargoed_session_values
    }
    noncausal_positions = {
        index
        for index in candidate_positions
        if index not in purged_positions
        and index not in embargoed_positions
        and require_information_before is not None
        and dataset.information_intervals[index].end >= require_information_before
    }
    pre_test_gap_values = _pre_test_gap_sessions(
        dataset,
        test_blocks=test_blocks,
        gap_sessions=pre_test_gap_sessions,
    )
    pre_test_gapped_positions = {
        index
        for index in candidate_positions
        if index not in purged_positions
        and index not in embargoed_positions
        and index not in noncausal_positions
        and dataset.sessions[index] in pre_test_gap_values
    }
    train_positions = tuple(
        index
        for index in candidate_positions
        if index not in purged_positions
        and index not in embargoed_positions
        and index not in noncausal_positions
        and index not in pre_test_gapped_positions
    )
    summary = ExclusionSummary(
        candidates=len(candidate_positions),
        purged=len(purged_positions),
        embargoed=len(embargoed_positions),
        retained=len(train_positions),
        pre_test_gapped=len(pre_test_gapped_positions),
        noncausal=len(noncausal_positions),
    )
    return ExclusionResult(
        train_positions=train_positions,
        summary=summary,
        trace=_build_trace(
            dataset,
            candidate_positions=candidate_positions,
            purged_positions=purged_positions,
            embargoed_positions=embargoed_positions,
            noncausal_positions=noncausal_positions,
            pre_test_gapped_positions=pre_test_gapped_positions,
            include_trace=include_trace,
        ),
    )


def overlapping_candidate_positions(
    dataset: ValidationDataset,
    *,
    candidate_positions: tuple[int, ...] | np.ndarray,
    protected_positions: tuple[int, ...] | np.ndarray,
) -> set[int]:
    """Return candidates overlapping any protected interval in near-linear time."""

    protected = tuple(
        dataset.information_intervals[int(position)] for position in protected_positions
    )
    merged = _merged_intervals(protected)
    if not merged:
        return set()
    starts = np.asarray([interval.start for interval in merged], dtype="datetime64[ns]")
    ends = np.asarray([interval.end for interval in merged], dtype="datetime64[ns]")
    overlapping: set[int] = set()
    for raw_position in candidate_positions:
        position = int(raw_position)
        candidate = dataset.information_intervals[position]
        index = int(np.searchsorted(starts, candidate.end, side="right")) - 1
        if index >= 0 and ends[index] >= candidate.start:
            overlapping.add(position)
    return overlapping


def _merged_intervals(
    intervals: tuple[InformationInterval, ...],
) -> tuple[InformationInterval, ...]:
    if not intervals:
        return ()
    ordered = sorted(intervals, key=lambda interval: (interval.start, interval.end))
    merged: list[InformationInterval] = [ordered[0]]
    for interval in ordered[1:]:
        previous = merged[-1]
        if interval.start <= previous.end:
            if interval.end > previous.end:
                merged[-1] = InformationInterval(previous.start, interval.end)
        else:
            merged.append(interval)
    return tuple(merged)


def _embargoed_sessions(
    dataset: ValidationDataset,
    *,
    test_positions: tuple[int, ...],
    test_blocks: tuple[TestBlock, ...],
    embargo_sessions: int,
) -> set[np.datetime64]:
    if embargo_sessions == 0:
        return set()
    embargoed: set[np.datetime64] = set()
    axis = np.asarray(dataset.session_axis, dtype="datetime64[ns]")
    for block in test_blocks:
        block_positions = tuple(
            position
            for position in test_positions
            if block.start_session <= dataset.sessions[position] <= block.end_session
        )
        if not block_positions:
            continue
        latest_information_end = max(
            dataset.information_intervals[position].end for position in block_positions
        )
        start = int(np.searchsorted(axis, latest_information_end, side="right"))
        stop = min(start + embargo_sessions, len(axis))
        embargoed.update(np.datetime64(value, "ns") for value in axis[start:stop])
    return embargoed


def _pre_test_gap_sessions(
    dataset: ValidationDataset,
    *,
    test_blocks: tuple[TestBlock, ...],
    gap_sessions: int,
) -> set[np.datetime64]:
    if gap_sessions == 0:
        return set()
    axis = np.asarray(dataset.session_axis, dtype="datetime64[ns]")
    excluded: set[np.datetime64] = set()
    for block in test_blocks:
        stop = int(np.searchsorted(axis, block.start_session, side="left"))
        start = max(0, stop - gap_sessions)
        excluded.update(np.datetime64(value, "ns") for value in axis[start:stop])
    return excluded


def _build_trace(
    dataset: ValidationDataset,
    *,
    candidate_positions: tuple[int, ...],
    purged_positions: set[int],
    embargoed_positions: set[int],
    noncausal_positions: set[int],
    pre_test_gapped_positions: set[int],
    include_trace: bool,
) -> tuple[ExclusionRecord, ...] | None:
    if not include_trace:
        return None
    records: list[ExclusionRecord] = []
    for position in candidate_positions:
        if position in purged_positions:
            reason = "purge-overlap"
        elif position in embargoed_positions:
            reason = "embargo"
        elif position in noncausal_positions:
            reason = "causal-future-information"
        elif position in pre_test_gapped_positions:
            reason = "pre-test-gap"
        else:
            continue
        records.append(
            ExclusionRecord(
                sample_id=dataset.sample_ids[position],
                position=position,
                reason=reason,
            )
        )
    return tuple(records)
