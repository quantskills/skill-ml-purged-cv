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
) -> ExclusionResult:
    """Purge exact overlaps, then Embargo each contiguous TestBlock."""

    protected = tuple(dataset.information_intervals[index] for index in test_positions)
    purged_positions = {
        index
        for index in candidate_positions
        if _overlaps_any(dataset.information_intervals[index], protected)
    }
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
    train_positions = tuple(
        index
        for index in candidate_positions
        if index not in purged_positions and index not in embargoed_positions
    )
    summary = ExclusionSummary(
        candidates=len(candidate_positions),
        purged=len(purged_positions),
        embargoed=len(embargoed_positions),
        retained=len(train_positions),
    )
    return ExclusionResult(
        train_positions=train_positions,
        summary=summary,
        trace=_build_trace(
            dataset,
            candidate_positions=candidate_positions,
            purged_positions=purged_positions,
            embargoed_positions=embargoed_positions,
            include_trace=include_trace,
        ),
    )


def _overlaps_any(
    candidate: InformationInterval, protected: tuple[InformationInterval, ...]
) -> bool:
    return any(candidate.overlaps(interval) for interval in protected)


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


def _build_trace(
    dataset: ValidationDataset,
    *,
    candidate_positions: tuple[int, ...],
    purged_positions: set[int],
    embargoed_positions: set[int],
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
