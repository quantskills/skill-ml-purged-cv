"""Cross-field, fail-closed Validation Dataset checks."""

from __future__ import annotations

from typing import Any, Hashable, TYPE_CHECKING

import numpy as np

from .domain import (
    InformationInterval,
    MissingValuePolicy,
    PITSnapshot,
    SCHEMA_VERSION,
    _array_json,
    _identity,
    _readonly_array,
    _time,
    _time_array_json,
    _time_text,
    canonical_digest,
    canonical_identity,
)
from .errors import (
    DatasetValidationError,
    PointInTimeValidationError,
    TemporalValidationError,
)

if TYPE_CHECKING:
    from .domain import ValidationDataset


def normalize_validation_dataset(dataset: ValidationDataset) -> dict[str, Any]:
    """Validate and normalize all row-aligned dataset evidence."""

    sample_ids = tuple(
        _identity(value, field_name="sample_ids") for value in dataset.sample_ids
    )
    if len(set(sample_ids)) != len(sample_ids):
        raise DatasetValidationError("sample_ids contains duplicate identities")
    if not sample_ids:
        raise DatasetValidationError(
            "ValidationDataset must contain at least one sample"
        )

    session_axis = tuple(
        _time(value, field_name="session_axis") for value in dataset.session_axis
    )
    if not session_axis:
        raise TemporalValidationError("session_axis must not be empty")
    if any(left >= right for left, right in zip(session_axis, session_axis[1:])):
        raise TemporalValidationError(
            "session_axis must be unique and strictly increasing"
        )

    sessions = tuple(_time(value, field_name="sessions") for value in dataset.sessions)
    intervals = tuple(dataset.information_intervals)
    if not all(isinstance(interval, InformationInterval) for interval in intervals):
        raise DatasetValidationError(
            "information_intervals must contain InformationInterval values"
        )
    features = _readonly_array(dataset.features, field_name="features", ndim=2)
    targets = _readonly_array(dataset.targets, field_name="targets", ndim=1)

    try:
        missing_value_policy = MissingValuePolicy(dataset.missing_value_policy)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(
            "missing_value_policy must be the supported 'reject' policy"
        ) from exc
    _require_finite_numeric(features, field_name="features")
    _require_finite_numeric(targets, field_name="targets")

    lengths = {
        "sample_ids": len(sample_ids),
        "sessions": len(sessions),
        "information_intervals": len(intervals),
        "features": features.shape[0],
        "targets": targets.shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise DatasetValidationError(
            "row-aligned fields have inconsistent lengths: "
            + ", ".join(f"{name}={length}" for name, length in lengths.items())
        )

    axis_positions = {session: index for index, session in enumerate(session_axis)}
    if any(session not in axis_positions for session in sessions):
        raise TemporalValidationError("sessions contains a value outside session_axis")
    session_positions = tuple(axis_positions[session] for session in sessions)
    if any(
        left > right for left, right in zip(session_positions, session_positions[1:])
    ):
        raise TemporalValidationError("samples must be ordered by session_axis")

    asset_ids = _normalize_asset_ids(dataset.asset_ids, len(sample_ids))
    decision_times = _normalize_time_array(
        dataset.decision_times,
        field_name="decision_times",
        expected_rows=len(sample_ids),
        expected_shape=None,
    )
    feature_availability = _normalize_time_array(
        dataset.feature_availability,
        field_name="feature_availability",
        expected_rows=len(sample_ids),
        expected_shape=features.shape,
    )

    if dataset.pit_snapshot is not None and not isinstance(
        dataset.pit_snapshot, PITSnapshot
    ):
        raise DatasetValidationError("pit_snapshot must be a PITSnapshot")

    normalized = {
        "schema_version": SCHEMA_VERSION,
        "sample_ids": [canonical_identity(value) for value in sample_ids],
        "session_axis": [_time_text(value) for value in session_axis],
        "sessions": [_time_text(value) for value in sessions],
        "information_intervals": [interval.canonical() for interval in intervals],
        "features": _array_json(features),
        "targets": _array_json(targets),
        "asset_ids": (
            None
            if asset_ids is None
            else [canonical_identity(value) for value in asset_ids]
        ),
        "decision_times": _time_array_json(decision_times),
        "feature_availability": _time_array_json(feature_availability),
        "pit_snapshot": (
            None
            if dataset.pit_snapshot is None
            else {
                "snapshot_id": dataset.pit_snapshot.snapshot_id,
                "source_digest": dataset.pit_snapshot.source_digest,
                "revision_policy": dataset.pit_snapshot.revision_policy,
            }
        ),
        "missing_value_policy": missing_value_policy.value,
    }
    return {
        "sample_ids": sample_ids,
        "session_axis": session_axis,
        "sessions": sessions,
        "information_intervals": intervals,
        "features": features,
        "targets": targets,
        "asset_ids": asset_ids,
        "decision_times": decision_times,
        "feature_availability": feature_availability,
        "missing_value_policy": missing_value_policy,
        "digest": canonical_digest(normalized),
    }


def formal_scoring_issues(dataset: ValidationDataset) -> tuple[str, ...]:
    """Return metadata-only reasons formal historical scoring is unavailable."""

    issues: list[str] = []
    if dataset.decision_times is None:
        issues.append("Decision Time evidence is missing")
    elif bool(np.isnat(dataset.decision_times).any()):
        issues.append("Decision Time evidence contains missing times")

    if dataset.feature_availability is None:
        issues.append("Feature Availability evidence is missing")
    elif bool(np.isnat(dataset.feature_availability).any()):
        issues.append("Feature Availability evidence contains missing times")

    if dataset.pit_snapshot is None:
        issues.append("PIT Snapshot provenance is missing")
    else:
        if not dataset.pit_snapshot.snapshot_id:
            issues.append("PIT Snapshot identity is missing")
        if not dataset.pit_snapshot.source_digest:
            issues.append("PIT Snapshot source digest is missing")
        if dataset.pit_snapshot.revision_policy != "point-in-time":
            issues.append("PIT Snapshot revision policy is not point-in-time")

    if dataset.decision_times is not None and dataset.feature_availability is not None:
        decisions = dataset.decision_times
        availability = dataset.feature_availability
        if availability.ndim == 2:
            decisions = decisions[:, np.newaxis]
            late_rows = np.nonzero(np.any(availability > decisions, axis=1))[0]
        else:
            late_rows = np.nonzero(availability > decisions)[0]
        if len(late_rows):
            issues.append(
                "Feature Availability is later than Decision Time at sample positions "
                + ",".join(str(int(position)) for position in late_rows)
            )
    return tuple(issues)


def require_formal_scoring(dataset: ValidationDataset) -> None:
    """Fail closed before any formal result can be created."""

    issues = formal_scoring_issues(dataset)
    if issues:
        raise PointInTimeValidationError(
            "formal scoring rejected dataset: " + "; ".join(issues)
        )


def _normalize_asset_ids(
    values: Any, expected_rows: int
) -> tuple[Hashable, ...] | None:
    if values is None:
        return None
    asset_ids = tuple(_identity(value, field_name="asset_ids") for value in values)
    if len(asset_ids) != expected_rows:
        raise DatasetValidationError(
            "row-aligned fields have inconsistent lengths: asset_ids"
        )
    return asset_ids


def _normalize_time_array(
    values: Any,
    *,
    field_name: str,
    expected_rows: int,
    expected_shape: tuple[int, ...] | None,
) -> np.ndarray | None:
    if values is None:
        return None
    try:
        array = np.array(values, dtype="datetime64[ns]", copy=True)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(f"{field_name} contains invalid times") from exc
    valid_shape = array.ndim == 1 and array.shape[0] == expected_rows
    if expected_shape is not None:
        valid_shape = valid_shape or array.shape == expected_shape
    if not valid_shape:
        if expected_shape is None:
            raise DatasetValidationError(
                f"{field_name} must be one-dimensional and row-aligned"
            )
        raise DatasetValidationError(
            f"{field_name} must be row-aligned or match features"
        )
    array.setflags(write=False)
    return array


def _require_finite_numeric(array: np.ndarray, *, field_name: str) -> None:
    if not np.issubdtype(array.dtype, np.number) or not bool(np.isfinite(array).all()):
        raise DatasetValidationError(
            f"{field_name} must contain finite numeric values under reject policy"
        )
