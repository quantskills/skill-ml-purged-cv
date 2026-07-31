"""Explicit pandas-to-ValidationDataset boundary adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable

import numpy as np
import pandas as pd

from ..domain import InformationInterval, PITSnapshot, ValidationDataset
from ..errors import AdapterValidationError


@dataclass(frozen=True, slots=True)
class PandasField:
    """An explicit reference to one DataFrame column or index level."""

    column: Hashable | None = None
    index_level: str | int | None = None

    def __post_init__(self) -> None:
        selected = int(self.column is not None) + int(self.index_level is not None)
        if selected != 1:
            raise AdapterValidationError(
                "PandasField must select exactly one column or index level"
            )


AvailabilityMapping = PandasField | tuple[PandasField, ...]


@dataclass(frozen=True, slots=True)
class PandasDatasetMapping:
    """Complete, non-inferred field map for the canonical dataset."""

    sample_id: PandasField
    session: PandasField
    interval_start: PandasField
    interval_end: PandasField
    decision_time: PandasField
    feature_availability: AvailabilityMapping | None
    features: tuple[PandasField, ...]
    target: PandasField
    asset_id: PandasField | None = None

    def __post_init__(self) -> None:
        required = (
            self.sample_id,
            self.session,
            self.interval_start,
            self.interval_end,
            self.decision_time,
            self.target,
        )
        if any(not isinstance(field, PandasField) for field in required):
            raise AdapterValidationError(
                "all required mappings must be PandasField values"
            )
        features = tuple(self.features)
        if not features or any(
            not isinstance(field, PandasField) for field in features
        ):
            raise AdapterValidationError(
                "features must explicitly map at least one field"
            )
        if self.feature_availability is None:
            raise AdapterValidationError(
                "feature_availability mapping is required; DataFrame attrs are ignored"
            )
        if isinstance(self.feature_availability, tuple):
            availability = tuple(self.feature_availability)
            if len(availability) != len(features) or any(
                not isinstance(field, PandasField) for field in availability
            ):
                raise AdapterValidationError(
                    "per-feature availability mappings must match feature mappings"
                )
            object.__setattr__(self, "feature_availability", availability)
        elif not isinstance(self.feature_availability, PandasField):
            raise AdapterValidationError(
                "feature_availability must explicitly map row or per-feature times"
            )
        if self.asset_id is not None and not isinstance(self.asset_id, PandasField):
            raise AdapterValidationError("asset_id mapping must be a PandasField")
        object.__setattr__(self, "features", features)


def validation_dataset_from_pandas(
    frame: pd.DataFrame,
    *,
    mapping: PandasDatasetMapping,
    session_axis: Iterable[Any],
    pit_snapshot: PITSnapshot,
) -> ValidationDataset:
    """Construct the canonical input using only explicitly supplied mappings."""

    if not isinstance(frame, pd.DataFrame):
        raise AdapterValidationError("frame must be a pandas DataFrame")
    if not isinstance(mapping, PandasDatasetMapping):
        raise AdapterValidationError("mapping must be a PandasDatasetMapping")
    if frame.index.has_duplicates:
        raise AdapterValidationError("DataFrame index contains duplicate entries")
    if not isinstance(pit_snapshot, PITSnapshot):
        raise AdapterValidationError("pit_snapshot must be supplied explicitly")

    sample_ids = _identities(_extract(frame, mapping.sample_id, "sample_id"))
    asset_ids = (
        None
        if mapping.asset_id is None
        else _identities(_extract(frame, mapping.asset_id, "asset_id"))
    )

    timezone_signatures: list[str | None] = []
    normalized_axis, timezone = _datetime_values(session_axis, "session_axis")
    timezone_signatures.append(timezone)
    sessions, timezone = _datetime_values(
        _extract(frame, mapping.session, "session"), "session"
    )
    timezone_signatures.append(timezone)
    starts, timezone = _datetime_values(
        _extract(frame, mapping.interval_start, "interval_start"), "interval_start"
    )
    timezone_signatures.append(timezone)
    ends, timezone = _datetime_values(
        _extract(frame, mapping.interval_end, "interval_end"), "interval_end"
    )
    timezone_signatures.append(timezone)
    decisions, timezone = _datetime_values(
        _extract(frame, mapping.decision_time, "decision_time"), "decision_time"
    )
    timezone_signatures.append(timezone)

    availability_mapping = mapping.feature_availability
    assert availability_mapping is not None
    if isinstance(availability_mapping, tuple):
        availability_columns: list[np.ndarray] = []
        for index, field in enumerate(availability_mapping):
            values, timezone = _datetime_values(
                _extract(frame, field, f"feature_availability[{index}]"),
                f"feature_availability[{index}]",
            )
            availability_columns.append(values)
            timezone_signatures.append(timezone)
        availability = np.column_stack(availability_columns)
    else:
        availability, timezone = _datetime_values(
            _extract(frame, availability_mapping, "feature_availability"),
            "feature_availability",
        )
        timezone_signatures.append(timezone)

    if len(set(timezone_signatures)) != 1:
        raise AdapterValidationError(
            "temporal mappings and session_axis must use one consistent timezone"
        )

    feature_columns = [
        np.asarray(_extract(frame, field, f"features[{index}]"))
        for index, field in enumerate(mapping.features)
    ]
    features = np.column_stack(feature_columns)
    targets = np.asarray(_extract(frame, mapping.target, "target"))

    return ValidationDataset(
        sample_ids=sample_ids,
        asset_ids=asset_ids,
        session_axis=normalized_axis,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(start, end) for start, end in zip(starts, ends)
        ),
        decision_times=decisions,
        feature_availability=availability,
        pit_snapshot=pit_snapshot,
        features=features,
        targets=targets,
    )


def _extract(frame: pd.DataFrame, field: PandasField, name: str) -> Any:
    if field.column is not None:
        if field.column not in frame.columns:
            raise AdapterValidationError(f"mapped {name} column is missing")
        values = frame[field.column]
        if isinstance(values, pd.DataFrame):
            raise AdapterValidationError(f"mapped {name} column is duplicated")
        return values
    try:
        index_level = field.index_level
        assert index_level is not None
        return frame.index.get_level_values(index_level)
    except (KeyError, IndexError, ValueError) as exc:
        raise AdapterValidationError(f"mapped {name} index level is missing") from exc


def _datetime_values(values: Any, name: str) -> tuple[np.ndarray, str | None]:
    try:
        index = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as exc:
        raise AdapterValidationError(f"mapped {name} contains invalid times") from exc
    timezone = None if index.tz is None else str(index.tz)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index.to_numpy(dtype="datetime64[ns]"), timezone


def _identities(values: Iterable[Any]) -> tuple[Hashable, ...]:
    normalized: list[Hashable] = []
    for value in values:
        normalized.append(value.item() if isinstance(value, np.generic) else value)
    return tuple(normalized)
