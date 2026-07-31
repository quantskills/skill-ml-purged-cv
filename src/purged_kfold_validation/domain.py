"""Immutable public evidence values used by the validation core."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Callable, Hashable, Iterable, Mapping

import numpy as np
import numpy.typing as npt

from .errors import (
    DatasetValidationError,
    InvalidFoldError,
    PointInTimeValidationError,
    TemporalValidationError,
)


SCHEMA_VERSION = "1"


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for normalized JSON-compatible evidence."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _time(value: Any, *, field_name: str) -> np.datetime64:
    try:
        normalized = np.datetime64(value, "ns")
    except (TypeError, ValueError) as exc:
        raise TemporalValidationError(f"{field_name} contains an invalid time") from exc
    if np.isnat(normalized):
        raise TemporalValidationError(f"{field_name} contains NaT")
    return normalized


def _time_text(value: np.datetime64) -> str:
    return np.datetime_as_string(value.astype("datetime64[ns]"), unit="ns")


def _identity(value: Hashable, *, field_name: str) -> Hashable:
    if value is None:
        raise DatasetValidationError(f"{field_name} contains a null identity")
    try:
        hash(value)
    except TypeError as exc:
        raise DatasetValidationError(
            f"{field_name} contains an unhashable identity"
        ) from exc
    if isinstance(value, str) and not value:
        raise DatasetValidationError(f"{field_name} contains an empty identity")
    return value


def _identity_json(value: Hashable) -> dict[str, str]:
    return {"type": type(value).__qualname__, "value": str(value)}


def _readonly_array(
    value: npt.ArrayLike,
    *,
    field_name: str,
    ndim: int | None = None,
    dtype: npt.DTypeLike | None = None,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise DatasetValidationError(f"{field_name} must have {ndim} dimensions")
    array.setflags(write=False)
    return array


def _array_json(array: np.ndarray) -> dict[str, Any]:
    values: list[Any] = []
    for raw in array.reshape(-1).tolist():
        if isinstance(raw, float):
            if np.isnan(raw):
                values.append({"float": "nan"})
                continue
            if np.isposinf(raw):
                values.append({"float": "+inf"})
                continue
            if np.isneginf(raw):
                values.append({"float": "-inf"})
                continue
        values.append(raw)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "values": values}


def _time_array_json(array: np.ndarray | None) -> Any:
    if array is None:
        return None
    return {
        "shape": list(array.shape),
        "values": [
            "NaT" if np.isnat(value) else _time_text(value)
            for value in array.reshape(-1)
        ],
    }


@dataclass(frozen=True, slots=True)
class InformationInterval:
    """Inclusive information interval associated with one sample."""

    start: Any
    end: Any

    def __post_init__(self) -> None:
        start = _time(self.start, field_name="information interval start")
        end = _time(self.end, field_name="information interval end")
        if start > end:
            raise TemporalValidationError(
                "information interval start must not be after end"
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def overlaps(self, other: InformationInterval) -> bool:
        """Return whether two inclusive intervals share any information time."""

        return bool(self.start <= other.end and other.start <= self.end)

    def canonical(self) -> dict[str, str]:
        return {"start": _time_text(self.start), "end": _time_text(self.end)}


@dataclass(frozen=True, slots=True)
class PITSnapshot:
    """Declared point-in-time source state used to construct feature values."""

    snapshot_id: str
    source_digest: str
    revision_policy: str = "point-in-time"
    provenance_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, str) or not isinstance(
            self.source_digest, str
        ):
            raise DatasetValidationError(
                "PIT Snapshot identity and source digest must be strings"
            )
        if not isinstance(self.revision_policy, str):
            raise DatasetValidationError(
                "PIT Snapshot revision policy must be a string"
            )
        object.__setattr__(
            self,
            "provenance_digest",
            canonical_digest(
                {
                    "snapshot_id": self.snapshot_id,
                    "source_digest": self.source_digest,
                    "revision_policy": self.revision_policy,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidationDataset:
    """Canonical immutable input for diagnostic splitting and formal evaluation."""

    sample_ids: Iterable[Hashable]
    session_axis: Iterable[Any]
    sessions: Iterable[Any]
    information_intervals: Iterable[InformationInterval]
    features: npt.ArrayLike
    targets: npt.ArrayLike
    asset_ids: Iterable[Hashable] | None = None
    decision_times: Iterable[Any] | None = None
    feature_availability: npt.ArrayLike | None = None
    pit_snapshot: PITSnapshot | None = None
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        sample_ids = tuple(
            _identity(value, field_name="sample_ids") for value in self.sample_ids
        )
        if len(set(sample_ids)) != len(sample_ids):
            raise DatasetValidationError("sample_ids contains duplicate identities")
        if not sample_ids:
            raise DatasetValidationError(
                "ValidationDataset must contain at least one sample"
            )

        session_axis = tuple(
            _time(value, field_name="session_axis") for value in self.session_axis
        )
        if not session_axis:
            raise TemporalValidationError("session_axis must not be empty")
        if any(left >= right for left, right in zip(session_axis, session_axis[1:])):
            raise TemporalValidationError(
                "session_axis must be unique and strictly increasing"
            )

        sessions = tuple(_time(value, field_name="sessions") for value in self.sessions)
        intervals = tuple(self.information_intervals)
        if not all(isinstance(interval, InformationInterval) for interval in intervals):
            raise DatasetValidationError(
                "information_intervals must contain InformationInterval values"
            )
        features = _readonly_array(self.features, field_name="features", ndim=2)
        targets = _readonly_array(self.targets, field_name="targets", ndim=1)

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
            raise TemporalValidationError(
                "sessions contains a value outside session_axis"
            )
        session_positions = tuple(axis_positions[session] for session in sessions)
        if any(
            left > right
            for left, right in zip(session_positions, session_positions[1:])
        ):
            raise TemporalValidationError("samples must be ordered by session_axis")

        asset_ids: tuple[Hashable, ...] | None = None
        if self.asset_ids is not None:
            asset_ids = tuple(
                _identity(value, field_name="asset_ids") for value in self.asset_ids
            )
            if len(asset_ids) != len(sample_ids):
                raise DatasetValidationError(
                    "row-aligned fields have inconsistent lengths: asset_ids"
                )

        decision_times: np.ndarray | None = None
        if self.decision_times is not None:
            decision_times = np.array(
                self.decision_times, dtype="datetime64[ns]", copy=True
            )
            if decision_times.ndim != 1 or decision_times.shape[0] != len(sample_ids):
                raise DatasetValidationError(
                    "decision_times must be one-dimensional and row-aligned"
                )
            decision_times.setflags(write=False)

        feature_availability: np.ndarray | None = None
        if self.feature_availability is not None:
            feature_availability = np.array(
                self.feature_availability, dtype="datetime64[ns]", copy=True
            )
            valid_shape = (
                feature_availability.ndim == 1
                and feature_availability.shape[0] == len(sample_ids)
            ) or feature_availability.shape == features.shape
            if not valid_shape:
                raise DatasetValidationError(
                    "feature_availability must be row-aligned or match features"
                )
            feature_availability.setflags(write=False)

        if self.pit_snapshot is not None and not isinstance(
            self.pit_snapshot, PITSnapshot
        ):
            raise DatasetValidationError("pit_snapshot must be a PITSnapshot")

        normalized = {
            "schema_version": SCHEMA_VERSION,
            "sample_ids": [_identity_json(value) for value in sample_ids],
            "session_axis": [_time_text(value) for value in session_axis],
            "sessions": [_time_text(value) for value in sessions],
            "information_intervals": [interval.canonical() for interval in intervals],
            "features": _array_json(features),
            "targets": _array_json(targets),
            "asset_ids": (
                None
                if asset_ids is None
                else [_identity_json(value) for value in asset_ids]
            ),
            "decision_times": _time_array_json(decision_times),
            "feature_availability": _time_array_json(feature_availability),
            "pit_snapshot": (
                None
                if self.pit_snapshot is None
                else {
                    "snapshot_id": self.pit_snapshot.snapshot_id,
                    "source_digest": self.pit_snapshot.source_digest,
                    "revision_policy": self.pit_snapshot.revision_policy,
                }
            ),
        }

        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "session_axis", session_axis)
        object.__setattr__(self, "sessions", sessions)
        object.__setattr__(self, "information_intervals", intervals)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "asset_ids", asset_ids)
        object.__setattr__(self, "decision_times", decision_times)
        object.__setattr__(self, "feature_availability", feature_availability)
        object.__setattr__(self, "digest", canonical_digest(normalized))

    @property
    def formal_scoring_issues(self) -> tuple[str, ...]:
        """Return metadata-only reasons this dataset cannot produce a formal score."""

        issues: list[str] = []
        if self.decision_times is None:
            issues.append("Decision Time evidence is missing")
        elif bool(np.isnat(self.decision_times).any()):
            issues.append("Decision Time evidence contains missing times")

        if self.feature_availability is None:
            issues.append("Feature Availability evidence is missing")
        elif bool(np.isnat(self.feature_availability).any()):
            issues.append("Feature Availability evidence contains missing times")

        if self.pit_snapshot is None:
            issues.append("PIT Snapshot provenance is missing")
        else:
            if not self.pit_snapshot.snapshot_id:
                issues.append("PIT Snapshot identity is missing")
            if not self.pit_snapshot.source_digest:
                issues.append("PIT Snapshot source digest is missing")
            if self.pit_snapshot.revision_policy != "point-in-time":
                issues.append("PIT Snapshot revision policy is not point-in-time")

        if self.decision_times is not None and self.feature_availability is not None:
            decisions = self.decision_times
            availability = self.feature_availability
            if availability.ndim == 2:
                decisions = decisions[:, np.newaxis]
            late_rows = (
                np.nonzero(np.any(availability > decisions, axis=1))[0]
                if availability.ndim == 2
                else np.nonzero(availability > decisions)[0]
            )
            if len(late_rows):
                issues.append(
                    "Feature Availability is later than Decision Time at sample positions "
                    + ",".join(str(int(position)) for position in late_rows)
                )
        return tuple(issues)

    def require_formal_scoring(self) -> None:
        """Fail closed when temporal feature evidence is not point-in-time safe."""

        issues = self.formal_scoring_issues
        if issues:
            raise PointInTimeValidationError(
                "formal scoring rejected dataset: " + "; ".join(issues)
            )


@dataclass(frozen=True, slots=True)
class ExclusionSummary:
    """Compact counts for candidate training exclusions."""

    candidates: int
    purged: int
    embargoed: int
    retained: int


@dataclass(frozen=True, slots=True)
class TestBlock:
    """One contiguous test block on the Session Axis."""

    start_session: np.datetime64
    end_session: np.datetime64
    session_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_session",
            _time(self.start_session, field_name="test block start"),
        )
        object.__setattr__(
            self, "end_session", _time(self.end_session, field_name="test block end")
        )


@dataclass(frozen=True, slots=True)
class ExclusionRecord:
    """Optional audit record for one excluded candidate training sample."""

    sample_id: Hashable
    position: int
    reason: str


@dataclass(frozen=True, slots=True, eq=False)
class FoldAssignment:
    """Immutable evidence-bearing train/test assignment."""

    fold_index: int
    train_sample_ids: tuple[Hashable, ...]
    test_sample_ids: tuple[Hashable, ...]
    train_positions: np.ndarray
    test_positions: np.ndarray
    test_blocks: tuple[TestBlock, ...]
    exclusion_summary: ExclusionSummary
    dataset_digest: str
    split_spec_digest: str
    split_id: str
    evidence_channel: str = "model-selection"
    schema_version: str = SCHEMA_VERSION
    exclusion_trace: tuple[ExclusionRecord, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "train_positions",
            _readonly_array(
                self.train_positions,
                field_name="train_positions",
                ndim=1,
                dtype=np.int64,
            ),
        )
        object.__setattr__(
            self,
            "test_positions",
            _readonly_array(
                self.test_positions,
                field_name="test_positions",
                ndim=1,
                dtype=np.int64,
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FoldAssignment):
            return NotImplemented
        return (
            self.fold_index == other.fold_index
            and self.train_sample_ids == other.train_sample_ids
            and self.test_sample_ids == other.test_sample_ids
            and np.array_equal(self.train_positions, other.train_positions)
            and np.array_equal(self.test_positions, other.test_positions)
            and self.test_blocks == other.test_blocks
            and self.exclusion_summary == other.exclusion_summary
            and self.dataset_digest == other.dataset_digest
            and self.split_spec_digest == other.split_spec_digest
            and self.split_id == other.split_id
            and self.evidence_channel == other.evidence_channel
            and self.schema_version == other.schema_version
            and self.exclusion_trace == other.exclusion_trace
        )


@dataclass(frozen=True, slots=True)
class InvalidFold:
    """Diagnostic candidate that cannot contribute formal evidence."""

    fold_index: int
    reasons: tuple[str, ...]
    test_sample_ids: tuple[Hashable, ...]
    test_positions: tuple[int, ...]
    exclusion_summary: ExclusionSummary
    exclusion_trace: tuple[ExclusionRecord, ...] | None = None


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Read-only diagnostic projection of all requested folds."""

    folds: tuple[FoldAssignment | InvalidFold, ...]
    dataset_digest: str
    split_spec_digest: str
    digest: str

    @property
    def invalid_folds(self) -> tuple[InvalidFold, ...]:
        return tuple(fold for fold in self.folds if isinstance(fold, InvalidFold))

    @property
    def assignments(self) -> tuple[FoldAssignment, ...]:
        return tuple(fold for fold in self.folds if isinstance(fold, FoldAssignment))

    def require_assignments(self) -> tuple[FoldAssignment, ...]:
        """Return formal assignments only when every requested fold is valid."""

        if self.invalid_folds:
            details = "; ".join(
                f"fold {fold.fold_index}: {', '.join(fold.reasons)}"
                for fold in self.invalid_folds
            )
            raise InvalidFoldError(
                f"formal assignments rejected invalid folds: {details}"
            )
        return self.assignments


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Versioned identity for one fixed model configuration."""

    name: str
    version: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise DatasetValidationError("model name and version must not be empty")
        parameters = dict(self.parameters)
        try:
            digest = canonical_digest(
                {
                    "kind": "model-spec",
                    "name": self.name,
                    "version": self.version,
                    "parameters": parameters,
                }
            )
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(
                "model parameters must be canonical JSON values"
            ) from exc
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "digest", digest)


MetricFunction = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Named, versioned pure projection from OOS observations."""

    name: str
    version: str
    function: MetricFunction = field(compare=False, repr=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise DatasetValidationError("metric name and version must not be empty")
        if not callable(self.function):
            raise DatasetValidationError("metric function must be callable")
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {"kind": "metric", "name": self.name, "version": self.version}
            ),
        )


@dataclass(frozen=True, slots=True)
class OOSObservation:
    """One authoritative out-of-sample fact."""

    run_id: str
    sample_id: Hashable
    session: np.datetime64
    asset_id: Hashable | None
    fold_index: int
    split_id: str
    target: float
    prediction: float
    dataset_digest: str
    split_spec_digest: str
    model_digest: str
    pit_snapshot_digest: str
    evidence_channel: str = "model-selection"


@dataclass(frozen=True, slots=True)
class OOSLedger:
    """Immutable source of truth for out-of-sample predictions."""

    observations: tuple[OOSObservation, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        normalized = [
            {
                "run_id": item.run_id,
                "sample_id": _identity_json(item.sample_id),
                "session": _time_text(item.session),
                "asset_id": (
                    None if item.asset_id is None else _identity_json(item.asset_id)
                ),
                "fold_index": item.fold_index,
                "split_id": item.split_id,
                "target": item.target,
                "prediction": item.prediction,
                "dataset_digest": item.dataset_digest,
                "split_spec_digest": item.split_spec_digest,
                "model_digest": item.model_digest,
                "pit_snapshot_digest": item.pit_snapshot_digest,
                "evidence_channel": item.evidence_channel,
            }
            for item in observations
        ]
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "digest", canonical_digest(normalized))

    @property
    def sample_ids(self) -> tuple[Hashable, ...]:
        return tuple(item.sample_id for item in self.observations)

    @property
    def predictions(self) -> np.ndarray:
        return _readonly_array(
            [item.prediction for item in self.observations],
            field_name="predictions",
            ndim=1,
            dtype=float,
        )

    @property
    def targets(self) -> np.ndarray:
        return _readonly_array(
            [item.target for item in self.observations],
            field_name="targets",
            ndim=1,
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class DerivedMetric:
    """Versioned metric values projected from one OOS Ledger."""

    name: str
    version: str
    overall: float
    per_fold: tuple[float, ...]
    observation_count: int
    coverage: float
    ledger_digest: str
    metric_digest: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Complete formal model-selection evidence for one evaluation run."""

    run_id: str
    plan_digest: str
    ledger: OOSLedger
    metrics: tuple[DerivedMetric, ...]
    evidence_channel: str = "model-selection"
