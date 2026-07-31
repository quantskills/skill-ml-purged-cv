"""Immutable public evidence values used by the validation core."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Callable, Hashable, Iterable, Mapping
from uuid import UUID

import numpy as np
import numpy.typing as npt

from .errors import (
    DatasetValidationError,
    InvalidFoldError,
    TemporalValidationError,
)


SCHEMA_VERSION = "1"


class EvidenceChannel(str, Enum):
    """Purpose boundary carried by validation evidence."""

    MODEL_SELECTION = "model-selection"


class MissingValuePolicy(str, Enum):
    """Declared handling of missing/non-finite model inputs."""

    REJECT = "reject"


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


def _deeply_frozen_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _deeply_frozen_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deeply_frozen_json(item) for item in value)
    return value


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
    if isinstance(value, np.integer) and not isinstance(value, np.bool_):
        value = int(value)
    if isinstance(value, str):
        if not value:
            raise DatasetValidationError(f"{field_name} contains an empty identity")
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, UUID):
        return value
    if isinstance(value, tuple):
        return tuple(_identity(item, field_name=field_name) for item in value)
    raise DatasetValidationError(
        f"{field_name} contains an unsupported stable identity type"
    )


def canonical_identity(value: Hashable) -> dict[str, Any]:
    """Encode a validated stable identity without process-local representations."""

    normalized = _identity(value, field_name="identity")
    if isinstance(normalized, str):
        return {"type": "string", "value": normalized}
    if isinstance(normalized, int):
        return {"type": "integer", "value": normalized}
    if isinstance(normalized, UUID):
        return {"type": "uuid", "value": str(normalized)}
    if isinstance(normalized, tuple):
        return {
            "type": "tuple",
            "value": [canonical_identity(item) for item in normalized],
        }
    raise AssertionError("validated identity was not normalized")


def _readonly_array(
    value: npt.ArrayLike,
    *,
    field_name: str,
    ndim: int | None = None,
    dtype: npt.DTypeLike | None = None,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=dtype, copy=True)
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(f"{field_name} contains invalid values") from exc
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


@dataclass(frozen=True, slots=True, init=False)
class ValidationDataset:
    """Canonical immutable input for diagnostic splitting and formal evaluation."""

    sample_ids: tuple[Hashable, ...]
    session_axis: tuple[np.datetime64, ...]
    sessions: tuple[np.datetime64, ...]
    information_intervals: tuple[InformationInterval, ...]
    features: np.ndarray
    targets: np.ndarray
    asset_ids: tuple[Hashable, ...] | None
    decision_times: np.ndarray | None
    feature_availability: np.ndarray | None
    pit_snapshot: PITSnapshot | None
    missing_value_policy: MissingValuePolicy
    digest: str = field(init=False)

    def __init__(
        self,
        *,
        sample_ids: Iterable[Hashable],
        session_axis: Iterable[Any],
        sessions: Iterable[Any],
        information_intervals: Iterable[InformationInterval],
        features: npt.ArrayLike,
        targets: npt.ArrayLike,
        asset_ids: Iterable[Hashable] | None = None,
        decision_times: Iterable[Any] | None = None,
        feature_availability: npt.ArrayLike | None = None,
        pit_snapshot: PITSnapshot | None = None,
        missing_value_policy: MissingValuePolicy | str = MissingValuePolicy.REJECT,
    ) -> None:
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "session_axis", session_axis)
        object.__setattr__(self, "sessions", sessions)
        object.__setattr__(self, "information_intervals", information_intervals)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "asset_ids", asset_ids)
        object.__setattr__(self, "decision_times", decision_times)
        object.__setattr__(self, "feature_availability", feature_availability)
        object.__setattr__(self, "pit_snapshot", pit_snapshot)
        object.__setattr__(self, "missing_value_policy", missing_value_policy)
        self.__post_init__()

    def __post_init__(self) -> None:
        from .validation import normalize_validation_dataset

        for name, value in normalize_validation_dataset(self).items():
            object.__setattr__(self, name, value)

    @property
    def formal_scoring_issues(self) -> tuple[str, ...]:
        """Return metadata-only reasons this dataset cannot produce a formal score."""

        from .validation import formal_scoring_issues

        return formal_scoring_issues(self)

    def require_formal_scoring(self) -> None:
        """Fail closed when temporal feature evidence is not point-in-time safe."""

        from .validation import require_formal_scoring

        require_formal_scoring(self)


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
    evidence_channel: EvidenceChannel = EvidenceChannel.MODEL_SELECTION
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
            normalized_parameters = json.loads(
                json.dumps(
                    parameters,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                )
            )
            digest = canonical_digest(
                {
                    "kind": "model-spec",
                    "name": self.name,
                    "version": self.version,
                    "parameters": normalized_parameters,
                }
            )
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(
                "model parameters must be canonical JSON values"
            ) from exc
        object.__setattr__(
            self, "parameters", _deeply_frozen_json(normalized_parameters)
        )
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
    evidence_channel: EvidenceChannel = EvidenceChannel.MODEL_SELECTION


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
                "sample_id": canonical_identity(item.sample_id),
                "session": _time_text(item.session),
                "asset_id": (
                    None if item.asset_id is None else canonical_identity(item.asset_id)
                ),
                "fold_index": item.fold_index,
                "split_id": item.split_id,
                "target": item.target,
                "prediction": item.prediction,
                "dataset_digest": item.dataset_digest,
                "split_spec_digest": item.split_spec_digest,
                "model_digest": item.model_digest,
                "pit_snapshot_digest": item.pit_snapshot_digest,
                "evidence_channel": item.evidence_channel.value,
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
    observation_coverage: float
    fold_count: int
    fold_coverage: float
    ledger_digest: str
    metric_digest: str

    @property
    def coverage(self) -> float:
        """Backward-compatible alias for observation coverage."""

        return self.observation_coverage


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Complete formal model-selection evidence for one evaluation run."""

    run_id: str
    plan_digest: str
    ledger: OOSLedger
    metrics: tuple[DerivedMetric, ...]
    evidence_channel: EvidenceChannel = EvidenceChannel.MODEL_SELECTION
