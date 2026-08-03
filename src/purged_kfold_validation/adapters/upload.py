"""Bounded local-file adapter for governed arbitrary-feature uploads."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, cast

import pandas as pd

from ..domain import PITSnapshot
from ..errors import AdapterValidationError, UploadLimitError
from ..features import (
    FeatureComputationScope,
    FeatureDefinition,
    FeatureManifest,
    GovernedFeatureDataset,
)
from .pandas import (
    PandasDatasetMapping,
    PandasField,
    governed_validation_dataset_from_pandas,
)


_CONFIG_BYTE_LIMIT = 1_048_576


@dataclass(frozen=True, slots=True)
class FeatureUploadLimits:
    """Resource ceilings applied before evaluation or evidence generation."""

    max_file_bytes: int = 536_870_912
    max_rows: int = 1_000_000
    max_features: int = 512
    max_combinations: int = 10_000
    max_columns: int = 2_048
    max_uncompressed_bytes: int = 2_147_483_648

    def __post_init__(self) -> None:
        for name, value in (
            ("max_file_bytes", self.max_file_bytes),
            ("max_rows", self.max_rows),
            ("max_features", self.max_features),
            ("max_combinations", self.max_combinations),
            ("max_columns", self.max_columns),
            ("max_uncompressed_bytes", self.max_uncompressed_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise UploadLimitError(f"{name} must be a positive integer")

    def canonical(self) -> dict[str, int]:
        return {
            "max_file_bytes": self.max_file_bytes,
            "max_rows": self.max_rows,
            "max_features": self.max_features,
            "max_combinations": self.max_combinations,
            "max_columns": self.max_columns,
            "max_uncompressed_bytes": self.max_uncompressed_bytes,
        }


@dataclass(frozen=True, slots=True)
class LoadedFeatureUpload:
    """Governed upload plus redacted source-file geometry."""

    governed: GovernedFeatureDataset
    file_name: str
    file_bytes: int


def load_governed_feature_upload(
    data_path: Path,
    *,
    manifest_path: Path,
    mapping_path: Path,
    limits: FeatureUploadLimits = FeatureUploadLimits(),
) -> LoadedFeatureUpload:
    """Load and govern one explicit local CSV/Parquet feature bundle."""

    data_path = _existing_file(data_path, "data")
    manifest_path = _existing_file(manifest_path, "manifest")
    mapping_path = _existing_file(mapping_path, "mapping")
    file_bytes = data_path.stat().st_size
    if file_bytes > limits.max_file_bytes:
        raise UploadLimitError("data file size exceeds max_file_bytes")

    manifest_document = _json_document(manifest_path, "manifest")
    mapping_document = _json_document(mapping_path, "mapping")
    manifest = _feature_manifest(manifest_document, limits=limits)
    mapping, session_axis, pit_snapshot = _dataset_mapping(
        mapping_document,
        limits=limits,
    )
    frame = _read_frame(
        data_path,
        expected_columns=_mapped_columns(mapping_document),
        limits=limits,
    )
    if len(frame) > limits.max_rows:
        raise UploadLimitError("data row count exceeds max_rows")
    governed = governed_validation_dataset_from_pandas(
        frame,
        mapping=mapping,
        session_axis=session_axis,
        pit_snapshot=pit_snapshot,
        feature_manifest=manifest,
    )
    return LoadedFeatureUpload(
        governed=governed,
        file_name=data_path.name,
        file_bytes=file_bytes,
    )


def _existing_file(path: Path, name: str) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise AdapterValidationError(f"{name} file is missing")
    return candidate


def _json_document(path: Path, name: str) -> Mapping[str, Any]:
    if path.stat().st_size > _CONFIG_BYTE_LIMIT:
        raise UploadLimitError(f"{name} JSON exceeds configuration size limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterValidationError(f"{name} JSON is invalid") from exc
    if not isinstance(value, dict):
        raise AdapterValidationError(f"{name} JSON must be an object")
    return value


def _read_frame(
    path: Path,
    *,
    expected_columns: tuple[str, ...],
    limits: FeatureUploadLimits,
) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            header = pd.read_csv(path, nrows=0)
            if len(header.columns) > limits.max_columns:
                raise UploadLimitError("data column count exceeds max_columns")
            missing = set(expected_columns).difference(
                str(value) for value in header.columns
            )
            if missing:
                raise AdapterValidationError("data file is missing mapped columns")
            return pd.read_csv(
                path,
                usecols=list(expected_columns),
                nrows=limits.max_rows + 1,
            )
        if suffix == ".parquet":
            import pyarrow.parquet as parquet

            parquet_module = cast(Any, parquet)
            metadata = parquet_module.ParquetFile(path).metadata
            if metadata.num_rows > limits.max_rows:
                raise UploadLimitError("data row count exceeds max_rows")
            if metadata.num_columns > limits.max_columns:
                raise UploadLimitError("data column count exceeds max_columns")
            uncompressed_bytes = sum(
                metadata.row_group(index).total_byte_size
                for index in range(metadata.num_row_groups)
            )
            if uncompressed_bytes > limits.max_uncompressed_bytes:
                raise UploadLimitError(
                    "Parquet uncompressed size exceeds max_uncompressed_bytes"
                )
            schema_names = set(metadata.schema.names)
            if set(expected_columns).difference(schema_names):
                raise AdapterValidationError("data file is missing mapped columns")
            return pd.read_parquet(path, columns=list(expected_columns))
    except (UploadLimitError, AdapterValidationError):
        raise
    except Exception as exc:
        raise AdapterValidationError("data file could not be parsed") from exc
    raise AdapterValidationError("data file must use .csv or .parquet")


def _mapped_columns(document: Mapping[str, Any]) -> tuple[str, ...]:
    scalar_names = (
        "sample_id_column",
        "session_column",
        "asset_id_column",
        "interval_start_column",
        "interval_end_column",
        "decision_time_column",
        "target_column",
    )
    values: list[str] = []
    for name in scalar_names:
        value = document[name]
        if value is not None:
            values.append(_string(value, name))
    values.extend(_string_array(document["feature_columns"], "feature_columns"))
    values.extend(
        _string_array(document["availability_columns"], "availability_columns")
    )
    return tuple(dict.fromkeys(values))


def _feature_manifest(
    document: Mapping[str, Any], *, limits: FeatureUploadLimits
) -> FeatureManifest:
    _exact_keys(
        document, {"schema_version", "source_bundle_digest", "features"}, "manifest"
    )
    _schema_version(document)
    feature_documents = document["features"]
    if not isinstance(feature_documents, list) or not feature_documents:
        raise AdapterValidationError("manifest features must be a non-empty array")
    if len(feature_documents) > limits.max_features:
        raise UploadLimitError("feature count exceeds max_features")
    definitions: list[FeatureDefinition] = []
    keys = {
        "name",
        "source_dataset",
        "source_fields",
        "source_digest",
        "transformation",
        "transformation_version",
        "code_digest",
        "parameters",
        "lookback_sessions",
        "computation_scope",
        "revision_policy",
        "uses_target",
    }
    for value in feature_documents:
        if not isinstance(value, dict):
            raise AdapterValidationError("every manifest feature must be an object")
        _exact_keys(value, keys, "manifest feature")
        source_fields = _string_array(value["source_fields"], "source_fields")
        parameters = value["parameters"]
        if not isinstance(parameters, dict):
            raise AdapterValidationError("feature parameters must be an object")
        try:
            computation_scope = FeatureComputationScope(
                _string(value["computation_scope"], "feature computation_scope")
            )
        except ValueError as exc:
            raise AdapterValidationError(
                "feature computation_scope is invalid"
            ) from exc
        definitions.append(
            FeatureDefinition(
                name=_string(value["name"], "feature name"),
                source_dataset=_string(
                    value["source_dataset"], "feature source_dataset"
                ),
                source_fields=source_fields,
                source_digest=_string(value["source_digest"], "feature source_digest"),
                transformation=_string(
                    value["transformation"], "feature transformation"
                ),
                transformation_version=_string(
                    value["transformation_version"],
                    "feature transformation_version",
                ),
                code_digest=_string(value["code_digest"], "feature code_digest"),
                parameters=parameters,
                lookback_sessions=_integer(
                    value["lookback_sessions"], "feature lookback_sessions"
                ),
                computation_scope=computation_scope,
                revision_policy=_string(
                    value["revision_policy"], "feature revision_policy"
                ),
                uses_target=_boolean(value["uses_target"], "feature uses_target"),
            )
        )
    return FeatureManifest(
        source_bundle_digest=_string(
            document["source_bundle_digest"], "source_bundle_digest"
        ),
        definitions=tuple(definitions),
    )


def _dataset_mapping(
    document: Mapping[str, Any], *, limits: FeatureUploadLimits
) -> tuple[PandasDatasetMapping, tuple[str, ...], PITSnapshot]:
    keys = {
        "schema_version",
        "sample_id_column",
        "session_column",
        "asset_id_column",
        "interval_start_column",
        "interval_end_column",
        "decision_time_column",
        "target_column",
        "feature_columns",
        "availability_columns",
        "session_axis",
        "pit_snapshot",
    }
    _exact_keys(document, keys, "mapping")
    _schema_version(document)
    features = _string_array(document["feature_columns"], "feature_columns")
    availability = _string_array(
        document["availability_columns"], "availability_columns"
    )
    if len(features) > limits.max_features:
        raise UploadLimitError("feature count exceeds max_features")
    if len(features) != len(availability):
        raise AdapterValidationError(
            "availability_columns must align with feature_columns"
        )
    axis = _string_array(document["session_axis"], "session_axis")
    snapshot = document["pit_snapshot"]
    if not isinstance(snapshot, dict):
        raise AdapterValidationError("pit_snapshot must be an object")
    _exact_keys(
        snapshot,
        {"snapshot_id", "source_digest", "revision_policy"},
        "pit_snapshot",
    )
    asset_value = document["asset_id_column"]
    asset_field = (
        None
        if asset_value is None
        else PandasField(column=_string(asset_value, "asset_id_column"))
    )
    mapping = PandasDatasetMapping(
        sample_id=PandasField(
            column=_string(document["sample_id_column"], "sample_id_column")
        ),
        session=PandasField(
            column=_string(document["session_column"], "session_column")
        ),
        asset_id=asset_field,
        interval_start=PandasField(
            column=_string(document["interval_start_column"], "interval_start_column")
        ),
        interval_end=PandasField(
            column=_string(document["interval_end_column"], "interval_end_column")
        ),
        decision_time=PandasField(
            column=_string(document["decision_time_column"], "decision_time_column")
        ),
        target=PandasField(column=_string(document["target_column"], "target_column")),
        features=tuple(PandasField(column=value) for value in features),
        feature_availability=tuple(PandasField(column=value) for value in availability),
    )
    return (
        mapping,
        axis,
        PITSnapshot(
            snapshot_id=_string(snapshot["snapshot_id"], "snapshot_id"),
            source_digest=_string(snapshot["source_digest"], "source_digest"),
            revision_policy=_string(snapshot["revision_policy"], "revision_policy"),
        ),
    )


def _schema_version(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != "1":
        raise AdapterValidationError("schema_version must be '1'")


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise AdapterValidationError(f"{name} fields do not match schema version 1")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _string_array(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise AdapterValidationError(f"{name} must be a non-empty string array")
    values = tuple(_string(item, name) for item in value)
    if len(set(values)) != len(values):
        raise AdapterValidationError(f"{name} must not contain duplicates")
    return values


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterValidationError(f"{name} must be an integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise AdapterValidationError(f"{name} must be boolean")
    return value
