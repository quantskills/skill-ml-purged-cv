"""Immutable availability and lineage governance for uploaded features."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Mapping

import numpy as np

from .domain import (
    SCHEMA_VERSION,
    ValidationDataset,
    _deeply_frozen_json,
    canonical_digest,
)
from .errors import DatasetValidationError


class FeatureComputationScope(str, Enum):
    """Declared lifecycle of a feature value presented for evaluation."""

    PRECOMPUTED_STATELESS = "precomputed-stateless"
    PRECOMPUTED_STATEFUL = "precomputed-stateful"
    FOLD_LOCAL = "fold-local"


def _digest(value: str, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DatasetValidationError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _parameters(value: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        normalized = json.loads(
            json.dumps(dict(value), ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(
            "feature parameters must be canonical JSON values"
        ) from exc
    frozen = _deeply_frozen_json(normalized)
    assert isinstance(frozen, Mapping)
    return frozen


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """One ordered uploaded feature's availability and lineage contract."""

    name: str
    source_dataset: str
    source_fields: tuple[str, ...]
    source_digest: str
    transformation: str
    transformation_version: str
    code_digest: str
    parameters: Mapping[str, Any]
    lookback_sessions: int
    computation_scope: FeatureComputationScope
    revision_policy: str = "point-in-time"
    uses_target: bool = False
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, field_name="feature name"))
        object.__setattr__(
            self,
            "source_dataset",
            _text(self.source_dataset, field_name="feature source dataset"),
        )
        source_fields = tuple(
            _text(value, field_name="feature source field")
            for value in self.source_fields
        )
        if not source_fields:
            raise DatasetValidationError("feature source_fields must not be empty")
        object.__setattr__(self, "source_fields", source_fields)
        object.__setattr__(
            self,
            "source_digest",
            _digest(self.source_digest, field_name="feature source_digest"),
        )
        object.__setattr__(
            self,
            "transformation",
            _text(self.transformation, field_name="feature transformation"),
        )
        object.__setattr__(
            self,
            "transformation_version",
            _text(
                self.transformation_version,
                field_name="feature transformation_version",
            ),
        )
        object.__setattr__(
            self,
            "code_digest",
            _digest(self.code_digest, field_name="feature code_digest"),
        )
        object.__setattr__(self, "parameters", _parameters(self.parameters))
        if (
            isinstance(self.lookback_sessions, bool)
            or not isinstance(self.lookback_sessions, int)
            or self.lookback_sessions < 1
        ):
            raise DatasetValidationError(
                "feature lookback_sessions must be an integer of at least 1"
            )
        try:
            scope = FeatureComputationScope(self.computation_scope)
        except (TypeError, ValueError) as exc:
            raise DatasetValidationError(
                "feature computation_scope is invalid"
            ) from exc
        object.__setattr__(self, "computation_scope", scope)
        if not isinstance(self.revision_policy, str):
            raise DatasetValidationError("feature revision_policy must be a string")
        if not isinstance(self.uses_target, bool):
            raise DatasetValidationError("feature uses_target must be boolean")
        object.__setattr__(self, "digest", canonical_digest(self.canonical()))

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "source_dataset": self.source_dataset,
            "source_fields": list(self.source_fields),
            "source_digest": self.source_digest,
            "transformation": self.transformation,
            "transformation_version": self.transformation_version,
            "code_digest": self.code_digest,
            "parameters": _plain_json(self.parameters),
            "lookback_sessions": self.lookback_sessions,
            "computation_scope": self.computation_scope.value,
            "revision_policy": self.revision_policy,
            "uses_target": self.uses_target,
        }


@dataclass(frozen=True, slots=True)
class FeatureManifest:
    """Ordered feature definitions bound to one PIT source bundle."""

    source_bundle_digest: str
    definitions: tuple[FeatureDefinition, ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_bundle_digest",
            _digest(self.source_bundle_digest, field_name="source_bundle_digest"),
        )
        definitions = tuple(self.definitions)
        if not definitions or any(
            not isinstance(value, FeatureDefinition) for value in definitions
        ):
            raise DatasetValidationError(
                "feature manifest must contain FeatureDefinition values"
            )
        names = tuple(value.name for value in definitions)
        if len(set(names)) != len(names):
            raise DatasetValidationError("feature manifest names must be unique")
        object.__setattr__(self, "definitions", definitions)
        object.__setattr__(self, "digest", canonical_digest(self.canonical()))

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_bundle_digest": self.source_bundle_digest,
            "definitions": [value.canonical() for value in self.definitions],
        }


@dataclass(frozen=True, slots=True)
class FeatureGovernanceReceipt:
    """Redacted deterministic evidence for one governed feature matrix."""

    dataset_digest: str
    manifest_digest: str
    source_bundle_digest: str
    observations: int
    features: int
    availability_cells: int
    maximum_lookback_sessions: int
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", canonical_digest(self.canonical()))

    def canonical(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_digest": self.dataset_digest,
            "manifest_digest": self.manifest_digest,
            "source_bundle_digest": self.source_bundle_digest,
            "observations": self.observations,
            "features": self.features,
            "availability_cells": self.availability_cells,
            "maximum_lookback_sessions": self.maximum_lookback_sessions,
        }


@dataclass(frozen=True, slots=True)
class GovernedFeatureDataset:
    dataset: ValidationDataset
    manifest: FeatureManifest
    receipt: FeatureGovernanceReceipt


def govern_feature_dataset(
    dataset: ValidationDataset, manifest: FeatureManifest
) -> GovernedFeatureDataset:
    """Bind explicit uploaded-feature lineage to canonical dataset evidence."""

    if not isinstance(dataset, ValidationDataset):
        raise DatasetValidationError("dataset must be a ValidationDataset")
    if not isinstance(manifest, FeatureManifest):
        raise DatasetValidationError("manifest must be a FeatureManifest")
    dataset.require_formal_scoring()
    snapshot = dataset.pit_snapshot
    assert snapshot is not None
    if snapshot.source_digest != manifest.source_bundle_digest:
        raise DatasetValidationError(
            "feature manifest source bundle does not match PIT Snapshot"
        )
    if len(manifest.definitions) != dataset.features.shape[1]:
        raise DatasetValidationError(
            "feature manifest definitions must match feature columns"
        )
    if any(
        value.computation_scope is not FeatureComputationScope.PRECOMPUTED_STATELESS
        for value in manifest.definitions
    ):
        raise DatasetValidationError(
            "uploaded features must be precomputed-stateless; learned state belongs "
            "in fold-local transformer factories"
        )
    if any(value.uses_target for value in manifest.definitions):
        raise DatasetValidationError(
            "uploaded features must be target-independent; target-derived state "
            "belongs in fold-local transformer factories"
        )
    if any(value.revision_policy != "point-in-time" for value in manifest.definitions):
        raise DatasetValidationError(
            "uploaded features require point-in-time revisions"
        )
    availability = dataset.feature_availability
    assert availability is not None
    if availability.ndim != 2 or availability.shape != dataset.features.shape:
        raise DatasetValidationError(
            "governed features require per-feature availability timestamps"
        )
    governed_dataset = ValidationDataset(
        sample_ids=dataset.sample_ids,
        asset_ids=dataset.asset_ids,
        session_axis=dataset.session_axis,
        sessions=dataset.sessions,
        information_intervals=dataset.information_intervals,
        decision_times=dataset.decision_times,
        feature_availability=dataset.feature_availability,
        pit_snapshot=dataset.pit_snapshot,
        feature_manifest_digest=manifest.digest,
        missing_value_policy=dataset.missing_value_policy,
        features=dataset.features,
        targets=dataset.targets,
    )
    receipt = FeatureGovernanceReceipt(
        dataset_digest=governed_dataset.digest,
        manifest_digest=manifest.digest,
        source_bundle_digest=manifest.source_bundle_digest,
        observations=len(governed_dataset.sample_ids),
        features=governed_dataset.features.shape[1],
        availability_cells=int(np.prod(availability.shape)),
        maximum_lookback_sessions=max(
            value.lookback_sessions for value in manifest.definitions
        ),
    )
    return GovernedFeatureDataset(governed_dataset, manifest, receipt)
