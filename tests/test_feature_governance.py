from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from purged_kfold_validation import (
    DatasetValidationError,
    FeatureComputationScope,
    FeatureDefinition,
    FeatureManifest,
    FactoryLifecycleError,
    InformationInterval,
    LeakageSafeEvaluator,
    ModelSpec,
    PITSnapshot,
    PointInTimeValidationError,
    PurgedKFold,
    TransformerSpec,
    ValidationDataset,
    govern_feature_dataset,
)


class MeanEstimator:
    def fit(self, features: np.ndarray, targets: np.ndarray) -> MeanEstimator:
        self.mean = float(targets.mean())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.mean)


def _dataset(*, availability: np.ndarray | None = None) -> ValidationDataset:
    sessions = np.array(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
        dtype="datetime64[ns]",
    )
    if availability is None:
        availability = np.array(
            [
                ["2025-01-02T09:00", "2025-01-02T10:00"],
                ["2025-01-03T09:00", "2025-01-03T10:00"],
                ["2025-01-06T09:00", "2025-01-06T10:00"],
                ["2025-01-07T09:00", "2025-01-07T10:00"],
            ],
            dtype="datetime64[ns]",
        )
    return ValidationDataset(
        sample_ids=("s0", "s1", "s2", "s3"),
        session_axis=sessions,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        decision_times=sessions + np.timedelta64(12, "h"),
        feature_availability=availability,
        pit_snapshot=PITSnapshot(
            snapshot_id="upload-vintage",
            source_digest="a" * 64,
        ),
        features=np.array([[100.0, 0.10], [101.0, 0.20], [102.0, 0.30], [103.0, 0.40]]),
        targets=np.array([0.0, 0.0, 1.0, 1.0]),
    )


def _manifest() -> FeatureManifest:
    return FeatureManifest(
        source_bundle_digest="a" * 64,
        definitions=(
            FeatureDefinition(
                name="close_raw",
                source_dataset="vendor.daily",
                source_fields=("close",),
                source_digest="b" * 64,
                transformation="identity",
                transformation_version="1",
                code_digest="c" * 64,
                parameters={},
                lookback_sessions=1,
                computation_scope=FeatureComputationScope.PRECOMPUTED_STATELESS,
            ),
            FeatureDefinition(
                name="momentum_2",
                source_dataset="vendor.daily",
                source_fields=("close",),
                source_digest="b" * 64,
                transformation="simple-return",
                transformation_version="1",
                code_digest="d" * 64,
                parameters={"periods": 2},
                lookback_sessions=2,
                computation_scope=FeatureComputationScope.PRECOMPUTED_STATELESS,
            ),
        ),
    )


def test_user_governs_arbitrary_features_and_evidence_reaches_oos() -> None:
    source = _dataset()
    manifest = _manifest()

    governed = govern_feature_dataset(source, manifest)

    assert governed.manifest == manifest
    assert governed.dataset.feature_manifest_digest == manifest.digest
    assert governed.dataset.digest != source.digest
    assert governed.receipt.observations == 4
    assert governed.receipt.features == 2
    assert governed.receipt.availability_cells == 8
    assert governed.receipt.maximum_lookback_sessions == 2
    assert governed.receipt.manifest_digest == manifest.digest
    assert manifest.canonical()["schema_version"] == "1"
    assert governed.receipt.canonical()["schema_version"] == "1"

    result = LeakageSafeEvaluator(
        splitter=PurgedKFold(n_splits=2),
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="mean", version="1"),
    ).evaluate(governed.dataset)

    assert {item.feature_manifest_digest for item in result.ledger.observations} == {
        manifest.digest
    }


def test_user_cannot_upload_a_globally_fitted_feature_matrix() -> None:
    honest = _manifest()
    stateful = replace(
        honest.definitions[1],
        computation_scope=FeatureComputationScope.PRECOMPUTED_STATEFUL,
    )
    manifest = FeatureManifest(
        source_bundle_digest=honest.source_bundle_digest,
        definitions=(honest.definitions[0], stateful),
    )

    with pytest.raises(
        DatasetValidationError,
        match="uploaded features must be precomputed-stateless",
    ):
        govern_feature_dataset(_dataset(), manifest)


def test_user_cannot_upload_target_derived_or_latest_revision_features() -> None:
    honest = _manifest()
    target_derived = replace(honest.definitions[1], uses_target=True)
    target_manifest = FeatureManifest(
        source_bundle_digest=honest.source_bundle_digest,
        definitions=(honest.definitions[0], target_derived),
    )

    with pytest.raises(DatasetValidationError, match="target-independent"):
        govern_feature_dataset(_dataset(), target_manifest)

    latest_revision = replace(honest.definitions[1], revision_policy="latest")
    latest_manifest = FeatureManifest(
        source_bundle_digest=honest.source_bundle_digest,
        definitions=(honest.definitions[0], latest_revision),
    )
    with pytest.raises(DatasetValidationError, match="point-in-time revisions"):
        govern_feature_dataset(_dataset(), latest_manifest)


class IdentityTransformer:
    def fit(self, features: np.ndarray, targets: np.ndarray) -> IdentityTransformer:
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features)


def test_fold_local_transformer_lineage_changes_evaluation_identity() -> None:
    governed = govern_feature_dataset(_dataset(), _manifest())
    first_spec = TransformerSpec(
        name="standardize",
        version="1",
        code_digest="e" * 64,
        parameters={"with_mean": True},
    )
    second_spec = replace(first_spec, version="2")

    first = LeakageSafeEvaluator(
        splitter=PurgedKFold(n_splits=2),
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="mean", version="1"),
        transformer_factories=(IdentityTransformer,),
        transformer_specs=(first_spec,),
    ).evaluate(governed.dataset)
    second = LeakageSafeEvaluator(
        splitter=PurgedKFold(n_splits=2),
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="mean", version="1"),
        transformer_factories=(IdentityTransformer,),
        transformer_specs=(second_spec,),
    ).evaluate(governed.dataset)

    assert first.run_id != second.run_id
    assert {item.transformer_spec_digests for item in first.ledger.observations} == {
        (first_spec.digest,)
    }


def test_feature_parameters_are_deeply_immutable_after_digesting() -> None:
    definition = replace(
        _manifest().definitions[1],
        parameters={"window": {"periods": 2}},
    )
    nested = cast(dict[str, int], definition.parameters["window"])

    with pytest.raises(TypeError):
        nested["periods"] = 99


def test_feature_governance_rejects_ambiguous_or_late_availability() -> None:
    sessions = np.array(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
        dtype="datetime64[ns]",
    )
    with pytest.raises(DatasetValidationError, match="per-feature availability"):
        govern_feature_dataset(_dataset(availability=sessions), _manifest())

    late = np.array(
        [
            ["2025-01-02T09:00", "2025-01-02T10:00"],
            ["2025-01-03T09:00", "2025-01-03T10:00"],
            ["2025-01-06T09:00", "2025-01-06T13:00"],
            ["2025-01-07T09:00", "2025-01-07T10:00"],
        ],
        dtype="datetime64[ns]",
    )
    with pytest.raises(PointInTimeValidationError, match="sample positions 2"):
        govern_feature_dataset(_dataset(availability=late), _manifest())


def test_feature_manifest_rejects_duplicate_names_and_source_mismatch() -> None:
    honest = _manifest()
    with pytest.raises(DatasetValidationError, match="names must be unique"):
        FeatureManifest(
            source_bundle_digest=honest.source_bundle_digest,
            definitions=(honest.definitions[0], honest.definitions[0]),
        )

    mismatch = replace(honest, source_bundle_digest="f" * 64)
    with pytest.raises(DatasetValidationError, match="does not match PIT Snapshot"):
        govern_feature_dataset(_dataset(), mismatch)


def test_transformer_factory_and_spec_must_be_bound_one_to_one() -> None:
    with pytest.raises(FactoryLifecycleError, match="one ordered TransformerSpec"):
        LeakageSafeEvaluator(
            splitter=PurgedKFold(n_splits=2),
            estimator_factory=MeanEstimator,
            model_spec=ModelSpec(name="mean", version="1"),
            transformer_factories=(IdentityTransformer,),
        )
