from __future__ import annotations

import pandas as pd
import pytest

from purged_kfold_validation import (
    DatasetValidationError,
    FeatureComputationScope,
    FeatureDefinition,
    FeatureManifest,
    PITSnapshot,
)
from purged_kfold_validation.adapters.pandas import (
    PandasDatasetMapping,
    PandasField,
    governed_validation_dataset_from_pandas,
)


def test_user_uploads_explicit_arbitrary_feature_columns_with_lineage() -> None:
    sessions = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    frame = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "session": sessions,
            "start": sessions,
            "end": sessions,
            "decision": sessions + pd.Timedelta(hours=12),
            "close_available": sessions + pd.Timedelta(hours=9),
            "momentum_available": sessions + pd.Timedelta(hours=10),
            "raw_close": [100.0, 101.0, 102.0, 103.0],
            "momentum_2": [0.0, 0.01, 0.0099, 0.0098],
            "target": [0.01, -0.01, 0.02, -0.02],
        }
    )
    mapping = PandasDatasetMapping(
        sample_id=PandasField(column="sample_id"),
        session=PandasField(column="session"),
        interval_start=PandasField(column="start"),
        interval_end=PandasField(column="end"),
        decision_time=PandasField(column="decision"),
        feature_availability=(
            PandasField(column="close_available"),
            PandasField(column="momentum_available"),
        ),
        features=(
            PandasField(column="raw_close"),
            PandasField(column="momentum_2"),
        ),
        target=PandasField(column="target"),
    )
    manifest = FeatureManifest(
        source_bundle_digest="a" * 64,
        definitions=(
            FeatureDefinition(
                name="raw_close",
                source_dataset="upload.parquet",
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
                source_dataset="upload.parquet",
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

    governed = governed_validation_dataset_from_pandas(
        frame,
        mapping=mapping,
        session_axis=sessions,
        pit_snapshot=PITSnapshot(
            snapshot_id="upload-vintage",
            source_digest="a" * 64,
        ),
        feature_manifest=manifest,
    )

    assert governed.dataset.features.shape == (4, 2)
    assert governed.manifest.digest == manifest.digest
    assert governed.receipt.availability_cells == 8

    reversed_manifest = FeatureManifest(
        source_bundle_digest=manifest.source_bundle_digest,
        definitions=tuple(reversed(manifest.definitions)),
    )
    with pytest.raises(DatasetValidationError, match="feature mapping order"):
        governed_validation_dataset_from_pandas(
            frame,
            mapping=mapping,
            session_axis=sessions,
            pit_snapshot=PITSnapshot(
                snapshot_id="upload-vintage",
                source_digest="a" * 64,
            ),
            feature_manifest=reversed_manifest,
        )
