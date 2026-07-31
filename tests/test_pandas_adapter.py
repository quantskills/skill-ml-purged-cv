from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from purged_kfold_validation import (
    AdapterValidationError,
    InformationInterval,
    LeakageSafeEvaluator,
    ModelSpec,
    PITSnapshot,
    PurgedKFold,
    TemporalValidationError,
    ValidationDataset,
)
from purged_kfold_validation.adapters.pandas import (
    PandasDatasetMapping,
    PandasField,
    validation_dataset_from_pandas,
)


class MeanEstimator:
    def fit(self, features: np.ndarray, targets: np.ndarray) -> MeanEstimator:
        self.mean = float(targets.mean())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.mean)


def test_explicit_multiindex_mapping_uses_the_canonical_evaluation_path() -> None:
    axis = pd.DatetimeIndex(
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
        name="session",
    )
    index = pd.MultiIndex.from_product([axis, ["A", "B"]], names=["session", "asset"])
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{index}" for index in range(8)],
            "interval_start": index.get_level_values("session"),
            "interval_end": index.get_level_values("session"),
            "decision_time": index.get_level_values("session") + pd.Timedelta(hours=12),
            "available_at": index.get_level_values("session") + pd.Timedelta(hours=9),
            "x1": np.arange(8, dtype=float),
            "x2": np.arange(8, 16, dtype=float),
            "target": np.array([0.0, 1.0] * 4),
        },
        index=index,
    )
    snapshot = PITSnapshot(
        snapshot_id="adapter-vintage",
        source_digest="adapter-source-digest",
    )
    mapping = PandasDatasetMapping(
        sample_id=PandasField(column="sample_id"),
        session=PandasField(index_level="session"),
        asset_id=PandasField(index_level="asset"),
        interval_start=PandasField(column="interval_start"),
        interval_end=PandasField(column="interval_end"),
        decision_time=PandasField(column="decision_time"),
        feature_availability=PandasField(column="available_at"),
        features=(PandasField(column="x1"), PandasField(column="x2")),
        target=PandasField(column="target"),
    )

    adapted = validation_dataset_from_pandas(
        frame,
        mapping=mapping,
        session_axis=axis,
        pit_snapshot=snapshot,
    )
    sessions = index.get_level_values("session").to_numpy(dtype="datetime64[ns]")
    direct = ValidationDataset(
        sample_ids=tuple(frame["sample_id"]),
        asset_ids=tuple(index.get_level_values("asset")),
        session_axis=axis.to_numpy(dtype="datetime64[ns]"),
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        decision_times=frame["decision_time"].to_numpy(dtype="datetime64[ns]"),
        feature_availability=frame["available_at"].to_numpy(dtype="datetime64[ns]"),
        pit_snapshot=snapshot,
        features=frame[["x1", "x2"]].to_numpy(),
        targets=frame["target"].to_numpy(),
    )

    assert adapted.digest == direct.digest
    splitter = PurgedKFold(n_splits=2)
    assert splitter.plan(adapted) == splitter.plan(direct)

    evaluator = LeakageSafeEvaluator(
        splitter=splitter,
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="adapter-mean", version="1"),
    )
    adapted_result = evaluator.evaluate(adapted)
    direct_result = evaluator.evaluate(direct)
    assert adapted_result.run_id == direct_result.run_id
    assert adapted_result.ledger.digest == direct_result.ledger.digest


def _column_frame() -> tuple[pd.DataFrame, PandasDatasetMapping, PITSnapshot]:
    frame = pd.DataFrame(
        {
            "sample_id": ("s0", "s1", "s2", "s3"),
            "session": pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
            ),
            "start": pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
            ),
            "end": pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
            ),
            "decision": pd.to_datetime(
                [
                    "2025-01-02T12:00",
                    "2025-01-03T12:00",
                    "2025-01-06T12:00",
                    "2025-01-07T12:00",
                ]
            ),
            "available": pd.to_datetime(
                ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
            ),
            "feature": np.arange(4, dtype=float),
            "target": np.arange(4, dtype=float),
        }
    )
    mapping = PandasDatasetMapping(
        sample_id=PandasField(column="sample_id"),
        session=PandasField(column="session"),
        interval_start=PandasField(column="start"),
        interval_end=PandasField(column="end"),
        decision_time=PandasField(column="decision"),
        feature_availability=PandasField(column="available"),
        features=(PandasField(column="feature"),),
        target=PandasField(column="target"),
    )
    snapshot = PITSnapshot(snapshot_id="fixture", source_digest="fixture-digest")
    return frame, mapping, snapshot


def test_adapter_rejects_ambiguous_index_time_and_session_evidence() -> None:
    frame, mapping, snapshot = _column_frame()
    duplicate_index = frame.copy()
    duplicate_index.index = [0, 0, 1, 2]
    with pytest.raises(AdapterValidationError, match="duplicate entries"):
        validation_dataset_from_pandas(
            duplicate_index,
            mapping=mapping,
            session_axis=frame["session"],
            pit_snapshot=snapshot,
        )

    aware_axis = pd.DatetimeIndex(frame["session"]).tz_localize("UTC")
    with pytest.raises(AdapterValidationError, match="consistent timezone"):
        validation_dataset_from_pandas(
            frame,
            mapping=mapping,
            session_axis=aware_axis,
            pit_snapshot=snapshot,
        )

    with pytest.raises(TemporalValidationError, match="outside session_axis"):
        validation_dataset_from_pandas(
            frame,
            mapping=mapping,
            session_axis=frame["session"].iloc[:3],
            pit_snapshot=snapshot,
        )


def test_adapter_ignores_hidden_attrs_and_requires_complete_explicit_mapping() -> None:
    frame, _, _ = _column_frame()
    frame = frame.drop(columns="available")
    frame.attrs["feature_availability"] = frame["decision"]
    mapping = PandasDatasetMapping(
        sample_id=PandasField(column="sample_id"),
        session=PandasField(column="session"),
        interval_start=PandasField(column="start"),
        interval_end=PandasField(column="end"),
        decision_time=PandasField(column="decision"),
        feature_availability=PandasField(column="available"),
        features=(PandasField(column="feature"),),
        target=PandasField(column="target"),
    )

    with pytest.raises(AdapterValidationError, match="column is missing"):
        validation_dataset_from_pandas(
            frame,
            mapping=mapping,
            session_axis=frame["session"],
            pit_snapshot=PITSnapshot(
                snapshot_id="explicit",
                source_digest="explicit-digest",
            ),
        )

    with pytest.raises(AdapterValidationError, match="must match feature mappings"):
        PandasDatasetMapping(
            sample_id=PandasField(column="sample_id"),
            session=PandasField(column="session"),
            interval_start=PandasField(column="start"),
            interval_end=PandasField(column="end"),
            decision_time=PandasField(column="decision"),
            feature_availability=(
                PandasField(column="decision"),
                PandasField(column="decision"),
            ),
            features=(PandasField(column="feature"),),
            target=PandasField(column="target"),
        )


def test_core_import_does_not_require_pandas() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
class BlockPandas:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pandas' or fullname.startswith('pandas.'):
            raise ImportError('pandas intentionally unavailable')
        return None
sys.meta_path.insert(0, BlockPandas())
import purged_kfold_validation
assert 'pandas' not in sys.modules
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
