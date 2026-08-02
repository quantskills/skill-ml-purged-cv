from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from purged_kfold_validation import (
    EffectivenessComparisonConfig,
    InformationInterval,
    ModelSpec,
    PITSnapshot,
    TransformerSpec,
    ValidationDataset,
    run_cpcv_effectiveness_comparison,
)


class FeatureEstimator:
    def fit(self, features: np.ndarray, targets: np.ndarray) -> FeatureEstimator:
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features[:, 0])


class FoldStandardizer:
    def fit(self, features: np.ndarray, targets: np.ndarray) -> FoldStandardizer:
        self.mean = features.mean(axis=0)
        scale = features.std(axis=0)
        self.scale = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.asarray((features - self.mean) / self.scale)


def _panel_dataset() -> ValidationDataset:
    session_axis = np.arange(
        np.datetime64("2025-01-02"),
        np.datetime64("2025-01-14"),
        dtype="datetime64[D]",
    )
    sessions = np.repeat(session_axis, 3)
    assets = tuple(asset for _ in session_axis for asset in ("A", "B", "C"))
    targets = np.asarray(
        [
            value
            for session_index in range(len(session_axis))
            for value in (
                -0.01 * (session_index + 1),
                0.0,
                0.01 * (session_index + 1),
            )
        ]
    )
    return ValidationDataset(
        sample_ids=tuple(
            (asset, str(session))
            for session in session_axis
            for asset in ("A", "B", "C")
        ),
        session_axis=session_axis,
        sessions=sessions,
        information_intervals=tuple(
            InformationInterval(session, session) for session in sessions
        ),
        decision_times=sessions + np.timedelta64(12, "h"),
        feature_availability=sessions,
        pit_snapshot=PITSnapshot(
            snapshot_id="effectiveness-fixture",
            source_digest="effectiveness-source",
        ),
        asset_ids=assets,
        features=targets.reshape(-1, 1),
        targets=targets,
    )


def test_user_gets_path_metrics_distributions_and_safe_channel_comparison() -> None:
    dataset = _panel_dataset()
    report = run_cpcv_effectiveness_comparison(
        dataset,
        label_roll_clean=np.ones(len(dataset.sample_ids), dtype=bool),
        estimator_factory=FeatureEstimator,
        model_spec=ModelSpec(name="feature-estimator", version="1"),
        config=EffectivenessComparisonConfig(
            n_groups=3,
            n_test_groups=2,
            walk_forward_splits=2,
            embargo_sessions=0,
            pre_test_gap_sessions=0,
            min_train_observations=1,
            min_train_sessions=1,
            min_train_assets=1,
        ),
    )

    assert tuple(channel.name for channel in report.channels) == (
        "purged-kfold",
        "cpcv",
        "causal-walk-forward",
    )
    assert all(channel.overlap_count == 0 for channel in report.channels)
    cpcv = report.channels[1]
    assert len(cpcv.native.paths) == 2
    assert all(path.metrics.mse == 0.0 for path in cpcv.native.paths)
    assert all(
        path.metrics.cross_sectional_spearman_ic == 1.0 for path in cpcv.native.paths
    )
    assert all(
        np.isfinite(path.metrics.diagnostic_sharpe) for path in cpcv.native.paths
    )
    assert cpcv.native.distributions.mse.median == 0.0
    assert cpcv.native.distributions.ic.worst == 1.0
    assert cpcv.common.unique_observations == 24
    assert cpcv.training_sufficiency.all_sufficient
    assert len(report.group_breadth) == 3


def test_high_level_report_binds_ordered_fold_local_transformer_specs() -> None:
    dataset = _panel_dataset()
    config = EffectivenessComparisonConfig(
        n_groups=3,
        n_test_groups=2,
        walk_forward_splits=2,
        embargo_sessions=0,
        pre_test_gap_sessions=0,
        min_train_observations=1,
        min_train_sessions=1,
        min_train_assets=1,
    )

    reports = tuple(
        run_cpcv_effectiveness_comparison(
            dataset,
            label_roll_clean=np.ones(len(dataset.sample_ids), dtype=bool),
            estimator_factory=FeatureEstimator,
            model_spec=ModelSpec(name="feature-estimator", version="1"),
            transformer_factories=(FoldStandardizer,),
            transformer_specs=(
                TransformerSpec(
                    name="standardizer",
                    version=version,
                    code_digest="d" * 64,
                ),
            ),
            config=config,
        )
        for version in ("1", "2")
    )

    assert reports[0].digest != reports[1].digest
    assert reports[0].canonical()["transformer_spec_digests"] == list(
        reports[0].transformer_spec_digests
    )
    assert reports[0].transformer_spec_digests != reports[1].transformer_spec_digests


def test_operator_runs_governed_full_cache_comparison_offline(tmp_path: Path) -> None:
    sessions = pd.date_range("2025-01-02", periods=18, freq="D")
    for asset_index, asset in enumerate(("A", "B", "C", "D")):
        drift = (-0.02, -0.005, 0.005, 0.02)[asset_index]
        close = [80.0 + asset_index * 15.0]
        for session_index in range(1, len(sessions)):
            close.append(close[-1] * (1.0 + drift * (1.0 + 0.03 * session_index)))
        frame = pd.DataFrame(
            {
                "date": [*sessions, sessions[-1]],
                "symbol": [f"{asset}_DOMINANT.X"] * len(sessions) + [f"{asset}2601.X"],
                "underlying_symbol": [asset] * (len(sessions) + 1),
                "dominant_id": [f"{asset}2501"] * 9
                + [f"{asset}2502"] * 9
                + [f"{asset}2601"],
                "close": [*close, close[-1]],
                "volume": [
                    100.0 * (asset_index + 1) + value for value in range(len(sessions))
                ]
                + [999.0],
                "open_interest": [
                    200.0 * (asset_index + 1) + 2 * value
                    for value in range(len(sessions))
                ]
                + [999.0],
            }
        )
        frame.to_parquet(tmp_path / f"{asset}_daily.parquet", index=False)
    pd.DataFrame(
        {
            "date": [sessions[-1]],
            "symbol": ["E2601.X"],
            "dominant_id": ["E2601"],
            "close": [100.0],
            "volume": [100.0],
            "open_interest": [200.0],
        }
    ).to_parquet(tmp_path / "E_daily.parquet", index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_full_pandaai_cpcv.py",
            "--data-dir",
            str(tmp_path),
            "--feature-columns",
            "close,volume,open_interest",
            "--label-horizon-sessions",
            "1",
            "--feature-lookback-sessions",
            "2",
            "--n-groups",
            "3",
            "--n-test-groups",
            "2",
            "--walk-forward-splits",
            "2",
            "--embargo-sessions",
            "0",
            "--pre-test-gap-sessions",
            "0",
            "--min-train-observations",
            "1",
            "--min-train-sessions",
            "1",
            "--min-train-assets",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "success"
    assert payload["input"]["files"] == 5
    assert payload["governance"]["selected_assets"] == ["A", "B", "C", "D"]
    assert payload["governance"]["excluded_assets"] == ["E"]
    assert payload["dataset"]["observations"] == 64
    assert [channel["name"] for channel in payload["report"]["channels"]] == [
        "purged-kfold",
        "cpcv",
        "causal-walk-forward",
    ]
    assert len(payload["report"]["channels"][1]["views"]["native"]["paths"]) == 2
