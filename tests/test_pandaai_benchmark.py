from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import json
import subprocess
import sys

import pytest

from purged_kfold_validation import AdapterValidationError, MetricSpec, ModelSpec
from purged_kfold_validation.adapters.pandaai import (
    PandaAIDailyConfig,
    PandaAIDailyMapping,
    validation_dataset_from_pandaai_daily,
)
from purged_kfold_validation.benchmark import run_validation_benchmark


def _daily_frame() -> pd.DataFrame:
    sessions = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
            "2025-01-07",
            "2025-01-08",
            "2025-01-09",
            "2025-01-10",
            "2025-01-13",
            "2025-01-14",
            "2025-01-15",
        ]
    )
    rows: list[dict[str, object]] = []
    for asset, offset in (("A", 0.0), ("B", 10.0)):
        for index, session in enumerate(sessions):
            rows.append(
                {
                    "date": session,
                    "symbol": asset,
                    "close": 100.0 + offset + index,
                    "volume": 1000.0 + 10 * index,
                }
            )
    return pd.DataFrame(rows)


def test_user_builds_explicit_pit_dataset_from_pandaai_daily_rows() -> None:
    dataset = validation_dataset_from_pandaai_daily(
        _daily_frame(),
        mapping=PandaAIDailyMapping(
            session="date",
            asset="symbol",
            close="close",
            features=("close", "volume"),
        ),
        config=PandaAIDailyConfig(
            label_horizon_sessions=1,
            feature_lookback_sessions=2,
            decision_time_offset_minutes=15 * 60,
            snapshot_id="pandaai-cache-2025-01-09",
            source_digest="sha256-local-cache",
        ),
    )

    assert len(dataset.sample_ids) == 16
    assert dataset.sample_ids[:2] == (
        ("A", "2025-01-03"),
        ("B", "2025-01-03"),
    )
    assert dataset.asset_ids is not None
    assert dataset.asset_ids[:2] == ("A", "B")
    assert dataset.features.shape == (16, 2)
    assert dataset.information_intervals[0].start == np.datetime64("2025-01-02", "ns")
    assert dataset.information_intervals[0].end == np.datetime64("2025-01-06", "ns")
    assert dataset.targets[0] == (102.0 / 101.0) - 1.0
    assert dataset.formal_scoring_issues == ()


class MeanEstimator:
    def fit(self, features: np.ndarray, targets: np.ndarray) -> MeanEstimator:
        self.mean = float(targets.mean())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.mean)


def test_benchmark_compares_four_channels_and_audits_actual_overlap() -> None:
    dataset = validation_dataset_from_pandaai_daily(
        _daily_frame(),
        mapping=PandaAIDailyMapping(
            session="date",
            asset="symbol",
            close="close",
            features=("close", "volume"),
        ),
        config=PandaAIDailyConfig(
            label_horizon_sessions=1,
            feature_lookback_sessions=2,
            decision_time_offset_minutes=15 * 60,
            snapshot_id="pandaai-cache-2025-01-09",
            source_digest="sha256-local-cache",
        ),
    )
    metric = MetricSpec(
        name="mse",
        version="1",
        function=lambda actual, predicted: float(np.mean((actual - predicted) ** 2)),
    )

    report = run_validation_benchmark(
        dataset,
        estimator_factory=MeanEstimator,
        model_spec=ModelSpec(name="mean", version="1"),
        metric=metric,
        n_splits=2,
        embargo_sessions=1,
        pre_test_gap_sessions=0,
        random_seed=7,
    )

    by_name = {channel.name: channel for channel in report.channels}
    assert tuple(by_name) == (
        "unsafe-shuffled-kfold",
        "chronological-no-purge",
        "purged-kfold",
        "causal-walk-forward",
    )
    assert by_name["chronological-no-purge"].overlap_count > 0
    assert by_name["purged-kfold"].overlap_count == 0
    assert by_name["causal-walk-forward"].overlap_count == 0
    assert by_name["purged-kfold"].observation_coverage == 1.0
    assert by_name["causal-walk-forward"].observation_coverage < 1.0
    assert (
        report.digest
        == run_validation_benchmark(
            dataset,
            estimator_factory=MeanEstimator,
            model_spec=ModelSpec(name="mean", version="1"),
            metric=metric,
            n_splits=2,
            embargo_sessions=1,
            pre_test_gap_sessions=0,
            random_seed=7,
        ).digest
    )


def test_operator_runs_offline_pandaai_parquet_benchmark(tmp_path: Path) -> None:
    frame = _daily_frame()
    for symbol, group in frame.groupby("symbol", sort=True):
        group.to_parquet(tmp_path / f"{symbol}_daily.parquet", index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_pandaai.py",
            "--data-dir",
            str(tmp_path),
            "--feature-columns",
            "close,volume",
            "--label-horizon-sessions",
            "1",
            "--feature-lookback-sessions",
            "2",
            "--n-splits",
            "2",
            "--embargo-sessions",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "success"
    assert payload["input"]["files"] == 2
    assert payload["input"]["assets"] == 2
    assert [channel["name"] for channel in payload["report"]["channels"]] == [
        "unsafe-shuffled-kfold",
        "chronological-no-purge",
        "purged-kfold",
        "causal-walk-forward",
    ]
    assert "rows" not in payload


def test_pandaai_adapter_rejects_duplicate_asset_sessions() -> None:
    frame = _daily_frame()
    frame = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)

    with pytest.raises(AdapterValidationError, match="duplicate asset sessions"):
        validation_dataset_from_pandaai_daily(
            frame,
            mapping=PandaAIDailyMapping(
                session="date",
                asset="symbol",
                close="close",
                features=("close", "volume"),
            ),
            config=PandaAIDailyConfig(
                label_horizon_sessions=1,
                feature_lookback_sessions=2,
                decision_time_offset_minutes=15 * 60,
                snapshot_id="duplicate-fixture",
                source_digest="duplicate-fixture-digest",
            ),
        )
