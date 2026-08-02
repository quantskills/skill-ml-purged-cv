"""Run an offline four-channel validation benchmark on PandaAI daily parquet caches."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_kfold_validation import (  # noqa: E402
    MetricSpec,
    ModelSpec,
    run_validation_benchmark,
)
from purged_kfold_validation.adapters.pandaai import (  # noqa: E402
    PandaAIDailyConfig,
    PandaAIDailyMapping,
    validation_dataset_from_pandaai_daily,
)


class RidgeEstimator:
    """Small deterministic fold-local regression baseline."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> RidgeEstimator:
        self.mean = features.mean(axis=0)
        scale = features.std(axis=0)
        self.scale = np.where(scale == 0.0, 1.0, scale)
        normalized = (features - self.mean) / self.scale
        design = np.column_stack((np.ones(len(normalized)), normalized))
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef = np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.mean is not None
        assert self.scale is not None
        assert self.coef is not None
        normalized = (features - self.mean) / self.scale
        design = np.column_stack((np.ones(len(normalized)), normalized))
        return np.asarray(design @ self.coef)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline PandaAI daily validation benchmark"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--session-column", default="date")
    parser.add_argument("--asset-column", default="symbol")
    parser.add_argument("--close-column", default="close")
    parser.add_argument("--feature-columns", required=True)
    parser.add_argument("--label-horizon-sessions", type=int, default=5)
    parser.add_argument("--feature-lookback-sessions", type=int, default=20)
    parser.add_argument("--decision-time-offset-minutes", type=int, default=15 * 60)
    parser.add_argument("--snapshot-id", default="")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--embargo-sessions", type=int, default=0)
    parser.add_argument("--pre-test-gap-sessions", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser.parse_args()


def selected_files(data_dir: Path, symbols: tuple[str, ...]) -> tuple[Path, ...]:
    if symbols:
        files = tuple(data_dir / f"{symbol}_daily.parquet" for symbol in symbols)
    else:
        files = tuple(sorted(data_dir.glob("*_daily.parquet")))
    missing = [path.name for path in files if not path.is_file()]
    if not files or missing:
        detail = "no matching cache files" if not files else f"missing files: {missing}"
        raise FileNotFoundError(detail)
    return files


def files_digest(files: tuple[Path, ...]) -> str:
    digest = sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    symbols = tuple(
        value.strip().upper() for value in args.symbols.split(",") if value.strip()
    )
    features = tuple(
        value.strip() for value in args.feature_columns.split(",") if value.strip()
    )
    files = selected_files(data_dir, symbols)
    source_digest = files_digest(files)
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    dataset = validation_dataset_from_pandaai_daily(
        frame,
        mapping=PandaAIDailyMapping(
            session=args.session_column,
            asset=args.asset_column,
            close=args.close_column,
            features=features,
        ),
        config=PandaAIDailyConfig(
            label_horizon_sessions=args.label_horizon_sessions,
            feature_lookback_sessions=args.feature_lookback_sessions,
            decision_time_offset_minutes=args.decision_time_offset_minutes,
            snapshot_id=(
                args.snapshot_id or f"pandaai-local-cache-{source_digest[:12]}"
            ),
            source_digest=source_digest,
        ),
    )
    metric = MetricSpec(
        name="mean-squared-error",
        version="1",
        function=lambda actual, predicted: float(np.mean((actual - predicted) ** 2)),
    )
    model = ModelSpec(
        name="fold-local-ridge",
        version="1",
        parameters={"alpha": args.ridge_alpha},
    )
    report = run_validation_benchmark(
        dataset,
        estimator_factory=lambda: RidgeEstimator(args.ridge_alpha),
        model_spec=model,
        metric=metric,
        n_splits=args.n_splits,
        embargo_sessions=args.embargo_sessions,
        pre_test_gap_sessions=args.pre_test_gap_sessions,
        random_seed=args.random_seed,
    )
    payload = {
        "status": "success",
        "input": {
            "files": len(files),
            "assets": len(set(dataset.asset_ids or ())),
            "sessions": len(set(dataset.sessions)),
            "observations": len(dataset.sample_ids),
            "source_digest": source_digest,
        },
        "report": report.canonical(),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
