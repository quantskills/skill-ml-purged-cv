"""Run the governed full-cache CPCV effectiveness comparison offline."""

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
    EffectivenessComparisonConfig,
    ModelSpec,
    run_cpcv_effectiveness_comparison,
)
from purged_kfold_validation.adapters.pandaai import (  # noqa: E402
    PandaAIContinuousMapping,
    PandaAIContinuousPolicy,
    PandaAIDailyConfig,
    PandaAIDailyMapping,
    governed_validation_dataset_from_pandaai_continuous_daily,
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
        description="Governed full-cache PandaAI CPCV effectiveness comparison"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--session-column", default="date")
    parser.add_argument("--series-symbol-column", default="symbol")
    parser.add_argument("--underlying-column", default="underlying_symbol")
    parser.add_argument("--active-contract-column", default="dominant_id")
    parser.add_argument("--close-column", default="close")
    parser.add_argument("--feature-columns", required=True)
    parser.add_argument("--label-horizon-sessions", type=int, default=5)
    parser.add_argument("--feature-lookback-sessions", type=int, default=20)
    parser.add_argument("--decision-time-offset-minutes", type=int, default=15 * 60)
    parser.add_argument("--snapshot-id", default="")
    parser.add_argument("--n-groups", type=int, default=6)
    parser.add_argument("--n-test-groups", type=int, default=2)
    parser.add_argument("--walk-forward-splits", type=int, default=5)
    parser.add_argument("--embargo-sessions", type=int, default=20)
    parser.add_argument("--pre-test-gap-sessions", type=int, default=20)
    parser.add_argument("--min-train-observations", type=int, default=10_000)
    parser.add_argument("--min-train-sessions", type=int, default=252)
    parser.add_argument("--min-train-assets", type=int, default=20)
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


def _file_asset(path: Path) -> str:
    suffix = "_daily"
    if not path.stem.endswith(suffix):
        raise ValueError(f"unexpected cache filename: {path.name}")
    asset = path.stem[: -len(suffix)].strip().upper()
    if not asset:
        raise ValueError(f"cache filename has no asset identity: {path.name}")
    return asset


def load_cache(
    files: tuple[Path, ...],
    *,
    required_columns: tuple[str, ...],
    optional_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in files:
        source = pd.read_parquet(path)
        missing = sorted(
            set(required_columns)
            .difference(source.columns)
            .difference(optional_columns)
        )
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        for column in optional_columns:
            if column not in source.columns:
                source[column] = pd.NA
        frame = source.loc[:, list(required_columns)].copy()
        frame["__file_asset"] = _file_asset(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    symbols = tuple(
        value.strip().upper() for value in args.symbols.split(",") if value.strip()
    )
    features = tuple(
        value.strip() for value in args.feature_columns.split(",") if value.strip()
    )
    if not features:
        raise ValueError("feature-columns must contain at least one column")
    files = selected_files(data_dir, symbols)
    source_digest = files_digest(files)
    required_columns = tuple(
        dict.fromkeys(
            (
                args.session_column,
                args.series_symbol_column,
                args.underlying_column,
                args.active_contract_column,
                args.close_column,
                *features,
            )
        )
    )
    frame = load_cache(
        files,
        required_columns=required_columns,
        optional_columns=(args.underlying_column,),
    )
    governed = governed_validation_dataset_from_pandaai_continuous_daily(
        frame,
        mapping=PandaAIDailyMapping(
            session=args.session_column,
            asset="__file_asset",
            close=args.close_column,
            features=features,
        ),
        config=PandaAIDailyConfig(
            label_horizon_sessions=args.label_horizon_sessions,
            feature_lookback_sessions=args.feature_lookback_sessions,
            decision_time_offset_minutes=args.decision_time_offset_minutes,
            snapshot_id=(
                args.snapshot_id or f"pandaai-continuous-{source_digest[:12]}"
            ),
            source_digest=source_digest,
        ),
        continuous_mapping=PandaAIContinuousMapping(
            series_symbol=args.series_symbol_column,
            asset="__file_asset",
            active_contract=args.active_contract_column,
            declared_underlying=args.underlying_column,
        ),
        policy=PandaAIContinuousPolicy(),
    )
    model = ModelSpec(
        name="fold-local-ridge",
        version="2-continuous-governed",
        parameters={"alpha": args.ridge_alpha, "features": features},
    )
    comparison = run_cpcv_effectiveness_comparison(
        governed.dataset,
        label_roll_clean=governed.label_roll_clean,
        estimator_factory=lambda: RidgeEstimator(args.ridge_alpha),
        model_spec=model,
        config=EffectivenessComparisonConfig(
            n_groups=args.n_groups,
            n_test_groups=args.n_test_groups,
            walk_forward_splits=args.walk_forward_splits,
            embargo_sessions=args.embargo_sessions,
            pre_test_gap_sessions=args.pre_test_gap_sessions,
            min_train_observations=args.min_train_observations,
            min_train_sessions=args.min_train_sessions,
            min_train_assets=args.min_train_assets,
        ),
    )
    dataset = governed.dataset
    payload = {
        "status": "success",
        "input": {
            "data_dir": str(data_dir),
            "files": len(files),
            "raw_rows": len(frame),
            "source_digest": source_digest,
        },
        "governance": governed.receipt.canonical(),
        "dataset": {
            "observations": len(dataset.sample_ids),
            "assets": len(set(dataset.asset_ids or ())),
            "sessions": len(set(dataset.sessions)),
            "digest": dataset.digest,
        },
        "report": comparison.canonical(),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
