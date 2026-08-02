"""Run a redacted five-year PandaData release gate on an existing local cache."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_kfold_validation import (  # noqa: E402
    CausalWalkForward,
    EffectivenessComparisonConfig,
    EvaluationProtocol,
    LeakageSafeEvaluator,
    LocalHoldoutStore,
    MetricSpec,
    ModelSpec,
    ValidationDataset,
    assess_model_ranking_stability,
    run_cpcv_effectiveness_comparison,
)
from purged_kfold_validation.adapters.pandaai import (  # noqa: E402
    PandaAIContinuousMapping,
    PandaAIContinuousPolicy,
    PandaAIDailyConfig,
    PandaAIDailyMapping,
    governed_validation_dataset_from_pandaai_continuous_daily,
)


class MeanEstimator:
    """Deterministic intercept-only baseline."""

    def __init__(self) -> None:
        self.mean: float | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> MeanEstimator:
        del features
        self.mean = float(targets.mean())
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert self.mean is not None
        return np.full(len(features), self.mean)


class RidgeEstimator:
    """Small standardized ridge model used only as a fixed baseline."""

    def __init__(self, alpha: float) -> None:
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


EstimatorFactory = Callable[[], Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Five-year PandaData leakage and final-Holdout release gate"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout-store", required=True)
    parser.add_argument("--existing-holdout-receipt", default="")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--holdout-sessions", type=int, default=252)
    parser.add_argument("--feature-columns", default="close,volume,open_interest")
    parser.add_argument("--label-horizon-sessions", type=int, default=5)
    parser.add_argument("--feature-lookback-sessions", type=int, default=20)
    return parser.parse_args()


def selected_files(data_dir: Path) -> tuple[Path, ...]:
    files = tuple(sorted(data_dir.glob("*_daily.parquet")))
    if not files:
        raise FileNotFoundError("no PandaData daily parquet files found")
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
        raise ValueError("cache filename does not follow *_daily.parquet")
    value = path.stem[: -len(suffix)].strip().upper()
    if not value:
        raise ValueError("cache filename has no asset identity")
    return value


def load_window(
    files: tuple[Path, ...], *, features: tuple[str, ...], years: int
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, int]:
    columns = tuple(
        dict.fromkeys(
            (
                "date",
                "symbol",
                "underlying_symbol",
                "dominant_id",
                "close",
                *features,
            )
        )
    )
    maximums: list[pd.Timestamp] = []
    for path in files:
        dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"])
        if not dates.empty:
            maximums.append(dates.max())
    if not maximums:
        raise ValueError("PandaData cache contains no dates")
    end = max(maximums).normalize()
    start = end - pd.DateOffset(years=years)
    frames: list[pd.DataFrame] = []
    raw_window_rows = 0
    for path in files:
        source = pd.read_parquet(path)
        missing = sorted(set(columns).difference(source.columns))
        if missing:
            if set(missing) == {"underlying_symbol"}:
                source["underlying_symbol"] = pd.NA
            else:
                raise ValueError(f"{path.name} is missing required columns: {missing}")
        dates = pd.to_datetime(source["date"], errors="raise").dt.normalize()
        mask = dates.between(start, end, inclusive="both")
        frame = source.loc[mask, list(columns)].copy()
        if frame.empty:
            continue
        frame["date"] = dates.loc[mask]
        frame["__file_asset"] = _file_asset(path)
        raw_window_rows += len(frame)
        frames.append(frame)
    if not frames:
        raise ValueError("five-year window contains no rows")
    return pd.concat(frames, ignore_index=True), start, end, raw_window_rows


def slice_dataset(
    dataset: ValidationDataset, positions: np.ndarray
) -> ValidationDataset:
    indexes = np.asarray(positions, dtype=int)
    decision_times = dataset.decision_times
    feature_availability = dataset.feature_availability
    assert decision_times is not None
    assert feature_availability is not None
    asset_ids = (
        None
        if dataset.asset_ids is None
        else tuple(dataset.asset_ids[int(index)] for index in indexes)
    )
    return ValidationDataset(
        sample_ids=tuple(dataset.sample_ids[int(index)] for index in indexes),
        asset_ids=asset_ids,
        session_axis=dataset.session_axis,
        sessions=tuple(dataset.sessions[int(index)] for index in indexes),
        information_intervals=tuple(
            dataset.information_intervals[int(index)] for index in indexes
        ),
        decision_times=decision_times[indexes],
        feature_availability=feature_availability[indexes],
        pit_snapshot=dataset.pit_snapshot,
        features=dataset.features[indexes],
        targets=dataset.targets[indexes],
        missing_value_policy=dataset.missing_value_policy,
        feature_manifest_digest=dataset.feature_manifest_digest,
    )


def model_candidates(
    features: tuple[str, ...],
) -> dict[str, tuple[ModelSpec, EstimatorFactory]]:
    return {
        "mean": (
            ModelSpec(name="mean", version="1-five-year", parameters={}),
            MeanEstimator,
        ),
        "ridge-1": (
            ModelSpec(
                name="ridge",
                version="1-five-year",
                parameters={"alpha": 1.0, "features": features},
            ),
            lambda: RidgeEstimator(1.0),
        ),
        "ridge-100": (
            ModelSpec(
                name="ridge",
                version="1-five-year",
                parameters={"alpha": 100.0, "features": features},
            ),
            lambda: RidgeEstimator(100.0),
        ),
    }


def regime_scores(
    observations_by_model: dict[str, tuple[Any, ...]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    first = next(iter(observations_by_model.values()))
    sessions = sorted({item.session for item in first})
    chunks = np.array_split(np.asarray(sessions), 3)
    regimes: dict[str, dict[str, float]] = {}
    overall: dict[str, float] = {}
    for model_name, observations in observations_by_model.items():
        targets = np.asarray([item.target for item in observations])
        predictions = np.asarray([item.prediction for item in observations])
        overall[model_name] = float(np.mean((targets - predictions) ** 2))
    for index, chunk in enumerate(chunks):
        session_set = {np.datetime64(value, "ns") for value in chunk}
        regime = f"chronological-{index + 1}"
        regimes[regime] = {}
        for model_name, observations in observations_by_model.items():
            selected = [item for item in observations if item.session in session_set]
            if not selected:
                raise ValueError(f"{regime} contains no comparable OOS observations")
            targets = np.asarray([item.target for item in selected])
            predictions = np.asarray([item.prediction for item in selected])
            regimes[regime][model_name] = float(np.mean((targets - predictions) ** 2))
    return regimes, overall


def holdout_diagnostics(observations: tuple[Any, ...]) -> dict[str, object]:
    targets = np.asarray([item.target for item in observations])
    predictions = np.asarray([item.prediction for item in observations])
    mse = float(np.mean((targets - predictions) ** 2))
    by_session: dict[np.datetime64, list[Any]] = {}
    for item in observations:
        by_session.setdefault(item.session, []).append(item)
    ics: list[float] = []
    daily_scores: list[float] = []
    for items in by_session.values():
        if len(items) >= 3:
            actual = pd.Series([item.target for item in items]).rank().to_numpy()
            predicted = pd.Series([item.prediction for item in items]).rank().to_numpy()
            correlation = float(np.corrcoef(actual, predicted)[0, 1])
            if np.isfinite(correlation):
                ics.append(correlation)
        signals = np.asarray([item.prediction for item in items])
        returns = np.asarray([item.target for item in items])
        gross = float(np.abs(signals - signals.mean()).sum())
        if gross > 0.0:
            daily_scores.append(
                float(np.sum((signals - signals.mean()) * returns) / gross)
            )
    sharpe = 0.0
    if len(daily_scores) >= 2 and float(np.std(daily_scores, ddof=1)) > 0.0:
        sharpe = float(
            np.sqrt(252.0) * np.mean(daily_scores) / np.std(daily_scores, ddof=1)
        )
    return {
        "mse": mse,
        "cross_sectional_ic": float(np.mean(ics)) if ics else 0.0,
        "diagnostic_sharpe": sharpe,
        "observations": len(observations),
        "sessions": len(by_session),
        "ic_sessions": len(ics),
    }


def main() -> int:
    args = parse_args()
    if args.years < 1 or args.holdout_sessions < 20:
        raise ValueError("years and holdout-sessions are below the supported minimum")
    data_dir = Path(args.data_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    holdout_store = Path(args.holdout_store).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            "output already exists; release-gate evidence is append-only"
        )
    features = tuple(
        value.strip() for value in args.feature_columns.split(",") if value.strip()
    )
    files = selected_files(data_dir)
    source_digest = files_digest(files)
    frame, window_start, window_end, raw_rows = load_window(
        files, features=features, years=args.years
    )
    governed = governed_validation_dataset_from_pandaai_continuous_daily(
        frame,
        mapping=PandaAIDailyMapping(
            session="date",
            asset="__file_asset",
            close="close",
            features=features,
        ),
        config=PandaAIDailyConfig(
            label_horizon_sessions=args.label_horizon_sessions,
            feature_lookback_sessions=args.feature_lookback_sessions,
            decision_time_offset_minutes=15 * 60,
            snapshot_id=f"pandadata-five-year-{source_digest[:12]}",
            source_digest=source_digest,
        ),
        continuous_mapping=PandaAIContinuousMapping(
            series_symbol="symbol",
            asset="__file_asset",
            active_contract="dominant_id",
            declared_underlying="underlying_symbol",
        ),
        policy=PandaAIContinuousPolicy(),
    )
    dataset = governed.dataset
    active_sessions = sorted(set(dataset.sessions))
    if len(active_sessions) <= args.holdout_sessions + 504:
        raise ValueError("five-year dataset is too short for development and holdout")
    holdout_start = active_sessions[-args.holdout_sessions]
    holdout_positions = np.flatnonzero(dataset.sessions >= holdout_start)
    minimum_holdout_information = min(
        dataset.information_intervals[int(index)].start for index in holdout_positions
    )
    development_positions = np.asarray(
        [
            index
            for index, interval in enumerate(dataset.information_intervals)
            if interval.end < minimum_holdout_information
        ],
        dtype=int,
    )
    development = slice_dataset(dataset, development_positions)
    holdout = slice_dataset(dataset, holdout_positions)
    development_roll_clean = governed.label_roll_clean[development_positions]

    development_sessions = sorted(set(development.sessions))
    test_sessions = max(
        40,
        (len(development_sessions) - 320) // 5,
    )
    walk_forward = CausalWalkForward(
        n_splits=5,
        test_sessions=test_sessions,
        pre_test_gap_sessions=args.feature_lookback_sessions,
        min_train_sessions=252,
        min_train_samples=5_000,
    )
    candidates = model_candidates(features)
    observations_by_model: dict[str, tuple[Any, ...]] = {}
    for name, (model_spec, factory) in candidates.items():
        result = LeakageSafeEvaluator(
            splitter=walk_forward,
            estimator_factory=factory,
            model_spec=model_spec,
        ).evaluate(development)
        observations_by_model[name] = result.ledger.observations
    scores, overall_mse = regime_scores(observations_by_model)
    ranking = assess_model_ranking_stability(
        scores, objective="minimize", min_pairwise_spearman=0.5
    )
    selected_name = min(overall_mse, key=overall_mse.__getitem__)
    selected_spec, selected_factory = candidates[selected_name]
    comparison_model_name = "ridge-1"
    comparison_spec, comparison_factory = candidates[comparison_model_name]

    comparison = run_cpcv_effectiveness_comparison(
        development,
        label_roll_clean=development_roll_clean,
        estimator_factory=comparison_factory,
        model_spec=comparison_spec,
        config=EffectivenessComparisonConfig(
            n_groups=6,
            n_test_groups=2,
            walk_forward_splits=2,
            embargo_sessions=args.feature_lookback_sessions,
            pre_test_gap_sessions=args.feature_lookback_sessions,
            min_train_observations=5_000,
            min_train_sessions=252,
            min_train_assets=20,
        ),
    )
    mse_metric = MetricSpec(
        name="mse",
        version="1",
        function=lambda actual, predicted: float(np.mean((actual - predicted) ** 2)),
    )
    protocol = EvaluationProtocol.freeze(
        protocol_id="pandadata-five-year-release-gate-v1",
        training_dataset=development,
        holdout_dataset=holdout,
        model_spec=selected_spec,
        metrics=(mse_metric,),
        search_policy={
            "kind": "five-year-causal-walk-forward-fixed-candidates",
            "candidate_model_digests": {
                name: spec.digest for name, (spec, _) in candidates.items()
            },
            "ranking_report_digest": ranking.digest,
            "selection_metric": "overall-walk-forward-mse",
            "selected_model": selected_name,
        },
        split_spec_digest=walk_forward.digest,
    )
    if args.existing_holdout_receipt:
        receipt_path = Path(args.existing_holdout_receipt).expanduser().resolve()
        receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt_payload, dict):
            raise ValueError("existing Holdout receipt must be a JSON object")
        if receipt_payload.get("protocol_digest") != protocol.digest:
            raise ValueError("existing Holdout receipt does not match the protocol")
        if receipt_payload.get("holdout_dataset_digest") != holdout.digest:
            raise ValueError("existing Holdout receipt does not match the dataset")
        metric_values = receipt_payload.get("metric_values")
        if not isinstance(metric_values, dict) or not isinstance(
            metric_values.get("mse@1"), (int, float)
        ):
            raise ValueError("existing Holdout receipt has no valid MSE")
        holdout_diagnostic_payload: dict[str, object] = {
            "mse": float(metric_values["mse@1"]),
            "observations": len(holdout.sample_ids),
            "sessions": len(set(holdout.sessions)),
            "cross_sectional_ic": None,
            "diagnostic_sharpe": None,
            "recovery_boundary": (
                "predictions were intentionally not persisted; IC and Sharpe "
                "cannot be reconstructed without prohibited Holdout reuse"
            ),
        }
    else:
        final = LocalHoldoutStore(holdout_store).evaluate_once(
            protocol,
            training_dataset=development,
            holdout_dataset=holdout,
            estimator_factory=selected_factory,
            model_spec=selected_spec,
            metrics=(mse_metric,),
        )
        receipt_payload = final.receipt.canonical()
        holdout_diagnostic_payload = holdout_diagnostics(
            final.result.ledger.observations
        )
    payload: dict[str, object] = {
        "status": "success",
        "window": {
            "start": str(window_start.date()),
            "end": str(window_end.date()),
            "raw_rows": raw_rows,
            "files": len(files),
            "source_digest": source_digest,
        },
        "governance": governed.receipt.canonical(),
        "dataset": {
            "eligible_observations": len(dataset.sample_ids),
            "assets": len(set(dataset.asset_ids or ())),
            "sessions": len(set(dataset.sessions)),
            "digest": dataset.digest,
        },
        "development": {
            "observations": len(development.sample_ids),
            "sessions": len(set(development.sessions)),
            "digest": development.digest,
            "walk_forward_overall_mse": overall_mse,
            "regime_mse": scores,
            "ranking": ranking.canonical(),
            "selected_model": selected_name,
            "comparison_model": comparison_model_name,
            "comparison": comparison.canonical(),
        },
        "holdout": {
            "observations": len(holdout.sample_ids),
            "sessions": len(set(holdout.sessions)),
            "start": str(pd.Timestamp(min(holdout.sessions)).date()),
            "digest": holdout.digest,
            "protocol_digest": protocol.digest,
            "receipt": receipt_payload,
            "diagnostics": holdout_diagnostic_payload,
        },
        "claim_boundary": (
            "five-year local validation evidence; no transaction costs, external "
            "metadata truth, profitability, remote CI, or deployment claim"
        ),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "success",
                "output": str(output),
                "selected_model": selected_name,
                "ranking_stable": ranking.stable,
                "holdout_receipt_digest": receipt_payload["digest"],
                "comparison_digest": comparison.digest,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
