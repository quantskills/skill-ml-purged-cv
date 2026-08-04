"""Installed JSON command for time-series strategy-selection benchmarks."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
import sys
from typing import Mapping, NoReturn, Sequence

import numpy as np

from .errors import StrategyBenchmarkError, ValidationError
from .strategy_acceptance import assess_time_series_benchmark
from .strategy_benchmark import (
    DensePricePanel,
    StrategyBenchmarkConfig,
    StrategyReturnMatrix,
    TimeSeriesBenchmarkConfig,
    analyze_strategy_return_matrix,
    run_time_series_strategy_benchmark,
)
from .temporal_model_benchmark import (
    TemporalDatasetSpec,
    TemporalModelBenchmarkConfig,
    build_temporal_supervised_dataset,
    run_temporal_model_benchmark,
)
from .temporal_models import registered_temporal_model_cases


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise StrategyBenchmarkError(message)


def _parser() -> _Parser:
    parser = _Parser(prog="purged-cv-strategy")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="run a deterministic TSMOM demonstration")
    run = commands.add_parser("run", help="run one JSON request")
    run.add_argument("--request", required=True, type=Path)
    schema = commands.add_parser("schema", help="print a packaged JSON schema")
    schema.add_argument("--kind", choices=("request", "result"), required=True)
    return parser


def _json(payload: object) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StrategyBenchmarkError(f"{name} must be a JSON object")
    return value


def _integer(options: Mapping[str, object], name: str, default: int) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise StrategyBenchmarkError(f"options.{name} must be an integer")
    return value


def _number(options: Mapping[str, object], name: str, default: float) -> float:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrategyBenchmarkError(f"options.{name} must be a number")
    return float(value)


def _config(options: Mapping[str, object]) -> StrategyBenchmarkConfig:
    allowed = {
        "annualization_sessions",
        "cscv_groups",
        "cpcv_groups",
        "cpcv_test_groups",
        "cpcv_embargo_sessions",
        "walk_forward_windows",
        "minimum_train_sessions",
        "cost_bps",
        "cost_scenarios_bps",
    }
    unknown = set(options) - allowed
    if unknown:
        raise StrategyBenchmarkError(
            f"unknown options: {', '.join(sorted(str(item) for item in unknown))}"
        )
    return StrategyBenchmarkConfig(
        annualization_sessions=_integer(options, "annualization_sessions", 252),
        cscv_groups=_integer(options, "cscv_groups", 8),
        cpcv_groups=_integer(options, "cpcv_groups", 6),
        cpcv_test_groups=_integer(options, "cpcv_test_groups", 2),
        cpcv_embargo_sessions=_integer(options, "cpcv_embargo_sessions", 0),
        walk_forward_windows=_integer(options, "walk_forward_windows", 5),
        minimum_train_sessions=_integer(options, "minimum_train_sessions", 63),
    )


def _temporal_options(
    options: Mapping[str, object],
) -> tuple[
    TemporalDatasetSpec, TemporalModelBenchmarkConfig, tuple[str, ...], int, int, float
]:
    allowed = {
        "lookback_sessions",
        "label_horizon_sessions",
        "n_splits",
        "embargo_sessions",
        "pre_test_gap_sessions",
        "walk_forward_test_sessions",
        "cpcv_groups",
        "cpcv_test_groups",
        "minimum_train_observations",
        "minimum_train_sessions",
        "random_seed",
        "models",
        "lightgbm_estimators",
        "lstm_epochs",
        "ridge_alpha",
    }
    unknown = set(options) - allowed
    if unknown:
        raise StrategyBenchmarkError(
            f"unknown options: {', '.join(sorted(str(item) for item in unknown))}"
        )
    raw_models = options.get("models", ["numpy-ridge", "lightgbm", "torch-lstm"])
    if (
        not isinstance(raw_models, list)
        or not raw_models
        or any(not isinstance(item, str) for item in raw_models)
    ):
        raise StrategyBenchmarkError("options.models must be a non-empty string array")
    models = tuple(str(item) for item in raw_models)
    allowed_models = {"numpy-ridge", "lightgbm", "torch-lstm"}
    if len(set(models)) != len(models) or set(models) - allowed_models:
        raise StrategyBenchmarkError(
            "options.models must be unique values from numpy-ridge, lightgbm, torch-lstm"
        )
    return (
        TemporalDatasetSpec(
            lookback_sessions=_integer(options, "lookback_sessions", 20),
            label_horizon_sessions=_integer(options, "label_horizon_sessions", 5),
        ),
        TemporalModelBenchmarkConfig(
            n_splits=_integer(options, "n_splits", 5),
            embargo_sessions=_integer(options, "embargo_sessions", 20),
            pre_test_gap_sessions=_integer(options, "pre_test_gap_sessions", 5),
            walk_forward_test_sessions=_integer(
                options, "walk_forward_test_sessions", 120
            ),
            cpcv_groups=_integer(options, "cpcv_groups", 6),
            cpcv_test_groups=_integer(options, "cpcv_test_groups", 2),
            minimum_train_observations=_integer(
                options, "minimum_train_observations", 3000
            ),
            minimum_train_sessions=_integer(options, "minimum_train_sessions", 252),
            random_seed=_integer(options, "random_seed", 20260804),
        ),
        models,
        _integer(options, "lightgbm_estimators", 40),
        _integer(options, "lstm_epochs", 2),
        _number(options, "ridge_alpha", 1.0),
    )


def _load_request(path: Path) -> tuple[Mapping[str, object], Path]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise StrategyBenchmarkError("request file does not exist")
    if resolved.stat().st_size > 1_048_576:
        raise StrategyBenchmarkError("request file exceeds 1 MiB")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StrategyBenchmarkError(f"request JSON cannot be read: {exc}") from exc
    request = _mapping(document, name="request")
    allowed = {"schema_version", "action", "data_path", "options"}
    unknown = set(request) - allowed
    if unknown:
        raise StrategyBenchmarkError(
            f"unknown request fields: {', '.join(sorted(str(item) for item in unknown))}"
        )
    if request.get("schema_version") != "1":
        raise StrategyBenchmarkError("schema_version must equal '1'")
    return request, resolved


def _data_path(request: Mapping[str, object], request_path: Path) -> Path:
    value = request.get("data_path")
    if not isinstance(value, str) or not value.strip():
        raise StrategyBenchmarkError("data_path must be a non-empty string")
    path = Path(value)
    resolved = (
        (request_path.parent / path).resolve()
        if not path.is_absolute()
        else path.resolve()
    )
    if not resolved.is_file() or resolved.suffix.lower() != ".npz":
        raise StrategyBenchmarkError("data_path must identify an existing .npz file")
    if resolved.stat().st_size > 536_870_912:
        raise StrategyBenchmarkError("data_path exceeds 512 MiB")
    return resolved


def _source_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(value: np.ndarray, *, name: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise StrategyBenchmarkError(f"{name} must be a one-dimensional string array")
    return tuple(str(item) for item in array.tolist())


def _execute(request_path: Path) -> tuple[str, dict[str, object]]:
    request, resolved_request = _load_request(request_path)
    action = request.get("action")
    if action not in {
        "analyze-return-matrix",
        "benchmark-tsmom",
        "benchmark-temporal-models",
    }:
        raise StrategyBenchmarkError(
            "action must be 'analyze-return-matrix', 'benchmark-tsmom', "
            "or 'benchmark-temporal-models'"
        )
    options = _mapping(request.get("options", {}), name="options")
    analysis = None if action == "benchmark-temporal-models" else _config(options)
    path = _data_path(request, resolved_request)
    source_digest = _source_digest(path)
    try:
        archive_context = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise StrategyBenchmarkError(f"NPZ input cannot be opened: {exc}") from exc
    with archive_context as archive:
        if action == "analyze-return-matrix":
            required = {
                "sessions",
                "candidate_ids",
                "gross_returns",
                "net_returns",
                "turnover",
            }
            if set(archive.files) != required:
                raise StrategyBenchmarkError(
                    "return-matrix NPZ keys must exactly equal "
                    + ", ".join(sorted(required))
                )
            matrix = StrategyReturnMatrix(
                sessions=np.asarray(archive["sessions"]),
                candidate_ids=_ids(archive["candidate_ids"], name="candidate_ids"),
                gross_returns=np.asarray(archive["gross_returns"]),
                net_returns=np.asarray(archive["net_returns"]),
                turnover=np.asarray(archive["turnover"]),
                source_digest=source_digest,
                cost_bps=_number(options, "cost_bps", 0.0),
            )
            assert analysis is not None
            return action, analyze_strategy_return_matrix(matrix, analysis).canonical()

        required = {"sessions", "asset_ids", "signal_prices", "tradable_returns"}
        if set(archive.files) != required:
            raise StrategyBenchmarkError(
                "TSMOM NPZ keys must exactly equal " + ", ".join(sorted(required))
            )
        costs_value = options.get("cost_scenarios_bps", [0.0, 1.0, 3.0, 5.0])
        if not isinstance(costs_value, list):
            raise StrategyBenchmarkError("options.cost_scenarios_bps must be an array")
        costs = tuple(float(item) for item in costs_value)
        panel = DensePricePanel(
            sessions=np.asarray(archive["sessions"]),
            asset_ids=_ids(archive["asset_ids"], name="asset_ids"),
            signal_prices=np.asarray(archive["signal_prices"]),
            tradable_returns=np.asarray(archive["tradable_returns"]),
            source_digest=source_digest,
        )
        if action == "benchmark-temporal-models":
            (
                dataset_spec,
                temporal_config,
                requested_models,
                lightgbm_estimators,
                lstm_epochs,
                ridge_alpha,
            ) = _temporal_options(options)
            available = {
                {
                    "numpy-ridge-lag-sequence": "numpy-ridge",
                    "lightgbm-lag-sequence": "lightgbm",
                    "torch-lstm-lag-sequence": "torch-lstm",
                }[case.model_spec.name]: case
                for case in registered_temporal_model_cases(
                    ridge_alpha=ridge_alpha,
                    lightgbm_estimators=lightgbm_estimators,
                    lstm_epochs=lstm_epochs,
                    random_seed=temporal_config.random_seed,
                    require_optional=False,
                )
            }
            missing = [name for name in requested_models if name not in available]
            if missing:
                raise StrategyBenchmarkError(
                    "requested temporal models are unavailable: "
                    + ", ".join(missing)
                    + "; install purged-kfold-validation[temporal-models]"
                )
            dataset = build_temporal_supervised_dataset(panel, dataset_spec)
            payload = run_temporal_model_benchmark(
                dataset,
                tuple(available[name] for name in requested_models),
                temporal_config,
            ).canonical()
            payload["dataset_spec"] = dataset_spec.canonical()
            payload["dataset_spec_digest"] = dataset_spec.digest
            return action, payload
        assert analysis is not None
        config = TimeSeriesBenchmarkConfig(
            cost_scenarios_bps=costs,
            analysis=analysis,
        )
        benchmark = run_time_series_strategy_benchmark(panel, config)
        payload = benchmark.canonical()
        payload["acceptance"] = assess_time_series_benchmark(benchmark).canonical()
        return action, payload


def _demo() -> dict[str, object]:
    rows = 320
    sessions = np.datetime64("2024-01-01", "D") + np.arange(rows).astype(
        "timedelta64[D]"
    )
    rng = np.random.default_rng(20260803)
    returns = rng.normal(0.0, 0.012, size=(rows, 3))
    returns[1:110] += np.asarray((0.0006, -0.0003, 0.0002))
    returns[110:220] += np.asarray((-0.0005, 0.0005, -0.0002))
    returns[220:] += np.asarray((0.0002, -0.0002, 0.0004))
    returns[0] = 0.0
    prices = np.asarray((100.0, 120.0, 80.0)) * np.cumprod(1.0 + returns, axis=0)
    panel = DensePricePanel(
        sessions=sessions,
        asset_ids=("DEMO-A", "DEMO-B", "DEMO-C"),
        signal_prices=prices,
        tradable_returns=returns,
        source_digest="deterministic-demo-v1",
    )
    config = TimeSeriesBenchmarkConfig()
    benchmark = run_time_series_strategy_benchmark(panel, config)
    payload = benchmark.canonical()
    payload["acceptance"] = assess_time_series_benchmark(benchmark).canonical()
    return payload


def _schema(kind: str) -> dict[str, object]:
    resource = files("purged_kfold_validation").joinpath(
        "resources", "strategy_benchmark", f"{kind}-v1.schema.json"
    )
    return _mapping(json.loads(resource.read_text(encoding="utf-8")), name="schema")  # type: ignore[return-value]


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.command == "schema":
            _json(_schema(str(args.kind)))
            return 0
        if args.command == "demo":
            action = "benchmark-tsmom"
            report = _demo()
        else:
            action, report = _execute(args.request)
        _json(
            {
                "schema_version": "1",
                "status": "success",
                "action": action,
                "report": report,
                "errors": [],
            }
        )
        return 0
    except (ValidationError, OSError, ValueError) as exc:
        _json(
            {
                "schema_version": "1",
                "status": "error",
                "action": None,
                "report": None,
                "errors": [
                    {"code": "strategy-benchmark-rejected", "message": str(exc)}
                ],
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
