from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from purged_kfold_validation.strategy_cli import main


def _sessions(count: int) -> np.ndarray:
    return np.datetime64("2025-01-01", "D") + np.arange(count).astype("timedelta64[D]")


def test_strategy_cli_demo_returns_standard_json(capsys: object) -> None:
    assert main(["demo"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)

    assert payload["status"] == "success"
    assert payload["action"] == "benchmark-tsmom"
    assert payload["report"]["benchmark_kind"] == (
        "tsmom-selection-overfitting-benchmark"
    )
    assert payload["report"]["scenarios"][0]["analysis"]["pbo"]["trial_count"] == 32
    assert payload["report"]["acceptance"]["validation_tool_status"] == "PASS"
    assert payload["report"]["acceptance"]["production_gate_status"] in {
        "PASS",
        "FAIL",
        "INCONCLUSIVE",
    }


def test_strategy_cli_analyzes_caller_owned_return_matrix(
    tmp_path: Path, capsys: object
) -> None:
    rows = 120
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0001, 0.01, size=(rows, 3))
    data_path = tmp_path / "returns.npz"
    np.savez(
        data_path,
        sessions=_sessions(rows),
        candidate_ids=np.asarray(("one", "two", "three")),
        gross_returns=returns,
        net_returns=returns,
        turnover=np.zeros_like(returns),
    )
    request = {
        "schema_version": "1",
        "action": "analyze-return-matrix",
        "data_path": data_path.name,
        "options": {
            "cscv_groups": 6,
            "cpcv_groups": 4,
            "cpcv_test_groups": 2,
            "walk_forward_windows": 3,
            "minimum_train_sessions": 30,
        },
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    assert main(["run", "--request", str(request_path)]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)

    assert payload["status"] == "success"
    assert payload["action"] == "analyze-return-matrix"
    assert payload["report"]["pbo"]["trial_count"] == 3
    assert payload["report"]["cpcv"]["path_count"] == 3


def test_strategy_cli_rejects_pickle_bearing_or_wrong_npz_contract(
    tmp_path: Path, capsys: object
) -> None:
    data_path = tmp_path / "bad.npz"
    np.savez(data_path, sessions=_sessions(10))
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "action": "analyze-return-matrix",
                "data_path": data_path.name,
            }
        ),
        encoding="utf-8",
    )

    assert main(["run", "--request", str(request_path)]) == 2
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)

    assert payload["status"] == "error"
    assert payload["errors"][0]["code"] == "strategy-benchmark-rejected"


def test_strategy_cli_rejects_object_identifiers_that_require_pickle(
    tmp_path: Path, capsys: object
) -> None:
    rows = 80
    returns = np.zeros((rows, 2), dtype=np.float64)
    data_path = tmp_path / "pickle-bearing.npz"
    np.savez(
        data_path,
        sessions=_sessions(rows),
        candidate_ids=np.asarray([{"id": "one"}, {"id": "two"}], dtype=object),
        gross_returns=returns,
        net_returns=returns,
        turnover=np.zeros_like(returns),
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "action": "analyze-return-matrix",
                "data_path": data_path.name,
            }
        ),
        encoding="utf-8",
    )

    assert main(["run", "--request", str(request_path)]) == 2
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)

    assert payload["status"] == "error"
    assert "Object arrays cannot be loaded" in payload["errors"][0]["message"]


def test_strategy_cli_runs_trainable_temporal_model_comparison(
    tmp_path: Path, capsys: object
) -> None:
    rows = 80
    rng = np.random.default_rng(20260804)
    returns = rng.normal(0.0001, 0.01, size=(rows, 3))
    returns[0] = 0.0
    prices = np.asarray((100.0, 110.0, 120.0)) * np.cumprod(1.0 + returns, axis=0)
    data_path = tmp_path / "panel.npz"
    np.savez(
        data_path,
        sessions=_sessions(rows),
        asset_ids=np.asarray(("A", "B", "C")),
        signal_prices=prices,
        tradable_returns=returns,
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "action": "benchmark-temporal-models",
                "data_path": data_path.name,
                "options": {
                    "models": ["numpy-ridge"],
                    "lookback_sessions": 5,
                    "label_horizon_sessions": 2,
                    "n_splits": 3,
                    "embargo_sessions": 5,
                    "pre_test_gap_sessions": 2,
                    "walk_forward_test_sessions": 8,
                    "cpcv_groups": 4,
                    "cpcv_test_groups": 2,
                    "minimum_train_observations": 30,
                    "minimum_train_sessions": 10,
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(["run", "--request", str(request_path)]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)

    assert payload["status"] == "success"
    assert payload["action"] == "benchmark-temporal-models"
    assert payload["report"]["decision"]["leakage_control_status"] == "PASS"
    assert payload["report"]["decision"]["production_authorization"] == (
        "NOT_AUTHORIZED"
    )
    assert len(payload["report"]["models"][0]["channels"]) == 6
