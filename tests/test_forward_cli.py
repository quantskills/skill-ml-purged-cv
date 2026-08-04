from __future__ import annotations

import json
from pathlib import Path

import pytest

from purged_kfold_validation.forward_cli import main
from purged_kfold_validation.temporal_forward import TemporalForwardProtocol


def _write_protocol(path: Path) -> TemporalForwardProtocol:
    protocol = TemporalForwardProtocol(
        protocol_id="cli-forward-v1",
        development_report_digest="a" * 64,
        development_dataset_digest="b" * 64,
        model_spec_digest="c" * 64,
        temporal_dataset_spec_digest="d" * 64,
        development_label_end_session="2026-08-03",
        forward_start_session="2026-08-04",
        label_horizon_sessions=5,
        minimum_matured_sessions=252,
        minimum_matured_observations=3000,
        minimum_assets=8,
        minimum_mean_session_spearman_ic=0.0,
        require_model_mse_not_worse_than_zero=True,
        selection_policy={"kind": "frozen"},
    )
    path.write_text(json.dumps(protocol.canonical()), encoding="utf-8")
    return protocol


def test_cli_init_and_status_return_redacted_waiting_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(protocol_path)
    store = tmp_path / "store"
    assert (
        main(
            [
                "init",
                "--protocol",
                str(protocol_path),
                "--store",
                str(store),
            ]
        )
        == 0
    )
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["result"]["status"] == "WAITING_FOR_FUTURE_DATA"
    assert initialized["result"]["protocol_digest"] == protocol.digest

    assert (
        main(
            [
                "status",
                "--protocol",
                str(protocol_path),
                "--store",
                str(store),
                "--persist",
            ]
        )
        == 0
    )
    status = json.loads(capsys.readouterr().out)
    assert status["result"]["production_authorization"] == "NOT_AUTHORIZED"
    assert len(list((store / "reports").glob("*.json"))) == 1


def test_cli_rejects_unknown_prediction_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    protocol_path = tmp_path / "protocol.json"
    _write_protocol(protocol_path)
    store = tmp_path / "store"
    assert main(["init", "--protocol", str(protocol_path), "--store", str(store)]) == 0
    capsys.readouterr()
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "predictions": [
                    {
                        "sample_id": "x",
                        "asset_id": "A",
                        "decision_session": "2026-08-04",
                        "label_end_session": "2026-08-11",
                        "label_available_at": "2099-01-01T00:00:00+00:00",
                        "prediction": 0.0,
                        "feature_snapshot_digest": "f" * 64,
                        "target": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "record",
                "--protocol",
                str(protocol_path),
                "--store",
                str(store),
                "--batch",
                str(batch),
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["errors"][0]["code"] == "forward-evidence-rejected"
    assert not (store / "predictions").exists()
