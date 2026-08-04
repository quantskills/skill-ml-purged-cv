from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_skill(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "purged_kfold_validation.agent_cli", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("kind", ("request", "result"))
def test_agent_skill_discovers_packaged_schema(kind: str) -> None:
    completed = _run_skill("schema", "--kind", kind)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert payload["type"] == "object"


def test_agent_skill_materializes_one_request_bundle_and_runs_it(
    tmp_path: Path,
) -> None:
    output = tmp_path / "quickstart"
    created = _run_skill("example", "--output-dir", str(output))

    assert created.returncode == 0, created.stdout + created.stderr
    assert json.loads(created.stdout)["files"] == [
        "features.csv",
        "manifest.json",
        "mapping.json",
        "request.json",
    ]

    completed = _run_skill("run", "--request", str(output / "request.json"))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "1"
    assert payload["status"] == "success"
    assert payload["action"] == "audit"
    assert len(payload["request_digest"]) == 64
    assert payload["authoritative_cli_result"]["stage"] == "audit"
    assert payload["errors"] == []
    assert str(tmp_path) not in completed.stdout
    assert "sample_id" not in completed.stdout
    assert "predictions" not in completed.stdout


def test_agent_skill_demo_is_one_command_smoke_test() -> None:
    completed = _run_skill("demo")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "success"
    assert payload["authoritative_cli_result"]["governance"]["features"] == 1


def test_agent_skill_rejects_unknown_request_fields_without_starting_engine(
    tmp_path: Path,
) -> None:
    request = {
        "schema_version": "1",
        "action": "audit",
        "data_path": "features.csv",
        "manifest_path": "manifest.json",
        "mapping_path": "mapping.json",
        "unsafe_extra": True,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    completed = _run_skill("run", "--request", str(request_path))

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "rejected"
    assert payload["errors"] == [
        {
            "error_type": "AgentRequestError",
            "message": "request fields do not match the closed version-1 contract",
        }
    ]


def test_agent_skill_rejects_out_of_range_option_before_starting_engine(
    tmp_path: Path,
) -> None:
    request = {
        "schema_version": "1",
        "action": "evaluate",
        "data_path": "features.csv",
        "manifest_path": "manifest.json",
        "mapping_path": "mapping.json",
        "options": {"n_groups": 1},
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    completed = _run_skill("run", "--request", str(request_path))

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["errors"][0]["message"] == "option n_groups must be at least 2"
