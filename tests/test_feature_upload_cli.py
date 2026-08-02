from __future__ import annotations

import json
from math import comb
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, cast

import pandas as pd
import pytest

from purged_kfold_validation import UploadLimitError
from purged_kfold_validation.adapters.upload import (
    FeatureUploadLimits,
    load_governed_feature_upload,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIGEST = "a" * 64
FEATURE_DIGEST = "b" * 64
CODE_DIGEST = "c" * 64


def _write_upload_bundle(
    root: Path,
    *,
    uses_target: bool = False,
    late_availability: bool = False,
) -> tuple[Path, Path, Path]:
    sessions = pd.date_range("2025-01-02", periods=12, freq="D")
    rows: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions):
        for asset_index, asset in enumerate(("A", "B", "C")):
            availability = session + pd.Timedelta(hours=18 if late_availability else 8)
            target = float(asset_index - 1) * (session_index + 1) / 100.0
            rows.append(
                {
                    "sample_id": f"{session.date()}::{asset}",
                    "session": session.isoformat(),
                    "asset": asset,
                    "interval_start": session.isoformat(),
                    "interval_end": session.isoformat(),
                    "decision_time": (session + pd.Timedelta(hours=12)).isoformat(),
                    "target": target,
                    "signal": target,
                    "signal_available": availability.isoformat(),
                }
            )
    data_path = root / "features.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    manifest = {
        "schema_version": "1",
        "source_bundle_digest": SOURCE_DIGEST,
        "features": [
            {
                "name": "signal",
                "source_dataset": "fixture.daily",
                "source_fields": ["close"],
                "source_digest": FEATURE_DIGEST,
                "transformation": "fixture-signal",
                "transformation_version": "1",
                "code_digest": CODE_DIGEST,
                "parameters": {},
                "lookback_sessions": 1,
                "computation_scope": "precomputed-stateless",
                "revision_policy": "point-in-time",
                "uses_target": uses_target,
            }
        ],
    }
    mapping = {
        "schema_version": "1",
        "sample_id_column": "sample_id",
        "session_column": "session",
        "asset_id_column": "asset",
        "interval_start_column": "interval_start",
        "interval_end_column": "interval_end",
        "decision_time_column": "decision_time",
        "target_column": "target",
        "feature_columns": ["signal"],
        "availability_columns": ["signal_available"],
        "session_axis": [session.isoformat() for session in sessions],
        "pit_snapshot": {
            "snapshot_id": "upload-fixture-v1",
            "source_digest": SOURCE_DIGEST,
            "revision_policy": "point-in-time",
        },
    }
    manifest_path = root / "manifest.json"
    mapping_path = root / "mapping.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    return data_path, manifest_path, mapping_path


def _run(
    command: str,
    data_path: Path,
    manifest_path: Path,
    mapping_path: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/audit_feature_upload.py",
            command,
            "--data",
            str(data_path),
            "--manifest",
            str(manifest_path),
            "--mapping",
            str(mapping_path),
            *extra,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _run_module(
    command: str,
    data_path: Path,
    manifest_path: Path,
    mapping_path: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "purged_kfold_validation",
            command,
            "--data",
            str(data_path),
            "--manifest",
            str(manifest_path),
            "--mapping",
            str(mapping_path),
            *extra,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_installed_style_module_audit_matches_legacy_script(tmp_path: Path) -> None:
    paths = _write_upload_bundle(tmp_path)

    legacy = _run("audit", *paths)
    module = _run_module("audit", *paths)

    assert module.returncode == 0, module.stdout + module.stderr
    assert json.loads(module.stdout) == json.loads(legacy.stdout)


@pytest.mark.parametrize(
    ("kind", "file_name"),
    (
        ("manifest", "feature-manifest.schema.json"),
        ("mapping", "upload-mapping.schema.json"),
    ),
)
def test_module_prints_installed_canonical_schema(kind: str, file_name: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "purged_kfold_validation",
            "schema",
            "--kind",
            kind,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    expected = json.loads(
        (ROOT / "config" / "feature-upload" / file_name).read_text(encoding="utf-8")
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == expected


@pytest.mark.parametrize(
    ("example", "audit_code"),
    (("raw", 0), ("stationary", 0), ("intentional-leak", 2)),
)
def test_module_materializes_installed_example_with_canonical_behavior(
    tmp_path: Path, example: str, audit_code: int
) -> None:
    output = tmp_path / f"{example}-example"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "purged_kfold_validation",
            "example",
            "--name",
            example,
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == {
        "example": example,
        "files": ["features.csv", "manifest.json", "mapping.json"],
        "stage": "example",
        "status": "success",
    }
    repository_example = ROOT / "config" / "feature-upload" / "examples" / example
    for name in ("features.csv", "manifest.json", "mapping.json"):
        assert (output / name).read_bytes() == (repository_example / name).read_bytes()

    audited = _run_module(
        "audit",
        output / "features.csv",
        output / "manifest.json",
        output / "mapping.json",
    )
    assert audited.returncode == audit_code, audited.stdout + audited.stderr


def test_example_export_refuses_known_conflict_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "conflict"
    output.mkdir()
    existing = output / "manifest.json"
    existing.write_text("owner-content", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "purged_kfold_validation",
            "example",
            "--name",
            "raw",
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["error_type"] == "AdapterValidationError"
    assert payload["message"] == "example export refuses to overwrite existing files"
    assert existing.read_text(encoding="utf-8") == "owner-content"
    assert not (output / "features.csv").exists()
    assert not (output / "mapping.json").exists()


def test_audit_accepts_explicit_csv_contract_without_exposing_rows(
    tmp_path: Path,
) -> None:
    paths = _write_upload_bundle(tmp_path)

    completed = _run("audit", *paths)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "success"
    assert payload["stage"] == "audit"
    assert payload["input"]["file_name"] == "features.csv"
    assert payload["dataset"]["observations"] == 36
    assert payload["dataset"]["features"] == 1
    assert payload["governance"]["availability_cells"] == 36
    assert str(tmp_path) not in completed.stdout
    assert "sample_id" not in completed.stdout
    assert "predictions" not in completed.stdout


def test_audit_rejects_target_dependent_feature_with_safe_json(tmp_path: Path) -> None:
    paths = _write_upload_bundle(tmp_path, uses_target=True)

    completed = _run("audit", *paths)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload == {
        "error_type": "DatasetValidationError",
        "message": (
            "uploaded features must be target-independent; target-derived state "
            "belongs in fold-local transformer factories"
        ),
        "stage": "audit",
        "status": "rejected",
    }
    assert completed.stderr == ""


def test_audit_rejects_row_limit_before_governance(tmp_path: Path) -> None:
    paths = _write_upload_bundle(tmp_path)

    completed = _run("audit", *paths, "--max-rows", "10")

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["error_type"] == "UploadLimitError"
    assert payload["message"] == "data row count exceeds max_rows"
    assert payload["stage"] == "audit"


def test_csv_row_budget_uses_bounded_read_before_full_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_upload_bundle(tmp_path)
    original = cast(Callable[..., pd.DataFrame], pd.read_csv)
    observed_nrows: list[int | None] = []

    def recording_read_csv(*args: object, **kwargs: object) -> pd.DataFrame:
        value = kwargs.get("nrows")
        observed_nrows.append(value if isinstance(value, int) else None)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)

    with pytest.raises(UploadLimitError, match="row count"):
        load_governed_feature_upload(
            paths[0],
            manifest_path=paths[1],
            mapping_path=paths[2],
            limits=FeatureUploadLimits(max_rows=10),
        )

    assert 11 in observed_nrows
    assert None not in observed_nrows


def test_parquet_footer_rejects_rows_before_table_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_path, manifest_path, mapping_path = _write_upload_bundle(tmp_path)
    parquet_path = tmp_path / "features.parquet"
    pd.read_csv(data_path).to_parquet(parquet_path, index=False)

    def forbidden_read(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("table materialization must not happen")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)

    with pytest.raises(UploadLimitError, match="row count"):
        load_governed_feature_upload(
            parquet_path,
            manifest_path=manifest_path,
            mapping_path=mapping_path,
            limits=FeatureUploadLimits(max_rows=10),
        )


@pytest.mark.parametrize(
    ("limits", "message"),
    (
        (FeatureUploadLimits(max_columns=1), "column count"),
        (FeatureUploadLimits(max_uncompressed_bytes=1), "uncompressed size"),
    ),
)
def test_parquet_footer_enforces_shape_and_expansion_budgets_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limits: FeatureUploadLimits,
    message: str,
) -> None:
    data_path, manifest_path, mapping_path = _write_upload_bundle(tmp_path)
    parquet_path = tmp_path / "features.parquet"
    pd.read_csv(data_path).to_parquet(parquet_path, index=False)

    def forbidden_read(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("table materialization must not happen")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)

    with pytest.raises(UploadLimitError, match=message):
        load_governed_feature_upload(
            parquet_path,
            manifest_path=manifest_path,
            mapping_path=mapping_path,
            limits=limits,
        )


def test_audit_rejects_file_byte_limit_before_parsing(tmp_path: Path) -> None:
    paths = _write_upload_bundle(tmp_path)

    completed = _run("audit", *paths, "--max-file-bytes", "10")

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["error_type"] == "UploadLimitError"
    assert payload["message"] == "data file size exceeds max_file_bytes"


def test_audit_rejects_manifest_feature_limit_before_data_loading(
    tmp_path: Path,
) -> None:
    data_path, manifest_path, mapping_path = _write_upload_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    second = dict(manifest["features"][0])
    second["name"] = "signal_2"
    manifest["features"].append(second)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    completed = _run(
        "audit",
        data_path,
        manifest_path,
        mapping_path,
        "--max-features",
        "1",
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["error_type"] == "UploadLimitError"
    assert payload["message"] == "feature count exceeds max_features"


def test_audit_accepts_parquet_through_the_same_contract(tmp_path: Path) -> None:
    data_path, manifest_path, mapping_path = _write_upload_bundle(tmp_path)
    parquet_path = tmp_path / "features.parquet"
    pd.read_csv(data_path).to_parquet(parquet_path, index=False)

    completed = _run("audit", parquet_path, manifest_path, mapping_path)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["input"]["file_name"] == "features.parquet"
    assert payload["dataset"]["observations"] == 36


def test_audit_rejects_feature_available_after_decision(tmp_path: Path) -> None:
    paths = _write_upload_bundle(tmp_path, late_availability=True)

    completed = _run("audit", *paths)

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["error_type"] == "PointInTimeValidationError"
    assert "availability" in payload["message"].lower()


def test_evaluate_runs_all_channels_after_audit(tmp_path: Path) -> None:
    paths = _write_upload_bundle(tmp_path)

    completed = _run(
        "evaluate",
        *paths,
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
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "success"
    assert payload["stage"] == "evaluate"
    assert [channel["name"] for channel in payload["report"]["channels"]] == [
        "purged-kfold",
        "cpcv",
        "causal-walk-forward",
    ]
    assert len(payload["report"]["channels"][1]["views"]["native"]["paths"]) == 2
    assert payload["evaluation"]["cpcv_combinations"] == comb(3, 2)
    assert "predictions" not in completed.stdout


def test_evaluate_rejects_excessive_combination_geometry_before_fitting(
    tmp_path: Path,
) -> None:
    paths = _write_upload_bundle(tmp_path)

    completed = _run(
        "evaluate",
        *paths,
        "--n-groups",
        "6",
        "--n-test-groups",
        "3",
        "--max-combinations",
        "10",
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["error_type"] == "UploadLimitError"
    assert payload["message"] == "CPCV combination count exceeds max_combinations"
    assert payload["stage"] == "evaluate"


@pytest.mark.parametrize(
    ("example", "expected_code"),
    (("raw", 0), ("stationary", 0), ("intentional-leak", 2)),
)
def test_checked_in_upload_examples_demonstrate_acceptance_boundary(
    example: str, expected_code: int
) -> None:
    directory = ROOT / "config" / "feature-upload" / "examples" / example

    completed = _run(
        "audit",
        directory / "features.csv",
        directory / "manifest.json",
        directory / "mapping.json",
    )

    assert completed.returncode == expected_code, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == ("success" if expected_code == 0 else "rejected")
