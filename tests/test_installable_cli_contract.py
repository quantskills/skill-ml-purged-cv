from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_distribution_declares_console_upload_extra_and_package_resources() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    assert project["project"]["version"] == "0.6.1"
    assert project["project"]["dependencies"] == ["numpy>=1.26"]
    assert project["project"]["scripts"] == {
        "purged-cv-upload": "purged_kfold_validation.cli:main"
    }
    assert project["project"]["optional-dependencies"]["upload"] == [
        "pandas>=2.1",
        "pyarrow>=15",
    ]
    assert project["tool"]["setuptools"]["package-data"]["purged_kfold_validation"] == [
        "resources/feature_upload/*.json",
        "resources/feature_upload/examples/*/*.csv",
        "resources/feature_upload/examples/*/*.json",
    ]


def test_schema_discovery_does_not_import_optional_pandas() -> None:
    code = """
import builtins
import sys

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "pandas" or name.startswith("pandas."):
        raise ImportError("pandas import blocked by canary")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from purged_kfold_validation.cli import main
sys.argv = ["purged-cv-upload", "schema", "--kind", "manifest"]
raise SystemExit(main())
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '"title": "Governed feature manifest v1"' in completed.stdout


def test_audit_without_upload_extra_fails_with_safe_install_guidance() -> None:
    code = """
import builtins
import sys

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "pandas" or name.startswith("pandas."):
        raise ImportError("pandas import blocked by canary")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from purged_kfold_validation.cli import main
sys.argv = [
    "purged-cv-upload", "audit", "--data", "features.csv",
    "--manifest", "manifest.json", "--mapping", "mapping.json"
]
raise SystemExit(main())
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    assert completed.stdout == (
        '{"error_type": "OptionalDependencyError", '
        '"message": "audit/evaluate require the upload extra: '
        'pip install purged-kfold-validation[upload]", '
        '"stage": "audit", "status": "rejected"}\n'
    )
