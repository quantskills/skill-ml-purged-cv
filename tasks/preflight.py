"""Generated cross-platform scaffold preflight."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "AGENTS.md",
    "PROGRAM.md",
    "CONTEXT.md",
    "README.md",
    "project.yaml",
    "docs/WORK_ITEMS.md",
)
REQUIRED_DIRS = ("src", "scripts", "tasks", "tests", "config", "docs")
MANIFESTS = ("pyproject.toml", "package.json", "go.mod")
EXPECTED_MANIFEST = "pyproject.toml"


def main() -> int:
    errors = []
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            errors.append(f"missing or empty: {relative}")
    for relative in REQUIRED_DIRS:
        if not (ROOT / relative).is_dir():
            errors.append(f"missing directory: {relative}")
    present = [name for name in MANIFESTS if (ROOT / name).is_file()]
    expected = [] if EXPECTED_MANIFEST is None else [EXPECTED_MANIFEST]
    if present != expected:
        errors.append(
            f"language manifest contract failed: expected={expected}, actual={present}"
        )
    report = {
        "status": "success" if not errors else "error",
        "root": str(ROOT),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
