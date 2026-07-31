"""Generated cross-platform language verification."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def required(command: str) -> str:
    found = shutil.which(command)
    if found is None:
        raise RuntimeError(f"required executable not found: {command}")
    return found


def run(argv: list[str], failures: list[str]) -> None:
    try:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            shell=False,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        failures.append(f"{argv[0]} failed to start: {exc}")
        return
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        failures.append(f"{' '.join(argv)} failed ({result.returncode}): {detail}")


def language_check(failures: list[str]) -> None:
    import compileall

    if not compileall.compile_dir(ROOT / "src", quiet=1):
        failures.append("Python source compilation failed")
    sys.path.insert(0, str(ROOT / "src"))
    try:
        __import__("purged_kfold_validation")
    except Exception as exc:
        failures.append(f"Python package import failed: {exc}")
    run([sys.executable, "-m", "pytest", "-q"], failures)
    run([sys.executable, "-m", "mypy", "src", "tests", "tasks"], failures)
    run([sys.executable, "-m", "ruff", "check", "."], failures)
    run([sys.executable, "-m", "ruff", "format", "--check", "."], failures)


def main() -> int:
    failures: list[str] = []
    run([sys.executable, "tasks/preflight.py"], failures)
    try:
        language_check(failures)
    except RuntimeError as exc:
        failures.append(str(exc))
    for failure in failures:
        print(failure, file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
