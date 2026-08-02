"""Compatibility wrapper for the installed governed-upload CLI."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from purged_kfold_validation.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
