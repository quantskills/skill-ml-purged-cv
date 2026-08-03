"""Compatibility wrapper for the installed governed-upload CLI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from purged_kfold_validation.cli import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
