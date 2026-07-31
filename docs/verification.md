# Verification

## Slice 1 local acceptance — 2026-07-31

Observed on Windows with Python 3.11.9, NumPy 2.4.6, pandas 3.0.3, Hypothesis 6.164.0,
pytest 9.1.1, and Ruff 0.15.21.

| Command / canary | Observed result |
|---|---|
| `python -m pip install -e ".[dev,pandas]"` | editable wheel built and installed successfully |
| `python tasks/preflight.py` | success; no contract errors |
| `python tasks/test.py` | success; includes pytest, Ruff check, and full-repo format check |
| `python -m pytest -q` | 41 passed |
| `python -m ruff check .` | all checks passed |
| `python -m ruff format --check .` | all Python files already formatted |
| core import with pandas blocked | passed in `test_core_import_does_not_require_pandas` |
| installed core and pandas-adapter imports outside the repository | version/API imports succeeded |
| `python -m pip check` | no broken requirements found |

This is local implementation evidence only. It does not establish cross-platform,
performance, deployment, CPCV, Causal Walk-Forward, or Untouched Holdout evidence.
