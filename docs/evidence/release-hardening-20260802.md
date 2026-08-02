# Release hardening and governed Holdout evidence — 2026-08-02

## Scope

Version 0.6.0 closes three local pre-release gaps: read-before-allocate upload budgets,
a frozen one-attempt final Holdout boundary, and fixed-model ranking stability across
declared regimes. It does not claim remote CI, external publication, profitability, or
that Holdout data was never accessed outside the governed store.

## Observed acceptance

| Check | Observed result |
|---|---|
| focused Holdout/ranking/upload tests | 34 passed |
| complete pytest suite | 132 passed |
| `python tasks/preflight.py` | success |
| `python tasks/test.py` | success |
| strict mypy | success; 48 source files |
| Ruff check / format check | all checks passed; 48 files formatted |
| `python -m pip check` | no broken requirements |
| `python -m build --no-isolation` | 0.6.0 sdist and universal wheel built |
| Twine check | both 0.6.0 artifacts passed |
| isolated wheel import | version 0.6.0 and Holdout/ranking public APIs imported from the venv |
| installed module schema canary | packaged manifest schema printed successfully |

The first local pip attempt printed a successful install before its outer wrapper timed
out. A final `--force-reinstall --no-deps` against the rebuilt wheel then exited `0`;
subsequent imports resolved from the isolated environment's `site-packages`.

## Safety findings

- CSV row ceilings now bound parsing to `max_rows + 1`; mapped columns alone are read.
- Parquet footer row, leaf-column, and declared uncompressed-byte budgets are checked
  before table materialization.
- The Holdout digest is claimed with exclusive file creation before any factory runs.
- Training Information Intervals must end strictly before the first Holdout interval;
  a later label horizon cannot cross the frozen boundary.
- A failed evaluation remains consumed and cannot be retried under a new protocol ID.
- Transformers and estimators fit on the frozen training dataset only.
- Durable receipts contain digests, metric summaries, and time, but not raw values or
  predictions.
- Nested search-policy configuration is deeply frozen into protocol identity.
- Rank reversal is surfaced as instability; it does not authorize Holdout reuse.

## Remaining external gates

- The worktree is not yet frozen into an immutable clean commit.
- No Git remote is configured, so the declared Linux/Windows/macOS CI matrix has not
  run remotely.
- No authenticated release target, rollback owner, or observation window is configured.
- The current contracts cannot establish the external truth of supplied timestamps,
  revisions, corporate-action treatment, or feature formula declarations.
