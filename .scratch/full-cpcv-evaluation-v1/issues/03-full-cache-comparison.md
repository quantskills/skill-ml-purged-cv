# EVAL-001 — Full-cache CPCV effectiveness comparison

Status: resolved
Blocked by: DATA-001, METRIC-001

- [x] Same dataset, model, target, and definitions across three safe channels.
- [x] Independent zero-overlap audit.
- [x] Per-fold training sufficiency and group breadth.
- [x] Offline redacted deterministic CLI.
- [x] Approximately 175k-row local cache receipt or bounded failure receipt.
- [x] Complete package acceptance.

## Answer

The governed comparison completed in 732.6 seconds on 175,129 observations. All
channels retained zero interval overlaps; every fold/combination passed the frozen
training sufficiency gates. See `docs/evidence/full-cpcv-effectiveness-20260801.md`.
