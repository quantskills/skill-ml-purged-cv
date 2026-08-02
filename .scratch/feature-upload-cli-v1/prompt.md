# Implementation prompt: governed arbitrary-feature upload v0.5.1

Implement the final user-facing delivery layer for `purged-kfold-validation` without
changing the underlying Purged K-Fold, CPCV, or Walk-Forward mathematics.

## Objective

Allow a user to submit a local CSV or Parquet matrix containing raw, stationary, or
otherwise precomputed numeric features, audit its availability and lineage declarations,
and, only after audit succeeds, evaluate it through Purged K-Fold, CPCV, and causal
Walk-Forward using the same deterministic fold-local ridge baseline.

## Required public seams

1. `python scripts/audit_feature_upload.py audit --data ... --manifest ... --mapping ...`
2. `python scripts/audit_feature_upload.py evaluate --data ... --manifest ... --mapping ...`
3. Versioned JSON Schemas at `config/feature-upload/feature-manifest.schema.json` and
   `config/feature-upload/upload-mapping.schema.json`.
4. Three non-secret example bundles: `raw`, `stationary`, and `intentional-leak`.
5. `run_cpcv_effectiveness_comparison()` accepts ordered transformer factories and one
   ordered `TransformerSpec` per factory, and binds those spec digests into the report.

## Safety and contract rules

- Accept only local `.csv` and `.parquet` data files; never execute uploaded code.
- Uploaded feature values must be finite, precomputed-stateless, target-independent,
  point-in-time values with one explicit availability timestamp per feature and row.
- Do not infer column roles, timestamps, feature formulas, stationarity, or provenance.
- `audit` validates and emits a redacted receipt but performs no model fitting.
- `evaluate` runs only after the exact same audit gate succeeds.
- Enforce configurable limits for file bytes, rows, features, and CPCV combinations.
- Emit deterministic JSON to stdout. Never emit source rows, feature values, targets,
  credentials, absolute paths, or tracebacks.
- Exit `0` on success and `2` for a governed rejection.
- Learned imputation, scaling, PCA, selection, and target encoding remain fold-local
  transformer work; the upload format never accepts executable transformer code.
- The output is diagnostic validation evidence, not profitability or deployment proof.

## Acceptance

Lock the public behavior with integration tests before implementation. Run focused tests,
`python tasks/preflight.py`, the complete `python tasks/test.py` suite, type checking,
lint, package build, Twine validation, and isolated wheel import. Record observed results
and close every local ticket only after its acceptance evidence exists.
