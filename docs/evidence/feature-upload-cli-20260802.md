# Governed local feature-upload evidence — 2026-08-02

## Delivery identity

- Version: 0.5.1
- Prompt: `.scratch/feature-upload-cli-v1/prompt.md`
- Specification: `.scratch/feature-upload-cli-v1/spec.md`
- Tickets: UPLOAD-001 through UPLOAD-004
- Public operator seam: `scripts/audit_feature_upload.py`
- Portable contracts: `config/feature-upload/*.schema.json`

## Implemented behavior

The local CSV/Parquet adapter parses exact version-1 manifest and mapping documents,
enforces file/row/feature ceilings, maps every temporal and feature field explicitly,
and delegates to canonical point-in-time and feature-lineage governance. `audit` stops
after emitting a redacted receipt. `evaluate` first passes that same gate, checks CPCV
combination geometry, and then produces Purged K-Fold, CPCV, and causal Walk-Forward
evidence using a deterministic fresh fold-local ridge estimator.

The high-level effectiveness seam now carries ordered trusted-code transformer factories
and ordered `TransformerSpec` identities through all three channels. Spec digests bind
the report identity. The file contract does not accept executable code.

## Example canaries

| Bundle | Observed status | Evidence |
|---|---|---|
| raw | accepted, exit 0 | 9 observations, 3 assets, 3 sessions; dataset digest `3967842bcebe35cccdc2c44c69fca97811b25fe6eefb9880b1080904b368b327` |
| stationary | accepted, exit 0 | 9 observations, 3 assets, 3 sessions; dataset digest `5a8121066291964f738de678b48151ebaf7834db9ead97a607bec70552f5762e` |
| intentional-leak | rejected, exit 2 | `DatasetValidationError`: target-derived uploaded feature |

These examples prove that stationarity is not an admission gate and that an explicitly
declared target dependency fails closed. They are audit fixtures, not model-effectiveness
or performance benchmarks.

## Verification receipt

| Check | Observed result |
|---|---|
| focused CLI/effectiveness suite | 15 passed |
| project preflight | success |
| complete project gate | success |
| complete pytest suite | 107 passed |
| strict mypy across source/tests/tasks/scripts | 41 source files; no issues |
| Ruff lint / format | all checks passed; 41 files formatted |
| dependency check | no broken requirements |
| package build | 0.5.1 sdist and universal wheel built |
| Twine | both artifacts passed |
| direct wheel canary | version 0.5.1, public limit error, upload adapter, and 10,000 default combination ceiling imported |

## Claim boundary

The receipt proves deterministic behavior against supplied files and declarations. It
cannot reconstruct feature formulas, authenticate vendor vintages, prove availability
timestamps truthful, establish profitability, replace untouched holdout evidence, or
authorize GitHub publication, remote CI, merging, or deployment.
