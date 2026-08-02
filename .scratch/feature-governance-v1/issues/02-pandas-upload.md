# FEATURE-002 — Explicit pandas arbitrary-feature upload boundary

Status: resolved
Blocked by: FEATURE-001

- [x] Existing explicit mapping reused without inference.
- [x] Ordered feature/availability columns matched to manifest.
- [x] Governed upload returned through one public seam.
- [x] Core import remains pandas-optional.

## Answer

Implemented by `governed_validation_dataset_from_pandas`; explicit upload and mismatch
tests pass.
