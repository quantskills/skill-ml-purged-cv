# FEATURE-003 — Fold-local transformer lineage

Status: resolved
Blocked by: FEATURE-001

- [x] Immutable versioned TransformerSpec.
- [x] One-to-one ordered factory/spec binding.
- [x] Transformer digests bound into run and OOS evidence.
- [x] Existing fresh-object and train-only guarantees retained.

## Answer

Implemented in the canonical evaluator; identity and failure-path regression tests pass.
