# FEATURE-001 — Immutable feature manifest and governed dataset

Status: resolved

- [x] Feature definition and manifest public values.
- [x] Per-feature availability and source-bundle binding.
- [x] Uploaded lifecycle/target/revision fail-closed rules.
- [x] Redacted deterministic governance receipt.
- [x] Manifest digest bound to dataset and OOS evidence.

## Answer

Implemented through `FeatureDefinition`, `FeatureManifest`, and
`govern_feature_dataset`; public-seam governance tests pass.
