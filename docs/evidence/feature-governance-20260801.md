# Arbitrary-feature governance acceptance — 2026-08-01

## Outcome

Version 0.5.0 adds a bounded, evidence-bearing upload contract for arbitrary finite
numeric features. It does not require raw or stationary inputs and does not alter
Purged K-Fold, Embargo, CPCV, Walk-Forward, or financial metric semantics.

Implemented public seams:

- `FeatureDefinition`, `FeatureManifest`, and `FeatureComputationScope`;
- `govern_feature_dataset` and redacted `FeatureGovernanceReceipt`;
- `governed_validation_dataset_from_pandas` for explicit uploaded columns;
- ordered `TransformerSpec` values paired one-to-one with fold-local factories.

The manifest digest is bound into the governed Validation Dataset and every OOS
observation. Ordered transformer digests are bound into run identity and OOS evidence.

## Fail-closed coverage

Observed tests reject:

- shared row-level rather than per-feature availability for governed uploads;
- availability later than Decision Time;
- duplicate feature names and PIT source-bundle mismatch;
- latest-revision, target-derived, and globally precomputed stateful features;
- mutable nested manifest parameters;
- pandas feature mappings whose names/order differ from the manifest;
- transformer factories without exactly one ordered versioned spec.

Existing fresh-object, train-only transformation, full-run abort, PIT, CPCV, and
pandas-optional core canaries remain green.

## Research receipt

Agent Reach Exa search was effective through `mcporter/exa`; result content hash:
`0821551def42a939f6e1d7255b0c0835212b700f086922b1fe2808ad8469aff4`.
Its GitHub backend was unavailable because `gh` was not installed and was not counted
as attempted/effective coverage. Official Feast, Databricks, scikit-learn, and
OpenLineage documentation was deep-read separately. Full routing and source notes are
in `.scratch/feature-governance-v1/research.md`.

## Local acceptance

| Check | Observed result |
|---|---|
| Feature-governance public-seam tests | 9 passed |
| Complete pytest suite | 94 passed |
| Strict mypy | success; 36 source files |
| Ruff check / format check | all checks passed; 38 files formatted |
| Project preflight and task gate | success |
| `pip check` | no broken requirements |
| 0.5.0 wheel and source build | success |
| Twine | both artifacts passed |
| Isolated wheel import | version, core governance API, TransformerSpec, and pandas wrapper passed |

## Claim boundary

The validator checks supplied metadata consistency, per-feature timestamps, source and
code identities, declared lifecycle, and actual fold-local factory isolation. Numeric
values cannot prove that a caller reported its formula, source vintage, or availability
truthfully. This slice is not a stationarity test, feature-quality score, Feature Store,
training/serving-parity system, Alpha result, or deployment approval.
