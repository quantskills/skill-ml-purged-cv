# Implementation prompt — arbitrary-feature availability and lineage governance

Implement a bounded feature-governance slice for `purged-kfold-validation` without
changing Purged K-Fold, Embargo, CPCV combination/path, Causal Walk-Forward, or metric
semantics.

## Objective

Allow callers to submit arbitrary finite numeric feature columns while making the
feature names, source lineage, point-in-time availability, revision policy,
transformation identity, lookback, target dependency, and computation lifecycle
explicit and evidence-bearing. Fail closed when an uploaded feature cannot legitimately
exist as one precomputed matrix before validation.

## Confirmed public seams

- `FeatureDefinition`, `FeatureManifest`, and `FeatureComputationScope`
- `govern_feature_dataset(...) -> GovernedFeatureDataset`
- `governed_validation_dataset_from_pandas(...) -> GovernedFeatureDataset`
- `TransformerSpec` bound one-to-one to fold-local transformer factories

Tests exercise only these public seams and `LeakageSafeEvaluator.evaluate()`.

## Feature definition contract

Every uploaded feature declares:

- a unique non-empty semantic name;
- one source dataset identity and at least one source field;
- a source content/vintage digest;
- transformation name, version, immutable parameters, and code digest;
- lookback measured in Trading Sessions, at least one;
- source revision policy, which must be `point-in-time` for formal governance;
- whether the feature depends on target values;
- computation scope: `precomputed-stateless`, `precomputed-stateful`, or `fold-local`.

The manifest preserves feature order and declares a source-bundle digest matching the
Validation Dataset's PIT Snapshot source digest. All definitions and the manifest have
deterministic digests.

## Uploaded-feature rules

- Feature count and order must match the uploaded feature matrix.
- Feature Availability must be a two-dimensional row-by-feature timestamp matrix.
- Every availability cell must be no later than its row's Decision Time.
- Uploaded features must be `precomputed-stateless`, target-independent, and use
  point-in-time source revisions.
- Globally fitted scaling, imputation, PCA, selection, encoding, or target-derived
  values fail closed. Callers must upload their raw/stateless inputs and express learned
  steps as fold-local transformer factories instead.
- The governed dataset and every OOS observation bind the feature manifest digest.
- The receipt reports rows, features, availability cells, maximum lookback, source and
  manifest digests without exposing feature values.

## Fold-local transformer lineage

- Add immutable `TransformerSpec` values with name, version, code digest, parameters,
  target-dependency declaration, and fixed `fold-local` fit scope.
- A `LeakageSafeEvaluator` with transformer factories requires exactly one ordered spec
  per factory. Specs without factories, factories without specs, reused transformer
  objects, or non-fold-local scopes fail closed.
- Transformer spec digests participate in run identity and OOS evidence so two
  materially different preprocessing pipelines cannot share an evidence identity.

## Pandas upload boundary

Wrap the existing explicit pandas adapter. Do not infer feature columns, availability
columns, session axes, formulae, source lineage, or hidden DataFrame attributes. Require
the existing mapping, PIT Snapshot, and a manifest; return the governed dataset and
receipt.

## Claim boundary

This proves consistency of supplied metadata, timestamps, lifecycle declarations,
digests, and fold-local execution. It cannot prove that a user's declared formula,
availability time, source vintage, or lineage is truthful merely by inspecting numeric
values. It does not judge stationarity, Alpha quality, profitability, or deployment
readiness.

## Acceptance

1. An honest two-feature upload produces a deterministic immutable receipt and binds
   the manifest digest into dataset/OOS evidence.
2. Missing/shared rather than per-feature availability fails closed.
3. A feature available after Decision Time fails before evaluation.
4. Target-derived, latest-revision, globally precomputed stateful, duplicate, malformed,
   or source-digest-mismatched definitions fail closed.
5. Learned transformations require fresh fold-local factories and matching specs; their
   spec digests alter run identity.
6. Existing tests remain green, the core import still does not require pandas, and
   package preflight, tests, strict typing, Ruff, build, Twine, and isolated wheel import
   pass.
