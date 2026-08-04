# Interface contract

This document describes the implemented Purged K-Fold model-selection, CPCV robustness,
Causal Walk-Forward, governed upload, ranking-stability, and final Holdout contracts.

## Slice 1 public contract

The supported runtime is Python 3.11 or newer with NumPy. Public core APIs are exported
from `purged_kfold_validation`; the optional pandas boundary is imported explicitly
from `purged_kfold_validation.adapters.pandas`.

### Diagnostic seam

`PurgedKFold.plan(ValidationDataset) -> SplitPlan` returns every requested candidate.
Valid candidates are immutable `FoldAssignment` values; invalid candidates retain typed
reasons and exclusion counts. Calling `SplitPlan.require_assignments()` fails when any
candidate is invalid. Planning never creates a score and does not require formal PIT
eligibility, so incomplete evidence can still be diagnosed.

`PurgedKFold` creates deterministic contiguous test-session blocks, has no shuffled
mode, Purges inclusive Information Interval overlap, and applies Embargo in Session Axis
positions after each TestBlock. All rows in one session are assigned together.

### Formal evaluation seam

`LeakageSafeEvaluator.evaluate(ValidationDataset) -> EvaluationResult` first requires
complete Decision Time, Feature Availability, and point-in-time `PITSnapshot` evidence.
It then requires a wholly valid SplitPlan, creates fresh transformers and an estimator
from factories for every fold, fits only on that fold's training side, and returns an
immutable raw `OOSLedger` plus named/versioned `DerivedMetric` projections.

The canonical evaluator has no pre-fitted-object parameter. Factory creation, fit,
transform, predict, output-shape, and metric failures abort the entire call. No skipped
fold, partial ledger, warning-only score, or NaN substitute is valid evidence.

### Canonical values and provenance

- `InformationInterval` uses inclusive boundaries.
- `ValidationDataset` owns stable sample identity, Session Axis membership, optional
  asset identity, features, targets, temporal feature evidence, and source provenance.
- The only Slice 1 `MissingValuePolicy` is fail-closed `reject`; features and targets
  must be finite numeric arrays.
- Public arrays are read-only copies and domain results are immutable.
- Dataset, split specification, fold, plan, model, metric, run, ledger, and PIT source
  state have deterministic digests or identities where applicable.
- `OOSObservation` retains sample, session, optional asset, fold, split, model, dataset,
  split-specification, PIT source, run, and Evidence Channel identity.
- Each `DerivedMetric` reports observation count/coverage and fold count/coverage.
- Public failures derive from `ValidationError`; dataset, temporal/PIT, split, factory,
  execution, shape, metric, and adapter failures remain distinguishable.

### Optional pandas boundary

`PandasField`, `PandasDatasetMapping`, and `validation_dataset_from_pandas` require every
column or index level to be mapped explicitly. They do not infer names, horizons, index
semantics, snapshot provenance, or hidden DataFrame attrs. The adapter normalizes one
consistent timezone and delegates all leakage semantics to the canonical core. Importing
the core does not import or require pandas.

## Evidence boundary and non-capabilities

Purged K-Fold results have the `model-selection` Evidence Channel. Two-sided training
is permitted, so those results are not causal deployment evidence.

`CausalWalkForward.plan(ValidationDataset) -> SplitPlan` creates deterministic past-only
test blocks. It applies inclusive Information Interval Purge, removes any candidate
whose information is not strictly earlier than the test block, optionally applies a
session-based Pre-Test Gap, and supports expanding or bounded sliding training history.
Assignments and evaluator results retain `causal-walk-forward` identity.

`validation_dataset_from_pandaai_daily` is an explicit optional pandas boundary for
already-loaded daily rows. `run_validation_benchmark` reports structural overlap,
coverage, and metrics for two diagnostic baselines plus the two safe channels. The CLI
loads local parquet only and emits redacted deterministic JSON.

### CPCV robustness seam

`CombinatorialPurgedCV.plan(ValidationDataset) -> SplitPlan` partitions active Trading
Sessions into chronological groups and enumerates every bounded lexicographic `N,k`
test-group combination. Each assignment reuses exact inclusive Purge, applies Embargo
after every contiguous selected test region, preserves panel-session grouping, and
carries the separate `cpcv-robustness` Evidence Channel.

`SplitPlan.path_decomposition` proves the deterministic assignment of every
combination/group occurrence to exactly `C(N-1,k-1)` complete paths. Every path contains
every chronological group once, and selected groups from one combination occupy
distinct paths. Excessive combination counts, invalid folds, incomplete occurrence
coverage, or incomplete evaluation paths fail closed.

The evaluator retains repeated CPCV predictions with combination, group, and path
identity. Derived metrics expose per-combination and per-path projections; observation
coverage counts unique sample identities. CPCV is two-sided robustness evidence, not
causal deployment evidence.

Nested HPO, general persistence writers, parallelism, scikit-learn compatibility, and
external metadata truth verification are not implemented.

### Governed final Holdout seam

`EvaluationProtocol.freeze` binds the training and Holdout dataset digests, fixed model,
ordered transformer specs, metrics, completed search policy, and split-specification
identity. The Holdout must be strictly later than training, every training Information
Interval must end before the first Holdout interval, and sample identities must not
overlap. Supplied components must exactly match the frozen protocol at execution.

`LocalHoldoutStore.evaluate_once` creates an exclusive claim keyed by Holdout dataset
digest before any transformer or estimator is created. Both successful and failed
attempts consume that identity, and a new protocol ID does not permit reuse. Every
transformer and the estimator fit only on the frozen training dataset. The returned
OOS evidence has the separate `holdout-confirmation` Evidence Channel.

Only a redacted `HoldoutReceipt` is persisted: protocol, Holdout, run and ledger
digests, metric values, and evaluation time. Raw rows, features, targets, and
predictions are not written. The contract governs access through this interface; it
cannot detect prior or parallel out-of-band inspection of the Holdout files.

### Cross-regime ranking seam

`assess_model_ranking_stability` requires at least two regimes and the exact same set
of at least two fixed models in each regime. It rejects non-finite or incomparable
scores and returns median/worst ranks, first-place counts, and every pairwise Spearman
correlation under a frozen threshold. It is robustness evidence, not Holdout evidence
or a profitability claim.

The approved requirements remain in `.scratch/purged-kfold-v1/spec.md`; their observed
implementation evidence is mapped in `docs/requirements-traceability.md`.

## Arbitrary-feature governance seam

`FeatureDefinition` records one uploaded feature's semantic name, source dataset and
fields, source/code digests, transformation/version/parameters, lookback sessions,
revision policy, target dependency, and computation scope. `FeatureManifest` preserves
the exact feature order and binds it to the PIT source-bundle digest.

`govern_feature_dataset(ValidationDataset, FeatureManifest)` requires a two-dimensional
per-feature availability matrix, source-bundle equality with `PITSnapshot`, feature
count equality, point-in-time revisions, target independence, and
`precomputed-stateless` scope. It returns a `GovernedFeatureDataset` and redacted
`FeatureGovernanceReceipt`; the manifest digest participates in the governed dataset
digest and every OOS observation.

`governed_validation_dataset_from_pandas` wraps the existing explicit pandas adapter.
Mapped feature column names and order must exactly match manifest names. It never infers
feature, availability, formula, session-axis, or provenance fields.

Any learned transformation must remain a fresh fold-local transformer factory.
`LeakageSafeEvaluator` requires one ordered `TransformerSpec` per transformer factory;
the spec digests participate in run identity and every OOS observation. A spec is an
identity/provenance declaration, not permission to share fitted state.

The library verifies metadata consistency, timestamp ordering, digests, declared
lifecycle, and actual fold-local factory use. It cannot infer stationarity, reconstruct
feature formulae, or prove that uploaded availability/source declarations are truthful
from values alone.

### Local feature-upload CLI

`scripts/audit_feature_upload.py` is the repository operator seam for a local `.csv` or
`.parquet` feature matrix. It requires a closed version-1 Feature Manifest and Upload
Mapping; schemas and examples live under `config/feature-upload/`. Unknown fields,
inferred roles, missing per-feature availability, unsupported suffixes, and inconsistent
feature order are rejected.

The `audit` subcommand performs bounded loading and produces dataset/manifest digests,
counts, limits, and a `FeatureGovernanceReceipt` without model fitting. The `evaluate`
subcommand passes the same audit gate, bounds `C(n_groups, n_test_groups)`, and then emits
the canonical Purged K-Fold, CPCV, and causal Walk-Forward effectiveness report using a
fresh fold-local ridge baseline. Success exits `0`; governed rejection exits `2` with
redacted JSON. Neither output includes source rows, feature values, targets, absolute
paths, or predictions.

CSV loading is capped with a bounded `max_rows + 1` parse rather than a full-file read.
Parquet footer metadata is inspected before table materialization to enforce row,
column, and declared uncompressed-byte ceilings. Mapped columns alone are materialized;
unsupported, missing, or over-budget input fails closed.

`run_cpcv_effectiveness_comparison` also accepts ordered `transformer_factories` and
`transformer_specs`. Each channel uses the same factory/spec sequence, while the report
binds ordered spec digests into its own identity. This supports trusted application code
that constructs fold-local preprocessing; the file upload contract itself never accepts
or executes such code.

### Installed command and resources

The wheel declares `purged-cv-upload = purged_kfold_validation.cli:main` and supports the
equivalent `python -m purged_kfold_validation` entry point. The repository script is a
compatibility wrapper around the same function, so audit/evaluate JSON and exit codes do
not diverge between installation styles.

`schema --kind manifest|mapping` prints the installed version-1 schema. `example --name
raw|stationary|intentional-leak --output-dir PATH` materializes the three fixed contract
files from package data. It checks every target before writing, refuses overwrite, and
does not expose an arbitrary package path. Schema/example discovery does not import
pandas. Audit/evaluate require the `upload` optional dependency and otherwise return a
redacted `OptionalDependencyError` with the installation command.

### Agent-neutral Skill seam

The root `SKILL.md` is the portable discovery and execution contract. Agent-specific UI
metadata contains no validation behavior. `purged-cv-skill run --request PATH` accepts one
closed, size-bounded version-1 JSON request, resolves relative inputs against that request,
and delegates to the installed `purged_kfold_validation` module in a bounded subprocess.

The command emits exactly one version-1 JSON envelope. It binds a canonical request digest,
engine identity, unchanged `authoritative_cli_result`, warnings, and typed errors without
exposing absolute input paths, rows, features, targets, or predictions. Unknown fields,
action-incompatible options, invalid ranges, timeout, unexpected stderr, and non-JSON engine
output fail closed with exit code `2`.

`purged-cv-skill demo` is the one-command installation smoke test. `example --output-dir`
atomically materializes a fixed raw feature bundle plus its single `request.json`, refusing an
existing target. `schema --kind request|result` prints the exact packaged machine contracts.
The wrapper does not implement Purge, Embargo, CPCV, Walk-Forward, feature governance, or
Holdout behavior; the maintained library remains the sole algorithm owner.

### Time-series strategy selection seam

`StrategyReturnMatrix` is the generic seam between caller-owned strategy generation and
selection-overfitting evidence. It contains strictly increasing Trading Sessions, unique
Candidate Strategy identities, and aligned finite gross-return, net-return, and turnover
matrices. `analyze_strategy_return_matrix` hides CSCV/PBO enumeration, DSR correction,
CPCV training-side selection/path reconstruction, and causal expanding-window selection
behind one immutable result interface.

`DensePricePanel` and `build_tsmom_return_matrix` form a separate built-in adapter at this
seam. The adapter uses each asset's own lagged price direction, lagged realized volatility,
frozen prior weights, and an offline turnover-cost rule. It does not perform cross-sectional
ranking, continuous-contract construction, order execution, capacity analysis, or external
PIT verification.

`purged-cv-strategy run --request PATH` accepts a closed version-1 request and an `.npz`
loaded with `allow_pickle=False`. `analyze-return-matrix` requires exactly `sessions`,
`candidate_ids`, `gross_returns`, `net_returns`, and `turnover`; `benchmark-tsmom` requires
exactly `sessions`, `asset_ids`, `signal_prices`, and `tradable_returns`. The file is capped
at 512 MiB, the request at 1 MiB, and unknown fields or archive members fail closed.

Cost scenarios remain separate reports and do not increase PBO/DSR trial counts. PBO is a
candidate-rank failure probability rather than a p-value. Every report warns that structural
or selection evidence is not profitability, execution, deployment, or investment advice.

### Strategy acceptance decision seam

`assess_time_series_benchmark` is the only public boundary that turns a completed
`TimeSeriesBenchmarkReport` into gate decisions. It receives an immutable,
digest-bound `StrategyAcceptancePolicy` and redacted `StrategyHoldoutEvidence`; the
benchmark engine never changes thresholds in response to observed results.

The decision reports `validation_tool_status`, `research_gate_status`, and
`production_gate_status` separately. Missing registered primary or stress cost scenarios
fail closed. A failed Research Gate forces Production to FAIL. A passing Research Gate
with a required but unrun Holdout is INCONCLUSIVE; only eligible `UNTOUCHED_PASS`
evidence can complete that gate. Reused Holdout evidence is ineligible.

Each check retains its code, observed value, threshold, comparison, and cost scenario.
The DSR track-record gap uses the report's selected/benchmark Sharpe, sample moments,
annualization, and observation count. It is explicitly a constant-distribution
approximation, not a guarantee and not a replacement for untouched confirmation.

The `benchmark-tsmom` CLI appends this canonical decision under `report.acceptance`.
`analyze-return-matrix` remains a single-cost evidence report and does not fabricate a
multi-cost production decision.

### Trainable temporal-model comparison seam

`build_temporal_supervised_dataset` maps one governed `DensePricePanel` into Asset by
Decision Session samples. Each sample holds an ordered own-return lag sequence, a future
return label, per-lag Feature Availability, and one inclusive Information Interval from
the earliest lag dependency through label completion. Panel Session grouping prevents
assets from the same Session crossing validation sides.

`run_temporal_model_benchmark` accepts that dataset, fixed `TemporalModelCase` factories,
and one digest-bound comparison configuration. It runs unsafe shuffled Session K-Fold,
chronological no-purge, Purged K-Fold without Embargo, Purged K-Fold with Embargo, CPCV,
and causal Walk-Forward through the maintained evaluator. Model adapters never choose or
implement split logic; splitters never construct or tune models.

The registered adapters are fold-local standardized NumPy Ridge, fixed-parameter
LightGBM, and a small CPU PyTorch LSTM that reshapes ordered lag columns as a sequence.
LightGBM and PyTorch are optional dependencies and are never silently substituted.

Every safe channel must retain zero Information Interval overlap. The report keeps MSE,
fold/path distributions, coverage, exclusions, minimum training geometry, and unsafe
optimism gaps. Embargo can validly have zero incremental exclusions when the complete
Information Interval Purge already covers its zone; this is disclosed separately from
whether the Embargo policy executed. Structural PASS never authorizes production.

`purged-cv-strategy` exposes the same seam as action `benchmark-temporal-models` over the
existing dense-panel NPZ contract. It does not accept executable user model code.

### Temporal forward-evidence seam

`TemporalForwardProtocol` freezes the consumed development report/data digests, selected
model and temporal dataset spec, development label boundary, strictly future start,
label horizon, maturity requirements, and metric checks. Its identity cannot be rebound
after predictions exist.

`LocalTemporalForwardStore` uses exclusive-create files for protocols, sample claims,
predictions, settlements, and report snapshots. A Prediction Receipt contains no target
or raw feature values and must be recorded before `label_available_at`. A Matured Label
Settlement must reference an existing prediction and can be created only after that
instant. Duplicate sample claims, duplicate settlements, orphan settlements, and digest
mismatches fail closed.

`ForwardEvidenceReport` exposes only counts, aggregate MSE/baseline MSE/mean per-session
Spearman IC, checks, status, and digests. `WAITING_FOR_FUTURE_DATA` means no eligible
forecast exists; `COLLECTING` means evidence is below a sufficiency gate;
`READY_FOR_REVIEW` and `FAIL` are emitted only after all sufficiency gates pass. The
report always returns `production_authorization=NOT_AUTHORIZED`.
It also returns `attestation_scope=LOCAL_APPEND_ONLY_NOT_EXTERNALLY_NOTARIZED` because
exclusive local files and caller-visible clocks are not an external timestamp or WORM
attestation. Production-grade non-repudiation requires an independently controlled sink.
