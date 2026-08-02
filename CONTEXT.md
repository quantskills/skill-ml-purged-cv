# Financial Validation Context

This context defines the language used to reason about leakage-safe financial time-series validation. It separates information-aware model selection from causal deployment evidence.

## Language

**Information Interval**:
The inclusive interval from the first information attributed to a sample through the final time needed to determine its label.
_Avoid_: Label window, horizon rows

**Trading Session**:
An ordered market-calendar unit used to group samples and measure session-based gaps across holidays and special trading days.
_Avoid_: Calendar day, row number

**Panel Session Group**:
All asset samples belonging to the same Trading Session and therefore assigned to the same validation side.
_Avoid_: Date rows, asset fold

**Session Axis**:
The authoritative ordered sequence of Trading Sessions against which sample membership, interval endpoints, and session-based distances are validated.
_Avoid_: Datetime index, row order

**Validation Dataset**:
The canonical collection of stable sample identities, Session Axis membership, Information Intervals, features, targets, and optional asset identities presented for validation.
_Avoid_: Input DataFrame, training frame

**Decision Time**:
The time at which a prediction represented by one sample is assumed to be made and against which feature availability is judged.
_Avoid_: Row timestamp, event start

**Feature Availability**:
The latest time at which all information represented by a feature value was actually available to the decision maker.
_Avoid_: Feature date, observation time

**Feature Definition**:
An immutable declaration of one uploaded feature's semantic name, ordered source
fields, source vintage, transformation/code identity, parameters, lookback, revision
policy, target dependency, and computation lifecycle.
_Avoid_: Column label, inferred formula

**Feature Manifest**:
The ordered collection of Feature Definitions bound to the PIT source bundle and then
to Validation Dataset and OOS evidence by a deterministic digest.
_Avoid_: Feature list, DataFrame schema

**Governed Feature Dataset**:
A Validation Dataset whose per-feature availability matrix and manifest have passed
the uploaded-feature lifecycle rules and produced a redacted governance receipt.
_Avoid_: Trusted values, leak-free by assertion

**Transformer Spec**:
The versioned code/parameter identity paired one-to-one with a fold-local transformer
factory so preprocessing state participates in the evaluation identity.
_Avoid_: Pipeline name, preprocessing comment

**Feature Upload Contract**:
The closed, versioned manifest and mapping documents that bind a local CSV/Parquet
matrix to column roles, Session Axis, PIT Snapshot, per-feature availability, lineage,
and bounded resource policy before evaluation.
_Avoid_: Automatic schema detection, arbitrary upload

**Installed Upload Command**:
The distribution-owned console/module boundary that exposes audit, evaluation, schema
discovery, and safe example materialization from the same packaged implementation.
_Avoid_: Repository script copy, unversioned helper

**PIT Snapshot**:
A point-in-time source snapshot that preserves the values and publication state available at its declared historical cutoff rather than later revisions.
_Avoid_: Historical export, latest data

**Pandas Adapter**:
A boundary translator that constructs a Validation Dataset from explicitly mapped pandas columns and index levels without inferring authoritative metadata.
_Avoid_: DataFrame-first API, automatic schema detection

**Purge**:
The exclusion of a training sample whose Information Interval overlaps a test Information Interval.
_Avoid_: Drop adjacent rows, trim

**Embargo**:
The post-test exclusion zone applied after each contiguous test block in two-sided validation.
_Avoid_: Walk-forward gap, purge window

**Pre-Test Gap**:
The exclusion zone immediately before a causal test block, used together with Purge when training is restricted to the past.
_Avoid_: Embargo

**Splitter**:
A component that produces validation assignments and their provenance without fitting transformations or estimators.
_Avoid_: Evaluator, backtest

**Fold Assignment**:
An immutable, evidence-bearing validation assignment containing stable sample identities, array positions, contiguous test blocks, exclusion summaries, and input/configuration provenance.
_Avoid_: Index tuple, fold indices

**Exclusion Summary**:
A compact account of how many candidate training samples were removed by Purge, Embargo, or Pre-Test Gap for one Fold Assignment.
_Avoid_: Dropped rows, filter count

**Exclusion Trace**:
An optional audit view that records the exclusion reason for each removed sample without changing the Fold Assignment.
_Avoid_: Debug log, warning list

**Split Plan**:
A read-only diagnostic projection of every requested Fold Assignment, including invalid candidates and the evidence explaining why they cannot be evaluated.
_Avoid_: Evaluation result, skipped folds

**Invalid Fold**:
A requested validation assignment that violates declared sample, session, coverage, or path-completeness constraints and therefore cannot contribute a score.
_Avoid_: Empty fold, warning-only fold

**Leakage-Safe Evaluator**:
The canonical high-level validation boundary that creates fresh fold-scoped transformations and estimators, executes Splitter assignments, and returns out-of-sample evidence.
_Avoid_: Splitter, global training loop

**Fold-Local**:
A learned operation whose fit state is derived only from the training side of one validation assignment.
_Avoid_: Precomputed, globally fitted

**Fold Factory**:
A provider that creates a new unfitted estimator or transformation for one validation assignment, preventing learned state from crossing fold boundaries.
_Avoid_: Shared model, fitted object, reusable instance

**OOS Ledger**:
The authoritative collection of path-scoped out-of-sample predictions, keyed by run, sample, split, combination, path, and model identity.
_Avoid_: Averaged predictions, score table

**Evidence Channel**:
A validation purpose whose observations and claims remain distinct from other purposes, such as model selection, CPCV robustness, causal walk-forward, or final holdout confirmation.
_Avoid_: Validation type, combined score

**Evaluation Protocol**:
An immutable declaration of frozen data boundaries, feature/model configuration, search policy, split specification, and metrics that authorizes final holdout evaluation.
_Avoid_: Experiment config, run parameters

**Holdout Receipt**:
An append-only record that binds one authorized holdout evaluation to its Evaluation Protocol, holdout dataset, result, and evaluation time.
_Avoid_: Score file, run log

**Reused Holdout**:
A holdout dataset observed before the current final evaluation claim and therefore ineligible to provide untouched confirmation, even under a new protocol identity.
_Avoid_: Refreshed holdout, second test run

**Derived Metric**:
A versioned calculation projected from an OOS Ledger for a declared split, path, or Evidence Channel.
_Avoid_: Stored truth, headline score

**CPCV Combination**:
One choice of test groups from the complete set of chronological groups in combinatorial purged cross-validation.
_Avoid_: CPCV path

**CPCV Path**:
A deterministic sequence of path-scoped out-of-sample predictions assembled from CPCV Combinations so that every chronological group appears exactly once in that path.
_Avoid_: Fold list, averaged predictions

**Path Decomposition**:
The deterministic assignment of every CPCV combination/test-group occurrence to complete CPCV Paths while preserving group coverage and combination identity.
_Avoid_: Path shuffle, fold concatenation

**Causal Walk-Forward**:
A validation sequence in which every training Information Interval precedes its test block and time advances without future training groups.
_Avoid_: Purged K-Fold, chronological K-Fold

**Untouched Holdout**:
A final evaluation interval excluded from model, feature, threshold, and hyperparameter decisions until the design is frozen.
_Avoid_: Validation fold, reusable test set

## Current five-year acceptance fact

The governed PandaData run covering 2021-06-18 through 2026-06-18 retained 88,417
eligible observations across 81 assets and 1,174 Trading Sessions. All three structural
channels reported zero retained Information Interval overlaps. A final 252-session
Holdout was consumed exactly once; its MSE is evidence about the frozen intercept-only
baseline, not a profitability or deployment claim. See
`docs/evidence/pandadata-five-year-release-gate-20260802.md`.
