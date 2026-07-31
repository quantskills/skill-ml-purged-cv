# Purged K-Fold V1 vertical slice

Status: resolved
Owner: project owner
Evidence channel: model-selection

## Problem Statement

Financial time-series samples are not independent observations. A sample can use feature information from an earlier window and label information from a later interval, while adjacent samples, assets observed in the same Trading Session, revised source data, and learned preprocessing can share information across a nominal train/test boundary. Ordinary K-Fold and row-based chronological splitting therefore produce apparently separate folds while allowing future information or shared event information into training.

The user needs a reusable validation library that makes these leakage boundaries explicit and fail-closed. It must distinguish leakage-aware model-selection evidence from causal deployment evidence, preserve enough provenance to audit every fold, prevent learned state from crossing folds, and refuse to score data whose temporal or point-in-time metadata is incomplete. A correct list of train/test indices is necessary but insufficient: the complete evaluation path must remain Fold-Local and evidence-bearing.

The repository currently contains a verified project scaffold and accepted domain/architecture decisions, but no product implementation. The first deliverable must establish one end-to-end Purged K-Fold vertical slice before CPCV, Causal Walk-Forward, nested HPO, or governed holdout access are added.

## Solution

Provide a NumPy-based leakage-safe validation core with two public behavioral seams:

1. A diagnostic Split Plan seam that exposes deterministic candidate Fold Assignments, contiguous test blocks, exclusions, provenance, and Invalid Fold reasons without producing a score.
2. A canonical Leakage-Safe Evaluator seam that validates an explicit Validation Dataset, obtains formal Fold Assignments, creates new unfitted transformations and estimators for every fold through Fold Factories, executes out-of-sample predictions, and returns an authoritative in-memory OOS Ledger plus versioned Derived Metrics.

The Validation Dataset carries stable sample identity, Session Axis membership, Information Intervals, Decision Time, Feature Availability, PIT Snapshot provenance, features, targets, and optional asset identity. Purge removes candidate training samples whose Information Intervals overlap test information. Embargo removes candidates in the configured Trading Sessions after each contiguous test block. Panel Session Groups remain indivisible. Formal evaluation fails as a whole when any requested fold or execution step is invalid.

The result retains stable sample, split, dataset, model, metric, and configuration identities. Summaries are derived from raw OOS observations and never replace them as the source of truth. The first slice supports fixed model configurations and an explicit optional pandas adapter; later slices add Causal Walk-Forward, governed holdout access, deterministic CPCV Path Decomposition, nested HPO, ecosystem adapters, persistence, and bounded parallelism.

## User Stories

1. As a quantitative researcher, I want training samples with labels overlapping a test Information Interval to be purged, so that validation does not learn from the test event.
2. As a quantitative researcher, I want Embargo measured in Trading Sessions, so that weekends and market holidays do not distort the exclusion period.
3. As a quantitative researcher, I want each test fold to contain contiguous sessions, so that fold geometry is understandable and reproducible.
4. As a quantitative researcher, I want every active session to appear in exactly one Purged K-Fold test assignment, so that model-selection coverage is complete.
5. As a quantitative researcher, I want two-sided training to be labeled explicitly as model-selection evidence, so that it is not mistaken for a deployment simulation.
6. As a quantitative researcher, I want raw OOS predictions retained, so that I can recompute or audit metrics without rerunning models.
7. As a quantitative researcher, I want fold-level metric distributions rather than only an average, so that unstable market-period performance remains visible.
8. As a quantitative researcher, I want difficult folds to fail rather than disappear, so that reported scores are not selectively optimistic.
9. As a quantitative researcher, I want missing event metadata to stop formal evaluation, so that a convenient default cannot silently disable Purge.
10. As a quantitative researcher, I want latest-revision features rejected from leakage-safe scoring, so that revised historical data is not treated as point-in-time evidence.
11. As a quantitative researcher, I want feature availability checked against Decision Time, so that future publications cannot enter historical predictions.
12. As a quantitative researcher, I want optional per-feature availability evidence, so that strict datasets can prove more than a row-level maximum.
13. As a quantitative researcher, I want labels with variable event horizons supported, so that validation is not limited to a fixed T+N approximation.
14. As a quantitative researcher, I want disjoint protected intervals handled separately, so that valid middle training periods are not over-purged.
15. As a quantitative researcher, I want an Exclusion Summary for every fold, so that I can see the cost of Purge and Embargo.
16. As a quantitative researcher, I want an optional Exclusion Trace, so that I can audit why a particular sample was removed.
17. As a multi-asset researcher, I want all assets from the same Trading Session assigned together, so that cross-sectional information cannot leak across fold sides.
18. As a multi-asset researcher, I want stable asset and sample identities preserved in OOS output, so that predictions can be joined without relying on row order.
19. As a data engineer, I want an explicit Session Axis, so that temporal distance uses an authoritative market calendar.
20. As a data engineer, I want unknown or duplicate sessions rejected, so that ambiguous calendar mappings cannot produce a score.
21. As a data engineer, I want PIT Snapshot identity included in dataset provenance, so that evaluations can be tied to the historical source state.
22. As a data engineer, I want deterministic dataset digests, so that equivalent normalized inputs can be recognized and changed inputs detected.
23. As a data engineer, I want adapters to require explicit field mappings, so that column names and index levels are never guessed.
24. As a pandas user, I want a supported DataFrame and MultiIndex adapter, so that I can construct the canonical input without rewriting my data pipeline.
25. As a pandas user, I want hidden attributes ignored as authoritative metadata, so that serialization or copy operations cannot silently change leakage behavior.
26. As a library user, I want immutable Fold Assignments, so that downstream code cannot mutate an already-audited split.
27. As a library user, I want both stable sample IDs and array positions in Fold Assignments, so that results are auditable and efficient to apply.
28. As a library user, I want deterministic split IDs and configuration digests, so that repeated runs can be compared reliably.
29. As a library user, I want typed errors with actionable context, so that invalid evidence is distinguishable from model-training failure.
30. As a library user, I want error messages to avoid feature values, so that sensitive data does not leak into logs.
31. As a machine-learning engineer, I want estimator objects created from Fold Factories, so that fitted state cannot cross fold boundaries.
32. As a machine-learning engineer, I want transformation objects created separately for every fold, so that imputation, scaling, and selection remain Fold-Local.
33. As a machine-learning engineer, I want transformations fitted only on the current training side, so that test statistics cannot affect preprocessing.
34. As a machine-learning engineer, I want fixed model specifications versioned in Slice 1, so that OOS evidence identifies the evaluated model configuration.
35. As a machine-learning engineer, I want arbitrary fit/predict implementations supported through protocols, so that the core is not limited to scikit-learn.
36. As a machine-learning engineer, I want a failure in factory creation, fitting, transformation, prediction, shape validation, or metrics to abort evaluation, so that partial results are not mistaken for complete evidence.
37. As a machine-learning engineer, I want no pre-fitted object accepted by the canonical evaluator, so that global training state cannot enter accidentally.
38. As a model reviewer, I want every OOS observation bound to dataset, split, and model digests, so that a score has traceable inputs.
39. As a model reviewer, I want Derived Metrics to carry names and versions, so that changes in calculations are visible.
40. As a model reviewer, I want observation and fold coverage reported, so that missing predictions cannot hide behind an aggregate.
41. As a model reviewer, I want diagnostic planning separated from formal scoring, so that invalid candidates can be inspected without being accepted.
42. As a model reviewer, I want all requested folds to satisfy minimum train/test constraints, so that tiny or empty folds cannot contaminate conclusions.
43. As a risk reviewer, I want Purged K-Fold results kept separate from Causal Walk-Forward and Untouched Holdout evidence, so that claims match validation purpose.
44. As a risk reviewer, I want the library to state what it cannot prove about external feature metadata, so that provenance declarations are not overstated.
45. As a risk reviewer, I want later holdout reuse marked explicitly, so that a previously observed test set cannot regain untouched status through renaming.
46. As a maintainer, I want a small NumPy core, so that the trusted leakage logic has a narrow dependency surface.
47. As a maintainer, I want pandas to remain optional, so that core installation and testing do not require an ecosystem adapter.
48. As a maintainer, I want no required scikit-learn dependency in the first slice, so that protocol compatibility remains broader than one framework.
49. As a maintainer, I want property-based tests for leakage invariants, so that correctness is not tied only to hand-selected examples.
50. As a maintainer, I want known counterexamples preserved as regression tests, so that over-purge, inert embargo, panel leakage, and missing metadata do not return.
51. As a maintainer, I want public results and arrays protected from mutation, so that provenance remains valid after return.
52. As a maintainer, I want no shuffled Purged K-Fold mode, so that unsupported semantics cannot be enabled accidentally.
53. As a maintainer, I want formal verification commands to pass before Slice 1 is complete, so that implementation and documentation evidence agree.
54. As a future CPCV user, I want Slice 1 contracts to preserve contiguous test-block identity, so that deterministic Path Decomposition can be added without replacing the result model.
55. As a future walk-forward user, I want Purge separated from Embargo and Pre-Test Gap terminology, so that causal and two-sided semantics do not become conflated.
56. As a future HPO user, I want fixed-parameter evaluation completed first, so that nested search can reuse a trusted outer evaluation boundary.

## Implementation Decisions

- The primary acceptance seam is the public Leakage-Safe Evaluator. Tests at this seam cover dataset validation, split construction, fold-scoped object lifecycle, prediction collection, provenance, and metrics as one external behavior.
- A second diagnostic seam is the Split Plan. It exists because users must be able to inspect Invalid Folds and exclusion evidence without performing model training or creating scores.
- The canonical input is an explicit immutable Validation Dataset. It includes stable sample IDs, Session Axis membership, Decision Time, Information Intervals, Feature Availability, PIT Snapshot provenance, features, targets, and optional asset IDs.
- DataFrame and MultiIndex inputs are translated by an optional Pandas Adapter using explicit mappings. Automatic column-name, index-level, label-horizon, or hidden-metadata inference is prohibited.
- Every row-aligned input has the same length. Sample IDs are unique and stable. Session membership is authoritative and all temporal fields are ordered and resolvable.
- Formal leakage-safe scoring requires point-in-time feature provenance and a non-empty snapshot digest. Latest-revision or missing-provenance datasets may be diagnosed but not scored.
- Panel Session Groups are indivisible assignment units. Fold geometry is constructed from unique ordered sessions, never from raw panel rows.
- Purged K-Fold creates deterministic contiguous test blocks whose session counts differ by at most one. No shuffled mode is supported.
- Candidate training data may exist before and after the test block because this Evidence Channel is for leakage-aware model selection, not causal deployment simulation.
- Purge removes a candidate training sample whenever its Information Interval overlaps any test Information Interval. Efficient merging may combine overlapping protected intervals but may not replace disjoint intervals with one global envelope.
- Embargo is applied independently after each contiguous test block and advances by Session Axis positions from the latest test Information Interval end for that block.
- Minimum training sessions, training samples, and test sessions are explicit configuration. Any requested Invalid Fold makes formal assignment and evaluation fail as a whole.
- Splitters return immutable Fold Assignments containing stable identities, array positions, contiguous block identity, Exclusion Summary, optional Exclusion Trace, deterministic IDs, digests, and schema version.
- Bare train/test tuples are not canonical Slice 1 output. A later compatibility adapter may derive them without changing the evidence-bearing result.
- The evaluator accepts estimator and transformation Fold Factories rather than pre-fitted instances. Every factory creates a new unfitted object for one fold.
- Fold-local transformations are fitted sequentially on training observations and applied to the current train/test observations only. The estimator is fitted after transformation and only on the current training side.
- Slice 1 evaluates fixed model configurations. Hyperparameter selection belongs to a later nested-HPO slice.
- Any factory, fit, transform, predict, shape, or metric failure terminates evaluation. Partial scores and warning-only degradation are not valid evidence.
- The OOS Ledger is the authoritative in-memory result. Each observation retains run, sample, session, optional asset, split, model, dataset, split-spec, and model-spec identity.
- Derived Metrics are pure, named, versioned projections from the OOS Ledger. Results expose per-fold values, observation counts, coverage, and distribution summaries rather than only a mean.
- Evidence Channels remain separate. Purged K-Fold scores cannot be promoted to Causal Walk-Forward or Untouched Holdout claims.
- Canonical serialization and cryptographic digests cover relevant normalized input, schema, split, model, and metric configuration so changes are observable.
- Public errors are typed under a common validation boundary and distinguish dataset, temporal, split-plan, fold-construction, factory-lifecycle, execution, shape, and metric failures.
- The required runtime is Python 3.11 or newer plus NumPy. Pandas is an optional adapter dependency. Hypothesis is a development dependency. Scikit-learn is not required in Slice 1.
- The architecture separates immutable domain values, cross-field validation, leakage rules, splitter planning, evaluation, metrics, and optional adapters. External adapters do not own leakage semantics.
- Slice 1 does not integrate with the existing factor-lab project. Integration begins only after the standalone behavior and evidence are complete.
- Later delivery slices add Causal Walk-Forward and governed holdout access, then general deterministic CPCV paths, then nested HPO and ecosystem/persistence adapters, followed by performance and release hardening.

## Testing Decisions

- Tests assert external behavior and domain invariants rather than private helper structure. Refactoring internal algorithms must not require rewriting tests when public evidence remains unchanged.
- The highest and most important seam is a complete Leakage-Safe Evaluator call using a deterministic toy estimator, deterministic transformations, an explicit Validation Dataset, and public result inspection.
- Split Plan tests exercise diagnosis independently because no valid evaluator result should be required to inspect Invalid Fold reasons, block geometry, or exclusion counts.
- Domain-contract tests cover immutability, read-only arrays, cross-field shape checks, stable identity, temporal order, Session Axis membership, canonical serialization, and digest sensitivity.
- Dataset tests cover duplicate/null sample IDs, unknown/duplicate sessions, invalid Information Intervals, Feature Availability later than Decision Time, absent PIT provenance, latest-revision inputs, panel grouping, missing-value policy, and sensitive error output.
- Leakage-engine tests cover exact interval overlap, boundary inclusivity, overlapping protected-interval merging, disjoint intervals, per-block Embargo, holiday/session arithmetic, deterministic order, summaries, and traces.
- Purged K-Fold behavioral tests prove complete test-session coverage, train/test disjointness, contiguous deterministic geometry, indivisible Panel Session Groups, two-sided model-selection labeling, post-exclusion minima, and whole-evaluation failure on Invalid Folds.
- Evaluator tests prove that each fold receives distinct estimator and transformer instances, fit state is Fold-Local, test observations never enter fitting, transformations remain ordered, and every execution failure aborts the run.
- OOS Ledger tests prove observation identity, coverage, provenance digests, result immutability, pure metric projection, metric versioning, and absence of silent averaging or missing predictions.
- Adapter tests prove explicit mapping, equivalent canonical datasets, MultiIndex panel preservation, timezone handling, duplicate-index rejection, unknown-session rejection, hidden-attribute non-authority, and successful core use without pandas installed.
- Regression tests preserve concrete failures discovered during project exploration: event overlap around T+5 labels, a disjoint protected-block min/max over-purge counterexample, an embargo configuration that previously had no effect, same-session multi-asset leakage, and latest-revision feature leakage.
- Property-based tests generate axes, panel groups, intervals, horizons, and gap configurations. They prove no retained training interval overlaps test information, every active session receives one test assignment, same-session rows share assignment, outputs are deterministic, non-session calendar days do not change session embargo, and relevant changes alter digests.
- Negative tests are first-class. Missing metadata, invalid sizes, unsupported provenance, factory misuse, and incomplete predictions must fail explicitly rather than be normalized into warnings or NaN scores.
- Existing repository tests only establish the scaffold contract; there is no existing product-level seam to reuse. The new public evaluator and diagnostic planner are therefore the minimum necessary seams.
- Slice 1 acceptance requires focused tests, regressions, properties, project preflight, the complete project test task, Ruff, package import checks, public documentation, and requirements-to-evidence traceability.
- Test evidence must distinguish locally observed results from future cross-platform or release evidence. A local pass does not authorize deployment or later-slice capability claims.

## Out of Scope

- CPCV combination generation, Path Decomposition, and path-level metric distributions.
- Causal Walk-Forward, rolling windows, expanding windows, and Pre-Test Gap policies.
- Evaluation Protocol persistence, final holdout authorization, and Holdout Receipts.
- Nested HPO, search-space execution, and outer/inner model-selection orchestration.
- Scikit-learn cloning and train/test tuple compatibility.
- Parquet, database, MLflow, object-store, or other persistent OOS writers.
- Parallel or distributed fold execution.
- Performance targets or optimization before a correctness baseline is measured.
- Production deployment, external-system writes, credential handling, or Registry integration.
- Direct integration with the existing factor-lab codebase.
- Claims that external source metadata is truthful beyond the timing, revision policy, and digest evidence supplied to the library.
- Claims that Purged K-Fold alone provides causal or deployment-grade OOS evidence.

## Further Notes

- This spec was accepted locally on 2026-07-31 after all seven Slice 1 tickets were resolved and the recorded verification gates passed.
- The approved domain glossary defines Information Interval, Trading Session, Validation Dataset, Purge, Embargo, Fold Assignment, Leakage-Safe Evaluator, Fold-Local, OOS Ledger, Evidence Channel, and related terms. Implementation and tests must use that vocabulary rather than drifting to ambiguous synonyms.
- Nine accepted architectural decisions govern evaluator ownership, explicit input, factories, evidence-bearing folds, strict failure, OOS facts, deterministic future CPCV paths, governed future holdout access, and feature availability evidence.
- Slice 1 is complete only when all associated implementation tickets are resolved with recorded acceptance evidence and every approved requirement maps to implementation and observed tests.
- Completion of Slice 1 authorizes work on later slices; it does not claim that those capabilities exist.
- Performance budgets remain deliberately unset until the correctness implementation can provide representative measurements. Any later optimization must preserve the public behavioral seams and evidence model.
