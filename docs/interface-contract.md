# Interface contract

Windows、Linux、macOS 跨平台可复用 Python 软件库，用于金融机器学习训练验证，提供基于显式事件区间和交易日历 session 的 Purged K-Fold、Embargo、CPCV 路径拼接、严格 walk-forward 与 untouched holdout；多资产 panel 按同日分组，预处理与嵌套 HPO 必须 fold-local，缺失泄漏元数据时 fail closed，并以属性测试和回归测试验证防未来函数与信息泄漏。

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
- Public arrays are read-only copies and domain results are immutable.
- Dataset, split specification, fold, plan, model, metric, run, ledger, and PIT source
  state have deterministic digests or identities where applicable.
- `OOSObservation` retains sample, session, optional asset, fold, split, model, dataset,
  split-specification, PIT source, run, and Evidence Channel identity.
- Public failures derive from `ValidationError`; dataset, temporal/PIT, split, factory,
  execution, shape, metric, and adapter failures remain distinguishable.

### Optional pandas boundary

`PandasField`, `PandasDatasetMapping`, and `validation_dataset_from_pandas` require every
column or index level to be mapped explicitly. They do not infer names, horizons, index
semantics, snapshot provenance, or hidden DataFrame attrs. The adapter normalizes one
consistent timezone and delegates all leakage semantics to the canonical core. Importing
the core does not import or require pandas.

## Evidence boundary and non-capabilities

Every Slice 1 result has the `model-selection` Evidence Channel. Two-sided training is
permitted, so these results are not causal deployment evidence. CPCV combinations and
paths, Causal Walk-Forward, Pre-Test Gap, Evaluation Protocols, Holdout Receipts,
Untouched Holdout claims, nested HPO, persistence writers, parallelism, scikit-learn
compatibility, and external metadata truth verification are not implemented.

The approved requirements remain in `.scratch/purged-kfold-v1/spec.md`; their observed
implementation evidence is mapped in `docs/requirements-traceability.md`.
