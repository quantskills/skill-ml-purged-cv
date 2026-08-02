# purged-kfold-validation

This repository is the maintained project for the purpose below.

## Project roadmap

The long-term roadmap is a cross-platform financial validation library spanning Purged
K-Fold, CPCV, Causal Walk-Forward, and governed holdout evidence. Capabilities are only
claimed when their delivery slice has implementation and observed acceptance evidence.

## Implemented scope

Version 0.6.1 implements explicit point-in-time
`ValidationDataset` input, diagnostic Purged K-Fold `SplitPlan`, session-based Embargo,
general deterministic CPCV combinations and Path Decomposition, causal Walk-Forward
with Pre-Test Gap, Fold-Local evaluation, raw path-scoped `OOSLedger`, versioned fold
and path metrics, an optional explicit pandas adapter, governed PandaAI continuous-
contract identity, per-path MSE/IC/diagnostic-Sharpe distributions, an offline
three-channel full-cache comparison, and arbitrary-feature availability/lineage
governance with evidence-bound fold-local transformer specifications. It also provides
an installable resource-bounded CSV/Parquet audit/evaluate CLI with versioned contracts
and packaged examples. It also implements a frozen Evaluation Protocol, an atomic
one-attempt local Holdout Store with redacted receipts, and fixed-model ranking
stability evidence across declared market regimes.

A governed five-year PandaData acceptance run covers 2021-06-18 through 2026-06-18,
81 assets, and 88,417 eligible observations. Purged K-Fold, CPCV, and causal
Walk-Forward retained zero overlapping training/test Information Intervals; the
one-attempt Holdout receipt is preserved separately. Exact results and the important
non-profitability boundary are recorded in
`docs/evidence/pandadata-five-year-release-gate-20260802.md`.

Purged K-Fold permits training observations before and after a test block. Its result
is leakage-aware **model-selection evidence**, not a causal deployment simulation.
Nested HPO, general persistence writers, parallel execution, and scikit-learn adapters
are not implemented in this slice. Holdout governance covers only access performed
through `LocalHoldoutStore`; it cannot prove that a caller did not inspect the data
through another path.

## Classification summary

Primary type: `library-cli`  
Domain: `finance`  
Runtime: `library`  
Risk: `R1`  
Language: `python`

## Program and governance

`PROGRAM.md` is the durable project direction and boundary contract. `CONTEXT.md` records current state, `README.md` explains use, and `docs/WORK_ITEMS.md` tracks current execution.

## Installation and setup

Create a virtual environment, then run `python -m pip install -e ".[dev]"`.
Install `.[pandas]` as well when the optional adapter is required.
Install `.[benchmark]` to run the offline PandaAI parquet benchmark.
Install `.[upload]` to use the CSV/Parquet upload CLI with its Parquet engine.

## Common commands

Run `python tasks/preflight.py` and `python tasks/test.py` from PowerShell, Command Prompt, or a POSIX shell. Bash is not required.

## Minimal diagnostic and formal evaluation

```python
import numpy as np

from purged_kfold_validation import (
    InformationInterval,
    LeakageSafeEvaluator,
    MetricSpec,
    ModelSpec,
    PITSnapshot,
    PurgedKFold,
    ValidationDataset,
)


class MeanEstimator:
    def fit(self, features, targets):
        self.mean = float(targets.mean())
        return self

    def predict(self, features):
        return np.full(features.shape[0], self.mean)


sessions = np.array(
    ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
    dtype="datetime64[D]",
)
dataset = ValidationDataset(
    sample_ids=("s0", "s1", "s2", "s3"),
    session_axis=sessions,
    sessions=sessions,
    information_intervals=tuple(
        InformationInterval(session, session) for session in sessions
    ),
    decision_times=sessions.astype("datetime64[ns]") + np.timedelta64(12, "h"),
    feature_availability=sessions,
    pit_snapshot=PITSnapshot(
        snapshot_id="vendor-vintage-2025-q1",
        source_digest="sha256-of-source-state",
    ),
    features=np.arange(8, dtype=float).reshape(4, 2),
    targets=np.array([0.0, 0.0, 1.0, 1.0]),
)

splitter = PurgedKFold(n_splits=2, embargo_sessions=1)

# Diagnostic only: invalid candidates and exclusions remain inspectable.
plan = splitter.plan(dataset)
for candidate in plan.folds:
    print(candidate)

# Formal scoring: every fold and every PIT declaration must be valid.
result = LeakageSafeEvaluator(
    splitter=splitter,
    estimator_factory=MeanEstimator,
    model_spec=ModelSpec(name="mean", version="1"),
    metrics=(
        MetricSpec(
            name="mse",
            version="1",
            function=lambda actual, predicted: float(
                np.mean((actual - predicted) ** 2)
            ),
        ),
    ),
).evaluate(dataset)

print(result.ledger.observations)  # authoritative OOS facts
print(result.metrics)              # derived, versioned projections
```

The evaluator accepts factories, never pre-fitted objects. Missing or latest-revision
PIT evidence, an Invalid Fold, reused fold objects, incomplete predictions, or any
execution failure aborts the run before a formal result is returned.

## Arbitrary feature uploads

Raw levels, returns, rolling features, factors, and embeddings are all acceptable finite
numeric inputs. The library does not require or infer stationarity. A governed upload
must provide one ordered `FeatureDefinition` per feature column and a two-dimensional
row-by-feature availability matrix:

```python
from purged_kfold_validation import (
    FeatureComputationScope,
    FeatureDefinition,
    FeatureManifest,
    govern_feature_dataset,
)

manifest = FeatureManifest(
    source_bundle_digest="<same SHA-256 as dataset.pit_snapshot.source_digest>",
    definitions=(
        FeatureDefinition(
            name="momentum_20",
            source_dataset="vendor.daily",
            source_fields=("close",),
            source_digest="<source-vintage SHA-256>",
            transformation="simple-return",
            transformation_version="1",
            code_digest="<feature-code SHA-256>",
            parameters={"periods": 20},
            lookback_sessions=20,
            computation_scope=FeatureComputationScope.PRECOMPUTED_STATELESS,
        ),
    ),
)
governed = govern_feature_dataset(dataset, manifest)
result = evaluator.evaluate(governed.dataset)
```

Uploaded matrices must be stateless, target-independent, point-in-time values. Learned
imputation, scaling, PCA, selection, or target encoding belongs in fresh fold-local
transformer factories and requires one ordered `TransformerSpec` per factory. The
manifest and transformer digests reach run and OOS evidence. These controls validate
the supplied declarations; they cannot reconstruct a formula or prove that a caller
reported its timestamp/source truthfully from numeric values alone.

### Installable CSV/Parquet audit and evaluation

The installed CLI makes governance a separate first stage. It accepts local data only,
executes no uploaded code, and emits redacted JSON rather than source rows:

```powershell
purged-cv-upload audit `
  --data config/feature-upload/examples/raw/features.csv `
  --manifest config/feature-upload/examples/raw/manifest.json `
  --mapping config/feature-upload/examples/raw/mapping.json
```

`python -m purged_kfold_validation` is an equivalent entry point. The earlier
`scripts/audit_feature_upload.py` path remains a compatibility wrapper for repository
users.

Use `evaluate` with the same three files to run the audited matrix through Purged
K-Fold, CPCV, and causal Walk-Forward. Real evaluation inputs must be large enough for
the declared split and training-sufficiency gates; the checked-in examples are deliberately
small audit fixtures. Default ceilings are 512 MiB compressed/on-disk bytes, 2 GiB
declared Parquet uncompressed bytes, 1,000,000 rows, 2,048 columns, 512 features, and
10,000 CPCV combinations, with explicit CLI overrides available. CSV uses a bounded
read; Parquet row, column, and uncompressed-size budgets are checked from footer
metadata before table materialization.

The version-1 contracts live in `config/feature-upload/`. Both raw levels and stationary
features are admissible when their declarations pass. The intentional-leak bundle shows
the expected rejection for a target-derived feature. Audit proves internal consistency
of supplied metadata, not the external truth of a timestamp or formula.

Installed users can inspect the packaged schemas and materialize an example without a
source checkout:

```powershell
purged-cv-upload schema --kind manifest
purged-cv-upload example --name raw --output-dir .\raw-example
```

Example export writes only `features.csv`, `manifest.json`, and `mapping.json` beneath
the explicit output directory and refuses to overwrite an existing target file.

## Governed final Holdout

Freeze the final training/Holdout boundary and every model-selection identity before
opening the Holdout through the library. `LocalHoldoutStore.evaluate_once` atomically
claims the Holdout dataset digest before it creates a transformer or estimator. A
failed attempt consumes the identity as well; changing `protocol_id` cannot make the
same Holdout untouched again.

```python
from pathlib import Path

from purged_kfold_validation import EvaluationProtocol, LocalHoldoutStore

protocol = EvaluationProtocol.freeze(
    protocol_id="release-candidate-1",
    training_dataset=training_dataset,
    holdout_dataset=holdout_dataset,
    model_spec=model_spec,
    transformer_specs=transformer_specs,
    metrics=metrics,
    search_policy={"kind": "completed-nested-cv", "frozen": True},
    split_spec_digest=split_spec_digest,
)
final = LocalHoldoutStore(Path("holdout-store")).evaluate_once(
    protocol,
    training_dataset=training_dataset,
    holdout_dataset=holdout_dataset,
    estimator_factory=estimator_factory,
    model_spec=model_spec,
    transformer_factories=transformer_factories,
    transformer_specs=transformer_specs,
    metrics=metrics,
)
```

The in-memory result has the separate `holdout-confirmation` Evidence Channel. The
durable receipt contains only digests, metric values, and evaluation time—not source
rows, features, targets, or predictions. The store governs its own interface; operating
system access controls are still required to prevent out-of-band inspection.

`assess_model_ranking_stability` separately summarizes the same fixed model set across
at least two declared regimes. It reports median/worst rank, first-place counts, and
pairwise Spearman rank correlations. A rank reversal is evidence of instability, not a
reason to query the Holdout again.

## Causal Walk-Forward

`CausalWalkForward` restricts every fold to past training sessions, applies exact
Information Interval Purge, optionally excludes a Session Axis `pre_test_gap_sessions`,
and supports expanding or bounded sliding history. Its assignments and OOS observations
carry the separate `causal-walk-forward` Evidence Channel.

```python
from purged_kfold_validation import CausalWalkForward

splitter = CausalWalkForward(
    n_splits=5,
    test_sessions=20,
    pre_test_gap_sessions=5,
)
causal_plan = splitter.plan(dataset)
```

## Combinatorial Purged Cross-Validation

`CombinatorialPurgedCV` keeps Purged K-Fold available for inexpensive development and
adds a separate `cpcv-robustness` Evidence Channel for exhaustive multi-group
combinations. For `N=6, k=2`, it plans 15 combinations and deterministically decomposes
their repeated OOS occurrences into 5 complete CPCV Paths.

```python
from purged_kfold_validation import CombinatorialPurgedCV

cpcv = CombinatorialPurgedCV(
    n_groups=6,
    n_test_groups=2,
    embargo_sessions=5,
    max_combinations=1_000,
)
cpcv_plan = cpcv.plan(dataset)
assignments = cpcv_plan.require_assignments()
paths = cpcv_plan.path_decomposition
```

Selected chronological groups may be disjoint. Purge protects all selected test
Information Intervals, while Embargo applies after every contiguous selected test
region. CPCV may train on groups later than a test group and therefore does not replace
Causal Walk-Forward or constitute deployment evidence.

## Offline PandaAI benchmark

The benchmark reads only user-supplied local `*_daily.parquet` caches. It never logs in
to PandaAI and never reads credentials. This example compares shuffled-session K-Fold,
chronological no-purge, Purged K-Fold, and Causal Walk-Forward using the same normalized
dataset and fold-local model:

```powershell
python scripts/benchmark_pandaai.py `
  --data-dir D:\path\to\pandaai_future_daily `
  --feature-columns close,volume,open_interest `
  --label-horizon-sessions 5 `
  --feature-lookback-sessions 20 `
  --n-splits 5
```

The JSON output contains digests, coverage, metrics, and retained overlap counts only;
it does not contain source rows or constitute a profitability claim. Snapshot identity
and source digests prove the supplied declaration, not the external vendor's truthfulness.

## Input and output contracts

The stable public contracts are documented in `docs/interface-contract.md`.
Runtime results are in-memory immutable evidence; persistence is intentionally outside
this slice.

## Dependency manifest source

Selected language manifest: `pyproject.toml`. Do not add a competing manifest without an explicit migration decision.
