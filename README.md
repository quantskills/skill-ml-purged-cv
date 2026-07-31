# purged-kfold-validation

This repository is the maintained project for the purpose below.

## Project roadmap

The long-term roadmap is a cross-platform financial validation library spanning Purged
K-Fold, CPCV, Causal Walk-Forward, and governed holdout evidence. Capabilities are only
claimed when their delivery slice has implementation and observed acceptance evidence.

## Implemented scope

Version 0.1 currently implements the first model-selection slice only: explicit point-in-time
`ValidationDataset` input, diagnostic Purged K-Fold `SplitPlan`, session-based Embargo,
Fold-Local evaluation, raw `OOSLedger`, versioned metrics, and an optional explicit
pandas adapter.

Purged K-Fold permits training observations before and after a test block. Its result
is leakage-aware **model-selection evidence**, not a causal deployment simulation.
CPCV, Causal Walk-Forward, Untouched Holdout governance, nested HPO, persistence,
parallel execution, and scikit-learn adapters are not implemented in this slice.

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

## Input and output contracts

The stable Slice 1 public contract is documented in `docs/interface-contract.md`.
Runtime results are in-memory immutable evidence; persistence is intentionally outside
this slice.

## Dependency manifest source

Selected language manifest: `pyproject.toml`. Do not add a competing manifest without an explicit migration decision.
