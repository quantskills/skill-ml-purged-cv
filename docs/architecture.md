# Architecture

The repository owns maintained source, tests, configuration contracts, entrypoints, and handoff documentation. External systems remain separately owned.

## Planned module boundaries

```text
src/purged_kfold_validation/
├── domain.py             # immutable domain values and digests
├── validation.py         # cross-field fail-closed dataset checks
├── leakage.py            # interval purge and session gap semantics
├── splitters/
│   ├── purged_kfold.py
│   ├── walk_forward.py   # later slice
│   └── cpcv.py           # later slice
├── paths.py              # later CPCV Path Decomposition
├── evaluation.py         # Fold Factories and Leakage-Safe Evaluator
├── metrics.py            # pure versioned Derived Metrics
├── protocols.py          # later Evaluation Protocol/Holdout Receipt
├── adapters/
│   ├── pandas.py         # optional Slice 1 adapter
│   └── sklearn.py        # later optional adapter
└── writers/              # later optional persistence
```

`domain.py`, `validation.py`, and `leakage.py` form the trusted core. Splitters consume normalized validated contracts, the evaluator consumes Fold Assignments, and adapters translate external ecosystems without owning leakage semantics. Raw OOS observations flow outward to metrics and optional writers; derived scores never become the source of truth.

## Delivery slices

1. Purged K-Fold vertical slice.
2. Causal Walk-Forward and governed holdout.
3. General deterministic CPCV paths.
4. Nested HPO and ecosystem/persistence adapters.
5. Parallelism, performance evidence, and release hardening.
