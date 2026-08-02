# Architecture

The repository owns maintained source, tests, configuration contracts, entrypoints, and handoff documentation. External systems remain separately owned.

## Planned module boundaries

```text
src/purged_kfold_validation/
├── domain.py             # immutable domain values and digests
├── validation.py         # cross-field fail-closed dataset checks
├── features.py           # arbitrary-feature manifests and governance receipts
├── leakage.py            # interval purge and session gap semantics
├── splitters/
│   ├── purged_kfold.py
│   ├── walk_forward.py   # causal expanding/sliding assignments
│   └── cpcv.py           # general bounded combination assignments
├── paths.py              # deterministic CPCV Path Decomposition
├── evaluation.py         # Fold Factories and Leakage-Safe Evaluator
├── holdout.py            # frozen protocol, one-attempt store, redacted receipt
├── ranking.py            # fixed-model cross-regime rank stability
├── cli.py                # installed governed-upload command implementation
├── __main__.py           # python -m distribution entry point
├── benchmark.py          # four-channel structural validation benchmark
├── metrics.py            # pure versioned Derived Metrics
├── adapters/
│   ├── pandas.py         # optional Slice 1 adapter
│   ├── upload.py         # bounded CSV/Parquet contract adapter
│   ├── pandaai.py        # optional offline daily-frame adapter
│   └── sklearn.py        # later optional adapter
└── writers/              # later optional persistence
```

`domain.py`, `validation.py`, and `leakage.py` form the trusted core. Splitters consume normalized validated contracts, the evaluator consumes Fold Assignments, and adapters translate external ecosystems without owning leakage semantics. Raw OOS observations flow outward to metrics and optional writers; derived scores never become the source of truth.

The implemented Slice 1 follows these boundaries: immutable values and digests remain
in `domain.py`, cross-field and PIT eligibility checks live in `validation.py`, and exact
Purge/Embargo decisions live in `leakage.py`.

Slice 2 adds `CausalWalkForward` without creating a second evaluator. Fold Assignments
carry their Evidence Channel into the existing fold-local evaluator and OOS Ledger.
The PandaAI adapter and benchmark remain optional boundaries; neither owns core leakage
semantics or authenticates to an external service.

Slice 3 adds `CombinatorialPurgedCV` as a consumer of the same leakage core. `paths.py`
properly edge-colors the combination/group incidence graph, producing general
deterministic paths without random or fixed-configuration templates. The existing
evaluator projects repeated OOS facts into combination and path metrics without
averaging away their identities.

Slice 5 adds `features.py` as a narrow evidence module. It does not compute or store
features. It binds ordered definitions, source/code/vintage digests, per-feature
availability, and uploaded-value lifecycle rules to the canonical dataset. The pandas
adapter remains the external translation boundary. Stateful learned transformations
stay inside the existing evaluator and now carry ordered `TransformerSpec` identities.

Slice 5.1 adds a delivery boundary, not a new validation engine. `adapters/upload.py`
parses closed version-1 manifest/mapping documents, enforces file/row/feature limits,
loads CSV or Parquet through pandas, and delegates to the same governed dataset seam.
`scripts/audit_feature_upload.py` performs either audit-only receipt generation or a
post-audit three-channel comparison. It does not execute feature or transformer code
supplied by users. CPCV combination geometry is bounded before fitting, while ordered
fold-local Transformer Specs flow through every high-level comparison channel.

Slice 5.2 moves the operator behavior into `cli.py`. Both the console entry point and
`__main__.py` call that implementation; the repository script contains only import-path
bootstrapping and delegation. Schemas and examples are immutable package data under
`resources/feature_upload/`. Schema/example commands return before the optional pandas
adapter is imported, preserving the NumPy-only base runtime. Audit/evaluate lazily load
the upload adapter and return installation guidance when the `upload` extra is absent.

Slice 6 adds `holdout.py` as a deliberately separate terminal evidence boundary. Its
filesystem store uses exclusive claim creation keyed by Holdout dataset digest before
fitting, so a process failure cannot silently restore untouched status. `ranking.py`
keeps model-selection stability evidence separate from Holdout confirmation. Upload
hardening uses bounded CSV parsing and Parquet footer budgets before materialization.

## Delivery slices

1. Purged K-Fold vertical slice.
2. Causal Walk-Forward and offline benchmark.
3. General deterministic CPCV paths.
4. Governed full-cache effectiveness comparison.
5. Arbitrary-feature availability and lineage governance.
6. Resource-bounded local feature upload audit/evaluation.
7. Distribution-native CLI and packaged upload resources.
8. Frozen one-attempt Holdout, ranking stability, and upload resource hardening.
9. Nested HPO and ecosystem/persistence adapters.
10. Parallelism, performance evidence, and external release hardening.
