# Work items

The `ready-for-agent` Slice 1 specification is `.scratch/purged-kfold-v1/spec.md`. Detailed local tickets live under `.scratch/purged-kfold-v1/issues/`.

| ID | Priority | Status | Depends on | Scope | Acceptance evidence |
|---|---|---|---|---|---|
| PKF-001 | P0 | resolved | — | Minimal leakage-safe Split Plan from an explicit single-asset dataset | 4 focused tests and Ruff observed passing |
| PKF-002 | P0 | resolved | PKF-001 | Fold-local training, raw OOS Ledger and one derived score | 4 evaluator tests in an 8-test focused pass |
| PKF-003 | P0 | resolved | PKF-001 | Complete strict Purged K-Fold planning through evaluation | 6 strict-planning cases in a 14-test focused pass |
| PKF-004 | P0 | resolved | PKF-003 | Financial time boundaries, per-block Embargo and panel grouping | 3 financial-boundary tests in a 17-test focused pass |
| PKF-005 | P0 | resolved | PKF-002 | PIT-safe formal scoring through the evaluator | 3 PIT scenarios in a 21-test full pass |
| PKF-006 | P1 | resolved | PKF-004, PKF-005 | Explicit pandas input through the same planning and evaluation path | 4 adapter tests in a 25-test full pass |
| PKF-007 | P0 | ready | PKF-002–006 | Slice 1 adversarial evidence, docs and traceability | Full acceptance gate in the spec |
| WF-001 | P1 | planned | PKF-007 | Causal Walk-Forward and Pre-Test Gap | Strict chronology properties |
| HOLDOUT-001 | P1 | planned | WF-001 | Evaluation Protocol, local store, Holdout Receipt | State-machine and atomicity tests |
| CPCV-001 | P1 | planned | PKF-007 | General combinations and deterministic Path Decomposition | Combinatorial/path invariants |
| HPO-001 | P2 | planned | PKF-005, CPCV-001 | Nested HPO and selection evidence | Outer/inner isolation tests |
| ADAPTER-001 | P2 | planned | PKF-007 | sklearn compatibility and persistence writers | Optional dependency/canary tests |

## Execution rule

Only `ready` and dependency-free work may start. A downstream item becomes ready when every dependency is resolved with recorded acceptance evidence. Slice 1 completion does not authorize claims for later slices.
