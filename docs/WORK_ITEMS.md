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
| PKF-007 | P0 | resolved | PKF-002–006 | Slice 1 adversarial evidence, docs and traceability | 41 tests plus full local acceptance and dual-axis review |
| WF-001 | P1 | resolved | PKF-007 | Causal Walk-Forward and Pre-Test Gap | 7 focused causal cases; 52-test full pass |
| BENCH-001 | P1 | resolved | WF-001 | Offline PandaAI adapter and four-channel benchmark | Synthetic controls plus five-asset real-cache receipt |
| HOLDOUT-001 | P1 | resolved | WF-001 | Evaluation Protocol, local store, Holdout Receipt | One-time consumption, train-only fit, redacted receipt, and boundary tests |
| CPCV-001 | P1 | resolved | PKF-007 | General combinations and deterministic Path Decomposition | 30 focused CPCV cases; 82-test full pass; five-asset receipt |
| DATA-001 | P0 | resolved | CPCV-001 | PandaAI continuous-contract identity governance | Synthetic receipt test plus governed 189,317-row local cache |
| METRIC-001 | P0 | resolved | DATA-001 | CPCV path financial metrics and distributions | Five native/common/roll-clean path tables |
| EVAL-001 | P0 | resolved | DATA-001, METRIC-001 | Full-cache three-channel effectiveness comparison | 175,129 observations; 15 CPCV combinations; zero overlap; all sufficiency gates pass |
| FEATURE-001 | P0 | resolved | EVAL-001 | Immutable arbitrary-feature manifest and governed dataset | Public-seam tests bind manifest into dataset and OOS evidence |
| FEATURE-002 | P0 | resolved | FEATURE-001 | Explicit pandas arbitrary-feature upload | Ordered feature/availability mapping test and pandas-optional core canary |
| FEATURE-003 | P0 | resolved | FEATURE-001 | Fold-local transformer lineage | One-to-one spec/factory binding and run-identity regression tests |
| UPLOAD-001 | P0 | resolved | FEATURE-002 | Versioned local upload contracts and audit command | Audit, rejection, late-availability, limits, and example tests |
| UPLOAD-002 | P0 | resolved | UPLOAD-001 | Three-channel evaluation and resource limits | Three-channel/path and CPCV-budget tests |
| UPLOAD-003 | P0 | resolved | FEATURE-003 | High-level transformer identity binding | Report-digest regression test |
| UPLOAD-004 | P0 | resolved | UPLOAD-001–003 | Examples, docs, packaging, final acceptance | 107 tests; typed/linted build and wheel canary |
| DISTCLI-001 | P0 | resolved | UPLOAD-004 | Package-native CLI and legacy compatibility wrapper | Module/script output equivalence and 19 CLI cases |
| DISTCLI-002 | P0 | resolved | DISTCLI-001 | Installed schemas and safe example materialization | Schema, three-example, optional-dependency, and overwrite cases |
| DISTCLI-003 | P0 | resolved | DISTCLI-001–002 | Console metadata and isolated-wheel acceptance | 117 tests; wheel resources; installed console/module canaries |
| RESOURCE-001 | P0 | resolved | UPLOAD-001 | Read-before-allocate upload budgets | CSV bounded read and Parquet footer rejection before materialization |
| ROBUST-001 | P1 | resolved | EVAL-001 | Multi-regime fixed-model ranking stability evidence | Stable and adversarial rank-reversal tests |
| REALGATE-001 | P0 | resolved | HOLDOUT-001, ROBUST-001 | Governed five-year PandaData release gate | 88,417 eligible observations; zero overlap; stable three-regime ranking; one-attempt Holdout receipt |
| SKILL-001 | P0 | resolved | DISTCLI-003, REALGATE-001 | Portable Skill discovery, one-request execution, schemas, Quick Start | Root Skill, standard JSON envelope, demo/example/schema commands, contract tests |
| TSBENCH-001 | P0 | resolved | CPCV-001, WF-001 | Generic strategy-return audit plus built-in TSMOM time-series benchmark | 150 full tests; strict quality gate; 0.7.0 artifacts; isolated 32-trial/4-cost canary |
| REALTS-001 | P0 | resolved | TSBENCH-001 | Five-year multi-asset PandaData TSMOM selection benchmark | 15 assets; 1,210 Sessions; formal CLI match; PBO pass; DSR production gate fail |
| ACCEPT-001 | P0 | resolved | REALTS-001 | Pre-registered strategy acceptance decisions, evidence gaps, and DSR track-record diagnostics | 159 tests; v0.7.1 build/Twine/wheel canary; PandaData PASS/FAIL/FAIL replay |
| TMODEL-001 | P0 | resolved | ACCEPT-001 | Trainable temporal-model leakage comparison across unsafe, Purged, Embargo, CPCV, and causal channels | 168 tests; 17,775-row three-model replay; unsafe overlaps detected; all safe overlaps zero |
| PROD-001 | P0 | ready-for-human | REALGATE-001 | Freeze v0.6.1 and execute remote cross-platform CI | Local gates/artifacts complete; remote URL and authenticated GitHub CLI still required |
| HPO-001 | P2 | ready | PKF-005, CPCV-001 | Nested HPO and selection evidence | Outer/inner isolation tests |
| ADAPTER-001 | P2 | planned | PKF-007 | sklearn compatibility and persistence writers | Optional dependency/canary tests |

## Execution rule

Only `ready` and dependency-free work may start. A downstream item becomes ready when every dependency is resolved with recorded acceptance evidence. Slice 1 completion does not authorize claims for later slices.
