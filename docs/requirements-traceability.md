# Requirements traceability

Slice 1 user-story numbers refer to `.scratch/purged-kfold-v1/spec.md`. “Boundary” means
the story deliberately describes a later slice and is verified here as an explicit
non-capability rather than an implemented feature.

| Spec stories / invariant | Implemented behavior | Observed verification | Status |
|---|---|---|---|
| 1, 13–16, 50 — exact Purge and exclusions | Inclusive Information Interval overlap, disjoint protected intervals, summaries and optional traces | `test_split_plan.py`; `test_financial_time_boundaries.py`; `test_properties.py` | verified |
| 2, 15, 50 — Session Axis Embargo | Embargo advances after each TestBlock by authoritative sessions rather than calendar days | `test_financial_time_boundaries.py`; generated calendar-gap property | verified |
| 3–5, 8, 17, 27–28, 41–42, 52 — strict K-Fold geometry | Deterministic contiguous session blocks, panel grouping, evidence-bearing assignments, invalid-fold diagnosis and formal failure | `test_strict_purged_kfold.py`; panel and coverage properties | verified |
| 6–7, 18, 34, 38–40 — OOS facts and metrics | Raw immutable OOS observations retain identities; named/versioned metrics expose overall, per-fold, observation coverage, and fold coverage | `test_evaluation.py`; `test_review_regressions.py` | verified |
| 9–12, 21, 29–30, 44 — PIT eligibility | Decision Time, row/per-feature availability and PIT source state gate formal scoring; errors redact values | `test_point_in_time_scoring.py` | verified |
| 19–20, 22, 26, 29–30, 51 — canonical dataset evidence | Authoritative ordered Session Axis, stable IDs, finite-value reject policy, typed conversion failures, deeply immutable configuration, read-only arrays and sensitivity-preserving digests | `test_split_plan.py`; `test_review_regressions.py`; digest property | verified |
| 23–25, 47–48 — explicit optional pandas input | Explicit columns/index levels, MultiIndex preservation, timezone/index/session rejection, hidden attrs ignored, core import canary | `test_pandas_adapter.py` | verified |
| 31–33, 35–37 — Fold-Local lifecycle | Fresh factory objects, sequential train-only transformations, arbitrary fit/predict protocols and whole-run abort | `test_evaluation.py`; `test_fail_closed_execution.py` | verified |
| 43, 45, 54–56 — later evidence channels and extension boundaries | Results remain `model-selection`; CPCV, causal walk-forward, holdout reuse governance and nested HPO remain explicit non-capabilities | `README.md`; `docs/interface-contract.md`; public evidence-channel assertions | boundary verified |
| 46, 49–50, 53 — narrow core and acceptance | NumPy runtime core, Hypothesis invariants, preserved regression counterexamples and repeatable project commands | `test_properties.py`; regression suites; `docs/verification.md` | verified |

The regression suite explicitly preserves: a label interval crossing a fold boundary, a
disjoint protected-interval min/max over-purge counterexample, session rather than
calendar-day Embargo, same-session panel leakage, factory object reuse, and
latest-revision feature leakage.
