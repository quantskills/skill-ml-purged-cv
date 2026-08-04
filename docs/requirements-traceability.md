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
| 43, 45, 54–56 — evidence channels and extension boundaries | Purged K-Fold remains `model-selection`; Holdout confirmation is a separate terminal channel; nested HPO remains an explicit non-capability | `README.md`; `docs/interface-contract.md`; public evidence-channel assertions | verified |
| 46, 49–50, 53 — narrow core and acceptance | NumPy runtime core, Hypothesis invariants, preserved regression counterexamples and repeatable project commands | `test_properties.py`; regression suites; `docs/verification.md` | verified |

The regression suite explicitly preserves: a label interval crossing a fold boundary, a
disjoint protected-interval min/max over-purge counterexample, session rather than
calendar-day Embargo, same-session panel leakage, factory object reuse, and
latest-revision feature leakage.

## Walk-Forward and benchmark slice

| Spec story / invariant | Implemented behavior | Observed verification | Status |
|---|---|---|---|
| WF 1–3, 6–7 | Past-only expanding/sliding assignments, exact Purge, noncausal information exclusion, Session Axis Pre-Test Gap, invalid-fold failure, causal channel propagation | `test_causal_walk_forward.py` | verified |
| WF 4–5 | Deterministic bounded history and indivisible panel sessions | `test_causal_walk_forward.py` | verified |
| Benchmark 8–9, 12 | Explicit offline PandaAI mapping, shifted per-asset horizons, PIT declaration, duplicate rejection, local parquet CLI | `test_pandaai_benchmark.py` | verified |
| Benchmark 10–11 | Same dataset/model/metric across four channels; independent retained-overlap audit without assuming score order | `test_pandaai_benchmark.py`; `docs/evidence/pandaai-benchmark-20260801.md` | verified |

## CPCV slice

| Spec story / invariant | Implemented behavior | Observed verification | Status |
|---|---|---|---|
| CPCV 1–3 | Complete lexicographic combinations, multi-block Purge/Embargo, adjacent-block merging, and panel-session grouping | `test_cpcv.py` combination and boundary cases | verified |
| CPCV 4 | General deterministic Path Decomposition with occurrence uniqueness, complete group coverage, and distinct paths within combinations | `test_cpcv.py` worked `6,2`, `5,3`, and all small `3≤N≤8` configurations | verified |
| CPCV 5 | Typed invalid folds and explicit combination budget fail closed | `test_cpcv.py` invalid-fold and budget cases | verified |
| CPCV 6–8 | Raw repeated OOS facts retain combination/group/path identity; unique coverage and per-path metrics derive from complete paths | `test_cpcv.py` evaluator path evidence | verified |
| CPCV 9 | Existing splitters, adapters, and benchmark behavior remain compatible | complete repository test suite | verified |
| Real-data structural probe | Five-asset PandaAI cache produced 15 combinations, 5 complete paths, and zero retained interval overlaps | `docs/evidence/cpcv-pandaai-probe-20260801.md` | verified |

## Arbitrary-feature governance slice

| Spec story / invariant | Implemented behavior | Observed verification | Status |
|---|---|---|---|
| Feature 1–3 | Ordered immutable definitions/manifests, source and transform lineage, per-feature availability, PIT bundle binding, deterministic receipt | `test_feature_governance.py` | verified |
| Feature 4 | Precomputed stateful, target-derived, latest-revision, late, ambiguous-availability, duplicate, and source-mismatched inputs fail closed | `test_feature_governance.py` | verified |
| Feature 5 | Ordered `TransformerSpec` values bind learned preprocessing into run and OOS identities while fresh fold-local factories remain mandatory | `test_feature_governance.py`; `test_evaluation.py`; `test_fail_closed_execution.py` | verified |
| Feature 6 | Explicit pandas upload wrapper matches manifest names/order and preserves optional dependency isolation | `test_feature_upload_pandas.py`; core import canary | verified |
| Feature 7 | Receipts expose counts and digests rather than feature values | `test_feature_governance.py` | verified |

## Governed local-upload delivery slice

| Spec invariant | Implemented behavior | Observed verification | Status |
|---|---|---|---|
| Upload 1–2 | Closed version-1 manifest/mapping schemas and bounded CSV/Parquet adapter delegate to canonical governance without inference | `test_feature_upload_cli.py`; `config/feature-upload/*.schema.json` | verified |
| Upload 3 | Audit-only command returns redacted receipt and never fits a model | `test_feature_upload_cli.py` audit success/rejection cases | verified |
| Upload 4 | Evaluate runs the audited dataset through Purged K-Fold, CPCV, and causal Walk-Forward with one fold-local ridge baseline | `test_feature_upload_cli.py` three-channel case | verified |
| Upload 5 | File, row, feature, and CPCV-combination limits fail closed before unbounded evaluation | `test_feature_upload_cli.py`; `FeatureUploadLimits` | verified |
| Upload 6 | Raw and stationary examples pass; declared target leakage and late availability fail | `test_feature_upload_cli.py` examples and adversarial cases | verified |
| Upload 7 | High-level comparisons pass ordered transformer factories/specs to every channel and bind spec digests into report identity | `test_effectiveness_comparison.py` transformer identity regression | verified |

## Installable upload CLI slice

| Spec invariant | Implemented behavior | Observed verification | Status |
|---|---|---|---|
| Dist CLI 1 | Console, module, and legacy script use one package-owned implementation | module/script equivalence in `test_feature_upload_cli.py`; isolated console canary | verified |
| Dist CLI 2 | Both schemas and all examples are packaged and reproduce canonical repository behavior | schema/example subprocess cases; wheel content inspection | verified |
| Dist CLI 3 | Example materialization uses fixed filenames and refuses known conflicts before writing | no-overwrite regression in `test_feature_upload_cli.py` | verified |
| Dist CLI 4 | Schema/example discovery preserves the pandas-optional base boundary; audit gives safe extra-install guidance | import-blocking canaries in `test_installable_cli_contract.py` | verified |
| Dist CLI 5 | Metadata declares version, console script, upload extra, and explicit package data | `test_installable_cli_contract.py`; Twine and wheel inspection | verified |
| Forward 1–7 | Frozen future boundary, prediction-before-label receipt, append-only settlement, sufficiency/metric state machine, redacted status, and Agent-neutral CLI | `test_temporal_forward.py`; `test_forward_cli.py`; ADR 0012; v0.9.0 wheel canary | verified; real state waiting for future data |
| Dist CLI 6 | Installed console prints schema, exports raw example, audits it, and module entry prints mapping schema | isolated venv receipt in `docs/evidence/installable-upload-cli-20260802.md` | verified |

## Agent-neutral Skill delivery slice

| Product invariant | Implemented behavior | Observed verification | Status |
|---|---|---|---|
| Skill 1 | Root `SKILL.md` exposes one concise portable workflow while detailed contracts load progressively | Skill official validator; `test_project_contract.py` | verified |
| Skill 2 | One request JSON delegates to the canonical CLI and preserves its result in a standard redacted envelope | `test_agent_skill_cli.py` request and demo cases | verified |
| Skill 3 | Unknown fields, invalid action options, bad ranges, engine timeout, stderr, or invalid JSON fail closed | `agent_cli.py`; rejection regressions in `test_agent_skill_cli.py` | verified |
| Skill 4 | Request/result schemas and minimal example are installed package resources | package metadata contract; schema/example subprocess cases | verified |
| Skill 5 | README provides one install command, one smoke-test command, one minimal request, and standard JSON shape | `README.md`; isolated-wheel verification pending final gate | implemented |

## Release-hardening and governed-Holdout slice

| Spec invariant | Implemented behavior | Observed verification | Status |
|---|---|---|---|
| Resource 1 | CSV row ceilings use a bounded parse; no unrestricted full-file read precedes the row gate | `test_csv_row_budget_uses_bounded_read_before_full_parse` | verified |
| Resource 2 | Parquet footer rows, columns, and declared uncompressed bytes are gated before table materialization | `test_parquet_footer_rejects_rows_before_table_materialization`; `FeatureUploadLimits` | verified |
| Holdout 1–2 | Protocol freezes training/Holdout/model/transformer/metric/search/split identities and requires a strictly future disjoint Holdout | `test_holdout.py` protocol and binding cases | verified |
| Holdout 3–5 | Exclusive pre-fit claim makes success and failure one-attempt; fitting sees training only; persisted receipt is redacted | `test_holdout.py` lifecycle, reuse, failure-consumption, and persisted-receipt cases | verified |
| Robustness 1–3 | Comparable fixed models receive cross-regime rank summaries and pairwise Spearman evidence; adversarial reversal is unstable | `test_ranking_stability.py` | verified |

## Five-year PandaData release gate

| Spec invariant | Implemented behavior | Observed verification | Status |
|---|---|---|---|
| Real gate 1 | Select an exact trailing five-calendar-year window from a local PandaData daily cache without credentials or network access | `scripts/evaluate_pandadata_five_year_release_gate.py`; redacted source digest in five-year evidence | verified |
| Real gate 2 | Continuous-contract governance precedes splitting; label and Information Interval roll crossings remain excluded | 102,605 input rows reduced to 88,417 eligible observations with governance counts recorded | verified |
| Real gate 3 | Purged K-Fold, CPCV, and causal Walk-Forward compare one fixed ridge baseline and independently audit retained interval overlap and training sufficiency | all channels report zero overlap; minimum training sizes and CPCV paths in `docs/evidence/pandadata-five-year-release-gate-20260802.md` | verified |
| Real gate 4 | Three chronological development regimes rank one frozen candidate set before final Holdout access | identical ranking in all regimes; minimum pairwise Spearman 1.0 | verified |
| Real gate 5 | Final 252-session Holdout is claimed before fitting and cannot be queried again after success or failure | persisted receipt `245bf378...1355bd`; recovery reused only the receipt, never the Holdout | verified |
| Real gate 6 | Evidence distinguishes leakage controls from feature/model effectiveness and profitability | intercept-only mean wins by a small margin; Holdout reports MSE only; no profitability/deployment claim | verified |

## Time-series strategy selection benchmark slice

| Spec invariant | Implemented behavior | Observed verification | Status |
|---|---|---|---|
| TS Bench 1 | Generic immutable Session × Candidate Strategy gross/net return and turnover seam with deterministic digests | `test_time_series_strategy_benchmark.py` input and determinism cases | verified |
| TS Bench 2 | Built-in 32-candidate, per-asset TSMOM family uses lagged prices/volatility and prior frozen weights | lag-causality and registered-grid tests | verified |
| TS Bench 3 | Offline turnover cost is applied to the same gross-return trajectory and cost scenarios remain separate from trials | cost monotonicity and scenario-isolation tests | verified |
| TS Bench 4 | CSCV/PBO and DSR report candidate trial count, rank-failure probability, non-normal moments, and deflated threshold | generic matrix and deliberate rank-reversal tests | verified |
| TS Bench 5 | Existing deterministic CPCV decomposition reconstructs complete selected-candidate paths and reports median/P10/worst/IQR | five-path coverage test | verified |
| TS Bench 6 | Expanding Walk-Forward selects only on prior Sessions and reports hindsight regret | future-mutation causality test | verified |
| TS Bench 7 | Installed JSON command accepts pickle-free NPZ inputs, publishes schemas, and fails closed on wrong archive contracts | `test_strategy_benchmark_cli.py` | verified |
