# Financial Validation Context

This context defines the language used to reason about leakage-safe financial time-series validation. It separates information-aware model selection from causal deployment evidence.

## 文档与实现分层

本文件维护领域词汇和当前事实，不执行验证算法。仓库中的信息按以下层次组织：

| 层次 | 主要文件 | 职责 |
|---|---|---|
| Agent 发现 | `SKILL.md`, `agents/openai.yaml` | 触发条件、执行顺序和薄适配 |
| 用户说明 | `README.md` | 中文介绍、逻辑、输入和使用流程 |
| 长期治理 | `PROGRAM.md` | 使命、边界、原则和成功门槛 |
| 领域上下文 | `CONTEXT.md` | 术语、语义和当前验收事实 |
| 详细契约 | `references/`, `docs/interface-contract.md` | 请求、结果、接口与解释规则 |
| 可执行实现 | `src/purged_kfold_validation/` | splitter、evaluator、feature governance、Holdout 和 CLI |
| 验证证据 | `tests/`, `docs/verification.md`, `docs/evidence/` | 回归、属性测试、构建和真实数据回执 |

Markdown 为 Agent 和维护者提供规则与上下文；运行时行为始终以 Python 实现、schema 和测试为准。若文档与可执行证据冲突，应先停止发布并修复漂移，不得让 Agent 自行选择更宽松的解释。

## Language

**Information Interval**:
The inclusive interval from the first information attributed to a sample through the final time needed to determine its label.
_Avoid_: Label window, horizon rows

**Trading Session**:
An ordered market-calendar unit used to group samples and measure session-based gaps across holidays and special trading days.
_Avoid_: Calendar day, row number

**Panel Session Group**:
All asset samples belonging to the same Trading Session and therefore assigned to the same validation side.
_Avoid_: Date rows, asset fold

**Session Axis**:
The authoritative ordered sequence of Trading Sessions against which sample membership, interval endpoints, and session-based distances are validated.
_Avoid_: Datetime index, row order

**Validation Dataset**:
The canonical collection of stable sample identities, Session Axis membership, Information Intervals, features, targets, and optional asset identities presented for validation.
_Avoid_: Input DataFrame, training frame

**Decision Time**:
The time at which a prediction represented by one sample is assumed to be made and against which feature availability is judged.
_Avoid_: Row timestamp, event start

**Feature Availability**:
The latest time at which all information represented by a feature value was actually available to the decision maker.
_Avoid_: Feature date, observation time

**Feature Definition**:
An immutable declaration of one uploaded feature's semantic name, ordered source
fields, source vintage, transformation/code identity, parameters, lookback, revision
policy, target dependency, and computation lifecycle.
_Avoid_: Column label, inferred formula

**Feature Manifest**:
The ordered collection of Feature Definitions bound to the PIT source bundle and then
to Validation Dataset and OOS evidence by a deterministic digest.
_Avoid_: Feature list, DataFrame schema

**Governed Feature Dataset**:
A Validation Dataset whose per-feature availability matrix and manifest have passed
the uploaded-feature lifecycle rules and produced a redacted governance receipt.
_Avoid_: Trusted values, leak-free by assertion

**Transformer Spec**:
The versioned code/parameter identity paired one-to-one with a fold-local transformer
factory so preprocessing state participates in the evaluation identity.
_Avoid_: Pipeline name, preprocessing comment

**Feature Upload Contract**:
The closed, versioned manifest and mapping documents that bind a local CSV/Parquet
matrix to column roles, Session Axis, PIT Snapshot, per-feature availability, lineage,
and bounded resource policy before evaluation.
_Avoid_: Automatic schema detection, arbitrary upload

**Installed Upload Command**:
The distribution-owned console/module boundary that exposes audit, evaluation, schema
discovery, and safe example materialization from the same packaged implementation.
_Avoid_: Repository script copy, unversioned helper

**PIT Snapshot**:
A point-in-time source snapshot that preserves the values and publication state available at its declared historical cutoff rather than later revisions.
_Avoid_: Historical export, latest data

**Pandas Adapter**:
A boundary translator that constructs a Validation Dataset from explicitly mapped pandas columns and index levels without inferring authoritative metadata.
_Avoid_: DataFrame-first API, automatic schema detection

**Purge**:
The exclusion of a training sample whose Information Interval overlaps a test Information Interval.
_Avoid_: Drop adjacent rows, trim

**Embargo**:
The post-test exclusion zone applied after each contiguous test block in two-sided validation.
_Avoid_: Walk-forward gap, purge window

**Pre-Test Gap**:
The exclusion zone immediately before a causal test block, used together with Purge when training is restricted to the past.
_Avoid_: Embargo

**Splitter**:
A component that produces validation assignments and their provenance without fitting transformations or estimators.
_Avoid_: Evaluator, backtest

**Fold Assignment**:
An immutable, evidence-bearing validation assignment containing stable sample identities, array positions, contiguous test blocks, exclusion summaries, and input/configuration provenance.
_Avoid_: Index tuple, fold indices

**Exclusion Summary**:
A compact account of how many candidate training samples were removed by Purge, Embargo, or Pre-Test Gap for one Fold Assignment.
_Avoid_: Dropped rows, filter count

**Exclusion Trace**:
An optional audit view that records the exclusion reason for each removed sample without changing the Fold Assignment.
_Avoid_: Debug log, warning list

**Split Plan**:
A read-only diagnostic projection of every requested Fold Assignment, including invalid candidates and the evidence explaining why they cannot be evaluated.
_Avoid_: Evaluation result, skipped folds

**Invalid Fold**:
A requested validation assignment that violates declared sample, session, coverage, or path-completeness constraints and therefore cannot contribute a score.
_Avoid_: Empty fold, warning-only fold

**Leakage-Safe Evaluator**:
The canonical high-level validation boundary that creates fresh fold-scoped transformations and estimators, executes Splitter assignments, and returns out-of-sample evidence.
_Avoid_: Splitter, global training loop

**Fold-Local**:
A learned operation whose fit state is derived only from the training side of one validation assignment.
_Avoid_: Precomputed, globally fitted

**Fold Factory**:
A provider that creates a new unfitted estimator or transformation for one validation assignment, preventing learned state from crossing fold boundaries.
_Avoid_: Shared model, fitted object, reusable instance

**OOS Ledger**:
The authoritative collection of path-scoped out-of-sample predictions, keyed by run, sample, split, combination, path, and model identity.
_Avoid_: Averaged predictions, score table

**Evidence Channel**:
A validation purpose whose observations and claims remain distinct from other purposes, such as model selection, CPCV robustness, causal walk-forward, or final holdout confirmation.
_Avoid_: Validation type, combined score

**Evaluation Protocol**:
An immutable declaration of frozen data boundaries, feature/model configuration, search policy, split specification, and metrics that authorizes final holdout evaluation.
_Avoid_: Experiment config, run parameters

**Holdout Receipt**:
An append-only record that binds one authorized holdout evaluation to its Evaluation Protocol, holdout dataset, result, and evaluation time.
_Avoid_: Score file, run log

**Reused Holdout**:
A holdout dataset observed before the current final evaluation claim and therefore ineligible to provide untouched confirmation, even under a new protocol identity.
_Avoid_: Refreshed holdout, second test run

**Derived Metric**:
A versioned calculation projected from an OOS Ledger for a declared split, path, or Evidence Channel.
_Avoid_: Stored truth, headline score

**CPCV Combination**:
One choice of test groups from the complete set of chronological groups in combinatorial purged cross-validation.
_Avoid_: CPCV path

**CPCV Path**:
A deterministic sequence of path-scoped out-of-sample predictions assembled from CPCV Combinations so that every chronological group appears exactly once in that path.
_Avoid_: Fold list, averaged predictions

**Path Decomposition**:
The deterministic assignment of every CPCV combination/test-group occurrence to complete CPCV Paths while preserving group coverage and combination identity.
_Avoid_: Path shuffle, fold concatenation

**Causal Walk-Forward**:
A validation sequence in which every training Information Interval precedes its test block and time advances without future training groups.
_Avoid_: Purged K-Fold, chronological K-Fold

**Candidate Strategy**:
One fully specified strategy and parameter configuration whose session-aligned net returns form a single selection trial.
_Avoid_: CPCV Path, cost scenario, model prediction

**Strategy Return Matrix**:
An immutable Trading Session by Candidate Strategy matrix of finite net returns used as the seam between caller-owned strategy generation and selection-overfitting evidence.
_Avoid_: Feature matrix, OOS Ledger, backtest summary

**Temporal Supervised Sample**:
One Asset at one Decision Session, with an ordered causal lag sequence, a future label, and an Information Interval spanning every dependency needed by both.
_Avoid_: Sequence row, sliding window

**Temporal Model Case**:
One fixed trainable estimator identity and fresh fold factory evaluated without changing its configuration across validation channels.
_Avoid_: Tuned winner, model family search

**Unsafe Overlap Canary**:
A deliberately unprotected validation channel retained only to reveal interval overlap and score optimism that safe evidence must exclude.
_Avoid_: Baseline approval, valid cross-validation

**Validation Optimism Gap**:
The safe-channel error minus unsafe-channel error for the same fixed model and data, disclosed as a diagnostic rather than a pure causal estimate of leakage bias.
_Avoid_: Leakage amount, guaranteed overfit penalty

**TSMOM Reference Family**:
The built-in non-cross-sectional benchmark family in which each asset's position direction is derived only from that asset's own lagged return history.
_Avoid_: Cross-sectional momentum, production strategy

**Selection Overfitting**:
The failure mode in which the best in-sample Candidate Strategy owes its apparent advantage to searching many trials and does not preserve its relative rank out of sample.
_Avoid_: Information leakage, poor absolute return

**CSCV/PBO Evidence**:
The symmetric complementary-slice evidence that measures how often an in-sample winning Candidate Strategy falls below the out-of-sample candidate median.
_Avoid_: CPCV Path evidence, p-value

**Deflated Sharpe Evidence**:
The probability evidence that a selected Sharpe exceeds a multiple-trial benchmark after accounting for sample length and non-normal returns.
_Avoid_: Raw Sharpe, profitability guarantee

**Acceptance Policy**:
An immutable, digest-bound set of cost scenarios and metric thresholds registered before strategy benchmark evidence is observed.
_Avoid_: Post-hoc cutoff, tuned pass criteria

**Research Gate**:
A decision over pre-registered PBO, Deflated Sharpe, CPCV path-tail, causal Walk-Forward, and cost-stress evidence that remains separate from final deployment authorization.
_Avoid_: Production approval, attractive backtest

**Production Gate**:
A governed decision that requires a passing Research Gate plus eligible one-time Untouched Holdout evidence when the Acceptance Policy requires it.
_Avoid_: Research score, local test pass

**Evidence Gap**:
A machine-readable statement of evidence still missing or ineligible for a requested claim, such as an unrun or reused strategy Holdout.
_Avoid_: Warning-only failure, assumed evidence

**Selection Regret**:
The difference between a causally selected candidate's test performance and the hindsight-best candidate performance on the same test block.
_Avoid_: Trading loss, prediction error

**Untouched Holdout**:
A final evaluation interval excluded from model, feature, threshold, and hyperparameter decisions until the design is frozen.
_Avoid_: Validation fold, reusable test set

**Temporal Forward Protocol**:
An immutable declaration binding consumed development evidence, one selected model, a strictly future start, label maturity, sufficiency gates, and acceptance checks before any eligible forecast is recorded.
_Avoid_: New backtest config, renamed Holdout

**Prediction Receipt**:
An append-only forecast identity durably recorded before its declared future label becomes available, without a target or raw feature payload.
_Avoid_: Replayed prediction, score row

**Matured Label Settlement**:
An append-only target binding created after label availability and referencing exactly one prior Prediction Receipt.
_Avoid_: Backfilled forecast, mutable result row

**Forward Evidence Ledger**:
The local ordered collection of Prediction Receipts and Matured Label Settlements used to project redacted future-evidence status and metrics.
_Avoid_: Historical OOS Ledger, editable CSV

**Evidence Maturity**:
The governed state `WAITING_FOR_FUTURE_DATA`, `COLLECTING`, `READY_FOR_REVIEW`, or `FAIL`, determined by pre-registered sample sufficiency and metric checks.
_Avoid_: Deployment approval, manual confidence label

## Current five-year acceptance fact

The governed PandaData run covering 2021-06-18 through 2026-06-18 retained 88,417
eligible observations across 81 assets and 1,174 Trading Sessions. All three structural
channels reported zero retained Information Interval overlaps. A final 252-session
Holdout was consumed exactly once; its MSE is evidence about the frozen intercept-only
baseline, not a profitability or deployment claim. See
`docs/evidence/pandadata-five-year-release-gate-20260802.md`.

## Current Agent consumption fact

The root `SKILL.md` and `purged-cv-skill` command provide a thin Agent-neutral consumption
layer over the same installed engine. One versioned request JSON produces one redacted,
versioned result envelope. The adapter contains no splitter, evaluator, feature-governance,
or Holdout logic and does not change the established evidence-channel boundaries.

## Current strategy-selection acceptance fact

Version 0.7.1 accepts a generic Strategy Return Matrix and supplies a built-in 32-candidate
TSMOM Reference Family. CSCV/PBO, Deflated Sharpe Evidence, CPCV selected paths, causal
Walk-Forward Selection Regret, four offline cost scenarios, package schemas, and the installed
command passed 150 local tests and an isolated-wheel canary. A subsequent authorized
PandaData run covered 15 assets and 1,210 common Sessions from 2021-08-03 through
2026-08-03. PBO and positive CPCV/Walk-Forward gates passed, while Deflated Sharpe
Evidence remained below the example 0.95 production gate. This is promising but
insufficient strategy evidence, not a failure of the validation tool. The versioned
Acceptance Policy now emits separate validation-tool, Research Gate, and Production Gate
statuses, exact failed checks, Holdout Evidence Gaps, and a non-guaranteed DSR track-record
approximation without retuning the registered candidate family.

## Current trainable temporal-model fact

Version 0.8.0 evaluates fixed NumPy Ridge, LightGBM, and PyTorch LSTM estimators through
unsafe shuffled K-Fold, chronological no-purge, Purged K-Fold, Purged K-Fold plus Embargo,
CPCV, and causal Walk-Forward. The governed lag-20/T+5 PandaData comparison contains
17,775 observations across 15 assets and 1,185 Decision Sessions. Unsafe shuffled folds
retain interval overlap while all formal channels retain zero overlap. Complete
Information Interval Purge already covers the registered 20-Session Embargo zone, so the
Embargo has no incremental exclusions in that run. This is structural validation evidence,
not model profitability, final Holdout, or production authorization.

## Current temporal forward-evidence fact

Version 0.9.0 freezes the observed v0.8 development comparison and selects its lowest
causal Walk-Forward MSE case, LightGBM lag-20/T+5, without further tuning. A new
Prediction Receipt must be persisted before label availability and can only be paired
with a Matured Label Settlement afterwards. The initial PandaData protocol starts after
the 2026-08-03 development label boundary and is honestly
`WAITING_FOR_FUTURE_DATA`. It requires 252 matured Decision Sessions, 3,000 settled
observations, and 8 assets before metric checks can produce `READY_FOR_REVIEW` or `FAIL`.
Neither state grants production authorization.
