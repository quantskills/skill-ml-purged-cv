---
name: skill-ml-purged-cv
description: 审计任意金融特征、可训练时序模型或候选策略收益，并执行防泄漏的 Purged K-Fold、Embargo、CPCV、Causal Walk-Forward、PBO、DSR、Governed Holdout 与 Temporal Forward Evidence。用于检查未来函数、信息区间重叠、特征可用时间和血缘、Fold-Local 预处理、CPCV Path 稳健性、策略选择过拟合、预测是否在标签成熟前登记，以及在接受金融模型或策略前生成结构化验证证据。
---

# 金融时序防泄漏验证

始终调用已安装的 `purged_kfold_validation` 实现。不得在本 Skill 或任何 Agent adapter 中复制 splitter、evaluator、特征治理或 Holdout 算法。

## 快速确认

先执行 `purged-cv-skill demo`。命令不可用时，要求安装当前仓库的 v0.9.0 wheel 或带 `upload` extra 的 Git 版本；不要自行重写验证算法。成功回执必须包含 `status=success`、当前 engine version 和底层 `authoritative_cli_result`，但不能据此声称模型盈利或可部署。

## 执行顺序

1. 要求调用方提供本地数据、`manifest.json` 和 `mapping.json`。
2. 先运行 audit，不拟合模型。
3. 遇到 PIT 缺失、特征晚于 Decision Time、目标派生特征、血缘不一致、Invalid Fold、CPCV Path 不完整或训练量不足时立即停止。
4. audit 通过后，才运行正式 evaluate。
5. 保留底层 `authoritative_cli_result`，把 Agent 的说明标记为派生解释。
6. 报告样本量、资产数、Session 数、各类排除数、保留区间重叠、fold/path 指标、中位数、最差分位数、离散程度和通道差异。
7. 当请求涉及已经冻结的未来表现时，先登记 Prediction Receipt，标签成熟后再结算；不得用历史重放替代 Forward Evidence。

使用统一入口：

```text
purged-cv-skill run --request <request.json>
```

需要最小样例或机器协议时执行：

```text
purged-cv-skill example --output-dir <empty-directory>
purged-cv-skill schema --kind request
purged-cv-skill schema --kind result
```

## 选择证据通道

- 比较特征或模型：使用 Purged K-Fold。
- 观察多条历史路径的指标和排名稳定性：增加 CPCV。
- 模拟只使用过去数据的持续训练：使用 Causal Walk-Forward。
- 设计完全冻结后的最终确认：使用一次性 Governed Holdout。
- 检查已冻结时序模型在真正未来数据上的表现：使用 Temporal Forward Evidence，先登记预测、标签成熟后再结算。
- 检查从多个时序策略参数中挑选赢家是否过拟合：使用候选净收益矩阵的 PBO、DSR、CPCV 选参路径和 Walk-Forward selection regret。

Purged K-Fold 与 CPCV 应并存。CPCV 不能替代 Walk-Forward，Holdout 不能参与调参。

## 时序策略选择过拟合

当用户问的是“这个时序策略或参数搜索是否过拟合”，不要只运行固定 Ridge/MSE 基线。优先要求用户提供 `Trading Session × Candidate Strategy` 的净收益矩阵；没有可复现策略时，可以使用内置 TSMOM 参考族作为标准 canary。

```text
purged-cv-strategy demo
purged-cv-strategy run --request <strategy-request.json>
purged-cv-strategy schema --kind request
purged-cv-strategy schema --kind result
```

对 `benchmark-tsmom`，必须优先读取 `report.acceptance`，并按以下顺序回答用户：

1. 报告 `validation_tool_status`，明确它只说明验证链路是否有效；
2. 报告 `research_gate_status` 和所有失败的 `checks`；
3. 报告 `production_gate_status`，不得把缺失 Holdout 的 `INCONCLUSIVE` 改写为通过；
4. 报告 `dsr_track_record_gap`，同时说明它是 constant-distribution approximation；
5. 报告 `evidence_gaps`，尤其是未运行或已复用的策略 Holdout。

不得根据已经看到的 benchmark 结果放宽 `StrategyAcceptancePolicy`、新增候选再复用同一历史数据，或为了获得 `PASS` 继续调 TSMOM 参数。默认 3 bps 为主情景、5 bps 为压力情景；缺少注册情景时应 fail closed。

当用户明确询问 Purged K-Fold / Embargo 是否对“可训练时序模型”生效时，运行 `benchmark-temporal-models`，不要用 TSMOM 收益矩阵或固定 Ridge 单通道代替。默认同时运行 `numpy-ridge`、`lightgbm` 和 `torch-lstm`，并报告六个通道的 MSE、overlap、CPCV Path 分布、最小训练量和 unsafe optimism gap。

解释顺序必须是：

1. 先报告全部安全通道的 overlap 是否为 0；
2. 再报告 unsafe/chronological 反例的 overlap；
3. 再比较相同模型的通道 MSE，明确分数差不是纯泄漏因果量；
4. 说明 Embargo 是产生了额外排除，还是在完整 interval-aware Purge 后没有增量；
5. 最后报告 `production_authorization`，没有一次性 Holdout 时不得写成可上线。

`analyze-return-matrix` 接受调用方自己的候选净收益，`benchmark-tsmom` 接受严格滞后的密集价格面板。成本情景不得计入 trial 数。PBO 是选择排名失败概率，不是 p-value；DSR、CPCV 尾部、Walk-Forward 和冻结 Holdout 必须联合解释。

## Temporal Forward Evidence

当历史数据已经参与模型、特征或阈值选择时，不得把重新切分或重新下载同一时期的数据称为 Holdout。使用 `purged-cv-forward init|record|settle|status`：先冻结开发报告、数据、模型和未来起点；再在标签可得前登记 Prediction Receipt；只有标签成熟后才能用 receipt digest 追加结算。

状态只能按预注册门槛从 `WAITING_FOR_FUTURE_DATA` 进入 `COLLECTING`，证据充分后进入 `READY_FOR_REVIEW` 或 `FAIL`。不得把 `READY_FOR_REVIEW` 改写成生产授权；所有回执的 `production_authorization` 均为 `NOT_AUTHORIZED`。必须同时披露 `attestation_scope`：本地追加式账本没有外部公证，不能描述成不可抵赖时间证明。若调用方提供的是已经知道 target 的历史文件，应拒绝将其登记为 Forward Evidence，并建议仅作为开发回放。

## 保持时间边界

- 根据显式 Information Interval 重叠执行 Purge，不使用固定删行代替。
- 按 Trading Session 在每个连续测试块之后执行 Embargo。
- 把 Walk-Forward Pre-Test Gap 与 Embargo 分开解释。
- 保持同一 Session 的多资产样本不可拆分。
- 通过全新 factory 在每一折训练侧拟合 transformer 和 estimator。

## 按需读取详细资料

创建特征请求、选择参数或解释结果时读取 [references/agent-contract.md](references/agent-contract.md)；运行策略选择审计时读取 [references/strategy-benchmark-contract.md](references/strategy-benchmark-contract.md)；登记未来预测或结算标签时读取 [references/forward-evidence-contract.md](references/forward-evidence-contract.md)。领域术语读取 `CONTEXT.md`，完整人类使用说明读取 `README.md`。不要把这些说明复制回本文件。

## 禁止事项

不得索取数据供应商凭据、登录远程系统、执行上传代码、修改源数据、跳过失败折、复用预拟合对象，或把零重叠、MSE、IC、diagnostic Sharpe、PBO、DSR 或 TSMOM benchmark 描述成盈利或部署证明。
