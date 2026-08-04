# Agent Skill 真实期货数据验收（2026-08-03）

## 结论

`purged-cv-skill` 的标准 `request.json -> authoritative_cli_result` 链路已在真实期货日频面板上通过。运行覆盖输入治理、Purged K-Fold、CPCV Path、Causal Walk-Forward、训练充分性门槛和红线 JSON 回执。

**本次运行是单一 Fold-Local Ridge 的真实数据执行 canary，不是时序过拟合 benchmark。** 它没有运行普通随机 K-Fold、Chronological No-Purge 等不安全对照组，因此不能量化泄漏造成的分数虚高，也不能仅凭一个模型判断模型选择过拟合。

本验收只证明 Skill 可以发现输入错误、拒绝不充分的验证配置，并在合格输入上执行安全验证通道；它不证明基线模型可盈利、模型没有过拟合、外部历史数据声明真实或已获得生产部署授权。

## 数据身份

- 来源：本地只读 `futures_offline_store.duckdb` 的 `features` 表；
- 来源 SHA-256：`d679d51c90ba56ab93d5d9ab25eb190242a31f137d5639740eeccf6413025e5f`；
- 时间窗口：2021-01-05 至 2025-12-30；
- 合格观测：82,850；
- 资产：81；
- Trading Session：1,210；
- 特征：`pct_chg_lag1` 至 `pct_chg_lag5`，以及滞后一 Session 的 `volume_z`、`open_interest_z`；
- 标签：每个资产下一可用 Trading Session 的真实 `pct_chg`；
- 治理后的数据摘要：`7d368bea226868d1e8c8fdce53850aa1445a059a6ae33fb199bdf6f0cb74dade`；
- manifest 摘要：`2257fa6c8a29573345b9a713458e08b97f823a4e9c6e54d98c6e2f0feb6709e2`。

真实数据和绝对本地路径未写入本回执，也不得进入版本库。

## 验证配置

- CPCV：`N=6, k=2`，15 个组合，5 条完整 Path；
- Embargo：20 个 Session；
- Causal Walk-Forward：3 段，Pre-Test Gap 为 5 个 Session；
- Fold-Local estimator：Ridge，`alpha=1.0`；
- 最低训练门槛：10,000 条观测、252 个 Session、20 个资产；
- 公共比较样本：62,969 条、906 个 Session；
- 成功请求摘要：`e59db356f54bda9b4c4f12a81039e2fc9244b7f4475707c099e0b9465214fb6d`；
- 报告摘要：`7cdd9b465ea960bd7b53e5b2be0abd8ffa7cd78873921c27879b071922b81724`。

## Fail-closed 探针

1. 输入先按资产再按日期排列时，运行以 `TemporalValidationError: samples must be ordered by session_axis` 拒绝；
2. 使用 5 段 Walk-Forward 且保持 252 Session 门槛时，首个训练窗只有 200 个 Session，运行以 `InvalidFoldError` 拒绝；
3. 改为全局 Session 排序和 3 段 Walk-Forward 后，保持 252 Session 门槛不变，运行成功。

这些拒绝说明 Skill 没有静默重排输入或跳过无效 Fold。

## 单模型执行结果（非过拟合 benchmark）

三个证据通道的保留 Information Interval 重叠数均为零，所有 Fold/Combination 均达到声明的训练充分性门槛。

| 通道 | 覆盖率 | 最小训练观测 | 最小训练 Session | 最小训练资产 | 保留重叠 |
|---|---:|---:|---:|---:|---:|
| Purged K-Fold | 100.00% | 66,726 | 985 | 81 | 0 |
| CPCV | 100.00% | 51,783 | 760 | 78 | 0 |
| Causal Walk-Forward | 76.00% | 19,551 | 299 | 69 | 0 |

在公共 62,969 条样本上的指标为：

| 通道 | MSE | Cross-sectional Spearman IC | Diagnostic Sharpe |
|---|---:|---:|---:|
| Purged K-Fold | 0.0001369365 | 0.011579 | 0.579069 |
| Causal Walk-Forward | 0.0001373619 | 0.006328 | 0.345030 |

CPCV 五条公共样本 Path 的分布为：

| 指标 | 中位数 | P10 | 最差值 | 标准差 |
|---|---:|---:|---:|---:|
| MSE | 0.0001369795 | 0.0001369370 | 0.0001370117 | 0.0000000300 |
| Cross-sectional Spearman IC | 0.010726 | 0.009341 | 0.008640 | 0.002155 |
| Diagnostic Sharpe | 0.434469 | 0.348872 | 0.344242 | 0.230102 |

Purged K-Fold 的 IC 和 diagnostic Sharpe 高于因果 Walk-Forward，说明这次单模型运行的非因果模型选择通道更乐观；但由于缺少普通 K-Fold、No-Purge 和多候选选择对照，不能把差异直接归因为时序过拟合。整体 IC 较低，也不能据此宣称模型有效或策略可交易。

本结果中的 IC 是按 Session 计算的截面 Spearman IC，`diagnostic Sharpe` 是预测分数诊断量。两者都不是包含持仓构造、换手、费用和滑点的交易策略回测。

## PIT 证据边界

离线特征库的 `ingestion_time` 是 2026-06-29 的一次批量入库时间。本次回放按照声明的事件时间可用性规则构造特征时间，其中成交量和持仓量额外滞后一 Session；它能验证运行时对这些声明的内部一致性检查，但不能从该离线快照证明 2021–2025 年每个历史时点确实已获得相同、未修订的数据。

若要形成生产级 PIT 证据，仍需供应商逐版本快照、发布日历或可审计的历史 ingestion log。

## 工程检查

- Skill 官方格式校验：算法仓库与生产 Skill 目录均通过；
- `python tasks/preflight.py`：两个目录均通过；
- 生产 Skill 契约测试：通过；
- 算法仓库完整测试、strict mypy、Ruff check 和 Ruff format check：通过；
- 隔离环境本地安装后，真实 `purged-cv-skill demo` 控制台入口返回退出码 0 和标准成功回执；
- 真实数据标准 Agent CLI：退出码 0，状态 `success`，引擎版本 `0.6.1`。
