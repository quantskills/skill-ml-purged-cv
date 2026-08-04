# 时序过拟合 Benchmark 研究与落地建议

日期：2026-08-03

## 结论先行

当前仓库**尚未拥有真正的时序交易策略 benchmark**。已有 benchmark 是一个很有价值的结构性泄漏 canary：它在相同数据、Ridge 回归器和 MSE 指标下比较 `unsafe-shuffled-kfold`、`chronological-no-purge`、`purged-kfold` 和 `causal-walk-forward`，并独立统计训练/测试信息区间重叠。但它没有生成逐资产交易信号、没有持仓和换仓、没有交易成本、没有候选策略选择过程，也没有用 PBO/DSR 衡量「从许多参数里挑最好者」的选择过拟合。因此，它能证明 splitter 是否阻断泄漏，不能回答一个时序策略族是否被历史数据过度挑选。

建议新增一个**非截面排名、逐资产独立生成信号的期货 TSMOM（Time-Series Momentum）策略族**作为真实 benchmark；把已有结构性 canary 保留为第一层，把 TSMOM 候选网格、CSCV/PBO、DSR、CPCV 路径、Walk-Forward 和冻结 Holdout 组合成第二层。TSMOM 的经典定义是用每个合约自身过去 12 个月的超额收益预测其未来方向，而不是在同一时点给资产做横截面排名；原论文覆盖 58 个股指、外汇、商品和债券期货/远期合约。[原论文 PDF](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2089463_code753937.pdf?abstractid=2089463&mirid=1)；[AQR 原始因子数据说明](https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Original-Paper-Data)

## 一、当前仓库 benchmark 审计

### 已有能力

- [`benchmark.py`](../../src/purged_kfold_validation/benchmark.py) 的 `run_validation_benchmark` 固定比较四个通道，并对安全通道执行实际信息区间重叠检查；安全通道只要保留一个重叠就失败。
- [`benchmark_pandaai.py`](../../scripts/benchmark_pandaai.py) 使用 fold-local Ridge 回归器和 MSE；每个 fold 重新拟合均值、标准差和系数，避免预处理跨 fold 泄漏。
- [既有 PandaAI 回执](../evidence/pandaai-benchmark-20260801.md) 证明真实缓存上四通道均被调用，并观测到 unsafe/chronological 通道存在重叠、安全通道为零重叠。
- 仓库还分别实现了 Purged K-Fold、CPCV 路径和 Causal Walk-Forward；这些是构建完整 benchmark 的必要积木。

### 为什么还不是真正的时序策略 benchmark

1. 输入特征是 `close/volume/open interest`，目标是未来收益回归，模型是 pooled Ridge；没有明确的逐资产 TSMOM 交易规则。
2. 输出是预测误差而非可交易的净收益序列；不存在前一日权重乘下一日收益、换手、手续费、滑点或合约换月成本。
3. 只有一个模型配置，没有候选参数矩阵，因而无法测量「挑最好参数」造成的选择偏差。
4. 四通道报告未集成 CPCV 多路径分布、CSCV/PBO、DSR、参数排名反转和选择 regret。
5. 既有测试刻意不要求 unsafe 分数一定优于 safe 分数；这适合作为 splitter correctness 测试，但不是过拟合效果 benchmark。

因此，现状应标注为：**泄漏结构 benchmark 已完成；时序策略选择过拟合 benchmark 未完成。**

## 二、推荐的真实时序策略族：期货 TSMOM

### 基础规则

对资产 `i` 和决策日 `t`：

```text
signal(i,t; L) = sign(product(1 + r[i,t-L:t-1]) - 1)
raw_weight(i,t) = signal(i,t; L) * target_vol / sigma(i,t)
weight(i,t) = clip(raw_weight, -leverage_cap, leverage_cap)
strategy_return(i,t+1) = weight(i,t) * tradable_return(i,t+1)
```

每个资产只使用自己的历史，不进行横截面排序。经典锚点配置为 12 个月回看、1 个月持有/再平衡、按单合约事前波动率缩放并跨资产等权组合；AQR 发布的数据说明明确给出了 12 个月 TSMOM、1 个月持有期和 58 个标的的口径。[AQR 数据说明](https://www.aqr.com/Insights/Datasets/Time-Series-Momentum-Original-Paper-Data)

### 有界候选网格

首版只保留可解释且计算量可控的 32 个候选：

| 维度 | 候选值 |
|---|---|
| 自身收益回看 `L` | 21、63、126、252 Sessions |
| 波动率窗口 | 20、60 Sessions |
| 再平衡频率 | 每周、每月 |
| 杠杆上限 | 2、4 |

`target_vol` 固定为 10% 组合目标（或固定单合约目标后再统一组合缩放），避免只改变收益单位的冗余 trial。成本情景 `0/1/3/5 bps × turnover` 是压力测试，不参与参数择优。原论文的 252/约 60/月频配置必须预注册为 anchor，即使它不是样本内最优也要始终报告。

### 因果执行与数据治理

- 信号用后复权/拼接连续价格，实际 PnL 用当时可交易合约及明确换月规则；两者必须分别留下 lineage。`pysystemtrade` 的期货文档明确区分用于信号的 stitched price，并指出不处理换月会制造异常收益，是工程实现的重要参考。[pysystemtrade backtesting 文档](https://github.com/pst-group/pysystemtrade/blob/develop/docs/backtesting.md)
- `t` 日收盘信息形成的信号和波动率只能在 `t+1` 执行，所有 rolling/EWMA 统计显式 `shift(1)`；一个可参考的 TSMOM 复现仓库也用 `.shift(1)` 和单元测试约束这一点。[aqr-tsmom-replication](https://github.com/gehlotmanthan/aqr-tsmom-replication)
- 资产纳入、退市、主力切换和合约映射必须是 point-in-time；禁止按完整样本期后的流动性筛选历史资产。
- 组合日收益用前一交易日冻结权重乘可交易收益，再扣 `abs(w_t-w_{t-1})` 对应的成本；缺失价格不得用未来值回填。
- 每个 sample 的信息区间应覆盖信号最大回看起点到标签/持有期结束；Purge 用真实 label interval，Embargo/Pre-Test Gap 用实际 Session Axis，而不是只按行数猜测。

## 三、PBO、DSR、CPCV 与 Walk-Forward 的分工

这些工具互补，不能互相替代：

| 工具 | 回答的问题 | 本 benchmark 的输入/输出 |
|---|---|---|
| Purged K-Fold | train/test 是否因重叠标签或相邻信息而泄漏 | 每折零区间重叠、候选 OOS 指标 |
| CPCV | 结论是否依赖某一条历史切分路径 | 每个候选的多条完整 OOS path；报告中位数、P10、最差、IQR |
| CSCV/PBO | 样本内挑中的候选有多常跌到样本外候选的后半区 | `T × N_candidates` 净收益矩阵；PBO、logit 分布、性能退化斜率 |
| DSR | 最佳 Sharpe 在非正态、小样本和多重试验修正后是否仍显著 | 候选 trial 数/有效 trial 数、Sharpe 方差、偏度、峰度、样本长度 |
| Walk-Forward | 按真实时间顺序反复选参并在下一段执行会怎样 | 每个窗口所选参数、下一窗净收益、排名稳定性、selection regret |
| 冻结 Holdout | 研究过程完全结束后能否在未触碰时期复现 | 一次性净收益、Sharpe/回撤和预注册 gate |

PBO 原论文把 CSCV 定义为通用、无模型且非参数的回测过拟合估计框架；它衡量的是**选择过程**而非单个策略。[PBO 原论文主页与 PDF](https://escholarship.org/uc/item/4w1110bb)；[作者 PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)。PBO 接近 0.5 表示样本内赢家的样本外排名接近随机；高于 0.5 表示更可能跌到样本外中位数以下，不能误设为「必须小于 0.05」这种 p-value 口径。

DSR 则把 Sharpe 的非正态误差和多次试验产生的期望最大 Sharpe 纳入基准，适合判断被挑中的赢家是否超过「试得足够多自然会出现」的水平。[DSR 原论文 PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

注意：CSCV 是 PBO 的对称 IS/OOS 划分底座；CPCV 是带 Purge/Embargo 并可重构多条 OOS 回测路径的验证器。两者相关但不是同一个算法，不能直接把 CPCV path 数值矩阵标成原论文 CSCV/PBO 而不声明转换假设。

## 四、建议落地接口

建议新增独立入口，避免改变现有 `run_validation_benchmark` 的结构性语义：

```python
run_time_series_strategy_benchmark(
    panel,
    strategy_family="tsmom-v1",
    candidates=[...],
    execution_spec=ExecutionSpec(...),
    cost_scenarios_bps=(0, 1, 3, 5),
    purged_kfold=...,
    cpcv=...,
    walk_forward=...,
    holdout_spec=...,
) -> TimeSeriesOverfittingReport
```

输入必须包含：`session`、`asset_id`、连续价格、可交易合约收益、决策时间、特征可用时间、label interval、主力/换月标识和 source digest。标准 JSON 输出至少包括：

- 数据覆盖、资产/Session 数、每资产样本量与 lineage digest；
- 每个通道的 split 配置、训练量、Purge/Embargo 数和实际 overlap count；
- 每个候选在各 fold/path/window 的**净收益**、Sharpe、Sortino、最大回撤、Calmar、换手和成本；
- unsafe 与 safe 的 optimism gap；
- CPCV 路径中位数、P10、最差值、IQR 和跨路径参数排名；
- CSCV/PBO、性能退化斜率、DSR/PSR、有效 trial 数；
- Walk-Forward 每窗选择及 regret；
- 冻结 Holdout 结果和最终 PASS/FAIL/INCONCLUSIVE，附原因码。

## 五、验收与过拟合判定

### 硬性正确性 gate

1. Purged K-Fold、CPCV、Walk-Forward 和 Holdout 的实际信息区间重叠均为 0。
2. 所有特征、波动率、资产池和权重在决策时刻可获得；泄漏注入 fixture 必须被 safe 通道拦截。
3. 再平衡收益严格使用上一时点冻结权重；成本和换月均进入净收益。
4. 同一候选在不同通道使用相同数据、策略逻辑、成本和指标；只改变验证方式。
5. 固定 seed、输入和版本必须生成相同 digest 与结果。

### 过拟合证据 gate

以下应联合判断，任一单指标都不应独立宣告策略有效：

- PBO `< 0.5` 才说明选择过程优于随机排名；建议生产候选采用更保守的预注册阈值（如 `< 0.2`），但阈值应写入 spec 而非跑完后调整。
- DSR 概率建议 `>= 0.95`，并同时披露 nominal 与 effective trial 数。
- CPCV 净 Sharpe 的中位数应为正，P10/最差分位、IQR 和最大回撤必须披露；不能只展示最好 path。
- Walk-Forward 选中候选相对各窗 hindsight-best 的 selection regret 应稳定，参数排名不能频繁反转；按牛/熊、趋势/震荡和高/低波动 regime 报告。
- 冻结 Holdout 只运行一次；若它失败，结果为 FAIL，不得回到网格修改参数后继续沿用同一 Holdout 名称。
- unsafe-minus-safe optimism gap 只作诊断，没有跨数据集通用阈值；真正硬 gate 是 overlap=0、PBO/DSR、路径尾部、WF 和 Holdout 的共同证据。

## 六、参考实现比较

| 实现 | 可借鉴能力 | 不应直接照搬的边界 |
|---|---|---|
| [eslazarev/purged-cross-validation](https://github.com/eslazarev/purged-cross-validation) | sklearn 接口；变量 label interval；Purged/Group/Walk-Forward/CPCV；路径重构；PSR/DSR/PBO；诊断断言。最接近本项目所需的参考面。 | 是通用验证库，不提供本项目的 PandaData lineage、连续合约治理或 TSMOM 执行协议；应作差分 oracle/设计参考，不宜替换本项目治理层。 |
| [OutOfSampleLab/oos-lab](https://github.com/OutOfSampleLab/oos-lab) | PBO/CSCV、DSR、Harvey-Liu haircut、Walk-Forward 与 CPCV 的小型统计 API；README 明确要求输入 `n_obs × n_variants` 的收益矩阵。 | 明确「不是 backtester、不是 strategy」；不能证明信号、换月、可用时间或区间清除正确。仓库历史和采用度当前很小，适合作独立数值对照，不应作为唯一真值。 |
| [skfolio](https://github.com/skfolio/skfolio) | 成熟的 sklearn 风格组合优化、CPCV 多路径对象和分布图；代码明确给出 `C(n_folds,n_test_folds)` 组合数与 `C(n,k)k/n` 路径数。[CPCV 源码](https://github.com/skfolio/skfolio/blob/main/src/skfolio/model_selection/_combinatorial.py) | 当前接口的 `purged_size/embargo_size` 是样本位置数量，不是本项目的逐样本可变 label interval；更偏资产组合优化，不等价于 TSMOM 事件级治理。 |
| [aqr-tsmom-replication](https://github.com/gehlotmanthan/aqr-tsmom-replication) / [rkohli3/TSMOM](https://github.com/rkohli3/TSMOM) | 前者强调 `.shift(1)`、lagged EWMA 和测试；后者提供 55 个期货数据与 TSMOM/波动缩放/组合函数，可作策略输出 sanity check。 | AQR 没有公开原论文 58 个合约的完整日频底层数据；前者公开结果主要基于 AQR 发布的聚合因子，后者是研究型仓库且数据来源/换月口径需另审。二者都不能直接充当本项目的真实数据 oracle。 |
| [pysystemtrade](https://github.com/pst-group/pysystemtrade) | 期货连续价格、波动率定标、实际合约交易与生产系统边界清晰，适合作工程执行参考。 | 不是 Purged/CPCV/PBO 库，也不是 MOP 原论文的一对一复现。 |

## 七、检索覆盖与证据边界

本次按 Agent Reach 的 evidence-audit 规则记录：

| 范围 | 方式 | 覆盖状态 | 说明 |
|---|---|---|---|
| GitHub | Agent Reach Exa 发现；随后直接打开仓库/源码 | effective + deep-read | 深读了上述 purgedcv、oos-lab、skfolio、TSMOM 与 pysystemtrade 页面；关键结论以仓库或源码直链为准。 |
| 其他网页/论文 | Agent Reach Exa + Web；直接打开论文/数据页 | effective + deep-read | 深读 AQR、SSRN/作者 PDF、PBO 学术存档、DSR 作者 PDF。 |
| 知乎 | Agent Reach/网页限定域搜索；打开可访问文章 | effective，但仅作线索 | 找到[《第七章—金融里的交叉验证》](https://zhuanlan.zhihu.com/p/350110799)等内容，文章自述为 AFML 摘录；未将其作为独立权威证据。 |
| 公众号 | Agent Reach/网页限定域搜索 | attempted，未形成可验证 deep-read | 没有获得可稳定深读、可追溯的一方公众号文章；因此不声称公众号覆盖成功，也不引用搜索摘要支撑技术结论。 |

知乎和公众号属于二手内容，最多用于发现关键词与实现线索。关于 TSMOM、PBO、DSR 和 CPCV 的定义与判定，应以原论文、作者数据页和可审计源码为准。

## 最终建议

不要替换现有 Ridge/MSE 四通道 benchmark；将它更名或明确定位为 `structural-leakage-canary`。新增 `tsmom-selection-overfitting-benchmark`：在同一真实期货面板上生成 32 个完全因果的逐资产 TSMOM 候选净收益，依次运行结构重叠审计、Purged K-Fold、CPCV 多路径、CSCV/PBO、DSR、Walk-Forward 选择和一次性 Holdout。这样才能同时回答两个不同问题：**验证器有没有泄漏**，以及**研究者是否从太多时序策略参数中挑出了历史偶然赢家**。
