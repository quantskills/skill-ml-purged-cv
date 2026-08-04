# 时序策略选择过拟合协议

## 两类输入

`purged-cv-strategy` 支持两种互不混淆的动作：

- `analyze-return-matrix`：用户已经拥有多个候选策略的逐 Session 毛收益、净收益和换手矩阵；
- `benchmark-tsmom`：用户提供密集、连续、已治理的信号价格与可交易收益面板，工具生成内置 TSMOM 候选族。

两者都使用 `.npz`，且以 `allow_pickle=False` 读取。Session 必须严格递增，ID 必须是 NumPy Unicode/bytes 一维数组，收益必须有限且大于 `-1`。

### 候选收益矩阵 NPZ

必须且只能包含：

```text
sessions          datetime64[T]
candidate_ids     unicode[N]
gross_returns     float64[T,N]
net_returns       float64[T,N]
turnover          float64[T,N]
```

### TSMOM 面板 NPZ

必须且只能包含：

```text
sessions          datetime64[T]
asset_ids         unicode[A]
signal_prices     float64[T,A]
tradable_returns  float64[T,A]
```

TSMOM 输入是密集面板。连续合约构造、换月映射、PIT 资产池和供应商口径必须在进入该接口前完成；本工具只验证矩阵内部约束，不能证明外部数据声明真实。

## 请求示例

```json
{
  "schema_version": "1",
  "action": "analyze-return-matrix",
  "data_path": "strategy-returns.npz",
  "options": {
    "annualization_sessions": 252,
    "cscv_groups": 8,
    "cpcv_groups": 6,
    "cpcv_test_groups": 2,
    "cpcv_embargo_sessions": 5,
    "walk_forward_windows": 5,
    "minimum_train_sessions": 252
  }
}
```

## 结果解释

- `pbo.probability`：CSCV 中样本内赢家跌到样本外候选中位数以下的比例；`0.5` 附近表示选择排名接近随机，不是显著性 p-value。
- `deflated_sharpe.probability`：被选 Sharpe 超过多 trial、非正态修正基准的概率；trial 数等于候选策略数。
- `cpcv.sharpe_distribution`：组合选参路径的净 Sharpe 中位数、P10、最差值和 IQR。
- `walk_forward.mean_selection_regret`：因果选中候选与同测试窗 hindsight-best 的平均 Sharpe 差。
- `selection_optimism_gap`：全样本最佳 Sharpe 与因果 Walk-Forward Sharpe 的差，仅作诊断，不存在跨市场通用阈值。

任何 PASS/FAIL 阈值都必须在运行前写入外部 protocol。该命令只返回证据，不自动授权盈利、上线或部署。
