# 02 — 一次性受治理 Holdout

Type: task
Status: resolved
Blocked by: None

- [x] 冻结版本化 Evaluation Protocol。
- [x] 训练/Holdout 边界、身份和组件摘要不一致时 fail closed。
- [x] Holdout identity 在模型拟合前原子消费，失败也不可重试。
- [x] 返回内存 OOS 证据并仅持久化脱敏 Holdout Receipt。
- [x] 状态机、生命周期和原子性测试通过。

## Answer

新增 `EvaluationProtocol`、`LocalHoldoutStore`、`HoldoutReceipt` 和独立
`holdout-confirmation` Evidence Channel。exclusive claim 在拟合前创建；失败尝试
同样消费 Holdout，且持久化回执不含行、特征、目标或预测。

## Comments
