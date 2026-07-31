# 05 — 阻止非 PIT 特征进入正式评分

Type: task
Status: resolved
Blocked by: 02 — 打通 Fold-Local 训练与 OOS 评分

**What to build:** 让使用者能够在评估前证明每项特征在对应 Decision Time 已经可用，并明确使用的 PIT Snapshot。合规数据可以沿既有 Evaluator 路径生成正式分数；latest revision、来源缺失或晚于决策时间的数据只能用于诊断，不能形成正式 CV 证据。

- [x] Feature Availability 不晚于对应 Decision Time 且 PIT Snapshot 明确的数据能够完成正式评估。
- [x] latest revision、缺失可用时间、缺失快照来源或晚到特征均被确定性地判定为不可正式评分。
- [x] 不合规数据可以返回结构化诊断，但不能产生正式 OOS Ledger、派生指标或部分有效分数。
- [x] 合规 OOS 结果保留数据与特征来源证据，使正式评分使用的快照可以被追踪。
- [x] 错误与诊断只暴露身份和元数据，不泄露原始特征值或目标值。
- [x] 覆盖“最新修订数据看似提升 CV 分数”的回归反例，证明该路径被 fail closed 拦截。

## Answer

已实现 Decision Time、行级或逐特征 Feature Availability、PIT Snapshot provenance，以及 Evaluator 的 formal-scoring 门禁。诊断保持可用，但所有不合规来源均在产生 Ledger 前 fail closed，且错误不包含特征值。观察证据：完整测试 21 passed，完整 Ruff 检查通过。

## Comments
