# 04 — 覆盖金融时间边界、Embargo 与面板数据

Type: task
Status: resolved
Blocked by: 03 — 扩展为完整、严格的 Purged K-Fold

**What to build:** 让使用者可以在具有可变标签期、非连续受保护区块、节假日以及多资产同会话数据的真实金融时间轴上规划并执行 Purged K-Fold。Purge 必须依据精确信息区间，Embargo 必须依据 Session Axis 且逐连续测试区块计算，面板中的同会话样本必须作为一个不可拆分的分配组。

- [x] 可变长度及多个受保护 Information Intervals 都按精确重叠关系执行 Purge。
- [x] 不连续测试区块分别计算 Embargo，不会错误排除两个区块之间的安全训练区域。
- [x] Embargo 按 Session Axis 位置而不是自然日计数，周末、节假日与缺失交易日的结果正确。
- [x] 多资产同会话样本始终位于同一分配侧，Panel Session Group 不跨训练集和测试集。
- [x] Exclusion Summary 和可选 Exclusion Trace 能确定性地区分 overlap、embargo 及其他排除原因。
- [x] 至少一个对抗性数据集证明普通 K-Fold 会泄漏，而计划后的全部保留训练样本满足时间隔离不变量并可完成评估。

## Answer

已实现按精确 Information Interval 的 Purge、逐 TestBlock 且基于 Session Axis 的 Embargo、稳定 Exclusion Summary/Trace，以及由会话构造而非面板行构造的分折几何。观察证据：金融时间、面板、严格计划、评估和基础计划测试合计 17 passed；完整 Ruff 检查通过。

## Comments
