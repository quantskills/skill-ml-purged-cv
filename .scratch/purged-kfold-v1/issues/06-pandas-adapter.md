# 06 — 提供显式 Pandas 接入通道

Type: task
Status: resolved
Blocked by: 04 — 覆盖金融时间边界、Embargo 与面板数据; 05 — 阻止非 PIT 特征进入正式评分

**What to build:** 让 pandas 使用者能够通过显式字段映射，把 DataFrame 或 MultiIndex 数据转换为相同的 Validation Dataset，并沿同一 Split Plan 与 Leakage-Safe Evaluator 路径获得结果。适配器必须保留面板会话分组与 PIT 证据，且不得用列名、索引层级、预测期或隐藏属性进行推断。

- [x] 显式映射的 DataFrame 和 MultiIndex 输入能够完成转换、Purged K-Fold 规划与正式评估。
- [x] 适配后的样本身份、Session Axis、Panel Session Group、Information Interval 与 PIT 元数据保持一致。
- [x] 相同语义的核心输入和 pandas 输入产生等价的 Fold Assignments、排除摘要及 OOS 结果。
- [x] 时区不一致、重复索引、未知会话、形状错误与缺失映射均以明确诊断失败。
- [x] 适配器不推断列名、索引层级、标签期或数据来源，也不从隐藏属性读取证据。
- [x] 未安装 pandas 时，核心包仍可导入、运行和测试；pandas 仅作为可选依赖存在。

## Answer

已实现完全显式的 PandasField/PandasDatasetMapping 和可选 pandas adapter，MultiIndex 面板、时间证据与 PIT provenance 均进入同一核心计划和评估路径。适配器拒绝歧义且核心不导入 pandas。观察证据：适配器测试 4 passed；完整测试 25 passed；Ruff 通过。

## Comments
