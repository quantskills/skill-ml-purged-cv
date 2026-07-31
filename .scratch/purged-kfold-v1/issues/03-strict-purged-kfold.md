# 03 — 扩展为完整、严格的 Purged K-Fold

Type: task
Status: resolved
Blocked by: 01 — 打通最小泄漏安全 Split Plan

**What to build:** 让使用者能够为任意合法 n_splits 生成连续、确定性的 Purged K-Fold 计划，检查每个候选折的有效性，并把有效计划交给同一评估路径执行。计划必须保留可审计的 Fold Assignments；若任一请求折无效，正式分配和正式评估都必须整体拒绝。

- [x] 活跃 Session Axis 上的会话被连续分配，且每个会话恰好在一个测试折中出现。
- [x] 每个 Fold Assignment 提供稳定的折身份、样本身份、位置、输入摘要与排除摘要。
- [x] 合法 n_splits 的完整计划能够经 Leakage-Safe Evaluator 运行，并产生预期的 OOS 测试覆盖。
- [x] 最小训练量、最小测试量和折数约束被显式检查，失败原因可确定地复现。
- [x] 诊断计划保留所有无效候选折及原因；只要一个请求折无效，正式分配与评估就不产生部分结果。
- [x] 允许双侧训练集，但所有输出明确标注为模型选择证据，而不是因果部署证据。

## Answer

已实现任意合法折数的连续 Session Axis 分块、完整测试覆盖、稳定 Fold Assignment、显式最小规模约束、诊断计划与 formal assignments/evaluation 整体 fail-closed。观察证据：三个公共接缝测试文件合计 14 passed，相关 Ruff 检查通过。

## Comments
