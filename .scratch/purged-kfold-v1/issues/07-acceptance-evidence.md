# 07 — 完成 Slice 1 的对抗性验收证据

Type: task
Status: resolved
Blocked by: 02 — 打通 Fold-Local 训练与 OOS 评分; 03 — 扩展为完整、严格的 Purged K-Fold; 04 — 覆盖金融时间边界、Embargo 与面板数据; 05 — 阻止非 PIT 特征进入正式评分; 06 — 提供显式 Pandas 接入通道

**What to build:** 让维护者能够通过一套可重复的验收流程，证明 Slice 1 在正常、边界和恶意输入下都满足已批准 Spec，并从需求一路追踪到实际通过的测试与文档。验收结果必须清楚限定为 Purged K-Fold 模型选择证据，不得暗示后续能力已经交付。

- [x] 每项 Slice 1 需求和不变量都能追踪到对应行为、测试及实际观察结果。
- [x] 性质测试覆盖无区间重叠、测试覆盖唯一、面板组不可拆分、确定性和 fail-closed 等核心不变量。
- [x] 回归反例覆盖普通 K-Fold 泄漏、按自然日错误 Embargo、跨不连续区块过度排除、复用折对象及非 PIT 特征评分。
- [x] 最小示例同时展示 Split Plan 诊断入口和 Leakage-Safe Evaluator 正式验收入口。
- [x] 预检、完整测试、静态检查、格式检查和无可选依赖导入 canary 全部通过并可重复执行。
- [x] 公共文档明确区分模型选择证据与因果部署证据，并明确 CPCV、Walk-Forward、Holdout Governance、Nested HPO 和持久化写入器不属于 Slice 1。

## Answer

已完成属性测试、失败注入、回归反例、双入口示例、公共接口文档、需求追踪、本地验收记录及 Standards/Spec 双轴代码审查。审查发现均已修复并复核关闭。最终本地证据：41 passed；预检、Ruff、全仓格式、editable 安装、核心无 pandas 导入、安装后 core/adapter 导入及 `pip check` 全部通过。

## Comments
