# 02 — 打通 Fold-Local 训练与 OOS 评分

Type: task
Status: resolved
Blocked by: 01 — 打通最小泄漏安全 Split Plan

**What to build:** 让使用者能够把最小 Split Plan 交给 Leakage-Safe Evaluator，由 Fold Factory 为该折创建隔离的转换器和估计器，完成拟合、预测与评分，并得到带来源证据的原始 OOS Ledger 和至少一个派生指标。任何步骤失败时，使用者都不能收到看似有效的部分总分。

- [x] 一个确定性的示例模型可以从已验证数据集和单折计划完成训练、预测，并返回逐观察值的 OOS Ledger。
- [x] 转换器与估计器只能通过 factory 创建；评估接口不接受预拟合实例。
- [x] 每次折执行使用全新的转换器和估计器身份，且拟合只接触该折训练集。
- [x] OOS 记录保留样本、折、数据、计划、模型和指标所需的来源标识，并能派生至少一个版本化指标。
- [x] factory、fit、transform、predict 或 metric 任一失败都会终止整个评估，且不返回部分正式分数。
- [x] 相同输入与确定性组件会产生相同的 OOS 顺序和评分结果。

## Answer

已实现 factory-only 的 Leakage-Safe Evaluator、顺序 Fold-Local 转换、固定模型规格、原始 OOS Ledger、版本化 Derived Metric 与整体失败语义。观察证据：`python -m pytest -q tests/test_evaluation.py tests/test_split_plan.py` 为 8 passed；相关 Ruff 检查通过。

## Comments
