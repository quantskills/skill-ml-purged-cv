# 01 — 打通最小泄漏安全 Split Plan

Type: task
Status: resolved
Blocked by: None — can start immediately

**What to build:** 让使用者能够提交一个最小、单资产、按交易会话排列的 Validation Dataset，并获得可审计、可重复的单折 Split Plan。该路径必须用标签的 Information Interval 排除会与测试信息重叠的训练样本，同时给出足够的身份、位置与排除证据，使结果可以独立检查，而不依赖后续完整 K-Fold 能力。

- [x] 一个结构合法的最小单资产数据集能够生成确定性的单折候选计划，并明确列出训练集与测试集。
- [x] 所有保留训练样本的 Information Interval 均不与受保护的测试区间重叠。
- [x] Split Plan 给出稳定的样本身份、位置、计划摘要和按原因统计的基础排除证据。
- [x] 重复输入产生相同的计划、顺序与摘要；会改变计划语义的输入变化能够被识别。
- [x] 重复身份、非法 Session Axis、区间倒置和形状不一致以确定性的类型化错误失败。
- [x] 核心路径无需 pandas 或 sklearn 即可导入、运行和测试。

## Answer

已实现不可变 Validation Dataset、Information Interval、确定性摘要、类型化数据错误，以及能够返回证据型 Fold Assignment 的最小 Purged K-Fold Split Plan。观察证据：`python -m pytest -q tests/test_split_plan.py` 为 4 passed；`python -m ruff check src tests/test_split_plan.py` 通过。

## Comments
