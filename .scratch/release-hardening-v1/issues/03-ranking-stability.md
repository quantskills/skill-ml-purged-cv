# 03 — 多 regime 模型排名稳定性

Type: task
Status: resolved
Blocked by: None

- [x] 接受至少两个 regime、两个固定模型的版本化分数矩阵。
- [x] 要求每个 regime 的模型集合完全一致且分数有限。
- [x] 输出每个模型的中位排名、最差排名、第一名次数和跨 regime Spearman 分布。
- [x] 排名反转在冻结阈值下明确标记为不稳定。
- [x] 合成稳定与对抗反转测试通过。

## Answer

新增 `assess_model_ranking_stability`。稳定样例得到 Spearman 1.0；完全反转的
对抗样例得到 -1.0 并明确标记不稳定；缺失模型和非有限分数 fail closed。

## Comments
