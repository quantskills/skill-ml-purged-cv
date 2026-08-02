# Release hardening v1

Status: resolved

## Execution prompt

在 `purged-kfold-validation` 中完成发布硬化：

1. 为任意特征 CSV/Parquet 上传增加读取前资源预算与压缩输入防护。
2. 完成独立 Holdout 验证能力，保证模型、特征、阈值和超参数选择不能接触 Holdout。
3. 增加合成对抗泄漏、多模型和多 regime 排名稳定性验证。
4. 把新能力接入公共 API、CLI 限额、文档、工作项、测试和验收证据。
5. 运行完整 preflight、lint、format、strict typecheck、测试、构建和安装 canary。
6. 不伪造远程 CI、GitHub 发布或生产部署结果；外部条件缺失时留下明确交接项。

## Acceptance contract

- CSV 行数预算通过 bounded read 在完整解析前生效。
- Parquet 行数、列数和声明的解压后字节数通过 footer metadata 在完整解析前生效。
- 资源拒绝为类型化、脱敏、fail-closed 的 `UploadLimitError`。
- `EvaluationProtocol` 冻结训练、Holdout、模型、变换器、指标和搜索/切分身份。
- Holdout 只能在时间上位于训练集之后，且样本身份不相交。
- 本地 Holdout store 在拟合前原子消费 Holdout identity；失败尝试也不得重试。
- Holdout 只用于一次最终预测，训练和变换器拟合只接触冻结训练集。
- Holdout 回执不持久化原始行、特征、目标或预测。
- 多 regime 排名报告要求模型集合一致，输出中位/最差排名和秩相关分布，并对排名反转给出不稳定结论。
- 现有 PKF、CPCV、Walk-Forward、上传 CLI 行为不回退。

## Non-goals

- 不在本切片实现远程仓库创建、GitHub 凭据、合并、部署或生产监控。
- 不把零泄漏重叠解释成模型盈利。
- 不把 Holdout 变成可重复查询的排行榜。
