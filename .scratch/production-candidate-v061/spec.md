# Production candidate v0.6.1

Status: claimed

## 可直接复用的执行提示词

你正在维护 `purged-kfold-validation` 金融时序验证库。算法核心不再扩展；本轮只
执行生产候选门禁，不调整 Purge、Embargo、CPCV 或 Walk-Forward 数学定义。

目标：把当前实现冻结为 v0.6.1 生产候选，并以可审计证据验证四项门禁：

1. 冻结干净版本：完整运行 preflight、pytest、strict mypy、Ruff、格式检查、
   dependency check、build、Twine 和隔离 wheel canary；确认无凭据、原始行情、
   Holdout 预测或 runtime store 进入 Git；提交全部项目内变更，创建本地注释标签
   `v0.6.1`，最终工作树必须干净。
2. 远程 CI：仅在明确的 GitHub 仓库 URL、可用 `gh` 和已认证会话存在时配置
   `origin`、推送候选分支和标签。CI 必须覆盖 Ubuntu Python 3.11/3.12/3.13、
   Windows 3.11、macOS 3.11，并安装 Parquet 测试依赖，运行 preflight、完整
   项目门禁、构建和 Twine。不得把“CI 文件存在”写成“远程 CI 已通过”。
3. 不可重复 Holdout：使用现有五年 PandaData 治理数据，冻结训练/Holdout、模型、
   指标、搜索和切分身份；Holdout 在拟合前原子消费，失败也不可重试。已有 receipt
   时只校验绑定并复用 receipt，不得重新读取 Holdout。记录 MSE 与声明边界，不持久化
   原始值或预测。
4. 排名稳定性：在 Holdout 之前的 development 数据上固定至少 3 个模型，在至少
   3 个连续市场 regime 中使用同一指标与同一候选集合，报告排名、中位/最差排名、
   pairwise Spearman 和反转结论；结果不得解释为盈利证明。

完成标准：四项门禁均有 receipt-backed 证据后才能标记 production-ready。任何
远程地址、认证、CI、部署或监控事实缺失时，状态保持 production-candidate，并
明确记录唯一外部阻塞项，不得伪造成功。

## 当前验收映射

| Gate | Current state | Evidence |
|---|---|---|
| clean v0.6.1 freeze | in progress | local 132-test/type/build/wheel evidence complete; commit/tag pending |
| remote cross-platform CI | external blocker | workflow present; remote URL and authenticated `gh` absent |
| one-attempt real Holdout | complete | five-year receipt `245bf378...1355bd` |
| 3 models × 3 regimes | complete | minimum pairwise Spearman 1.0; mean ranked first |

## Production label rule

`production-ready` is forbidden until the remote CI gate reports success against the
same immutable commit. Until then the highest honest label is `production-candidate`.
