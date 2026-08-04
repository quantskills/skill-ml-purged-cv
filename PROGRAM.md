# 项目长期纲领

本文件定义 `skill-ml-purged-cv` 的长期使命、产品边界、不可变原则和成功门槛。短期实施状态写入 `CONTEXT.md` 与 `docs/WORK_ITEMS.md`，具体接口写入 `docs/interface-contract.md`，不得用短期任务悄悄改变本纲领。

## 使命

为金融机器学习提供一套跨平台、可审计、默认拒绝不完整证据的时序验证基础设施，使研究人员和 Agent 能够区分“模型确实具有样本外信息”与“验证过程发生未来函数或信息泄漏”。

项目以显式 Information Interval 和 Trading Session 为时间基础，提供 Purged K-Fold、Embargo、CPCV Path、Causal Walk-Forward、Fold-Local 评估、任意特征血缘治理和一次性最终 Holdout。

## 服务对象

- 训练金融时序模型的量化研究人员；
- 评估任意用户特征、因子或 Embedding 的平台；
- 需要自动运行防泄漏验证的 Codex、Hermes、Claude 等 Agent；
- 需要结构化回执、模型排名稳定性和验证审计证据的工程团队。

## 预期结果

项目长期应确保：

1. 每条样本都具有稳定身份、Session、Information Interval、Decision Time 和 PIT 证据；
2. 多资产同 Session 样本不可跨训练/测试侧拆分；
3. Purge 由真实区间重叠决定，Embargo 由 Session Axis 距离决定；
4. 所有学习型预处理和 estimator 均为 Fold-Local；
5. Purged K-Fold、CPCV、Walk-Forward 和 Holdout 证据通道保持独立；
6. 缺失时间、血缘、路径完整性或训练充分性时 fail closed；
7. 普通用户和 Agent 通过同一 Python/CLI 核心获得可复现结果；
8. 输出明确区分结构安全、模型效果、盈利能力和部署授权。

## 产品表面

项目拥有四类稳定表面：

- Python API：高级用户自定义 splitter、metric、transformer 和 estimator；
- `purged-cv-upload`：本地 CSV/Parquet 审计与三通道固定基线评估；
- `purged-cv-skill`：一个请求 JSON 对应一个标准 JSON 回执的 Agent 无关入口；
- `SKILL.md` 与 references：Agent 发现、执行顺序、输入输出和解释边界。

所有表面必须委托给同一套领域对象和验证实现，不得为不同 Agent 复制算法。

## 不可变原则

### 时间与泄漏

- 使用 Trading Session，不使用行号或自然日近似市场时间；
- 使用包含边界的 Information Interval 判断 Purge；
- Embargo 作用于连续测试块之后，不冒充 Walk-Forward Pre-Test Gap；
- Causal Walk-Forward 的训练信息必须严格早于测试信息；
- Holdout 在设计冻结前不得打开，失败尝试同样消耗一次机会。

### 特征与模型

- 任意有限数值特征可以提交，但不得自动推断平稳性、公式或时间含义；
- 特征 manifest、source/code digest、可用时间和目标依赖必须显式；
- 学习型转换必须通过全新工厂在每折训练侧拟合；
- 任何预拟合对象复用、部分预测或跳过无效折都不能形成正式证据。

### Agent 与接口

- `SKILL.md` 负责导航，不保存算法实现；
- Agent 专用 adapter 只能包含展示或调用信息；
- 标准回执保留底层 authoritative result，不允许 Agent 改写为更好看的结果；
- 输出不泄露原始行、特征值、目标、预测或绝对本地路径；
- Agent 不得索取供应商凭据、登录远程数据源或自动部署交易系统。

## 项目范围

项目负责：

- 版本化领域对象、splitter、evaluator 和证据通道；
- PIT、特征可用性、血缘和 transformer identity；
- 受资源限制的本地 CSV/Parquet 边界；
- CPCV combination、Path Decomposition 和路径指标；
- 通用候选策略净收益矩阵的选择过拟合审计，以及仅用于离线验证的内置 TSMOM 参考策略族；
- CSCV/PBO、DSR、CPCV 选参路径、Walk-Forward selection regret 和预注册成本压力情景；
- 模型跨 regime 排名稳定性；
- 冻结协议、一次性 Holdout 状态与脱敏回执；
- 跨平台安装包、CLI、schema、示例、测试和验证证据。

## 非目标

项目不负责：

- 自动发现或承诺盈利策略；
- 订单执行、真实成交滑点、容量和风控系统；离线 benchmark 只允许使用预注册的简化换手成本压力模型；
- 验证外部供应商时间戳和数据声明的绝对真实性；
- 阻止调用方绕过 Holdout 接口直接查看文件；
- 在缺乏证据时自动猜测字段、标签期、特征公式或部署结论；
- 未经明确设计和验收的外部写入、凭据管理或生产部署。

## 交付路径

1. 保持领域术语和时间边界明确；
2. 先定义可观测验收标准，再实现接口；
3. 使用属性测试、反例和回归测试保护防泄漏不变量；
4. 使用真实本地数据验证训练充分性和多通道差异；
5. 构建并从 wheel 独立验证安装、schema、示例和 CLI；
6. 只有在本地、跨平台 CI、发布和观察证据一致后扩大生产声明。

## 成功门槛

- `python tasks/preflight.py` 和 `python tasks/test.py` 通过；
- Ruff、格式检查、严格 MyPy 和全量 Pytest 通过；
- wheel/sdist、Twine 和安装后 canary 通过；
- Skill frontmatter 和 Agent 元数据有效；
- 每个正式通道的保留 Information Interval 重叠为零；
- CPCV Path 完整且训练样本满足声明门槛；
- 文档、接口、schema、代码、测试和验收证据相互一致；
- 没有凭据、私有数据、运行状态、日志或 Holdout 原始内容进入版本库。

## 变更控制

只有长期使命、范围、不可变原则、产品表面或成功门槛发生变化时才修改本文件。当前事实更新 `CONTEXT.md`，任务更新 `docs/WORK_ITEMS.md`，接口细节更新 `docs/interface-contract.md`，验收结果更新 `docs/verification.md`。
