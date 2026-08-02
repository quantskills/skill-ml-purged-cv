# 01 — 上传读取前资源预检

Type: task
Status: resolved
Blocked by: None

- [x] CSV bounded read 在完整文件解析前执行 `max_rows`。
- [x] Parquet footer 在读取表数据前执行行、列和解压后字节预算。
- [x] CLI 暴露新预算且审计回执记录生效限额。
- [x] 类型检查、拒绝输出和兼容性测试通过。

## Answer

`FeatureUploadLimits` 新增列数和 Parquet 解压后字节预算。CSV 只读取映射列并以
`max_rows + 1` 有界解析；Parquet 在 `pandas.read_parquet` 前检查 footer。
完整仓库 132 tests、strict mypy、Ruff、构建和 wheel canary 通过。

## Comments
