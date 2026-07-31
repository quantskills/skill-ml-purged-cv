# Agent handoff

Follow `AGENTS.md`; this is not a competing read order.

## Problem solved

Windows、Linux、macOS 跨平台可复用 Python 软件库，用于金融机器学习训练验证，提供基于显式事件区间和交易日历 session 的 Purged K-Fold、Embargo、CPCV 路径拼接、严格 walk-forward 与 untouched holdout；多资产 panel 按同日分组，预处理与嵌套 HPO 必须 fold-local，缺失泄漏元数据时 fail closed，并以属性测试和回归测试验证防未来函数与信息泄漏。

## Current structure

`src/`, `scripts/`, `tasks/`, `tests/`, `config/`, and `docs/` have explicit ownership.

## Program authority

`PROGRAM.md` defines durable direction and boundaries; change it only for an explicit long-term program decision.

## Verification

Canonical entrypoints are `python tasks/preflight.py` and `python tasks/test.py`; they run natively on Windows and POSIX.

## Known unknowns

Product schemas, external wiring, deployment targets, and production readiness remain pending owner definition.
