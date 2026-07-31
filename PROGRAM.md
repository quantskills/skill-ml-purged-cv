# Project program

This document is the durable direction and boundary contract for the project.

## Mission

Windows、Linux、macOS 跨平台可复用 Python 软件库，用于金融机器学习训练验证，提供基于显式事件区间和交易日历 session 的 Purged K-Fold、Embargo、CPCV 路径拼接、严格 walk-forward 与 untouched holdout；多资产 panel 按同日分组，预处理与嵌套 HPO 必须 fold-local，缺失泄漏元数据时 fail closed，并以属性测试和回归测试验证防未来函数与信息泄漏。

## Intended outcomes

Deliver and maintain the stated mission as an owned project with explicit contracts, observable acceptance evidence, and a safe path from scaffold to verified behavior.

## Scope

Own versioned source, tests, non-secret configuration contracts, documentation, and verification for this `library-cli` project in the `finance` domain.

## Non-goals

The scaffold does not authorize deployment, production readiness, external-system writes, credential provisioning, or invented product schemas. It does not claim behavior that has not been implemented and verified.

## Operating principles

Follow `AGENTS.md` for safety and execution rules. Prefer plan/read-only behavior, preserve external ownership, keep secrets and runtime artifacts out of canonical source, and record observed evidence instead of inferred success.

## Delivery path

1. Preserve the mission and boundaries in this program.
2. Define the first measurable behavior in `docs/WORK_ITEMS.md`.
3. Implement source and tests within repository ownership.
4. Run `python tasks/preflight.py` and `python tasks/test.py`.
5. Record evidence before claiming readiness or expanding scope.

## Success gates

The canonical documents remain consistent; preflight and project tests pass; requirements map to implementation and evidence; no secret or unowned runtime state is committed; deployment or external writes remain separately authorized.

## Change control

Update `PROGRAM.md` only when the long-term mission, scope, non-goals, operating principles, delivery path, or success gates change. Track current execution in `docs/WORK_ITEMS.md` and current facts in `CONTEXT.md`.
