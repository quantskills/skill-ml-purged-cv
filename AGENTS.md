# 开发维护说明（AGENTS.md）

本文件只面向修改、测试和发布本仓库的开发 Agent，是开发阶段的 onboarding 与安全约束。它不是 Skill 的运行入口，不参与 Skill 触发，也不是普通用户或消费端 Agent 的必读文件；使用本项目能力时应从根目录 `SKILL.md` 开始。

## Canonical onboarding

Read `AGENTS.md`, `PROGRAM.md`, `CONTEXT.md`, and `README.md`; run `python tasks/preflight.py`; then read only the active work item and task-specific documents.

## Safety boundaries

Plan and read-only inspection are defaults. Never commit credentials, private data, runtime state, logs, or caches. Never write external owners or production without authorization.

## Verification

Run focused checks, `python tasks/preflight.py`, and `python tasks/test.py`; record observed evidence. A local pass does not authorize deployment.

## Extending

Keep product source in `src/`, helpers in `scripts/`, entrypoints in `tasks/`, tests in `tests/`, non-secret examples in `config/`, and contracts/evidence in `docs/`.

## Agent skills

### Issue tracker

Issues and specs are tracked as local Markdown under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the canonical labels `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository using root `CONTEXT.md` and system-wide ADRs under `docs/adr/`. See `docs/agents/domain.md`.
