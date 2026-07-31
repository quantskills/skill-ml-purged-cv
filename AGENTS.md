# Agent contract

This file is the canonical onboarding and safety contract.

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
