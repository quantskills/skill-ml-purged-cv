# General deterministic CPCV vertical slice

Status: resolved
Owner: project owner
Evidence channel: cpcv-robustness

## Problem

Purged K-Fold provides inexpensive leakage-aware model-selection evidence, while
Causal Walk-Forward provides past-only evidence. The project cannot yet measure how
model behavior varies across the complete family of legal multi-group test
combinations or assemble those repeated predictions into independently auditable CPCV
Paths.

## Solution

Add a general deterministic `CombinatorialPurgedCV` splitter and verified Path
Decomposition. Reuse the existing trusted leakage and evaluator cores, retain every
path-scoped OOS fact, and keep CPCV robustness evidence separate from model-selection
and causal channels.

## Acceptance stories

1. Every chronological test-group combination is planned exactly once.
2. Multi-block Purge and per-contiguous-block Embargo have zero retained overlap.
3. Panel sessions remain indivisible.
4. General deterministic paths satisfy occurrence, group, path, and combination invariants.
5. Invalid combinations and excessive combination counts fail closed.
6. Repeated CPCV observations retain combination/group/path identity.
7. Every complete path covers every sample exactly once.
8. Fold and path metrics remain derived from the raw OOS Ledger.
9. Existing Purged K-Fold, Walk-Forward, adapters, and benchmark remain unchanged.

## Out of scope

Nested HPO, governed holdout access, parallel scheduling, persistence, live PandaAI,
deployment, and profitability/performance claims.

## Resolution evidence

- General deterministic combination planning and Path Decomposition are implemented
  without replacing Purged K-Fold or Causal Walk-Forward.
- Focused CPCV verification covers 30 cases, including every small configuration with
  `3≤N≤8` and `2≤k<N`.
- Local acceptance on 2026-08-01: 82 tests passed; preflight, project test gate,
  strict mypy, Ruff, build, Twine, dependency, and isolated-wheel checks passed.
- The five-asset PandaAI structural receipt is recorded in
  `docs/evidence/cpcv-pandaai-probe-20260801.md`.
