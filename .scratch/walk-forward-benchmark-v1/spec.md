# Walk-Forward and PandaAI benchmark vertical slice

Status: resolved
Owner: project owner
Evidence channels: causal-walk-forward, diagnostic benchmark

## Problem

Slice 1 produces leakage-aware model-selection evidence with two-sided Purged K-Fold,
but cannot prove that a model can be trained only on information available before a
future test block. The project also lacks a repeatable real-data probe showing how
ordinary validation differs from interval-aware and causal validation.

## Solution

Add an evidence-bearing Causal Walk-Forward splitter and an offline PandaAI daily-data
adapter/benchmark. Reuse `ValidationDataset`, `SplitPlan`, `FoldAssignment`,
`LeakageSafeEvaluator`, and `OOSLedger`; preserve channel identity rather than creating
a parallel scoring system for safe channels. Unsafe baselines exist only inside the
benchmark and must be labeled diagnostic.

## Acceptance stories

1. Training in every causal fold is strictly earlier than the test block.
2. Overlapping training label intervals are purged even when their sample session is earlier.
3. Pre-Test Gap uses Session Axis positions and works across holidays/weekends.
4. Expanding and bounded sliding training histories are deterministic.
5. Panel rows sharing one session never cross assignment sides.
6. Invalid causal folds remain visible and abort formal evaluation.
7. Evaluator output carries `causal-walk-forward` rather than `model-selection`.
8. PandaAI-style daily frames require explicit mapping and PIT snapshot declarations.
9. Forward labels and information intervals use actual per-asset shifted sessions.
10. One benchmark compares unsafe shuffled, chronological, Purged K-Fold, and causal channels.
11. Benchmark overlap counts are structural facts; score ordering is not assumed.
12. The operator script reads only a user-supplied local cache and emits deterministic redacted JSON.

## Out of scope

CPCV, nested HPO, governed holdout, live PandaAI authentication, strategy PnL,
deployment, persistence services, and performance optimization.

## Resolution evidence

- `CausalWalkForward` implements deterministic expanding/sliding past-only folds,
  inclusive interval Purge, session-based Pre-Test Gap, and causal evidence identity.
- The optional PandaAI adapter builds point-in-time validation datasets from explicit
  local daily-parquet mappings without network access or credentials.
- The four-channel benchmark independently audits retained interval overlap and emits
  deterministic, redacted JSON.
- Local acceptance on 2026-08-01: 52 tests passed; strict mypy, Ruff, build, Twine,
  dependency, preflight, and isolated-wheel checks passed.
- A five-asset PandaAI cache receipt is recorded in
  `docs/evidence/pandaai-benchmark-20260801.md`.
