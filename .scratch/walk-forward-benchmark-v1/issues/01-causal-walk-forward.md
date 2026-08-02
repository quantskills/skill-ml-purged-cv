# WF-001 — Causal Walk-Forward and Pre-Test Gap

Type: task
Status: resolved
Blocked by: PKF-007

- [x] Public deterministic `CausalWalkForward.plan()` seam.
- [x] Past-only expanding and optional sliding history.
- [x] Inclusive interval Purge and session-based Pre-Test Gap.
- [x] Evidence-bearing summaries/traces and causal Evidence Channel.
- [x] Formal evaluator preserves the assignment channel.
- [x] Boundary, panel, invalid-fold, determinism, and evaluator tests pass.

## Answer

Implemented through the canonical Split Plan and evaluator seams. Seven focused causal
tests cover chronology, noncausal information, sliding windows, panel grouping, invalid
folds, weekend/session gaps, determinism, and evidence-channel propagation. The final
full suite passed with 52 tests on 2026-08-01.
