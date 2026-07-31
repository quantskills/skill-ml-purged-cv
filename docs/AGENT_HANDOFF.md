# Agent handoff

Follow `AGENTS.md`; this is not a competing read order.

## Current delivered slice

Slice 1 provides leakage-safe Purged K-Fold model-selection evidence: explicit
Information Intervals and PIT provenance, Session Axis Embargo, indivisible panel
sessions, Fold-Local evaluation, raw OOS facts, versioned metrics, and an optional
explicit pandas adapter.

CPCV, Causal Walk-Forward, holdout governance, nested HPO, persistence, and production
deployment remain later work items and must not be claimed from Slice 1 results.

## Current structure

`domain.py` owns immutable evidence, `validation.py` owns cross-field/PIT checks,
`leakage.py` owns Purge/Embargo semantics, splitters own geometry, and the evaluator owns
Fold-Local execution. Optional adapters translate inputs without owning leakage rules.

## Program authority

`PROGRAM.md` defines durable direction and boundaries; change it only for an explicit long-term program decision.

## Verification

Canonical entrypoints are `python tasks/preflight.py` and `python tasks/test.py`; they run natively on Windows and POSIX.

## Known unknowns

Product schemas, external wiring, deployment targets, and production readiness remain pending owner definition.
