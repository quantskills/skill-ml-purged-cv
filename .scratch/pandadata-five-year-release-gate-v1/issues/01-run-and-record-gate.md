# 01 — Run and record the five-year governed gate

Type: task
Status: resolved
Blocked by: None

- [x] Confirm the local cache covers an exact trailing five-year window.
- [x] Apply continuous-contract and roll-crossing governance.
- [x] Rank frozen candidates on causal development regimes.
- [x] Compare PKF, CPCV, and causal Walk-Forward with overlap/sufficiency audits.
- [x] Consume the final Holdout once and preserve a redacted receipt.
- [x] Recover reporting without querying the consumed Holdout.
- [x] Record evidence and claim boundaries.

## Answer

The 2021-06-18 through 2026-06-18 run retained 88,417 eligible observations across 81
assets and 1,174 sessions. All structural channels reported zero overlap and sufficient
training samples. Ranking was stable, but an intercept-only mean model marginally beat
both ridge candidates. The one-attempt Holdout MSE was 0.0015686022812105493; its receipt
was reused only as immutable evidence after a report-serialization failure.

## Comments

Canonical evidence: `docs/evidence/pandadata-five-year-release-gate-20260802.md`.
