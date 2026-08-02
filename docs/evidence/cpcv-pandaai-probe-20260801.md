# CPCV PandaAI structural probe receipt — 2026-08-01

Status: observed local evidence

## Configuration

- Source: five local PandaAI futures daily parquet snapshots (`SF`, `LH`, `CU`, `FG`, `SM`)
- Network/authentication: none
- Eligible observations: 12,259
- Assets: 5
- Trading Sessions: 2,745
- Label horizon: 5 per-asset sessions
- Feature information lookback: 20 per-asset sessions
- Features: close, volume, open interest
- CPCV groups/test groups: `N=6`, `k=2`
- Combination budget: default 10,000
- Embargo: 20 sessions
- Source digest: `a8ff074c4efebb330a494dff8b0f1f1ba8e79861d24da0e4b7030f4e89f8a615`
- Dataset digest: `369a978b6ba0876c6a677b653fe286d014809d57ef2a1793bf1cc42b6dbda917`
- Split-spec digest: `d9e330ef6992db4abd84344867b1a6b4a3964b66c2b1720d38846a1d742f8f92`
- Plan digest: `93b8e3c5be2ef99e24defd46684d3ba5b38a5c4d5c22209f5708c9ef314eca5c`

## Structural result

| Evidence | Observed value |
|---|---:|
| combinations | 15 |
| complete CPCV Paths | 5 |
| Purged candidate observations across combinations | 4,224 |
| Embargoed candidate observations across combinations | 88 |
| independently audited retained interval overlaps | 0 |

## Claim boundary

This receipt proves bounded CPCV planning and retained-overlap behavior on the declared
local cache. It does not contain a strategy result and does not prove profitability,
causal deployment performance, vendor metadata truthfulness, Untouched Holdout
performance, cross-platform behavior, or production readiness.
