# PandaAI offline benchmark receipt — 2026-08-01

Status: observed local evidence

## Configuration

- Source: five local PandaAI futures daily parquet snapshots (`SF`, `LH`, `CU`, `FG`, `SM`)
- Network/authentication: none
- Rows after declared warm-up/tail eligibility: 12,259
- Assets: 5
- Trading Sessions: 2,745
- Label horizon: 5 per-asset sessions
- Feature information lookback: 20 per-asset sessions
- Features: close, volume, open interest
- Folds: 5
- Purged K-Fold Embargo: 20 sessions
- Causal Pre-Test Gap: 20 sessions
- Model: fold-local deterministic ridge baseline
- Metric: mean squared error
- Dataset digest: `8e128009ce97ff32c08f64136f6d6df391b2df09d06bb5297030f4a5f02955048`
- Source digest: `a8ff074c4efebb330a494dff8b0f1f1ba8e79861d24da0e4b7030f4e89f8a615`
- Report digest: `840cc962c6a04fc8c38024089f0f025cd92b18de43fd9cb42e420265b1058b08`

## Results

| Channel | Evidence channel | OOS observations | Coverage | Retained overlap count | MSE |
|---|---|---:|---:|---:|---:|
| unsafe-shuffled-kfold | diagnostic-unsafe-baseline | 12,259 | 1.0000 | 49,036 | 0.0014990205 |
| chronological-no-purge | diagnostic-chronological-baseline | 10,419 | 0.8499 | 528 | 0.0015228816 |
| purged-kfold | model-selection | 12,259 | 1.0000 | 0 | 0.0015025414 |
| causal-walk-forward | causal-walk-forward | 10,419 | 0.8499 | 0 | 0.0015236896 |

## Interpretation boundary

This receipt proves that the implementation ran on the declared local cache and that
the retained-overlap audit distinguished unsafe, chronological, Purged, and causal
channels. It does not prove profitability, vendor metadata truthfulness, untouched
holdout performance, cross-platform behavior, or production readiness. Metric ordering
is reported as observed and is not an acceptance assumption.
