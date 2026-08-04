# PandaData Trainable Temporal-Model Leakage Benchmark v0.8.0 — 2026-08-04

## Decision

The same governed five-year dense panel was converted into a real supervised sequence
dataset and evaluated with fixed NumPy Ridge, LightGBM, and CPU PyTorch LSTM models.

```text
leakage control: PASS
unsafe overlap canary: DETECTED
Embargo incremental status: NO_INCREMENTAL_EXCLUSION_AFTER_FULL_INTERVAL_PURGE
production authorization: NOT_AUTHORIZED
```

This proves that Purge, Embargo policy execution, CPCV, and causal Walk-Forward enter the
actual fold-local training path for trainable temporal estimators. It does not prove that
any tested model has deployable forecasting or trading performance.

## Frozen data and model design

| Item | Registered value |
|---|---|
| source | existing PandaData dense NPZ; no new download or credential use |
| source window | 2021-08-03 through 2026-08-03 |
| supervised Decision Sessions | 1,185; 2021-08-31 through 2026-07-27 |
| assets / observations | 15 / 17,775 |
| feature sequence | each asset's own prior 20 Session returns, latest at `t-1` |
| label | simple forward return through `t+5` |
| Information Interval | earliest lag dependency through label completion |
| models | NumPy Ridge alpha 1; LightGBM 40 trees/depth 3; LSTM hidden 8/2 epochs |
| split policy | 5 folds; 20 Session Embargo; 5 Session pre-test gap |
| Walk-Forward | five fixed 120-Session test windows; minimum train 252 Sessions |
| CPCV | 6 groups / 2 test groups / 15 combinations / 5 complete paths |
| final report digest | `c3e8f962b950e307958ff622694b8595afd9d581bc73b9316f71f948a3f9cd38` |

The three model configurations and dependency versions participate in ModelSpec digests.
Every fold creates a fresh estimator. Ridge and LSTM standardization is fit only on that
fold's training side. No model or threshold was selected from these results.

## Structural overlap evidence

Overlap counts are sums of retained training positions whose complete Information
Intervals overlap the protected test intervals across the requested folds.

| Channel | Retained overlap | Formal evidence |
|---|---:|---|
| unsafe shuffled Session K-Fold | 71,100 | no; deliberate canary |
| chronological no-purge | 1,875 | no; deliberate canary |
| Purged K-Fold, no Embargo | 0 | yes |
| Purged K-Fold + 20 Session Embargo | 0 | yes |
| CPCV | 0 | yes |
| causal Walk-Forward | 0 | yes |

The counts are model-independent and were identical for all three estimators. Minimum
training geometry remained sufficient: Purged+Embargo retained at least 13,470
observations / 898 Sessions, CPCV 10,350 / 690, and Walk-Forward 8,400 / 560.

Purge removed 3,000 candidate positions across the five Purged folds, 15,000 across CPCV
combinations, and 1,875 across Walk-Forward windows. Embargo added zero incremental
exclusions because complete sample intervals already start 20 lags before the decision
and end at T+5. A post-test sample within the 20-Session Embargo zone therefore already
overlaps a protected test interval and is removed by interval-aware Purge. The system
keeps that truthful interval instead of narrowing it to manufacture an Embargo count.

## Model MSE comparison

Lower MSE is better. Unsafe gaps are safe-channel MSE minus shuffled K-Fold MSE;
positive means the unsafe result looked better, but the difference is not a pure causal
estimate of leakage bias.

| Model | unsafe shuffled | chronological no-purge | Purged | Purged+Embargo | CPCV overall | Walk-Forward | unsafe gap to Purged+Embargo | unsafe gap to WF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NumPy Ridge | 0.000959572 | 0.000851402 | 0.000963238 | 0.000963238 | 0.000963548 | 0.000851211 | +0.000003666 | -0.000108361 |
| LightGBM | 0.000959834 | 0.000843927 | 0.000959169 | 0.000959169 | 0.000963798 | 0.000844166 | -0.000000665 | -0.000115668 |
| PyTorch LSTM | 0.001012548 | 0.001613601 | 0.001012735 | 0.001012735 | 0.001801139 | 0.001820150 | +0.000000187 | +0.000807602 |

The score effect is model-dependent. Shuffled K-Fold is slightly optimistic versus
Purged Ridge and LSTM, but slightly worse than Purged LightGBM. Causal Walk-Forward is
better than shuffled validation for Ridge and LightGBM, while LSTM degrades strongly.
Therefore “ordinary K-Fold always reports a better number” is not supported; “ordinary
K-Fold retains forbidden information overlap” is directly supported.

## CPCV path MSE

| Model | median | P10 | P90 | worst |
|---|---:|---:|---:|---:|
| NumPy Ridge | 0.000963389 | 0.000962566 | 0.000964601 | 0.000965011 |
| LightGBM | 0.000963283 | 0.000961034 | 0.000967089 | 0.000968561 |
| PyTorch LSTM | 0.001810444 | 0.001700005 | 0.001892560 | 0.001898832 |

Ridge and LightGBM are close on CPCV paths. The fixed LSTM is both worse and less stable,
which is useful negative evidence rather than a reason to tune it on the same history.

## Interpretation and limits

- Purged K-Fold + Embargo is active on real trainable temporal estimators because every
  model is fitted on the splitter's retained positions and every safe overlap count is zero.
- Purge/Embargo is a validity control, not an accuracy enhancer.
- The continuous ratio-adjusted price panel is an offline approximation; exact contract
  fills, revisions, supplier PIT truth, transaction costs, capacity, and execution are not
  proven here.
- This five-year history is development evidence. It was not converted into a final
  untouched Holdout after observation.
- No model passed a production gate. LightGBM's lower causal MSE is not authorization to
  choose or deploy it without a frozen independent confirmation protocol.

## Verification

| Check | Observed result |
|---|---|
| new temporal-model and CLI focused tests | 17 passed |
| complete Pytest suite | 168 passed |
| complete repository quality gate | success; Ruff, format, strict MyPy, tests, pip check |
| full three-model formal CLI | success; current digest shown above |
| repeated numeric replay | Ridge, LightGBM, and LSTM metrics reproduced exactly |
| build / Twine | v0.8.0 wheel and sdist built; both artifacts passed |
| isolated installed-wheel canary | v0.8.0; six channels; PASS/NOT_AUTHORIZED decision preserved |

Local evidence does not publish, merge, deploy, or observe a production model.
