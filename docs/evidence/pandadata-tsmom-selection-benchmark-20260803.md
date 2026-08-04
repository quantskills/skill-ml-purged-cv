# PandaData TSMOM Selection-Overfitting Benchmark — 2026-08-03

## Decision

The real five-year PandaData TSMOM run proves that the installed time-series benchmark
works end to end on a multi-asset continuous-price panel. The validation tool passes.
The tested 32-candidate TSMOM family does **not** pass the pre-registered production
statistical-strength example because DSR remains below 0.95 in every cost scenario.

Canonical decision:

```text
validation tool: PASS
candidate-rank stability: PASS
path and causal return sign: PASS
deflated statistical strength: FAIL
production strategy acceptance: FAIL
```

This is not evidence that the Purged/CPCV project fails on time series. It is evidence
that the project rejects an attractive full-sample result when multiple-trial-adjusted
strength is insufficient.

## Governed input

| Item | Observed value |
|---|---|
| PandaData method | `get_future_daily_post`, `close_pcr` |
| requested / downloaded symbols | 15 / 15; no failed symbol |
| window | 2021-08-03 through 2026-08-03 |
| common dense Sessions | 1,210 |
| assets | A, AL, AU, CF, CU, IF, M, MA, P, RB, RU, SC, SR, TA, Y |
| candidate strategies | 32 pre-registered TSMOM configurations |
| cost scenarios | 0, 1, 3, 5 bps; not counted as additional trials |
| CSCV | 8 groups; 70 complementary splits |
| CPCV | 6 groups, 2 test groups, 15 combinations, 5 paths, 5-Session embargo |
| Walk-Forward | 252-Session minimum training, 5 expanding windows |
| source digest | `6a649a8ce9e5af73c729c5e7daedcc44a1e4f008cdbbe0443455cf157e779976` |
| direct report digest | `0cfbf6b5ed4cf239f5971e82d1ac49a7d7e6eae490c9f92afe3a021c1996c5d0` |
| formal NPZ/JSON CLI report digest | `4cc75b047f8ec2b1484b0f043d7f8a6240513b64e2adb2f4d1b0c33594a19a60` |

The two report digests differ because the direct run binds the governed extracted-panel
digest while the CLI binds the serialized NPZ file digest. Their financial and selection
metrics agree to floating-point precision.

PandaData authentication was used only for this requested download. Credentials were
passed through process environment, removed from the Python process after token
initialization, and were not written to source, request, NPZ, Parquet, JSON evidence, or
Git-tracked files. Raw runtime files remain under the repository's ignored
`.scratch/**/runtime/` path.

## Results

The selected full-sample candidate is identical in all four cost scenarios:
`tsmom-l021-v020-r21-c2` — 21-Session own-return direction, 20-Session lagged volatility,
21-Session rebalance, and leverage cap 2.

| Cost (bps) | PBO | DSR probability | Best full-sample Sharpe | CPCV median / P10 / worst | Walk-Forward Sharpe | Mean regret | Optimism gap |
|---:|---:|---:|---:|---|---:|---:|---:|
| 0 | 0.0714 | 0.7553 | 1.0827 | 0.8850 / 0.8361 / 0.8035 | 0.7665 | 0.8651 | 0.3162 |
| 1 | 0.0714 | 0.7390 | 1.0612 | 0.8636 / 0.8160 / 0.7842 | 0.7412 | 0.8715 | 0.3200 |
| 3 | 0.0571 | 0.7034 | 1.0182 | 0.8207 / 0.7756 / 0.7455 | 0.6905 | 0.8841 | 0.3277 |
| 5 | 0.0429 | 0.6640 | 0.9750 | 0.7068 / 0.5389 / 0.5258 | 0.6396 | 0.8967 | 0.3353 |

At zero cost, the best candidate has cumulative return 22.74% and maximum drawdown
3.37% on the continuous-return approximation. The causal Walk-Forward track has
cumulative return 10.93%, maximum drawdown 3.95%, and Sharpe 0.7665. At 5 bps, the
Walk-Forward cumulative return falls to 8.99%, maximum drawdown rises to 4.32%, and
Sharpe falls to 0.6396.

## Gate interpretation

| Gate | Example threshold | Observed | Result |
|---|---:|---:|---|
| PBO | `< 0.20` | 0.0429–0.0714 | PASS |
| DSR probability | `>= 0.95` | 0.6640–0.7553 | FAIL |
| CPCV median Sharpe | `> 0` | 0.7068–0.8850 | PASS |
| CPCV worst Sharpe | `> 0` | 0.5258–0.8035 | PASS |
| Walk-Forward Sharpe | `> 0` | 0.6396–0.7665 | PASS |
| cost degradation | monotonic disclosure | Sharpe and return decline as cost rises | PASS |

PBO says the parameter winner's relative rank is not behaving like a random historical
accident. CPCV says the selected strategy remains positive across reconstructed paths.
Walk-Forward says causal selection remains positive. DSR still says the observed Sharpe
does not clear the required multiple-testing-adjusted confidence. Therefore the correct
decision is not “time-series validation failed”; it is “promising but statistically
insufficient for production acceptance.”

## Limitations

- `close_pcr` ratio-adjusted dominant continuous prices are used for both signal and
  continuous-return approximation. They remove roll discontinuities but do not reproduce
  exact old-contract/new-contract fills.
- The bps model covers turnover pressure, not bid/ask spread, market impact, capacity,
  margin, limit moves, or broker execution.
- The panel contains 15 diversified Chinese futures underlyings, not every available
  market.
- This run uses CPCV and expanding Walk-Forward but does not consume a new one-attempt
  frozen Holdout. It cannot authorize deployment.

The v0.7.1 pre-registered decision replay and DSR sample-gap diagnostic are recorded in
`docs/evidence/strategy-acceptance-decision-v071-20260804.md`.
