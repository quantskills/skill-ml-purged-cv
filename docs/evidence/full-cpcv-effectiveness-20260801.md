# Full PandaAI CPCV effectiveness receipt — 2026-08-01

## Outcome

The governed offline run completed successfully in 732.6 seconds. It evaluated the
same immutable dataset and fold-local ridge baseline through Purged K-Fold, CPCV
`N=6, k=2`, and Causal Walk-Forward. All three channels retained zero overlapping
Information Intervals. This is model-selection/robustness evidence, not an executable
strategy, costed backtest, untouched holdout, profitability result, or deployment
evidence.

## Identity and continuous-contract governance

| Item | Observed result |
|---|---:|
| Local parquet files | 90 |
| Raw rows | 189,317 |
| Vendor-declared continuous rows | 177,073 |
| Concrete-contract rows discarded | 12,244 |
| Eligible observations after lookback/horizon | 175,129 |
| Assets / sessions | 81 / 2,745 |
| Active-contract transitions | 3,826 |
| Five-session labels crossing a roll | 19,012 (10.856%) |
| Full information intervals crossing a roll | 81,923 (46.779%) |
| Roll-clean observations | 156,117 |

The authoritative asset is the commodity code derived from each cache filename.
Only rows whose vendor series symbol contains `_DOMINANT.` are selected. Concrete
contracts are never relabelled into that asset/session panel. The selected continuous
rows all supplied `underlying_symbol` matching the filename asset and a non-empty
`dominant_id`. Nine concrete-only files were excluded because they had no declared
continuous rows: `AD`, `BZ`, `L_F`, `OP`, `PD`, `PL`, `PP_F`, `PT`, and `V_F`.
The vendor's adjustment/back-adjustment method remains unverified.

Source digest: `7776c9be6ed2bb6edceb0f61322db56da4904b1de67331c4557591c176d009cc`

Dataset digest: `8020a52a869617af6de00162089064aa574da87cb40c5ffaaba5e30be8023c14`

Governance receipt: `ea07894079358be54d2c2c40ffa09ca5b2e09ccd1931c754f5d74b47bb2f235d`

## CPCV native path results

Every path covers all 175,129 eligible observations and 2,745 sessions.

| Path | MSE | Cross-sectional IC | Diagnostic Sharpe |
|---:|---:|---:|---:|
| 0 | 0.007839078 | -0.013508 | -1.009359 |
| 1 | 0.007839227 | -0.000214 | -0.855980 |
| 2 | 0.007838394 | -0.006892 | -0.987205 |
| 3 | 0.007838887 | -0.001001 | -0.966524 |
| 4 | 0.007839121 | -0.002934 | -0.934046 |

| Metric | Median | Worst | Standard deviation | p10 | p90 |
|---|---:|---:|---:|---:|---:|
| MSE | 0.007839078 | 0.007839227 | 0.000000295 | 0.007838591 | 0.007839185 |
| IC | -0.002934 | -0.013508 | 0.004880 | -0.010862 | -0.000528 |
| Diagnostic Sharpe | -0.966524 | -1.009359 | 0.053429 | -1.000497 | -0.887206 |

## CPCV common-period path results

The common sample is the 152,508 observations and 2,285 sessions scored by causal
Walk-Forward. It is the fair primary comparison view across channels.

| Path | MSE | Cross-sectional IC | Diagnostic Sharpe |
|---:|---:|---:|---:|
| 0 | 0.008774748 | -0.011763 | -1.024581 |
| 1 | 0.008774918 | -0.003066 | -1.164926 |
| 2 | 0.008773839 | -0.005432 | -1.092724 |
| 3 | 0.008774393 | 0.002257 | -1.007219 |
| 4 | 0.008774669 | -0.000132 | -0.971121 |

| Metric | Median | Worst | Standard deviation | p10 | p90 |
|---|---:|---:|---:|---:|---:|
| MSE | 0.008774669 | 0.008774918 | 0.000000378 | 0.008774060 | 0.008774850 |
| IC | -0.003066 | -0.011763 | 0.004829 | -0.009231 | 0.001301 |
| Diagnostic Sharpe | -1.024581 | -1.164926 | 0.068861 | -1.136045 | -0.985560 |

## CPCV roll-clean sensitivity paths

This view removes only labels whose five-session target horizon crosses an active
contract transition. It does not replace the primary all-sample result.

| Path | MSE | Cross-sectional IC | Diagnostic Sharpe |
|---:|---:|---:|---:|
| 0 | 0.002727135 | -0.014937 | -0.940065 |
| 1 | 0.002727156 | 0.001314 | -0.481045 |
| 2 | 0.002726284 | -0.004988 | -0.411797 |
| 3 | 0.002727149 | 0.004371 | -0.345845 |
| 4 | 0.002727537 | -0.005067 | -0.878704 |

| Metric | Median | Worst | Standard deviation | p10 | p90 |
|---|---:|---:|---:|---:|---:|
| MSE | 0.002727149 | 0.002727537 | 0.000000413 | 0.002726624 | 0.002727385 |
| IC | -0.004988 | -0.014937 | 0.006632 | -0.010989 | 0.003148 |
| Diagnostic Sharpe | -0.481045 | -0.940065 | 0.247720 | -0.915521 | -0.372226 |

## Common-period channel comparison

| Channel | Observations | MSE | IC | Diagnostic Sharpe |
|---|---:|---:|---:|---:|
| Purged K-Fold | 152,508 | 0.008774310 | -0.001644 | -1.128645 |
| CPCV path median | 152,508 per path | 0.008774669 | -0.003066 | -1.024581 |
| Causal Walk-Forward | 152,508 | 0.008774267 | -0.002284 | -0.854780 |

MSE is effectively indistinguishable across the three channels at this precision.
CPCV exposes meaningful path risk in IC and diagnostic Sharpe that one Purged K-Fold
score cannot show. The fixed baseline has negative median IC and negative diagnostic
Sharpe in every primary/common comparison, so it provides no evidence of positive
predictive or trading performance.

## Training sufficiency and universe breadth

The task-specific thresholds were 10,000 observations, 252 sessions, and 20 assets.

| Channel | Minimum train observations | Minimum train sessions | Minimum train assets | Gate |
|---|---:|---:|---:|---|
| Purged K-Fold | 136,552 | 2,213 | 78 | pass |
| CPCV (15 combinations) | 102,001 | 1,707 | 69 | pass |
| Causal Walk-Forward | 21,421 | 436 | 50 | pass |

CPCV chronological groups contain 22,521, 23,490, 27,394, 31,247, 33,772, and
36,705 observations. Asset breadth rises from 50 in the earliest group to 81 in the
latest group, so universe drift remains a material interpretation caveat even though
every combination is comfortably above the frozen sufficiency gates.

Comparison report digest: `c6401149a2c5c9713fec5b719bce3b2cd7d1bf1706a3db39f600c69bdddf565a`
