# PandaData five-year governed release gate — 2026-08-02

## Decision

The leakage-control implementation passes the local five-year structural and governance
gate. It is suitable for a controlled local/beta release, but the observed baseline is
not evidence of trading value or production deployment readiness.

The local cache already covered the requested window, so this run did not authenticate
to PandaData, make a network request, or persist credentials. Runtime Parquet data and
Holdout store files remain outside canonical package inputs.

## Repository and artifact acceptance

| Check | Observed result |
|---|---|
| `python tasks/preflight.py` | success |
| `python tasks/test.py` | success |
| complete pytest suite | 132 passed |
| strict mypy | success; 49 source files |
| Ruff check / format check | all checks passed; 49 files formatted |
| `python -m pip check` | no broken requirements |
| `python -m build --no-isolation` | 0.6.1 sdist and universal wheel built |
| Twine check | both 0.6.1 artifacts passed |
| isolated installed-wheel canary | version 0.6.1, JSON ranking report, and packaged manifest schema succeeded |

The initial combined venv-install wrapper timed out after pip had reported a successful
installation. The subsequent import/schema canary was run separately from the same
venv, resolved the package from its `site-packages`, and exited successfully.

## Frozen input and boundaries

| Item | Observed value |
|---|---|
| window | 2021-06-18 through 2026-06-18 |
| local files / input rows | 90 / 102,605 |
| source digest | `e26f9073e32b0fbe45d89cded68f83cff91e368d1ce88b8f13c49c445c516797` |
| eligible dataset | 88,417 observations, 81 assets, 1,174 sessions |
| dataset digest | `3a5e027512d14fa1d881491db295bab6d179c2f71c78a7d5e6ab6f703fb352d9` |
| development | 66,061 observations, 898 sessions |
| final Holdout | 20,412 observations, 252 sessions, beginning 2025-05-09 |

Continuous-contract governance selected 90,361 declared-continuous rows, discarded
12,244 concrete-contract rows, and excluded no asset for insufficient history. It
identified 46,890 Information Interval roll-crossing rows, 11,080 label roll-crossing
rows, and 2,242 active-contract transitions. Eligibility applies the combined
governance and finite/PIT requirements; the counts are diagnostic categories and must
not be summed as disjoint rejection reasons.

## Development model selection

The frozen candidates were an intercept-only mean and standardized ridge models with
alpha 1 and 100. Selection used causal Walk-Forward MSE on development data only.

| Candidate | Overall development MSE |
|---|---:|
| mean | 0.0011892708254843901 |
| ridge-100 | 0.0011904686574452687 |
| ridge-1 | 0.001190501905978517 |

The ordering was identical in all three chronological regimes (minimum pairwise
Spearman 1.0). Mean MSE rose from 0.0009666698642839209 to 0.0011729879470912252 and
then 0.0014067469177058585. The ranking is stable, but the margins are small and the
more complex models did not improve on an intercept. This is negative feature/model
effectiveness evidence, not a failure of the leakage controls.

## Structural channel comparison

The same fixed ridge-1 model was used for the structural comparison so IC and
diagnostic Sharpe remained defined. Every channel retained zero training/test
Information Interval overlaps.

| Channel | Common paths | Coverage | Median MSE | Median cross-sectional IC | Median diagnostic Sharpe | Minimum train observations / sessions / assets |
|---|---:|---:|---:|---:|---:|---|
| Purged K-Fold | 1 | 1.0000 | 0.001184221678527877 | -0.0192539397757504 | -3.0100055005823627 | 50,617 / 699 / 78 |
| CPCV | 5 | 1.0000 | 0.0011849285333831517 | 0.00020523221959390086 | -1.1866100996616875 | 36,554 / 500 / 78 |
| Causal Walk-Forward | 1 | 0.6861 | 0.001185375524277935 | 0.004547313432522602 | -0.45699192372537956 | 19,054 / 276 / 70 |

CPCV path MSE ranged from 0.0011844019571498456 to 0.001185356710974918; path IC
ranged from -0.007775302670370636 to 0.006530220124702608; diagnostic Sharpe ranged
from -2.190842427993466 to -0.8290056364938254. Each common path contained 45,327
observations across 598 sessions. This narrow MSE range shows path consistency, while
IC near zero and negative diagnostic Sharpe do not support a trading-performance claim.

Comparison digest:
`501401aae88d403aa40a4191d392764f7db282b31e783124472da8f4c0667c61`.

## One-attempt Holdout and recovery receipt

The selected mean model evaluated the Holdout once and produced MSE
0.0015686022812105493. Protocol digest:
`2d875f28d4fed7520bd70dd91e91145bca60f427fdc0125a826a46e4ea77c92c`.
Receipt digest:
`245bf378ef474b95337771beb03822bb13ae9ce067fe5a4606ddadb9dc1355bd`.

After the receipt was durably written, report serialization failed because a NumPy
integer was not JSON serializable. The Holdout was not run again. The serializer was
corrected to emit native scalars and validate the complete payload before exclusive
file creation. Recovery recomputed development/structural evidence, loaded the existing
receipt, and verified its protocol and Holdout-dataset digests. Because predictions are
intentionally absent from the redacted receipt, Holdout IC and Sharpe remain null rather
than being reconstructed through prohibited Holdout reuse.

## Claim boundary and release recommendation

This establishes local implementation behavior on a substantial five-year panel:
interval Purge, session Embargo/Pre-Test Gap, CPCV path construction, training
sufficiency, ranking stability, and one-attempt Holdout governance. It does not prove
vendor metadata truth, feature lineage truth, stationarity, transaction-cost-adjusted
performance, profitability, remote CI, package publication, deployment, rollback, or
production observation. A controlled beta is reasonable because the complete repository
gate and 0.6.1 artifact checks pass; public production release still requires the
external CI/repository/deployment gates.
