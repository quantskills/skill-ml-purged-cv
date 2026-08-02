# Implementation prompt — governed full-cache CPCV effectiveness evaluation

Implement the next evidence slice of `purged-kfold-validation` without changing the
meaning of existing Purged K-Fold, CPCV, Causal Walk-Forward, PandaAI adapter, or
four-channel benchmark behavior.

## Objective

Govern PandaAI asset identity and continuous-contract selection, then run one bounded
offline full-cache comparison of Purged K-Fold, general CPCV, and Causal Walk-Forward
using the same immutable dataset, fold-local baseline model, targets, and financial
diagnostic definitions.

## Public seams

- `governed_validation_dataset_from_pandaai_continuous_daily(...) -> GovernedPandaAIDataset`
- `run_cpcv_effectiveness_comparison(...) -> EffectivenessComparisonReport`
- `python scripts/evaluate_full_pandaai_cpcv.py --data-dir ...`

Tests must exercise these public seams rather than private helpers.

## Asset and continuous-contract governance

- Accept explicit columns for Session, series symbol, file-level/underlying asset,
  active contract (`dominant_id`), close, and features.
- Select only vendor-declared continuous rows whose series symbol contains the explicit
  `_DOMINANT.` token. Never relabel concrete contract rows into the continuous panel.
- Require exactly one continuous series identity per asset, one row per asset/session,
  a non-empty active-contract identity, and a non-empty explicit asset identity.
- Optionally cross-check a declared underlying column against the authoritative asset.
- Exclude assets with fewer source sessions than lookback plus label horizon and report
  their identities without failing an otherwise valid panel.
- Preserve an immutable governance receipt: input/selected/discarded rows, selected and
  excluded assets, active-contract transitions, eligible rows, label horizons crossing
  a transition, full Information Intervals crossing a transition, group breadth, source
  digest, and policy digest.
- Use the existing PandaAI adapter after governance so labels, Information Intervals,
  Decision Time, Feature Availability, and PIT evidence retain their canonical meaning.
- Treat vendor adjustment as declared but unverified; do not claim back-adjustment truth.

## Frozen metric definitions

- MSE: mean squared error of raw five-session forward simple returns.
- IC: mean per-session cross-sectional Spearman correlation between prediction and
  target, using only sessions with at least three assets and non-constant ranks.
- Diagnostic Sharpe: annualized mean/std of per-session zero-cost target returns formed
  by demeaning predictions cross-sectionally and normalizing absolute weights to unit
  gross exposure. Use `sqrt(252)`, no costs, no turnover or contract sizing.
- Every metric reports observations and valid sessions. Non-computable values fail
  closed rather than becoming NaN evidence.
- Sharpe is diagnostic only and must never be labeled executable or profitable.

## Comparison requirements

- Evaluate Causal Walk-Forward first to define the common OOS sample set, then evaluate
  Purged K-Fold and CPCV using the same estimator factory and model identity.
- Report native-coverage metrics for every channel and metrics restricted to the common
  Walk-Forward sample set.
- For CPCV, retain every path separately and report path metric values plus median,
  worst value, standard deviation, p10, and p90. MSE worst is maximum; IC and Sharpe
  worst are minimum.
- Report fold/combination training observations, sessions, and assets as min/median/max.
- For this full-cache baseline, audit explicit thresholds of at least 10,000 training
  observations, 252 training sessions, and 20 training assets per fold. These are
  task-specific gates, not universal statistical guarantees.
- Audit retained Information Interval overlap independently for every channel; safe
  channels must report zero.
- Report chronological CPCV group row and asset breadth so universe drift remains visible.
- Add a roll-clean sensitivity view restricted to samples whose five-session label does
  not cross an active-contract transition. Do not discard those rows from the primary
  175k dataset.

## Full-cache operator constraints

- Read only user-supplied local `*_daily.parquet` files; no credentials or network.
- Add file-level asset identity before concatenation and hash selected files.
- Emit deterministic redacted JSON only. Do not emit source rows, feature values, or
  credentials.
- Stop the bounded probe if it exceeds 20 minutes or exhausts memory.
- The result is a fixed ridge-baseline validation comparison, not an Alpha conclusion,
  strategy backtest, untouched holdout, production result, or profitability claim.

## Acceptance

1. Synthetic governance rejects mixed/duplicate/missing identity evidence.
2. Synthetic roll transitions produce exact receipt counts and clean masks.
3. Worked metric fixtures prove MSE, cross-sectional IC, and diagnostic Sharpe.
4. CPCV path distributions and common-period comparisons are deterministic.
5. Training sufficiency and overlap audits fail closed.
6. Existing tests remain green.
7. The full local cache run emits a receipt or truthfully records its bounded failure.
8. Preflight, project tests, strict typing, Ruff, build, Twine, and isolated wheel import pass.
