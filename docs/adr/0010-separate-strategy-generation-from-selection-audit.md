# ADR 0010: Separate Strategy Generation from Selection Audit

## Status

Accepted

## Context

The existing benchmark proves structural leakage controls with one fold-local model. A real overfitting benchmark needs many candidate net-return tracks, while users may supply strategies unrelated to the built-in reference strategy. Embedding TSMOM assumptions inside the cross-validation core would make the validation library strategy-specific; accepting only a generic return matrix would leave the project without a reproducible end-to-end canary.

## Decision

Place a generic `StrategyReturnMatrix` analysis seam between strategy generation and selection-overfitting statistics. Provide TSMOM as one deterministic built-in adapter at that seam. Keep the existing Ridge/MSE comparison as a separately named structural leakage canary.

The selection audit owns CSCV/PBO, DSR, CPCV selected paths and causal walk-forward selection. The TSMOM adapter owns only strictly lagged signal, position, turnover and offline cost construction. Neither module owns order execution, capacity analysis or a profitability claim.

## Consequences

- Any caller can audit its own candidate strategy returns without adopting TSMOM.
- The project retains an end-to-end, reproducible time-series reference benchmark.
- Strategy and validation bugs can be tested independently through the same return-matrix seam.
- Continuous-contract and point-in-time source truth remain caller governance responsibilities.
