# Arbitrary-feature availability and lineage governance

Status: verified
Owner: project owner
Risk: high; silent future information and preprocessing contamination

## Acceptance stories

1. A caller can attach an ordered immutable manifest to any finite numeric feature
   matrix without the library assuming that raw or stationary features are required.
2. Every feature has source, transformation, code/vintage, lookback, revision, target
   dependency, and lifecycle evidence.
3. Every uploaded feature cell has an explicit availability time no later than Decision
   Time.
4. Precomputed stateful or target-dependent uploaded features are rejected and directed
   to fold-local transformer factories.
5. Fold-local transformer identity is bound into the evaluation run and OOS Ledger.
6. The existing explicit pandas boundary can return a governed upload without inferring
   metadata.
7. Receipts and failures reveal no feature values.

## Non-goals

Automatic stationarity tests, formula reconstruction from values, truth verification of
user declarations, AS-OF storage engines, online serving, training/serving parity,
feature computation, costed backtests, Alpha claims, and deployment.
