# Feature governance research receipt — 2026-08-01

## Agent Reach coverage

- Agent Reach doctor initially appeared unavailable inside the Windows sandbox; the
  approved read-only WSL retry reported Exa ready and GitHub degraded because `gh` is
  not installed.
- Exa semantic search was executed through `mcporter/exa` and returned eight public
  candidates. Coverage: **effective**, not deep-read. Content hash:
  `0821551def42a939f6e1d7255b0c0835212b700f086922b1fe2808ad8469aff4`.
- The planned Agent Reach GitHub route was not executed because its declared backend
  was unavailable. It is not counted as attempted/effective GitHub coverage.
- Strong candidates were then deep-read through their public official documentation.

## Design findings

1. Point-in-time correctness requires selecting values available no later than each
   observation's decision timestamp, normally with an AS-OF join. A row/event timestamp
   by itself is not availability evidence.
2. Feature history needs a declared source vintage. Latest revised historical values
   cannot substantiate what was known at an earlier decision time.
3. A feature contract needs stable schema/name, source fields, transformation identity,
   version/code identity, parameters, and dependencies. Column-level lineage should
   distinguish identity, transformation, aggregation, join/window, and indirect inputs.
4. Stateful preprocessing and target-dependent transformations cannot be accepted as
   one globally precomputed uploaded matrix. Their state must be fitted inside each
   fold and their specifications must be bound into evaluation identity.
5. The validator can verify declared timestamps, digests, lifecycle rules, and fold
   isolation. It cannot infer from numeric values whether a user lied about a formula,
   timestamp, revision policy, or source.

## Primary sources deep-read

- Feast point-in-time joins:
  https://docs.feast.dev/v0.18-branch/getting-started/concepts/point-in-time-joins
- Databricks point-in-time feature joins:
  https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series
- scikit-learn common data-leakage pitfalls and Pipeline guidance:
  https://scikit-learn.org/stable/common_pitfalls.html
- OpenLineage facets and column-level lineage:
  https://openlineage.io/docs/spec/facets/
  https://openlineage.io/docs/spec/facets/dataset-facets/column_lineage_facet/

## Bounded conclusion

Implement a small evidence contract, not a feature store: immutable feature definitions
and manifest, per-feature availability enforcement, source/transform lineage digests,
uploaded-feature lifecycle restrictions, a governance receipt, and explicit fold-local
transformer specifications. Do not attempt automatic stationarity or semantic leakage
detection.
