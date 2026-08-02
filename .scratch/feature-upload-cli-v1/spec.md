# Spec: governed arbitrary-feature upload v0.5.1

Status: approved

## Problem

The library can govern arbitrary in-memory feature matrices, but a user cannot yet hand
it a local file through a stable, versioned contract. A safe delivery layer must separate
metadata audit from model evaluation and must reject unverifiable lifecycle declarations
before any fitting begins.

## Input contracts

The data file is CSV or Parquet. The mapping document explicitly names the sample,
session, asset, information-interval, decision-time, target, ordered feature, and ordered
availability columns, the authoritative Session Axis, and the PIT Snapshot. The manifest
contains one ordered Feature Definition per mapped feature and binds all definitions to
the PIT source-bundle digest. Both documents use `schema_version: "1"` and reject unknown
fields.

The implementation validates the contract without adding a runtime JSON-Schema
dependency. The checked-in JSON Schemas are the portable authoring/interchange contract.

## Audit command

`audit` performs bounded file loading, exact contract parsing, pandas boundary mapping,
canonical dataset validation, and feature lifecycle governance. Success returns redacted
input metadata, applied limits, dataset counts/digests, and the governance receipt.
It must not instantiate a transformer or estimator.

## Evaluate command

`evaluate` first executes the audit path, checks `C(n_groups, n_test_groups)` against the
combination limit, and evaluates the governed dataset with a deterministic fold-local
ridge estimator. The output adds the canonical three-channel effectiveness report and
retains its explicit claim boundary.

## Resource defaults

- maximum data-file bytes: 536,870,912
- maximum rows: 1,000,000
- maximum features: 512
- maximum CPCV combinations: 10,000

Every limit is overridable downwards or upwards by an explicit positive CLI integer.
The combination check occurs before split planning/evaluation.

## Rejection behavior

Contract, parsing, governance, limit, split, or evaluation failures return exit code `2`
and a JSON object containing only `status`, `stage`, `error_type`, and a safe message.
Unexpected parser details and source values are not exposed.

## High-level transformer binding

`run_cpcv_effectiveness_comparison()` accepts `transformer_factories` and
`transformer_specs`, passes them unchanged into every `LeakageSafeEvaluator`, and adds the
ordered spec digests to `EffectivenessComparisonReport`. The existing evaluator enforces
one-to-one identity and fresh instances per fold.

## Non-goals

HTTP upload, a web UI, remote storage, automatic feature engineering, arbitrary user-code
execution, formula verification from numeric values, nested HPO, deployment, and profit
claims are outside this slice.

## Acceptance criteria

1. Valid CSV audit succeeds without training and contains no row values or absolute path.
2. Valid CSV evaluation reports Purged K-Fold, CPCV, and causal Walk-Forward channels.
3. Parquet follows the same loader contract when the optional engine is installed.
4. Late availability, target-dependent/stateful features, mismatched ordering/digests,
   malformed mappings, unsupported suffixes, and exceeded limits fail closed with code 2.
5. Transformer spec changes alter the high-level report digest.
6. Raw and stationary examples audit successfully; intentional leak fails.
7. Full repository verification and isolated wheel import pass.
