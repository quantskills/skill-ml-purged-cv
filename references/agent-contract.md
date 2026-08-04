# Agent request and result contract

## Request v1

Pass one UTF-8 JSON object to `purged-cv-skill run --request`. Relative input paths resolve
against the request file directory. Unknown fields and options fail closed.

```json
{
  "schema_version": "1",
  "action": "evaluate",
  "data_path": "features.parquet",
  "manifest_path": "manifest.json",
  "mapping_path": "mapping.json",
  "timeout_seconds": 900,
  "options": {
    "n_groups": 6,
    "n_test_groups": 2,
    "walk_forward_splits": 5,
    "embargo_sessions": 20,
    "pre_test_gap_sessions": 5,
    "max_combinations": 10000,
    "min_train_observations": 1000,
    "min_train_sessions": 252,
    "min_train_assets": 1,
    "ridge_alpha": 1.0
  }
}
```

The request is only an orchestration envelope. The feature manifest and mapping remain the
authoritative lineage and physical-column contracts. Generate their current schemas with
`purged-cv-upload schema --kind manifest` and `--kind mapping`.

## Result v1

The command emits exactly one JSON document. `authoritative_cli_result` is the unchanged
output of the maintained validation engine. `request_digest` binds the receipt to the
canonical request without exposing local paths.

```json
{
  "schema_version": "1",
  "status": "success",
  "action": "audit",
  "request_digest": "<64 lowercase hexadecimal characters>",
  "engine": {
    "name": "purged-kfold-validation",
    "version": "0.9.0"
  },
  "authoritative_cli_result": {
    "status": "success",
    "stage": "audit",
    "dataset": {
      "observations": 36,
      "features": 1,
      "sessions": 12,
      "assets": 3
    }
  },
  "warnings": [
    "Structural leakage controls and model metrics are not profitability claims."
  ],
  "errors": []
}
```

Use `purged-cv-skill schema --kind request|result` for the complete machine-readable schema.

## Evidence selection

| Decision | Required evidence |
|---|---|
| Compare features or models without interval overlap | Purged K-Fold |
| Inspect sensitivity across historical test-group combinations | CPCV Path distribution |
| Approximate causal retraining and prediction | Causal Walk-Forward |
| Confirm one frozen design once | Governed Holdout |
| Audit selection across candidate strategy returns | CSCV/PBO + DSR + CPCV selected paths + Causal Walk-Forward |

Purged K-Fold and CPCV coexist. CPCV does not replace Purged K-Fold or Causal Walk-Forward.

## Required interpretation

Report dataset/configuration/model/manifest identities, eligible observations, assets,
Trading Sessions, training coverage per fold, Purge/Embargo/Pre-Test Gap exclusions, retained
Information Interval overlaps, per-fold or per-path metrics, median, worst quantile,
dispersion, ranking stability, channel differences, rejections, and limitations.

Structural success means the retained assignments satisfy the declared leakage controls. It
does not establish profitability, external timestamp truth, market capacity, execution cost,
or deployment readiness.
