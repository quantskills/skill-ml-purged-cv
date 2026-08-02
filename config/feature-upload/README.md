# Governed feature-upload contracts

`feature-manifest.schema.json` and `upload-mapping.schema.json` are the version-1
authoring contracts. The runtime parser applies the same closed-field policy without a
JSON-Schema dependency.

The `raw` and `stationary` bundles are both valid: stationarity is not an admission
criterion. `intentional-leak` declares a target-derived feature and must be rejected.

From the repository root:

```powershell
python scripts/audit_feature_upload.py audit `
  --data config/feature-upload/examples/raw/features.csv `
  --manifest config/feature-upload/examples/raw/manifest.json `
  --mapping config/feature-upload/examples/raw/mapping.json
```

Replace `raw` with `stationary` for the second passing case or `intentional-leak` for
the fail-closed demonstration. The examples are intentionally small audit fixtures, not
effectiveness benchmarks.
