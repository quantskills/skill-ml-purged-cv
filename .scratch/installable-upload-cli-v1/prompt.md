# Implementation prompt: installable governed-upload CLI v0.5.2

Turn the repository-only governed feature-upload script into a distribution-native CLI
without changing audit/evaluation semantics or weakening the NumPy-only core boundary.

## Objective

A user who installs the wheel with the upload extra must be able to audit/evaluate local
CSV or Parquet features, inspect the exact versioned JSON Schemas, and materialize any
checked-in example bundle without cloning the repository.

## Confirmed public seams

1. `purged-cv-upload <command>` console entry point.
2. `python -m purged_kfold_validation <command>` equivalent module entry point.
3. Existing `python scripts/audit_feature_upload.py <command>` remains compatible.
4. `schema --kind manifest|mapping` prints the installed canonical schema as JSON.
5. `example --name raw|stationary|intentional-leak --output-dir PATH` safely writes one
   installed example bundle and refuses to overwrite an existing target file.

## Packaging and dependency rules

- Move the executable behavior into package source; the repository script becomes a thin
  compatibility wrapper.
- Package schemas and the three examples as wheel data and test them against the canonical
  repository copies to prevent drift.
- Add an `upload` optional dependency containing pandas and a Parquet engine. Do not add
  pandas to the base NumPy-only runtime.
- Add the console script through standard project metadata.
- Schema/example discovery must work without importing pandas.
- Audit/evaluate behavior, JSON structure, redaction, limits, and exit codes remain
  compatible with v0.5.1.
- Resource export may write only beneath the explicit output directory, must validate
  the fixed package-owned filenames, and must fail before overwriting any existing file.
- Never execute uploaded code, access a network, read credentials, or infer metadata.

## Verification

Use public subprocess seams and the built wheel. Observe both module and console commands
in an isolated environment, resource equality, safe no-overwrite behavior, legacy script
compatibility, full tests, strict typing, lint, build, Twine, and wheel contents.
