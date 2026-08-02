# Spec: installable governed-upload CLI v0.5.2

Status: approved

## Problem

Version 0.5.1 ships the upload adapter in the wheel but leaves its operator script,
schemas, and examples in repository-only paths. An installed user therefore cannot use
the documented workflow without cloning the source tree.

## Command contract

`purged-cv-upload` and `python -m purged_kfold_validation` expose identical `audit`,
`evaluate`, `schema`, and `example` commands. The legacy repository script delegates to
the same `main()` implementation. Audit/evaluate retain the v0.5.1 JSON and exit contract.

`schema --kind manifest` and `schema --kind mapping` emit the exact installed JSON Schema
document to stdout and exit 0. `example --name ... --output-dir ...` writes only
`features.csv`, `manifest.json`, and `mapping.json`, then emits a JSON receipt containing
the example name and those relative filenames. If any target already exists, it writes
nothing and exits 2 with safe JSON.

## Distribution contract

The wheel contains both schemas and all three example bundles under package resources.
Repository and packaged resources are byte-identical. Project metadata declares:

- version `0.5.2`;
- console entry point `purged-cv-upload`;
- optional `upload` extra with pandas and pyarrow;
- explicit package-data patterns.

The base dependency list remains NumPy-only.

## Acceptance criteria

1. Module schema command works from source and the installed wheel.
2. Materialized raw/stationary bundles audit successfully; intentional leak rejects.
3. Example export never overwrites and does not partially write when a conflict is known.
4. Module audit output matches the legacy script for the same inputs.
5. Wheel metadata exposes the console command and upload extra.
6. The isolated installed console command can print a schema, materialize an example, and
   audit that example.
7. Complete repository and packaging gates pass.

## Non-goals

HTTP upload, GUI, remote storage, formula attestation, arbitrary code execution, automatic
dependency installation, deployment, GitHub publication, and profitability claims remain
outside this slice.
