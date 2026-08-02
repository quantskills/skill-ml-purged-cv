# Installable governed-upload CLI evidence — 2026-08-02

## Delivery identity

- Version: 0.5.2
- Prompt: `.scratch/installable-upload-cli-v1/prompt.md`
- Specification: `.scratch/installable-upload-cli-v1/spec.md`
- Tickets: DISTCLI-001 through DISTCLI-003
- Console command: `purged-cv-upload`
- Module command: `python -m purged_kfold_validation`
- Compatibility wrapper: `scripts/audit_feature_upload.py`

## Distribution outcome

The wheel owns the executable implementation, two version-1 JSON Schemas, and the raw,
stationary, and intentional-leak example bundles. Standard project metadata declares the
console command and an `upload` extra containing pandas and pyarrow while leaving the
base dependency set NumPy-only.

Schema and example commands complete before the optional pandas adapter is imported.
Audit/evaluate lazily load it and return deterministic installation guidance instead of
a traceback when the upload extra is unavailable. Example export uses only three fixed
filenames and rejects any known conflict before creating another target.

## Observed acceptance

| Check | Result |
|---|---|
| focused public seams | 22 passed |
| complete project gate | success |
| complete pytest | 117 passed |
| strict typing | 44 source files; no issues |
| lint / formatting | all checks passed; 44 files formatted |
| dependency consistency | no broken requirements |
| package build | 0.5.2 sdist and universal wheel built |
| Twine | both artifacts passed |
| wheel contents | entry point, 2 schemas, 3 CSVs, and 6 example JSON files present |
| optional-boundary canaries | schema works with pandas blocked; audit returns safe exit 2 guidance |

## Isolated installed-wheel canary

A temporary Windows venv was created with system site packages, then the local 0.5.2
wheel was installed with `--no-deps`. The installed console command and module command
observed the following:

1. Package version printed `0.5.2`.
2. `purged-cv-upload schema --kind manifest` returned the packaged manifest schema.
3. `purged-cv-upload example --name raw` wrote exactly three files.
4. The installed console audited that exported bundle successfully: 9 observations,
   3 assets, 3 sessions, dataset digest
   `3967842bcebe35cccdc2c44c69fca97811b25fe6eefb9880b1080904b368b327`.
5. `python -m purged_kfold_validation schema --kind mapping` returned the packaged
   mapping schema.
6. The verified temporary run directory was removed after the canary.

## Claim boundary

This proves local wheel contents, command routing, packaged resources, safe export, and
audit compatibility. It does not prove external package-index dependency resolution,
cross-platform remote CI, user metadata truth, profitability, GitHub publication,
deployment, or production operation.
