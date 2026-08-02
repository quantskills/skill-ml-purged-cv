# Verification

## Slice 1 local acceptance — 2026-07-31

Observed on Windows with Python 3.11.9, NumPy 2.4.6, pandas 3.0.3, Hypothesis 6.164.0,
pytest 9.1.1, mypy 2.3.0, and Ruff 0.15.21.

| Command / canary | Observed result |
|---|---|
| `python -m pip install -e ".[dev,pandas]"` | editable wheel built and installed successfully |
| `python tasks/preflight.py` | success; no contract errors |
| `python tasks/test.py` | success; includes pytest, strict mypy, Ruff check, and full-repo format check |
| `python -m pytest -q` | 41 passed |
| `python -m mypy src tests tasks` | success; no issues in 22 source files |
| `python -m ruff check .` | all checks passed |
| `python -m ruff format --check .` | all Python files already formatted |
| core import with pandas blocked | passed in `test_core_import_does_not_require_pandas` |
| installed core and pandas-adapter imports outside the repository | version/API imports succeeded |
| `python -m pip check` | no broken requirements found |
| `python -m build --no-isolation` | wheel and source distribution built successfully |
| `python -m twine check dist\\*` | both distributions passed without warnings |
| isolated import directly from the built wheel | package import succeeded |

The repository also contains a GitHub Actions matrix for Linux (Python 3.11–3.13),
Windows 3.11, and macOS 3.11, followed by a distribution-build job. This workflow has
not yet run remotely because the local repository has no configured GitHub remote.

This is local implementation and build evidence only. It does not establish remote CI,
cross-platform, performance, deployment, CPCV, Causal Walk-Forward, or Untouched
Holdout evidence.

## Slice 2 local acceptance — 2026-08-01

| Command / canary | Observed result |
|---|---|
| `python tasks/preflight.py` | success; no contract errors |
| `python tasks/test.py` | success; complete repository quality gate |
| `python -m pytest -q` | 52 passed |
| `python -m mypy src tests tasks scripts/benchmark_pandaai.py` | success; no issues in 28 source files |
| `python -m ruff check .` | all checks passed |
| `python -m ruff format --check .` | 28 files already formatted |
| `python -m pip check` | no broken requirements found |
| `python -m build --no-isolation` | 0.2.0 wheel and source distribution built successfully |
| `python -m twine check` on both 0.2.0 artifacts | both distributions passed |
| isolated install/import from the 0.2.0 wheel | version, `CausalWalkForward`, and benchmark API succeeded |
| offline PandaAI five-asset benchmark | 12,259 observations across 2,745 sessions; both safe channels had zero retained interval overlaps |

The detailed real-cache inputs, digests, structural overlap findings, metrics, and claim
boundary are recorded in `docs/evidence/pandaai-benchmark-20260801.md`. This evidence
establishes local Causal Walk-Forward and offline benchmark behavior. It still does not
establish remote CI, cross-platform behavior, general performance, deployment, CPCV, or
Untouched Holdout evidence.

## Slice 3 CPCV local acceptance — 2026-08-01

| Command / canary | Observed result |
|---|---|
| focused `tests/test_cpcv.py` | 30 passed, including all small `3≤N≤8`, `2≤k<N` configurations |
| `python tasks/preflight.py` | success; no contract errors |
| `python tasks/test.py` | success; complete repository quality gate |
| `python -m pytest -q` | 82 passed |
| `python -m mypy src tests tasks scripts/benchmark_pandaai.py` | success; no issues in 31 source files |
| `python -m ruff check .` | all checks passed |
| `python -m ruff format --check .` | 31 files already formatted |
| `python -m pip check` | no broken requirements found |
| `python -m build --no-isolation` | 0.3.0 wheel and source distribution built successfully |
| `python -m twine check` on both 0.3.0 artifacts | both distributions passed |
| isolated install/import from the 0.3.0 wheel | version, CPCV splitter, path type, and channel API succeeded |
| offline PandaAI CPCV structural probe | 15 combinations, 5 paths, 12,259 observations, and zero retained interval overlaps |

The real-cache structural receipt and claim boundary are recorded in
`docs/evidence/cpcv-pandaai-probe-20260801.md`. This evidence establishes local general
CPCV combination, Path Decomposition, and path-scoped evaluator behavior. It does not
establish remote CI, cross-platform behavior, general performance, causal deployment,
Untouched Holdout evidence, profitability, or production readiness.

## Slice 4 governed full-cache effectiveness — 2026-08-01

| Command / canary | Observed result |
|---|---|
| focused governance and effectiveness tests | 3 passed, including offline Parquet CLI |
| `python -m pytest -q` before the full run | 85 passed |
| `python -m mypy` | success; no issues in 33 source files |
| `python -m ruff check .` | all checks passed |
| governed full-cache comparison | success in 732.6 seconds; 175,129 observations, 81 assets, 2,745 sessions |
| CPCV structure | 15 combinations, 5 complete paths, 175,129 observations per native path |
| independent overlap audit | zero in Purged K-Fold, CPCV, and Causal Walk-Forward |
| task-specific training gates | all folds/combinations exceed 10,000 observations, 252 sessions, and 20 assets |
| final `python tasks/preflight.py` | success; no contract errors |
| final `python tasks/test.py` | success; complete repository quality gate |
| final `python -m pytest -q` | 85 passed |
| final strict mypy / Ruff / format check | success; 33 source files typed, 35 Python files formatted |
| `python -m pip check` | no broken requirements found |
| `python -m build --no-isolation` | 0.4.0 wheel and source distribution built successfully |
| Twine check on 0.4.0 artifacts | wheel and source distribution passed |
| isolated direct import from 0.4.0 wheel | version, effectiveness API, and governed PandaAI adapter imported successfully |

Detailed governance counts, every CPCV path, metric distributions, channel comparison,
training sufficiency, digests, and claim boundaries are recorded in
`docs/evidence/full-cpcv-effectiveness-20260801.md`.

## Slice 5 arbitrary-feature governance — 2026-08-01

| Command / canary | Observed result |
|---|---|
| Feature-governance public-seam tests | 9 passed |
| `python tasks/preflight.py` | success; no contract errors |
| `python tasks/test.py` | success; complete repository quality gate |
| `python -m pytest -q` | 94 passed |
| strict mypy | success; no issues in 36 source files |
| Ruff check and format check | all checks passed; 38 Python files formatted |
| `python -m pip check` | no broken requirements found |
| `python -m build --no-isolation` | 0.5.0 wheel and source distribution built successfully |
| Twine check | both 0.5.0 artifacts passed |
| isolated import directly from 0.5.0 wheel | version, feature governance, TransformerSpec, and pandas upload APIs succeeded |

Research coverage, failure cases, implementation outcome, and claim boundaries are
recorded in `docs/evidence/feature-governance-20260801.md`.

## Slice 5.1 governed local feature upload — 2026-08-02

| Command / canary | Observed result |
|---|---|
| focused upload and high-level transformer tests | 15 passed |
| `python tasks/preflight.py` | success; no contract errors |
| `python tasks/test.py` | success; complete repository quality gate, including scripts in strict mypy |
| `python -m pytest -q` | 107 passed |
| `python -m mypy src tests tasks scripts` | success; no issues in 41 source files |
| `python -m ruff check .` | all checks passed |
| `python -m ruff format --check .` | 41 files already formatted |
| `python -m pip check` | no broken requirements found |
| raw/stationary/leak audit canaries | raw and stationary exited 0; declared target leak exited 2 |
| `python -m build --no-isolation` | 0.5.1 wheel and source distribution built successfully |
| Twine check | both 0.5.1 artifacts passed |
| isolated import directly from 0.5.1 wheel | version, `UploadLimitError`, upload adapter, and default 10,000-combination limit imported successfully |

The versioned input contract, redacted examples, resource gates, fold-local transformer
binding, and claim boundaries are recorded in
`docs/evidence/feature-upload-cli-20260802.md`. This evidence establishes a local
repository CLI and installable adapter. It does not prove user-supplied metadata is
truthful, establish strategy profitability, authorize remote publication, or constitute
deployment evidence.

## Slice 5.2 installable governed-upload CLI — 2026-08-02

| Command / canary | Observed result |
|---|---|
| CLI/distribution focused tests | 22 passed |
| `python tasks/preflight.py` | success; no contract errors |
| `python tasks/test.py` | success; complete repository gate |
| `python -m pytest -q` | 117 passed |
| strict mypy | success; no issues in 44 source files |
| Ruff check / format check | all checks passed; 44 Python files formatted |
| `python -m pip check` | no broken requirements found |
| `python -m build --no-isolation` | 0.5.2 wheel and source distribution built successfully |
| Twine check | both 0.5.2 artifacts passed |
| wheel content inspection | console entry-point metadata, 2 schemas, and all 9 example files present |
| pandas-blocked schema canary | schema discovery succeeded without importing pandas |
| pandas-blocked audit canary | exit 2 with redacted upload-extra installation guidance |
| isolated venv console/module canary | version 0.5.2; schema, raw export, raw audit, and module mapping schema succeeded |

Detailed distribution identity, canaries, and limitations are recorded in
`docs/evidence/installable-upload-cli-20260802.md`. The isolated venv reused locally
installed runtime dependencies and installed the 0.5.2 wheel with `--no-deps`; it proves
wheel command/resource behavior, not dependency resolution from an external index or
remote deployment.

## Slice 6 release hardening and governed Holdout — 2026-08-02

| Command / canary | Observed result |
|---|---|
| focused Holdout/ranking/upload tests | 34 passed |
| `python tasks/preflight.py` | success; no contract errors |
| `python tasks/test.py` | success; complete repository quality gate |
| `python -m pytest -q` | 132 passed |
| strict mypy | success; no issues in 48 source files |
| Ruff check / format check | all checks passed; 48 Python files formatted |
| `python -m pip check` | no broken requirements found |
| `python -m build --no-isolation` | 0.6.0 wheel and source distribution built successfully |
| Twine check | both 0.6.0 artifacts passed |
| isolated installed-wheel canary | version 0.6.0; Holdout/ranking APIs and packaged schema succeeded |

The detailed state-machine, upload-preflight, ranking-reversal, build, and claim
boundaries are recorded in `docs/evidence/release-hardening-20260802.md`. Local success
does not establish remote CI, an immutable Git commit, external metadata truth,
profitability, publication, deployment, rollback, or production observation.

## Five-year PandaData governed release gate — 2026-08-02

| Check | Observed result |
|---|---|
| source window | 2021-06-18 through 2026-06-18; 90 local Parquet files; no login or download |
| governed dataset | 102,605 input rows; 88,417 eligible observations; 81 assets; 1,174 sessions |
| development boundary | 66,061 observations and 898 sessions, strictly before Holdout information |
| model ranking | mean, ridge-100, ridge-1 in the same order across 3 chronological regimes; minimum Spearman 1.0 |
| structural comparison | Purged K-Fold, 5 CPCV paths, and causal Walk-Forward all retained zero overlapping intervals |
| minimum training samples | PKF 50,617/699 sessions; CPCV 36,554/500; Walk-Forward 19,054/276 |
| final Holdout | 20,412 observations; 252 sessions from 2025-05-09; MSE 0.0015686022812105493 |
| one-attempt governance | receipt persisted after the only evaluation; no Holdout rerun during report recovery |
| complete repository gate | `tasks/test.py` success; 132 pytest tests; strict mypy over 49 files; Ruff and pip check pass |
| 0.6.1 artifacts | wheel and sdist built; Twine passed; isolated installed-wheel version/schema/ranking canary passed |

The full metrics, digests, recovery boundary, and interpretation are recorded in
`docs/evidence/pandadata-five-year-release-gate-20260802.md`.
