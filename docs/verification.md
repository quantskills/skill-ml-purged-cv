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

## Agent-neutral Skill productization — 2026-08-03

| Command / canary | Observed result |
|---|---|
| Skill official validator | valid root `SKILL.md` and frontmatter |
| Agent CLI focused tests | 10 passed |
| `python tasks/preflight.py` | success; no contract errors |
| `python tasks/test.py` | success; complete repository quality gate |
| `python -m pytest -q` | 138 passed |
| strict mypy | success; no issues in 47 source files during the complete gate |
| Ruff check / format check | all checks passed after formatting 2 new files |
| `python -m build --no-isolation` | 0.6.1 wheel and source distribution built successfully |
| Twine check | both 0.6.1 artifacts passed |
| isolated wheel canary | request schema and one-command demo succeeded outside the source tree |
| installed metadata canary | both `purged-cv-skill` and `purged-cv-upload` entry points present |

The Skill layer adds only discovery, orchestration, packaged schemas, and a redacted result
envelope. It does not alter algorithm behavior or turn local structural/model evidence into a
profitability, deployment, or external metadata-truth claim.

## Time-series strategy selection benchmark — 2026-08-03

| Command / canary | Observed result |
|---|---|
| focused strategy/CLI tests | 12 passed |
| `python tasks/preflight.py` | success |
| `python tasks/test.py` | success; complete repository quality gate |
| `python -m pytest -q` | 150 passed |
| strict MyPy and Ruff | success |
| `python -m pip check` | no broken requirements |
| `python -m build --no-isolation` | 0.7.0 wheel and sdist built |
| Twine check | both 0.7.0 artifacts passed |
| isolated installed-wheel canary | version 0.7.0; strategy schema; 32 trials; 4 cost scenarios |
| installed metadata canary | all three `purged-cv-*` entry points present |

The deterministic demo intentionally produces high PBO (0.8286 at zero cost) and DSR
probability below 0.95 (0.8808), demonstrating that an attractive full-sample Sharpe is
not automatically accepted. A subsequent authorized PandaData API run covered 15 assets
and 1,210 common Sessions. It passed PBO and positive CPCV/Walk-Forward gates but failed
DSR `>= 0.95`; see `docs/evidence/pandadata-tsmom-selection-benchmark-20260803.md`.

## Pre-registered strategy acceptance decision — 2026-08-04

| Command / canary | Observed result |
|---|---|
| focused acceptance/strategy/CLI tests | 21 passed |
| `python tasks/preflight.py` | success |
| `python tasks/test.py` | success; complete repository quality gate |
| `python -m pytest -q` | 159 passed |
| strict MyPy and Ruff | success |
| `python -m build --no-isolation` | 0.7.1 wheel and sdist built |
| Twine check | both 0.7.1 artifacts passed |
| isolated installed-wheel canary | version 0.7.1; 32 trials; 4 cost scenarios; acceptance statuses present |
| PandaData v0.7.1 replay | validation PASS; research FAIL; production FAIL |

At the 3 bps primary scenario, DSR is 0.7034292572 and fails the pre-registered 0.95
threshold while PBO, CPCV path-tail, and causal Walk-Forward checks pass. The
constant-distribution approximation estimates 11,460 total Sessions (10,250 additional),
but is not a guarantee. The untouched strategy Holdout remains unrun. Full evidence is
recorded in `docs/evidence/strategy-acceptance-decision-v071-20260804.md`.

## Trainable temporal-model leakage comparison — 2026-08-04

| Command / canary | Observed result |
|---|---|
| focused temporal-model/CLI tests | 17 passed |
| complete Pytest suite | 168 passed |
| `python tasks/preflight.py` | success |
| `python tasks/test.py` | success; Ruff, format, strict MyPy, Pytest, pip check |
| full PandaData three-model CLI | 17,775 observations; 15 assets; 1,185 Sessions |
| unsafe overlap canaries | shuffled 71,100; chronological no-purge 1,875 |
| formal channel overlap | Purged, Purged+Embargo, CPCV, Walk-Forward all zero |
| current report digest | `c3e8f962b950e307958ff622694b8595afd9d581bc73b9316f71f948a3f9cd38` |
| `python -m build --no-isolation` | 0.8.0 wheel and sdist built |
| Twine check | both 0.8.0 artifacts passed |
| isolated installed-wheel canary | version 0.8.0; six channels; structural PASS; production NOT_AUTHORIZED |

The fixed NumPy Ridge, LightGBM, and CPU PyTorch LSTM models all used the same lag-20/T+5
dataset and fold-local estimator lifecycle. Complete intervals made the 20-Session
Embargo incrementally redundant after Purge; the policy remained active and disclosed.
Structural leakage control passed; production authorization remained NOT_AUTHORIZED.
Full evidence is in `docs/evidence/pandadata-trainable-temporal-models-v080-20260804.md`.

## Temporal Forward Evidence protocol — 2026-08-04

| Command / canary | Observed result |
|---|---|
| focused forward store/CLI tests | 8 passed |
| complete Pytest suite | 176 passed |
| `python tasks/preflight.py` | success |
| `python tasks/test.py` | success; complete repository quality gate |
| Ruff check / format check / strict MyPy | success |
| frozen forward protocol | LightGBM lag-20/T+5; start 2026-08-04; digest `478d3a749462d9cca022b19c6b676783679a941be8d8b5ec1b3fe9f727553b70` |
| initial evidence state | 0 predictions; 0 settlements; `WAITING_FOR_FUTURE_DATA`; production `NOT_AUTHORIZED` |
| `python -m build --no-isolation` | 0.9.0 wheel and sdist built |
| Twine check | both 0.9.0 artifacts passed |
| isolated installed-wheel canary | version 0.9.0; public protocol import; `purged-cv-forward init` returned the expected waiting receipt |

The implementation enforces durable prediction recording before label availability and
append-only T+5 settlement afterwards. Synthetic tests demonstrate `COLLECTING`,
`READY_FOR_REVIEW`, and `FAIL`, but the real PandaData protocol remains waiting because
independent future labels do not yet exist. See
`docs/evidence/pandadata-temporal-forward-protocol-v090-20260804.md`.

## Clean production Skill release — 2026-08-04

| Command / canary | Observed result |
|---|---|
| Skill Creator `quick_validate.py` | valid |
| README/Skill contract tests | passed |
| complete Pytest suite | 179 passed |
| `python tasks/preflight.py` / `python tasks/test.py` | success |
| Ruff check / format check / strict MyPy | success |
| `python -m build --no-isolation` / Twine | v0.9.0 wheel and sdist passed |
| installed `purged-cv-skill demo` outside source checkout | success; engine 0.9.0; audit |
| installed example materialization and run | success |
| installed strategy demo | validation PASS; research/production FAIL |
| installed forward init | waiting; not authorized; local-not-notarized scope |
| repository cleanup | all `.scratch` paths removed from production index and ignored |

The independent forward-test detected a legacy machine-global distribution registration and
missing PATH entry. Release acceptance therefore relies on the isolated installed v0.9.0
entrypoints, not an implicit source-tree import. Full evidence is in
`docs/evidence/clean-production-skill-release-v090-20260804.md`.
