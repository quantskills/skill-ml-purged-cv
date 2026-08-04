# Strategy Acceptance Decision v0.7.1 — 2026-08-04

## Outcome

v0.7.1 adds a pre-registered decision layer without changing the TSMOM candidate
family or retuning it against observed PandaData results. The same five-year panel now
returns explicit validation-tool, Research Gate, and Production Gate statuses.

```text
validation tool: PASS
research gate: FAIL
production gate: FAIL
reason codes: DSR_BELOW_THRESHOLD,
              UNTOUCHED_HOLDOUT_NOT_RUN,
              RESEARCH_GATE_FAILED
```

The result means the validation Skill works and rejects insufficient strategy evidence.
It does not mean CPCV or the time-series validation engine failed.

## Pre-registered default policy

| Check | Scenario | Threshold |
|---|---:|---:|
| PBO | 3 bps | `<= 0.20` |
| DSR probability | 3 bps | `>= 0.95` |
| CPCV median / P10 / worst Sharpe | 3 bps | each `>= 0` |
| Walk-Forward Sharpe | 3 bps | `>= 0` |
| CPCV worst Sharpe | 5 bps | `>= 0` |
| Walk-Forward Sharpe | 5 bps | `>= 0` |
| final evidence | frozen strategy | one untouched Holdout |

The policy is immutable and digest-bound. Missing registered cost scenarios fail closed.

## PandaData replay

The existing governed NPZ was replayed through the formal v0.7.1 JSON CLI. It contains
15 assets and 1,210 common Trading Sessions from 2021-08-03 through 2026-08-03. No new
download, credential use, candidate search, or Holdout consumption occurred.

| Metric | 3 bps primary | 5 bps stress |
|---|---:|---:|
| PBO | 0.0571428571 | 0.0428571429 |
| DSR probability | 0.7034292572 | 0.6640122312 |
| CPCV worst Sharpe | 0.7455369278 | 0.5258280105 |
| Walk-Forward Sharpe | 0.6904516588 | 0.6396166568 |

All registered path-tail and causal-return checks pass. The primary DSR check fails.
The decision digest is
`95acb4ebbe12a35fd3745c6e67efe2175792957f32892daa681b4bd9372dd332` and the
v0.7.1 report digest is
`b89742225500e0cc392094ce9a99bd20909aec145401a29073dc3b84df2ebdbd`.

## DSR evidence gap

At the 3 bps primary scenario, the current 1,210 Sessions produce DSR probability
0.7034292572. Under the explicit `constant-distribution approximation`, the same
selected/benchmark Sharpe and sample moments would require approximately 11,460 total
Sessions, or 10,250 additional Sessions, to reach probability 0.95.

This is a diagnostic extrapolation, not a forecast or guarantee. It assumes the return
distribution and edge remain constant over a very long extension and therefore cannot
be used as production authorization. A genuinely untouched strategy Holdout remains
missing; the five-year development window cannot be relabeled as one after selection.

## Verification

| Check | Observed result |
|---|---|
| focused acceptance/strategy/CLI tests | 21 passed |
| complete Pytest suite | 159 passed |
| `python tasks/preflight.py` | success |
| `python tasks/test.py` | success; Ruff, format, strict MyPy, Pytest, pip check |
| build | v0.7.1 wheel and sdist built |
| Twine | both v0.7.1 artifacts passed |
| isolated wheel canary | v0.7.1; 32 trials; 4 costs; three gate statuses present |
| real formal CLI replay | PASS / FAIL / FAIL with exact reason codes above |

Local completion does not publish the package, merge a branch, run remote cross-platform
CI, execute a final Holdout, deploy a strategy, or observe production behavior.
