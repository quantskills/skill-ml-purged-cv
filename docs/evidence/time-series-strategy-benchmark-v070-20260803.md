# Time-Series Strategy Selection Benchmark v0.7.0 — 2026-08-03

## Decision

The new strategy-selection benchmark passes the complete local code, package, and
installed-wheel gates. The repository now has a real time-series strategy benchmark
surface rather than describing the fixed Ridge/MSE structural comparison as a trading
strategy test.

The accepted product boundary has two layers:

1. the existing four-channel comparison remains a structural leakage canary;
2. `StrategyReturnMatrix` plus the built-in TSMOM adapter provides candidate-selection
   overfitting evidence through CSCV/PBO, DSR, CPCV selected paths, and causal
   Walk-Forward.

This acceptance does not establish that TSMOM or any user strategy is profitable.

## Automated acceptance

| Check | Observed result |
|---|---|
| focused strategy benchmark and CLI tests | 12 passed |
| complete Pytest suite | 150 passed |
| `python tasks/preflight.py` | success |
| `python tasks/test.py` | success; strict MyPy, Ruff check, Ruff format check, and full tests |
| `python -m pip check` | no broken requirements |
| `python -m build --no-isolation` | 0.7.0 wheel and sdist built |
| Twine check | wheel and sdist passed |
| isolated installed-wheel version | 0.7.0 |
| installed strategy request schema | `Time-series strategy benchmark request v1` |
| installed console entry points | `purged-cv-skill`, `purged-cv-strategy`, `purged-cv-upload` |
| installed TSMOM demo | success; 32 trials; 4 separate cost scenarios |

The Windows venv install wrapper twice reached its 120-second outer timeout after pip
had explicitly reported a successful installation. Schema/demo/version checks were then
run separately from the same isolated venv and completed successfully in about one
second. The timeout is recorded as environment overhead rather than silently presented
as a one-command pass.

## Adversarial evidence

- Changing signal prices at Session `t` or later does not change the strategy return at
  `t`; later eligible rebalances do change, proving the signal is not accidentally
  detached from input.
- Increasing `cost_bps` leaves gross returns and turnover unchanged and cannot increase
  the same candidate's cumulative net return.
- A deliberately regime-specific candidate matrix produces PBO at or above 0.75.
- Mutating every return from the first Walk-Forward test boundary onward cannot change
  the candidate selected for that first window.
- Each `6 choose 2` CPCV audit reconstructs five complete paths, each covering every
  Session exactly once.
- Wrong NPZ members and pickle-bearing/object-style identifiers are not accepted by the
  installed input seam.

## Deterministic demo

The installed demo uses three generated assets, 320 Sessions, a fixed RNG seed, three
different drift regimes, and the registered 32-candidate TSMOM grid. It is deliberately
an installation and interpretation canary rather than a favorable strategy fixture.

| Cost (bps) | PBO | DSR probability | Best full-sample Sharpe | CPCV median / P10 / worst | Walk-Forward Sharpe | Mean selection regret |
|---:|---:|---:|---:|---|---:|---:|
| 0 | 0.8286 | 0.8808 | 2.1351 | 0.9203 / 0.5826 / 0.5030 | 1.7504 | 1.8496 |
| 1 | 0.8429 | 0.8794 | 2.1301 | 0.9114 / 0.3149 / 0.1979 | 1.7422 | 1.8479 |
| 3 | 0.8429 | 0.8766 | 2.1199 | 0.8935 / 0.2939 / 0.1797 | 1.7259 | 1.8446 |
| 5 | 0.8286 | 0.8737 | 2.1097 | 0.8757 / 0.2729 / 0.1614 | 1.7096 | 1.8411 |

The demo is an example of a negative selection conclusion: PBO is high and DSR remains
below a pre-registered 0.95 acceptance example even though the best full-sample Sharpe
looks attractive. This is precisely why the benchmark must not report only the winning
Sharpe.

## Real-data follow-up

After the initial local cache search, the user explicitly authorized a new PandaData API
download. The installed benchmark was then run on 15 underlyings and 1,210 common Sessions
from 2021-08-03 through 2026-08-03. The real-data run passes PBO, CPCV path-sign, causal
Walk-Forward-sign, and monotonic cost-disclosure gates, but fails the example DSR `>= 0.95`
gate. See `docs/evidence/pandadata-tsmom-selection-benchmark-20260803.md` for exact metrics,
data lineage, credential boundary, and limitations.
