# Clean production Skill release v0.9.0 — 2026-08-04

## Outcome

The repository is prepared as a maintained, installable QuantSkills Skill without tracked
`.scratch` process artifacts. The root Skill validates, the Chinese README has an automated
public-workflow contract, and installed console commands run outside the source checkout.

## Release boundary

- `.scratch/` is ignored in full and its 53 previously tracked process files are staged for
  removal from the remote repository while remaining available locally.
- Source, tests, CI, Agent metadata, progressive references, ADRs, redacted evidence, and
  non-secret examples remain in the maintained repository.
- `dist/`, runtime ledgers, caches, raw data, credentials, tokens, and private keys are not
  release inputs.

## Skill and README validation

| Check | Result |
|---|---|
| `quick_validate.py` with UTF-8 mode | Skill is valid |
| independent Agent forward-test | found the global legacy installation mismatch; module fallback succeeded |
| README/Skill release-contract tests | passed |
| complete Pytest suite | 179 passed |
| Ruff / format / strict MyPy | passed |
| `tasks/preflight.py` / `tasks/test.py` | passed |
| v0.9.0 wheel and sdist | built |
| Twine | both artifacts passed |

The independent test correctly rejected the machine-global `0.1.0` registration and missing
PATH entry rather than accepting a source-tree import as an installed Skill. The release was
therefore canaried again from the existing isolated v0.9.0 wheel environment, with its working
directory outside the source checkout.

## Installed console canary

| Surface | Observed result |
|---|---|
| installed distribution | 0.9.0 |
| `purged-cv-skill demo` | success; engine 0.9.0; authoritative stage `audit` |
| `purged-cv-skill example` then `run` | success; generated bundle audited |
| `purged-cv-strategy demo` | success; validation PASS; research FAIL; production FAIL |
| `purged-cv-forward init` | `WAITING_FOR_FUTURE_DATA`; `NOT_AUTHORIZED`; local-not-notarized scope disclosed |

These canaries demonstrate that the optimized Skill executes through installed entrypoints.
They do not convert historical development evidence into future model performance or production
authorization.
