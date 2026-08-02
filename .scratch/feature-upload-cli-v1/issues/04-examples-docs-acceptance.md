# UPLOAD-004 — Examples, documentation, and release acceptance

Type: task
Status: resolved
Blocked by: 01, 02, 03

Provide raw, stationary, and intentional-leak bundles; document the user workflow and
claim boundary; bump to 0.5.1; run and record complete verification.

## Acceptance

- Raw and stationary audit successfully; intentional leak is rejected.
- Preflight, tests, mypy, Ruff, build, Twine, and isolated wheel import pass.

## Comments

- 2026-08-02: Claimed for v0.5.1 implementation.

## Answer

Added all three examples and user/contracts documentation, released local version 0.5.1,
and observed the complete tests, static checks, build, Twine, and wheel canary passing.
Evidence: `docs/evidence/feature-upload-cli-20260802.md`.
