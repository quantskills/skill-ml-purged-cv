# DISTCLI-002 — Installed schemas and example bundles

Type: task
Status: resolved
Blocked by: 01

Package byte-identical schemas/examples and expose read-only schema plus safe example
materialization commands.

## Acceptance

- All installed resources match repository canonical copies.
- Export refuses overwrite before writing.

## Comments

- 2026-08-02: Claimed for v0.5.2.

## Answer

The wheel packages both schemas and all examples; schema discovery, three example
behaviors, optional dependency isolation, and no-overwrite export are verified. Evidence:
`docs/evidence/installable-upload-cli-20260802.md`.
