# UPLOAD-003 — Bind fold-local transformer identity at the comparison seam

Type: task
Status: resolved
Blocked by: none

Extend the high-level effectiveness comparison to accept transformer factories/specs,
pass them into all channels, and expose ordered spec digests in canonical report evidence.

## Acceptance

- One-to-one validation remains fail closed.
- Changing a Transformer Spec changes the report digest.

## Comments

- 2026-08-02: Claimed for v0.5.1 implementation.

## Answer

All comparison channels now receive the ordered transformer factories/specs and the
report binds their digests. Evidence: `docs/evidence/feature-upload-cli-20260802.md`.
