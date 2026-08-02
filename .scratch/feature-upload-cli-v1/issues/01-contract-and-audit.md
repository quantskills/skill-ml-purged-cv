# UPLOAD-001 — Versioned upload contract and audit command

Type: task
Status: resolved
Blocked by: none

Implement exact JSON parsing, CSV/Parquet loading, resource gates, and the redacted
`audit` command. Add JSON Schemas and integration tests for success and fail-closed paths.

## Acceptance

- Valid CSV returns code 0 and a governance receipt without fitting.
- Invalid declarations and resource excess return code 2 with safe JSON.

## Comments

- 2026-08-02: Claimed for v0.5.1 implementation.

## Answer

Implemented the closed schemas, bounded adapter, audit command, redacted rejection path,
and audit/example integration tests. Evidence: `docs/evidence/feature-upload-cli-20260802.md`.
