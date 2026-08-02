# DISTCLI-001 — Package-native CLI and compatibility wrapper

Type: task
Status: resolved
Blocked by: none

Move CLI behavior into package source, add the module entry point, and preserve the
repository script as a thin compatibility wrapper.

## Acceptance

- Module audit/evaluate behavior matches the existing script.
- No validation logic remains duplicated in the wrapper.

## Comments

- 2026-08-02: Claimed for v0.5.2.

## Answer

Package `cli.py` and `__main__.py` now own command execution; the repository script is a
thin compatibility wrapper. Evidence: `docs/evidence/installable-upload-cli-20260802.md`.
