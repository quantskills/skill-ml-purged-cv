# DISTCLI-003 — Console metadata and isolated-wheel acceptance

Type: task
Status: resolved
Blocked by: 01, 02

Add the console entry point, upload extra, package data, documentation, version bump, and
observe module/console commands from an isolated installed wheel.

## Acceptance

- Full quality/build gates pass.
- Installed `purged-cv-upload` completes schema, example, and audit canaries.

## Comments

- 2026-08-02: Claimed for v0.5.2.

## Answer

Version 0.5.2 declares the console entry point, upload extra, and package data. Full
quality/build gates and installed console/module canaries passed. Evidence:
`docs/evidence/installable-upload-cli-20260802.md`.
