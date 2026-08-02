# 01 — Freeze v0.6.1 and execute remote CI

Type: task
Status: claimed
Blocked by: GitHub repository URL, installed/authenticated `gh`

- [x] Real one-attempt Holdout receipt exists and is bound to the frozen protocol.
- [x] Three fixed models are ranked across three chronological regimes.
- [x] Cross-platform workflow covers Linux, Windows, macOS and production artifacts.
- [x] CI installs `benchmark`/`upload` extras so Parquet tests have `pyarrow`.
- [x] Local complete test/type/lint/build/install gates pass.
- [ ] Commit current v0.6.1 scope and create local annotated tag.
- [ ] Configure `origin` from an explicit repository URL.
- [ ] Push branch/tag and observe every GitHub Actions matrix job.
- [ ] Record immutable commit, workflow URL, job conclusions, and production decision.

## Answer

Holdout and regime gates are complete. The remaining remote gate cannot be truthfully
completed until the repository target and GitHub authentication are available.

## Comments

Execution prompt: `.scratch/production-candidate-v061/spec.md`.
