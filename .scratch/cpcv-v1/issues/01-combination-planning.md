# CPCV-001A — General combination planning

Type: task
Status: resolved
Blocked by: PKF-007

- [x] Public deterministic `CombinatorialPurgedCV.plan()` seam.
- [x] General lexicographic `N,k` combinations with an explicit budget.
- [x] Multi-block Purge and per-contiguous-block Embargo.
- [x] Panel grouping, typed invalid folds, provenance, and CPCV Evidence Channel.
- [x] Focused boundary and determinism tests pass.

## Answer

Implemented with complete lexicographic combinations, explicit combination budgeting,
multi-block leakage exclusions, adjacent-block merging, typed invalid folds, immutable
assignments, and `cpcv-robustness` evidence. Focused and complete suites passed.
