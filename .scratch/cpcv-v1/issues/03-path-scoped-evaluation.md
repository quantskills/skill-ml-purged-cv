# CPCV-001C — Path-scoped evaluation evidence

Type: task
Status: resolved
Blocked by: CPCV-001A, CPCV-001B

- [x] OOS facts retain combination, group, and path identity.
- [x] Repeated sample predictions are preserved without premature averaging.
- [x] Unique-sample coverage and complete per-path metrics are correct.
- [x] Incomplete/duplicate path evidence fails closed.
- [x] Full compatibility and package acceptance pass.

## Answer

The canonical evaluator now retains raw repeated CPCV observations, validates complete
sample coverage in every path, and derives per-combination and per-path metrics. The
0.3.0 wheel passed metadata and isolated public-API import checks.
