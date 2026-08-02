# CPCV-001B — Deterministic general Path Decomposition

Type: task
Status: resolved
Blocked by: CPCV-001A

- [x] Exactly `C(N-1,k-1)` complete paths.
- [x] Every occurrence assigned once and every path contains every group once.
- [x] Same-combination test groups occupy distinct paths.
- [x] General deterministic construction and immutable digest.
- [x] Worked and non-special-case path tests pass.

## Answer

Implemented using deterministic proper edge-coloring of the combination/group
incidence graph. The worked `6,2`, non-special-case `5,3`, and all small configurations
from 3 through 8 groups passed complete path invariants.
