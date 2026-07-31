# Build deterministic general CPCV paths

CPCV will support general valid `N,k` configurations and derive Path Decomposition deterministically from canonical group order and the split specification. Construction succeeds only when every combination/test-group occurrence is assigned once, each path contains every group exactly once, test groups from one combination occupy distinct paths, and the expected `C(N-1,k-1)` complete paths are proven; fixed `N=6,k=2`, random assignment, incomplete paths, and scheduling-dependent identities were rejected.
