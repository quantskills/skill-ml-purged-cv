# Return evidence-bearing fold assignments

Splitters will return immutable Fold Assignments that preserve stable sample identities, array positions, contiguous test blocks, CPCV identities, exclusion summaries, and dataset/configuration digests. Bare `(train_idx, test_idx)` tuples remain available only through compatibility adapters because they discard the provenance needed to audit blockwise embargo, path-scoped OOS predictions, and the exact inputs behind a validation result; per-sample Exclusion Traces remain optional to control memory use.
