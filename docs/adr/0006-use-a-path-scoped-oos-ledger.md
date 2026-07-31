# Use a path-scoped OOS ledger

Raw path-scoped OOS predictions are the authoritative result, and fold, path, and channel metrics are versioned projections from that ledger. Repeated CPCV predictions for the same sample will retain split, combination, and path identity rather than being averaged before path construction; Purged K-Fold, CPCV, Causal Walk-Forward, and Untouched Holdout remain separate Evidence Channels, while persistence is delegated to optional writers so the core can return an in-memory result.
