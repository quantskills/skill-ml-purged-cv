# Fail closed on invalid folds

A Split Plan may diagnose invalid candidates, but a formal evaluation will fail as a whole if any requested fold violates minimum training/test size, coverage, or CPCV path-completeness constraints, or if fold-scoped fitting fails. Silently skipping folds, emitting partial scores, or substituting `NaN` was rejected because difficult market intervals would disappear from the evidence and invalidate configured fold counts, combination coverage, and path-level comparisons.
