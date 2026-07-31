# Own splitters and leakage-safe evaluation

V1 will expose low-level Splitters and a canonical Leakage-Safe Evaluator that creates transformations and estimators inside each fold. A splitter-only library was rejected because it could generate correct indices while still allowing globally fitted preprocessing or hyperparameter selection to leak test information; owning the evaluator makes Fold-Local behavior enforceable while preserving expert access to the lower-level split contract.
