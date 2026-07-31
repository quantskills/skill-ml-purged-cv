# Govern final holdout access

V1 will provide an Evaluation Protocol state machine that freezes data boundaries, feature/model/search configuration, split specifications, and metrics before authorizing one final holdout evaluation, then records an append-only Holdout Receipt through a Protocol Store. A plain holdout splitter was rejected because repeated inspection cannot support an untouched claim; new protocol identities do not rehabilitate a Reused Holdout, and the library will state explicitly that it governs only accesses made through its own interfaces.
