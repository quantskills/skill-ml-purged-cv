# Implementation prompt — General deterministic CPCV

You are implementing the next verified delivery slice of
`purged-kfold-validation`. Work inside this repository and preserve `AGENTS.md`,
`PROGRAM.md`, `CONTEXT.md`, the accepted ADRs, and all already observed Purged K-Fold,
Causal Walk-Forward, evaluator, adapter, and benchmark behavior.

## Objective

Implement `CPCV-001`: general deterministic Combinatorial Purged Cross-Validation as
an additional robustness Evidence Channel. CPCV must reuse the canonical Validation
Dataset, exact Information Interval Purge, session-based Embargo, Split Plan,
fold-local evaluator, and raw OOS Ledger. It must not replace Purged K-Fold or be
presented as causal deployment evidence.

## Public seams

- `CombinatorialPurgedCV.plan(ValidationDataset) -> SplitPlan`
- `SplitPlan.path_decomposition -> CPCVPathDecomposition | None`
- `LeakageSafeEvaluator(...).evaluate(dataset) -> EvaluationResult`
- path-scoped fields on CPCV `OOSObservation` values and path-scoped Derived Metrics

Tests must use these public seams and observable evidence, never private helpers.

## Combination planning requirements

- Divide active Trading Sessions into `n_groups` deterministic, chronological,
  non-empty contiguous groups using the authoritative Session Axis.
- Enumerate every lexicographic choice of `n_test_groups` from `n_groups` exactly once.
- Require `2 <= n_test_groups < n_groups`; reject booleans and invalid integers.
- Reject `n_groups` greater than active-session count.
- Compute the combination count before enumeration and fail when it exceeds an
  explicit `max_combinations` budget.
- For one combination, all selected group sessions are test sessions; every other
  active session is a candidate training session. Panel rows sharing a session remain
  indivisible.
- Purge every candidate whose inclusive Information Interval overlaps any selected
  test sample interval.
- Apply Embargo after every contiguous selected test region in Session Axis positions.
  Adjacent selected groups form one contiguous Test Block; disjoint groups remain
  separate Test Blocks.
- Preserve minimum train sessions, train samples, and test sessions. Invalid
  combinations remain typed Invalid Folds and formal evaluation fails closed.
- Every assignment must carry `EvidenceChannel.CPCV_ROBUSTNESS`, its zero-based
  combination index, selected group identities, deterministic split identity, and
  exclusion evidence.

## General deterministic Path Decomposition

For `N=n_groups`, `k=n_test_groups`, and `phi=C(N-1,k-1)`:

- Produce exactly `phi` CPCV Paths.
- Treat every `(combination, selected group)` occurrence as one indivisible item.
- Assign every occurrence to exactly one path.
- Every path must contain every chronological group exactly once.
- Test groups belonging to the same combination must occupy distinct paths.
- Preserve chronological group order inside each path.
- Construction must be deterministic and scheduling-independent for every accepted
  general `N,k`, not special-cased to `N=6,k=2`.
- The decomposition and its digest must be immutable and independently validate all
  coverage, uniqueness, and identity invariants.

Use a deterministic proper edge-coloring of the bipartite
`combination <-> selected group` incidence graph, or another construction that proves
the same invariants. Never use random path assignment.

## Evaluation and ledger requirements

- Create fresh fold-local transformers and estimator for every CPCV Combination.
- Retain repeated OOS predictions; never average a sample across combinations before
  path construction.
- Every CPCV observation must carry combination index, chronological group index, and
  path index in addition to run, split, sample, model, dataset, PIT, and channel identity.
- Sort ledger evidence deterministically without discarding repeated observations.
- Overall metrics may use all raw CPCV observations. `per_fold` reports combinations;
  `per_path` reports complete CPCV Paths.
- Observation coverage is based on unique OOS sample identities, so a complete CPCV
  run reports 1.0 coverage rather than `phi`.
- A path metric may be emitted only when every path covers every dataset sample once;
  incomplete or duplicate path observations fail closed.

## Negative controls

Tests must include:

- the worked `N=6,k=2` case: 15 combinations and 5 complete paths;
- a different general configuration such as `N=5,k=3`;
- deterministic repeated planning and changed-configuration digests;
- an interval crossing one of multiple test blocks that Purge removes;
- Embargo after disjoint blocks and adjacency merging;
- panel-session indivisibility;
- invalid minimum-training combinations and combination-budget rejection;
- evaluator evidence proving repeated predictions remain separated by combination and
  path while every path covers each sample exactly once.

## Delivery constraints

- Python 3.11+, NumPy core, no mandatory new dependency.
- No causal, profitability, holdout, nested-HPO, deployment, or performance claim.
- No live PandaAI calls or credentials. A local PandaAI cache may be used only for a
  bounded structural acceptance probe after synthetic correctness is established.
- Follow one public-behavior red-green cycle at a time.
- Update exports, README, architecture, interface contract, work items, requirements
  traceability, and verification only for implemented and observed behavior.

## Acceptance commands

Run and record:

1. focused CPCV planning tests;
2. focused CPCV evaluator/path tests;
3. all pre-existing tests;
4. `python tasks/preflight.py`;
5. `python tasks/test.py`;
6. package build, metadata validation, and isolated wheel import.

Do not resolve CPCV-001 until every invariant and acceptance command is observed.
