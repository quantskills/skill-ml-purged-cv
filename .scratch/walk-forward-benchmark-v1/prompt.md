# Implementation prompt — Causal Walk-Forward + PandaAI validation benchmark

You are implementing the next verified delivery slice of
`purged-kfold-validation`. Work inside this repository and preserve the canonical
contracts in `AGENTS.md`, `PROGRAM.md`, `CONTEXT.md`, the accepted ADRs, and the
existing Slice 1 interface.

## Objective

Deliver two connected capabilities:

1. `WF-001`: a deterministic, evidence-bearing `CausalWalkForward` splitter with
   Purge and a session-based Pre-Test Gap. Every retained training Information
   Interval must precede the test block; future observations may never train a
   causal fold.
2. `BENCH-001`: an offline PandaAI daily-futures benchmark that compares a clearly
   unsafe shuffled K-Fold baseline, a chronological no-purge baseline, Purged
   K-Fold, and Causal Walk-Forward on the same normalized dataset and estimator.

The benchmark is diagnostic evidence, not a trading strategy, profitability
claim, deployment simulation, or final holdout result.

## Public seams

- `CausalWalkForward.plan(ValidationDataset) -> SplitPlan`
- `LeakageSafeEvaluator(...).evaluate(dataset) -> EvaluationResult` with the
  assignment's Evidence Channel preserved into the run, ledger, and result
- `validation_dataset_from_pandaai_daily(frame, mapping, config) -> ValidationDataset`
- `run_validation_benchmark(dataset, estimator_factory, model_spec, ...) -> BenchmarkReport`
- `python scripts/benchmark_pandaai.py --data-dir ...` as the offline operator entrypoint

Tests must exercise these public seams and observable evidence, not private helpers.

## Causal Walk-Forward requirements

- Use the authoritative Session Axis and keep all rows from one Trading Session on
  the same side.
- Produce deterministic chronological test blocks.
- Restrict candidate training samples to sessions strictly before the current test
  block.
- Purge every candidate whose inclusive Information Interval overlaps a test
  Information Interval.
- Apply `pre_test_gap_sessions` immediately before the test block in Session Axis
  positions. Do not call it Embargo.
- Support expanding history by default and an optional `max_train_sessions` sliding
  history limit.
- Expose minimum train sessions, train samples, and test sessions; invalid requested
  folds remain diagnostic `InvalidFold` values and formal evaluation fails closed.
- Emit `EvidenceChannel.CAUSAL_WALK_FORWARD` on assignments, observations, ledgers,
  evaluation results, and deterministic run identity.
- Extend exclusion evidence to count and trace `pre-test-gap` removals without
  breaking existing Purged K-Fold callers.
- Preserve immutable arrays, deterministic digests, typed errors, and complete-run
  failure semantics.

## PandaAI adapter requirements

- The core library must not authenticate to PandaAI, read credentials, or access the
  network.
- Accept an already loaded pandas DataFrame through an explicit mapping. A script may
  load local `*_daily.parquet` files from a user-supplied directory.
- Required source columns are explicit: session/date, asset/symbol, close, and any
  selected numeric feature columns. Do not infer hidden DataFrame attributes.
- Build a forward-return label with an explicit session horizon and derive each
  sample's Information Interval from the actual per-asset feature lookback start to
  the actual label end.
- Set Decision Time and Feature Availability explicitly. Require a non-empty snapshot
  identity and source digest; state that the library verifies declarations, not the
  vendor's truthfulness.
- Drop warm-up/tail rows only before constructing the canonical dataset, report the
  resulting row/session/asset counts, and fail on duplicates, non-finite values, or
  insufficient history.
- Keep pandas and parquet engines optional; importing the NumPy core must not import
  pandas.

## Benchmark requirements

- Evaluate all channels on the same immutable `ValidationDataset`, estimator
  factory, and metric definition.
- Shuffle session groups, never individual panel rows, in the unsafe K-Fold baseline.
- Keep the chronological baseline past-only but deliberately omit interval Purge so
  its remaining overlaps are measurable.
- Count actual train/test Information Interval overlaps independently for every
  channel. Do not assert that an unsafe score must always be better.
- Report per-channel fold count, observation count/coverage, metric values, overlap
  count, Evidence Channel, dataset digest, and model digest.
- Safe channels must have zero retained overlap. Causal Walk-Forward must additionally
  prove `max(train information end) < min(test session)` for every valid fold.
- Output deterministic JSON from the script. Never print source rows, credentials, or
  feature values.

## Negative controls

Tests must include:

- a label interval crossing a causal boundary that chronological splitting retains
  but Purge removes;
- a Pre-Test Gap crossing a weekend/closed-market period;
- a panel session proving same-session rows remain indivisible;
- a future-available feature declaration that formal evaluation rejects;
- a fixed-session Embargo/Pre-Test Gap counterexample showing why interval overlap and
  feature availability remain separate checks.

## Delivery constraints

- Python 3.11+, NumPy core; pandas remains optional.
- No live PandaAI calls, credentials, deployment, GitHub publication, CPCV, nested
  HPO, holdout access, or performance claims in this slice.
- Follow red-green TDD one public behavior at a time.
- Preserve existing Slice 1 behavior and tests.
- Update exports, README, architecture, interface contract, work items, and
  requirements traceability only for behavior actually implemented and observed.

## Acceptance commands

Run and record:

1. focused Walk-Forward tests;
2. focused PandaAI adapter/benchmark tests;
3. `python tasks/preflight.py`;
4. `python tasks/test.py`;
5. package build and metadata validation.

Do not mark tickets resolved unless their acceptance evidence is observed. If local
PandaAI parquet data is unavailable, the adapter and benchmark fixture may be fully
verified while the real-data execution remains explicitly open; do not fabricate a
real-data receipt.
