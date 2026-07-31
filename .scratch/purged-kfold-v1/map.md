# Purged K-Fold V1 map

Status: resolved

## Notes

- The approved specification is `spec.md`.
- The approved tracer-bullet breakdown is published under `issues/`; work follows declared blocking edges rather than strict ticket-number order.
- Ticket 01 is the initial frontier. Tickets 02 and 03 may proceed in parallel after 01; tickets 04 and 05 may proceed in parallel after their respective blockers.
- Formal evidence must remain fail-closed; diagnostic output is not a score.
- Ticket 01 is resolved; its implementation and observed evidence are recorded in `issues/01-minimal-leakage-safe-split-plan.md`.
- Ticket 02 is resolved; its factory lifecycle, OOS Ledger, metric, and abort evidence are recorded in `issues/02-fold-local-oos-evaluation.md`.
- Ticket 03 is resolved; its coverage, constraints, diagnostic-plan, and whole-run failure evidence are recorded in `issues/03-strict-purged-kfold.md`.
- Ticket 04 is resolved; its interval, Session Axis Embargo, exclusion trace, and panel evidence are recorded in `issues/04-financial-time-and-panel-boundaries.md`.
- Ticket 05 is resolved; its PIT eligibility, latest-revision rejection, redaction, and OOS provenance evidence are recorded in `issues/05-pit-safe-formal-scoring.md`.
- Ticket 06 is resolved; its explicit mapping, equivalence, ambiguity rejection, and optional-dependency evidence are recorded in `issues/06-pandas-adapter.md`.
- Ticket 07 is resolved; property, regression, documentation, packaging, and dual-axis review evidence are recorded in `issues/07-acceptance-evidence.md` and `docs/verification.md`.

## Decisions so far

- ADR-0001 through ADR-0009 define the accepted architecture and domain boundaries.
- Slice 1 is a vertical Purged K-Fold evaluation path; CPCV, walk-forward, holdout governance, nested HPO, and persistent writers are later slices.
- The published spec follows the project PRD template and has passed final local acceptance.
- The implementation tickets are vertical, independently verifiable slices; the former horizontal layer tickets have been superseded.
- All Slice 1 tickets are resolved. Later work remains planned and requires a separately claimed frontier ticket.

## Fog

- Performance targets are intentionally unset until correctness baselines can be measured.
- Exact public names may change during test-first implementation, but domain meanings and accepted ADR boundaries may not drift silently.
