# Domain Docs

This repository uses a single-context domain-document layout.

## Before exploring

Read:

- Root `CONTEXT.md`
- Relevant ADRs under `docs/adr/`

If an expected document does not exist, proceed silently. Domain documents and ADRs are created lazily when terminology or decisions are resolved.

## Vocabulary

Use the domain terminology defined in `CONTEXT.md`. Avoid introducing synonyms that conflict with its glossary.

If a required concept is missing, reconsider whether new terminology is necessary or record the gap for domain modeling.

## ADR conflicts

If proposed work contradicts an existing ADR, surface the conflict explicitly rather than silently overriding it.
