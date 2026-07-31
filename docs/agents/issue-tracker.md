# Issue tracker: Local Markdown

Issues and specs for this repo live as markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Triage state is recorded as a `Status:` line near the top
- Comments append under a `## Comments` heading

## Publishing and fetching

When a skill says “publish to the issue tracker”, create the corresponding file under `.scratch/<feature-slug>/`.

When a skill says “fetch the relevant ticket”, read the referenced path or issue number.

## Wayfinding operations

- Map: `.scratch/<effort>/map.md`
- Child ticket: `.scratch/<effort>/issues/NN-<slug>.md`
- Ticket type: `Type: research|prototype|grilling|task`
- Ticket status: `Status: claimed|resolved`
- Dependencies: `Blocked by: NN, NN`
- Claim before work by setting `Status: claimed`
- Resolve by adding `## Answer`, setting `Status: resolved`, and recording a context pointer in `map.md`
