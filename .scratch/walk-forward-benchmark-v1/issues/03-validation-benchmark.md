# BENCH-001B — Four-channel validation benchmark

Type: task
Status: resolved
Blocked by: WF-001, BENCH-001A

- [x] Same dataset/model/metric across shuffled, chronological, Purged, and causal channels.
- [x] Session-group baselines and independent interval-overlap audit.
- [x] Structured immutable report with coverage, metric, channel, and provenance.
- [x] Deterministic redacted JSON operator output for local PandaAI parquet caches.
- [x] Synthetic negative controls pass.
- [x] Real PandaAI cache run is recorded when the configured cache is available.

## Answer

Implemented with four explicitly labeled evidence channels and no assumed metric
ordering. The real five-asset receipt reports 49,036 unsafe shuffled overlaps, 528
chronological overlaps, and zero overlaps for both Purged and causal channels; see
`docs/evidence/pandaai-benchmark-20260801.md`.
