# BENCH-001A — Explicit PandaAI daily-data adapter

Type: task
Status: resolved
Blocked by: WF-001

- [x] Explicit pandas mapping and benchmark configuration.
- [x] Per-asset forward label and actual shifted Information Interval construction.
- [x] Explicit Decision Time, Feature Availability, PIT snapshot, and source digest.
- [x] Duplicate, finite-value, warm-up/tail, and insufficient-history checks.
- [x] Core import remains pandas-free.
- [x] Public adapter tests pass without network or credentials.

## Answer

Implemented as an optional pandas adapter. It reads caller-provided frames and the
operator CLI reads only caller-provided local parquet files. Synthetic adapter cases and
an offline five-asset PandaAI cache run passed without authentication or network access.
