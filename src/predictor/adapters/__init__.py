"""
src.predictor.adapters
----------------------
Best-effort data adapters that expand the predictor's feature horizon
(docs/PREDICTOR.md §3). Each adapter is independent and degrades gracefully:
a network failure or missing datum yields NaN/empty, never an exception that
blocks a signal. Every fetch records (source, rows, seconds, stale_pct) telemetry.

Data-reality note: free news/fundamentals have no point-in-time historical API, so
these features accrue FORWARD (run daily, store dated parquet) rather than being
backfillable across the training window. They are wired as live signal enrichment
first; the training ablation (hp_002/hp_003) runs once enough history accumulates.
"""
