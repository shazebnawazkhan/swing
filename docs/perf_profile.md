# Performance Profile

_Generated 2026-06-05 23:55:23 — 13 symbols, 39 tasks_

## Stage Breakdown

| Stage | Time (s) | Share |
|---|---|---|
| fetch | 28.04 | 89.6% |
| strategies | 2.17 | 6.9% |
| build_report | 1.07 | 3.4% |
| save_csvs | 0.01 | 0.0% |
| batch_precompute | 0.00 | 0.0% |
| **Total wall time** | **31.29** | 100% |

## Stage 1 — Per-Stock Fetch Times

Mean: 14.85s  Median: 17.36s  Max: 17.92s

### Slowest fetches (top 10)

| Symbol | Time (s) | Rows |
|---|---|---|
| HONASA | 17.92 | 148 |
| GRAVITA | 17.84 | 148 |
| FORCEMOT | 17.75 | 148 |
| MAZDOCK | 17.64 | 148 |
| INDUSTOWER | 17.49 | 148 |
| INFY | 17.38 | 148 |
| KALYANKJIL | 17.36 | 148 |
| E2E | 17.08 | 148 |
| OLECTRA | 10.83 | 148 |
| PCJEWELLER | 10.66 | 148 |

## Stage 2 — Per-Task (Symbol × Strategy) Times

Mean: 284.8ms  Median: 205.4ms  Max: 948.3ms

### Slowest tasks (top 15)

| Symbol | Strategy | Time (ms) |
|---|---|---|
| GRAVITA | Delivery + OI | 948.3 |
| HONASA | Delivery + OI | 855.5 |
| FORCEMOT | Delivery + OI | 669.1 |
| INDUSTOWER | Delivery + OI | 657.7 |
| MAZDOCK | Delivery + OI | 614.2 |
| OLECTRA | Delivery + OI | 589.9 |
| WHIRLPOOL | Delivery + OI | 569.2 |
| KALYANKJIL | Delivery + OI | 557.2 |
| GRAVITA | Volume + EMA Cross | 505.4 |
| INFY | Delivery + OI | 418.4 |
| PERSISTENT | Delivery + OI | 408.4 |
| GRAVITA | EMA + Bollinger Bands | 349.8 |
| PCJEWELLER | Delivery + OI | 346.2 |
| E2E | Delivery + OI | 329.7 |
| TBZ | Delivery + OI | 320.5 |

### Average time per strategy

| Strategy | Avg (ms) | Max (ms) | Tasks |
|---|---|---|---|
| Delivery + OI | 560.3 | 948.3 | 13 |
| EMA + Bollinger Bands | 118.6 | 349.8 | 13 |
| Volume + EMA Cross | 175.5 | 505.4 | 13 |

## Data Shape

Non-empty stocks: 13 / 13  Total rows: 1,924  In-memory: 0.3 MB

| Symbol | Rows | Columns |
|---|---|---|
| E2E | 148 | 12 |
| KALYANKJIL | 148 | 12 |
| INFY | 148 | 12 |
| INDUSTOWER | 148 | 12 |
| MAZDOCK | 148 | 12 |
| FORCEMOT | 148 | 12 |
| GRAVITA | 148 | 12 |
| HONASA | 148 | 12 |
| WHIRLPOOL | 148 | 12 |
| TBZ | 148 | 12 |
| OLECTRA | 148 | 12 |
| PERSISTENT | 148 | 12 |
| PCJEWELLER | 148 | 12 |

## Report Build — Trade Record Counts

Total trade records embedded in HTML: 40  Avg per result: 1.0

### Top 10 results by trade count

| Symbol | Strategy | Trades |
|---|---|---|
| HONASA | EMA + Bollinger Bands | 3 |
| WHIRLPOOL | EMA + Bollinger Bands | 3 |
| FORCEMOT | Volume + EMA Cross | 3 |
| FORCEMOT | EMA + Bollinger Bands | 3 |
| TBZ | EMA + Bollinger Bands | 2 |
| TBZ | Volume + EMA Cross | 2 |
| MAZDOCK | Volume + EMA Cross | 2 |
| INDUSTOWER | EMA + Bollinger Bands | 2 |
| KALYANKJIL | Volume + EMA Cross | 2 |
| E2E | EMA + Bollinger Bands | 2 |

## Hotspot Observations

1. **Network I/O dominates** (28.0s / 90% of wall time). Each stock requires sequential date-range requests to the NSE archive. Bottleneck: TCP round-trips, not CPU.
   - `DATA_WORKERS=8` already parallelises fetches; raising it beyond ~8 risks NSE rate-limiting (429 responses).
   - **Mitigation**: pre-warm the `.nse_cache/` disk cache on first run; subsequent runs skip network entirely and drop fetch time to <1s.

2. **Strategy simulation** (2.2s / 7%). Fast when Numba JIT is active (nogil kernels run truly in parallel). Slow on first cold run due to JIT compilation (~1–2s one-time cost).
   - **Delivery + OI is 4.7× slower** than EMA + Bollinger Bands (avg 560ms vs 118ms). Its `generate_signals()` computes VWAP rolling windows, 30-day support-level sweeps, and 20-day breakout scans in pure pandas — all O(n²) in Python loops. The other two strategies use only vectorised EWM and rolling operations.
   - On a 500-stock universe, Delivery + OI alone would take ~4.7 minutes of CPU time; EMA+BB would take ~1 minute.

3. **`build_report`** (1.07s) — iterates every row of every signal DataFrame to serialise indicator values to JSON using `iterrows()`. Scales linearly with `n_stocks × n_rows × n_signal_cols`. For large universes (>100 stocks) this becomes the CPU bottleneck.
   - **Mitigation**: replace the Python `iterrows()` loop with `df[sig_cols].to_dict(orient='records')` (vectorised, ~10× faster).

## Key Recommendations

| Priority | Change | Expected gain |
|---|---|---|
| **High** | Warm `.nse_cache/` before bulk runs | Eliminates fetch stage (28s → <1s, ~90% reduction) |
| **High** | Vectorise support/breakout scan in Delivery+OI `generate_signals()` | ~4× faster per-task, closes the 4.7× gap vs EMA+BB |
| Medium | Replace `iterrows()` in `build_report` with `.to_dict(orient='records')` | 5–10× faster signal serialisation at scale |
| Medium | Pre-compile Numba kernels at startup (`--warmup` flag) | Removes 1–2s JIT cost on cold runs |
| Low | Increase `DATA_WORKERS` to 12 with exponential-backoff retry | ~30% faster uncached fetches |
| Low | Switch `signal_data` embedding from full history to last 90 days | Cuts HTML payload size by ~50% for long backtests |

---

## Test Run — `fetch_data.py` on 200 stocks

_Run 2026-06-06 00:26  ·  8 workers  ·  incremental (warm `.nse_cache/`)_

### Summary

| Metric | Value |
|---|---|
| Total universe | 200 stocks |
| Fetched (new data) | 200 |
| No data (not in NSE archive) | 0 |
| Errors | 0 |
| Total new rows written | 35,134 |
| Wall time (8 workers) | **536.8s** |
| Throughput | 0.4 stocks/s |
| Rows/sec | 65 |

### Per-Symbol Timing

| Stat | Value |
|---|---|
| Mean | 21.31s |
| Median | 21.09s |
| Stdev | 2.15s |
| Min | 11.33s |
| Max | 27.29s |
| p95 | 24.80s |

#### Slowest 15 symbols

| Symbol | Time (s) | Rows |
|---|---|---|
| TEGA | 27.29 | 184 |
| THEINVEST | 26.38 | 184 |
| TASTYBITE | 26.31 | 184 |
| TATASTEEL | 26.28 | 184 |
| TRU | 25.53 | 184 |
| TRANSWORLD | 25.51 | 184 |
| TRF | 25.29 | 184 |
| TAKE | 25.09 | 184 |
| UNIMECH | 25.03 | 184 |
| UJJIVANSFB | 24.80 | 184 |
| MODINATUR | 24.70 | 127 |
| UMAEXPORTS | 24.55 | 184 |
| MSPL | 24.52 | 184 |
| VIJIFIN | 24.50 | 184 |
| UCAL | 24.43 | 184 |

#### Time distribution

| Bucket | Count |
|---|---|
| <1s | 0 |
| 1–2s | 0 |
| 2–5s | 0 |
| 5–10s | 0 |
| 10–20s | 56 |
| ≥20s | 144 |

### Key Observations

- **Per-symbol time is ~21s even with warm `.nse_cache/`.** The archive cache removes the network layer, but `get_delivery_data()` still reads ~185 daily CSV files sequentially per symbol, parses each (~45K rows), and filters to one row. This is pure disk-I/O + pandas-CSV overhead — not network.
- **Parallelism is CPU-I/O bound, not network bound.** 8 workers × 25 batches × ~21s/batch = 536s. Raising `DATA_WORKERS` beyond 8 would help here (no rate-limit risk since we're reading from disk), scaling roughly as `200 × 21s / workers`.
- **At 16 workers the same 200-stock fetch would complete in ~268s (~4.5 min); at 32 workers ~134s (~2 min)** — straightforward win with no code changes.
- **The parquet layer is the real payoff.** Now that `data/fetched/*.parquet` is populated, `--from-cache` loads all 200 stocks in <1s by reading 200-row typed binary files instead of 185 large CSVs per symbol. Strategy re-runs become instant.
- **Full 2,364-stock universe** at 8 workers ≈ 8,800s (~2.5 hrs); at 32 workers ≈ 2,200s (~37 min). Viable as a nightly job.
- **Throughput ceiling per worker:** 1 stock / 21s = ~0.048 stocks/s/worker — set by CSV scan speed of the NSE archive format. A future optimisation is to invert the loop: scan each day's CSV once and extract all symbols, reducing per-symbol disk reads from 185 to 1.
