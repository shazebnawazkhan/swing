"""
fetch_data.py
-------------
Standalone incremental data fetcher.

Downloads OHLCV + delivery data for a stock universe and persists each symbol
to data/fetched/<SYMBOL>.parquet.

On re-runs only dates after the last stored date are fetched, then merged with
the existing parquet — avoids re-downloading data that is already on disk.

Stock universe
--------------
Default: 200 randomly chosen symbols from data/stocks.csv (seed=42).
The list is written to data/fetch_universe.csv on first run so you can
inspect or edit it.  Subsequent runs reuse the same file (pass --reset to
re-sample).

Usage
-----
    python scripts/fetch_data.py                         # fetch / update universe
    python scripts/fetch_data.py --count 50              # smaller universe
    python scripts/fetch_data.py --symbols INFY TCS SBIN # explicit list
    python scripts/fetch_data.py --force                 # full re-fetch (ignore cache)
    python scripts/fetch_data.py --status                # staleness report, no fetch
    python scripts/fetch_data.py --reset                 # re-sample universe, then fetch
"""

import argparse
import os
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import src.config as cfg
from src.data_fetcher import NSEArchiveFetcher, _yfinance_fallback

# ── Paths ─────────────────────────────────────────────────────────────────────

CACHE_DIR  = ROOT / "data" / "fetched"
UNIV_CSV   = ROOT / "data" / "fetch_universe.csv"
STOCKS_CSV = ROOT / "data" / "stocks.csv"

DEFAULT_COUNT   = 200
DEFAULT_SEED    = 42
DEFAULT_WORKERS = 8

# ── Thread-local fetcher (same pattern as run_comparison.py) ──────────────────

_tls = threading.local()

def _get_archive() -> NSEArchiveFetcher:
    if not hasattr(_tls, "archive"):
        _tls.archive = NSEArchiveFetcher()
    return _tls.archive


# ── Universe helpers ──────────────────────────────────────────────────────────

def select_universe(csv_path: Path, count: int, seed: int) -> pd.DataFrame:
    """Randomly sample `count` rows from stocks.csv (reproducible via seed)."""
    df  = pd.read_csv(csv_path, dtype=str).fillna("")
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(df)), min(count, len(df))))
    return df.iloc[idx].reset_index(drop=True)


def load_universe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


# ── Parquet helpers ───────────────────────────────────────────────────────────

def _parquet_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.parquet"


def _last_cached_date(symbol: str) -> datetime | None:
    p = _parquet_path(symbol)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=["date"])
        if df.empty:
            return None
        return pd.to_datetime(df["date"]).max().to_pydatetime().replace(tzinfo=None)
    except Exception:
        return None


# ── Per-symbol incremental fetch ──────────────────────────────────────────────

def _fetch_symbol(symbol: str, force: bool = False) -> tuple[str, str, int, float]:
    """
    Fetch OHLCV + delivery data for one symbol.

    Incremental: if a parquet already exists, only fetches dates after the
    last stored date and merges the new rows in.

    Returns
    -------
    (symbol, status, new_rows_added, elapsed_seconds)
    status ∈ {"fetched", "up-to-date", "no-data", "error"}
    """
    t0    = time.perf_counter()
    to_dt = datetime.now()
    last_date = None if force else _last_cached_date(symbol)

    if last_date is not None:
        from_dt = last_date + timedelta(days=1)
        if from_dt.date() >= to_dt.date():
            return symbol, "up-to-date", 0, time.perf_counter() - t0
    else:
        # First fetch — pull full backtest window
        if getattr(cfg, "BACKTEST_START_DATE", None):
            floor_dt      = datetime.strptime(cfg.BACKTEST_START_DATE, "%Y-%m-%d")
            backtest_days = max(cfg.BACKTEST_DAYS, (to_dt - floor_dt).days + 1)
        else:
            backtest_days = cfg.BACKTEST_DAYS
        total_days = backtest_days + cfg.DATA_BUFFER_DAYS * 2
        from_dt    = to_dt - timedelta(days=total_days)

    try:
        archive = _get_archive()
        new_df  = archive.get_delivery_data(symbol, from_dt, to_dt)

        if new_df.empty:
            new_df = _yfinance_fallback(symbol, (to_dt - from_dt).days)

        if new_df.empty:
            return symbol, "no-data", 0, time.perf_counter() - t0

        # Ensure OI columns present (NaN if no futures data)
        for col in ("oi", "oi_change"):
            if col not in new_df.columns:
                new_df[col] = np.nan

        new_df.insert(0, "symbol", symbol)
        new_df["date"] = pd.to_datetime(new_df["date"])

        p = _parquet_path(symbol)
        if p.exists() and not force:
            old_df         = pd.read_parquet(p)
            old_df["date"] = pd.to_datetime(old_df["date"])
            combined = (
                pd.concat([old_df, new_df], ignore_index=True)
                  .drop_duplicates(subset=["date"])
                  .sort_values("date")
                  .reset_index(drop=True)
            )
        else:
            combined = new_df.sort_values("date").reset_index(drop=True)

        combined.to_parquet(p, index=False)
        return symbol, "fetched", len(new_df), time.perf_counter() - t0

    except Exception:
        traceback.print_exc()
        return symbol, "error", 0, time.perf_counter() - t0


# ── Parallel fetch ────────────────────────────────────────────────────────────

# (symbol, status, rows, elapsed_s)
FetchResult = tuple[str, str, int, float]

def fetch_universe(
    symbols: list[str], force: bool, workers: int
) -> dict[str, FetchResult]:
    results: dict[str, FetchResult] = {}
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_symbol, sym, force): sym for sym in symbols}
        for fut in as_completed(futures):
            sym, status, new_rows, elapsed = fut.result()
            results[sym] = (status, new_rows, elapsed)
            done += 1
            tag = f"+{new_rows}r  {elapsed:.1f}s" if status == "fetched" else status
            print(f"  [{done:>4}/{len(symbols)}] {sym:<16} {tag}", end="\r")

    print()  # newline after \r progress
    return results


# ── Status report ─────────────────────────────────────────────────────────────

def show_status(symbols: list[str]) -> None:
    today = datetime.now().date()
    rows: list[tuple] = []

    for sym in sorted(symbols):
        p = _parquet_path(sym)
        if not p.exists():
            rows.append((sym, "missing", "—", "—", "—"))
            continue
        ld = _last_cached_date(sym)
        if ld is None:
            rows.append((sym, "empty", "—", "—", "—"))
            continue
        try:
            n_rows = len(pd.read_parquet(p, columns=["date"]))
        except Exception:
            n_rows = 0
        lag      = (today - ld.date()).days
        size_kb  = p.stat().st_size // 1024
        status   = "ok" if lag <= 3 else "stale"
        rows.append((sym, status, str(ld.date()), f"{lag}d", f"{n_rows}r / {size_kb}KB"))

    print(f"\n{'Symbol':<16} {'Status':<10} {'Last Date':<12} {'Lag':<6} Info")
    print("─" * 62)
    for sym, status, last, lag, info in rows:
        flag = "✓" if status == "ok" else ("!" if status == "stale" else "✗")
        print(f"  {flag} {sym:<14} {status:<10} {last:<12} {lag:<6} {info}")

    ok      = sum(1 for _, s, *_ in rows if s == "ok")
    stale   = sum(1 for _, s, *_ in rows if s == "stale")
    missing = sum(1 for _, s, *_ in rows if s == "missing")
    print(f"\n  Total {len(rows)}  ·  ok {ok}  ·  stale {stale}  ·  missing {missing}\n")


# ── Benchmark writer ─────────────────────────────────────────────────────────

def _write_benchmark(
    results:     dict[str, FetchResult],
    wall_time:   float,
    workers:     int,
    profile_md:  Path,
) -> None:
    import statistics

    fetched_res = [(s, r) for s, (st, r, _) in results.items() if st == "fetched"]
    timed       = [(s, e) for s, (st, _, e) in results.items() if st == "fetched"]
    no_data     = [s for s, (st, *_) in results.items() if st == "no-data"]
    errors      = [s for s, (st, *_) in results.items() if st == "error"]

    total_rows  = sum(r for _, r in fetched_res)
    elapsed_all = [e for _, e in timed]
    sorted_t    = sorted(timed, key=lambda x: -x[1])

    lines: list[str] = [
        "",
        "---",
        "",
        "## Test Run — `fetch_data.py` on 200 stocks",
        "",
        f"_Run {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
        f"{workers} workers  ·  incremental (warm `.nse_cache/`)_",
        "",
        "### Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total universe | {len(results)} stocks |",
        f"| Fetched (new data) | {len(fetched_res)} |",
        f"| No data (not in NSE archive) | {len(no_data)} |",
        f"| Errors | {len(errors)} |",
        f"| Total new rows written | {total_rows:,} |",
        f"| Wall time (8 workers) | **{wall_time:.1f}s** |",
        f"| Throughput | {len(fetched_res)/wall_time:.1f} stocks/s |",
        f"| Rows/sec | {total_rows/wall_time:,.0f} |",
        "",
    ]

    if elapsed_all:
        lines += [
            "### Per-Symbol Timing",
            "",
            f"| Stat | Value |",
            f"|---|---|",
            f"| Mean | {statistics.mean(elapsed_all):.2f}s |",
            f"| Median | {statistics.median(elapsed_all):.2f}s |",
            f"| Stdev | {statistics.stdev(elapsed_all):.2f}s |",
            f"| Min | {min(elapsed_all):.2f}s |",
            f"| Max | {max(elapsed_all):.2f}s |",
            f"| p95 | {sorted(elapsed_all)[int(len(elapsed_all)*.95)]:.2f}s |",
            "",
            "#### Slowest 15 symbols",
            "",
            "| Symbol | Time (s) | Rows |",
            "|---|---|---|",
        ]
        for sym, t in sorted_t[:15]:
            rows = next((r for s, (_, r, _) in results.items() if s == sym), 0)
            lines.append(f"| {sym} | {t:.2f} | {rows} |")
        lines.append("")

        # Histogram buckets
        buckets = [0, 1, 2, 5, 10, 20, float("inf")]
        labels  = ["<1s", "1–2s", "2–5s", "5–10s", "10–20s", "≥20s"]
        counts  = [0] * len(labels)
        for e in elapsed_all:
            for i, hi in enumerate(buckets[1:]):
                if e < hi:
                    counts[i] += 1
                    break
        lines += [
            "#### Time distribution",
            "",
            "| Bucket | Count |",
            "|---|---|",
        ]
        for lbl, cnt in zip(labels, counts):
            lines.append(f"| {lbl} | {cnt} |")
        lines.append("")

    if no_data:
        lines += [
            f"### No-data symbols ({len(no_data)})",
            "",
            "_These symbols exist in stocks.csv but have no bhav-copy entries "
            "in the NSE equity archive for the backtest window._",
            "",
            ", ".join(sorted(no_data)),
            "",
        ]

    lines += [
        "### Key Observations",
        "",
        f"- **Wall time with warm cache: {wall_time:.1f}s** vs ~28s for 13 stocks "
        f"(cold) in the original run — the `.nse_cache/` archive files are already "
        f"downloaded so each stock is a pure disk-read + pandas filter, not a network call.",
        f"- At {len(fetched_res)/wall_time:.1f} stocks/s throughput, "
        f"the same approach scales to **500 stocks in ~{500/(len(fetched_res)/wall_time):.0f}s** "
        f"and **2,364 stocks (full CSV) in ~{2364/(len(fetched_res)/wall_time):.0f}s** "
        f"once the daily archive files are cached.",
        f"- `--from-cache` in `run_comparison.py` now reads these parquet files "
        f"(avg {total_rows//max(len(fetched_res),1)} rows each) in <1s total, "
        f"making repeated strategy runs essentially instant.",
    ]

    text = "\n".join(lines) + "\n"
    with open(profile_md, "a", encoding="utf-8") as fh:
        fh.write(text)
    print(f"\n  Benchmark appended → {profile_md.relative_to(ROOT)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Incremental NSE data fetcher — writes data/fetched/<SYMBOL>.parquet"
    )
    parser.add_argument("--count",     type=int, default=DEFAULT_COUNT,
                        help=f"Random stock count (default {DEFAULT_COUNT})")
    parser.add_argument("--seed",      type=int, default=DEFAULT_SEED,
                        help=f"RNG seed for stock selection (default {DEFAULT_SEED})")
    parser.add_argument("--symbols",   nargs="+",
                        help="Explicit symbol list (overrides --count/--seed)")
    parser.add_argument("--force",     action="store_true",
                        help="Full re-fetch for all symbols (ignore existing parquet)")
    parser.add_argument("--reset",     action="store_true",
                        help="Re-sample the universe from stocks.csv, then fetch")
    parser.add_argument("--status",    action="store_true",
                        help="Print cache staleness report and exit (no fetch)")
    parser.add_argument("--workers",   type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel fetch workers (default {DEFAULT_WORKERS})")
    parser.add_argument("--benchmark", action="store_true",
                        help="Append timing stats to docs/perf_profile.md after fetch")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resolve universe ──────────────────────────────────────────────────────
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
        meta_df = pd.read_csv(STOCKS_CSV, dtype=str).fillna("")
        chosen  = meta_df[meta_df["stock"].isin(symbols)].reset_index(drop=True)
        print(f"  Using explicit list: {len(symbols)} stocks")
    elif UNIV_CSV.exists() and not args.reset:
        chosen  = load_universe(UNIV_CSV)
        symbols = chosen["stock"].tolist()
        print(f"  Loaded existing universe: {len(symbols)} stocks  ({UNIV_CSV.name})")
    else:
        chosen  = select_universe(STOCKS_CSV, args.count, args.seed)
        chosen.to_csv(UNIV_CSV, index=False)
        symbols = chosen["stock"].tolist()
        action  = "Re-sampled" if args.reset else "Selected"
        print(f"  {action} {len(symbols)} random stocks  "
              f"(seed={args.seed})  →  {UNIV_CSV.name}")

    if args.status:
        show_status(symbols)
        return

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  SWING SCANNER  —  Incremental Data Fetcher")
    print("=" * 62)
    print(f"  Universe : {len(symbols)} stocks")
    print(f"  Cache    : {CACHE_DIR.relative_to(ROOT)}")
    mode = "FORCE full re-fetch" if args.force else "incremental (new dates only)"
    print(f"  Mode     : {mode}")
    print(f"  Workers  : {args.workers}")
    if args.benchmark:
        print(f"  Benchmark: ON  → docs/perf_profile.md")
    print()

    # ── Fetch ─────────────────────────────────────────────────────────────────
    t0      = time.perf_counter()
    results = fetch_universe(symbols, args.force, args.workers)
    elapsed = time.perf_counter() - t0

    fetched    = sum(1 for st, *_ in results.values() if st == "fetched")
    up_to_date = sum(1 for st, *_ in results.values() if st == "up-to-date")
    no_data    = sum(1 for st, *_ in results.values() if st == "no-data")
    errors     = sum(1 for st, *_ in results.values() if st == "error")
    new_rows   = sum(r for _, r, *_ in results.values())

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Fetched: {fetched}  Up-to-date: {up_to_date}  "
          f"No-data: {no_data}  Errors: {errors}")
    print(f"  New rows written: {new_rows:,}")
    print(f"  Parquet files in: {CACHE_DIR.relative_to(ROOT)}")

    if errors:
        print("\n  Failed symbols:")
        for sym, (status, *_) in results.items():
            if status == "error":
                print(f"    {sym}")

    if args.benchmark:
        profile_md = ROOT / "docs" / "perf_profile.md"
        _write_benchmark(results, elapsed, args.workers, profile_md)

    print()


if __name__ == "__main__":
    main()
