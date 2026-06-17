"""
bulk_fetch.py
-------------
Date-major bulk fetcher for large universes (e.g. all halal stocks).

Why a separate script from fetch_data.py
----------------------------------------
fetch_data.py is symbol-major: fetching one symbol re-parses every daily
bhav-copy CSV.  Fine for tens of symbols; for 2,000+ symbols x 250 days it
would mean ~500k CSV parses.  This script inverts the loop:

    1. ensure every daily archive file for the window is in .nse_cache/
       (parallel download of the missing ones; holidays remembered)
    2. parse each daily file ONCE, filter SERIES==EQ + symbol in universe
    3. group the combined frame by symbol -> merge into data/fetched/<SYM>.parquet

A manifest (data/.bulk_manifest.json) records which archive dates are already
merged, so re-runs only parse new days.  Output schema matches fetch_data.py:

    symbol, date, open, high, low, close, total_volume,
    delivery_qty, delivery_pct, vwap, oi, oi_change

Usage
-----
    python scripts/bulk_fetch.py                          # halal universe, from 2025-06-01
    python scripts/bulk_fetch.py --start 2025-01-01       # longer warm-up window
    python scripts/bulk_fetch.py --universe all           # every stock in stocks.csv
    python scripts/bulk_fetch.py --status                 # coverage report, no fetch
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from src.data_fetcher import NSEArchiveFetcher

NSE_CACHE   = ROOT / ".nse_cache"
FETCHED_DIR = ROOT / "data" / "fetched"
STOCKS_CSV  = ROOT / "data" / "stocks.csv"
MANIFEST    = ROOT / "data" / ".bulk_manifest.json"
HOLIDAYS    = ROOT / "data" / ".archive_holidays.json"

DEFAULT_START   = "2025-06-01"   # 1y backtest window + indicator warm-up
DEFAULT_WORKERS = 8

_tls = threading.local()


def _get_archive() -> NSEArchiveFetcher:
    if not hasattr(_tls, "archive"):
        _tls.archive = NSEArchiveFetcher()
    return _tls.archive


# ── Universe ──────────────────────────────────────────────────────────────────

def load_universe(kind: str) -> list[str]:
    df = pd.read_csv(STOCKS_CSV, dtype=str).fillna("")
    if kind == "halal":
        df = df[df["halal"].str.upper() == "Y"]
    return sorted(df["stock"].str.strip().unique())


# ── Date bookkeeping ──────────────────────────────────────────────────────────

def _weekdays(start: datetime, end: datetime) -> list[datetime]:
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _load_json_set(path: Path) -> set[str]:
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _save_json_set(path: Path, items: set[str]) -> None:
    path.write_text(json.dumps(sorted(items)))


def _cache_file(day: datetime) -> Path:
    return NSE_CACHE / f"eq_{day:%d%m%Y}.csv"


# ── Step 1: ensure archive files are cached ───────────────────────────────────

def download_missing(days: list[datetime], workers: int) -> tuple[int, int]:
    """Download archive CSVs not yet in .nse_cache. Returns (downloaded, holidays_hit)."""
    holidays = _load_json_set(HOLIDAYS)
    today = datetime.now().date()
    todo = [d for d in days
            if not _cache_file(d).exists()
            and d.strftime("%Y-%m-%d") not in holidays
            and d.date() < today]                     # today's file appears after EOD

    if not todo:
        return 0, 0

    print(f"  Downloading {len(todo)} missing archive files "
          f"({todo[0]:%Y-%m-%d} … {todo[-1]:%Y-%m-%d}, {workers} workers)")

    new_holidays: set[str] = set()
    done = ok = 0

    def _one(day: datetime) -> tuple[datetime, bool]:
        arch = _get_archive()
        url = arch.EQ_URL.format(date=day.strftime("%d%m%Y"))
        raw = arch._cached_get(url, str(_cache_file(day)))
        return day, raw is not None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, d) for d in todo]
        for fut in as_completed(futures):
            day, success = fut.result()
            done += 1
            if success:
                ok += 1
            else:
                new_holidays.add(day.strftime("%Y-%m-%d"))
            print(f"    [{done:>3}/{len(todo)}] {day:%Y-%m-%d} "
                  f"{'ok' if success else 'no file (holiday?)'}", end="\r")
    print()

    if new_holidays:
        _save_json_set(HOLIDAYS, holidays | new_holidays)
    return ok, len(new_holidays)


# ── Step 2: parse daily files once each ───────────────────────────────────────

_KEEP = {
    "OPEN_PRICE": "open", "HIGH_PRICE": "high", "LOW_PRICE": "low",
    "CLOSE_PRICE": "close", "AVG_PRICE": "vwap", "TTL_TRD_QNTY": "total_volume",
    "DELIV_QTY": "delivery_qty", "DELIV_PER": "delivery_pct",
}


def parse_day(day: datetime, universe: set[str]) -> pd.DataFrame | None:
    """Parse one cached bhav copy; return normalised rows for universe symbols."""
    path = _cache_file(day)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, skipinitialspace=True, dtype=str)
    except Exception:
        return None
    df.columns = [c.strip().upper() for c in df.columns]
    if "SYMBOL" not in df.columns or "SERIES" not in df.columns:
        return None

    df["SYMBOL"] = df["SYMBOL"].str.strip()
    mask = (df["SERIES"].str.strip() == "EQ") & df["SYMBOL"].isin(universe)
    df = df.loc[mask, ["SYMBOL"] + [c for c in _KEEP if c in df.columns]]
    if df.empty:
        return None

    out = df.rename(columns={"SYMBOL": "symbol", **_KEEP})
    for col in _KEEP.values():
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col].str.replace(",", "", regex=False), errors="coerce")
        else:
            out[col] = np.nan
    out["date"] = pd.Timestamp(day.date())
    return out


# ── Step 3: merge into per-symbol parquets ────────────────────────────────────

def merge_to_parquets(big: pd.DataFrame) -> tuple[int, int]:
    """Merge the combined frame into data/fetched/<SYM>.parquet. Returns (symbols, rows)."""
    FETCHED_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["symbol", "date", "open", "high", "low", "close", "total_volume",
            "delivery_qty", "delivery_pct", "vwap", "oi", "oi_change"]
    for c in ("oi", "oi_change"):
        big[c] = np.nan
    big = big[cols]

    n_syms = n_rows = 0
    groups = big.groupby("symbol", sort=True)
    total = groups.ngroups
    for sym, g in groups:
        p = FETCHED_DIR / f"{sym}.parquet"
        g = g.sort_values("date")
        if p.exists():
            old = pd.read_parquet(p)
            old["date"] = pd.to_datetime(old["date"])
            g = (pd.concat([old, g], ignore_index=True)
                 .drop_duplicates(subset=["date"], keep="first")
                 .sort_values("date").reset_index(drop=True))
        g.to_parquet(p, index=False)
        n_syms += 1
        n_rows += len(g)
        if n_syms % 100 == 0:
            print(f"    [{n_syms:>4}/{total}] parquets written", end="\r")
    print()
    return n_syms, n_rows


# ── Status report ─────────────────────────────────────────────────────────────

def show_status(universe: list[str]) -> None:
    have = {p.stem for p in FETCHED_DIR.glob("*.parquet")}
    missing = [s for s in universe if s not in have]
    spans = []
    for sym in universe[:0] or sorted(have & set(universe))[:2000]:
        try:
            d = pd.read_parquet(FETCHED_DIR / f"{sym}.parquet", columns=["date"])
            spans.append((len(d), pd.to_datetime(d["date"]).min(),
                          pd.to_datetime(d["date"]).max()))
        except Exception:
            pass
    print(f"\n  Universe          : {len(universe)} symbols")
    print(f"  Parquets present  : {len(have & set(universe))}")
    print(f"  Missing           : {len(missing)}"
          + (f"  e.g. {missing[:8]}" if missing else ""))
    if spans:
        rows = [n for n, *_ in spans]
        print(f"  Rows/symbol       : min {min(rows)}  median "
              f"{int(np.median(rows))}  max {max(rows)}")
        print(f"  Date range        : {min(s for _, s, _ in spans):%Y-%m-%d} → "
              f"{max(e for *_, e in spans):%Y-%m-%d}")
    manifest = _load_json_set(MANIFEST)
    print(f"  Manifest days     : {len(manifest)} merged archive dates\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Date-major bulk NSE fetcher")
    ap.add_argument("--universe", choices=["halal", "all"], default="halal")
    ap.add_argument("--start", default=DEFAULT_START, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--status", action="store_true", help="coverage report only")
    ap.add_argument("--force", action="store_true",
                    help="re-parse all days, ignore manifest")
    args = ap.parse_args()

    universe = load_universe(args.universe)
    if args.status:
        show_status(universe)
        return

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = (datetime.strptime(args.end, "%Y-%m-%d") if args.end
           else datetime.now() - timedelta(days=1))
    days = _weekdays(start, end)

    print("=" * 62)
    print("  BULK FETCH — date-major NSE archive → per-symbol parquet")
    print("=" * 62)
    print(f"  Universe : {args.universe} ({len(universe)} symbols)")
    print(f"  Window   : {start:%Y-%m-%d} → {end:%Y-%m-%d} ({len(days)} weekdays)")

    t0 = time.perf_counter()
    downloaded, holidays = download_missing(days, args.workers)
    t_dl = time.perf_counter() - t0

    manifest = set() if args.force else _load_json_set(MANIFEST)
    todo = [d for d in days
            if d.strftime("%Y-%m-%d") not in manifest and _cache_file(d).exists()]
    print(f"  Archive files: +{downloaded} downloaded ({holidays} holidays), "
          f"{len(todo)} days to parse ({t_dl:.0f}s)")

    if not todo:
        print("  Nothing new to merge — up to date.\n")
        show_status(universe)
        return

    t1 = time.perf_counter()
    uni_set = set(universe)
    frames = []
    for i, day in enumerate(sorted(todo), 1):
        f = parse_day(day, uni_set)
        if f is not None:
            frames.append(f)
        if i % 25 == 0:
            print(f"    [{i:>3}/{len(todo)}] days parsed", end="\r")
    print()
    if not frames:
        print("  No parseable data found.\n")
        return
    big = pd.concat(frames, ignore_index=True)
    t_parse = time.perf_counter() - t1
    print(f"  Parsed {len(frames)} days → {len(big):,} rows "
          f"({big['symbol'].nunique()} symbols) in {t_parse:.0f}s")

    t2 = time.perf_counter()
    n_syms, n_rows = merge_to_parquets(big)
    t_merge = time.perf_counter() - t2

    _save_json_set(MANIFEST, manifest | {d.strftime("%Y-%m-%d") for d in todo})

    print(f"  Merged → {n_syms} parquets ({n_rows:,} total rows) in {t_merge:.0f}s")
    print(f"  Total wall time: {time.perf_counter() - t0:.0f}s\n")
    show_status(universe)


if __name__ == "__main__":
    main()
