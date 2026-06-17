"""
run_comparison.py
-----------------
Bulk backtester: run ALL registered strategies on ALL stocks marked
enabled=Y in stocks.csv.  Outputs comparison_report.html.

Parallelism
-----------
Stage 1  Data fetch   ThreadPoolExecutor(DATA_WORKERS)
         Each thread owns its own NSEArchiveFetcher (thread-local Session) so
         concurrent downloads never race on a shared requests.Session.
         NSE archive files are cached to .nse_cache/ after the first download.

Stage 2  Strategies   ThreadPoolExecutor(STRAT_WORKERS)
         Numba JIT simulation (nogil=True) lets threads run truly in parallel.
         Without Numba the GIL is released by most numpy/pandas operations.

Stage 2b CUDA / Numba prange batch pre-computation
         If CuPy or Numba is available, EMA + Bollinger Bands are computed for
         ALL stocks in one vectorised GPU/SIMD kernel call before dispatching.

Usage
-----
    python run_comparison.py
    python run_comparison.py --csv my.csv
    python run_comparison.py --strategy "Volume + EMA Cross"
    python run_comparison.py --workers 8
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import src.config as cfg
import src.fast_indicators as fi
from src.backtester import Backtester
from src.data_fetcher import NSEArchiveFetcher, _yfinance_fallback
from src.strategies import all_strategies, get_strategy, list_strategy_names
from src.strategies.base import BacktestResult, TradeRecord

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s  %(name)s  %(message)s")

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_WORKERS  = 8    # conservative: avoids NSE rate-limiting
STRAT_WORKERS = max(4, (os.cpu_count() or 4) * 2)
OUTPUT_HTML   = "outputs/comparison_report.html"

# ── Profiling state (populated when --profile is passed) ─────────────────────

_PROFILE      = False
_prof_lock    = threading.Lock()
_prof_fetch:  list[tuple[str, float, int]] = []   # (symbol, elapsed_s, rows)
_prof_tasks:  list[tuple[str, str, float]] = []   # (symbol, strategy, elapsed_s)
_prof_stages: dict[str, float]             = {}   # stage_name -> elapsed_s

# ── Thread-local archive fetcher ──────────────────────────────────────────────
# Each thread gets its own NSEArchiveFetcher (its own requests.Session).
# This eliminates all shared-session race conditions without sacrificing
# parallelism — urllib3 connection pools within a session ARE thread-safe
# for that one session object.

_tls = threading.local()

def _get_archive() -> NSEArchiveFetcher:
    if not hasattr(_tls, "archive"):
        _tls.archive = NSEArchiveFetcher()
    return _tls.archive


# ── Startup banner ────────────────────────────────────────────────────────────

def _banner():
    print()
    print("=" * 68)
    print("  SWING SCANNER  —  Bulk Comparison Runner")
    print("=" * 68)
    print(f"  Acceleration : {fi.backend()}")
    print(f"  Fetch workers: {DATA_WORKERS}   Strategy workers: {STRAT_WORKERS}")
    print("=" * 68)
    print()


# ── Stock list ────────────────────────────────────────────────────────────────

def load_enabled_stocks(csv_path: str = "stocks.csv", all_stocks: bool = False) -> pd.DataFrame:
    df  = pd.read_csv(csv_path, dtype=str).fillna("")
    if not all_stocks:
        mask = df["enabled"].str.strip().str.upper() == "Y"
        df   = df[mask].copy().reset_index(drop=True)
    label = "All" if all_stocks else "Enabled"
    print(f"  {label} stocks: {len(df)}  (from {csv_path})")
    return df


# ── Cache loader (--from-cache mode) ─────────────────────────────────────────

def load_from_cache(
    symbols:   list[str],
    cache_dir: str = "data/fetched",
) -> dict[str, pd.DataFrame]:
    """
    Load pre-fetched DataFrames from parquet files and trim to the configured
    backtest window.  Symbols without a parquet file are skipped silently.
    """
    from pathlib import Path as _Path

    p     = _Path(cache_dir)
    to_dt = datetime.now()

    if getattr(cfg, "BACKTEST_START_DATE", None):
        floor_dt      = datetime.strptime(cfg.BACKTEST_START_DATE, "%Y-%m-%d")
        backtest_days = max(cfg.BACKTEST_DAYS, (to_dt - floor_dt).days + 1)
    else:
        backtest_days = cfg.BACKTEST_DAYS

    cutoff = to_dt - timedelta(days=backtest_days + cfg.DATA_BUFFER_DAYS)
    if getattr(cfg, "BACKTEST_START_DATE", None):
        cutoff = min(cutoff, floor_dt - timedelta(days=cfg.DATA_BUFFER_DAYS))

    stock_data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        fp = p / f"{sym}.parquet"
        if not fp.exists():
            continue
        try:
            df         = pd.read_parquet(fp)
            df["date"] = pd.to_datetime(df["date"])
            df         = df[df["date"] >= cutoff].reset_index(drop=True)
            if df.empty:
                continue
            if "symbol" not in df.columns:
                df.insert(0, "symbol", sym)
            for col in ("oi", "oi_change"):
                if col not in df.columns:
                    df[col] = np.nan
            stock_data[sym] = df
        except Exception:
            traceback.print_exc()

    ok = len(stock_data)
    print(f"  Loaded {ok}/{len(symbols)} stocks from cache  ({cache_dir})\n")
    return stock_data


def _discover_cache_symbols(cache_dir: str) -> list[str]:
    """Return all symbol names that have a parquet file in cache_dir."""
    from pathlib import Path as _Path
    return sorted(p.stem for p in _Path(cache_dir).glob("*.parquet"))


# ── Stage 1: parallel data fetch ─────────────────────────────────────────────

def _fetch_one(symbol: str) -> tuple[str, pd.DataFrame]:
    """
    Fetch OHLCV + delivery data for one symbol using a thread-local
    NSEArchiveFetcher so concurrent threads never share a Session object.
    Falls back to yfinance if the archive returns nothing.
    """
    _t0   = time.perf_counter()
    to_dt = datetime.now()

    # Honour BACKTEST_START_DATE if set; fallback to rolling BACKTEST_DAYS
    if getattr(cfg, "BACKTEST_START_DATE", None):
        floor_dt    = datetime.strptime(cfg.BACKTEST_START_DATE, "%Y-%m-%d")
        backtest_days = max(cfg.BACKTEST_DAYS, (to_dt - floor_dt).days + 1)
    else:
        backtest_days = cfg.BACKTEST_DAYS

    total_days = backtest_days + cfg.DATA_BUFFER_DAYS * 2
    from_dt    = to_dt - timedelta(days=total_days)

    try:
        archive = _get_archive()
        df = archive.get_delivery_data(symbol, from_dt, to_dt)

        if df.empty:
            df = _yfinance_fallback(symbol, total_days)

        if df.empty:
            return symbol, pd.DataFrame()

        # Ensure OI columns exist (will be NaN — Delivery+OI cond_oi will be False)
        for col in ("oi", "oi_change"):
            if col not in df.columns:
                df[col] = np.nan

        # Trim to the backtest window (keep buffer for indicator warm-up)
        cutoff = to_dt - timedelta(days=backtest_days + cfg.DATA_BUFFER_DAYS)
        if getattr(cfg, "BACKTEST_START_DATE", None):
            cutoff = min(cutoff, floor_dt - timedelta(days=cfg.DATA_BUFFER_DAYS))
        df = df[df["date"] >= cutoff].reset_index(drop=True)

        if df.empty:
            if _PROFILE:
                with _prof_lock:
                    _prof_fetch.append((symbol, time.perf_counter() - _t0, 0))
            return symbol, pd.DataFrame()

        df.insert(0, "symbol", symbol)
        if _PROFILE:
            with _prof_lock:
                _prof_fetch.append((symbol, time.perf_counter() - _t0, len(df)))
        return symbol, df

    except Exception:
        # Log so the user can diagnose, but never crash the whole pool
        traceback.print_exc()
        if _PROFILE:
            with _prof_lock:
                _prof_fetch.append((symbol, time.perf_counter() - _t0, 0))
        return symbol, pd.DataFrame()


def fetch_all(symbols: list[str]) -> dict[str, pd.DataFrame]:
    print(f"\n[Stage 1/2] Fetching {len(symbols)} stock(s)  ({DATA_WORKERS} workers)…")
    t0         = time.perf_counter()
    stock_data: dict[str, pd.DataFrame] = {}
    done       = 0

    with ThreadPoolExecutor(max_workers=DATA_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym, df = fut.result()
            stock_data[sym] = df
            done += 1
            status = f"{len(df)} rows" if not df.empty else "NO DATA"
            print(f"  [{done:>4}/{len(symbols)}] {sym:<16} {status}", end="\r")

    elapsed  = time.perf_counter() - t0
    ok_count = sum(1 for d in stock_data.values() if not d.empty)
    print(f"\n  Fetch done in {elapsed:.1f}s — {ok_count}/{len(symbols)} stocks have data\n")
    return stock_data


# ── Optional: batch GPU/SIMD pre-computation ──────────────────────────────────

def _batch_precompute(stock_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    If CUDA or Numba is available, compute shared indicators for ALL stocks
    simultaneously in one kernel call, then inject back into each DataFrame.
    """
    if not ("cuda" in fi.backend() or "numba" in fi.backend()):
        return stock_data

    non_empty = {s: d for s, d in stock_data.items() if not d.empty}
    if not non_empty:
        return stock_data

    T       = max(len(d) for d in non_empty.values())
    symbols = list(non_empty.keys())
    N       = len(symbols)

    # Build (N × T) close-price matrix, left-aligned, NaN-padded on the right
    matrix = np.full((N, T), np.nan)
    lengths: list[int] = []
    for i, sym in enumerate(symbols):
        c = non_empty[sym]["close"].to_numpy(dtype=np.float64)
        matrix[i, :len(c)] = c
        lengths.append(len(c))

    # Batch compute in one kernel call
    spans = [9, 10, 21, 30]
    ema_results  = {span: fi.batch_ema(matrix, span)          for span in spans}
    bb_mean      = fi.batch_rolling_mean(matrix, 15)

    # Inject pre-computed columns — slice [:rows] (left-aligned)
    updated = {}
    for i, sym in enumerate(symbols):
        rows = lengths[i]
        df   = non_empty[sym].copy()
        for span, mat in ema_results.items():
            df[f"_pre_ema{span}"] = mat[i, :rows]
        df["_pre_bb_mid15"] = bb_mean[i, :rows]
        updated[sym] = df

    print(f"  Batch pre-computed EMA{spans}+BB15 for {N} stocks via {fi.backend().split()[0]}")
    return {**stock_data, **updated}


# ── Stage 2: Numba-accelerated backtester ─────────────────────────────────────

class FastBacktester(Backtester):
    """
    Replaces Backtester._simulate with a Numba JIT kernel.
    nogil=True on the kernel lets ThreadPoolExecutor threads run in true
    parallel — one active simulation per CPU core.
    Falls back transparently to the pandas loop when Numba is unavailable.
    """

    def _simulate(self, sig_df: pd.DataFrame, symbol: str, strategy_name: str):
        fast = fi.simulate_trades(
            closes    = sig_df["close"].ffill().to_numpy(dtype=np.float64),
            signals   = sig_df["buy_signal"].fillna(False).to_numpy(dtype=np.bool_),
            dates_ord = sig_df["date"].apply(lambda d: d.toordinal()).to_numpy(dtype=np.int32),
            sl_pct    = self.cfg.STOP_LOSS_PCT,
            tp_pct    = self.cfg.TARGET_PCT,
            max_hold  = self.cfg.MAX_HOLD_DAYS,
            pos_size  = float(self.cfg.CAPITAL) * self.cfg.POSITION_SIZE_PCT / 100,
        )
        if fast is None:
            # Numba not available — use the standard pandas loop
            return super()._simulate(sig_df, symbol, strategy_name)

        if len(fast["entry_idx"]) == 0:
            return [], float(self.cfg.CAPITAL)

        dates       = sig_df["date"].tolist()
        pos_size    = float(self.cfg.CAPITAL) * self.cfg.POSITION_SIZE_PCT / 100
        running_cap = float(self.cfg.CAPITAL)
        trades      = []

        for k in range(len(fast["entry_idx"])):
            ep    = fast["entry_prices"][k]
            xp    = fast["exit_prices"][k]
            hd    = int(fast["hold_days"][k])
            pct   = float(fast["pnl_pcts"][k])
            gross = float(fast["gross_pnls"][k])
            reason = fast["exit_codes"][k]
            ei    = int(fast["entry_idx"][k])
            xi    = int(fast["exit_idx"][k])

            shares = int(pos_size / ep) if ep > 0 else 0
            if shares == 0:
                continue  # can't afford even 1 share at this price
            running_cap += gross

            trades.append(TradeRecord(
                symbol      = symbol,
                strategy    = strategy_name,
                entry_date  = dates[ei].strftime("%Y-%m-%d"),
                exit_date   = dates[xi].strftime("%Y-%m-%d"),
                entry_price = round(ep, 2),
                exit_price  = round(xp, 2),
                shares      = shares,
                pnl_pct     = round(pct, 2),
                gross_pnl   = round(gross, 2),
                hold_days   = hd,
                exit_reason = reason,
            ))

        return trades, running_cap


# ── Columns to embed per strategy for signal-validation view ──────────────────

_SIGNAL_COLS: dict[str, list[str]] = {
    "Delivery + OI": [
        "buy_signal", "conditions_met",
        "cond_delivery_up", "cond_price_above_vwap",
        "cond_oi_increasing", "cond_support_or_break",
        "near_support", "is_breakout",
    ],
    "Volume + EMA Cross": [
        "buy_signal", "conditions_met",
        "cond_ema_cross", "cond_vol_surge", "cond_ema_confirm", "cond_bullish",
        "ema_fast", "ema_slow", "vol_avg",
    ],
    "EMA + Bollinger Bands": [
        "buy_signal", "conditions_met",
        "cond_ema_aligned", "cond_bb_pullback",
        "ema_fast", "ema_slow", "bb_upper", "bb_mid", "bb_lower",
    ],
}

_BOOL_COLS = {"buy_signal", "near_support", "is_breakout"}


# ── Stage 2 worker ────────────────────────────────────────────────────────────

def _run_one(
    symbol:       str,
    meta:         dict,
    df:           pd.DataFrame,
    strategy_cls,
    engine:       FastBacktester,
) -> tuple[BacktestResult | None, dict, pd.DataFrame | None]:
    _t0 = time.perf_counter() if _PROFILE else 0.0
    try:
        strategy = strategy_cls()
        result, sig_df = engine.run_with_signals(strategy, df)
        if _PROFILE:
            with _prof_lock:
                _prof_tasks.append((symbol, strategy_cls.params.name, time.perf_counter() - _t0))
        return result, meta, sig_df
    except Exception:
        print(f"\n  [ERROR] {symbol} / {strategy_cls.params.name}:")
        traceback.print_exc()
        if _PROFILE:
            with _prof_lock:
                _prof_tasks.append((symbol, strategy_cls.params.name, time.perf_counter() - _t0))
        return None, meta, None


def run_all_strategies(
    stock_data:     dict[str, pd.DataFrame],
    stock_meta:     dict[str, dict],
    strategy_names: list[str] | None,
) -> list[tuple[BacktestResult, dict, pd.DataFrame | None]]:
    strategy_classes = (
        [get_strategy(n) for n in strategy_names]
        if strategy_names else all_strategies()
    )

    tasks = [
        (sym, stock_meta.get(sym, {}), df, cls)
        for sym, df in stock_data.items()
        if not df.empty
        for cls in strategy_classes
    ]

    if not tasks:
        print("  [!] No tasks to run — all fetched DataFrames are empty.")
        return []

    print(f"[Stage 2/2] Running {len(tasks)} tasks  "
          f"({STRAT_WORKERS} workers)…")
    t0     = time.perf_counter()
    engine = FastBacktester(cfg)
    pairs: list[tuple[BacktestResult, dict, pd.DataFrame | None]] = []
    done  = 0

    with ThreadPoolExecutor(max_workers=STRAT_WORKERS) as pool:
        futures = [
            pool.submit(_run_one, sym, meta, df, cls, engine)
            for sym, meta, df, cls in tasks
        ]
        for fut in as_completed(futures):
            result, meta, sig_df = fut.result()
            done += 1
            if result is not None:
                pairs.append((result, meta, sig_df))
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done:>5}/{len(tasks)} …", end="\r")

    elapsed = time.perf_counter() - t0
    ok      = len(pairs)
    zeros   = sum(1 for r, *_ in pairs if r.total_trades == 0)
    print(f"\n  Strategy runs done in {elapsed:.1f}s")
    print(f"  Results: {ok} collected  |  {zeros} with 0 trades  |  "
          f"{ok - zeros} with >=1 trade\n")
    return pairs


# ── HTML Report ───────────────────────────────────────────────────────────────

def build_report(
    pairs:          list[tuple[BacktestResult, dict, "pd.DataFrame | None"]],
    strategy_names: list[str],
    n_enabled:      int,
    stock_data:     dict | None = None,
) -> str:
    by_strat: dict[str, list[dict]] = {n: [] for n in strategy_names}

    for res, meta, sig_df in pairs:
        sn = res.strategy_name
        if sn not in by_strat:
            by_strat[sn] = []
        row = res.to_dict()
        row["company_name"] = meta.get("company_name", "")
        row["sector"]       = meta.get("sector", "")
        row["sub_sector"]   = meta.get("sub_sector", "")
        row["trades"]       = res.trades      # list[dict] of individual trade records
        sym = res.symbol
        if stock_data and sym in stock_data and not stock_data[sym].empty:
            df_s = stock_data[sym]
            def _safe_float(val, prec=2):
                try:
                    v = float(val)
                    return round(v, prec) if not (v != v) else None  # NaN → None
                except (TypeError, ValueError):
                    return None
            def _safe_int(val):
                try:
                    v = int(val)
                    return v
                except (TypeError, ValueError):
                    return None
            ohlcv_rows = []
            for _, r in df_s.iterrows():
                if not pd.notna(r.get("close")):
                    continue
                ohlcv_rows.append({
                    "time":         str(r["date"])[:10],
                    "open":         _safe_float(r.get("open",  r["close"])),
                    "high":         _safe_float(r.get("high",  r["close"])),
                    "low":          _safe_float(r.get("low",   r["close"])),
                    "close":        _safe_float(r["close"]),
                    "volume":       _safe_int(r.get("total_volume", r.get("volume", 0))),
                    "delivery_pct": _safe_float(r.get("delivery_pct"), 2),
                    "delivery_qty": _safe_int(r.get("delivery_qty")),
                    "vwap":         _safe_float(r.get("vwap")),
                    "oi":           _safe_int(r.get("oi")),
                    "oi_change":    _safe_int(r.get("oi_change")),
                })
            row["ohlcv"] = ohlcv_rows
        else:
            row["ohlcv"] = []

        # Embed per-day indicator/condition values for signal-validation view
        sig_cols = _SIGNAL_COLS.get(sn, [])
        if sig_df is not None and not sig_df.empty and sig_cols:
            signal_rows = []
            for _, sr in sig_df.iterrows():
                srow: dict = {"time": str(sr["date"])[:10]}
                for col in sig_cols:
                    if col not in sr.index:
                        continue
                    val = sr[col]
                    if not pd.notna(val):
                        srow[col] = None
                    elif col.startswith("cond_") or col in _BOOL_COLS:
                        srow[col] = bool(val)
                    elif col == "conditions_met":
                        srow[col] = int(val)
                    else:
                        srow[col] = round(float(val), 2)
                signal_rows.append(srow)
            row["signal_data"] = signal_rows
        else:
            row["signal_data"] = []

        by_strat[sn].append(row)

    for sn in by_strat:
        by_strat[sn].sort(key=lambda r: r.get("total_pnl_pct", 0), reverse=True)

    total_with_trades = sum(
        sum(1 for r in rows if r.get("total_trades", 0) > 0)
        for rows in by_strat.values()
    )

    payload = json.dumps({
        "run_date":           datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_enabled":          n_enabled,
        "backtest_days":      cfg.BACKTEST_DAYS,
        "capital":            cfg.CAPITAL,
        "total_with_trades":  total_with_trades,
        "strategies": [
            {"name": sn, "stocks": by_strat.get(sn, [])}
            for sn in strategy_names
        ],
    }, default=str)

    html = _REPORT_TEMPLATE.replace("__PAYLOAD__", payload)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    return OUTPUT_HTML


# ── HTML template ─────────────────────────────────────────────────────────────

_REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Strategy Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:#ffffff; --surface:#f8f9ff; --border:#c8d0e0;
  --text:#0f1623; --muted:#3d4a62; --accent:#1d4ed8;
  --green:#15803d; --red:#b91c1c; --r:6px;
}
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
       background:var(--bg); color:var(--text); font-size:14px; line-height:1.55; }
header { display:flex; align-items:center; gap:12px; flex-wrap:wrap;
         padding:11px 24px; background:#fff; border-bottom:2px solid var(--border);
         position:sticky; top:0; z-index:100; box-shadow:0 1px 4px rgba(0,0,0,.06); }
header h1 { font-size:15px; font-weight:800; color:var(--accent); letter-spacing:.05em; }
.badge { background:var(--surface); border:1.5px solid var(--border); border-radius:20px;
         padding:3px 11px; font-size:12px; font-weight:600; color:var(--muted); }
.back { margin-left:auto; font-size:12px; font-weight:600; color:var(--accent);
        text-decoration:none; border:1.5px solid var(--border); border-radius:var(--r);
        padding:5px 13px; transition:border-color .15s; }
.back:hover { border-color:var(--accent); }
#strategy-bar { display:flex; align-items:center; gap:10px;
                padding:10px 24px; background:#fff; border-bottom:1px solid var(--border); }
#strategy-bar label { font-size:11px; font-weight:700; color:var(--muted);
                      text-transform:uppercase; letter-spacing:.06em; }
#strat-sel { padding:6px 12px; border:1.5px solid var(--border); border-radius:var(--r);
             font-size:13px; font-weight:600; background:var(--surface); color:var(--text);
             cursor:pointer; outline:none; min-width:240px; }
#strat-sel:focus { border-color:var(--accent); }

/* ── Run bar ────────────────────────────────────────────────────────────── */
#run-bar { display:flex; align-items:center; gap:12px; flex-wrap:wrap;
           padding:9px 24px; background:#f0f4ff; border-bottom:1px solid var(--border); }
#run-bar label { font-size:11px; font-weight:800; color:var(--muted);
                 text-transform:uppercase; letter-spacing:.06em; }
#cfg-capital { padding:5px 10px; border:1.5px solid var(--border); border-radius:var(--r);
               font-size:13px; font-weight:700; width:130px; outline:none;
               background:#fff; color:var(--text); }
#cfg-capital:focus { border-color:var(--accent); }
#btn-rerun { padding:6px 16px; background:var(--accent); color:#fff; border:none;
             border-radius:var(--r); font-size:13px; font-weight:700; cursor:pointer;
             display:flex; align-items:center; gap:6px; transition:opacity .15s; }
#btn-rerun:hover { opacity:.88; }
#btn-rerun:disabled { opacity:.45; cursor:not-allowed; }
#run-status { font-size:12px; font-weight:600; display:flex; align-items:center; gap:6px; }
.rs-idle    { color:var(--muted); }
.rs-running { color:#d97706; }
.rs-done    { color:#15803d; }
.rs-error   { color:#b91c1c; }
.spin { display:inline-block; animation:spin .8s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
#run-log-wrap { display:none; width:100%; margin-top:4px; }
#run-log-wrap.open { display:block; }
#run-log { background:#1e2433; color:#a8b4cc; font-family:monospace; font-size:11px;
           padding:8px 12px; border-radius:var(--r); max-height:160px; overflow-y:auto;
           white-space:pre; line-height:1.5; }
#run-log-toggle { font-size:11px; font-weight:600; color:var(--accent);
                  cursor:pointer; text-decoration:underline; margin-left:4px; }
.run-bar-sep { width:1px; height:22px; background:var(--border); flex-shrink:0; }

/* ── Status toast (fixed bottom-right) ──────────────────────────────────── */
#status-toast { position:fixed; bottom:24px; right:24px; z-index:999;
                min-width:260px; max-width:380px; border-radius:10px;
                box-shadow:0 4px 24px rgba(0,0,0,.18); overflow:hidden;
                font-size:13px; font-weight:600; pointer-events:none;
                opacity:0; transform:translateY(12px);
                transition:opacity .25s, transform .25s; }
#status-toast.visible { opacity:1; transform:translateY(0); pointer-events:auto; }
#toast-header { display:flex; align-items:center; gap:10px; padding:12px 16px 10px; }
#toast-icon { font-size:18px; line-height:1; flex-shrink:0; }
#toast-title { flex:1; font-size:13px; font-weight:800; }
#toast-close { background:none; border:none; cursor:pointer; font-size:16px;
               opacity:.6; padding:0 2px; line-height:1; color:inherit; }
#toast-close:hover { opacity:1; }
#toast-body { padding:0 16px 12px; font-size:12px; font-weight:500; opacity:.85; }
#toast-progress { height:3px; width:100%; transition:width .4s linear; }
/* state colours */
#status-toast.st-running { background:#fffbeb; color:#92400e; border:1.5px solid #fcd34d; }
#status-toast.st-running #toast-progress { background:#f59e0b; }
#status-toast.st-done    { background:#f0fdf4; color:#14532d; border:1.5px solid #86efac; }
#status-toast.st-done    #toast-progress { background:#22c55e; width:100% !important; }
#status-toast.st-error   { background:#fff1f2; color:#881337; border:1.5px solid #fda4af; }
#status-toast.st-error   #toast-progress { background:#f43f5e; width:100% !important; }

/* ── Indicator chip bar ─────────────────────────────────────────────────── */
#chip-bar { display:flex; align-items:center; gap:8px; flex-wrap:wrap;
            padding:9px 24px; background:#fff; border-bottom:2px solid var(--border);
            min-height:46px; }
.chip-bar-lbl { font-size:10px; font-weight:800; color:var(--muted);
                text-transform:uppercase; letter-spacing:.07em; flex-shrink:0;
                margin-right:4px; }
.chip { display:inline-flex; align-items:center; gap:5px; padding:4px 10px 4px 10px;
        border-radius:20px; font-size:12px; font-weight:700; color:#fff;
        white-space:nowrap; cursor:default; }
.chip-cond { font-size:10px; font-weight:700; opacity:.85;
             background:rgba(0,0,0,.18); border-radius:3px; padding:0 5px; }

/* ── Panel & tabs ───────────────────────────────────────────────────────── */
.panel { display:none; }
.panel.active { display:block; }
.tab-bar { display:flex; border-bottom:2px solid var(--border);
           background:#fff; padding:0 24px; position:sticky;
           top:86px; z-index:90; }
.tab-btn { padding:10px 22px; font-size:13px; font-weight:700; background:none;
           border:none; border-bottom:3px solid transparent; margin-bottom:-2px;
           cursor:pointer; color:var(--muted); transition:color .15s,border-color .15s; }
.tab-btn:hover { color:var(--text); }
.tab-btn.active { color:var(--accent); border-bottom-color:var(--accent); }
.tab-content { display:none; padding:24px; max-width:1440px; margin:0 auto; }
.tab-content.active { display:block; }

/* ── Trade detail (overview tab) ────────────────────────────────────────── */
.trade-chart-wrap { margin-bottom:12px; }
.eq-stats-bar { display:flex; gap:16px; flex-wrap:wrap; padding:5px 10px 4px;
                font-size:11px; font-weight:700; background:var(--surface);
                border:1px solid var(--border); border-radius:var(--r) var(--r) 0 0;
                border-bottom:none; }
.eqs { display:flex; align-items:center; gap:4px; }
.eqs.peak  { color:#0891b2; } .eqs.final { color:#1d4ed8; }
.eqs.dd    { color:#b91c1c; } .eqs.dur   { color:#7c3aed; }
.eqs .dot  { width:8px; height:8px; border-radius:50%; display:inline-block; flex-shrink:0; }
.eqs.peak  .dot { background:#0891b2; }
.eqs.final .dot { background:#1d4ed8; }
.eqs.dd    .dot { background:#b91c1c; }
.eqs.dur   .dot { background:#7c3aed; border-radius:1px; height:3px; width:12px; }
.eq-wrap { height:160px; border:1px solid var(--border);
           border-radius:0 0 var(--r) var(--r); overflow:hidden;
           background:var(--surface); margin-bottom:8px; }
.lc-wrap { height:280px; border:1px solid var(--border);
           border-radius:var(--r); overflow:hidden; }
.detail-inner { padding:12px 20px 16px; }

/* ── Summary cards & bar chart ──────────────────────────────────────────── */
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(155px,1fr));
         gap:12px; margin-bottom:24px; }
.card { background:var(--surface); border:1.5px solid var(--border);
        border-radius:var(--r); padding:14px 18px; }
.cl { font-size:11px; font-weight:700; color:var(--muted); text-transform:uppercase;
      letter-spacing:.06em; margin-bottom:5px; }
.cv { font-size:22px; font-weight:800; }
/* ── Overview: side-by-side table + chart ───────────────────────────────── */
.overview-body { display:flex; gap:16px; align-items:flex-start; }
.overview-left { flex:1; min-width:0; }
.overview-right { width:340px; flex-shrink:0;
                  position:sticky; top:132px;
                  height:calc(100vh - 160px);
                  background:var(--surface); border:1.5px solid var(--border);
                  border-radius:var(--r);
                  display:flex; flex-direction:column; overflow:hidden; }
.chart-right-hdr { padding:10px 14px 8px; font-size:11px; font-weight:800;
                   color:var(--muted); text-transform:uppercase; letter-spacing:.06em;
                   border-bottom:1px solid var(--border); flex-shrink:0; }
.chart-scroll-wrap { flex:1; overflow-y:auto; overflow-x:hidden; padding:8px 4px 8px 8px; }
.chart-scroll-wrap canvas { display:block; }
.chart-drag-handle { height:10px; cursor:ns-resize; background:var(--surface);
                     border:1.5px solid var(--border); border-top:none;
                     border-radius:0 0 var(--r) var(--r); margin-bottom:12px;
                     display:flex; align-items:center; justify-content:center; }
.chart-drag-handle::after { content:''; width:32px; height:3px;
                             background:var(--border); border-radius:2px; }
.chart-drag-handle:hover { background:var(--border); }

/* ── Stock table ────────────────────────────────────────────────────────── */
.tbl-wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { background:var(--surface); font-weight:700; text-align:left;
     padding:8px 10px; border-bottom:2px solid var(--border);
     font-size:11px; text-transform:uppercase; letter-spacing:.05em;
     color:var(--muted); cursor:pointer; white-space:nowrap; user-select:none; }
th:hover { color:var(--accent); }
th.sort-asc::after  { content:" ▲"; color:var(--accent); }
th.sort-desc::after { content:" ▼"; color:var(--accent); }
td { padding:7px 10px; border-bottom:1px solid #eef0f6; vertical-align:middle; }
tr:hover td { background:#f5f7ff; }
tr.profit td:first-child { border-left:3px solid #86efac; }
tr.loss   td:first-child { border-left:3px solid #fca5a5; }
tr.no-trade td { color:var(--muted); }
.sym { font-weight:800; font-family:monospace; font-size:13px; }
.pos { color:var(--green); font-weight:700; }
.neg { color:var(--red);   font-weight:700; }
.tbl-controls { display:flex; gap:10px; align-items:center; margin-bottom:12px; }
.tbl-controls input[type=text] { padding:6px 10px; border:1.5px solid var(--border);
  border-radius:var(--r); font-size:13px; width:240px; outline:none; }
.tbl-controls input[type=text]:focus { border-color:var(--accent); }
.tbl-controls label { font-size:12px; font-weight:600; color:var(--muted);
                       display:flex; align-items:center; gap:5px; cursor:pointer; }
.empty-note { padding:32px; text-align:center; color:var(--muted); font-size:13px; }
.tog { cursor:pointer; font-size:10px; color:var(--accent); padding:1px 5px;
       border:1px solid var(--border); border-radius:3px; margin-left:5px; user-select:none; }
.tog:hover { background:var(--accent); color:#fff; border-color:var(--accent); }
tr.detail-row td { padding:0; background:#f0f4ff; border-bottom:2px solid var(--border); }
.tl-wrap { padding:10px 20px 14px; }
.tl { width:100%; border-collapse:collapse; font-size:12px; }
.tl th { background:#e8ecf8; font-size:10px; font-weight:700; padding:5px 8px;
         border-bottom:1.5px solid var(--border); color:var(--muted); text-transform:uppercase;
         white-space:nowrap; }
.tl td { padding:5px 8px; border-bottom:1px solid #dde3f0; white-space:nowrap; }
.tl-trade-hdr td { background:#dde3f5; font-size:11px; font-weight:700; padding:7px 8px; }
.tl-tn { background:var(--accent); color:#fff; border-radius:3px;
         padding:1px 7px; margin-right:5px; font-size:10px; font-weight:800; }
tr.tl-buy  td { background:#e6f4ea; }
tr.tl-sell td { background:#fde8e8; }
tr.tl-ctx  td { background:#f8f9ff; color:var(--muted); }
tr.tl-ctx  td:first-child,
tr.tl-ctx  td:nth-child(2) { font-weight:700; color:var(--text); }
tr.tl-sep  td { height:8px; background:transparent !important;
                border:none !important; padding:0 !important; }

/* ── Signal Validation tab ──────────────────────────────────────────────── */
.sig-controls { display:flex; align-items:center; gap:12px; flex-wrap:wrap;
                margin-bottom:16px; }
.sig-controls label { font-size:11px; font-weight:700; color:var(--muted);
                      text-transform:uppercase; letter-spacing:.06em; }
.sig-controls select { padding:6px 12px; border:1.5px solid var(--border);
                       border-radius:var(--r); font-size:13px; font-weight:600;
                       background:var(--surface); color:var(--text); cursor:pointer;
                       min-width:220px; outline:none; }
.sig-controls select:focus { border-color:var(--accent); }
.sig-chart-wrap { height:340px; border:1px solid var(--border);
                  border-radius:var(--r); overflow:hidden; margin-bottom:10px; }
.sig-legend { display:flex; gap:14px; flex-wrap:wrap; padding:4px 2px 12px;
              align-items:center; }
.sig-leg-item { display:flex; align-items:center; gap:5px;
                font-size:11px; font-weight:600; color:var(--muted); }
.sig-leg-line { width:18px; height:2px; border-radius:2px; flex-shrink:0; }
.sig-tbl-header { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
.sig-tbl-header span { font-size:12px; font-weight:700; color:var(--muted); }
.sig-tbl-header label { font-size:12px; font-weight:600; color:var(--muted);
                         display:flex; align-items:center; gap:5px; cursor:pointer; }
.cond-tbl-wrap { overflow-x:auto; max-height:420px; overflow-y:auto;
                 border:1px solid var(--border); border-radius:var(--r); }
.cond-tbl { border-collapse:collapse; font-size:12px; min-width:100%; }
.cond-tbl thead th { position:sticky; top:0; z-index:1; }
.cond-tbl th { padding:6px 10px; background:#edf0f8; font-size:10px; font-weight:800;
               text-transform:uppercase; color:var(--muted); border-bottom:2px solid var(--border);
               white-space:nowrap; letter-spacing:.04em; }
.cond-tbl td { padding:5px 10px; border-bottom:1px solid #eef0f6; white-space:nowrap; }
tr.cond-buy td { background:#dcfce7; }
tr.cond-buy td:first-child { border-left:3px solid #16a34a; font-weight:700; }
.c-yes { color:#15803d; font-weight:800; font-size:15px; line-height:1; }
.c-no  { color:#d1d5db; font-size:15px; line-height:1; }
.c-sig { display:inline-block; border-radius:3px; padding:1px 8px;
         font-size:10px; font-weight:800; letter-spacing:.03em; color:#fff; }
.c-sig-buy       { background:#1d4ed8; }
.c-sig-sell-win  { background:#15803d; }
.c-sig-sell-loss { background:#b91c1c; }
.c-num { font-family:monospace; }
tr.cond-sell td          { background:#fff1f2; }
tr.cond-sell td:first-child { border-left:3px solid #b91c1c; font-weight:700; }
tr.cond-buy-sell td         { background:#fef9c3; }
tr.cond-buy-sell td:first-child { border-left:3px solid #ca8a04; font-weight:700; }
</style>
</head>
<body>
<header>
  <h1>Strategy Comparison</h1>
  <span class="badge" id="h-stocks"></span>
  <span class="badge" id="h-period"></span>
  <span class="badge" id="h-date"></span>
  <a class="back" href="dashboard.html">&#8592; Dashboard</a>
</header>
<div id="strategy-bar">
  <label for="strat-sel">Strategy</label>
  <select id="strat-sel"></select>
</div>
<div id="run-bar">
  <label for="cfg-capital">Capital ₹</label>
  <input type="number" id="cfg-capital" step="50000" min="10000">
  <div class="run-bar-sep"></div>
  <button id="btn-rerun" onclick="triggerRun()">&#x21BB; Re-run Backtest</button>
  <span id="run-status" class="rs-idle">—</span>
  <span id="run-log-toggle" style="display:none" onclick="toggleLog()">show log</span>
  <div id="run-log-wrap"><pre id="run-log"></pre></div>
</div>
<div id="chip-bar"></div>
<div id="panels"></div>

<!-- Fixed status toast -->
<div id="status-toast">
  <div id="toast-header">
    <span id="toast-icon"></span>
    <span id="toast-title"></span>
    <button id="toast-close" onclick="hideToast()">✕</button>
  </div>
  <div id="toast-body"></div>
  <div id="toast-progress" style="width:0%"></div>
</div>

<script>
const DATA = __PAYLOAD__;

document.getElementById('h-stocks').textContent =
  DATA.n_enabled + ' stocks  |  ' + DATA.total_with_trades + ' results with trades';
document.getElementById('h-period').textContent =
  DATA.backtest_days + 'd  ₹' + Number(DATA.capital).toLocaleString('en-IN');
document.getElementById('h-date').textContent = DATA.run_date;

// ── Strategy metadata: chips and per-condition definitions ────────────────────
const STRATEGY_META = {
  "Delivery + OI": {
    chips: [
      { label:"Delivery Qty",   bg:"#0891b2", cond:"C1", desc:"Delivery qty > prev day" },
      { label:"Delivery %",     bg:"#06b6d4", cond:"C1", desc:"Delivery % > prev day" },
      { label:"VWAP",           bg:"#7c3aed", cond:"C2", desc:"Close > VWAP" },
      { label:"Futures OI",     bg:"#d97706", cond:"C3", desc:"OI positive & growing" },
      { label:"Support (30d)",  bg:"#16a34a", cond:"C4", desc:"Price within 2% of support" },
      { label:"Breakout (20d)", bg:"#15803d", cond:"C4", desc:"Price ≥ 20-bar high" },
    ],
    conditions: [
      { key:"cond_delivery_up",      label:"Delivery Up",  bg:"#0891b2" },
      { key:"cond_price_above_vwap", label:"Price > VWAP", bg:"#7c3aed" },
      { key:"cond_oi_increasing",    label:"OI ↑",         bg:"#d97706" },
      { key:"cond_support_or_break", label:"Sup/Break",    bg:"#16a34a" },
    ],
    overlays: [
      { key:"vwap", label:"VWAP", color:"#7c3aed", lineWidth:1, lineStyle:0, src:"ohlcv" },
    ],
    numericCols: [],
  },
  "Volume + EMA Cross": {
    chips: [
      { label:"EMA(9)",          bg:"#1d4ed8", cond:"C1", desc:"Fast EMA" },
      { label:"EMA(21)",         bg:"#7c3aed", cond:"C1", desc:"Slow EMA" },
      { label:"Golden Cross",    bg:"#f59e0b", cond:"C1", desc:"EMA(9) crosses above EMA(21)" },
      { label:"Vol MA(20)",      bg:"#d97706", cond:"C2", desc:"20-day volume moving average" },
      { label:"Vol Surge ×1.5",  bg:"#ea580c", cond:"C2", desc:"Volume > 1.5× avg" },
      { label:"Price > EMA(21)", bg:"#16a34a", cond:"C3", desc:"Uptrend confirmation" },
      { label:"Bullish Candle",  bg:"#0891b2", cond:"C4", desc:"Close > Open" },
    ],
    conditions: [
      { key:"cond_ema_cross",   label:"EMA Cross",   bg:"#f59e0b" },
      { key:"cond_vol_surge",   label:"Vol Surge",   bg:"#ea580c" },
      { key:"cond_ema_confirm", label:"EMA Confirm", bg:"#16a34a" },
      { key:"cond_bullish",     label:"Bullish",     bg:"#0891b2" },
    ],
    overlays: [
      { key:"ema_fast", label:"EMA(9)",  color:"#1d4ed8", lineWidth:1.5, lineStyle:0, src:"signal_data" },
      { key:"ema_slow", label:"EMA(21)", color:"#7c3aed", lineWidth:1.5, lineStyle:0, src:"signal_data" },
    ],
    numericCols: ["ema_fast","ema_slow","vol_avg"],
  },
  "EMA + Bollinger Bands": {
    chips: [
      { label:"EMA(10)",       bg:"#1d4ed8", cond:"C1", desc:"Fast EMA" },
      { label:"EMA(30)",       bg:"#7c3aed", cond:"C1", desc:"Slow EMA" },
      { label:"Align (7 bars)",bg:"#f59e0b", cond:"C1", desc:"EMA(10) > EMA(30) for 7 bars" },
      { label:"BB (15, 1.5σ)", bg:"#d97706", cond:"C2", desc:"Bollinger Bands 15-period 1.5σ" },
      { label:"BB Pullback",   bg:"#0891b2", cond:"C2", desc:"Close ≤ lower Bollinger Band" },
    ],
    conditions: [
      { key:"cond_ema_aligned", label:"EMA Aligned", bg:"#f59e0b" },
      { key:"cond_bb_pullback", label:"BB Pullback",  bg:"#0891b2" },
    ],
    overlays: [
      { key:"ema_fast", label:"EMA(10)",  color:"#1d4ed8",          lineWidth:1.5, lineStyle:0, src:"signal_data" },
      { key:"ema_slow", label:"EMA(30)",  color:"#7c3aed",          lineWidth:1.5, lineStyle:0, src:"signal_data" },
      { key:"bb_upper", label:"BB Upper", color:"rgba(217,119,6,.7)",lineWidth:1,   lineStyle:2, src:"signal_data" },
      { key:"bb_mid",   label:"BB Mid",   color:"rgba(217,119,6,.5)",lineWidth:1,   lineStyle:2, src:"signal_data" },
      { key:"bb_lower", label:"BB Lower", color:"rgba(217,119,6,.7)",lineWidth:1,   lineStyle:2, src:"signal_data" },
    ],
    numericCols: ["ema_fast","ema_slow","bb_upper","bb_mid","bb_lower"],
  },
};

// ── Chip bar ──────────────────────────────────────────────────────────────────
function updateChips(stratName) {
  const bar  = document.getElementById('chip-bar');
  if (!bar) return;
  const meta = STRATEGY_META[stratName];
  if (!meta || !meta.chips.length) { bar.innerHTML = ''; return; }
  bar.innerHTML =
    '<span class="chip-bar-lbl">Indicators</span>' +
    meta.chips.map(c =>
      '<span class="chip" style="background:' + c.bg + '" title="' + c.desc + '">' +
        c.label +
        '<span class="chip-cond">' + c.cond + '</span>' +
      '</span>'
    ).join('');
}

// ── Panel construction ────────────────────────────────────────────────────────
const selEl    = document.getElementById('strat-sel');
const panelsEl = document.getElementById('panels');

const _charts    = {};
const _tableData = {};
const _sortState = {};
const _sigCharts = {};
let   _resizing  = null;
function startChartResize(e, si) {
  e.preventDefault();
  const wrap = document.getElementById('chart-wrap-' + si);
  if (!wrap) return;
  _resizing = { wrap, startY: e.clientY, startH: wrap.offsetHeight };
  document.addEventListener('mousemove', _onChartResize);
  document.addEventListener('mouseup',   _stopChartResize);
}
function _onChartResize(e) {
  if (!_resizing) return;
  const h = Math.max(120, _resizing.startH + e.clientY - _resizing.startY);
  _resizing.wrap.style.height = h + 'px';
}
function _stopChartResize() {
  _resizing = null;
  document.removeEventListener('mousemove', _onChartResize);
  document.removeEventListener('mouseup',   _stopChartResize);
}
const _SORT_KEYS = ['symbol','company_name','sector','total_trades','win_rate',
                    'total_pnl','total_pnl_pct','max_drawdown_pct','final_capital'];

DATA.strategies.forEach((strat, si) => {
  const opt = document.createElement('option');
  opt.value = si;
  opt.textContent = strat.name + ' (' + strat.stocks.length + ')';
  selEl.appendChild(opt);

  const panel = document.createElement('div');
  panel.className = 'panel' + (si === 0 ? ' active' : '');
  panel.id = 'panel-' + si;
  panel.innerHTML = buildPanel(strat, si);
  panelsEl.appendChild(panel);
  wireTable(si, strat.stocks);
  if (si === 0) {
    try { buildChart(strat, si); } catch(_) {}
  }
});

updateChips(DATA.strategies[0] ? DATA.strategies[0].name : '');
selEl.addEventListener('change', () => activateStrategy(+selEl.value));

// ── Run-bar: capital config + re-run ──────────────────────────────────────────
const SERVER_MODE = window.location.protocol !== 'file:';
const API         = window.location.origin;
let   _pollTimer  = null;
let   _toastTimer = null;

// Pre-fill capital from embedded data
document.getElementById('cfg-capital').value = DATA.capital || 500000;

// ── Toast helpers ─────────────────────────────────────────────────────────────
function showToast(state, icon, title, body, progressPct) {
  const toast = document.getElementById('status-toast');
  toast.className = 'visible st-' + state;
  document.getElementById('toast-icon').textContent  = icon;
  document.getElementById('toast-title').textContent = title;
  document.getElementById('toast-body').textContent  = body;
  document.getElementById('toast-progress').style.width = progressPct + '%';
  clearTimeout(_toastTimer);
}

function hideToast() {
  const toast = document.getElementById('status-toast');
  toast.classList.remove('visible');
}

function autoHideToast(ms) {
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(hideToast, ms);
}

// ── Run-bar inline status (small, always visible in the bar) ──────────────────
function setBarStatus(cls, html, showToggle) {
  const el = document.getElementById('run-status');
  const tg = document.getElementById('run-log-toggle');
  el.className = cls;
  el.innerHTML = html;
  tg.style.display = showToggle ? 'inline' : 'none';
}

function toggleLog() {
  const wrap = document.getElementById('run-log-wrap');
  const tog  = document.getElementById('run-log-toggle');
  const open = wrap.classList.toggle('open');
  tog.textContent = open ? 'hide log' : 'show log';
}

// ── Trigger run ───────────────────────────────────────────────────────────────
async function triggerRun() {
  if (!SERVER_MODE) {
    alert('Open the dashboard via the server to use Re-run.\n\nRun:  python scripts/server.py');
    return;
  }
  const capital = parseInt(document.getElementById('cfg-capital').value, 10);
  if (!capital || capital < 10000) { alert('Enter a valid capital (≥ ₹10,000).'); return; }

  document.getElementById('btn-rerun').disabled = true;
  setBarStatus('rs-running', '<span class="spin">⟳</span> Saving…', false);
  showToast('running', '⟳', 'Saving config…', '₹' + capital.toLocaleString('en-IN'), 5);

  try {
    const cfgRes = await fetch(API + '/api/config', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ capital }),
    });
    if (!cfgRes.ok) throw new Error(await cfgRes.text());

    setBarStatus('rs-running', '<span class="spin">⟳</span> Starting…', false);
    showToast('running', '⟳', 'Starting backtest…', 'Fetching data for all stocks…', 8);

    const runRes = await fetch(API + '/api/run', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({}),
    });
    if (!runRes.ok) {
      const err = await runRes.json();
      throw new Error(err.error || 'Run failed to start');
    }

    _pollTimer = setInterval(pollStatus, 1500);
  } catch (e) {
    setBarStatus('rs-error', '✗ Error', false);
    showToast('error', '✗', 'Failed to start', e.message, 100);
    document.getElementById('btn-rerun').disabled = false;
  }
}

// ── Poll status ───────────────────────────────────────────────────────────────
async function pollStatus() {
  try {
    const res  = await fetch(API + '/api/status');
    const data = await res.json();

    // Update inline log panel
    const logEl = document.getElementById('run-log');
    if (data.log.length) {
      logEl.textContent = data.log.join('\n');
      logEl.scrollTop   = logEl.scrollHeight;
    }

    if (data.status === 'running') {
      // Estimate rough progress from log line count (fetch ~60% of time, strategies ~40%)
      const pct = Math.min(90, Math.round(data.log.length / 0.8));
      const lastLine = data.log[data.log.length - 1] || '';
      const stage = lastLine.includes('Stage 2') ? 'Running strategies…'
                  : lastLine.includes('Stage 1') ? 'Fetching data…'
                  : 'Running…';
      setBarStatus('rs-running',
        '<span class="spin">⟳</span> ' + stage + ' ' + data.elapsed + 's',
        data.log.length > 0);
      showToast('running', '⟳', stage, data.elapsed + 's elapsed', pct);

    } else {
      clearInterval(_pollTimer);
      _pollTimer = null;
      document.getElementById('btn-rerun').disabled = false;

      if (data.status === 'done') {
        setBarStatus('rs-done', '✓ Done in ' + data.elapsed + 's', true);
        showToast('done', '✓', 'Backtest complete', 'Done in ' + data.elapsed + 's — reloading…', 100);
        setTimeout(() => window.location.reload(), 1500);
      } else {
        setBarStatus('rs-error', '✗ Failed', true);
        showToast('error', '✗', 'Run failed', 'Check the log below for details.', 100);
        document.getElementById('run-log-wrap').classList.add('open');
        document.getElementById('run-log-toggle').textContent = 'hide log';
      }
    }
  } catch (_) { /* network blip — keep polling */ }
}

// On load: if server is mid-run (e.g. page refresh during a run), resume polling
if (SERVER_MODE) {
  fetch(API + '/api/status').then(r => r.json()).then(d => {
    if (d.status === 'running') {
      document.getElementById('btn-rerun').disabled = true;
      setBarStatus('rs-running',
        '<span class="spin">⟳</span> Running… ' + d.elapsed + 's', false);
      showToast('running', '⟳', 'Backtest in progress', d.elapsed + 's elapsed', 30);
      _pollTimer = setInterval(pollStatus, 1500);
    }
  }).catch(() => {});
}

function activateStrategy(idx) {
  document.querySelectorAll('.panel').forEach(p =>
    p.classList.toggle('active', p.id === 'panel-' + idx));
  updateChips(DATA.strategies[idx] ? DATA.strategies[idx].name : '');
  if (_charts[idx]) { try { _charts[idx].resize(); } catch(_) {} }
  else { try { buildChart(DATA.strategies[idx], idx); } catch(_) {} }
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(si, tabName) {
  const panel = document.getElementById('panel-' + si);
  if (!panel) return;
  panel.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === tabName));
  panel.querySelectorAll('.tab-content').forEach(c =>
    c.classList.toggle('active', c.dataset.tab === tabName));
  if (tabName === 'signals') {
    requestAnimationFrame(() => renderSignalView(si));
  }
}

// ── Build panel HTML ──────────────────────────────────────────────────────────
function buildPanel(strat, si) {
  return (
    '<div class="tab-bar">' +
      '<button class="tab-btn active" data-tab="overview" ' +
        'onclick="switchTab(' + si + ',\'overview\')">Overview</button>' +
      '<button class="tab-btn" data-tab="signals" ' +
        'onclick="switchTab(' + si + ',\'signals\')">Signal Validation</button>' +
    '</div>' +
    '<div class="tab-content active" data-tab="overview">' + buildOverviewHTML(strat, si) + '</div>' +
    '<div class="tab-content" data-tab="signals">' + buildSignalsHTML(strat, si) + '</div>'
  );
}

function buildOverviewHTML(strat, si) {
  const stocks = strat.stocks;
  const traded = stocks.filter(r => +r.total_trades > 0);
  const profit = traded.filter(r => +r.total_pnl_pct > 0);
  const avgRet = traded.length
    ? traded.reduce((s, r) => s + +r.total_pnl_pct, 0) / traded.length : 0;
  const best  = traded.reduce((b,r) => +r.total_pnl_pct > +(b||{total_pnl_pct:-1e9}).total_pnl_pct ? r : b, null);
  const worst = traded.reduce((b,r) => +r.total_pnl_pct < +(b||{total_pnl_pct:+1e9}).total_pnl_pct ? r : b, null);
  const cards = [
    ['Stocks',      stocks.length],
    ['With Trades', traded.length],
    ['Profitable',  '<span class="pos">' + profit.length + '</span>'],
    ['Avg Return',  '<span class="' + (avgRet>=0?'pos':'neg') + '">' + (avgRet>=0?'+':'') + avgRet.toFixed(1) + '%</span>'],
    ['Best',        best  ? '<b>' + best.symbol  + '</b> <span class="pos">+' + fmt(best.total_pnl_pct)  + '%</span>' : '—'],
    ['Worst',       worst ? '<b>' + worst.symbol + '</b> <span class="neg">'  + fmt(worst.total_pnl_pct) + '%</span>' : '—'],
  ];
  const chartH = Math.max(180, traded.length * 18 + 60);
  return (
    '<div class="cards">' +
      cards.map(([l,v]) =>
        '<div class="card"><div class="cl">' + l + '</div><div class="cv">' + v + '</div></div>'
      ).join('') +
    '</div>' +
    (traded.length
      ? '<div id="chart-wrap-' + si + '" class="chart-scroll-wrap" style="height:420px;overflow-y:auto;overflow-x:hidden;border:1.5px solid var(--border);border-radius:var(--r) var(--r) 0 0;margin-bottom:0;">' +
          '<canvas id="chart-' + si + '" height="' + chartH + '"></canvas>' +
        '</div>' +
        '<div class="chart-drag-handle" onmousedown="startChartResize(event,' + si + ')"></div>'
      : '') +
    '<div class="tbl-controls">' +
      '<input type="text" id="filter-' + si + '" placeholder="Filter symbol / company / sector…" oninput="renderTable(' + si + ')">' +
      '<label><input type="checkbox" id="hide-' + si + '" onchange="renderTable(' + si + ')"> Hide no-trade stocks</label>' +
    '</div>' +
    '<div class="tbl-wrap"><table id="tbl-' + si + '">' +
      '<thead><tr>' +
        ['Symbol','Company','Sector','Trades','Win%','Total P&L','Return%','Max DD%','Final Capital']
          .map((h, i) => '<th onclick="sortTable(' + si + ',' + i + ')">' + h + '</th>').join('') +
      '</tr></thead><tbody id="tbody-' + si + '"></tbody></table></div>'
  );
}

function buildSignalsHTML(strat, si) {
  const stocks = strat.stocks.filter(r => r.ohlcv && r.ohlcv.length > 0);
  if (!stocks.length)
    return '<div class="empty-note">No OHLCV data available — re-run the comparison to generate signal data.</div>';

  const meta = STRATEGY_META[strat.name] || {};
  const overlays = meta.overlays || [];

  const opts = stocks.map(r =>
    '<option value="' + r.symbol + '">' +
      r.symbol + (r.company_name ? '  —  ' + r.company_name : '') +
    '</option>'
  ).join('');

  const markerLegend =
    '<span class="sig-leg-item"><span style="color:#1d4ed8;font-size:13px">▲</span> BUY executed</span>' +
    '<span class="sig-leg-item"><span style="color:#15803d;font-size:13px">▼</span> SELL (profit)</span>' +
    '<span class="sig-leg-item"><span style="color:#b91c1c;font-size:13px">▼</span> SELL (loss)</span>' +
    '<span class="sig-leg-item"><span style="color:#94a3b8;font-size:13px">●</span> Signal skipped (in trade)</span>';

  const legendHtml = '<div class="sig-legend">' +
    markerLegend +
    (overlays.length
      ? overlays.map(ov =>
          '<span class="sig-leg-item"><span class="sig-leg-line" style="' +
          (ov.lineStyle === 2
            ? 'border-top:2px dashed ' + ov.color + ';background:none;height:0'
            : 'background:' + ov.color) +
          '"></span>' + ov.label + '</span>'
        ).join('')
      : '') +
    '</div>';

  return (
    '<div class="sig-controls">' +
      '<label for="sig-sel-' + si + '">Stock</label>' +
      '<select id="sig-sel-' + si + '" onchange="renderSignalView(' + si + ')">' + opts + '</select>' +
    '</div>' +
    '<div class="sig-chart-wrap" id="sig-chart-' + si + '"></div>' +
    legendHtml +
    '<div class="sig-tbl-header">' +
      '<span>Conditions per Day</span>' +
      '<label><input type="checkbox" id="sig-all-' + si + '" onchange="refreshCondTable(' + si + ')"> Show all days</label>' +
    '</div>' +
    '<div id="sig-cond-' + si + '"><div class="empty-note">Select a stock above to view signal details.</div></div>'
  );
}

// ── Signal Validation: chart ──────────────────────────────────────────────────
function renderSignalView(si) {
  requestAnimationFrame(() => {
    buildSignalChart(si);
    refreshCondTable(si);
  });
}

function buildSignalChart(si) {
  const selEl = document.getElementById('sig-sel-' + si);
  if (!selEl) return;
  const sym   = selEl.value;
  const strat = DATA.strategies[si];
  const stock = strat.stocks.find(s => s.symbol === sym);
  if (!stock) return;

  const container = document.getElementById('sig-chart-' + si);
  if (!container) return;

  // Destroy existing chart
  if (_sigCharts[si]) {
    try { _sigCharts[si].remove(); } catch(_) {}
    _sigCharts[si] = null;
  }
  container.innerHTML = '';

  const ohlcv    = (stock.ohlcv || []).filter(d => d.open != null && d.close != null);
  const sigData  = stock.signal_data || [];
  const meta     = STRATEGY_META[strat.name] || {};
  const overlays = meta.overlays || [];

  if (!ohlcv.length) {
    container.innerHTML = '<div style="padding:20px;text-align:center;color:#999;font-size:12px">No OHLCV data</div>';
    return;
  }

  if (typeof LightweightCharts === 'undefined') {
    container.innerHTML = '<div style="padding:20px;text-align:center;color:#999;font-size:12px">Chart library not loaded</div>';
    return;
  }

  const chart = LightweightCharts.createChart(container, {
    width:  container.clientWidth || 900,
    height: 338,
    layout: { background:{ color:'#f8f9ff' }, textColor:'#0f1623' },
    grid:   { vertLines:{ color:'#eef0f6' }, horzLines:{ color:'#eef0f6' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    timeScale: { borderColor:'#c8d0e0', timeVisible:true, secondsVisible:false },
    rightPriceScale: { borderColor:'#c8d0e0' },
    handleScroll: true, handleScale: true,
  });
  _sigCharts[si] = chart;

  // Candlesticks
  const candleSeries = chart.addCandlestickSeries({
    upColor:'#15803d', downColor:'#b91c1c',
    borderUpColor:'#15803d', borderDownColor:'#b91c1c',
    wickUpColor:'#15803d', wickDownColor:'#b91c1c',
  });
  candleSeries.setData(ohlcv.map(d => ({ time:d.time, open:d.open, high:d.high, low:d.low, close:d.close })));

  // Volume histogram
  const volSeries = chart.addHistogramSeries({
    priceFormat:{ type:'volume' }, priceScaleId:'', scaleMargins:{ top:0.82, bottom:0 },
  });
  volSeries.setData(ohlcv.filter(d => d.volume != null).map(d => ({
    time:d.time, value:d.volume || 0,
    color:(+d.close) >= (+d.open || +d.close) ? 'rgba(21,128,61,0.3)' : 'rgba(185,28,28,0.3)',
  })));

  // Indicator overlays (EMA, BB bands, VWAP)
  const sigMap = {};
  for (const sd of sigData) sigMap[sd.time] = sd;

  for (const ov of overlays) {
    let lineData;
    if (ov.src === 'ohlcv') {
      lineData = ohlcv.filter(d => d[ov.key] != null).map(d => ({ time:d.time, value:+d[ov.key] }));
    } else {
      lineData = sigData.filter(sd => sd[ov.key] != null).map(sd => ({ time:sd.time, value:sd[ov.key] }));
    }
    if (!lineData.length) continue;
    const series = chart.addLineSeries({
      color: ov.color, lineWidth: ov.lineWidth || 1, lineStyle: ov.lineStyle || 0,
      lastValueVisible: false, priceLineVisible: false,
    });
    series.setData(lineData);
  }

  // All signal days (from signal_data) — circle markers for every BUY signal fired
  const executedEntries = new Set((stock.trades || []).map(t => t.entry_date));
  const markers = [];
  for (const sd of sigData) {
    if (!sd.buy_signal) continue;
    if (executedEntries.has(sd.time)) continue;  // executed trade will get an arrow instead
    markers.push({
      time: sd.time, position: 'belowBar',
      color: '#94a3b8', shape: 'circle', text: '●', size: 0,
    });
  }
  // Executed trades — arrow up for entry, arrow down for exit
  for (const t of (stock.trades || [])) {
    if (t.entry_date) markers.push({
      time:t.entry_date, position:'belowBar', color:'#1d4ed8', shape:'arrowUp', text:'BUY', size:1,
    });
    if (t.exit_date) markers.push({
      time:t.exit_date, position:'aboveBar',
      color: +t.gross_pnl >= 0 ? '#15803d' : '#b91c1c',
      shape:'arrowDown', text:'SELL', size:1,
    });
  }
  markers.sort((a,b) => a.time.localeCompare(b.time));
  candleSeries.setMarkers(markers);

  chart.timeScale().fitContent();
}

// ── Signal Validation: conditions table ───────────────────────────────────────
function refreshCondTable(si) {
  const selEl = document.getElementById('sig-sel-' + si);
  if (!selEl) return;
  const sym     = selEl.value;
  const showAll = document.getElementById('sig-all-' + si)?.checked || false;
  const div     = document.getElementById('sig-cond-' + si);
  if (div) div.innerHTML = buildConditionTable(si, sym, showAll);
}

function buildConditionTable(si, sym, showAll) {
  const strat   = DATA.strategies[si];
  const stock   = strat.stocks.find(s => s.symbol === sym);
  if (!stock) return '<div class="empty-note">Stock not found.</div>';

  const meta    = STRATEGY_META[strat.name] || {};
  const conds   = meta.conditions || [];
  const numCols = meta.numericCols || [];
  const sigData = stock.signal_data || [];
  if (!sigData.length)
    return '<div class="empty-note">No signal data for this stock — re-run the comparison.</div>';

  // OHLCV lookup
  const oMap = {};
  for (const d of (stock.ohlcv || [])) oMap[d.time] = d;

  // Exit-date lookup: date → list of trades that exit on that day
  const exitMap = {};
  for (const t of (stock.trades || [])) {
    if (!t.exit_date) continue;
    (exitMap[t.exit_date] = exitMap[t.exit_date] || []).push(t);
  }

  const rows = showAll ? sigData : sigData.slice(-60);

  const numLabels = {
    ema_fast:'EMA Fast', ema_slow:'EMA Slow', vol_avg:'Vol Avg',
    bb_upper:'BB Upper', bb_mid:'BB Mid', bb_lower:'BB Lower',
  };

  const thead = '<tr>' +
    '<th>Date</th><th>Close</th><th>Signal</th><th>Met</th>' +
    conds.map(c => '<th>' + c.label + '</th>').join('') +
    numCols.map(c => '<th>' + (numLabels[c] || c) + '</th>').join('') +
    '</tr>';

  let tbody = '';
  for (const sd of rows) {
    const bar    = oMap[sd.time] || {};
    const isBuy  = !!sd.buy_signal;
    const exits  = exitMap[sd.time] || [];
    const isSell = exits.length > 0;

    let cls = '';
    if (isBuy && isSell)  cls = 'cond-buy-sell';
    else if (isBuy)       cls = 'cond-buy';
    else if (isSell)      cls = 'cond-sell';

    // Build signal badge(s)
    let badges = '';
    if (isBuy)
      badges += '<span class="c-sig c-sig-buy">BUY</span> ';
    for (const t of exits) {
      const win    = +t.gross_pnl >= 0;
      const reason = (t.exit_reason || 'EXIT').replace(/_/g, ' ');
      const pct    = t.pnl_pct != null ? ' · ' + (win ? '+' : '') + t.pnl_pct + '%' : '';
      badges += '<span class="c-sig ' + (win ? 'c-sig-sell-win' : 'c-sig-sell-loss') + '">' +
        'SELL · ' + reason + pct + '</span> ';
    }

    let tr = '<tr class="' + cls + '">';
    tr += '<td>' + sd.time + '</td>';
    tr += '<td class="c-num">' + (bar.close != null ? bar.close : '—') + '</td>';
    tr += '<td style="white-space:nowrap">' + badges + '</td>';
    tr += '<td class="c-num">' + (sd.conditions_met != null ? sd.conditions_met : '—') + '</td>';
    for (const c of conds) {
      const v = sd[c.key];
      tr += '<td class="' + (v ? 'c-yes' : 'c-no') + '">' + (v ? '✓' : '✗') + '</td>';
    }
    for (const c of numCols) {
      tr += '<td class="c-num">' + (sd[c] != null ? sd[c] : '—') + '</td>';
    }
    tr += '</tr>';
    tbody += tr;
  }

  const totalRows  = sigData.length;
  const buyDays    = sigData.filter(s => s.buy_signal).length;
  const sellDays   = Object.keys(exitMap).length;
  const shown      = rows.length;

  return (
    '<div style="margin-bottom:8px;font-size:12px;color:var(--muted)">' +
      'Showing <b>' + shown + '</b> of <b>' + totalRows + '</b> days' +
      (buyDays  ? '  ·  <b style="color:#16a34a">'  + buyDays  + ' BUY'  + (buyDays >1?'s':'') + '</b>' : '') +
      (sellDays ? '  ·  <b style="color:#b91c1c">'  + sellDays + ' SELL' + (sellDays>1?'s':'') + '</b>' : '') +
    '</div>' +
    '<div class="cond-tbl-wrap"><table class="cond-tbl"><thead>' + thead + '</thead>' +
    '<tbody>' + tbody + '</tbody></table></div>'
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmt(v, d) {
  d = d == null ? 2 : d;
  return (v == null || v === '' || isNaN(+v)) ? '—' : Number(v).toFixed(d);
}
function fmtInr(v) {
  return (v == null || isNaN(+v)) ? '—'
    : '₹' + Math.abs(+v).toLocaleString('en-IN', {maximumFractionDigits:0});
}
function signed(v, d) {
  d = d == null ? 1 : d;
  if (v == null || isNaN(+v)) return '—';
  const cls = +v >= 0 ? 'pos' : 'neg';
  return '<span class="' + cls + '">' + (+v >= 0 ? '+' : '') + Number(v).toFixed(d) + '%</span>';
}

function buildChart(strat, si) {
  const traded = strat.stocks.filter(r => +r.total_trades > 0)
    .sort((a,b) => +b.total_pnl_pct - +a.total_pnl_pct);
  if (!traded.length) return;
  const canvas = document.getElementById('chart-' + si);
  if (!canvas) return;
  if (_charts[si]) _charts[si].destroy();
  const wrap = canvas.parentElement;
  const w = wrap ? wrap.clientWidth || 308 : 308;
  const h = Math.max(180, traded.length * 18 + 60);
  canvas.width  = w;
  canvas.height = h;
  const data   = traded.map(r => +r.total_pnl_pct || 0);
  const colors = data.map(v => v >= 0 ? 'rgba(21,128,61,.75)' : 'rgba(185,28,28,.75)');
  _charts[si] = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: traded.map(r => r.symbol),
      datasets: [{ label:'Return %', data, backgroundColor:colors,
                   borderColor:colors.map(c => c.replace('.75','1')),
                   borderWidth:1, borderRadius:3, barThickness:10 }]
    },
    options: {
      indexAxis:'y', responsive:false, maintainAspectRatio:false,
      plugins: { legend:{display:false},
                 tooltip:{ callbacks:{ label: ctx => (ctx.parsed.x>=0?'+':'')+ctx.parsed.x.toFixed(2)+'%' }}},
      scales: {
        x: { grid:{color:'#e5e7eb'}, ticks:{callback:v=>(v>=0?'+':'')+v.toFixed(1)+'%', font:{size:11}}},
        y: { ticks:{font:{size:10,weight:'700'}}}
      }
    }
  });
}

function wireTable(si, stocks) {
  _tableData[si] = stocks;
  _sortState[si] = { col:6, dir:-1 };
  renderTable(si);
}

function sortTable(si, col) {
  const s = _sortState[si];
  s.dir = s.col === col ? -s.dir : -1;
  s.col = col;
  document.querySelectorAll('#tbl-' + si + ' th').forEach((th, i) => {
    th.classList.remove('sort-asc','sort-desc');
    if (i === col) th.classList.add(s.dir === 1 ? 'sort-asc' : 'sort-desc');
  });
  renderTable(si);
}

function toggleTrades(si, sym) {
  const rowId = 'sr-' + si + '-' + sym;
  const detId = 'dr-' + si + '-' + sym;
  const row   = document.getElementById(rowId);
  const exist = document.getElementById(detId);
  if (exist) { exist.remove(); return; }
  const strat = DATA.strategies[si];
  const stock = strat.stocks.find(s => s.symbol === sym);
  if (!stock || !stock.trades || !stock.trades.length) return;
  const eqStatsId = 'eq-stats-' + si + '-' + sym;
  const eqId      = 'eq-' + si + '-' + sym;
  const lcId      = 'lc-' + si + '-' + sym;
  const det = document.createElement('tr');
  det.id = detId;
  det.className = 'detail-row';
  det.innerHTML = '<td colspan="9"><div class="detail-inner">' +
    '<div class="trade-chart-wrap">' +
      '<div class="eq-stats-bar" id="' + eqStatsId + '"></div>' +
      '<div class="eq-wrap" id="' + eqId + '"></div>' +
      '<div class="lc-wrap" id="' + lcId + '"></div>' +
    '</div>' +
    '<div class="tl-wrap">' + buildTradeLog(stock) + '</div>' +
    '</div></td>';
  row.after(det);
  requestAnimationFrame(() => {
    let eqChart = null, lcChart = null;
    try { eqChart = buildEquityChart(stock, eqId, eqStatsId); } catch(_) {}
    try { lcChart = buildCandleChart(stock, lcId); } catch(_) {}
    if (eqChart && lcChart) {
      let _syncing = false;
      eqChart.timeScale().subscribeVisibleLogicalRangeChange(r => {
        if (_syncing || !r) return;
        _syncing = true; lcChart.timeScale().setVisibleLogicalRange(r); _syncing = false;
      });
      lcChart.timeScale().subscribeVisibleLogicalRangeChange(r => {
        if (_syncing || !r) return;
        _syncing = true; eqChart.timeScale().setVisibleLogicalRange(r); _syncing = false;
      });
    }
  });
}

function buildEquityChart(stock, containerId, statsId) {
  const container = document.getElementById(containerId);
  if (!container || typeof LightweightCharts === 'undefined') return;

  const cap    = +stock.initial_capital || 100000;
  const trades = [...(stock.trades || [])].sort((a, b) => a.entry_date.localeCompare(b.entry_date));
  const ohlcv  = stock.ohlcv || [];
  if (!ohlcv.length || !trades.length) return;

  let realized = 0, tIdx = 0;
  let runPeak = 100, peakDate = ohlcv[0].time, ddPeakDate = ohlcv[0].time;
  let maxDD = 0, maxDDDate = ohlcv[0].time, ddDur = 0;
  const equityData = [];

  for (const bar of ohlcv) {
    const date  = bar.time;
    const close = +bar.close;
    while (tIdx < trades.length && trades[tIdx].exit_date <= date) {
      realized += +trades[tIdx].gross_pnl; tIdx++;
    }
    let unrealized = 0;
    for (let i = tIdx; i < trades.length; i++) {
      const t = trades[i];
      if (t.entry_date > date) break;
      if (t.exit_date  > date) { unrealized = +t.shares * (close - +t.entry_price); break; }
    }
    const pct = Math.round((cap + realized + unrealized) / cap * 10000) / 100;
    equityData.push({ time: date, value: pct });
    if (pct > runPeak) { runPeak = pct; peakDate = date; ddPeakDate = date; }
    const dd = pct - runPeak;
    if (dd < maxDD) {
      maxDD = dd; maxDDDate = date;
      ddDur = Math.round((new Date(date) - new Date(ddPeakDate)) / 86400000);
    }
  }

  const finalPct  = equityData[equityData.length - 1].value;
  const peakPct   = runPeak;
  const ddPct     = Math.round(maxDD * 100) / 100;
  const lastDate  = ohlcv[ohlcv.length - 1].time;
  const lineColor = finalPct >= 100 ? '#15803d' : '#b91c1c';
  const fillTop   = finalPct >= 100 ? 'rgba(21,128,61,0.18)' : 'rgba(185,28,28,0.18)';

  const chart = LightweightCharts.createChart(container, {
    width: container.clientWidth || 900, height: 158,
    layout: { background:{ color:'#f8f9ff' }, textColor:'#0f1623' },
    grid:   { vertLines:{ color:'#eef0f6' }, horzLines:{ color:'#eef0f6' } },
    rightPriceScale: { borderColor:'#c8d0e0' },
    timeScale: { borderColor:'#c8d0e0', timeVisible:true, secondsVisible:false },
    handleScroll: true, handleScale: true,
  });
  const equitySeries = chart.addAreaSeries({
    lineColor, topColor:fillTop,
    bottomColor: finalPct >= 100 ? 'rgba(21,128,61,0.02)' : 'rgba(185,28,28,0.02)',
    lineWidth:2, lastValueVisible:true, priceLineVisible:false,
  });
  equitySeries.setData(equityData);
  if (maxDD < 0) {
    equitySeries.createPriceLine({ price:peakPct, color:'#ef4444', lineWidth:1, lineStyle:2, axisLabelVisible:false });
  }
  const markers = [];
  if (peakDate !== ohlcv[0].time && peakDate !== lastDate)
    markers.push({ time:peakDate,  position:'aboveBar', color:'#0891b2', shape:'circle', text:'Peak', size:1 });
  if (maxDD < 0)
    markers.push({ time:maxDDDate, position:'belowBar', color:'#ef4444', shape:'circle', text:'DD',   size:1 });
  markers.push(  { time:lastDate,  position:'aboveBar', color:'#1d4ed8', shape:'circle', text:'End',  size:1 });
  markers.sort((a,b) => a.time.localeCompare(b.time));
  equitySeries.setMarkers(markers);
  chart.timeScale().fitContent();

  const statsEl = statsId ? document.getElementById(statsId) : null;
  if (statsEl) {
    statsEl.innerHTML =
      '<span class="eqs peak"><span class="dot"></span> Peak '   + peakPct.toFixed(2)  + '%</span>' +
      '<span class="eqs final"><span class="dot"></span> Final ' + finalPct.toFixed(2) + '%</span>' +
      '<span class="eqs dd"><span class="dot"></span> Max DD '   + ddPct.toFixed(2)    + '%</span>' +
      (ddDur > 0 ? '<span class="eqs dur"><span class="dot"></span> Max DD Dur. (' + ddDur + 'd)</span>' : '');
  }
  return chart;
}

function buildCandleChart(stock, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (typeof LightweightCharts === 'undefined') {
    container.innerHTML = '<div style="padding:20px;text-align:center;color:#999;font-size:12px">Chart library not loaded</div>';
    return;
  }
  const ohlcv = stock.ohlcv || [];
  if (!ohlcv.length) {
    container.innerHTML = '<div style="padding:20px;text-align:center;color:#999;font-size:12px">No OHLCV data available</div>';
    return;
  }
  const chart = LightweightCharts.createChart(container, {
    width: container.clientWidth || 900, height: 280,
    layout: { background:{ color:'#f8f9ff' }, textColor:'#0f1623' },
    grid:   { vertLines:{ color:'#eef0f6' }, horzLines:{ color:'#eef0f6' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    timeScale: { timeVisible:true, secondsVisible:false, borderColor:'#c8d0e0' },
    rightPriceScale: { borderColor:'#c8d0e0' },
    handleScroll: true, handleScale: true,
  });
  const candleSeries = chart.addCandlestickSeries({
    upColor:'#15803d', downColor:'#b91c1c',
    borderUpColor:'#15803d', borderDownColor:'#b91c1c',
    wickUpColor:'#15803d', wickDownColor:'#b91c1c',
  });
  candleSeries.setData(ohlcv
    .filter(d => d.open != null && d.high != null && d.low != null && d.close != null)
    .map(d => ({ time:d.time, open:d.open, high:d.high, low:d.low, close:d.close })));
  const volSeries = chart.addHistogramSeries({
    priceFormat:{ type:'volume' }, priceScaleId:'', scaleMargins:{ top:0.85, bottom:0 },
  });
  volSeries.setData(ohlcv.filter(d => d.close != null).map(d => ({
    time:d.time, value:d.volume != null ? d.volume : 0,
    color:(+d.close)>=(+d.open||+d.close)?'rgba(21,128,61,0.35)':'rgba(185,28,28,0.35)',
  })));
  const markers = [];
  for (const t of (stock.trades || [])) {
    if (t.entry_date) markers.push({
      time:t.entry_date, position:'belowBar', color:'#1d4ed8', shape:'arrowUp', text:'B', size:1,
    });
    if (t.exit_date) markers.push({
      time:t.exit_date, position:'aboveBar',
      color:+t.gross_pnl>=0?'#15803d':'#b91c1c', shape:'arrowDown', text:'S', size:1,
    });
  }
  markers.sort((a,b) => a.time.localeCompare(b.time));
  candleSeries.setMarkers(markers);
  chart.timeScale().fitContent();
  return chart;
}

function buildTradeLog(stock) {
  const trades = stock.trades || [];
  const ohlcv  = stock.ohlcv  || [];
  const sname  = stock.strategy_name || '';
  const oMap   = {};
  const oDates = [];
  for (const d of ohlcv) { oMap[d.time] = d; oDates.push(d.time); }
  oDates.sort();

  function adjDate(dateStr, offset) {
    const idx = oDates.indexOf(dateStr);
    if (idx < 0) return null;
    const ni = idx + offset;
    return (ni >= 0 && ni < oDates.length) ? oDates[ni] : null;
  }

  const isDelivOI  = sname.includes('Delivery') || sname.includes('OI');
  const extraHeads = isDelivOI ? ['Del %','Del Qty','VWAP','OI Chg'] : ['VWAP'];
  const allHeads   = ['Date','Type','Open','High','Low','Close','Volume', ...extraHeads];
  const thHtml     = '<tr>' + allHeads.map(h => '<th>' + h + '</th>').join('') + '</tr>';
  const NC         = allHeads.length;

  function extraCells(d) {
    if (!d) return Array(extraHeads.length).fill('—').map(v => '<td>' + v + '</td>').join('');
    if (isDelivOI) return [
      d.delivery_pct != null ? fmt(d.delivery_pct, 1) + '%' : '—',
      d.delivery_qty != null ? (+d.delivery_qty).toLocaleString('en-IN') : '—',
      d.vwap         != null ? fmt(d.vwap) : '—',
      d.oi_change    != null ? (+d.oi_change).toLocaleString('en-IN') : '—',
    ].map(v => '<td>' + v + '</td>').join('');
    return '<td>' + (d.vwap != null ? fmt(d.vwap) : '—') + '</td>';
  }

  function dataRow(dateStr, label, rowCls) {
    const d = oMap[dateStr];
    if (!d) return '<tr class="tl-ctx"><td>' + (dateStr || '—') + '</td><td class="pos">' + label + '</td>'
      + Array(NC - 2).fill('<td>—</td>').join('') + '</tr>';
    return '<tr class="' + rowCls + '">' +
      '<td>' + d.time + '</td>' +
      '<td><b>' + label + '</b></td>' +
      '<td>' + fmt(d.open)  + '</td>' +
      '<td>' + fmt(d.high)  + '</td>' +
      '<td>' + fmt(d.low)   + '</td>' +
      '<td>' + fmt(d.close) + '</td>' +
      '<td>' + (d.volume != null ? (+d.volume).toLocaleString('en-IN') : '—') + '</td>' +
      extraCells(d) +
      '</tr>';
  }

  let html = '';
  for (let i = 0; i < trades.length; i++) {
    const t   = trades[i];
    const pos = +t.gross_pnl >= 0;
    html +=
      '<tr class="tl-trade-hdr"><td colspan="' + NC + '">' +
        '<span class="tl-tn">T' + (i + 1) + '</span>' +
        'BUY ' + t.entry_date + ' → SELL ' + t.exit_date +
        ' &nbsp;·&nbsp; ' + t.shares + ' shares' +
        ' &nbsp;·&nbsp; Entry ₹' + fmt(t.entry_price) + '  Exit ₹' + fmt(t.exit_price) +
        ' &nbsp;·&nbsp; Hold ' + t.hold_days + 'd  [' + t.exit_reason + ']' +
        ' &nbsp;·&nbsp; <span class="' + (pos ? 'pos' : 'neg') + '">' +
          (pos ? '+' : '') + t.pnl_pct + '%  ' +
          (pos ? '+' : '-') + fmtInr(Math.abs(+t.gross_pnl)) +
        '</span>' +
      '</td></tr>';
    html += thHtml;
    const pre  = adjDate(t.entry_date, -1);
    const post = adjDate(t.exit_date,  +1);
    html += dataRow(pre,          'Pre-Buy',   'tl-ctx');
    html += dataRow(t.entry_date, 'BUY ▸',     'tl-buy');
    html += dataRow(t.exit_date,  'SELL ◂',    'tl-sell');
    html += dataRow(post,         'Post-Sell', 'tl-ctx');
    if (i < trades.length - 1)
      html += '<tr class="tl-sep"><td colspan="' + NC + '"></td></tr>';
  }
  return '<table class="tl"><tbody>' + html + '</tbody></table>';
}

function renderTable(si) {
  const q         = (document.getElementById('filter-' + si)?.value || '').toLowerCase();
  const hideEmpty = document.getElementById('hide-' + si)?.checked;
  const { col, dir } = _sortState[si] || { col:6, dir:-1 };
  const key = _SORT_KEYS[col] || 'total_pnl_pct';

  let rows = (_tableData[si] || []).filter(r => {
    if (hideEmpty && +r.total_trades === 0) return false;
    if (!q) return true;
    return (r.symbol + ' ' + r.company_name + ' ' + r.sector).toLowerCase().includes(q);
  }).sort((a, b) => {
    const av = a[key], bv = b[key];
    if (typeof av === 'string') return av.localeCompare(bv) * dir;
    return ((+av || 0) - (+bv || 0)) * dir;
  });

  const tbody = document.getElementById('tbody-' + si);
  if (!tbody) return;

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-note">No results match the current filter.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(r => {
    const nt  = +r.total_trades === 0;
    const cls = nt ? 'no-trade' : (+r.total_pnl_pct >= 0 ? 'profit' : 'loss');
    const pnl = nt ? '—'
      : '<span class="' + (+r.total_pnl >= 0 ? 'pos' : 'neg') + '">'
        + (+r.total_pnl >= 0 ? '+' : '-') + fmtInr(r.total_pnl) + '</span>';
    const togBtn = (r.trades && r.trades.length)
      ? '<button class="tog" onclick="toggleTrades(' + si + ',\'' + r.symbol + '\')">' + r.trades.length + '▸</button>'
      : '';
    return '<tr id="sr-' + si + '-' + r.symbol + '" class="' + cls + '">' +
      '<td><span class="sym">' + r.symbol + '</span>' + togBtn + '</td>' +
      '<td>' + (r.company_name || '—') + '</td>' +
      '<td style="color:var(--muted)">' + (r.sector || '—') + '</td>' +
      '<td>' + (nt ? '—' : r.total_trades) + '</td>' +
      '<td>' + (nt ? '—' : fmt(r.win_rate) + '%') + '</td>' +
      '<td>' + pnl + '</td>' +
      '<td>' + (nt ? '—' : signed(r.total_pnl_pct)) + '</td>' +
      '<td>' + (nt ? '—' : fmt(r.max_drawdown_pct) + '%') + '</td>' +
      '<td>' + (nt ? '—' : fmtInr(r.final_capital)) + '</td>' +
      '</tr>';
  }).join('');
}
</script>
</body>
</html>
"""


# ── CSV output ────────────────────────────────────────────────────────────────

def save_csvs(pairs: list[tuple]) -> None:
    """Write comparison_summary.csv and comparison_trades.csv."""
    summary_rows: list[dict] = []
    trade_rows:   list[dict] = []

    for res, meta, *_ in pairs:
        base = {
            "strategy":     res.strategy_name,
            "symbol":       res.symbol,
            "company_name": meta.get("company_name", ""),
            "sector":       meta.get("sector", ""),
            "sub_sector":   meta.get("sub_sector", ""),
        }
        summary_rows.append({
            **base,
            "total_trades":     res.total_trades,
            "wins":             res.wins,
            "losses":           res.losses,
            "win_rate":         res.win_rate,
            "total_pnl":        res.total_pnl,
            "total_pnl_pct":    res.total_pnl_pct,
            "max_drawdown_pct": res.max_drawdown_pct,
            "avg_hold_days":    res.avg_hold_days,
            "final_capital":    res.final_capital,
        })
        for t in res.trades:   # already list[dict] via asdict()
            trade_rows.append({**base, **t})

    pd.DataFrame(summary_rows).to_csv("outputs/comparison_summary.csv", index=False)
    pd.DataFrame(trade_rows).to_csv("outputs/comparison_trades.csv", index=False)
    print(f"  CSVs: outputs/comparison_summary.csv ({len(summary_rows)} rows)  "
          f"outputs/comparison_trades.csv ({len(trade_rows)} trades)")


# ── Perf profiling writer ─────────────────────────────────────────────────────

def _write_perf_profile(
    n_symbols: int,
    stock_data: dict[str, pd.DataFrame],
    pairs: list,
    out_path: str = "docs/perf_profile.md",
) -> None:
    import statistics

    lines: list[str] = [
        "# Performance Profile",
        f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — "
        f"{n_symbols} symbols, {len(_prof_tasks)} tasks_\n",
    ]

    # ── Stage breakdown ───────────────────────────────────────────────────────
    total = sum(_prof_stages.values())
    lines += ["## Stage Breakdown\n", "| Stage | Time (s) | Share |", "|---|---|---|"]
    for stage, t in sorted(_prof_stages.items(), key=lambda x: -x[1]):
        lines.append(f"| {stage} | {t:.2f} | {t/total*100:.1f}% |")
    lines.append(f"| **Total wall time** | **{total:.2f}** | 100% |\n")

    # ── Fetch hotspots ────────────────────────────────────────────────────────
    if _prof_fetch:
        sorted_fetch = sorted(_prof_fetch, key=lambda x: -x[1])
        fetch_times  = [x[1] for x in _prof_fetch]
        lines += [
            "## Stage 1 — Per-Stock Fetch Times\n",
            f"Mean: {statistics.mean(fetch_times):.2f}s  "
            f"Median: {statistics.median(fetch_times):.2f}s  "
            f"Max: {max(fetch_times):.2f}s\n",
            "### Slowest fetches (top 10)\n",
            "| Symbol | Time (s) | Rows |", "|---|---|---|",
        ]
        for sym, t, rows in sorted_fetch[:10]:
            lines.append(f"| {sym} | {t:.2f} | {rows} |")
        lines.append("")

        no_data = [sym for sym, _, rows in _prof_fetch if rows == 0]
        if no_data:
            lines.append(f"**No-data symbols ({len(no_data)}):** {', '.join(no_data)}\n")

    # ── Strategy run hotspots ─────────────────────────────────────────────────
    if _prof_tasks:
        sorted_tasks = sorted(_prof_tasks, key=lambda x: -x[2])
        task_times   = [x[2] for x in _prof_tasks]
        lines += [
            "## Stage 2 — Per-Task (Symbol × Strategy) Times\n",
            f"Mean: {statistics.mean(task_times)*1000:.1f}ms  "
            f"Median: {statistics.median(task_times)*1000:.1f}ms  "
            f"Max: {max(task_times)*1000:.1f}ms\n",
            "### Slowest tasks (top 15)\n",
            "| Symbol | Strategy | Time (ms) |", "|---|---|---|",
        ]
        for sym, strat, t in sorted_tasks[:15]:
            lines.append(f"| {sym} | {strat} | {t*1000:.1f} |")
        lines.append("")

        # Per-strategy averages
        from collections import defaultdict
        strat_times: dict[str, list[float]] = defaultdict(list)
        for _, strat, t in _prof_tasks:
            strat_times[strat].append(t)
        lines += ["### Average time per strategy\n",
                  "| Strategy | Avg (ms) | Max (ms) | Tasks |", "|---|---|---|---|"]
        for strat, ts in sorted(strat_times.items()):
            lines.append(f"| {strat} | {statistics.mean(ts)*1000:.1f} | {max(ts)*1000:.1f} | {len(ts)} |")
        lines.append("")

    # ── Data shape ────────────────────────────────────────────────────────────
    non_empty = {s: d for s, d in stock_data.items() if not d.empty}
    if non_empty:
        row_counts = sorted([(s, len(d)) for s, d in non_empty.items()], key=lambda x: -x[1])
        mem_mb     = sum(d.memory_usage(deep=True).sum() for d in non_empty.values()) / 1e6
        lines += [
            "## Data Shape\n",
            f"Non-empty stocks: {len(non_empty)} / {n_symbols}  "
            f"Total rows: {sum(r for _, r in row_counts):,}  "
            f"In-memory: {mem_mb:.1f} MB\n",
            "| Symbol | Rows | Columns |", "|---|---|---|",
        ]
        for sym, rows in row_counts[:20]:
            cols = len(stock_data[sym].columns)
            lines.append(f"| {sym} | {rows} | {cols} |")
        lines.append("")

    # ── Signal data sizes (report build cost proxy) ───────────────────────────
    if pairs:
        sig_sizes = [(r.symbol, r.strategy_name, len(r.trades))
                     for r, *_ in pairs if r.total_trades > 0]
        sig_sizes.sort(key=lambda x: -x[2])
        total_trades = sum(x[2] for x in sig_sizes)
        lines += [
            "## Report Build — Trade Record Counts\n",
            f"Total trade records embedded in HTML: {total_trades}  "
            f"Avg per result: {total_trades/max(len(pairs),1):.1f}\n",
        ]
        if sig_sizes:
            lines += ["### Top 10 results by trade count\n",
                      "| Symbol | Strategy | Trades |", "|---|---|---|"]
            for sym, strat, n in sig_sizes[:10]:
                lines.append(f"| {sym} | {strat} | {n} |")
            lines.append("")

    # ── Hotspot summary ───────────────────────────────────────────────────────
    lines += [
        "## Hotspot Observations\n",
    ]
    stage_order = sorted(_prof_stages.items(), key=lambda x: -x[1])
    dominant = stage_order[0][0] if stage_order else "fetch"

    fetch_t  = _prof_stages.get("fetch", 0)
    report_t = _prof_stages.get("build_report", 0)
    strat_t  = _prof_stages.get("strategies", 0)

    if fetch_t > 0 and fetch_t / total > 0.5:
        lines.append(
            f"1. **Network I/O dominates** ({fetch_t:.1f}s / {fetch_t/total*100:.0f}% of wall time). "
            "Each stock requires sequential date-range requests to the NSE archive. "
            "Bottleneck: TCP round-trips, not CPU."
        )
        lines.append(
            f"   - `DATA_WORKERS={DATA_WORKERS}` already parallelises fetches; "
            "raising it beyond ~8 risks NSE rate-limiting (429 responses)."
        )
        lines.append(
            "   - **Mitigation**: pre-warm the `.nse_cache/` disk cache on first run; "
            "subsequent runs skip network entirely and drop fetch time to <1s."
        )
    if strat_t > 0:
        strat_share = strat_t / total * 100
        lines.append(
            f"\n2. **Strategy simulation** ({strat_t:.1f}s / {strat_share:.0f}%). "
            "Fast when Numba JIT is active (nogil kernels run truly in parallel). "
            "Slow on first cold run due to JIT compilation (~1–2s one-time cost)."
        )
        if strat_t > 5:
            lines.append(
                "   - With Numba unavailable the GIL still limits parallelism on pure-Python loops; "
                "install `numba` for a 5–10× speedup on large universes."
            )
    if report_t > 0:
        lines.append(
            f"\n3. **`build_report`** ({report_t:.2f}s) — iterates every row of every "
            "signal DataFrame to serialise indicator values to JSON. "
            "Scales linearly with `n_stocks × n_rows × n_signal_cols`. "
            "For large universes (>100 stocks) this becomes the CPU bottleneck."
        )
        lines.append(
            "   - **Mitigation**: replace the Python `iterrows()` loop with "
            "`df[sig_cols].to_dict(orient='records')` (vectorised, ~10× faster)."
        )

    lines += [
        "\n## Key Recommendations\n",
        "| Priority | Change | Expected gain |",
        "|---|---|---|",
        "| High | Warm `.nse_cache/` before bulk runs | Eliminates fetch stage (~70% of total) |",
        "| Medium | Replace `iterrows()` in `build_report` with `.to_dict(orient='records')` | 5–10× faster signal serialisation |",
        "| Medium | Pre-compile Numba kernels at startup (`--warmup` flag) | Removes 1–2s JIT cost on cold runs |",
        "| Low | Increase `DATA_WORKERS` to 12 with exponential-backoff retry | ~30% faster uncached fetches |",
        "| Low | Switch `signal_data` embedding from full history to last 90 days | Cuts HTML size by ~50% for long backtests |",
    ]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Perf profile written -> {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all strategies on all enabled stocks → comparison_report.html"
    )
    parser.add_argument("--csv",      default="data/stocks.csv")
    parser.add_argument("--strategy", nargs="+",  help="Strategy name(s)")
    parser.add_argument("--workers",  type=int,   help="Override STRAT_WORKERS")
    parser.add_argument("--all",        action="store_true",
                        help="Run on ALL stocks in the CSV, ignoring the enabled flag")
    parser.add_argument("--from-cache", action="store_true",
                        help="Load data from parquet cache instead of fetching from NSE")
    parser.add_argument("--cache-dir",  default="data/fetched",
                        help="Parquet cache directory (default: data/fetched)")
    parser.add_argument("--profile",    action="store_true",
                        help="Capture per-stock/per-task timing and write docs/perf_profile.md")
    args = parser.parse_args()

    global STRAT_WORKERS, _PROFILE
    if args.workers:
        STRAT_WORKERS = args.workers
    if args.profile:
        _PROFILE = True

    _banner()
    if args.all:
        print("  [!] --all flag set: running on every stock in the CSV (ignoring enabled flag)\n")
    if getattr(args, "from_cache", False):
        print(f"  [!] --from-cache: loading parquet files from {args.cache_dir}\n")
    if args.profile:
        print("  [!] --profile flag set: per-stage and per-task timing will be collected\n")

    # Validate requested strategy names
    strategy_names: list[str] | None = None
    if args.strategy:
        available = list_strategy_names()
        bad = [s for s in args.strategy if s not in available]
        if bad:
            print(f"Unknown strategies: {bad}\nAvailable: {available}")
            sys.exit(1)
        strategy_names = args.strategy
    all_names = strategy_names or list_strategy_names()

    from_cache = getattr(args, "from_cache", False)

    # ── Resolve symbol list ───────────────────────────────────────────────────
    if from_cache and args.all:
        # Discover every symbol that has a parquet file
        symbols   = _discover_cache_symbols(args.cache_dir)
        meta_df   = pd.read_csv(args.csv, dtype=str).fillna("")
        stocks_df = meta_df[meta_df["stock"].isin(symbols)].reset_index(drop=True)
        print(f"  Cache-discovered stocks: {len(symbols)}  ({args.cache_dir})")
    else:
        stocks_df = load_enabled_stocks(args.csv, all_stocks=args.all)
        symbols   = stocks_df["stock"].tolist()

    if not symbols:
        print("No stocks found. Check data/stocks.csv or the cache directory.")
        sys.exit(0)

    # Build metadata lookup for report annotations
    stock_meta: dict[str, dict] = {
        row["stock"]: {
            "company_name": row.get("company_name", ""),
            "sector":       row.get("sector", ""),
            "sub_sector":   row.get("sub_sector", ""),
        }
        for _, row in stocks_df.iterrows()
    }

    # Stage 1: fetch from NSE or load from parquet cache
    t0 = time.perf_counter()
    if from_cache:
        stock_data = load_from_cache(symbols, args.cache_dir)
    else:
        stock_data = fetch_all(symbols)
    _prof_stages["fetch"] = time.perf_counter() - t0

    # Stage 1b: batch pre-compute indicators via GPU/SIMD if available
    t0         = time.perf_counter()
    stock_data = _batch_precompute(stock_data)
    _prof_stages["batch_precompute"] = time.perf_counter() - t0

    # Stage 2: parallel strategy runs
    t0    = time.perf_counter()
    pairs = run_all_strategies(stock_data, stock_meta, strategy_names)
    _prof_stages["strategies"] = time.perf_counter() - t0

    if not pairs:
        print("[!] No results collected — check that enabled stocks have accessible data.")
        print("    Generating an empty report anyway so you can see the structure.")

    t0 = time.perf_counter()
    save_csvs(pairs)
    _prof_stages["save_csvs"] = time.perf_counter() - t0

    t0  = time.perf_counter()
    out = build_report(pairs, all_names, len(symbols), stock_data=stock_data)
    _prof_stages["build_report"] = time.perf_counter() - t0

    print(f"  Report written -> {out}")
    print(f"  Open it via Dashboard > Comparison button, or directly in a browser.\n")

    if args.profile:
        _write_perf_profile(len(symbols), stock_data, pairs)


if __name__ == "__main__":
    main()
