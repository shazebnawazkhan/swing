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

def load_enabled_stocks(csv_path: str = "stocks.csv") -> pd.DataFrame:
    df   = pd.read_csv(csv_path, dtype=str).fillna("")
    mask = df["enabled"].str.strip().str.upper() == "Y"
    out  = df[mask].copy().reset_index(drop=True)
    print(f"  Enabled stocks: {len(out)}  (from {csv_path})")
    return out


# ── Stage 1: parallel data fetch ─────────────────────────────────────────────

def _fetch_one(symbol: str) -> tuple[str, pd.DataFrame]:
    """
    Fetch OHLCV + delivery data for one symbol using a thread-local
    NSEArchiveFetcher so concurrent threads never share a Session object.
    Falls back to yfinance if the archive returns nothing.
    """
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
            return symbol, pd.DataFrame()

        df.insert(0, "symbol", symbol)
        return symbol, df

    except Exception:
        # Log so the user can diagnose, but never crash the whole pool
        traceback.print_exc()
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


# ── Stage 2 worker ────────────────────────────────────────────────────────────

def _run_one(
    symbol:       str,
    meta:         dict,
    df:           pd.DataFrame,
    strategy_cls,
    engine:       FastBacktester,
) -> tuple[BacktestResult | None, dict]:
    try:
        strategy = strategy_cls()
        result   = engine.run(strategy, df)
        return result, meta
    except Exception:
        print(f"\n  [ERROR] {symbol} / {strategy_cls.params.name}:")
        traceback.print_exc()
        return None, meta


def run_all_strategies(
    stock_data:     dict[str, pd.DataFrame],
    stock_meta:     dict[str, dict],
    strategy_names: list[str] | None,
) -> list[tuple[BacktestResult, dict]]:
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
    pairs: list[tuple[BacktestResult, dict]] = []
    done  = 0

    with ThreadPoolExecutor(max_workers=STRAT_WORKERS) as pool:
        futures = [
            pool.submit(_run_one, sym, meta, df, cls, engine)
            for sym, meta, df, cls in tasks
        ]
        for fut in as_completed(futures):
            result, meta = fut.result()
            done += 1
            if result is not None:
                pairs.append((result, meta))
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done:>5}/{len(tasks)} …", end="\r")

    elapsed = time.perf_counter() - t0
    ok      = len(pairs)
    zeros   = sum(1 for r, _ in pairs if r.total_trades == 0)
    print(f"\n  Strategy runs done in {elapsed:.1f}s")
    print(f"  Results: {ok} collected  |  {zeros} with 0 trades  |  "
          f"{ok - zeros} with >=1 trade\n")
    return pairs


# ── HTML Report ───────────────────────────────────────────────────────────────

def build_report(
    pairs:          list[tuple[BacktestResult, dict]],
    strategy_names: list[str],
    n_enabled:      int,
    stock_data:     dict | None = None,
) -> str:
    by_strat: dict[str, list[dict]] = {n: [] for n in strategy_names}

    for res, meta in pairs:
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
                padding:10px 24px; background:#fff; border-bottom:2px solid var(--border); }
#strategy-bar label { font-size:11px; font-weight:700; color:var(--muted);
                      text-transform:uppercase; letter-spacing:.06em; }
#strat-sel { padding:6px 12px; border:1.5px solid var(--border); border-radius:var(--r);
             font-size:13px; font-weight:600; background:var(--surface); color:var(--text);
             cursor:pointer; outline:none; min-width:240px; }
#strat-sel:focus { border-color:var(--accent); }
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
.panel { display:none; padding:24px; max-width:1440px; margin:0 auto; }
.panel.active { display:block; }
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(155px,1fr));
         gap:12px; margin-bottom:24px; }
.card { background:var(--surface); border:1.5px solid var(--border);
        border-radius:var(--r); padding:14px 18px; }
.cl { font-size:11px; font-weight:700; color:var(--muted); text-transform:uppercase;
      letter-spacing:.06em; margin-bottom:5px; }
.cv { font-size:22px; font-weight:800; }
.chart-wrap { background:var(--surface); border:1.5px solid var(--border);
              border-radius:var(--r); padding:16px; margin-bottom:24px; overflow-x:auto; }
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
.tl-trade-hdr td { background:#dde3f5; font-size:11px; font-weight:700;
                   padding:7px 8px; }
.tl-tn { background:var(--accent); color:#fff; border-radius:3px;
         padding:1px 7px; margin-right:5px; font-size:10px; font-weight:800; }
tr.tl-buy  td { background:#e6f4ea; }
tr.tl-sell td { background:#fde8e8; }
tr.tl-ctx  td { background:#f8f9ff; color:var(--muted); }
tr.tl-ctx  td:first-child,
tr.tl-ctx  td:nth-child(2) { font-weight:700; color:var(--text); }
tr.tl-sep  td { height:8px; background:transparent !important;
                border:none !important; padding:0 !important; }
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
<div id="panels"></div>

<script>
const DATA = __PAYLOAD__;

document.getElementById('h-stocks').textContent =
  DATA.n_enabled + ' stocks  |  ' + DATA.total_with_trades + ' results with trades';
document.getElementById('h-period').textContent =
  DATA.backtest_days + 'd  ₹' + Number(DATA.capital).toLocaleString('en-IN');
document.getElementById('h-date').textContent = DATA.run_date;

const selEl    = document.getElementById('strat-sel');
const panelsEl = document.getElementById('panels');

const _charts    = {};
const _tableData = {};
const _sortState = {};
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
  if (si === 0) { try { buildChart(strat, si); } catch(_) {} }
});

selEl.addEventListener('change', () => activateStrategy(+selEl.value));

function activateStrategy(idx) {
  document.querySelectorAll('.panel').forEach(p =>
    p.classList.toggle('active', p.id === 'panel-' + idx));
  if (_charts[idx]) { _charts[idx].resize(); }
  else { try { buildChart(DATA.strategies[idx], idx); } catch(_) {} }
}

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

function buildPanel(strat, si) {
  const stocks  = strat.stocks;
  const traded  = stocks.filter(r => +r.total_trades > 0);
  const profit  = traded.filter(r => +r.total_pnl_pct > 0);
  const avgRet  = traded.length
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
  const ht = Math.min(500, Math.max(180, traded.length * 24 + 48));
  return '<div class="cards">' +
    cards.map(([l,v]) =>
      '<div class="card"><div class="cl">' + l + '</div><div class="cv">' + v + '</div></div>'
    ).join('') + '</div>' +
    (traded.length ? '<div class="chart-wrap" style="height:' + ht + 'px"><canvas id="chart-' + si + '"></canvas></div>' : '') +
    '<div class="tbl-controls">' +
      '<input type="text" id="filter-' + si + '" placeholder="Filter symbol / company / sector…" oninput="renderTable(' + si + ')">' +
      '<label><input type="checkbox" id="hide-' + si + '" onchange="renderTable(' + si + ')"> Hide no-trade stocks</label>' +
    '</div>' +
    '<div class="tbl-wrap"><table id="tbl-' + si + '">' +
      '<thead><tr>' +
        ['Symbol','Company','Sector','Trades','Win%','Total P&L','Return%','Max DD%','Final Capital']
          .map((h, i) => '<th onclick="sortTable(' + si + ',' + i + ')">' + h + '</th>').join('') +
      '</tr></thead><tbody id="tbody-' + si + '"></tbody></table></div>';
}

function buildChart(strat, si) {
  const traded = strat.stocks.filter(r => +r.total_trades > 0)
    .sort((a,b) => +b.total_pnl_pct - +a.total_pnl_pct);
  if (!traded.length) return;
  const canvas = document.getElementById('chart-' + si);
  if (!canvas) return;
  if (_charts[si]) _charts[si].destroy();
  const data   = traded.map(r => +r.total_pnl_pct || 0);
  const colors = data.map(v => v >= 0 ? 'rgba(21,128,61,.75)' : 'rgba(185,28,28,.75)');
  _charts[si] = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: traded.map(r => r.symbol),
      datasets: [{ label: 'Return %', data, backgroundColor: colors,
                   borderColor: colors.map(c => c.replace('.75','1')),
                   borderWidth:1, borderRadius:3 }]
    },
    options: {
      indexAxis: 'y', responsive:true, maintainAspectRatio:false,
      plugins: { legend:{display:false},
                 tooltip:{ callbacks:{ label: ctx => (ctx.parsed.x>=0?'+':'')+ctx.parsed.x.toFixed(2)+'%' }}},
      scales: {
        x: { grid:{color:'#e5e7eb'}, ticks:{callback:v=>(v>=0?'+':'')+v.toFixed(1)+'%', font:{size:11}}},
        y: { ticks:{font:{size:11,weight:'700'}}}
      }
    }
  });
}

function wireTable(si, stocks) {
  _tableData[si] = stocks;
  _sortState[si] = { col:6, dir:-1 };    // default: Return% desc
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
  // Defer chart creation one RAF so the newly-inserted DOM is fully painted first.
  // This prevents lightweight-charts v4.2 async "Value is null" RAF errors.
  requestAnimationFrame(() => {
    let eqChart = null, lcChart = null;
    try { eqChart = buildEquityChart(stock, eqId, eqStatsId); } catch(_) {}
    try { lcChart = buildCandleChart(stock, lcId); } catch(_) {}
    if (eqChart && lcChart) {
      // Sync time-scale zoom/pan between equity and OHLC charts.
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

  // ── Daily mark-to-market equity = realized P&L + unrealized open position ──
  let realized = 0, tIdx = 0;
  let runPeak = 100, peakDate = ohlcv[0].time, ddPeakDate = ohlcv[0].time;
  let maxDD = 0, maxDDDate = ohlcv[0].time, ddDur = 0;

  const equityData = [];
  const peakData   = [];

  for (const bar of ohlcv) {
    const date  = bar.time;
    const close = +bar.close;

    // Advance realized bucket: all trades whose exit_date ≤ today
    while (tIdx < trades.length && trades[tIdx].exit_date <= date) {
      realized += +trades[tIdx].gross_pnl;
      tIdx++;
    }

    // Unrealized bucket: the one open trade (entry ≤ today < exit)
    let unrealized = 0;
    for (let i = tIdx; i < trades.length; i++) {
      const t = trades[i];
      if (t.entry_date > date) break;
      if (t.exit_date  > date) { unrealized = +t.shares * (close - +t.entry_price); break; }
    }

    const pct = Math.round((cap + realized + unrealized) / cap * 10000) / 100;
    equityData.push({ time: date, value: pct });

    // Rolling peak and drawdown tracking
    if (pct > runPeak) { runPeak = pct; peakDate = date; ddPeakDate = date; }
    peakData.push({ time: date, value: runPeak });

    const dd = pct - runPeak;   // ≤ 0 during drawdown
    if (dd < maxDD) {
      maxDD     = dd;
      maxDDDate = date;
      ddDur     = Math.round((new Date(date) - new Date(ddPeakDate)) / 86400000);
    }
  }

  const finalPct  = equityData[equityData.length - 1].value;
  const peakPct   = runPeak;
  const ddPct     = Math.round(maxDD * 100) / 100;
  const lastDate  = ohlcv[ohlcv.length - 1].time;
  const lineColor = finalPct >= 100 ? '#15803d' : '#b91c1c';
  const fillTop   = finalPct >= 100 ? 'rgba(21,128,61,0.18)' : 'rgba(185,28,28,0.18)';

  // ── Chart ─────────────────────────────────────────────────────────────────
  const chart = LightweightCharts.createChart(container, {
    width:  container.clientWidth || 900,
    height: 158,
    layout: { background: { color: '#f8f9ff' }, textColor: '#0f1623' },
    grid:   { vertLines: { color: '#eef0f6' }, horzLines: { color: '#eef0f6' } },
    rightPriceScale: { borderColor: '#c8d0e0' },
    timeScale: { borderColor: '#c8d0e0', timeVisible: true, secondsVisible: false },
    handleScroll: true, handleScale: true,
  });

  // ── Equity area series (single series — avoids multi-area price-scale conflicts)
  const equitySeries = chart.addAreaSeries({
    lineColor:        lineColor,
    topColor:         fillTop,
    bottomColor:      finalPct >= 100 ? 'rgba(21,128,61,0.02)' : 'rgba(185,28,28,0.02)',
    lineWidth:        2,
    lastValueVisible: true,
    priceLineVisible: false,
  });
  equitySeries.setData(equityData);

  // ── Peak price line: horizontal dashed red line across the chart at peak level.
  // axisLabelVisible:false avoids the duplicate-label async RAF error in LW-Charts v4.
  if (maxDD < 0) {
    equitySeries.createPriceLine({
      price:            peakPct,
      color:            '#ef4444',
      lineWidth:        1,
      lineStyle:        2,
      axisLabelVisible: false,
    });
  }

  // ── Markers: peak (cyan), max-DD trough (red), final (blue)
  const markers = [];
  if (peakDate !== ohlcv[0].time && peakDate !== lastDate)
    markers.push({ time: peakDate,  position: 'aboveBar', color: '#0891b2', shape: 'circle', text: 'Peak', size: 1 });
  if (maxDD < 0)
    markers.push({ time: maxDDDate, position: 'belowBar', color: '#ef4444', shape: 'circle', text: 'DD',   size: 1 });
  markers.push(   { time: lastDate,  position: 'aboveBar', color: '#1d4ed8', shape: 'circle', text: 'End',  size: 1 });
  markers.sort((a, b) => a.time.localeCompare(b.time));
  equitySeries.setMarkers(markers);

  chart.timeScale().fitContent();

  // ── Stats bar ─────────────────────────────────────────────────────────────
  const statsEl = statsId ? document.getElementById(statsId) : null;
  if (statsEl) {
    statsEl.innerHTML =
      '<span class="eqs peak"><span class="dot"></span> Peak '  + peakPct.toFixed(2)  + '%</span>' +
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
    width:  container.clientWidth || 900,
    height: 280,
    layout: { background: { color: '#f8f9ff' }, textColor: '#0f1623' },
    grid:   { vertLines: { color: '#eef0f6' }, horzLines: { color: '#eef0f6' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    timeScale: { timeVisible: true, secondsVisible: false, borderColor: '#c8d0e0' },
    rightPriceScale: { borderColor: '#c8d0e0' },
    handleScroll: true, handleScale: true,
  });
  const candleSeries = chart.addCandlestickSeries({
    upColor: '#15803d', downColor: '#b91c1c',
    borderUpColor: '#15803d', borderDownColor: '#b91c1c',
    wickUpColor: '#15803d', wickDownColor: '#b91c1c',
  });
  candleSeries.setData(ohlcv
    .filter(d => d.open != null && d.high != null && d.low != null && d.close != null)
    .map(d => ({ time: d.time, open: d.open, high: d.high, low: d.low, close: d.close })));
  const volSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: '',
    scaleMargins: { top: 0.85, bottom: 0 },
  });
  volSeries.setData(ohlcv.filter(d => d.close != null).map(d => ({
    time: d.time, value: d.volume != null ? d.volume : 0,
    color: (+d.close) >= (+d.open || +d.close) ? 'rgba(21,128,61,0.35)' : 'rgba(185,28,28,0.35)',
  })));
  const markers = [];
  for (const t of (stock.trades || [])) {
    if (t.entry_date) markers.push({
      time: t.entry_date, position: 'belowBar',
      color: '#1d4ed8', shape: 'arrowUp', text: 'B', size: 1,
    });
    if (t.exit_date) markers.push({
      time: t.exit_date, position: 'aboveBar',
      color: +t.gross_pnl >= 0 ? '#15803d' : '#b91c1c',
      shape: 'arrowDown', text: 'S', size: 1,
    });
  }
  markers.sort((a, b) => a.time.localeCompare(b.time));
  candleSeries.setMarkers(markers);
  chart.timeScale().fitContent();
  return chart;
}

function buildTradeLog(stock) {
  const trades = stock.trades || [];
  const ohlcv  = stock.ohlcv  || [];
  const sname  = stock.strategy_name || '';

  // Build date-indexed OHLCV map and sorted date list
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

  // Strategy-specific extra columns
  const isDelivOI = sname.includes('Delivery') || sname.includes('OI');
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
    // Trade summary header row
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
    // Column headers
    html += thHtml;
    // ±1 context rows
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

def save_csvs(pairs: list[tuple[BacktestResult, dict]]) -> None:
    """Write comparison_summary.csv and comparison_trades.csv."""
    summary_rows: list[dict] = []
    trade_rows:   list[dict] = []

    for res, meta in pairs:
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


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all strategies on all enabled stocks → comparison_report.html"
    )
    parser.add_argument("--csv",      default="data/stocks.csv")
    parser.add_argument("--strategy", nargs="+",  help="Strategy name(s)")
    parser.add_argument("--workers",  type=int,   help="Override STRAT_WORKERS")
    args = parser.parse_args()

    global STRAT_WORKERS
    if args.workers:
        STRAT_WORKERS = args.workers

    _banner()

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

    # Load enabled stocks
    stocks_df = load_enabled_stocks(args.csv)
    if stocks_df.empty:
        print("No enabled stocks found. Edit data/stocks.csv and set enabled=Y for at least one row.")
        sys.exit(0)

    symbols   = stocks_df["stock"].tolist()
    stock_meta: dict[str, dict] = {
        row["stock"]: {
            "company_name": row.get("company_name", ""),
            "sector":       row.get("sector", ""),
            "sub_sector":   row.get("sub_sector", ""),
        }
        for _, row in stocks_df.iterrows()
    }

    # Stage 1: parallel data fetch
    stock_data = fetch_all(symbols)

    # Stage 1b: batch pre-compute indicators via GPU/SIMD if available
    stock_data = _batch_precompute(stock_data)

    # Stage 2: parallel strategy runs
    pairs = run_all_strategies(stock_data, stock_meta, strategy_names)

    if not pairs:
        print("[!] No results collected — check that enabled stocks have accessible data.")
        print("    Generating an empty report anyway so you can see the structure.")

    save_csvs(pairs)
    out = build_report(pairs, all_names, len(symbols), stock_data=stock_data)
    print(f"  Report written -> {out}")
    print(f"  Open it via Dashboard > Comparison button, or directly in a browser.\n")


if __name__ == "__main__":
    main()
