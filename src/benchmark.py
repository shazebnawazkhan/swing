"""
benchmark.py
------------
A daily market benchmark for honest, market-adjusted strategy evaluation.

Why this exists
---------------
The first experiment batch (docs/research/batch1.md) was run in a year where the
median universe stock fell ~14.8%.  Raw profit-factor / Sharpe cannot tell whether
a strategy *lost less than the market* (real alpha) or simply rode it down.  Kissell
(*Algorithmic Trading Methods*, ch.3 — Index-Adjusted Performance Metric) prescribes
judging performance against a benchmark return rather than an absolute return.  This
module provides that benchmark and a derived market regime.

Two benchmark sources, in order of preference:
  1. NIFTY 50 (``^NSEI``) via yfinance — the real market index, cached to
     ``data/fetched/_NIFTY.parquet`` (incremental, offline-tolerant).
  2. A synthetic *equal-weight universe index* built from the cached per-symbol
     parquets — needs no network and is the natural benchmark for "did this beat
     just holding the universe?".

Public API
----------
    load_benchmark(prefer="nifty") -> pd.DataFrame   # columns: date, close
    benchmark_return(bench, d0, d1) -> float          # % return between two dates
    regime_series(bench, span=50) -> pd.Series        # "bull"/"bear" by close vs EMA
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FETCHED_DIR = ROOT / "data" / "fetched"
NIFTY_CACHE = FETCHED_DIR / "_NIFTY.parquet"
UNIV_CACHE = FETCHED_DIR / "_UNIVERSE_INDEX.parquet"


# ── NIFTY (real index) ──────────────────────────────────────────────────────────

def fetch_nifty(start: str = "2024-06-01", end: str | None = None) -> pd.DataFrame | None:
    """Fetch ^NSEI daily closes via yfinance; cache to NIFTY_CACHE. None if offline."""
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        raw = yf.download("^NSEI", start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or len(raw) == 0:
            return None
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):       # yfinance multiindex → squeeze
            close = close.iloc[:, 0]
        out = pd.DataFrame({
            "date": pd.to_datetime(close.index).tz_localize(None),
            "close": close.to_numpy(dtype=float),
        }).reset_index(drop=True)
        FETCHED_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(NIFTY_CACHE, index=False)
        return out
    except Exception:
        return None


# ── Synthetic universe index (offline fallback) ──────────────────────────────────

def build_universe_index(min_rows: int = 150, min_price: float = 10.0) -> pd.DataFrame:
    """
    Equal-weight universe index: the daily mean of per-symbol simple returns,
    compounded into a price level (base 1000).  Offline; uses cached parquets only.
    """
    daily: dict[pd.Timestamp, list[float]] = {}
    for p in sorted(FETCHED_DIR.glob("*.parquet")):
        if p.stem.startswith("_"):                # skip benchmark caches
            continue
        df = pd.read_parquet(p, columns=["date", "close"])
        if len(df) < min_rows or df["close"].median() < min_price:
            continue
        df = df.sort_values("date")
        ret = df["close"].pct_change()
        for d, r in zip(df["date"], ret):
            if pd.notna(r) and abs(r) < 0.5:      # drop corp-action / bad-tick spikes
                daily.setdefault(pd.Timestamp(d), []).append(float(r))

    if not daily:
        raise RuntimeError("No cached symbols available to build a universe index.")

    dates = sorted(daily)
    mean_ret = np.array([np.mean(daily[d]) for d in dates])
    level = 1000.0 * np.cumprod(1.0 + mean_ret)
    out = pd.DataFrame({"date": pd.to_datetime(dates), "close": level})
    FETCHED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(UNIV_CACHE, index=False)
    return out


# ── Loader ───────────────────────────────────────────────────────────────────────

def load_benchmark(prefer: str = "nifty", refresh: bool = False) -> tuple[pd.DataFrame, str]:
    """
    Return (benchmark_df, source_label).  benchmark_df has columns date, close.
    prefer="nifty" tries the real index first (cached/fetch), else synthetic.
    prefer="universe" forces the synthetic equal-weight universe index.
    """
    if prefer == "nifty":
        if not refresh and NIFTY_CACHE.exists():
            return pd.read_parquet(NIFTY_CACHE), "nifty50 (^NSEI, cached)"
        fetched = fetch_nifty()
        if fetched is not None and len(fetched):
            return fetched, "nifty50 (^NSEI, fetched)"
        # fall through to synthetic when offline

    if not refresh and UNIV_CACHE.exists():
        return pd.read_parquet(UNIV_CACHE), "synthetic equal-weight universe index"
    return build_universe_index(), "synthetic equal-weight universe index"


# ── Derived quantities ────────────────────────────────────────────────────────────

def _aligned_close(bench: pd.DataFrame) -> pd.Series:
    s = bench.copy()
    s["date"] = pd.to_datetime(s["date"])
    return s.set_index("date").sort_index()["close"]


def benchmark_return(bench: pd.DataFrame, d0, d1) -> float:
    """
    Percent return of the benchmark between two dates (inclusive of d1), using the
    last available close on/before each date (handles non-trading days).  0.0 if
    either side cannot be located.
    """
    s = _aligned_close(bench)
    d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
    a = s.asof(d0)
    b = s.asof(d1)
    if pd.isna(a) or pd.isna(b) or a <= 0:
        return 0.0
    return float((b / a - 1.0) * 100.0)


def regime_series(bench: pd.DataFrame, span: int = 50) -> pd.Series:
    """Boolean 'bull' series: benchmark close >= its `span`-day EMA, indexed by date."""
    s = _aligned_close(bench)
    ema = s.ewm(span=span, adjust=False).mean()
    return (s >= ema).rename("bull")


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    bench, src = load_benchmark(prefer="nifty", refresh=True)
    bench["date"] = pd.to_datetime(bench["date"])
    reg = regime_series(bench)
    bull_days = int(reg.sum())
    print(f"Benchmark source : {src}")
    print(f"Rows             : {len(bench)}  ({bench['date'].min():%Y-%m-%d} → {bench['date'].max():%Y-%m-%d})")
    print(f"Total return     : {(bench['close'].iloc[-1]/bench['close'].iloc[0]-1)*100:.1f}%")
    print(f"Regime (vs 50EMA): {bull_days}/{len(reg)} bull days ({bull_days/len(reg)*100:.0f}%)")
    # also build the offline synthetic index so the fallback is ready
    try:
        uni = build_universe_index()
        print(f"Universe index   : built, {len(uni)} days, "
              f"total {(uni['close'].iloc[-1]/uni['close'].iloc[0]-1)*100:.1f}%")
    except Exception as e:
        print(f"Universe index   : skipped ({e})")
