"""
src.predictor.adapters.intraday_adapter
---------------------------------------
Intraday 5-minute entry-timing features from yfinance (docs/PREDICTOR.md §3,
hypothesis hp_006). yfinance only serves ~60 days of 5m history for `<SYM>.NS`,
so these features are **optional/off** by design and can only be tested on a
short, recent window — never used to form the daily label.

Each feature describes day T's *completed* session (known at the close of T) and
is used to predict the T+1 next-open swing entry, so there is no lookahead: the
entry decision happens after T closes.

Graceful degradation: any fetch/parse failure for a symbol yields no rows (never
raises); its panel features come out NaN.

Public API
  fetch_intraday(symbols, ...) -> DataFrame[symbol, datetime, open, high, low, close, volume]
  build_features(panel_ids, bars) -> DataFrame[date, symbol, INTRADAY_FEATURES]
  INTRADAY_FEATURES                          the feature columns it contributes
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

INTRADAY_FEATURES = ["id_close_pos", "id_ret", "id_first_hour_ret",
                     "id_last_hour_ret", "id_vwap_dev", "id_vol", "id_range_pct"]

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "features" / "intraday"
_TZ = "Asia/Kolkata"
_BARS_PER_HOUR = 12          # 5-minute bars in one hour


def fetch_intraday(symbols: list[str], period: str = "60d", interval: str = "5m",
                   throttle: float = 0.3, verbose: bool = True) -> pd.DataFrame:
    """5m bars per symbol via yfinance. Long frame [symbol, datetime, o/h/l/c, volume]."""
    import yfinance as yf

    out, t0, ok = [], time.time(), 0
    for i, sym in enumerate(symbols, 1):
        try:
            h = yf.Ticker(f"{sym}.NS").history(period=period, interval=interval, auto_adjust=False)
        except Exception:
            h = None
        if h is not None and len(h):
            df = h.reset_index()
            tcol = "Datetime" if "Datetime" in df.columns else df.columns[0]
            df = df.rename(columns={tcol: "datetime", "Open": "open", "High": "high",
                                    "Low": "low", "Close": "close", "Volume": "volume"})
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(_TZ)
            df["symbol"] = sym
            keep = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
            out.append(df[[c for c in keep if c in df.columns]])
            ok += 1
        if throttle:
            time.sleep(throttle)
        if verbose and i % 25 == 0:
            print(f"      intraday {i}/{len(symbols)} … ({ok} with data)")

    secs = time.time() - t0
    if not out:
        if verbose:
            print(f"      intraday: 0/{len(symbols)} symbols returned data ({secs:.1f}s)")
        return pd.DataFrame(columns=["symbol", "datetime", "open", "high", "low", "close", "volume"])
    res = pd.concat(out, ignore_index=True)
    if verbose:
        cov = ok / max(len(symbols), 1) * 100
        print(f"      intraday: {ok}/{len(symbols)} symbols ({cov:.0f}% coverage), "
              f"{len(res):,} 5m-bars, {secs:.1f}s")
    return res


def cache_path(asof: str) -> Path:
    return _CACHE_DIR / f"{asof}.parquet"


def save(bars: pd.DataFrame, asof: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(asof)
    bars.to_parquet(p, index=False)
    return p


def _session_features(g: pd.DataFrame) -> dict:
    """Entry-timing features for one symbol-day's 5m bars (chronological)."""
    o = g["open"].to_numpy(dtype=float)
    h = g["high"].to_numpy(dtype=float)
    low = g["low"].to_numpy(dtype=float)
    c = g["close"].to_numpy(dtype=float)
    v = g["volume"].to_numpy(dtype=float)
    n = len(c)
    if n < 2 or o[0] <= 0:
        return {k: np.nan for k in INTRADAY_FEATURES}
    day_hi, day_lo = np.nanmax(h), np.nanmin(low)
    rng = day_hi - day_lo
    vsum = np.nansum(v)
    vwap = float(np.nansum(c * v) / vsum) if vsum > 0 else float(np.nanmean(c))
    rets = np.diff(c) / c[:-1]
    fh = min(_BARS_PER_HOUR, n - 1)
    return {
        "id_close_pos": float((c[-1] - day_lo) / rng) if rng > 0 else 0.5,
        "id_ret": float(c[-1] / o[0] - 1.0),
        "id_first_hour_ret": float(c[fh] / o[0] - 1.0),
        "id_last_hour_ret": float(c[-1] / o[max(0, n - _BARS_PER_HOUR)] - 1.0),
        "id_vwap_dev": float(c[-1] / vwap - 1.0) if vwap > 0 else 0.0,
        "id_vol": float(np.nanstd(rets)) if len(rets) else 0.0,
        "id_range_pct": float(rng / o[0]) if o[0] > 0 else 0.0,
    }


def build_features(panel_ids: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(date, symbol) intraday entry-timing features, joined onto panel ids.
    Features are from day T's session (known at close T → predicts T+1), no shift.
    Returns [date, symbol, *INTRADAY_FEATURES]; uncovered rows → NaN.
    """
    ids = panel_ids[["date", "symbol"]].copy()
    ids["date"] = pd.to_datetime(ids["date"]).dt.normalize()
    if bars is None or bars.empty:
        for col in INTRADAY_FEATURES:
            ids[col] = np.nan
        return ids

    b = bars.copy()
    b["datetime"] = pd.to_datetime(b["datetime"], utc=True).dt.tz_convert(_TZ)
    b["date"] = b["datetime"].dt.normalize().dt.tz_localize(None)
    b = b.sort_values(["symbol", "datetime"])

    recs = []
    for (sym, d), g in b.groupby(["symbol", "date"], sort=False):
        f = _session_features(g)
        f["symbol"], f["date"] = sym, d
        recs.append(f)
    feat = pd.DataFrame(recs)
    return ids.merge(feat, on=["date", "symbol"], how="left")
