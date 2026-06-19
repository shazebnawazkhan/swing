"""
src.predictor.features
----------------------
Build the (date, symbol) feature panel from cached daily bars.

Contract (docs/PREDICTOR.md §4, §11.1): every feature at row T uses ONLY data
visible at close[T]; labels (labels.py) are strictly forward. Missing values stay
NaN (LightGBM-native). Cross-sectional features are ranked within each date so the
model is robust to universe-wide regime drift.

Public API
----------
  FEATURE_COLS                      -> list[str]   the model input columns
  build_panel(data, nifty)          -> pd.DataFrame  long panel, one row per (date,symbol)
  feature_manifest()                -> dict        col -> {source, lag, kind}
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import fast_indicators as fi

# ── Feature catalog ────────────────────────────────────────────────────────────
# Grouped by source; the union (minus identifiers) is what the model consumes.
_PRICE_FEATS = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "ema_gap_20", "ema_gap_50", "ema_gap_200",
    "rsi_14", "rsi_2", "atr_pct_14", "bb_pctb_20",
    "dist_120_high", "gap_pct", "vol_z_20", "above_ema200",
]
_MICRO_FEATS = [
    "delivery_pct", "deliv_vs_avg", "deliv_z_20", "vwap_dev", "oi_change",
]
_XS_FEATS = [
    "rs_rank_60", "ret_20_xs", "deliv_pct_xs",
]
_CTX_FEATS = [
    "mkt_ret_20", "mkt_above_ema50", "rel_ret_20", "dow",
]

FEATURE_COLS: list[str] = _PRICE_FEATS + _MICRO_FEATS + _XS_FEATS + _CTX_FEATS
ID_COLS = ["date", "symbol"]


# ── Small indicator helpers (vectorised, NaN-safe) ─────────────────────────────

def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = fi.rolling_mean(gain, period)
    avg_loss = fi.rolling_mean(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, rsi)
    return rsi


def _atr_pct(high, low, close, period) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    atr = fi.rolling_mean(tr, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(close > 0, atr / close * 100.0, np.nan)


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    return -fi.rolling_min(-arr, window)


# ── Per-symbol feature computation ─────────────────────────────────────────────

def _features_one(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-symbol features. Input sorted by date; output same length."""
    out = df[["date", "symbol"]].copy()
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    openp = df["open"].to_numpy(dtype=np.float64)
    vol = df["total_volume"].to_numpy(dtype=np.float64)

    def ret(n):
        prev = np.roll(close, n)
        prev[:n] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(prev > 0, close / prev - 1.0, np.nan)

    out["ret_1"] = ret(1)
    out["ret_3"] = ret(3)
    out["ret_5"] = ret(5)
    out["ret_10"] = ret(10)
    out["ret_20"] = ret(20)

    for span in (20, 50, 200):
        e = fi.ema(close, span)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"ema_gap_{span}"] = np.where(e > 0, close / e - 1.0, np.nan)
    out["above_ema200"] = (close > fi.ema(close, 200)).astype(float)

    out["rsi_14"] = _rsi(close, 14)
    out["rsi_2"] = _rsi(close, 2)
    out["atr_pct_14"] = _atr_pct(high, low, close, 14)

    # Bollinger %B (20, 2σ)
    mid = fi.rolling_mean(close, 20)
    sd = fi.rolling_std(close, 20, ddof=0)
    upper, lower = mid + 2 * sd, mid - 2 * sd
    with np.errstate(divide="ignore", invalid="ignore"):
        out["bb_pctb_20"] = np.where(upper > lower, (close - lower) / (upper - lower), np.nan)

    hi = _rolling_max(close, min(120, len(close)))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dist_120_high"] = np.where(hi > 0, close / hi - 1.0, np.nan)

    prev_close = np.roll(close, 1); prev_close[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        out["gap_pct"] = np.where(prev_close > 0, openp / prev_close - 1.0, np.nan)

    vmean = fi.rolling_mean(vol, 20)
    vstd = fi.rolling_std(vol, 20, ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["vol_z_20"] = np.where(vstd > 0, (vol - vmean) / vstd, np.nan)

    # Microstructure (NSE) — absent-tolerant
    if "delivery_pct" in df:
        dp = df["delivery_pct"].to_numpy(dtype=np.float64)
        out["delivery_pct"] = dp
        davg = fi.rolling_mean(dp, 20)
        dstd = fi.rolling_std(dp, 20, ddof=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["deliv_vs_avg"] = np.where(davg > 0, dp / davg - 1.0, np.nan)
            out["deliv_z_20"] = np.where(dstd > 0, (dp - davg) / dstd, np.nan)
    else:
        out["delivery_pct"] = np.nan
        out["deliv_vs_avg"] = np.nan
        out["deliv_z_20"] = np.nan

    if "vwap" in df:
        vwap = df["vwap"].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["vwap_dev"] = np.where(vwap > 0, close / vwap - 1.0, np.nan)
    else:
        out["vwap_dev"] = np.nan

    out["oi_change"] = df["oi_change"].to_numpy(dtype=np.float64) if "oi_change" in df else np.nan
    return out


# ── Cross-sectional + market-context features (need the whole panel) ───────────

def _add_market_features(panel: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    if nifty is None or len(nifty) == 0:
        panel["mkt_ret_20"] = np.nan
        panel["mkt_above_ema50"] = np.nan
    else:
        nclose = nifty["close"].to_numpy(dtype=np.float64)
        mkt_ret = np.full(len(nclose), np.nan)
        prev = np.roll(nclose, 20); prev[:20] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            mkt_ret = np.where(prev > 0, nclose / prev - 1.0, np.nan)
        mkt_above = (nclose > fi.ema(nclose, 50)).astype(float)
        nmap = pd.DataFrame({"date": nifty["date"], "mkt_ret_20": mkt_ret, "mkt_above_ema50": mkt_above})
        panel = panel.merge(nmap, on="date", how="left")
    panel["rel_ret_20"] = panel["ret_20"] - panel["mkt_ret_20"]
    panel["dow"] = panel["date"].dt.dayofweek.astype(float)
    return panel


def _add_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional ranks (pct in [0,1]); robust to regime drift."""
    g = panel.groupby("date", sort=False)
    panel["rs_rank_60"] = g["ret_20"].rank(pct=True)        # 20d used as 60d proxy on short history
    panel["ret_20_xs"] = g["ret_20"].rank(pct=True)
    panel["deliv_pct_xs"] = g["delivery_pct"].rank(pct=True)
    return panel


def build_panel(data: dict[str, pd.DataFrame], nifty: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Assemble the long feature panel. `data` is {symbol: bars_df} from data.load_universe.
    Returns one row per (date, symbol) with FEATURE_COLS (+ id cols). NaNs preserved.
    """
    parts = []
    for sym, df in data.items():
        try:
            parts.append(_features_one(df))
        except Exception as e:                  # one bad symbol never kills the panel
            print(f"    ! features failed for {sym}: {e}")
    panel = pd.concat(parts, ignore_index=True)
    panel = _add_market_features(panel, nifty)
    panel = _add_cross_sectional(panel)
    # Stable column order; guarantee every declared feature exists
    for c in FEATURE_COLS:
        if c not in panel.columns:
            panel[c] = np.nan
    return panel[ID_COLS + FEATURE_COLS].sort_values(["date", "symbol"]).reset_index(drop=True)


def feature_manifest() -> dict:
    """col -> {source, lag_bars, kind} (docs/PREDICTOR.md §11.1)."""
    man = {}
    for c in _PRICE_FEATS:
        man[c] = {"source": "daily_bars", "lag_bars": 0, "kind": "price"}
    for c in _MICRO_FEATS:
        man[c] = {"source": "daily_bars", "lag_bars": 0, "kind": "microstructure"}
    for c in _XS_FEATS:
        man[c] = {"source": "panel", "lag_bars": 0, "kind": "cross_sectional"}
    for c in _CTX_FEATS:
        man[c] = {"source": "nifty/calendar", "lag_bars": 0, "kind": "context"}
    return man
