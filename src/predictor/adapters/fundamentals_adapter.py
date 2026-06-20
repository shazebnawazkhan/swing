"""
src.predictor.adapters.fundamentals_adapter
--------------------------------------------
Quarterly EPS / earnings features from yfinance (docs/PREDICTOR.md §3, hypothesis
hp_003). Unlike news, earnings *report dates* are historically public, so these
features are genuinely BACKFILLABLE point-in-time across the training window — the
report date is the available_date, and the feature store only ever joins on
report_date < T (strictly, so a same-day post-close announcement never leaks).

This adapter covers the backfillable fundamentals (EPS surprise, EPS YoY growth,
days-since-result, post-result drift window). Analyst targets / ratings are
current-only on free sources (no history), so they stay forward-accrual like news
and are out of scope here.

Graceful degradation: any network/parse failure for a symbol yields no rows for
that symbol (never raises); its panel features come out NaN.

Public API
  fetch_earnings(symbols, ...) -> DataFrame[symbol, report_date, eps_est,
                                            reported_eps, surprise_pct]
  build_features(panel_ids, earnings) -> DataFrame[date, symbol, FUND_FEATURES]
  FUND_FEATURES                                the feature columns it contributes
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

FUND_FEATURES = ["eps_surprise_pct", "eps_surprise_sign", "eps_yoy",
                 "days_since_result", "in_result_window"]

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "features" / "fundamentals"
_RESULT_WINDOW_DAYS = 5      # post-result drift window flag
_FRESH_MAX_DAYS = 120        # a result older than ~1 quarter is "stale" (no fresh fundamentals)


def fetch_earnings(symbols: list[str], limit: int = 16, throttle: float = 0.4,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Historical earnings dates + EPS surprise per symbol via yfinance (`<SYM>.NS`).

    `limit` = how many past quarters to pull (16 ≈ 4 years). Returns a long frame
    [symbol, report_date, eps_est, reported_eps, surprise_pct]; report_date is the
    date the result was published (point-in-time available_date). Rows with no
    reported EPS (future/unannounced) are dropped.
    """
    import yfinance as yf

    out, t0, ok = [], time.time(), 0
    for i, sym in enumerate(symbols, 1):
        try:
            ed = yf.Ticker(f"{sym}.NS").get_earnings_dates(limit=limit)
        except Exception:
            ed = None
        if ed is not None and len(ed):
            df = ed.reset_index()
            df.columns = [str(c).strip() for c in df.columns]
            ren = {"Earnings Date": "report_date", "EPS Estimate": "eps_est",
                   "Reported EPS": "reported_eps", "Surprise(%)": "surprise_pct"}
            df = df.rename(columns=ren)
            if "report_date" in df and "reported_eps" in df:
                df = df[df["reported_eps"].notna()].copy()
                if len(df):
                    # Drop tz, keep the calendar date the result was published.
                    df["report_date"] = pd.to_datetime(df["report_date"], utc=True).dt.tz_localize(None).dt.normalize()
                    df["symbol"] = sym
                    keep = ["symbol", "report_date", "eps_est", "reported_eps", "surprise_pct"]
                    out.append(df[[c for c in keep if c in df.columns]])
                    ok += 1
        if throttle:
            time.sleep(throttle)
        if verbose and i % 25 == 0:
            print(f"      fundamentals {i}/{len(symbols)} … ({ok} with data)")

    secs = time.time() - t0
    if not out:
        if verbose:
            print(f"      fundamentals: 0/{len(symbols)} symbols returned data ({secs:.1f}s)")
        return pd.DataFrame(columns=["symbol", "report_date", "eps_est",
                                     "reported_eps", "surprise_pct"])
    res = pd.concat(out, ignore_index=True).sort_values(["symbol", "report_date"])
    if verbose:
        cov = ok / max(len(symbols), 1) * 100
        print(f"      fundamentals: {ok}/{len(symbols)} symbols ({cov:.0f}% coverage), "
              f"{len(res)} quarter-rows, {secs:.1f}s")
    return res.reset_index(drop=True)


def cache_path(asof: str) -> Path:
    return _CACHE_DIR / f"{asof}.parquet"


def save(earnings: pd.DataFrame, asof: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(asof)
    earnings.to_parquet(p, index=False)
    return p


def build_features(panel_ids: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """
    Point-in-time as-of join of earnings onto panel (date, symbol) rows.

    For each row at date T we attach the most recent earnings report STRICTLY before
    T (`report_date < T`, so a result announced on T — possibly after close — never
    leaks into a T-close feature used to predict T+1):

      eps_surprise_pct  : surprise% of that last report
      eps_surprise_sign : sign of the surprise (-1/0/+1)
      eps_yoy           : reported_eps vs the same quarter a year earlier (4 reports back)
      days_since_result : trading-calendar days since that report
      in_result_window  : 1 if within `_RESULT_WINDOW_DAYS` of the report (post-result drift)

    Returns [date, symbol, *FUND_FEATURES]; symbols/dates with no prior report → NaN.
    """
    ids = panel_ids[["date", "symbol"]].copy()
    ids["date"] = pd.to_datetime(ids["date"])
    if earnings is None or earnings.empty:
        for c in FUND_FEATURES:
            ids[c] = np.nan
        return ids

    e = earnings.copy()
    e["report_date"] = pd.to_datetime(e["report_date"])
    e = e.sort_values(["symbol", "report_date"])
    # YoY EPS growth: this quarter's reported EPS vs 4 quarters earlier, per symbol.
    e["eps_yoy"] = e.groupby("symbol")["reported_eps"].transform(
        lambda s: s / s.shift(4) - 1.0)

    parts = []
    for sym, g in ids.groupby("symbol", sort=False):
        eg = e[e["symbol"] == sym]
        g = g.sort_values("date")
        if eg.empty:
            for c in FUND_FEATURES:
                g[c] = np.nan
            parts.append(g)
            continue
        eg = eg.sort_values("report_date")
        m = pd.merge_asof(
            g, eg[["report_date", "surprise_pct", "eps_yoy"]],
            left_on="date", right_on="report_date",
            direction="backward", allow_exact_matches=False,  # strictly report_date < T
        )
        m["eps_surprise_pct"] = m["surprise_pct"]
        m["eps_surprise_sign"] = np.sign(m["surprise_pct"].fillna(0)).where(m["surprise_pct"].notna())
        m["days_since_result"] = (m["date"] - m["report_date"]).dt.days
        m["in_result_window"] = (m["days_since_result"] <= _RESULT_WINDOW_DAYS).astype(float)
        m.loc[m["days_since_result"].isna(), "in_result_window"] = np.nan
        parts.append(m[["date", "symbol", *FUND_FEATURES]])

    return pd.concat(parts, ignore_index=True)


def fresh_mask(feat: pd.DataFrame, max_days: int = _FRESH_MAX_DAYS) -> pd.Series:
    """Rows with a recent (<= max_days) earnings report — the hp_003 'fresh fundamentals' subset."""
    return feat["days_since_result"].notna() & (feat["days_since_result"] <= max_days)
