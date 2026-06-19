"""
src.predictor.data
------------------
Universe + benchmark loading for the predictor, reusing the same cached parquet
store and halal filter as scripts/run_experiments.py (do not diverge from it).

Pure loading + light validation; no feature logic (that's features.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
FETCHED_DIR = ROOT / "data" / "fetched"
STOCKS_CSV = ROOT / "data" / "stocks.csv"
NIFTY_PARQUET = FETCHED_DIR / "_NIFTY.parquet"

MIN_ROWS = 120          # need enough history for 200-window indicators + a test tail
MIN_PRICE = 20.0        # drop sub-₹20 names (penny noise, matches run_experiments)

# Columns every bar parquet is expected to carry (optional ones tolerated absent).
CORE_COLS = ["date", "open", "high", "low", "close", "total_volume"]
OPT_COLS = ["delivery_qty", "delivery_pct", "vwap", "oi", "oi_change"]


def halal_symbols() -> set[str]:
    """Symbols flagged halal='Y' in stocks.csv (the project's default universe)."""
    stocks = pd.read_csv(STOCKS_CSV, dtype=str).fillna("")
    return set(stocks.loc[stocks["halal"].str.upper() == "Y", "stock"].str.strip())


def sector_map() -> dict[str, str]:
    """symbol -> sector (for diversification caps / UI grouping)."""
    stocks = pd.read_csv(STOCKS_CSV, dtype=str).fillna("")
    return dict(zip(stocks["stock"].str.strip(), stocks["sector"].str.strip()))


def load_universe(universe: str = "halal", asof: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Load per-symbol daily bars into memory once.

    universe : "halal" (default) or "all".
    asof     : optional ISO date; rows after it are dropped so a historical run
               sees exactly what was knowable on that date (point-in-time replay).
    """
    wanted = halal_symbols() if universe == "halal" else None
    cutoff = pd.Timestamp(asof) if asof else None

    data: dict[str, pd.DataFrame] = {}
    skipped_short = skipped_penny = 0
    for p in sorted(FETCHED_DIR.glob("*.parquet")):
        if p.stem.startswith("_"):           # _NIFTY, _UNIVERSE_INDEX are not tradables
            continue
        if wanted is not None and p.stem not in wanted:
            continue
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        if cutoff is not None:
            df = df[df["date"] <= cutoff]
        if len(df) < MIN_ROWS:
            skipped_short += 1
            continue
        if float(df["close"].median()) < MIN_PRICE:
            skipped_penny += 1
            continue
        df = df.sort_values("date").reset_index(drop=True)
        df["symbol"] = p.stem
        data[p.stem] = df

    print(f"  Universe '{universe}': {len(data)} symbols loaded "
          f"(skipped {skipped_short} short-history, {skipped_penny} sub-Rs{MIN_PRICE:.0f})"
          + (f"; as-of {asof}" if asof else ""))
    return data


def load_nifty(asof: str | None = None) -> pd.DataFrame:
    """NIFTY benchmark (date, close) for regime / market-relative features."""
    if not NIFTY_PARQUET.exists():
        return pd.DataFrame(columns=["date", "close"])
    df = pd.read_parquet(NIFTY_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    if asof:
        df = df[df["date"] <= pd.Timestamp(asof)]
    return df.sort_values("date").reset_index(drop=True)
