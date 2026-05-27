"""
strategies/delivery_oi.py
-------------------------
Delivery Volume + Futures OI strategy.

BUY signal when ALL four conditions are simultaneously true:
  C1  Delivery quantity AND delivery % are higher than the previous day
  C2  Closing price is above the day's VWAP
  C3  Futures Open Interest is positive AND higher than the previous day
  C4  Price is near a support level OR is breaking out from a recent range

Required columns: date, open, high, low, close, total_volume,
                  delivery_qty, delivery_pct, vwap, oi, oi_change
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import register_strategy
from .base import Strategy, StrategyParams


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class DeliveryOIParams(StrategyParams):
    name: str = "Delivery + OI"
    description: str = (
        "BUY when delivery volume & % rise, price > VWAP, "
        "futures OI increases, and price is near support or breaking out."
    )
    # support / resistance
    support_lookback: int = 30       # candles to look back for swing lows
    support_proximity_pct: float = 2.0  # within X% of support = "near support"
    # breakout
    breakout_lookback: int = 20      # candles to look back for recent high
    breakout_buffer_pct: float = 0.5 # price >= recent_high * (1 - X%) = breakout
    # OI handling
    require_oi: bool = True          # False = generate signal even when OI data missing


# ── Support / resistance helpers ──────────────────────────────────────────────

def _support_levels(df: pd.DataFrame, lookback: int, window: int = 4) -> list[float]:
    """Fractal swing lows + classic pivot S1/S2 over the last ``lookback`` bars."""
    recent = df.tail(lookback)
    if len(recent) < window * 2 + 2:
        return []

    lows   = recent["low"].values
    levels: list[float] = []

    for i in range(window, len(lows) - window):
        if (all(lows[i] <= lows[i - j] for j in range(1, window + 1))
                and all(lows[i] <= lows[i + j] for j in range(1, window + 1))):
            levels.append(float(lows[i]))

    h     = float(recent["high"].max())
    lo    = float(recent["low"].min())
    c     = float(recent["close"].iloc[-1])
    pivot = (h + lo + c) / 3
    levels += [2 * pivot - h, pivot - (h - lo)]

    # Deduplicate levels within 1% of each other
    clean: list[float] = []
    for lvl in sorted(set(levels)):
        if lvl > 0 and not any(abs(lvl - e) / e < 0.01 for e in clean):
            clean.append(lvl)
    return clean


def _near_support(price: float, levels: list[float], pct: float) -> bool:
    return any(lvl > 0 and abs(price - lvl) / lvl * 100 <= pct for lvl in levels)


def _is_breakout(price: float, df: pd.DataFrame, lookback: int, buf: float) -> bool:
    if len(df) < lookback:
        return False
    recent_high = float(df["high"].tail(lookback).max())
    return price >= recent_high * (1 - buf / 100)


# ── Strategy ──────────────────────────────────────────────────────────────────

@register_strategy
class DeliveryOIStrategy(Strategy):
    """Delivery + Futures OI swing strategy (row-by-row evaluation)."""

    params = DeliveryOIParams()

    def required_columns(self) -> list[str]:
        return [
            "date", "open", "high", "low", "close", "total_volume",
            "delivery_qty", "delivery_pct", "vwap", "oi", "oi_change",
        ]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signal_rows = []

        for i in range(1, len(df)):
            row  = df.iloc[i]
            prev = df.iloc[i - 1]
            hist = df.iloc[: i + 1]

            # C1 – delivery qty AND delivery % both higher than previous day
            c1 = bool(
                pd.notna(row.get("delivery_qty")) and pd.notna(prev.get("delivery_qty"))
                and float(row["delivery_qty"]) > float(prev["delivery_qty"])
                and pd.notna(row.get("delivery_pct")) and pd.notna(prev.get("delivery_pct"))
                and float(row["delivery_pct"]) > float(prev["delivery_pct"])
            )

            # C2 – close > VWAP
            c2 = bool(
                pd.notna(row.get("close")) and pd.notna(row.get("vwap"))
                and float(row["close"]) > float(row["vwap"])
            )

            # C3 – futures OI positive and growing
            oi_now  = row.get("oi")
            oi_prev = prev.get("oi")
            oi_ok   = pd.notna(oi_now) and float(oi_now) > 0
            if oi_ok:
                c3 = bool(pd.notna(oi_prev) and float(oi_now) > float(oi_prev))
            elif not p.require_oi:
                c3 = True   # waive when data missing and config permits
            else:
                c3 = False

            # C4 – near support or breaking out
            levels   = _support_levels(hist, p.support_lookback)
            near_sup = _near_support(float(row["close"]), levels, p.support_proximity_pct)
            breakout = _is_breakout(float(row["close"]), hist,
                                    p.breakout_lookback, p.breakout_buffer_pct)
            c4 = near_sup or breakout

            signal_rows.append({
                "date":                  row["date"],
                "buy_signal":            c1 and c2 and c3 and c4,
                "conditions_met":        int(c1 + c2 + c3 + c4),
                "cond_delivery_up":      c1,
                "cond_price_above_vwap": c2,
                "cond_oi_increasing":    c3,
                "cond_support_or_break": c4,
                "near_support":          near_sup,
                "is_breakout":           breakout,
                "oi_available":          oi_ok,
            })

        if not signal_rows:
            return df

        sig_df  = pd.DataFrame(signal_rows)
        extra   = [c for c in sig_df.columns if c != "date"]
        base    = df.iloc[1:].reset_index(drop=True)
        return base.merge(sig_df[["date"] + extra], on="date", how="left")
