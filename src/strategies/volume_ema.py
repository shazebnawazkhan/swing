"""
strategies/volume_ema.py
------------------------
Volume Surge + EMA Crossover strategy.

BUY signal when ALL four conditions are true:
  C1  Fast EMA crosses above slow EMA today (golden cross)
  C2  Today's volume > surge_mult x rolling average volume
  C3  Close is above the slow EMA (trend confirmation)
  C4  Today is a bullish candle (close > open)

This strategy works on OHLCV data alone -- no delivery % or futures OI
required.  It acts as a useful baseline comparison against the Delivery+OI
strategy and will generate signals even when delivery data is unavailable
(e.g. when yfinance fallback is used).

Required columns: date, open, high, low, close, total_volume
"""

from dataclasses import dataclass

import pandas as pd

from . import register_strategy
from .base import Strategy, StrategyParams


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class VolumeEMAParams(StrategyParams):
    name: str = "Volume + EMA Cross"
    description: str = (
        "BUY on EMA golden cross with a volume surge and bullish candle. "
        "Uses OHLCV only -- no delivery or futures data required."
    )
    fast_ema: int = 9
    slow_ema: int = 21
    volume_avg_period: int = 20
    surge_mult: float = 1.5      # volume must exceed this multiple of its rolling average
    min_above_ema_pct: float = 0.0  # price must be >= slow_ema * (1 + X/100)


# ── Strategy ──────────────────────────────────────────────────────────────────

@register_strategy
class VolumeEMAStrategy(Strategy):
    """Volume + EMA crossover strategy (fully vectorised, pandas-based)."""

    params = VolumeEMAParams()

    def required_columns(self) -> list[str]:
        return ["date", "open", "high", "low", "close", "total_volume"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p   = self.params
        out = df.copy()

        # Indicators (computed on full series for accuracy)
        out["_ema_fast"]  = out["close"].ewm(span=p.fast_ema,       adjust=False).mean()
        out["_ema_slow"]  = out["close"].ewm(span=p.slow_ema,       adjust=False).mean()
        out["_vol_avg"]   = out["total_volume"].rolling(p.volume_avg_period).mean()
        out["_prev_fast"] = out["_ema_fast"].shift(1)
        out["_prev_slow"] = out["_ema_slow"].shift(1)

        # Expose useful indicator columns for diagnostics
        out["ema_fast"] = out["_ema_fast"].round(2)
        out["ema_slow"] = out["_ema_slow"].round(2)
        out["vol_avg"]  = out["_vol_avg"].round(0)

        # C1 – EMA golden cross (fast crossed above slow since yesterday)
        c1 = (out["_ema_fast"] > out["_ema_slow"]) & (out["_prev_fast"] <= out["_prev_slow"])

        # C2 – Volume surge
        c2 = out["total_volume"] > (p.surge_mult * out["_vol_avg"])

        # C3 – Price above slow EMA (confirming uptrend)
        c3 = out["close"] >= out["_ema_slow"] * (1 + p.min_above_ema_pct / 100)

        # C4 – Bullish candle
        c4 = out["close"] > out["open"]

        out["cond_ema_cross"]    = c1
        out["cond_vol_surge"]    = c2
        out["cond_ema_confirm"]  = c3
        out["cond_bullish"]      = c4
        out["conditions_met"]    = c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int)
        out["buy_signal"]        = c1 & c2 & c3 & c4

        # Drop internal working columns
        out = out.drop(columns=["_ema_fast", "_ema_slow", "_vol_avg",
                                 "_prev_fast", "_prev_slow"], errors="ignore")

        # Trim warm-up rows where slow EMA and volume average are not yet stable
        warmup = max(p.slow_ema, p.volume_avg_period)
        return out.iloc[warmup:].reset_index(drop=True)
