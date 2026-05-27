"""
strategies/ema_bb.py
--------------------
EMA Alignment + Bollinger Band Pullback strategy.

BUY signal when ALL conditions are true:
  C1  EMA_fast has been consistently above EMA_slow for the last
      ``alignment_bars`` candles (trend confirmation)
  C2  Close touches or dips below the lower Bollinger Band
      (pullback / mean-reversion entry)

Adapted from the intraday MyStrat (EMA 10/30 + BB 15/1.5) in backtesting.py
to daily OHLCV data.  Uses pandas EWM and rolling std -- no talib required.

Required columns: date, open, high, low, close, total_volume
"""

from dataclasses import dataclass

import pandas as pd

from . import register_strategy
from .base import Strategy, StrategyParams


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class EMABollingerParams(StrategyParams):
    name: str = "EMA + Bollinger Bands"
    description: str = (
        "BUY when EMA fast is consistently above EMA slow (trend) "
        "AND close touches or dips below the lower Bollinger Band (pullback entry)."
    )
    fast_ema: int = 10
    slow_ema: int = 30
    bb_period: int = 15
    bb_std: float = 1.5
    alignment_bars: int = 7   # consecutive bars EMA_fast must be above EMA_slow


# ── Strategy ──────────────────────────────────────────────────────────────────

@register_strategy
class EMABollingerStrategy(Strategy):
    """EMA alignment + Bollinger Band pullback strategy (daily OHLCV, no talib)."""

    params = EMABollingerParams()

    def required_columns(self) -> list[str]:
        return ["date", "open", "high", "low", "close", "total_volume"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p   = self.params
        out = df.copy()

        # ── Indicators ─────────────────────────────────────────────────────────
        out["_ema_fast"] = out["close"].ewm(span=p.fast_ema, adjust=False).mean()
        out["_ema_slow"] = out["close"].ewm(span=p.slow_ema, adjust=False).mean()

        bb_mid          = out["close"].rolling(p.bb_period).mean()
        bb_std_ser      = out["close"].rolling(p.bb_period).std(ddof=0)
        out["bb_upper"] = (bb_mid + p.bb_std * bb_std_ser).round(2)
        out["bb_mid"]   = bb_mid.round(2)
        out["bb_lower"] = (bb_mid - p.bb_std * bb_std_ser).round(2)

        # Expose for diagnostics / live signal display
        out["ema_fast"] = out["_ema_fast"].round(2)
        out["ema_slow"] = out["_ema_slow"].round(2)

        # ── C1 — EMA consistently aligned over rolling window ────────────────
        # rolling(alignment_bars).min() == 1 means every bar in the window
        # had fast > slow
        ema_above = (out["_ema_fast"] > out["_ema_slow"]).astype(int)
        c1 = ema_above.rolling(p.alignment_bars).min() == 1

        # ── C2 — Close at or below lower Bollinger Band (pullback entry) ─────
        c2 = out["close"] <= out["bb_lower"]

        out["cond_ema_aligned"] = c1
        out["cond_bb_pullback"] = c2
        out["conditions_met"]   = c1.astype(int) + c2.astype(int)
        out["buy_signal"]       = c1 & c2

        out = out.drop(columns=["_ema_fast", "_ema_slow"], errors="ignore")

        warmup = max(p.slow_ema, p.bb_period, p.alignment_bars)
        return out.iloc[warmup:].reset_index(drop=True)
