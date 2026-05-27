"""
strategies/base.py
------------------
Abstract base class for all trading strategies, plus shared data structures.

Design:
  - StrategyParams  : dataclass for signal-generation parameters (strategy-specific)
  - Strategy        : ABC with a single required method: generate_signals(df) -> df
  - TradeRecord     : immutable record of one completed trade
  - BacktestResult  : aggregated stats for one (strategy, symbol) backtest run

Trade management parameters (SL %, target %, position size) live in config.py
and are intentionally NOT part of the strategy, so the same set of strategies
can be compared under identical risk rules.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import pandas as pd


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class StrategyParams:
    """
    Base class for strategy parameters.
    Subclass and add fields for each strategy's specific knobs.
    Being a dataclass makes params serialisable to JSON for reproducibility.
    """
    name: str
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    symbol: str
    strategy: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl_pct: float
    gross_pnl: float
    hold_days: int
    exit_reason: str


# ── Backtest result ───────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Standardised output for one (strategy × symbol) backtest run."""
    strategy_name: str
    symbol: str
    period_days: int
    initial_capital: float
    # --- trade stats ---
    total_trades: int
    wins: int
    losses: int
    win_rate: float           # %
    # --- P&L ---
    total_pnl: float          # INR
    total_pnl_pct: float      # % return on initial capital
    avg_pnl_per_trade: float  # avg % gain per trade
    best_trade_pnl: float     # INR
    worst_trade_pnl: float    # INR
    # --- risk ---
    max_drawdown_pct: float
    avg_hold_days: float
    final_capital: float
    # --- raw trades (excluded from summary views) ---
    trades: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return scalar fields only (no trade list) for comparison tables."""
        d = asdict(self)
        d.pop("trades")
        return d

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ── Strategy ABC ──────────────────────────────────────────────────────────────

class Strategy(ABC):
    """
    Abstract base for all trading strategies.

    Contract
    --------
    Every subclass must:
      1. Define a CLASS-LEVEL ``params`` attribute (a StrategyParams instance).
      2. Implement ``generate_signals(df)`` which enriches the DataFrame with
         at least a boolean ``buy_signal`` column.

    The strategy is responsible ONLY for signal generation.
    Trade management (stop-loss, target, hold limit) is handled by Backtester.

    Example
    -------
    >>> @register_strategy
    ... class MyStrategy(Strategy):
    ...     params = MyParams(name="My Strategy")
    ...
    ...     def generate_signals(self, df):
    ...         df = df.copy()
    ...         df["buy_signal"] = df["close"] > df["close"].shift(1)
    ...         return df.iloc[1:]
    """

    # Subclasses must set this at class level
    params: StrategyParams

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def description(self) -> str:
        return self.params.description

    # ── interface ─────────────────────────────────────────────────────────────

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add signal columns to ``df`` and return the enriched DataFrame.

        Required output column
        ~~~~~~~~~~~~~~~~~~~~~~
        ``buy_signal`` (bool) – True on days where entry is recommended.

        Recommended diagnostic columns (for printing and analysis)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``conditions_met`` (int) – how many sub-conditions were satisfied.
        Individual condition booleans (e.g. ``cond_delivery_up``).

        Notes
        ~~~~~
        - May return fewer rows than the input (e.g. after indicator warm-up).
        - Should not modify the caller's DataFrame; work on a copy.
        - Missing data for a condition should cause that condition = False,
          never raise an exception.
        """

    def required_columns(self) -> list[str]:
        """
        Documents which input columns this strategy uses.
        Not enforced at runtime; used for documentation and validation helpers.
        """
        return ["date", "open", "high", "low", "close", "total_volume"]

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
