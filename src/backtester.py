"""
backtester.py
-------------
Multi-strategy backtesting engine.

The Backtester class:
  - Runs any registered Strategy against one stock's DataFrame
  - Returns a standardised BacktestResult
  - Provides static helpers to compare and summarise results across
    multiple (strategy x symbol) combinations

Usage
-----
    import config as cfg
    from strategies import all_strategies, get_strategy
    from backtester import Backtester

    engine   = Backtester(cfg)
    strategy = get_strategy("Delivery + OI")()

    result   = engine.run(strategy, stock_df)
    print(result.total_pnl_pct)

    # Compare several strategies on the same stock
    results = [engine.run(cls(), stock_df) for cls in all_strategies()]
    print(Backtester.compare(results))
"""

from dataclasses import asdict

import numpy as np
import pandas as pd

from .strategies.base import BacktestResult, Strategy, TradeRecord


class Backtester:
    """
    Runs a Strategy on a stock DataFrame and returns a BacktestResult.

    Trade management parameters (stop-loss, target, position size, max hold)
    are read from ``cfg`` so they stay consistent across strategy comparisons.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, strategy: Strategy, df: pd.DataFrame) -> BacktestResult:
        """
        Parameters
        ----------
        strategy : Strategy instance (already instantiated)
        df       : stock DataFrame from DataFetcher (must include date + close)

        Returns
        -------
        BacktestResult with all metrics and a list of individual trade dicts.
        """
        symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns else "UNKNOWN"

        # Let the strategy enrich the DataFrame with buy_signal
        try:
            sig_df = strategy.generate_signals(df.copy())
        except Exception as exc:
            print(f"  [WARN] {strategy.name} signal generation failed for {symbol}: {exc}")
            return self._empty_result(strategy.name, symbol)

        if sig_df.empty or "buy_signal" not in sig_df.columns:
            return self._empty_result(strategy.name, symbol)

        # Simulate trades
        trades, final_cap = self._simulate(sig_df, symbol, strategy.name)

        return self._build_result(strategy.name, symbol, trades, final_cap)

    # ── simulation ───────────────────────────────────────────────────────────

    def _simulate(
        self, sig_df: pd.DataFrame, symbol: str, strategy_name: str
    ) -> tuple[list[TradeRecord], float]:
        cfg         = self.cfg
        capital     = float(cfg.CAPITAL)
        pos_size    = capital * cfg.POSITION_SIZE_PCT / 100
        trades: list[TradeRecord] = []

        in_trade    = False
        entry_price = 0.0
        entry_date  = None
        hold_days   = 0
        running_cap = capital

        for i in range(len(sig_df)):
            row = sig_df.iloc[i]
            if pd.isna(row.get("close")):
                continue

            if in_trade:
                hold_days += 1
                cur      = float(row["close"])
                pnl_pct  = (cur - entry_price) / entry_price * 100
                reason   = None

                if   pnl_pct <= -cfg.STOP_LOSS_PCT:   reason = "STOP_LOSS"
                elif pnl_pct >=  cfg.TARGET_PCT:       reason = "TARGET_HIT"
                elif hold_days >= cfg.MAX_HOLD_DAYS:   reason = "MAX_HOLD"

                if reason:
                    shares       = int(pos_size / entry_price)
                    gross        = shares * (cur - entry_price)
                    running_cap += gross
                    trades.append(TradeRecord(
                        symbol      = symbol,
                        strategy    = strategy_name,
                        entry_date  = entry_date.strftime("%Y-%m-%d"),
                        exit_date   = row["date"].strftime("%Y-%m-%d"),
                        entry_price = round(entry_price, 2),
                        exit_price  = round(cur, 2),
                        shares      = shares,
                        pnl_pct     = round(pnl_pct, 2),
                        gross_pnl   = round(gross, 2),
                        hold_days   = hold_days,
                        exit_reason = reason,
                    ))
                    in_trade = False

            elif row.get("buy_signal") is True or row.get("buy_signal") == 1:
                ep = float(row["close"])
                if ep > 0 and int(pos_size / ep) == 0:
                    continue  # can't afford even 1 share; skip signal
                entry_price = ep
                entry_date  = row["date"]
                in_trade    = True
                hold_days   = 0

        # Close any still-open trade at the last available price
        if in_trade and len(sig_df):
            last         = sig_df.iloc[-1]
            cur          = float(last["close"]) if pd.notna(last.get("close")) else entry_price
            pnl_pct      = (cur - entry_price) / entry_price * 100
            shares       = int(pos_size / entry_price)
            gross        = shares * (cur - entry_price)
            running_cap += gross
            trades.append(TradeRecord(
                symbol      = symbol,
                strategy    = strategy_name,
                entry_date  = entry_date.strftime("%Y-%m-%d"),
                exit_date   = last["date"].strftime("%Y-%m-%d"),
                entry_price = round(entry_price, 2),
                exit_price  = round(cur, 2),
                shares      = shares,
                pnl_pct     = round(pnl_pct, 2),
                gross_pnl   = round(gross, 2),
                hold_days   = hold_days,
                exit_reason = "OPEN_AT_END",
            ))

        return trades, running_cap

    # ── result builders ───────────────────────────────────────────────────────

    def _build_result(
        self,
        strategy_name: str,
        symbol: str,
        trades: list[TradeRecord],
        final_cap: float,
    ) -> BacktestResult:
        capital = float(self.cfg.CAPITAL)

        if not trades:
            return self._empty_result(strategy_name, symbol)

        tdf   = pd.DataFrame([asdict(t) for t in trades])
        wins  = tdf[tdf["gross_pnl"] > 0]

        # Max drawdown: peak-to-trough on cumulative P&L curve
        running  = capital + tdf["gross_pnl"].cumsum()
        peak     = running.cummax()
        max_dd   = float(((peak - running) / peak * 100).max())

        return BacktestResult(
            strategy_name     = strategy_name,
            symbol            = symbol,
            period_days       = self.cfg.BACKTEST_DAYS,
            initial_capital   = capital,
            total_trades      = len(tdf),
            wins              = len(wins),
            losses            = len(tdf) - len(wins),
            win_rate          = round(len(wins) / len(tdf) * 100, 1),
            total_pnl         = round(float(tdf["gross_pnl"].sum()), 2),
            total_pnl_pct     = round((final_cap - capital) / capital * 100, 2),
            avg_pnl_per_trade = round(float(tdf["pnl_pct"].mean()), 2),
            best_trade_pnl    = round(float(tdf["gross_pnl"].max()), 2),
            worst_trade_pnl   = round(float(tdf["gross_pnl"].min()), 2),
            max_drawdown_pct  = round(max_dd, 2),
            avg_hold_days     = round(float(tdf["hold_days"].mean()), 1),
            final_capital     = round(final_cap, 2),
            trades            = [asdict(t) for t in trades],
        )

    def _empty_result(self, strategy_name: str, symbol: str) -> BacktestResult:
        capital = float(self.cfg.CAPITAL)
        return BacktestResult(
            strategy_name=strategy_name, symbol=symbol,
            period_days=self.cfg.BACKTEST_DAYS, initial_capital=capital,
            total_trades=0, wins=0, losses=0, win_rate=0.0,
            total_pnl=0.0, total_pnl_pct=0.0, avg_pnl_per_trade=0.0,
            best_trade_pnl=0.0, worst_trade_pnl=0.0,
            max_drawdown_pct=0.0, avg_hold_days=0.0,
            final_capital=capital, trades=[],
        )

    # ── comparison helpers ────────────────────────────────────────────────────

    @staticmethod
    def compare(results: list[BacktestResult]) -> pd.DataFrame:
        """
        One row per (strategy, symbol).  Sorted by strategy then symbol.
        """
        if not results:
            return pd.DataFrame()

        cols_order = [
            "strategy_name", "symbol",
            "total_trades", "wins", "losses", "win_rate",
            "total_pnl", "total_pnl_pct", "avg_pnl_per_trade",
            "best_trade_pnl", "worst_trade_pnl",
            "max_drawdown_pct", "avg_hold_days", "final_capital",
        ]
        df = pd.DataFrame([r.to_dict() for r in results])
        df = df[[c for c in cols_order if c in df.columns]]
        return df.sort_values(["strategy_name", "symbol"]).reset_index(drop=True)

    @staticmethod
    def summary_by_strategy(results: list[BacktestResult]) -> pd.DataFrame:
        """
        Aggregate all symbols for each strategy into one summary row.
        P&L figures are summed; rates and averages are mean-aggregated.
        """
        comp = Backtester.compare(results)
        if comp.empty:
            return comp

        agg_rules: dict[str, str] = {
            "total_trades":      "sum",
            "wins":              "sum",
            "losses":            "sum",
            "win_rate":          "mean",
            "total_pnl":         "sum",
            "total_pnl_pct":     "mean",
            "avg_pnl_per_trade": "mean",
            "best_trade_pnl":    "max",
            "worst_trade_pnl":   "min",
            "max_drawdown_pct":  "max",
            "avg_hold_days":     "mean",
            "final_capital":     "sum",
        }
        valid = {k: v for k, v in agg_rules.items() if k in comp.columns}
        summary = (
            comp.groupby("strategy_name", sort=False)
            .agg(valid)
            .reset_index()
        )
        # Re-derive win_rate from aggregated counts
        if "wins" in summary.columns and "total_trades" in summary.columns:
            summary["win_rate"] = (
                summary["wins"] / summary["total_trades"].replace(0, np.nan) * 100
            ).round(1)

        for col in ["total_pnl_pct", "avg_pnl_per_trade",
                    "max_drawdown_pct", "avg_hold_days"]:
            if col in summary.columns:
                summary[col] = summary[col].round(2)

        return summary.sort_values("total_pnl", ascending=False).reset_index(drop=True)
