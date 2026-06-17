"""
metrics.py
----------
Strategy-evaluation metrics computed from a list of completed trades.

Pure functions, no I/O.  Used by the experiment runner (scripts/run_experiments.py)
and anywhere a BacktestResult needs enriching.

Note on Sharpe: trades are discrete (no daily mark-to-market here), so Sharpe is
approximated from per-trade returns annualised by average holding period:
    sharpe ≈ mean(r) / std(r) * sqrt(252 / avg_hold_days)
This is comparable ACROSS strategies evaluated the same way, which is what the
research loop needs; it is not directly comparable to a daily-NAV Sharpe.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def trade_metrics(trades: pd.DataFrame, capital: float) -> dict:
    """
    Compute evaluation metrics from a trades DataFrame.

    Required columns: pnl_pct (net %, per trade), gross_pnl (net INR),
                      hold_days, exit_date (sortable).
    Returns a flat dict; all-zero dict when there are no trades.
    """
    if trades is None or len(trades) == 0:
        return _empty()

    t = trades.sort_values("exit_date").reset_index(drop=True)
    r = t["pnl_pct"].to_numpy(dtype=float) / 100.0
    pnl = t["gross_pnl"].to_numpy(dtype=float)

    wins = pnl > 0
    gross_win = float(pnl[wins].sum())
    gross_loss = float(-pnl[~wins].sum())

    # Max drawdown on the pooled, exit-date-ordered cumulative P&L curve
    equity = capital + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    max_dd = float(((peak - equity) / peak * 100).max())

    avg_hold = float(t["hold_days"].mean())
    sharpe = 0.0
    if len(r) > 1 and r.std(ddof=1) > 0 and avg_hold > 0:
        sharpe = float(
            r.mean() / r.std(ddof=1)
            * np.sqrt(TRADING_DAYS_PER_YEAR / max(avg_hold, 1.0))
        )

    return {
        "trades":          int(len(t)),
        "wins":            int(wins.sum()),
        "win_rate":        round(float(wins.mean()) * 100, 1),
        "total_pnl":       round(float(pnl.sum()), 2),
        "total_pnl_pct":   round(float(pnl.sum()) / capital * 100, 2),
        "expectancy_pct":  round(float(r.mean()) * 100, 3),
        "profit_factor":   round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_approx":   round(sharpe, 2),
        "avg_hold_days":   round(avg_hold, 1),
        "best_trade_pct":  round(float(t["pnl_pct"].max()), 2),
        "worst_trade_pct": round(float(t["pnl_pct"].min()), 2),
    }


def _empty() -> dict:
    return {
        "trades": 0, "wins": 0, "win_rate": 0.0,
        "total_pnl": 0.0, "total_pnl_pct": 0.0, "expectancy_pct": 0.0,
        "profit_factor": 0.0, "max_drawdown_pct": 0.0, "sharpe_approx": 0.0,
        "avg_hold_days": 0.0, "best_trade_pct": 0.0, "worst_trade_pct": 0.0,
    }


def market_adjusted_metrics(trades: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    """
    Index-adjusted performance (Kissell, *Algorithmic Trading Methods* ch.3).

    For each trade, alpha = (net trade return) − (benchmark return over the trade's
    holding window).  This separates skill from market beta: a strategy that lost
    less than the market over its trade windows has *positive* alpha even with a
    raw profit factor < 1.

    Required trade columns: pnl_pct (net %), entry_date, exit_date (YYYY-MM-DD).
    `benchmark` has columns date, close.  Returns a dict prefixed with `alpha_`.
    """
    if trades is None or len(trades) == 0:
        return _empty_alpha()

    from src.benchmark import benchmark_return        # local import avoids cycle

    t = trades.sort_values("exit_date").reset_index(drop=True)
    bench_rets = np.array([
        benchmark_return(benchmark, e, x)
        for e, x in zip(t["entry_date"], t["exit_date"])
    ], dtype=float)
    trade_rets = t["pnl_pct"].to_numpy(dtype=float)
    alpha = trade_rets - bench_rets                   # excess return over market, %

    a = alpha / 100.0
    beat = alpha > 0
    gross_pos = float(alpha[beat].sum())
    gross_neg = float(-alpha[~beat].sum())
    avg_hold = float(t["hold_days"].mean()) if "hold_days" in t else 1.0

    info_ratio = 0.0
    if len(a) > 1 and a.std(ddof=1) > 0 and avg_hold > 0:
        info_ratio = float(
            a.mean() / a.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR / max(avg_hold, 1.0))
        )

    return {
        "alpha_expectancy_pct": round(float(alpha.mean()), 3),
        "alpha_total_pct":      round(float(alpha.sum()), 2),
        "pct_beating_market":   round(float(beat.mean()) * 100, 1),
        "alpha_profit_factor":  round(gross_pos / gross_neg, 2) if gross_neg > 0 else float("inf"),
        "info_ratio_approx":    round(info_ratio, 2),
        "avg_bench_ret_pct":    round(float(bench_rets.mean()), 3),
    }


def _empty_alpha() -> dict:
    return {
        "alpha_expectancy_pct": 0.0, "alpha_total_pct": 0.0,
        "pct_beating_market": 0.0, "alpha_profit_factor": 0.0,
        "info_ratio_approx": 0.0, "avg_bench_ret_pct": 0.0,
    }


def apply_round_trip_costs(pnl_pcts: np.ndarray, cost_pct: float) -> np.ndarray:
    """Deduct round-trip transaction costs (brokerage+STT+slippage) from raw trade returns."""
    return pnl_pcts - cost_pct
