"""
swing_scanner.py  —  Multi-Strategy Swing Scanner

Usage:
    python swing_scanner.py                                  # backtest + live signals (all)
    python swing_scanner.py backtest                         # backtest only
    python swing_scanner.py signals                          # live signals only
    python swing_scanner.py backtest TATASTEEL SBIN          # specific symbols
    python swing_scanner.py --strategy "Delivery + OI"       # specific strategy
    python swing_scanner.py backtest --strategy "Volume + EMA Cross" RELIANCE
"""

import os as _os
import sys
import logging
from datetime import datetime, timedelta

sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

# Force UTF-8 output on Windows so rupee symbol and other chars render correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

import src.config as cfg
from src.data_fetcher import DataFetcher
from src.strategies import all_strategies, get_strategy, list_strategy_names
from src.strategies.base import BacktestResult
from src.backtester import Backtester

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _next_trading_day(dt: datetime) -> datetime:
    """Advance past any weekend days."""
    nxt = dt + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def _fmt_inr(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}₹{val:,.2f}"


def _sep(char: str = "-", width: int = 72) -> str:
    return char * width


# ── Printing ──────────────────────────────────────────────────────────────────

def _print_trade_log(result: BacktestResult):
    if not result.trades:
        return
    print(f"\n  Trade log — {result.strategy_name} / {result.symbol}")
    print(f"  {'ENTRY DATE':<12} {'EXIT DATE':<12} {'ENTRY':>8} {'EXIT':>8} "
          f"{'SHARES':>7} {'P&L%':>7} {'GROSS P&L':>12} {'REASON'}")
    print("  " + _sep(width=85))
    for t in result.trades:
        print(
            f"  {t['entry_date']:<12} {t['exit_date']:<12} "
            f"₹{t['entry_price']:>7,.2f} ₹{t['exit_price']:>7,.2f} "
            f"{t['shares']:>7} {t['pnl_pct']:>+6.1f}% "
            f"{_fmt_inr(t['gross_pnl']):>12}  {t['exit_reason']}"
        )


def _print_backtest_summary(results: list[BacktestResult]):
    compare = Backtester.compare(results)
    summary = Backtester.summary_by_strategy(results)

    print()
    print(_sep("="))
    print("  BACKTEST SUMMARY")
    print(f"  Period  : last {cfg.BACKTEST_DAYS} calendar days")
    print(f"  Capital : ₹{cfg.CAPITAL:,}  |  "
          f"Position size : {cfg.POSITION_SIZE_PCT}%  |  "
          f"SL : {cfg.STOP_LOSS_PCT}%  |  Target : {cfg.TARGET_PCT}%")
    print(_sep("="))

    # Per (strategy, symbol) rows
    print()
    print("  By Strategy / Symbol:")
    hdr = (
        f"  {'STRATEGY':<22} {'SYMBOL':<12} {'TRADES':>6} {'WINS':>5} {'WIN%':>6} "
        f"{'TOTAL P&L':>12} {'RETURN%':>8} {'FINAL CAP':>14}"
    )
    print(hdr)
    print("  " + _sep(width=90))
    for _, row in compare.iterrows():
        pnl_str = _fmt_inr(row["total_pnl"])
        ret_str = f"{row['total_pnl_pct']:+.1f}%"
        cap_str = f"₹{row['final_capital']:,.0f}"
        print(
            f"  {row['strategy_name']:<22} {row['symbol']:<12} {int(row['total_trades']):>6} "
            f"{int(row['wins']):>5} {row['win_rate']:>5.1f}% {pnl_str:>12} "
            f"{ret_str:>8} {cap_str:>14}"
        )

    # Aggregate by strategy
    print()
    print("  Aggregate by Strategy:")
    hdr2 = (
        f"  {'STRATEGY':<22} {'TRADES':>6} {'WINS':>5} {'WIN%':>6} "
        f"{'TOTAL P&L':>12} {'AVG RETURN%':>12} {'MAX DD%':>8}"
    )
    print(hdr2)
    print("  " + _sep(width=80))
    for _, row in summary.iterrows():
        pnl_str = _fmt_inr(row["total_pnl"])
        print(
            f"  {row['strategy_name']:<22} {int(row['total_trades']):>6} "
            f"{int(row['wins']):>5} {row['win_rate']:>5.1f}% {pnl_str:>12} "
            f"{row['total_pnl_pct']:>+11.2f}% {row['max_drawdown_pct']:>7.1f}%"
        )
    print(_sep("="))


def _print_live_signals(signals: list[dict]):
    print()
    print(_sep("="))
    print("  LIVE BUY SIGNALS  -  Upcoming trading day")
    print(_sep("="))

    if not signals:
        print("  No BUY signals today.  All conditions must be met simultaneously.")
        print(_sep("="))
        return

    for s in signals:
        print(f"\n  >>> {s['symbol']}  [{s['strategy']}]")
        print(f"     Signal date     : {s['signal_date']}")
        print(f"     Suggested entry : {s['buy_date']}  (next trading day)")
        print(f"     Entry price     : ₹{s['close']:,.2f}")
        print(f"     Stop loss       : ₹{s['stop_loss']:,.2f}  (-{cfg.STOP_LOSS_PCT}%)")
        print(f"     Target          : ₹{s['target']:,.2f}  (+{cfg.TARGET_PCT}%)")
        print(f"     Conditions met  : {s['conditions_met']}/4")

        strat = s.get("strategy", "")

        if strat == "Delivery + OI":
            print(f"       [+] Delivery qty/% up : {s.get('cond_delivery_up', 'N/A')}")
            print(f"       [+] Close > VWAP      : {s.get('cond_price_above_vwap', 'N/A')}")
            print(f"       [+] OI increasing     : {s.get('cond_oi_increasing', 'N/A')}")
            c4 = s.get('cond_support_or_break', 'N/A')
            ns = s.get('near_support'); bk = s.get('is_breakout')
            print(f"       [+] Support/Breakout  : {c4}  (near_support={ns}, breakout={bk})")
            dqty = s.get('delivery_qty'); dpct = s.get('delivery_pct')
            vwap = s.get('vwap');         oi   = s.get('oi')
            print(f"     Delivery qty    : {dqty:,.0f}" if pd.notna(dqty) else "     Delivery qty    : N/A")
            print(f"     Delivery %      : {dpct:.1f}%" if pd.notna(dpct) else "     Delivery %      : N/A")
            print(f"     VWAP            : ₹{vwap:,.2f}" if pd.notna(vwap) else "     VWAP            : N/A")
            print(f"     Futures OI      : {oi:,.0f}" if pd.notna(oi) else "     Futures OI      : N/A")

        elif strat == "Volume + EMA Cross":
            print(f"       [+] EMA golden cross  : {s.get('cond_ema_cross', 'N/A')}")
            print(f"       [+] Volume surge      : {s.get('cond_vol_surge', 'N/A')}")
            print(f"       [+] Price above EMA   : {s.get('cond_ema_confirm', 'N/A')}")
            print(f"       [+] Bullish candle    : {s.get('cond_bullish', 'N/A')}")
            ef = s.get('ema_fast'); es = s.get('ema_slow'); va = s.get('vol_avg')
            if pd.notna(ef) and pd.notna(es):
                print(f"     EMA fast / slow : ₹{ef:,.2f} / ₹{es:,.2f}")
            if pd.notna(va):
                print(f"     Vol avg (20d)   : {va:,.0f}")

        elif strat == "EMA + Bollinger Bands":
            print(f"       [+] EMA aligned (7d)  : {s.get('cond_ema_aligned', 'N/A')}")
            print(f"       [+] BB pullback        : {s.get('cond_bb_pullback', 'N/A')}")
            ef = s.get('ema_fast'); es = s.get('ema_slow')
            bbl = s.get('bb_lower'); bbm = s.get('bb_mid'); bbu = s.get('bb_upper')
            if pd.notna(ef) and pd.notna(es):
                print(f"     EMA fast / slow : ₹{ef:,.2f} / ₹{es:,.2f}")
            if pd.notna(bbl) and pd.notna(bbu):
                print(f"     BB lower/mid/up : ₹{bbl:,.2f} / ₹{bbm:,.2f} / ₹{bbu:,.2f}")

    print()
    print(_sep("="))


# ── Main runners ──────────────────────────────────────────────────────────────

def run_backtest(
    symbols: list[str] | None = None,
    strategy_names: list[str] | None = None,
) -> list[BacktestResult]:
    strat_display = ", ".join(strategy_names or list_strategy_names())
    print(_sep("="))
    print("  SWING SCANNER  -  BACK-TEST MODE")
    print(f"  Date       : {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  Strategies : {strat_display}")
    print(_sep("="))

    stock_map = {
        k: v for k, v in cfg.STOCKS.items()
        if (symbols is None or k in symbols)
    }
    strategy_classes = (
        [get_strategy(n) for n in strategy_names]
        if strategy_names else all_strategies()
    )

    fetcher = DataFetcher(cfg)
    engine  = Backtester(cfg)
    all_results: list[BacktestResult] = []

    try:
        for symbol, info in stock_map.items():
            print(f"\n  [{symbol}]  fetching data…")
            df = fetcher.get_stock_data(
                symbol         = info["nse_symbol"],
                stockedge_id   = info["stockedge_id"],
                stockedge_slug = info["stockedge_slug"],
                has_futures    = info["has_futures"],
                days           = cfg.BACKTEST_DAYS + cfg.DATA_BUFFER_DAYS,
            )

            if df.empty:
                print(f"  ✗ Skipping {symbol}: no data available")
                continue

            for StrategyCls in strategy_classes:
                strategy = StrategyCls()
                result   = engine.run(strategy, df)
                all_results.append(result)
                print(f"    [{strategy.name}]  trades={result.total_trades}  "
                      f"win_rate={result.win_rate}%  pnl={_fmt_inr(result.total_pnl)}")
                _print_trade_log(result)

    finally:
        fetcher.close()

    if all_results:
        _print_backtest_summary(all_results)
    return all_results


def run_live_signals(
    symbols: list[str] | None = None,
    strategy_names: list[str] | None = None,
) -> list[dict]:
    print(_sep("="))
    print("  SWING SCANNER  -  LIVE SIGNAL MODE")
    print(f"  Scan date : {datetime.now():%Y-%m-%d %H:%M}")
    print(_sep("="))

    stock_map = {
        k: v for k, v in cfg.STOCKS.items()
        if (symbols is None or k in symbols)
    }
    strategy_classes = (
        [get_strategy(n) for n in strategy_names]
        if strategy_names else all_strategies()
    )

    fetcher = DataFetcher(cfg)
    buy_signals: list[dict] = []

    try:
        for symbol, info in stock_map.items():
            print(f"\n  Scanning {symbol}…")
            df = fetcher.get_stock_data(
                symbol         = info["nse_symbol"],
                stockedge_id   = info["stockedge_id"],
                stockedge_slug = info["stockedge_slug"],
                has_futures    = info["has_futures"],
                days           = cfg.SUPPORT_LOOKBACK_DAYS + cfg.DATA_BUFFER_DAYS + 10,
            )

            if df.empty or len(df) < 2:
                print(f"  ✗ Insufficient data for {symbol}")
                continue

            for StrategyCls in strategy_classes:
                strategy = StrategyCls()
                try:
                    sig_df = strategy.generate_signals(df.copy())
                except Exception as exc:
                    print(f"    [{strategy.name}] signal generation failed: {exc}")
                    continue

                if sig_df.empty or "buy_signal" not in sig_df.columns:
                    continue

                last = sig_df.iloc[-1]
                is_signal = last.get("buy_signal") is True or last.get("buy_signal") == 1
                conds = int(last.get("conditions_met", 0))

                if not is_signal:
                    print(f"    [{strategy.name}] No signal  (conditions met: {conds}/4)")
                    continue

                close = float(last["close"])
                sig = dict(
                    symbol         = symbol,
                    strategy       = strategy.name,
                    signal_date    = last["date"].strftime("%Y-%m-%d"),
                    buy_date       = _next_trading_day(last["date"]).strftime("%Y-%m-%d"),
                    close          = close,
                    stop_loss      = round(close * (1 - cfg.STOP_LOSS_PCT / 100), 2),
                    target         = round(close * (1 + cfg.TARGET_PCT / 100),   2),
                    conditions_met = conds,
                )

                # Attach strategy-specific diagnostic columns
                if strategy.name == "Delivery + OI":
                    for col in ["cond_delivery_up", "cond_price_above_vwap",
                                "cond_oi_increasing", "cond_support_or_break",
                                "near_support", "is_breakout",
                                "delivery_qty", "delivery_pct", "vwap", "oi", "oi_change"]:
                        sig[col] = last.get(col)
                elif strategy.name == "Volume + EMA Cross":
                    for col in ["cond_ema_cross", "cond_vol_surge", "cond_ema_confirm",
                                "cond_bullish", "ema_fast", "ema_slow", "vol_avg"]:
                        sig[col] = last.get(col)
                elif strategy.name == "EMA + Bollinger Bands":
                    for col in ["cond_ema_aligned", "cond_bb_pullback",
                                "ema_fast", "ema_slow", "bb_upper", "bb_mid", "bb_lower"]:
                        sig[col] = last.get(col)

                buy_signals.append(sig)
                print(f"    [{strategy.name}]  ◀ BUY SIGNAL")

    finally:
        fetcher.close()

    _print_live_signals(buy_signals)
    return buy_signals


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> tuple[str, list[str] | None, list[str] | None]:
    """
    Format:  swing_scanner.py [backtest|signals|both] [SYMBOL ...] [--strategy NAME ...]
    """
    args = sys.argv[1:]
    mode_choices = {"backtest", "signals", "both"}
    mode        = "both"
    symbols     = None
    strat_names = None

    # Extract --strategy NAME [NAME ...] anywhere in args
    if "--strategy" in args:
        idx = args.index("--strategy")
        strat_args: list[str] = []
        j = idx + 1
        while j < len(args) and not args[j].startswith("-"):
            strat_args.append(args[j])
            j += 1
        args = args[:idx] + args[j:]
        if strat_args:
            available = list_strategy_names()
            invalid = [s for s in strat_args if s not in available]
            if invalid:
                print(f"Unknown strategies: {invalid}")
                print(f"Available: {available}")
                sys.exit(1)
            strat_names = strat_args

    if args and args[0].lower() in mode_choices:
        mode = args[0].lower()
        rest = args[1:]
    else:
        rest = args

    if rest:
        symbols = [s.upper() for s in rest]
        invalid = [s for s in symbols if s not in cfg.STOCKS]
        if invalid:
            print(f"Unknown symbols: {invalid}")
            print(f"Available: {list(cfg.STOCKS.keys())}")
            sys.exit(1)

    return mode, symbols, strat_names


if __name__ == "__main__":
    mode, symbols, strat_names = _parse_args()

    print()
    print("=" * 66)
    print("       SWING SCANNER  -  Multi-Strategy Edition")
    print("=" * 66)
    print()
    print(f"  Data source : NSE India (+ StockEdge if USE_STOCKEDGE=True)")
    print(f"  Stocks      : {', '.join(symbols or list(cfg.STOCKS.keys()))}")
    print(f"  Strategies  : {', '.join(strat_names or list_strategy_names())}")
    print(f"  Capital     : ₹{cfg.CAPITAL:,}")
    print()

    if mode in ("backtest", "both"):
        run_backtest(symbols, strat_names)

    if mode in ("signals", "both"):
        run_live_signals(symbols, strat_names)
