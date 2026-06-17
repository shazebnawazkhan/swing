"""Seed the first 10 experiment strategy specs into data/strategies/.

Each spec is validated before writing. Re-running overwrites in place (specs
are still draft — no experiment has referenced them yet).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategies.spec import SCHEMA_VERSION, SPEC_DIR, validate_spec

CREATED = "2026-06-12"


def spec(id_, name, desc, params, indicators, entry, exit_, regimes=None):
    return {
        "schema": SCHEMA_VERSION, "id": id_, "name": name, "description": desc,
        "markets": ["nse"], "regimes": regimes or ["bull", "sideways"],
        "params": params, "indicators": indicators,
        "entry": entry, "exit": exit_,
        "provenance": {"created": CREATED, "hypothesis_id": None,
                       "parent_id": None, "experiments": []},
    }


SPECS = [
    spec(
        "delivery_breakout_v1", "Delivery Breakout v1",
        "20-day price breakout confirmed by abnormally high delivery percentage "
        "(strong hands taking delivery, not intraday churn).",
        {"del_mult": 1.5, "lookback": 20},
        {
            "hhv20":      {"fn": "hhv", "on": "close", "window": "@lookback"},
            "hhv20_prev": {"fn": "shift", "on": "hhv20", "n": 1},
            "del_avg20":  {"fn": "sma", "on": "delivery_pct", "window": 20},
        },
        {"all": ["close > hhv20_prev", "delivery_pct > @del_mult * del_avg20"]},
        {"stop_loss_pct": 5, "target_pct": 10, "max_hold_days": 10},
    ),
    spec(
        "rsi2_meanrev_v1", "RSI2 Mean Reversion v1",
        "Buy 2-day RSI oversold dips inside a medium-term uptrend; quick exit.",
        {"rsi_buy": 10, "trend_span": 100},
        {
            "rsi2":   {"fn": "rsi", "on": "close", "period": 2},
            "ema100": {"fn": "ema", "on": "close", "span": "@trend_span"},
        },
        {"all": ["rsi2 < @rsi_buy", "close > ema100"]},
        {"stop_loss_pct": 5, "target_pct": 6, "max_hold_days": 6},
        regimes=["sideways", "bull"],
    ),
    spec(
        "pullback_trend_v1", "Pullback Trend v1",
        "Uptrend (EMA20>EMA50), price pulls back to touch EMA20, closes bullish.",
        {},
        {
            "ema20": {"fn": "ema", "on": "close", "span": 20},
            "ema50": {"fn": "ema", "on": "close", "span": 50},
        },
        {"all": ["ema20 > ema50", "low <= ema20", "close > open"]},
        {"stop_loss_pct": 4, "target_pct": 8, "max_hold_days": 10},
        regimes=["bull"],
    ),
    spec(
        "bb_squeeze_breakout_v1", "BB Squeeze Breakout v1",
        "Bollinger band width near a 60-day low (volatility squeeze), then close "
        "above the upper band on elevated volume.",
        {"vol_mult": 1.5, "squeeze_tol": 1.2},
        {
            "bbw":          {"fn": "bb_width", "on": "close", "window": 20, "k": 2},
            "bbw_min60":    {"fn": "llv", "on": "bbw", "window": 60},
            "bbw_prev":     {"fn": "shift", "on": "bbw", "n": 1},
            "bbw_min_prev": {"fn": "shift", "on": "bbw_min60", "n": 1},
            "bb_up":        {"fn": "bb_upper", "on": "close", "window": 20, "k": 2},
            "vol20":        {"fn": "sma", "on": "total_volume", "window": 20},
        },
        {"all": ["bbw_prev <= @squeeze_tol * bbw_min_prev",
                 "close > bb_up", "total_volume > @vol_mult * vol20"]},
        {"stop_loss_pct": 5, "target_pct": 12, "max_hold_days": 12},
    ),
    spec(
        "vwap_reclaim_delivery_v1", "VWAP Reclaim + Delivery v1",
        "Close crosses back above VWAP after a below-VWAP day, with delivery "
        "percentage above its 20-day average (institutional accumulation).",
        {},
        {
            "close_prev": {"fn": "shift", "on": "close", "n": 1},
            "vwap_prev":  {"fn": "shift", "on": "vwap", "n": 1},
            "del_avg20":  {"fn": "sma", "on": "delivery_pct", "window": 20},
        },
        {"all": ["close > vwap", "close_prev < vwap_prev",
                 "delivery_pct > del_avg20"]},
        {"stop_loss_pct": 5, "target_pct": 10, "max_hold_days": 10},
    ),
    spec(
        "high_momentum_52w_v1", "Near-High Momentum v1",
        "Price within 2% of its 200-day high with volume confirmation — "
        "momentum continuation near 52-week-high territory.",
        {"proximity": 0.98, "vol_mult": 1.5},
        {
            "hhv200": {"fn": "hhv", "on": "close", "window": 200},
            "vol20":  {"fn": "sma", "on": "total_volume", "window": 20},
        },
        {"all": ["close >= @proximity * hhv200",
                 "total_volume > @vol_mult * vol20"]},
        {"stop_loss_pct": 6, "target_pct": 12, "max_hold_days": 15},
        regimes=["bull"],
    ),
    spec(
        "gap_up_continuation_v1", "Gap-Up Continuation v1",
        "Opening gap above the previous high that holds (bullish close) with "
        "rising delivery quantity.",
        {},
        {
            "high_prev": {"fn": "shift", "on": "high", "n": 1},
            "del_prev":  {"fn": "shift", "on": "delivery_qty", "n": 1},
        },
        {"all": ["open > high_prev", "close > open", "delivery_qty > del_prev"]},
        {"stop_loss_pct": 4, "target_pct": 8, "max_hold_days": 7},
    ),
    spec(
        "oversold_snapback_v1", "Oversold Snapback v1",
        "5-day fall of 10%+ followed by the first up-day on elevated volume — "
        "capitulation snapback.",
        {"drop_pct": -10, "vol_mult": 1.3},
        {
            "roc5":       {"fn": "roc", "on": "close", "period": 5},
            "close_prev": {"fn": "shift", "on": "close", "n": 1},
            "vol20":      {"fn": "sma", "on": "total_volume", "window": 20},
        },
        {"all": ["roc5 < @drop_pct", "close > close_prev",
                 "total_volume > @vol_mult * vol20"]},
        {"stop_loss_pct": 5, "target_pct": 8, "max_hold_days": 5},
        regimes=["sideways", "high_vol"],
    ),
    spec(
        "ema_ribbon_cross_v1", "EMA Ribbon Cross v1",
        "Fresh EMA8/EMA21 bullish crossover above a rising EMA50 — early trend "
        "ignition entry.",
        {},
        {
            "ema8":       {"fn": "ema", "on": "close", "span": 8},
            "ema21":      {"fn": "ema", "on": "close", "span": 21},
            "ema50":      {"fn": "ema", "on": "close", "span": 50},
            "ema8_prev":  {"fn": "shift", "on": "ema8", "n": 1},
            "ema21_prev": {"fn": "shift", "on": "ema21", "n": 1},
        },
        {"all": ["ema8 > ema21", "ema8_prev <= ema21_prev", "close > ema50"]},
        {"stop_loss_pct": 5, "target_pct": 10, "max_hold_days": 12},
        regimes=["bull"],
    ),
    spec(
        "delivery_absorption_v1", "Delivery Absorption v1",
        "Down day where delivery quantity jumps 30%+ with high delivery share — "
        "weak-hand selling absorbed by strong hands; buy the absorption.",
        {"del_jump": 1.3, "del_floor": 40},
        {
            "close_prev": {"fn": "shift", "on": "close", "n": 1},
            "del_prev":   {"fn": "shift", "on": "delivery_qty", "n": 1},
        },
        {"all": ["close < close_prev", "delivery_qty > @del_jump * del_prev",
                 "delivery_pct > @del_floor"]},
        {"stop_loss_pct": 5, "target_pct": 10, "max_hold_days": 10},
    ),
]


def main():
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    for s in SPECS:
        errs = validate_spec(s)
        if errs:
            print(f"  INVALID {s['id']}: {errs}")
            continue
        path = SPEC_DIR / f"{s['id']}.json"
        path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {path.relative_to(ROOT)}")
    print(f"\n  {len(SPECS)} specs in {SPEC_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
