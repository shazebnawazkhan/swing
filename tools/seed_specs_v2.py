"""Refinement batch: v2 children of the best batch-1 specs.

Finding from batch-1 (see docs/research/batch1.md): the 2025-26 window was a
bear year for the halal universe (median stock -14.8%); every all-long spec
lost after costs.  v2 adds trend gates (EMA200) and tighter exits.
One change-family per child, parent recorded in provenance.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.strategies.spec import SCHEMA_VERSION, SPEC_DIR, validate_spec

CREATED = "2026-06-12"


def child(id_, parent, name, desc, params, indicators, entry, exit_, regimes=None):
    return {
        "schema": SCHEMA_VERSION, "id": id_, "name": name, "description": desc,
        "markets": ["nse"], "regimes": regimes or ["bull"],
        "params": params, "indicators": indicators,
        "entry": entry, "exit": exit_,
        "provenance": {"created": CREATED, "hypothesis_id": "h_011",
                       "parent_id": parent, "experiments": []},
    }


SPECS = [
    child(
        "delivery_breakout_v2", "delivery_breakout_v1",
        "Delivery Breakout v2 (trend-gated)",
        "v1 + EMA200 trend gate and tighter exits — only break out of bases "
        "that are already in long-term uptrends.",
        {"del_mult": 1.5, "lookback": 20},
        {
            "hhv20":      {"fn": "hhv", "on": "close", "window": "@lookback"},
            "hhv20_prev": {"fn": "shift", "on": "hhv20", "n": 1},
            "del_avg20":  {"fn": "sma", "on": "delivery_pct", "window": 20},
            "ema200":     {"fn": "ema", "on": "close", "span": 200},
        },
        {"all": ["close > hhv20_prev", "delivery_pct > @del_mult * del_avg20",
                 "close > ema200"]},
        {"stop_loss_pct": 4, "target_pct": 8, "max_hold_days": 8},
    ),
    child(
        "high_momentum_52w_v2", "high_momentum_52w_v1",
        "Near-High Momentum v2 (delivery-confirmed)",
        "v1 + delivery% confirmation: near-high breakouts only when strong "
        "hands are taking delivery above the 20-day average.",
        {"proximity": 0.98, "vol_mult": 1.5},
        {
            "hhv200":    {"fn": "hhv", "on": "close", "window": 200},
            "vol20":     {"fn": "sma", "on": "total_volume", "window": 20},
            "del_avg20": {"fn": "sma", "on": "delivery_pct", "window": 20},
        },
        {"all": ["close >= @proximity * hhv200", "total_volume > @vol_mult * vol20",
                 "delivery_pct > del_avg20"]},
        {"stop_loss_pct": 5, "target_pct": 10, "max_hold_days": 12},
    ),
    child(
        "rsi2_meanrev_v2", "rsi2_meanrev_v1",
        "RSI2 Mean Reversion v2 (deep dip, quick exit)",
        "v1 with a deeper oversold trigger (RSI2<5), stronger trend gate "
        "(EMA200) and a fast 4-day exit — scalp the bounce, don't marry it.",
        {"rsi_buy": 5, "trend_span": 200},
        {
            "rsi2":     {"fn": "rsi", "on": "close", "period": 2},
            "ema_t":    {"fn": "ema", "on": "close", "span": "@trend_span"},
        },
        {"all": ["rsi2 < @rsi_buy", "close > ema_t"]},
        {"stop_loss_pct": 4, "target_pct": 4, "max_hold_days": 4},
        regimes=["sideways", "bull"],
    ),
    child(
        "gap_up_continuation_v2", "gap_up_continuation_v1",
        "Gap-Up Continuation v2 (trend-gated)",
        "v1 + EMA200 trend gate — gaps in downtrends are exits, not entries.",
        {},
        {
            "high_prev": {"fn": "shift", "on": "high", "n": 1},
            "del_prev":  {"fn": "shift", "on": "delivery_qty", "n": 1},
            "ema200":    {"fn": "ema", "on": "close", "span": 200},
        },
        {"all": ["open > high_prev", "close > open", "delivery_qty > del_prev",
                 "close > ema200"]},
        {"stop_loss_pct": 4, "target_pct": 8, "max_hold_days": 7},
    ),
    child(
        "pullback_trend_v2", "pullback_trend_v1",
        "Pullback Trend v2 (strong-trend only, quick exit)",
        "v1 restricted to strong uptrends (EMA50>EMA200 as well) with a "
        "tighter 6% target and 7-day hold.",
        {},
        {
            "ema20":  {"fn": "ema", "on": "close", "span": 20},
            "ema50":  {"fn": "ema", "on": "close", "span": 50},
            "ema200": {"fn": "ema", "on": "close", "span": 200},
        },
        {"all": ["ema20 > ema50", "ema50 > ema200", "low <= ema20",
                 "close > open"]},
        {"stop_loss_pct": 4, "target_pct": 6, "max_hold_days": 7},
    ),
]


def main():
    for s in SPECS:
        errs = validate_spec(s)
        if errs:
            print(f"  INVALID {s['id']}: {errs}")
            continue
        path = SPEC_DIR / f"{s['id']}.json"
        path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
