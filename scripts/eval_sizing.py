"""
scripts/eval_sizing.py
----------------------
Position-sizing test for the swing head (docs/PREDICTOR.md §9, hypothesis hp_004):
does sizing the top-decile basket by calibrated P(win) (Kelly-capped) lower max
drawdown vs equal-weight, *without* giving up CAGR?

Method: walk-forward folds (non-overlapping by default so the validation windows
tile the timeline), train the swing head per fold, concatenate every fold's
out-of-sample (date, symbol, p, fwd_ret_pct), then build one daily-rebalanced
equity curve under each sizing scheme on identical daily picks. Because both
schemes trade the same names, the CAGR / max-DD gap isolates sizing.

Verdict (hp_004): max-DD drops by >= 2 points AND CAGR stays within +-10% of
equal-weight (a sizing scheme that just trades less and earns less is not a win).

Outputs
  outputs/predictor_sizing_<asof>.json   both curves + delta + verdict
  data/results/experiments.jsonl         one kind:"predict" record (append-only)

Run
  python scripts/eval_sizing.py
  python scripts/eval_sizing.py --kelly-cap 0.25 --top-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config as cfg
from src.predictor import features, model

FEATURES_DIR = ROOT / "data" / "features"
OUT_DIR = ROOT / "outputs"
RESULTS = ROOT / "data" / "results" / "experiments.jsonl"


def latest_panel() -> Path:
    panels = sorted(FEATURES_DIR.glob("panel_*.parquet"))
    if not panels:
        raise SystemExit("No panel found — run scripts/train_predictor.py first.")
    return panels[-1]


def oos_predictions(panel, folds) -> pd.DataFrame:
    """Concatenate every fold's calibrated validation predictions for the swing head.

    Folds are non-overlapping by default, but if any (date, symbol) repeats we keep
    the most recent fold's prediction so the equity curve never double-counts a day.
    """
    frames = []
    for i, (vs, ve, tr, va) in enumerate(folds, 1):
        try:
            art = model.train_head(panel, features.FEATURE_COLS, "y_swing", "swing_w", tr, va)
        except ValueError as e:
            print(f"   fold {i} ({vs.date()}→{ve.date()}): skipped ({e})")
            continue
        vdf = art["valid_df"]
        print(f"   fold {i} {vs.date()}→{ve.date()}: AUC {art['metrics']['auc']:.3f}  "
              f"n_valid={len(vdf):,}")
        frames.append(vdf)
    if not frames:
        raise SystemExit("No usable folds — widen the window.")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("date").drop_duplicates(["date", "symbol"], keep="last")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-months", type=int, default=10)
    ap.add_argument("--valid-months", type=int, default=2)
    ap.add_argument("--slide", type=int, default=None,
                    help="months to slide folds (default = valid-months => non-overlapping)")
    ap.add_argument("--top-frac", type=float, default=0.1)
    ap.add_argument("--kelly-cap", type=float, default=0.25,
                    help="max per-name fraction (cap on full Kelly)")
    args = ap.parse_args()
    slide = args.slide if args.slide is not None else args.valid_months
    kelly_b = cfg.TARGET_PCT / cfg.STOP_LOSS_PCT          # payoff odds from the barriers

    t0 = time.time()
    ppath = latest_panel()
    panel = pd.read_parquet(ppath)
    panel["date"] = pd.to_datetime(panel["date"])
    asof = str(panel["date"].max().date())
    print(f"Panel {ppath.name}  rows={len(panel):,}  range "
          f"{panel['date'].min().date()}→{asof}")

    folds = model.walk_forward_folds(panel, args.train_months, args.valid_months,
                                     slide, purge_days=cfg.MAX_HOLD_DAYS)
    print(f"Walk-forward: {len(folds)} folds "
          f"({args.train_months}mo train / {args.valid_months}mo validate, slide {slide}mo)\n"
          f"Kelly odds b=TARGET/STOP={cfg.TARGET_PCT:.0f}/{cfg.STOP_LOSS_PCT:.0f}={kelly_b:.2f}  "
          f"cap={args.kelly_cap}  top_frac={args.top_frac}\n")
    if not folds:
        raise SystemExit("Not enough history for a single fold.")

    oos = oos_predictions(panel, folds)
    common = dict(top_frac=args.top_frac, cost_pct=cfg.ROUND_TRIP_COST_PCT)
    kcommon = dict(kelly_b=kelly_b, kelly_cap=args.kelly_cap, **common)
    equal = model.portfolio_curve(oos, sizing="equal", **common)
    kelly = model.portfolio_curve(oos, sizing="kelly", **kcommon)
    kelly_full = model.portfolio_curve(oos, sizing="kelly_full", **kcommon)

    delta_dd = round(kelly["max_dd_pct"] - equal["max_dd_pct"], 2)
    base_cagr = equal["cagr_pct"]
    cagr_retained = (kelly["cagr_pct"] / base_cagr) if base_cagr > 0 else float("nan")
    cagr_ok = (base_cagr <= 0) or (abs(kelly["cagr_pct"] - base_cagr) <= 0.10 * abs(base_cagr))
    ok = (delta_dd <= -2.0) and cagr_ok
    v = "confirm" if ok else "reject"

    print("=" * 70)
    print(f"{'SIZING':10s} {'CAGR%':>8s} {'MaxDD%':>8s} {'Sharpe':>7s} "
          f"{'TotRet%':>9s} {'Expo':>6s} {'Trades':>7s}")
    print("-" * 70)
    for name, c in (("equal", equal), ("kelly", kelly), ("kelly_full", kelly_full)):
        print(f"{name:10s} {c['cagr_pct']:>8.2f} {c['max_dd_pct']:>8.2f} "
              f"{c['sharpe']:>7.2f} {c['total_return_pct']:>9.2f} "
              f"{c['avg_exposure']:>6.2f} {c['n_trades']:>7d}")
    print("-" * 70)
    print(f"ΔMaxDD = {delta_dd:+.2f} pts (need ≤ -2.00)   "
          f"CAGR retained = {cagr_retained:.0%}" if base_cagr > 0
          else f"ΔMaxDD = {delta_dd:+.2f} pts (need ≤ -2.00)   CAGR≤0 → gate on DD only")
    print(f"VERDICT (hp_004): {v.upper()}")
    print("=" * 70)

    metrics = {"equal_weight": equal, "kelly_capped": kelly, "kelly_full": kelly_full,
               "delta_max_dd_pct": delta_dd,
               "delta_max_dd_pct_tilt_only": round(kelly_full["max_dd_pct"] - equal["max_dd_pct"], 2),
               "cagr_retained": (round(cagr_retained, 3) if base_cagr > 0 else None),
               "kelly_b": round(kelly_b, 3), "kelly_cap": args.kelly_cap}

    OUT_DIR.mkdir(exist_ok=True)
    outp = OUT_DIR / f"predictor_sizing_{asof}.json"
    outp.write_text(json.dumps({"asof": asof, "config": vars(args), "metrics": metrics,
                                "verdict": v}, indent=2))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_predict_sizing",
            "ts": datetime.now().isoformat(timespec="seconds"), "kind": "predict",
            "spec_id": None, "strategy": "predictor:swing",
            "params": {"head": "swing", "feature_set": "core", "model": "lightgbm",
                       "sizing": "kelly_capped", "compare_to": "equal_weight",
                       "kelly_b": round(kelly_b, 3), "kelly_cap": args.kelly_cap,
                       "top_frac": args.top_frac,
                       "wf": {"train_months": args.train_months,
                              "valid_months": args.valid_months, "slide_months": slide}},
            "universe": "halal", "market": "nse",
            "window": [str(panel["date"].min().date()), asof],
            "costs": {"round_trip_pct": cfg.ROUND_TRIP_COST_PCT}, "fill": "next_open",
            "metrics": metrics, "hypothesis_id": "hp_004", "verdict": v, "runtime_s": None,
        }) + "\n")

    print(f"Wrote {outp.relative_to(ROOT)}  + 1 predict record  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
