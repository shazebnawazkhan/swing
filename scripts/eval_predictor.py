"""
scripts/eval_predictor.py
-------------------------
Rolling walk-forward evaluation of the predictor heads (docs/PREDICTOR.md §7,
hypothesis hp_007). Answers the question a single train/validate split cannot:
is the edge real across time, or an artefact of one lucky 2-month window?

For each head it trains on every fold (default 10mo train / 2mo validate, slide 1mo,
purge = label horizon), computes validation AUC + net-of-cost top-decile basket
metrics per fold, then aggregates mean ± dispersion and applies the promotion gate.

Outputs
  outputs/predictor_eval_<asof>.json    per-head fold table + aggregate + verdict
  data/results/experiments.jsonl        one kind:"predict" record per head (append-only)

Run
  python scripts/eval_predictor.py                       # uses latest saved panel
  python scripts/eval_predictor.py --train-months 10 --valid-months 2 --slide 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config as cfg
from src.predictor import features, model

FEATURES_DIR = ROOT / "data" / "features"
OUT_DIR = ROOT / "outputs"
RESULTS = ROOT / "data" / "results" / "experiments.jsonl"

HEADS = [("dir1d", "y_dir1d", None), ("swing", "y_swing", "swing_w")]


def latest_panel() -> Path:
    panels = sorted(FEATURES_DIR.glob("panel_*.parquet"))
    if not panels:
        raise SystemExit("No panel found — run scripts/train_predictor.py first.")
    return panels[-1]


def aggregate(fold_rows: list[dict]) -> dict:
    """Mean ± std across folds for the metrics that drive the verdict."""
    def ms(key, sub=None):
        vals = [(r[sub][key] if sub else r[key]) for r in fold_rows]
        vals = [v for v in vals if np.isfinite(v)]
        return (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)) if vals else (0.0, 0.0)

    auc_m, auc_s = ms("auc")
    pf_m, pf_s = ms("profit_factor", "basket")
    prec_m, _ = ms("precision_at_n", "basket")
    exp_m, _ = ms("expectancy_pct", "basket")
    beats = sum(1 for r in fold_rows if r["beats_baseline"])
    trades = int(sum(r["basket"]["n_trades"] for r in fold_rows))
    return {
        "folds": len(fold_rows),
        "auc_mean": auc_m, "auc_std": auc_s,
        "basket_pf_mean": pf_m, "basket_pf_std": pf_s,
        "precision_at_n_mean": prec_m, "expectancy_pct_mean": exp_m,
        "folds_beating_baseline": beats, "total_basket_trades": trades,
    }


def verdict(agg: dict) -> str:
    """Promotion gate (docs/PREDICTOR.md §7) on walk-forward aggregates."""
    ok = (agg["auc_mean"] >= 0.55
          and agg["folds_beating_baseline"] >= (agg["folds"] + 1) // 2
          and agg["basket_pf_mean"] >= 1.3
          and agg["total_basket_trades"] >= 100)
    return "promote" if ok else "reject"


def eval_head(panel, head, label_col, weight_col, folds, cost_pct) -> tuple[dict, list[dict]]:
    fold_rows = []
    for i, (vs, ve, tr, va) in enumerate(folds, 1):
        try:
            art = model.train_head(panel, features.FEATURE_COLS, label_col, weight_col, tr, va)
        except ValueError as e:
            print(f"      fold {i} ({vs.date()}→{ve.date()}): skipped ({e})")
            continue
        basket = model.basket_metrics(art["valid_df"], label_col, cost_pct=cost_pct)
        m = art["metrics"]
        row = {"fold": i, "valid_start": str(vs.date()), "valid_end": str(ve.date()),
               "auc": round(m["auc"], 4), "base_auc": round(m["base_auc"], 4),
               "beats_baseline": bool(m["beats_baseline"]), "basket": basket}
        fold_rows.append(row)
        print(f"      fold {i} {vs.date()}→{ve.date()}: AUC {m['auc']:.3f} "
              f"(base {m['base_auc']:.3f})  PF {basket['profit_factor']:.2f}  "
              f"prec@N {basket['precision_at_n']:.0f}%  n={basket['n_trades']}")
    return aggregate(fold_rows), fold_rows


def append_record(record: dict):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-months", type=int, default=10)
    ap.add_argument("--valid-months", type=int, default=2)
    ap.add_argument("--slide", type=int, default=1)
    ap.add_argument("--stop", type=float, default=cfg.STOP_LOSS_PCT)
    ap.add_argument("--target", type=float, default=cfg.TARGET_PCT)
    ap.add_argument("--hold", type=int, default=cfg.MAX_HOLD_DAYS)
    args = ap.parse_args()

    t0 = time.time()
    ppath = latest_panel()
    panel = pd.read_parquet(ppath)
    panel["date"] = pd.to_datetime(panel["date"])
    asof = str(panel["date"].max().date())
    print(f"Panel {ppath.name}  rows={len(panel):,}  range "
          f"{panel['date'].min().date()}→{asof}")

    folds = model.walk_forward_folds(panel, args.train_months, args.valid_months,
                                     args.slide, purge_days=args.hold)
    print(f"Walk-forward: {len(folds)} folds "
          f"({args.train_months}mo train / {args.valid_months}mo validate, slide {args.slide}mo)")
    if not folds:
        raise SystemExit("Not enough history for a single fold — widen the window.")

    summary, eval_out = [], {"asof": asof, "config": vars(args), "heads": {}}
    for head, label_col, weight_col in HEADS:
        print(f"\n[{head}]")
        agg, fold_rows = eval_head(panel, head, label_col, weight_col, folds, cfg.ROUND_TRIP_COST_PCT)
        v = verdict(agg)
        eval_out["heads"][head] = {"aggregate": agg, "folds": fold_rows, "verdict": v}
        summary.append((head, agg, v))

        append_record({
            "id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_predict_{head}",
            "ts": datetime.now().isoformat(timespec="seconds"), "kind": "predict",
            "spec_id": None, "strategy": f"predictor:{head}",
            "params": {"head": head, "feature_set": "core", "model": "lightgbm",
                       "label_cfg": {"stop_pct": args.stop, "target_pct": args.target,
                                     "max_hold": args.hold},
                       "wf": {"train_months": args.train_months, "valid_months": args.valid_months,
                              "slide_months": args.slide}},
            "universe": "halal", "market": "nse", "window": [str(panel["date"].min().date()), asof],
            "costs": {"round_trip_pct": cfg.ROUND_TRIP_COST_PCT}, "fill": "next_open",
            "metrics": agg, "hypothesis_id": "hp_007", "verdict": v, "runtime_s": None,
        })

    OUT_DIR.mkdir(exist_ok=True)
    outp = OUT_DIR / f"predictor_eval_{asof}.json"
    outp.write_text(json.dumps(eval_out, indent=2))

    print("\n" + "=" * 78)
    print(f"{'HEAD':7s} {'FOLDS':>5s} {'AUC(mean±sd)':>16s} {'PF(mean±sd)':>15s} "
          f"{'BEAT':>5s} {'VERDICT':>9s}")
    print("-" * 78)
    for head, a, v in summary:
        print(f"{head:7s} {a['folds']:>5d} "
              f"{a['auc_mean']:>7.3f}±{a['auc_std']:<7.3f} "
              f"{a['basket_pf_mean']:>6.2f}±{a['basket_pf_std']:<6.2f} "
              f"{a['folds_beating_baseline']:>2d}/{a['folds']:<2d} {v:>9s}")
    print("=" * 78)
    print(f"Wrote {outp.relative_to(ROOT)}  + {len(summary)} predict records  "
          f"in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
