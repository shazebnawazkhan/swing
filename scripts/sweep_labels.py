"""
scripts/sweep_labels.py
-----------------------
Label sweep for the swing head (docs/PREDICTOR.md §5, hypothesis hp_005).

Features are held fixed; only the triple-barrier label (horizon H, take-profit,
stop-loss) varies. For each barrier set we relabel, train the swing head on a
walk-forward split, and compare net-of-cost basket expectancy / profit factor to
the config default (SL/TP/H). The winner is the barrier set the predictor should
adopt; it should then be confirmed with the full eval_predictor walk-forward.

Outputs
  outputs/label_sweep_<asof>.json     ranked barrier sets + delta vs default
  data/results/experiments.jsonl      one kind:"predict" record per barrier set

Run
  python scripts/sweep_labels.py                          # default grid
  python scripts/sweep_labels.py --holds 5 10 15 --tps 8 10 14 --sls 3 5 --folds 2
"""

from __future__ import annotations

import argparse
import itertools
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
from src.predictor import data, features, labels, model

OUT_DIR = ROOT / "outputs"
RESULTS = ROOT / "data" / "results" / "experiments.jsonl"


def eval_barrier(panel_feats, label_frame, folds, cost_pct) -> dict:
    """Train the swing head per fold for one barrier set; return mean basket stats."""
    panel = panel_feats.merge(label_frame, on=["date", "symbol"], how="left")
    pfs, exps, precs, aucs, beats = [], [], [], [], 0
    for vs, ve, tr, va in folds:
        try:
            art = model.train_head(panel, features.FEATURE_COLS, "y_swing", "swing_w", tr, va)
        except ValueError:
            continue
        b = model.basket_metrics(art["valid_df"], "y_swing", cost_pct=cost_pct)
        if not np.isfinite(b["profit_factor"]):
            b["profit_factor"] = 0.0
        pfs.append(b["profit_factor"]); exps.append(b["expectancy_pct"])
        precs.append(b["precision_at_n"]); aucs.append(art["metrics"]["auc"])
        beats += int(art["metrics"]["beats_baseline"])
    if not pfs:
        return {"basket_pf_mean": 0.0, "expectancy_pct_mean": 0.0, "precision_at_n_mean": 0.0,
                "auc_mean": 0.0, "folds_beating_baseline": 0, "folds": 0, "pos_rate": 0.0}
    return {
        "basket_pf_mean": round(float(np.mean(pfs)), 3),
        "expectancy_pct_mean": round(float(np.mean(exps)), 3),
        "precision_at_n_mean": round(float(np.mean(precs)), 1),
        "auc_mean": round(float(np.mean(aucs)), 3),
        "folds_beating_baseline": beats, "folds": len(pfs),
        "pos_rate": round(float(panel["y_swing"].dropna().mean()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["halal", "all"], default="halal")
    ap.add_argument("--holds", type=int, nargs="+", default=[5, 10, 15])
    ap.add_argument("--tps", type=float, nargs="+", default=[8, 10, 14])
    ap.add_argument("--sls", type=float, nargs="+", default=[3, 5])
    ap.add_argument("--folds", type=int, default=2, help="most-recent walk-forward folds per combo")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    print("[1/3] Loading bars + building features once …")
    d = data.load_universe(args.universe)
    if args.limit:
        d = dict(list(d.items())[:args.limit])
    nifty = data.load_nifty()
    panel_feats = features.build_panel(d, nifty)
    asof = str(panel_feats["date"].max().date())

    # Use the most-recent N walk-forward folds (built on a label-free copy; masks are
    # date-based so any label set reuses them).
    all_folds = model.walk_forward_folds(panel_feats.assign(y_swing=np.nan),
                                         train_months=10, valid_months=2, slide_months=1,
                                         purge_days=max(args.holds))
    folds = all_folds[-args.folds:] if args.folds else all_folds
    print(f"      {len(d)} symbols, asof {asof}, {len(folds)} folds/combo")

    default = (cfg.STOP_LOSS_PCT, cfg.TARGET_PCT, cfg.MAX_HOLD_DAYS)
    combos = list(itertools.product(args.sls, args.tps, args.holds))
    if default not in combos:
        combos.append(default)
    print(f"[2/3] Sweeping {len(combos)} barrier sets …")

    rows = []
    for i, (sl, tp, hold) in enumerate(combos, 1):
        lf = labels.swing_label_frame(d, stop_pct=sl, target_pct=tp, max_hold=hold)
        m = eval_barrier(panel_feats, lf, folds, cfg.ROUND_TRIP_COST_PCT)
        is_def = (sl, tp, hold) == default
        rows.append({"sl": sl, "tp": tp, "hold": hold, "is_default": is_def, **m})
        print(f"      [{i:>2d}/{len(combos)}] SL{sl:g}/TP{tp:g}/H{hold:<2d}"
              f"{'  (default)' if is_def else '':<11s} "
              f"exp {m['expectancy_pct_mean']:+.3f}%  PF {m['basket_pf_mean']:.2f}  "
              f"prec@N {m['precision_at_n_mean']:.0f}%  pos {m['pos_rate']*100:.0f}%")

    rows.sort(key=lambda r: r["expectancy_pct_mean"], reverse=True)
    dflt = next(r for r in rows if r["is_default"])
    best = rows[0]
    delta = round(best["expectancy_pct_mean"] - dflt["expectancy_pct_mean"], 3)
    win = best if not best["is_default"] else (rows[1] if len(rows) > 1 else best)
    verdict = "confirmed" if delta >= 0.1 and not best["is_default"] else "rejected"

    OUT_DIR.mkdir(exist_ok=True)
    outp = OUT_DIR / f"label_sweep_{asof}.json"
    outp.write_text(json.dumps({"asof": asof, "default": dflt, "best": best,
                                "delta_expectancy": delta, "verdict": verdict,
                                "ranked": rows}, indent=2))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_predict_swing_sl{r['sl']:g}tp{r['tp']:g}h{r['hold']}",
                "ts": datetime.now().isoformat(timespec="seconds"), "kind": "predict",
                "spec_id": None, "strategy": "predictor:swing",
                "params": {"head": "swing", "feature_set": "core", "model": "lightgbm",
                           "label_cfg": {"stop_pct": r["sl"], "target_pct": r["tp"], "max_hold": r["hold"]}},
                "universe": args.universe, "market": "nse", "window": [None, asof],
                "costs": {"round_trip_pct": cfg.ROUND_TRIP_COST_PCT}, "fill": "next_open",
                "metrics": {k: r[k] for k in ("basket_pf_mean", "expectancy_pct_mean",
                                              "precision_at_n_mean", "auc_mean", "pos_rate")},
                "hypothesis_id": "hp_005", "verdict": None, "runtime_s": None}) + "\n")

    print("\n" + "=" * 70)
    print(f"{'RANK':>4s} {'BARRIER':<14s} {'EXP%':>7s} {'PF':>6s} {'PREC@N':>7s} {'AUC':>6s}")
    print("-" * 70)
    for i, r in enumerate(rows, 1):
        tag = " *default" if r["is_default"] else ""
        print(f"{i:>4d} SL{r['sl']:g}/TP{r['tp']:g}/H{r['hold']:<3d} "
              f"{r['expectancy_pct_mean']:>+7.3f} {r['basket_pf_mean']:>6.2f} "
              f"{r['precision_at_n_mean']:>6.0f}% {r['auc_mean']:>6.3f}{tag}")
    print("=" * 70)
    print(f"Best: SL{best['sl']:g}/TP{best['tp']:g}/H{best['hold']}  "
          f"exp {best['expectancy_pct_mean']:+.3f}%  vs default {dflt['expectancy_pct_mean']:+.3f}%  "
          f"(Δ {delta:+.3f}%)  → {verdict}")
    print(f"Wrote {outp.relative_to(ROOT)} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
