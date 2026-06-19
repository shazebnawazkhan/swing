"""
scripts/train_predictor.py
--------------------------
Build the feature+label panel from cached daily bars and train both predictor
heads (dir1d, swing) on the user's split: train ~12mo, validate last ~2mo, with a
purge gap = label horizon (docs/PREDICTOR.md §6–§7).

Outputs
  data/features/panel_<asof>.parquet      the (date,symbol) feature+label panel
  data/features/feature_manifest.json     col -> {source, lag, kind}
  models/<head>/<run_id>/model.joblib      fitted gbt+calibrator+baseline+scaler
  models/<head>/<run_id>/metrics.json      validation + basket metrics (schema §11.3)
  models/<head>/<run_id>/importance.csv    feature importances
  models/<head>/latest.txt                 pointer to the live run_id

Run
  python scripts/train_predictor.py                 # full halal universe
  python scripts/train_predictor.py --limit 200     # quick subset for a smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config as cfg
from src.predictor import data, features, labels, model

FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR = ROOT / "models"

HEADS = [
    # (head name, label column, sample-weight column or None)
    ("dir1d", "y_dir1d", None),
    ("swing", "y_swing", "swing_w"),
]


def build_panel(universe: str, asof: str | None, limit: int | None,
                stop: float, target: float, hold: int) -> pd.DataFrame:
    print("[1/3] Loading universe + building feature panel …")
    d = data.load_universe(universe, asof=asof)
    if limit:
        d = dict(list(d.items())[:limit])
        print(f"      (limited to {len(d)} symbols)")
    nifty = data.load_nifty(asof=asof)
    panel = features.build_panel(d, nifty)
    panel = labels.add_labels(panel, d, stop_pct=stop, target_pct=target, max_hold=hold)
    print(labels.label_summary(panel))
    return panel, d


def save_panel(panel: pd.DataFrame, asof: str) -> Path:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    pth = FEATURES_DIR / f"panel_{asof}.parquet"
    panel.to_parquet(pth, index=False)
    (FEATURES_DIR / "feature_manifest.json").write_text(
        json.dumps(features.feature_manifest(), indent=2))
    print(f"      panel saved -> {pth.relative_to(ROOT)} ({len(panel):,} rows)")
    return pth


def train_and_save(panel, run_id, train_mask, valid_mask, valid_start,
                   stop, target, hold, cost_pct) -> list[dict]:
    rows = []
    for head, label_col, weight_col in HEADS:
        print(f"[3/3] Training head '{head}' …")
        art = model.train_head(panel, features.FEATURE_COLS, label_col, weight_col,
                               train_mask, valid_mask)
        basket = model.basket_metrics(art["valid_df"], label_col, cost_pct=cost_pct)
        m = art["metrics"]

        outdir = MODELS_DIR / head / run_id
        outdir.mkdir(parents=True, exist_ok=True)
        joblib.dump({k: art[k] for k in
                     ("gbt", "calibrator", "baseline", "scaler", "impute", "feature_cols")},
                    outdir / "model.joblib")
        art["importance"].to_csv(outdir / "importance.csv", header=["importance"])

        record = {
            "head": head, "run_id": run_id, "trained": datetime.now().isoformat(timespec="seconds"),
            "valid_start": str(valid_start.date()),
            "label_cfg": {"stop_pct": stop, "target_pct": target, "max_hold": hold},
            "auc": round(m["auc"], 4), "pr_auc": round(m["pr_auc"], 4),
            "brier": round(m["brier"], 4), "base_auc": round(m["base_auc"], 4),
            "base_rate": round(m["base_rate"], 4),
            "beats_baseline": bool(m["beats_baseline"]),
            "basket": basket, "n_features": len(features.FEATURE_COLS),
            "n_train": m["n_train"], "n_valid": m["n_valid"],
        }
        # Promotion gate (docs/PREDICTOR.md §7) — single-split flag; WF confirms later.
        record["status"] = ("candidate" if (
            m["auc"] >= 0.55 and m["beats_baseline"]
            and basket["profit_factor"] >= 1.3 and basket["n_trades"] >= 100
        ) else "rejected")
        (outdir / "metrics.json").write_text(json.dumps(record, indent=2))
        (MODELS_DIR / head / "latest.txt").write_text(run_id)

        record["top_features"] = list(art["importance"].head(8).index)
        rows.append(record)
    return rows


def print_summary(rows: list[dict]):
    print("\n" + "=" * 74)
    print(f"{'HEAD':7s} {'AUC':>6s} {'vsBASE':>7s} {'BRIER':>6s} "
          f"{'BASKET_PF':>9s} {'PREC@N':>7s} {'TRADES':>6s} {'STATUS':>10s}")
    print("-" * 74)
    for r in rows:
        b = r["basket"]
        print(f"{r['head']:7s} {r['auc']:6.3f} "
              f"{r['auc'] - r['base_auc']:+7.3f} {r['brier']:6.3f} "
              f"{b['profit_factor']:9.2f} {b['precision_at_n']:7.1f} "
              f"{b['n_trades']:6d} {r['status']:>10s}")
    print("=" * 74)
    for r in rows:
        print(f"  {r['head']} top features: {', '.join(r['top_features'][:6])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["halal", "all"], default="halal")
    ap.add_argument("--asof", default=None, help="point-in-time cutoff (ISO date)")
    ap.add_argument("--limit", type=int, default=None, help="cap #symbols (quick smoke run)")
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--valid-months", type=int, default=2)
    ap.add_argument("--stop", type=float, default=cfg.STOP_LOSS_PCT)
    ap.add_argument("--target", type=float, default=cfg.TARGET_PCT)
    ap.add_argument("--hold", type=int, default=cfg.MAX_HOLD_DAYS)
    args = ap.parse_args()

    t0 = time.time()
    panel, _ = build_panel(args.universe, args.asof, args.limit,
                           args.stop, args.target, args.hold)
    asof = args.asof or str(panel["date"].max().date())
    save_panel(panel, asof)

    print("[2/3] Time split (train %dmo / validate %dmo, purge=%dd) …"
          % (args.train_months, args.valid_months, args.hold))
    train_mask, valid_mask, valid_start = model.time_split(
        panel, args.train_months, args.valid_months, purge_days=args.hold)
    print(f"      train rows={int(train_mask.sum()):,}  valid rows={int(valid_mask.sum()):,}  "
          f"valid_start={valid_start.date()}")

    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = train_and_save(panel, run_id, train_mask, valid_mask, valid_start,
                          args.stop, args.target, args.hold, cfg.ROUND_TRIP_COST_PCT)
    print_summary(rows)
    print(f"\nDone in {time.time() - t0:.1f}s  (run_id={run_id})")


if __name__ == "__main__":
    main()
