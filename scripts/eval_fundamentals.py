"""
scripts/eval_fundamentals.py
----------------------------
Fundamentals adapter ablation for the swing head (docs/PREDICTOR.md §3/§9,
hypothesis hp_003): do EPS-surprise / earnings features lift basket profit factor
by >= 0.1 on the subset of stocks with FRESH fundamentals?

Because free analyst targets have no history, this tests only the backfillable
earnings features (fundamentals_adapter.FUND_FEATURES). It is a true ablation:
core FEATURE_COLS vs core + FUND_FEATURES, same walk-forward folds, same picks
logic, evaluated on the fresh-fundamentals subset only.

Earnings are fetched once per symbol-batch and cached at
data/features/fundamentals/<asof>.parquet; rerun reuses the cache (use --refresh
to refetch). Batch the universe with --limit / --offset to spread the slow
yfinance pull across runs.

Outputs
  outputs/predictor_fundamentals_<asof>.json   per-fold core-vs-fund basket table
  data/results/experiments.jsonl               kind:"predict" record (hp_003)

Run
  python scripts/eval_fundamentals.py --limit 250            # first 250 symbols
  python scripts/eval_fundamentals.py --limit 250 --offset 250 --refresh
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
from src.predictor.adapters import fundamentals_adapter as fa

FEATURES_DIR = ROOT / "data" / "features"
OUT_DIR = ROOT / "outputs"
RESULTS = ROOT / "data" / "results" / "experiments.jsonl"


def latest_panel() -> Path:
    panels = sorted(FEATURES_DIR.glob("panel_*.parquet"))
    if not panels:
        raise SystemExit("No panel found — run scripts/train_predictor.py first.")
    return panels[-1]


def load_or_fetch(symbols, asof, refresh) -> pd.DataFrame:
    cp = fa.cache_path(asof)
    cached = pd.read_parquet(cp) if cp.exists() else pd.DataFrame()
    have = set(cached["symbol"].unique()) if len(cached) else set()
    todo = [s for s in symbols if s not in have] if not refresh else list(symbols)
    if todo:
        print(f"Fetching earnings for {len(todo)} symbols "
              f"({len(have)} already cached)…")
        fresh = fa.fetch_earnings(todo)
        merged = pd.concat([cached[~cached["symbol"].isin(todo)], fresh], ignore_index=True) \
            if len(cached) else fresh
        fa.save(merged, asof)
        cached = merged
    else:
        print(f"All {len(symbols)} symbols already cached.")
    return cached[cached["symbol"].isin(symbols)].copy()


def basket_on_subset(art, label_col, keep_mask_df, cost_pct):
    """Basket metrics restricted to fresh-fundamentals (date,symbol) rows."""
    vdf = art["valid_df"].merge(keep_mask_df, on=["date", "symbol"], how="inner")
    return model.basket_metrics(vdf, label_col, cost_pct=cost_pct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-months", type=int, default=10)
    ap.add_argument("--valid-months", type=int, default=2)
    ap.add_argument("--slide", type=int, default=2)
    ap.add_argument("--limit", type=int, default=250, help="symbols to include (after offset)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--refresh", action="store_true", help="refetch earnings (ignore cache)")
    args = ap.parse_args()

    t0 = time.time()
    ppath = latest_panel()
    panel = pd.read_parquet(ppath)
    panel["date"] = pd.to_datetime(panel["date"])
    asof = str(panel["date"].max().date())

    all_syms = sorted(panel["symbol"].unique())
    syms = all_syms[args.offset:args.offset + args.limit]
    print(f"Panel {ppath.name}  rows={len(panel):,}  asof {asof}")
    print(f"Universe slice: {len(syms)} / {len(all_syms)} symbols "
          f"(offset {args.offset}, limit {args.limit})\n")

    earnings = load_or_fetch(syms, asof, args.refresh)

    # Restrict the panel to the slice, attach fundamentals, flag the fresh subset.
    panel = panel[panel["symbol"].isin(syms)].copy()
    feat = fa.build_features(panel[["date", "symbol"]], earnings)
    panel = panel.merge(feat, on=["date", "symbol"], how="left")
    fresh = fa.fresh_mask(panel)
    panel = panel[fresh].copy()                       # hp_003: fresh-fundamentals subset only
    keep_ids = panel[["date", "symbol"]].copy()
    cov = panel["symbol"].nunique()
    print(f"Fresh-fundamentals subset: {len(panel):,} rows, {cov} symbols "
          f"({panel['date'].min().date()}→{panel['date'].max().date()})\n")
    if len(panel) < 2000:
        print("WARNING: thin subset — treat the result as a first read, not a verdict.")

    folds = model.walk_forward_folds(panel, args.train_months, args.valid_months,
                                     args.slide, purge_days=cfg.MAX_HOLD_DAYS)
    if not folds:
        raise SystemExit("Not enough history in the subset for a fold — widen --limit.")
    print(f"Walk-forward: {len(folds)} folds "
          f"({args.train_months}mo/{args.valid_months}mo, slide {args.slide}mo)\n")

    core_cols = features.FEATURE_COLS
    fund_cols = features.FEATURE_COLS + fa.FUND_FEATURES
    rows = []
    for i, (vs, ve, tr, va) in enumerate(folds, 1):
        try:
            a_core = model.train_head(panel, core_cols, "y_swing", "swing_w", tr, va)
            a_fund = model.train_head(panel, fund_cols, "y_swing", "swing_w", tr, va)
        except ValueError as e:
            print(f"   fold {i} ({vs.date()}→{ve.date()}): skipped ({e})")
            continue
        b_core = basket_on_subset(a_core, "y_swing", keep_ids, cfg.ROUND_TRIP_COST_PCT)
        b_fund = basket_on_subset(a_fund, "y_swing", keep_ids, cfg.ROUND_TRIP_COST_PCT)
        rows.append({"fold": i, "valid_start": str(vs.date()), "valid_end": str(ve.date()),
                     "auc_core": round(a_core["metrics"]["auc"], 4),
                     "auc_fund": round(a_fund["metrics"]["auc"], 4),
                     "pf_core": b_core["profit_factor"], "pf_fund": b_fund["profit_factor"],
                     "n_core": b_core["n_trades"], "n_fund": b_fund["n_trades"]})
        print(f"   fold {i} {vs.date()}→{ve.date()}: "
              f"AUC {a_core['metrics']['auc']:.3f}→{a_fund['metrics']['auc']:.3f}  "
              f"PF {b_core['profit_factor']:.2f}→{b_fund['profit_factor']:.2f}  "
              f"n={b_core['n_trades']}")

    if not rows:
        raise SystemExit("No usable folds.")

    def mean(k):
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    pf_core_m, pf_fund_m = mean("pf_core"), mean("pf_fund")
    auc_core_m, auc_fund_m = mean("auc_core"), mean("auc_fund")
    delta_pf = round(pf_fund_m - pf_core_m, 3)
    delta_auc = round(auc_fund_m - auc_core_m, 4)
    ok = delta_pf >= 0.1
    v = "confirm" if ok else "reject"

    print("\n" + "=" * 60)
    print(f"{'':14s} {'AUC':>8s} {'BasketPF':>10s}")
    print(f"{'core':14s} {auc_core_m:>8.3f} {pf_core_m:>10.2f}")
    print(f"{'core+fund':14s} {auc_fund_m:>8.3f} {pf_fund_m:>10.2f}")
    print(f"{'delta':14s} {delta_auc:>+8.4f} {delta_pf:>+10.2f}")
    print(f"VERDICT (hp_003): {v.upper()}  (need ΔbasketPF ≥ +0.10)")
    print("=" * 60)

    metrics = {"folds": rows, "pf_core_mean": pf_core_m, "pf_fund_mean": pf_fund_m,
               "delta_basket_pf": delta_pf, "auc_core_mean": auc_core_m,
               "auc_fund_mean": auc_fund_m, "delta_auc": delta_auc,
               "subset_rows": int(len(panel)), "subset_symbols": int(cov),
               "universe_slice": [args.offset, args.offset + args.limit]}

    OUT_DIR.mkdir(exist_ok=True)
    outp = OUT_DIR / f"predictor_fundamentals_{asof}.json"
    outp.write_text(json.dumps({"asof": asof, "config": vars(args),
                                "metrics": metrics, "verdict": v}, indent=2))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_predict_fundamentals",
            "ts": datetime.now().isoformat(timespec="seconds"), "kind": "predict",
            "spec_id": None, "strategy": "predictor:swing",
            "params": {"head": "swing", "feature_set": "core+fundamentals",
                       "model": "lightgbm", "adapters": ["fundamentals"],
                       "subset": "fresh_fundamentals",
                       "wf": {"train_months": args.train_months,
                              "valid_months": args.valid_months, "slide_months": args.slide}},
            "universe": "halal", "market": "nse",
            "window": [str(panel["date"].min().date()), asof],
            "costs": {"round_trip_pct": cfg.ROUND_TRIP_COST_PCT}, "fill": "next_open",
            "metrics": metrics, "hypothesis_id": "hp_003", "verdict": v, "runtime_s": None,
        }) + "\n")

    print(f"Wrote {outp.relative_to(ROOT)}  + 1 predict record  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
