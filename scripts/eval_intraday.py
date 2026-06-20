"""
scripts/eval_intraday.py
------------------------
Intraday adapter ablation for the swing head (docs/PREDICTOR.md §3/§9, hypothesis
hp_006): do 5-minute entry-timing features lift basket profit factor by >= 0.1?

DATA REALITY (read this before trusting any number): yfinance serves only ~60 days
of 5m history, while the deployed swing label needs a 15-day forward window. A
60-day window with a 15-day label and a proper purge leaves almost no clean split.
So this script deliberately uses a REDUCED setup:
  * a SHORT-HORIZON swing label (default H=5, TP/SL scaled down) recomputed on the
    fly, so a purged train/validate split fits inside the intraday window;
  * a day-count split (default train 20 / validate 8 trading days, purge = horizon).
Even so the sample is small and the result is a FIRST READ, not a verdict — which
is exactly why PREDICTOR.md keeps intraday optional/off. The verdict is reported
as 'inconclusive' unless the trade count clears a minimum.

Intraday 5m bars are cached at data/features/intraday/<asof>.parquet (reuse on
rerun; --refresh to refetch). Batch with --limit / --offset.

Outputs
  outputs/predictor_intraday_<asof>.json   core-vs-intraday basket table
  data/results/experiments.jsonl           kind:"predict" record (hp_006)

Run
  python scripts/eval_intraday.py --limit 150
  python scripts/eval_intraday.py --limit 150 --horizon 5 --target 6 --stop 3
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
from src.predictor import data as pdata, features, labels, model
from src.predictor.adapters import intraday_adapter as ia

FEATURES_DIR = ROOT / "data" / "features"
OUT_DIR = ROOT / "outputs"
RESULTS = ROOT / "data" / "results" / "experiments.jsonl"
MIN_TRADES = 60          # below this the basket PF is pure noise -> inconclusive


def latest_panel() -> Path:
    panels = sorted(FEATURES_DIR.glob("panel_*.parquet"))
    if not panels:
        raise SystemExit("No panel found — run scripts/train_predictor.py first.")
    return panels[-1]


def load_or_fetch(symbols, asof, period, refresh) -> pd.DataFrame:
    cp = ia.cache_path(asof)
    cached = pd.read_parquet(cp) if cp.exists() else pd.DataFrame()
    have = set(cached["symbol"].unique()) if len(cached) else set()
    todo = [s for s in symbols if s not in have] if not refresh else list(symbols)
    if todo:
        print(f"Fetching 5m bars for {len(todo)} symbols ({len(have)} cached)…")
        fresh = ia.fetch_intraday(todo, period=period)
        cached = pd.concat([cached[~cached["symbol"].isin(todo)], fresh], ignore_index=True) \
            if len(cached) else fresh
        ia.save(cached, asof)
    else:
        print(f"All {len(symbols)} symbols already cached.")
    return cached[cached["symbol"].isin(symbols)].copy()


def day_split(dates: np.ndarray, valid_days: int, train_days: int, purge: int):
    """One purged split on the tail of `dates` (sorted unique). Returns (train_set, valid_set)."""
    d = np.sort(np.unique(dates))
    if len(d) < valid_days + purge + 5:
        return None
    valid = set(d[-valid_days:])
    purge_cut = d[-valid_days]                       # first valid day
    train_pool = d[d < purge_cut]
    train_pool = train_pool[:-purge] if purge > 0 and len(train_pool) > purge else train_pool
    train = set(train_pool[-train_days:]) if train_days else set(train_pool)
    return train, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--period", default="60d")
    ap.add_argument("--horizon", type=int, default=5, help="short swing label hold (trading days)")
    ap.add_argument("--target", type=float, default=6.0)
    ap.add_argument("--stop", type=float, default=3.0)
    ap.add_argument("--train-days", type=int, default=20)
    ap.add_argument("--valid-days", type=int, default=8)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ppath = latest_panel()
    panel = pd.read_parquet(ppath)
    panel["date"] = pd.to_datetime(panel["date"])
    asof = str(panel["date"].max().date())
    all_syms = sorted(panel["symbol"].unique())
    syms = all_syms[args.offset:args.offset + args.limit]
    panel = panel[panel["symbol"].isin(syms)].copy()
    print(f"Panel {ppath.name}  asof {asof}  slice {len(syms)}/{len(all_syms)} symbols\n")

    # 1) Short-horizon swing label (so a purged split fits the ~60d intraday window).
    bars = pdata.load_universe("halal")
    bars = {s: df for s, df in bars.items() if s in syms}
    lab = labels.swing_label_frame(bars, args.stop, args.target, args.horizon)
    lab["date"] = pd.to_datetime(lab["date"])
    panel = panel.drop(columns=["y_swing", "swing_w", "fwd_ret_pct"], errors="ignore") \
                 .merge(lab, on=["date", "symbol"], how="left")
    print(f"Short label: TP{args.target:.0f}/SL{args.stop:.0f}/H{args.horizon}  "
          f"pos_rate={panel['y_swing'].dropna().mean()*100:.1f}%")

    # 2) Intraday features over the recent window.
    intr = load_or_fetch(syms, asof, args.period, args.refresh)
    feat = ia.build_features(panel[["date", "symbol"]], intr)
    panel = panel.merge(feat, on=["date", "symbol"], how="left")

    # 3) Keep rows that have intraday coverage AND a usable label.
    covered = panel[ia.INTRADAY_FEATURES].notna().any(axis=1)
    panel = panel[covered & panel["y_swing"].notna()].copy()
    if panel.empty:
        raise SystemExit("No intraday-covered labelled rows — widen --limit/--period.")
    d0, d1 = panel["date"].min().date(), panel["date"].max().date()
    print(f"Intraday-covered labelled rows: {len(panel):,}, {panel['symbol'].nunique()} symbols "
          f"({d0}→{d1})\n")

    sp = day_split(panel["date"].values, args.valid_days, args.train_days, args.horizon)
    if sp is None:
        raise SystemExit("Not enough covered days for a purged split — widen --period/--limit.")
    train_set, valid_set = sp
    tr = panel["date"].isin(train_set)
    va = panel["date"].isin(valid_set)
    print(f"Split: train {len(train_set)}d ({int(tr.sum())} rows) / "
          f"validate {len(valid_set)}d ({int(va.sum())} rows), purge {args.horizon}d\n")

    core_cols = features.FEATURE_COLS
    intr_cols = features.FEATURE_COLS + ia.INTRADAY_FEATURES
    try:
        a_core = model.train_head(panel, core_cols, "y_swing", "swing_w", tr, va)
        a_intr = model.train_head(panel, intr_cols, "y_swing", "swing_w", tr, va)
    except ValueError as e:
        raise SystemExit(f"degenerate split ({e}) — widen the window.")

    b_core = model.basket_metrics(a_core["valid_df"], "y_swing", cost_pct=cfg.ROUND_TRIP_COST_PCT)
    b_intr = model.basket_metrics(a_intr["valid_df"], "y_swing", cost_pct=cfg.ROUND_TRIP_COST_PCT)
    auc_c, auc_i = a_core["metrics"]["auc"], a_intr["metrics"]["auc"]
    delta_pf = round(b_intr["profit_factor"] - b_core["profit_factor"], 3)
    delta_auc = round(auc_i - auc_c, 4)

    n = min(b_core["n_trades"], b_intr["n_trades"])
    if n < MIN_TRADES:
        v = "inconclusive"
    else:
        v = "confirm" if delta_pf >= 0.1 else "reject"

    print("=" * 60)
    print(f"{'':14s} {'AUC':>8s} {'BasketPF':>10s} {'n':>6s}")
    print(f"{'core':14s} {auc_c:>8.3f} {b_core['profit_factor']:>10.2f} {b_core['n_trades']:>6d}")
    print(f"{'core+intraday':14s} {auc_i:>8.3f} {b_intr['profit_factor']:>10.2f} {b_intr['n_trades']:>6d}")
    print(f"{'delta':14s} {delta_auc:>+8.4f} {delta_pf:>+10.2f}")
    print(f"VERDICT (hp_006): {v.upper()}"
          + (f"  (n={n} < {MIN_TRADES}: data-limited)" if v == "inconclusive"
             else "  (need ΔbasketPF ≥ +0.10)"))
    print("=" * 60)

    metrics = {"auc_core": round(auc_c, 4), "auc_intraday": round(auc_i, 4),
               "delta_auc": delta_auc, "pf_core": b_core["profit_factor"],
               "pf_intraday": b_intr["profit_factor"], "delta_basket_pf": delta_pf,
               "n_trades": int(n), "n_rows_valid": int(va.sum()),
               "symbols_covered": int(panel["symbol"].nunique()),
               "label": {"horizon": args.horizon, "target": args.target, "stop": args.stop},
               "window": [str(d0), str(d1)]}

    OUT_DIR.mkdir(exist_ok=True)
    outp = OUT_DIR / f"predictor_intraday_{asof}.json"
    outp.write_text(json.dumps({"asof": asof, "config": vars(args),
                                "metrics": metrics, "verdict": v}, indent=2))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_predict_intraday",
            "ts": datetime.now().isoformat(timespec="seconds"), "kind": "predict",
            "spec_id": None, "strategy": "predictor:swing",
            "params": {"head": "swing", "feature_set": "core+intraday", "model": "lightgbm",
                       "adapters": ["intraday"],
                       "label_cfg": {"horizon": args.horizon, "target": args.target, "stop": args.stop},
                       "split": {"train_days": args.train_days, "valid_days": args.valid_days}},
            "universe": "halal", "market": "nse", "window": [str(d0), str(d1)],
            "costs": {"round_trip_pct": cfg.ROUND_TRIP_COST_PCT}, "fill": "next_open",
            "metrics": metrics, "hypothesis_id": "hp_006", "verdict": v, "runtime_s": None,
        }) + "\n")

    print(f"Wrote {outp.relative_to(ROOT)}  + 1 predict record  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
