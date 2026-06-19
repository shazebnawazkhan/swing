"""
scripts/predict_daily.py
------------------------
Score the latest available bar for every stock with both trained heads and emit a
ranked next-session BUY/SELL list (docs/PREDICTOR.md §8).

  outputs/predicted_signals_<date>.json   the daily product (schema §11.2)
  data/paper_trades.jsonl                  every emitted signal, for the live-vs-backtest gap

Direction rule: BUY when both heads agree above threshold, SELL/AVOID when both agree
below (1-threshold), else HOLD. BUY candidates ranked by swing P(win). No live
execution — paper ledger only.

Run
  python scripts/predict_daily.py                 # latest cached date, halal universe
  python scripts/predict_daily.py --asof 2026-06-11 --top 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.config as cfg
from src.predictor import data, features, model
from src.predictor.adapters import news_adapter

MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
NEWS_DIR = ROOT / "data" / "features" / "news"
PAPER_LEDGER = ROOT / "data" / "paper_trades.jsonl"
HEADS = ["dir1d", "swing"]


def load_latest_model(head: str) -> tuple[str, dict] | tuple[None, None]:
    ptr = MODELS_DIR / head / "latest.txt"
    if not ptr.exists():
        return None, None
    run_id = ptr.read_text().strip()
    art = joblib.load(MODELS_DIR / head / run_id / "model.joblib")
    return run_id, art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["halal", "all"], default="halal")
    ap.add_argument("--asof", default=None, help="score as-of this date (ISO); default=latest cached")
    ap.add_argument("--top", type=int, default=25, help="how many ranked candidates to surface")
    ap.add_argument("--dir-thr", type=float, default=0.52, help="p_up gate for a BUY badge")
    ap.add_argument("--win-thr", type=float, default=0.30, help="p_win gate for a BUY badge")
    ap.add_argument("--no-news", action="store_true",
                    help="skip news/order-win enrichment (offline / fast mode)")
    args = ap.parse_args()

    t0 = time.time()
    print("[1/3] Building features (no labels) for inference …")
    d = data.load_universe(args.universe, asof=args.asof)
    nifty = data.load_nifty(asof=args.asof)
    panel = features.build_panel(d, nifty)
    asof = args.asof or str(panel["date"].max().date())

    # Latest bar per symbol on/at the as-of date (the row we predict FROM).
    asof_ts = pd.Timestamp(asof)
    latest = (panel[panel["date"] <= asof_ts]
              .sort_values("date").groupby("symbol", as_index=False).tail(1)
              .reset_index(drop=True))
    # Only keep symbols whose most-recent bar is actually the as-of date (fresh).
    latest = latest[latest["date"] == latest["date"].max()].reset_index(drop=True)
    print(f"      scoring {len(latest)} symbols as-of {asof}")

    print("[2/3] Loading models + scoring both heads …")
    runs, preds, arts = {}, {}, {}
    for head in HEADS:
        run_id, art = load_latest_model(head)
        if art is None:
            print(f"      ! no trained model for '{head}' — run train_predictor.py first")
            return
        runs[head], arts[head] = run_id, art
        preds[head] = model.predict_head(art, latest)
    latest["p_up"] = preds["dir1d"]
    latest["p_win"] = preds["swing"]

    dir_thr, win_thr = args.dir_thr, args.win_thr
    sec = data.sector_map()
    sl, tp = cfg.STOP_LOSS_PCT, cfg.TARGET_PCT

    def direction(row):
        if row.p_up >= dir_thr and row.p_win >= win_thr:
            return "BUY"
        if row.p_up <= (1 - dir_thr) and row.p_win <= win_thr / 2:
            return "AVOID"
        return "WATCH"

    latest["direction"] = latest.apply(direction, axis=1)

    # Reference levels off the last close (entry fills next open in live use).
    close_map = {sym: float(df["close"].iloc[-1]) for sym, df in d.items()}
    latest["ref_close"] = latest["symbol"].map(close_map)
    latest["stop"] = (latest["ref_close"] * (1 - sl / 100.0)).round(2)
    latest["target"] = (latest["ref_close"] * (1 + tp / 100.0)).round(2)
    latest["expected_value_pct"] = (latest["p_win"] * tp - (1 - latest["p_win"]) * sl
                                    - cfg.ROUND_TRIP_COST_PCT).round(2)

    # Always surface a ranked watchlist (top-N by P(win)) so the page is never blank;
    # the BUY/WATCH/AVOID badge tells the user the conviction. On a risk-off day the
    # list is all WATCH — that is the model honestly declining to buy.
    buys = latest.sort_values(["p_win", "p_up"], ascending=False).head(args.top)

    # Cheap "why": global top-importance features (swing head) + this row's values.
    gbt_imp = pd.Series(arts["swing"]["gbt"].feature_importances_,
                        index=features.FEATURE_COLS).sort_values(ascending=False)
    why_cols = list(gbt_imp.head(4).index)

    # News + order-win enrichment — only the surfaced candidates (data-cost rule).
    news_map = {}
    if not args.no_news:
        print("[2b] Fetching news/order-win for candidates …")
        cmap = {s.strip(): c.strip() for s, c in
                zip(pd.read_csv(data.STOCKS_CSV, dtype=str).fillna("")["stock"],
                    pd.read_csv(data.STOCKS_CSV, dtype=str).fillna("")["company_name"])}
        nsyms = buys["symbol"].tolist()
        ndf = news_adapter.fetch_news(nsyms, cmap, verbose=True)
        news_map = {r["symbol"]: r for _, r in ndf.iterrows()}
        NEWS_DIR.mkdir(parents=True, exist_ok=True)
        ndf.assign(date=asof).to_parquet(NEWS_DIR / f"{asof}.parquet", index=False)

    def _news_fields(sym):
        n = news_map.get(sym)
        if n is None:
            return {"news_sent": None, "news_count": 0, "order_win": 0,
                    "news_age_days": None, "headline": ""}
        return {"news_sent": None if pd.isna(n["news_sent"]) else float(n["news_sent"]),
                "news_count": int(n["news_count"]), "order_win": int(n["order_win_flag"]),
                "news_age_days": None if pd.isna(n["days_since_news"]) else float(n["days_since_news"]),
                "headline": str(n.get("top_headline", ""))[:140]}

    signals = []
    for rank, (_, r) in enumerate(buys.iterrows(), 1):
        signals.append({
            "symbol": r["symbol"], "market": "nse", "direction": r["direction"], "rank": rank,
            "p_up": round(float(r["p_up"]), 3), "p_win": round(float(r["p_win"]), 3),
            "agree": bool(r["direction"] == "BUY"),
            "expected_value_pct": float(r["expected_value_pct"]),
            "entry_type": "next_open", "ref_close": r["ref_close"],
            "stop": float(r["stop"]), "target": float(r["target"]),
            "sector": sec.get(r["symbol"], ""),
            "why": [[c, round(float(r[c]), 4) if pd.notna(r[c]) else None] for c in why_cols],
            **_news_fields(r["symbol"]),
        })

    OUT_DIR.mkdir(exist_ok=True)
    payload = {
        "date": asof, "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "model_runs": runs, "universe": args.universe,
        "thresholds": {"dir": dir_thr, "win": win_thr},
        "exit_rule": {"stop_pct": sl, "target_pct": tp, "max_hold": cfg.MAX_HOLD_DAYS},
        "counts": {"scored": int(len(latest)),
                   "buy": int((latest["direction"] == "BUY").sum()),
                   "avoid": int((latest["direction"] == "AVOID").sum()),
                   "watch": int((latest["direction"] == "WATCH").sum())},
        "signals": signals,
    }
    out = OUT_DIR / f"predicted_signals_{asof}.json"
    out.write_text(json.dumps(payload, indent=2))
    # Stable pointer the dashboard reads.
    (OUT_DIR / "predicted_signals_latest.json").write_text(json.dumps(payload, indent=2))

    # Paper ledger append — only genuine BUY badges (outcome filled later from data).
    with open(PAPER_LEDGER, "a", encoding="utf-8") as f:
        for s in signals:
            if s["direction"] != "BUY":
                continue
            f.write(json.dumps({"date": asof, "symbol": s["symbol"], "direction": "BUY",
                                "p_win": s["p_win"], "p_up": s["p_up"],
                                "ref_close": s["ref_close"], "stop": s["stop"],
                                "target": s["target"], "outcome": None}) + "\n")

    print("[3/3] Top signals:")
    c = payload["counts"]
    print(f"      scored={c['scored']}  BUY={c['buy']}  WATCH={c['watch']}  AVOID={c['avoid']}")
    if c["buy"] == 0:
        print("      (risk-off: no high-conviction BUYs today — showing ranked watchlist)")
    print(f"      {'RANK':>4s} {'SYMBOL':<14s} {'DIR':>5s} {'P_UP':>5s} {'P_WIN':>6s} {'EV%':>6s}  SECTOR")
    for s in signals[:15]:
        print(f"      {s['rank']:>4d} {s['symbol']:<14s} {s['direction']:>5s} {s['p_up']:>5.2f} "
              f"{s['p_win']:>6.2f} {s['expected_value_pct']:>6.2f}  {s['sector']}")
    print(f"\nWrote {out.relative_to(ROOT)}  (+ paper ledger)  in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
