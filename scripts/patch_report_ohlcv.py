"""
patch_report_ohlcv.py
----------------------
Fetches OHLCV for each symbol in comparison_report.html, injects it as
stock.ohlcv, then re-writes the HTML using the current template from
run_comparison.py (which now includes the lightweight-charts panel).
"""
import json
import os as _os
import sys
from datetime import datetime, timedelta

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _os.path.join(_ROOT, "scripts"))

import pandas as pd

import src.config as cfg
from src.data_fetcher import NSEArchiveFetcher
import run_comparison as rc

HTML_PATH = "outputs/comparison_report.html"


def extract_data(html: str) -> dict:
    marker = "const DATA = "
    idx = html.index(marker)
    raw = html[idx + len(marker):]
    data, _ = json.JSONDecoder().raw_decode(raw)
    return data


def fetch_ohlcv(symbols: list[str]) -> dict[str, list[dict]]:
    to_dt = datetime.now()
    if getattr(cfg, "BACKTEST_START_DATE", None):
        floor_dt      = datetime.strptime(cfg.BACKTEST_START_DATE, "%Y-%m-%d")
        backtest_days = max(cfg.BACKTEST_DAYS, (to_dt - floor_dt).days + 1)
    else:
        backtest_days = cfg.BACKTEST_DAYS
    total_days = backtest_days + cfg.DATA_BUFFER_DAYS * 2
    from_dt   = to_dt - timedelta(days=total_days)
    cutoff    = to_dt - timedelta(days=backtest_days + cfg.DATA_BUFFER_DAYS)
    if getattr(cfg, "BACKTEST_START_DATE", None):
        cutoff = min(cutoff, floor_dt - timedelta(days=cfg.DATA_BUFFER_DAYS))

    fetcher = NSEArchiveFetcher()
    result: dict[str, list[dict]] = {}

    print(f"Fetching OHLCV for {len(symbols)} symbols…")
    for i, sym in enumerate(symbols, 1):
        print(f"  [{i}/{len(symbols)}] {sym}", end="  ", flush=True)
        try:
            df = fetcher.get_delivery_data(sym, from_dt, to_dt)
            if df.empty:
                print("NO DATA")
                result[sym] = []
                continue
            df = df[df["date"] >= cutoff].reset_index(drop=True)
            def _sf(val, prec=2):
                try:
                    v = float(val)
                    return round(v, prec) if v == v else None
                except (TypeError, ValueError):
                    return None
            def _si(val):
                try: return int(val)
                except (TypeError, ValueError): return None

            rows = []
            for _, r in df.iterrows():
                if not pd.notna(r.get("close")):
                    continue
                rows.append({
                    "time":         str(r["date"])[:10],
                    "open":         _sf(r.get("open",  r["close"])),
                    "high":         _sf(r.get("high",  r["close"])),
                    "low":          _sf(r.get("low",   r["close"])),
                    "close":        _sf(r["close"]),
                    "volume":       _si(r.get("total_volume", r.get("volume", 0))),
                    "delivery_pct": _sf(r.get("delivery_pct"), 2),
                    "delivery_qty": _si(r.get("delivery_qty")),
                    "vwap":         _sf(r.get("vwap")),
                    "oi":           _si(r.get("oi")),
                    "oi_change":    _si(r.get("oi_change")),
                })
            result[sym] = rows
            print(f"{len(rows)} candles")
        except Exception as exc:
            print(f"ERROR: {exc}")
            result[sym] = []

    return result


def main():
    with open(HTML_PATH, "r", encoding="utf-8") as fh:
        html = fh.read()

    data = extract_data(html)
    print(f"Loaded DATA: {len(data['strategies'])} strategies")

    # Collect all unique symbols
    symbols = sorted({
        stock["symbol"]
        for strat in data["strategies"]
        for stock in strat["stocks"]
    })

    ohlcv_map = fetch_ohlcv(symbols)

    # Inject ohlcv into every stock entry across all strategies
    for strat in data["strategies"]:
        for stock in strat["stocks"]:
            stock["ohlcv"] = ohlcv_map.get(stock["symbol"], [])

    payload = json.dumps(data, default=str)
    new_html = rc._REPORT_TEMPLATE.replace("__PAYLOAD__", payload)

    with open(HTML_PATH, "w", encoding="utf-8") as fh:
        fh.write(new_html)

    print(f"\nDone — {HTML_PATH} updated with OHLCV data and new chart template.")


if __name__ == "__main__":
    main()
