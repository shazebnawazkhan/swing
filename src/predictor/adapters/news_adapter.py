"""
src.predictor.adapters.news_adapter
-----------------------------------
Google News RSS per symbol → keyword-scored sentiment + order-win event flags
(docs/PREDICTOR.md §3, hypotheses hp_002). Free, best-effort, no API key, no LLM.

Graceful degradation: any network/parse failure for a symbol yields a null feature
row (never raises). Results are stored dated so the feature accrues forward into a
point-in-time history (`data/features/news/<asof>.parquet`).

Public API
  fetch_symbol(name, company=None) -> dict        recent-news features for one symbol
  fetch_news(symbols, company_map, ...) -> DataFrame   batch, with telemetry
  NEWS_FEATURES                                   the feature columns it contributes
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

from src.predictor.adapters.sentiment_words import score_headline

NEWS_FEATURES = ["news_sent", "news_count", "order_win_flag", "days_since_news"]
_HALF_LIFE_DAYS = 5.0        # recency weighting: older headlines count less
_RSS = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
_UA = {"User-Agent": "Mozilla/5.0 (compatible; swing-predictor/0.1)"}


def _empty_row(symbol: str) -> dict:
    return {"symbol": symbol, "news_sent": np.nan, "news_count": 0,
            "order_win_flag": 0, "days_since_news": np.nan, "top_headline": ""}


def fetch_symbol(symbol: str, company: str | None = None, lookback_days: int = 21,
                 timeout: int = 12) -> dict:
    """Recent-news features for one symbol. Returns a null row on any failure."""
    query = f'"{company or symbol}" stock NSE'
    url = _RSS.format(q=urllib.parse.quote(query))
    try:
        req = urllib.request.Request(url, headers=_UA)
        raw = urllib.request.urlopen(req, timeout=timeout).read()
        items = ET.fromstring(raw).findall(".//item")
    except Exception:
        return _empty_row(symbol)

    now = datetime.now(timezone.utc)
    rows, w_sent, w_sum, order_win, newest = [], 0.0, 0.0, 0, None
    for it in items:
        title = (it.findtext("title") or "").strip()
        pub = it.findtext("pubDate")
        try:
            dt = parsedate_to_datetime(pub) if pub else None
        except Exception:
            dt = None
        if dt is None:
            continue
        age = (now - dt).total_seconds() / 86400.0
        if age < 0 or age > lookback_days:
            continue
        s = score_headline(title)
        weight = 0.5 ** (age / _HALF_LIFE_DAYS)        # recency decay
        signed = s["pos"] - s["neg"]
        if signed != 0:
            w_sent += weight * np.sign(signed)
            w_sum += weight
        order_win = max(order_win, s["order_win"])
        if newest is None or dt > newest:
            newest = dt
            top = title
        rows.append(1)

    if not rows:
        return _empty_row(symbol)
    return {
        "symbol": symbol,
        "news_sent": round(float(w_sent / w_sum), 3) if w_sum > 0 else 0.0,
        "news_count": int(len(rows)),
        "order_win_flag": int(order_win),
        "days_since_news": round((now - newest).total_seconds() / 86400.0, 1) if newest else np.nan,
        "top_headline": top if rows else "",
    }


def fetch_news(symbols: list[str], company_map: dict[str, str] | None = None,
               lookback_days: int = 21, throttle: float = 0.3, verbose: bool = True) -> pd.DataFrame:
    """
    Fetch + score news for a list of symbols (intended for the daily candidate set,
    NOT the whole universe — that is the data-cost optimization). Returns a DataFrame
    with NEWS_FEATURES + top_headline + `news_age_days` staleness, and prints telemetry.
    """
    company_map = company_map or {}
    out, t0, ok = [], time.time(), 0
    for i, sym in enumerate(symbols, 1):
        row = fetch_symbol(sym, company_map.get(sym), lookback_days)
        out.append(row)
        if row["news_count"] > 0:
            ok += 1
        if throttle:
            time.sleep(throttle)
        if verbose and i % 10 == 0:
            print(f"      news {i}/{len(symbols)} …")
    df = pd.DataFrame(out)
    df["news_age_days"] = df["days_since_news"]
    secs = time.time() - t0
    if verbose:
        cov = ok / max(len(symbols), 1) * 100
        print(f"      news: {len(symbols)} symbols, {ok} with items ({cov:.0f}% coverage), "
              f"{int(df['order_win_flag'].sum())} order-win flags, {secs:.1f}s")
    return df
