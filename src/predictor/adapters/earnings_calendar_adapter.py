"""
src.predictor.adapters.earnings_calendar_adapter
------------------------------------------------
Upcoming quarterly-RESULTS calendar for NSE, plus per-company enrichment with the
KPIs that move a stock around its result (revenue / profit / EPS growth, margins,
valuation) and the analyst-firm target price.

Unlike the fundamentals_adapter (which builds point-in-time BACKFILLABLE features
for the model), this adapter is a FORWARD-looking, current-snapshot view meant for
the results dashboard — "who reports next week, and what does the street expect."
Analyst targets and consensus growth are current-only on free sources, so nothing
here is used as a training feature; it is presentation data with source links.

Sources (all free, best-effort, graceful degradation — never raises for one symbol):
  * Results calendar  NSE forthcoming events / board-meetings JSON API, purpose
                      filtered to financial results. Fallback: yfinance earnings dates.
  * KPIs / growth     yfinance `.info` (revenueGrowth, earningsGrowth, margins,
                      EPS, PE) + upcoming-quarter EPS estimate vs the year-ago report.
  * Analyst targets   yfinance analyst price targets (mean/low/high, #analysts, reco).
  * Reference links   NSE quote, Yahoo Finance analysis, Screener.in — so every number
                      on the dashboard is traceable to a public page.

Public API
  fetch_results_calendar(start, end, session=None, verbose=True) -> DataFrame
      [symbol, company, date, purpose, source_url]
  enrich(symbol, company=None) -> dict            KPIs + target + reference links
  reference_links(symbol) -> dict                 the traceability URLs for one symbol
  build_calendar(start, end, company_map=None, halal=None, limit=None, ...) -> dict
      full dashboard payload, grouped by date
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.predictor.adapters import stockanalysis_adapter

DEFAULT_WORKERS = 8

# Purpose strings NSE uses for a quarterly-results board meeting. Matched case-insensitively
# as substrings, so "To consider and approve the unaudited financial results…" also hits.
_RESULT_KEYWORDS = ("financial result", "quarterly result", "unaudited result",
                    "audited result", "quarterly & audited", "results for the quarter")

_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")


# ── NSE session ───────────────────────────────────────────────────────────────

def _nse_session():
    """A requests.Session warmed with NSE's JS cookies (same pattern as data_fetcher)."""
    import requests

    s = requests.Session()
    s.headers.update({
        "User-Agent": _BROWSER_UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-board-meetings",
    })
    for page in ("https://www.nseindia.com",
                 "https://www.nseindia.com/companies-listing/corporate-filings-board-meetings"):
        try:
            s.get(page, timeout=12)
            time.sleep(1.0)
        except Exception:
            pass
    return s


_tls = threading.local()


def _sa_session():
    """
    Per-thread requests.Session for stockanalysis.com. A single Session is not safe to
    share across worker threads, so each thread lazily builds and reuses its own (same
    thread-local pattern as scripts/fetch_data.py).
    """
    if not hasattr(_tls, "sa"):
        import requests
        _tls.sa = requests.Session()
    return _tls.sa


def _is_result_purpose(purpose: str) -> bool:
    p = (purpose or "").lower()
    return any(k in p for k in _RESULT_KEYWORDS)


def _parse_nse_date(raw: str):
    """NSE serves dates as e.g. '07-Aug-2026'. Return a python date or None."""
    if not raw:
        return None
    for fmt in ("%d-%b-%Y", "%d-%b-%Y %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(raw).strip()[:len(fmt) + 4], fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(raw, dayfirst=True).date()
    except Exception:
        return None


# ── Results calendar (authoritative: NSE) ─────────────────────────────────────

def fetch_results_calendar(start: date, end: date, session=None,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Companies with a board meeting to approve financial results in [start, end].

    Tries NSE's forthcoming-events API, then the board-meetings API. Each record is
    normalised across the two endpoints' differing field names. Returns
    [symbol, company, date, purpose, source_url]; empty frame on total failure.
    """
    session = session or _nse_session()
    endpoints = [
        ("https://www.nseindia.com/api/event-calendar",
         {"symbol": "symbol", "company": "company", "date": "date", "purpose": "purpose"}),
        ("https://www.nseindia.com/api/corporate-board-meetings?index=equities",
         {"symbol": "bm_symbol", "company": "sm_name", "date": "bm_date", "purpose": "bm_purpose"}),
    ]

    rows = []
    for url, fmap in endpoints:
        try:
            resp = session.get(url, timeout=18)
            if resp.status_code != 200:
                continue
            data = resp.json()
        except Exception as exc:
            if verbose:
                print(f"      calendar: {url.split('/api/')[-1]} failed ({exc})")
            continue

        records = data if isinstance(data, list) else data.get("data", data)
        if not isinstance(records, list):
            continue

        for r in records:
            sym = (r.get(fmap["symbol"]) or "").strip().upper()
            purpose = r.get(fmap["purpose"]) or ""
            dt = _parse_nse_date(r.get(fmap["date"]))
            if not sym or dt is None or not _is_result_purpose(purpose):
                continue
            if not (start <= dt <= end):
                continue
            rows.append({
                "symbol": sym,
                "company": (r.get(fmap["company"]) or "").strip(),
                "date": dt,
                "purpose": purpose.strip(),
                "source_url": ("https://www.nseindia.com/companies-listing/"
                               "corporate-filings-board-meetings"),
            })
        if rows:
            break  # first endpoint that yields data wins

    if not rows:
        if verbose:
            print("      calendar: NSE returned no result-board-meetings "
                  "(blocked/empty) — trying yfinance fallback if a universe was given.")
        return pd.DataFrame(columns=["symbol", "company", "date", "purpose", "source_url"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol", "date"]).sort_values(["date", "symbol"])
    if verbose:
        print(f"      calendar: {len(df)} companies reporting between {start} and {end}")
    return df.reset_index(drop=True)


def fetch_calendar_yfinance(symbols: list[str], start: date, end: date,
                            company_map: dict | None = None, throttle: float = 0.3,
                            verbose: bool = True) -> pd.DataFrame:
    """
    Fallback calendar: scan a supplied universe for a yfinance next-earnings date in
    [start, end]. Only usable when NSE is blocked AND a symbol list is available, and
    only covers symbols yfinance tracks — hence a fallback, not the primary source.
    """
    import yfinance as yf

    company_map = company_map or {}
    rows = []
    for i, sym in enumerate(symbols, 1):
        try:
            ed = yf.Ticker(f"{sym}.NS").get_earnings_dates(limit=8)
        except Exception:
            ed = None
        if ed is not None and len(ed):
            fut = ed[ed["Reported EPS"].isna()] if "Reported EPS" in ed else ed
            for ts in fut.index:
                dt = pd.Timestamp(ts).date()
                if start <= dt <= end:
                    rows.append({"symbol": sym, "company": company_map.get(sym, ""),
                                 "date": dt, "purpose": "Financial Results (yfinance est.)",
                                 "source_url": f"https://finance.yahoo.com/quote/{sym}.NS"})
        if throttle:
            time.sleep(throttle)
        if verbose and i % 25 == 0:
            print(f"      calendar(yf) {i}/{len(symbols)} … ({len(rows)} hits)")
    return pd.DataFrame(rows, columns=["symbol", "company", "date", "purpose", "source_url"])


# ── Per-company enrichment ────────────────────────────────────────────────────

def reference_links(symbol: str) -> dict:
    """
    Current, India-specific pages that back the numbers shown for a symbol.

    Yahoo Finance's NSE coverage is stale/thin, so it is deliberately NOT used.

    Coverage reality: forward TARGETS only exist for analyst-followed stocks, and every
    India source that carries them (Trendlyne, stockanalysis, MoneyControl) hides the
    page behind an internal id or covers only large/mid-caps — so none can be linked
    from the ticker alone without 404ing on the small-caps in a full-NSE list. Hence:
      * NSE quote       — official board-meeting / results filing (universal, direct).
      * Target forecast — a Trendlyne-biased web search that always resolves and lands
                          on the CURRENT consensus target + estimates for this exact
                          ticker (Trendlyne Forecaster / stockanalysis / MoneyControl).
      * Screener        — reported quarterly results + fundamentals (near-universal, direct).
    """
    from urllib.parse import quote
    q = quote(symbol, safe="")   # encode '&', spaces etc. so URLs don't break
    target_query = quote(f"{symbol} NSE share price target forecast estimates trendlyne")
    return {
        "NSE filing": f"https://www.nseindia.com/get-quotes/equity?symbol={q}",
        "Target forecast": f"https://www.google.com/search?q={target_query}",
        "Quarterly (Screener)": f"https://www.screener.in/company/{q}/consolidated/",
    }


def _pct(x):
    """yfinance gives growth ratios as fractions (0.12 = +12%). Scale to % or None."""
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return round(float(x) * 100, 1)
    except (TypeError, ValueError):
        return None


def _num(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _expected_eps_growth(tk) -> tuple:
    """
    (expected_eps, expected_eps_growth_pct) for the UPCOMING quarter: the analyst EPS
    estimate for the next unreported quarter vs the reported EPS 4 quarters earlier.
    Returns (None, None) if the estimate or the year-ago actual is unavailable.
    """
    try:
        ed = tk.get_earnings_dates(limit=12)
    except Exception:
        return None, None
    if ed is None or not len(ed) or "EPS Estimate" not in ed:
        return None, None
    ed = ed.sort_index()
    future = ed[ed["Reported EPS"].isna() & ed["EPS Estimate"].notna()] if "Reported EPS" in ed else pd.DataFrame()
    if not len(future):
        return None, None
    est = float(future.iloc[0]["EPS Estimate"])
    reported = ed[ed["Reported EPS"].notna()] if "Reported EPS" in ed else pd.DataFrame()
    if len(reported) >= 4:
        year_ago = float(reported.iloc[-4]["Reported EPS"])
        if year_ago not in (0, None) and not np.isnan(year_ago):
            growth = (est - year_ago) / abs(year_ago) * 100
            return round(est, 2), round(growth, 1)
    return round(est, 2), None


def enrich(symbol: str, company: str | None = None, verbose: bool = False) -> dict:
    """
    Current KPI snapshot + analyst target + reference links for one symbol.

    Returns a dict of price-relevant KPIs (revenue / earnings / EPS growth, margins,
    valuation) and the street's price target. Every field degrades to None on failure;
    the function never raises for a single symbol.
    """
    out = {
        "symbol": symbol, "company": company or "",
        "current_price": None, "expected_eps": None, "expected_eps_growth_pct": None,
        "revenue_growth_pct": None, "earnings_growth_pct": None, "earnings_q_growth_pct": None,
        "profit_margin_pct": None, "gross_margin_pct": None,
        "trailing_pe": None, "forward_pe": None, "trailing_eps": None, "forward_eps": None,
        # Forward analyst estimates (primary: stockanalysis.com; see below).
        "fwd_rev_growth_pct": None, "fwd_eps_growth_pct": None, "fwd_eps_est": None,
        "target_mean": None, "target_low": None, "target_high": None,
        "n_analysts": None, "recommendation": None, "target_upside_pct": None,
        "target_source": None, "target_as_of": None, "kpi_source": None,
        "operating_margin_pct": None, "market_cap": None,
        "refs": reference_links(symbol),
    }

    # ── KPIs. PRIMARY: stockanalysis.com ──────────────────────────────────────
    # Yahoo cannot carry a full-universe run: under thread fan-out it starts returning
    # 401 "Invalid Crumb" and drops the majority of symbols (a 916-name run came back
    # ~44% filled, degrading the further it got). The stockanalysis quote/statistics
    # routes answer for essentially every listed NSE name, so they lead and yfinance is
    # only consulted for what they leave blank.
    kpi = {}
    try:
        kpi = stockanalysis_adapter.fetch_kpis(symbol, session=_sa_session()) or {}
    except Exception:
        kpi = {}

    if kpi:
        out["kpi_source"] = "stockanalysis"
        for key in ("current_price", "revenue_growth_pct", "earnings_growth_pct",
                    "trailing_eps", "trailing_pe", "forward_pe", "profit_margin_pct",
                    "gross_margin_pct", "operating_margin_pct", "market_cap"):
            if kpi.get(key) is not None:
                out[key] = kpi[key]

    # yfinance is now a gap-filler: skip it entirely when stockanalysis answered with a
    # price, which both avoids Yahoo's rate limiter and keeps one source per row.
    need_yf = not kpi or out["current_price"] is None
    tk = None
    info = {}
    if need_yf:
        try:
            import yfinance as yf
            tk = yf.Ticker(f"{symbol}.NS")
            info = tk.info or {}
        except Exception:
            info = {}

        if info:
            out["kpi_source"] = "stockanalysis+yfinance" if kpi else "yfinance"
        for key, val in (
            ("current_price", _num(info.get("currentPrice") or info.get("regularMarketPrice"))),
            ("revenue_growth_pct", _pct(info.get("revenueGrowth"))),
            ("earnings_growth_pct", _pct(info.get("earningsGrowth"))),
            ("profit_margin_pct", _pct(info.get("profitMargins"))),
            ("gross_margin_pct", _pct(info.get("grossMargins"))),
            ("trailing_pe", _num(info.get("trailingPE"))),
            ("forward_pe", _num(info.get("forwardPE"))),
            ("trailing_eps", _num(info.get("trailingEps"))),
            ("forward_eps", _num(info.get("forwardEps"))),
        ):
            if out.get(key) is None and val is not None:
                out[key] = val
        # Quarterly YoY earnings growth has no stockanalysis equivalent on these routes.
        out["earnings_q_growth_pct"] = _pct(info.get("earningsQuarterlyGrowth"))

    if company is None or not company:
        out["company"] = info.get("longName") or info.get("shortName") or ""

    # Analyst price target + forward growth.
    # PRIMARY: stockanalysis.com (India-specific, S&P Global consensus, carries a dated
    # freshness stamp and next-FY growth estimates). FALLBACK: yfinance/Yahoo, which is
    # thinner and undated. Whichever supplies the target is recorded in `target_source`.
    fc = {}
    try:
        fc = stockanalysis_adapter.fetch_forecast(symbol, session=_sa_session()) or {}
    except Exception:
        fc = {}

    if fc.get("target_mean") is not None:
        out["target_mean"] = fc.get("target_mean")
        out["target_low"] = fc.get("target_low")
        out["target_high"] = fc.get("target_high")
        out["n_analysts"] = fc.get("n_analysts")
        out["recommendation"] = fc.get("recommendation") or out["recommendation"]
        out["target_source"] = "stockanalysis"
        out["target_as_of"] = fc.get("target_as_of")
    elif kpi.get("target_mean") is not None:
        # The quote route carries a consensus target for many names the forecast route
        # does not cover — no low/high band or `updated` stamp, but a real number.
        out["target_mean"] = kpi["target_mean"]
        out["n_analysts"] = kpi.get("n_analysts")
        out["recommendation"] = kpi.get("recommendation") or out["recommendation"]
        out["target_source"] = "stockanalysis_quote"
    else:
        # yfinance fallback for names neither stockanalysis route covers.
        tgt = {}
        if tk is not None:
            try:
                tgt = tk.get_analyst_price_targets() or {}
            except Exception:
                tgt = {}
        out["target_mean"] = _num(tgt.get("mean") or info.get("targetMeanPrice"))
        out["target_low"] = _num(tgt.get("low") or info.get("targetLowPrice"))
        out["target_high"] = _num(tgt.get("high") or info.get("targetHighPrice"))
        out["n_analysts"] = info.get("numberOfAnalystOpinions") or kpi.get("n_analysts")
        out["recommendation"] = info.get("recommendationKey") or kpi.get("recommendation")
        if out["target_mean"] is not None:
            out["target_source"] = "yfinance"

    # Forward growth estimates: stockanalysis next-FY numbers when available.
    out["fwd_rev_growth_pct"] = fc.get("fwd_rev_growth_pct")
    out["fwd_eps_growth_pct"] = fc.get("fwd_eps_growth_pct")
    out["fwd_eps_est"] = fc.get("fwd_eps_est")

    if out["target_mean"] and out["current_price"]:
        out["target_upside_pct"] = round(
            (out["target_mean"] - out["current_price"]) / out["current_price"] * 100, 1)

    # Upcoming-quarter EPS estimate + YoY growth (yfinance) — supplementary column, and
    # only available when we had a reason to build a Ticker at all.
    if tk is not None:
        exp_eps, exp_growth = _expected_eps_growth(tk)
        out["expected_eps"] = exp_eps
        out["expected_eps_growth_pct"] = exp_growth
    return out


# ── Orchestration: the dashboard payload ──────────────────────────────────────

def build_calendar(start: date, end: date, company_map: dict | None = None,
                   halal: set | None = None, universe: list | None = None,
                   limit: int | None = None, throttle: float = 0.0,
                   workers: int = DEFAULT_WORKERS, verbose: bool = True) -> dict:
    """
    Assemble the full dashboard payload for [start, end], grouped by date.

    company_map : symbol -> company name (from stocks.csv) to fill NSE blanks + tag "mine".
    halal       : set of halal symbols, for the per-row halal tag.
    universe    : symbol list for the yfinance calendar fallback (used only if NSE fails).
    limit       : cap the number of enriched symbols (dev/rate-limit guard).
    workers     : thread-pool size for the (network-bound) per-symbol enrichment.
    throttle    : optional per-symbol politeness delay INSIDE each worker (0 = none);
                  the effective request rate is bounded by `workers`, not this.

    Returns {generated, start, end, counts, dates:[{date, weekday, rows:[...]}]}.
    """
    company_map = company_map or {}
    halal = halal or set()

    cal = fetch_results_calendar(start, end, verbose=verbose)
    if cal.empty and universe:
        cal = fetch_calendar_yfinance(universe, start, end, company_map, verbose=verbose)

    if not cal.empty and limit:
        keep = cal["symbol"].drop_duplicates().head(limit)
        cal = cal[cal["symbol"].isin(keep)]

    syms = list(cal["symbol"].unique()) if not cal.empty else []
    if verbose:
        print(f"      enriching {len(syms)} companies (KPIs + analyst targets) "
              f"across {workers} threads…")

    def _work(sym: str):
        if throttle:
            time.sleep(throttle)
        return sym, enrich(sym, company_map.get(sym) or None)

    # Enrichment is network-bound (yfinance + stockanalysis), so fan it out over a
    # thread pool; each worker uses its own thread-local stockanalysis session.
    enriched, done = {}, 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_work, s) for s in syms]
        for fut in as_completed(futures):
            sym, e = fut.result()
            enriched[sym] = e
            done += 1
            if verbose and (done % 20 == 0 or done == len(syms)):
                print(f"      enrich {done}/{len(syms)} …", flush=True)

    # Group into one bucket per calendar date.
    buckets = {}
    for r in cal.itertuples() if not cal.empty else []:
        e = enriched.get(r.symbol, {})
        company = r.company or e.get("company") or company_map.get(r.symbol, "")
        row = {
            "symbol": r.symbol, "company": company,
            "purpose": r.purpose, "source_url": r.source_url,
            "in_universe": r.symbol in company_map, "halal": r.symbol in halal,
            **{k: v for k, v in e.items() if k not in ("symbol", "company")},
        }
        buckets.setdefault(r.date, []).append(row)

    dates = []
    for d in sorted(buckets):
        rows = sorted(buckets[d], key=lambda x: (x.get("target_upside_pct") is None,
                                                 -(x.get("target_upside_pct") or 0)))
        dates.append({
            "date": d.isoformat(),
            "weekday": d.strftime("%A"),
            "label": d.strftime("%a, %d %b"),
            "count": len(rows),
            "rows": rows,
        })

    total = sum(len(b) for b in buckets.values())
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "start": start.isoformat(), "end": end.isoformat(),
        "counts": {
            "companies": total,
            "days": len(dates),
            "in_universe": sum(1 for d in dates for r in d["rows"] if r["in_universe"]),
            "halal": sum(1 for d in dates for r in d["rows"] if r["halal"]),
        },
        "dates": dates,
    }
