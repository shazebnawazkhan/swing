"""
src.predictor.adapters.stockanalysis_adapter
---------------------------------------------
Fresh, India-specific analyst forecasts for NSE stocks from stockanalysis.com —
the forward-looking numbers yfinance/Yahoo serves stale or not at all.

stockanalysis.com is a SvelteKit app whose routes expose structured server-data
endpoints:
    https://stockanalysis.com/quote/nse/<SYMBOL>/forecast/__data.json
    https://stockanalysis.com/quote/nse/<SYMBOL>/__data.json             (quote/KPIs)
    https://stockanalysis.com/quote/nse/<SYMBOL>/statistics/__data.json  (margins)
That JSON uses SvelteKit's "devalue" flat-array format (each value is an index into
one big array); `_deref` rebuilds the nested object.

Two levels of coverage, and they are very different:
  * FORECAST (price target, next-FY estimates) — analyst-followed large/mid-caps only;
    unfollowed small-caps 404 or return no targets.
  * QUOTE + STATISTICS (growth, EPS, PE, margins) — essentially every listed NSE name,
    micro-caps included. This is why they are the PRIMARY KPI source: Yahoo/yfinance
    rate-limits hard under fan-out (HTTP 401 "Invalid Crumb") and silently drops the
    majority of a full-universe run, while these endpoints answer reliably.

Graceful: any failure yields {} (no raise).

Public API
  fetch_forecast(symbol) -> dict     target + forward-growth estimates for one symbol
  fetch_kpis(symbol)     -> dict     price/growth/EPS/PE/margin snapshot for one symbol
  FORECAST_FIELDS / KPI_FIELDS       the keys each populates
"""

from __future__ import annotations

import re

import requests

FORECAST_FIELDS = ["target_mean", "target_low", "target_high", "n_analysts",
                   "recommendation", "target_currency", "target_as_of",
                   "fwd_rev_growth_pct", "fwd_eps_growth_pct", "fwd_eps_est"]

KPI_FIELDS = ["current_price", "revenue_growth_pct", "earnings_growth_pct",
              "eps_growth_pct", "trailing_eps", "trailing_pe", "forward_pe",
              "profit_margin_pct", "gross_margin_pct", "operating_margin_pct",
              "market_cap", "recommendation", "n_analysts", "target_mean",
              "target_upside_pct", "earnings_date"]

_URL = "https://stockanalysis.com/quote/nse/{sym}/forecast/__data.json"
_QUOTE_URL = "https://stockanalysis.com/quote/nse/{sym}/__data.json"
_STATS_URL = "https://stockanalysis.com/quote/nse/{sym}/statistics/__data.json"
_UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
       "Accept": "application/json"}


def _deref(arr: list):
    """
    Rebuild a devalue flat array into nested objects.

    In SvelteKit's format the payload is one big list; a dict/list holds *indices*
    into that list rather than inline values. We resolve arr[0] (the root) following
    every index, with a visited-guard against the format's self-references.
    """
    def walk(i, seen):
        if not isinstance(i, int) or i < 0 or i >= len(arr) or i in seen:
            return None if isinstance(i, int) else i
        seen = seen | {i}
        v = arr[i]
        if isinstance(v, dict):
            return {k: walk(x, seen) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x, seen) for x in v]
        return v
    return walk(0, frozenset())


def _page_root(url: str, timeout: int, session):
    """
    Fetch a `__data.json` route and deref its page node.

    Layout nodes (session/cookies/theme) come first and are small; the page's own data
    is the largest `data` list, so we pick by length. Returns None on any failure.
    """
    get = (session or requests).get
    try:
        r = get(url, headers=_UA, timeout=timeout)
        if r.status_code != 200:
            return None
        nodes = r.json().get("nodes", [])
    except Exception:
        return None
    best, best_len = None, -1
    for n in nodes:
        d = n.get("data")
        if isinstance(d, list) and len(d) > best_len:
            best, best_len = d, len(d)
    if best is None:
        return None
    try:
        root = _deref(best)
    except Exception:
        return None
    return root if isinstance(root, dict) else None


def _snum(v):
    """
    Number out of stockanalysis's display strings: '29.85', '1,234.5', '36.63%',
    '415.96B'. 'n/a'/None/'' -> None. Suffixes K/M/B/T are expanded.
    """
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.strip().replace(",", "").replace("₹", "")
    if not s or s.lower() in ("n/a", "na", "-", "--"):
        return None
    s = s.rstrip("%")
    mult = 1.0
    if s and s[-1] in "KMBT":
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[s[-1]]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


_TARGET_RE = re.compile(r"^\s*([\d,.]+)\s*(?:\(\s*([+-]?[\d.]+)%\s*\))?")


def _parse_target(v):
    """'674.13 (+29.64%)' -> (674.13, 29.64). 'n/a' -> (None, None)."""
    if not isinstance(v, str):
        return None, None
    m = _TARGET_RE.match(v.replace("₹", ""))
    if not m:
        return None, None
    return _snum(m.group(1)), (_snum(m.group(2)) if m.group(2) else None)


def _last_close(root: dict):
    """
    Latest close from the quote page's 1-year chart.

    Some names carry no chart series at all; there the 1-year change plus the price a
    year ago reconstructs it (`price1y * (1 + ch1y/100)`), which is what the site's own
    header does. Returns None when neither is available.
    """
    chart = root.get("chart")
    pts = chart.get("data") if isinstance(chart, dict) else None
    if isinstance(pts, list):
        for p in reversed(pts):
            if isinstance(p, dict) and isinstance(p.get("c"), (int, float)):
                return round(float(p["c"]), 2)
    changes = root.get("changes")
    base = (changes or {}).get("price1y") if isinstance(changes, dict) else None
    ch1y = _snum(root.get("ch1y"))
    if isinstance(base, (int, float)) and ch1y is not None:
        return round(float(base) * (1.0 + ch1y / 100.0), 2)
    return None


def _margins(root: dict) -> dict:
    """Margin block of the statistics page -> {id: pct}. Missing/n-a ids are skipped."""
    block = root.get("margins") if isinstance(root, dict) else None
    data = block.get("data") if isinstance(block, dict) else None
    if not isinstance(data, list):
        return {}
    out = {}
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            val = _snum(item.get("hover") or item.get("value"))
            if val is not None:
                out[item["id"]] = round(val, 2)
    return out


def fetch_kpis(symbol: str, timeout: int = 15, session: requests.Session | None = None,
               with_margins: bool = True) -> dict:
    """
    Current KPI snapshot for one NSE symbol — the yfinance-`info` replacement.

    One request to the quote route (price, revenue/earnings growth, EPS, PE, rating,
    consensus target) plus, when `with_margins`, one to the statistics route for
    gross/operating/profit margins. Returns {} if the symbol is not covered; individual
    fields are omitted rather than None-filled, so callers can layer a fallback.
    """
    root = _page_root(_QUOTE_URL.format(sym=symbol), timeout, session)
    if not root or "revenueGrowth" not in root:
        return {}

    out: dict = {}

    def put(key, val):
        if val is not None:
            out[key] = val

    put("current_price", _last_close(root))
    put("revenue_growth_pct", _round(_snum(root.get("revenueGrowth")), 1))
    put("earnings_growth_pct", _round(_snum(root.get("netIncomeGrowth")), 1))
    put("eps_growth_pct", _round(_snum(root.get("epsGrowth")), 1))
    put("trailing_eps", _round(_snum(root.get("eps")), 2))
    put("trailing_pe", _round(_snum(root.get("peRatio")), 2))
    put("forward_pe", _round(_snum(root.get("forwardPE")), 2))
    put("market_cap", _snum(root.get("marketCap")))
    put("earnings_date", root.get("earningsDate") or None)

    rating = root.get("analysts")
    if isinstance(rating, str) and rating.strip() and rating.strip().lower() != "n/a":
        out["recommendation"] = rating.strip().lower().replace(" ", "_")

    # The rating histogram doubles as the analyst count when the forecast route is thin.
    hist = root.get("analystChart")
    if isinstance(hist, dict):
        n = sum(v for v in hist.values() if isinstance(v, int))
        if n > 0:
            out["n_analysts"] = n

    tgt, upside = _parse_target(root.get("target"))
    put("target_mean", tgt)
    put("target_upside_pct", upside)

    if with_margins:
        stats = _page_root(_STATS_URL.format(sym=symbol), timeout, session)
        if stats:
            m = _margins(stats)
            put("gross_margin_pct", m.get("grossMargin"))
            put("operating_margin_pct", m.get("operatingMargin"))
            put("profit_margin_pct", m.get("profitMargin"))
    return out


def _round(v, nd):
    return None if v is None else round(v, nd)


def _first_fy(chart: dict):
    """First non-PRO fiscal-year bucket from an estimatesCharts sub-object (next FY)."""
    if not isinstance(chart, dict):
        return None
    for _, v in sorted(chart.items()):
        if isinstance(v, dict) and isinstance(v.get("avg"), (int, float)):
            return v
    return None


def fetch_forecast(symbol: str, timeout: int = 15, session: requests.Session | None = None) -> dict:
    """
    Analyst price target + next-FY growth estimates for one NSE symbol.

    Returns a dict with FORECAST_FIELDS (any missing field omitted), or {} when the
    stock is not covered / the request fails. `target_as_of` is stockanalysis's own
    "updated" date, so callers can show how fresh the consensus is.
    """
    get = (session or requests).get
    try:
        r = get(_URL.format(sym=symbol), headers=_UA, timeout=timeout)
        if r.status_code != 200:
            return {}
        payload = r.json()
    except Exception:
        return {}

    # Find the route node whose root carries the forecast keys, then deref it.
    node = None
    for n in payload.get("nodes", []):
        data = n.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict) \
                and ("priceTargets" in data[0] or "targets" in data[0]):
            node = data
            break
    if node is None:
        return {}
    try:
        root = _deref(node)
    except Exception:
        return {}

    out: dict = {}
    pt = root.get("priceTargets") or {}
    tg = root.get("targets") or {}
    # priceTargets (S&P Global) is the reliable consensus; `targets` is the fallback block.
    avg = pt.get("avg") if isinstance(pt.get("avg"), (int, float)) else tg.get("average")
    if isinstance(avg, (int, float)):
        out["target_mean"] = round(float(avg), 2)
        lo = pt.get("low") if pt.get("low") is not None else tg.get("low")
        hi = pt.get("high") if pt.get("high") is not None else tg.get("high")
        if isinstance(lo, (int, float)):
            out["target_low"] = round(float(lo), 2)
        if isinstance(hi, (int, float)):
            out["target_high"] = round(float(hi), 2)
        n = pt.get("numPriceTargets") or tg.get("count")
        if isinstance(n, int) and n > 0:
            out["n_analysts"] = n
        out["target_currency"] = pt.get("currency") or tg.get("currency") or "INR"
        out["target_as_of"] = tg.get("updated") or None

    ratings = root.get("currentRatings") or {}
    if ratings.get("consensus"):
        out["recommendation"] = str(ratings["consensus"]).lower().replace(" ", "_")

    charts = root.get("estimatesCharts") or {}
    rev = _first_fy(charts.get("revenueGrowth"))
    eps = _first_fy(charts.get("epsGrowth"))
    eps_est = _first_fy(charts.get("eps"))
    if rev:
        out["fwd_rev_growth_pct"] = round(float(rev["avg"]), 1)
    if eps:
        out["fwd_eps_growth_pct"] = round(float(eps["avg"]), 1)
    if eps_est:
        out["fwd_eps_est"] = round(float(eps_est["avg"]), 2)
    return out
