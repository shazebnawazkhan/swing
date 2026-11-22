"""
src.watchlist_enrich
--------------------
Per-symbol enrichment for the personal watchlist dashboard.

    data/watchlist/enrich.json        {SYMBOL: {section: {...,"as_of","source"}}}
    data/watchlist/charts/<SYM>.json  price series for 1D / 1W / 1M..1Y / 5Y / All

Sections and sources (everything here runs from Python, so the dashboard's
"Refresh data" button and newly added stocks get it without Claude):

  news          Google News RSS, last 30 days, with article links + trigger tags
  target_news   Google News RSS for "<company> target price"; target parsed from headline
  triggers      NSE event calendar (board meetings) + NSE corporate actions (ex-dates)
                + trigger-tagged headlines
  delivery      NSE sec_bhavdata_full archives: delivery % for 1D, 1W (5d), 1M (~21d)
  oi            NSE F&O bhavcopy: stock-futures OI (all expiries), 1D/1W/1M change,
                and the price/OI buildup label. None = not an F&O stock.
  chart         yfinance: 5m (1D), 30m (1W), daily (1Y), weekly (max). A small watchlist
                is fine for Yahoo; this is NOT for universe-wide fan-out (see CLAUDE.md).
  shareholding  fallback only: NSE promoter vs public split. The Groww import
                (scripts/build_watchlist.py import-groww --shareholding) gives the full
                Promoter / FII / DII / Public split and takes precedence.
  quarterly     Groww import only (import-groww --financials): NSE's results-comparison
                API is stale and stockanalysis' quarterly route returns no data.

Every fetch degrades per symbol and per section: a failure keeps the previous data.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO, StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WL_DIR = ROOT / "data" / "watchlist"
ENRICH_PATH = WL_DIR / "enrich.json"
CHART_DIR = WL_DIR / "charts"

_UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
_RSS = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
_lock = threading.RLock()

# headline keyword -> trigger tag (first match wins per tag; a headline can carry several)
TRIGGER_TAGS = [
    ("results", r"\bq[1-4]\b|result|earnings|profit (?:rises|falls|jumps|drops)|quarterly"),
    ("dividend", r"dividend|record date"),
    ("bonus/split", r"\bbonus\b|stock split|split"),
    ("buyback", r"buy-?back"),
    ("order win", r"\border\b|contract|bags|wins|secures"),
    ("fund raise", r"\bqip\b|fund ?rais|rights issue|preferential|ncd"),
    ("rating", r"upgrade|downgrade|target price|\btarget\b|initiates? coverage|\bbuy\b|\bsell\b"),
    ("stake", r"block deal|bulk deal|stake|promoter (?:buy|sell|pledge)|pledge"),
    ("M&A", r"acqui|merger|demerger|takeover"),
    ("regulatory", r"\bsebi\b|penalty|probe|raid|\bed\b|income tax|show cause"),
]
_TAG_RE = [(t, re.compile(p, re.I)) for t, p in TRIGGER_TAGS]
_TARGET_RE = re.compile(r"(?:target(?: price)?|\bTP\b)[^0-9₹]{0,30}?(Rs\.?|₹|INR)?\s?([\d,]{2,7}(?:\.\d+)?)", re.I)
# "Rs 330 target" (price before the word) is checked before the target-then-price form
_TARGET_PRE_RE = re.compile(r"(Rs\.?|₹|INR)\s?([\d,]{2,7}(?:\.\d+)?)\s*(?:price\s*)?target", re.I)
# SEO "share price target 2026, 2027, 2030" forecast pages are not analyst calls
_SEO_TARGET_RE = re.compile(r"target\s*(?:for\s*)?20\d\d\b", re.I)


def _relevant(title: str, sym: str, name: str) -> bool:
    """Google News matches loosely; keep headlines that actually name the company."""
    t = title.lower()
    words = [w for w in re.findall(r"[a-z0-9&]+", name.lower()) if len(w) >= 4]
    return sym.lower() in t or (words and words[0] in t) or any(w in t for w in words[1:2])
_SUFFIX_RE = re.compile(r"\b(?:limited|ltd\.?|company|corporation|corp\.?|enterprises?)\b|\(india\)|\bindia$", re.I)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def short_name(company: str, symbol: str) -> str:
    """'Kalyan Jewellers India Limited' -> 'Kalyan Jewellers' (for news queries)."""
    n = (company or "").strip()
    for _ in range(2):
        n = _SUFFIX_RE.sub("", n).strip(" .,-&")
    n = " ".join(n.split())
    return n if len(n) >= 3 else symbol


# ── storage ─────────────────────────────────────────────────────────────────────

def load() -> dict:
    try:
        return json.loads(ENRICH_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def merge(sections: dict[str, dict]) -> None:
    """{SYM: {section: value}} -> enrich.json, section-level overwrite."""
    from src.watchlist_store import _write
    with _lock:
        cur = load()
        for sym, secs in sections.items():
            cur.setdefault(sym, {}).update(secs)
        _write(ENRICH_PATH, cur)


def load_chart(sym: str) -> dict | None:
    try:
        return json.loads((CHART_DIR / f"{sym}.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ── news ────────────────────────────────────────────────────────────────────────

def _rss(query: str, days: int, n: int, session) -> list[dict]:
    url = _RSS.format(q=urllib.parse.quote(query))
    raw = session.get(url, headers=_UA, timeout=12).content
    now = datetime.now(timezone.utc)
    out, seen = [], set()
    for it in ET.fromstring(raw).findall(".//item"):
        title = (it.findtext("title") or "").strip()
        src = (it.findtext("source") or "").strip()
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3].strip()
        try:
            dt = parsedate_to_datetime(it.findtext("pubDate"))
        except Exception:
            continue
        age = (now - dt).total_seconds() / 86400
        key = title.lower()[:70]
        if age < 0 or age > days or key in seen:
            continue
        seen.add(key)
        out.append({"title": title, "url": it.findtext("link"), "source": src,
                    "published": dt.astimezone().isoformat(timespec="minutes"),
                    "tags": [t for t, rx in _TAG_RE if rx.search(title)]})
    out.sort(key=lambda r: r["published"], reverse=True)
    return out[:n]


def fetch_news(sym: str, company: str, session, n: int = 8) -> dict:
    name = short_name(company, sym)
    items = [i for i in _rss(f'"{name}" share OR stock', 30, n * 2, session) if _relevant(i["title"], sym, name)]
    if len(items) < 3 and name != sym:               # thin coverage: widen to the ticker
        extra = _rss(f'"{sym}" NSE', 30, n, session)
        have = {i["title"].lower()[:70] for i in items}
        items += [i for i in extra if i["title"].lower()[:70] not in have and _relevant(i["title"], sym, name)]
    items = sorted(items, key=lambda r: r["published"], reverse=True)[:n]
    return {"items": items, "query": name, "as_of": _now(), "source": "Google News RSS"}


def fetch_target_news(sym: str, company: str, session, n: int = 5) -> dict:
    name = short_name(company, sym)
    items = _rss(f'"{name}" target price', 120, n * 2, session)
    keep = []
    for it in items:
        if ("target" not in it["title"].lower() or _SEO_TARGET_RE.search(it["title"])
                or not _relevant(it["title"], sym, name)):
            continue
        m = _TARGET_PRE_RE.search(it["title"]) or _TARGET_RE.search(it["title"])
        tgt = None
        if m:
            try:
                tgt = float(m.group(2).replace(",", ""))
            except ValueError:
                pass
            if tgt is not None and not m.group(1) and 2000 <= tgt <= 2100:
                tgt = None                       # a bare year ("target 2026"), not a price
        it["target"] = tgt
        keep.append(it)
    return {"items": keep[:n], "as_of": _now(), "source": "Google News RSS"}


# ── NSE: triggers, shareholding fallback ───────────────────────────────────────

def _nse_get(session, url: str):
    try:
        r = session.get(url, timeout=18)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _nse_date(s):
    for fmt in ("%d-%b-%Y", "%d-%b-%Y %H:%M:%S", "%d-%B-%Y"):
        try:
            return datetime.strptime(str(s).strip().title(), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def fetch_triggers(symbols: list[str], news: dict[str, dict], nse) -> dict[str, dict]:
    """Forward-looking events per symbol: board meetings, ex-dates, tagged headlines."""
    today = date.today()
    out = {s: [] for s in symbols}
    cal = _nse_get(nse, "https://www.nseindia.com/api/event-calendar") or []
    for r in cal if isinstance(cal, list) else []:
        s = (r.get("symbol") or "").upper()
        d = _nse_date(r.get("date"))
        if s in out and d and d >= today:
            out[s].append({"date": d.isoformat(), "kind": "board meeting",
                           "label": (r.get("purpose") or "Board meeting").strip(),
                           "detail": (r.get("bm_desc") or "").strip()[:200],
                           "url": "https://www.nseindia.com/companies-listing/corporate-filings-event-calendar"})
    for s in symbols:
        ca = _nse_get(nse, f"https://www.nseindia.com/api/corporates-corporateActions?index=equities&symbol={urllib.parse.quote(s)}")
        for r in ca if isinstance(ca, list) else []:
            d = _nse_date(r.get("exDate"))
            if d and d >= today:
                out[s].append({"date": d.isoformat(), "kind": "ex-date",
                               "label": (r.get("subject") or "Corporate action").strip(),
                               "detail": f"record date {r.get('recDate') or '-'}",
                               "url": "https://www.nseindia.com/companies-listing/corporate-filings-actions"})
        time.sleep(0.3)
    res = {}
    for s in symbols:
        tagged = [{"date": i["published"][:10], "kind": "news", "label": ", ".join(i["tags"]),
                   "detail": i["title"], "url": i["url"]}
                  for i in (news.get(s) or {}).get("items", []) if i["tags"]][:4]
        events = sorted(out[s], key=lambda e: e["date"])
        res[s] = {"events": events, "news_signals": tagged, "as_of": _now(),
                  "source": "NSE event calendar + corporate actions; tagged headlines"}
    return res


def fetch_nse_shareholding(sym: str, nse) -> dict | None:
    j = _nse_get(nse, f"https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={urllib.parse.quote(sym)}")
    if not isinstance(j, list) or not j:
        return None
    r = j[0]
    try:
        pro, pub = float(r.get("pr_and_prgrp")), float(r.get("public_val"))
    except (TypeError, ValueError):
        return None
    prev = None
    if len(j) > 1:
        try:
            prev = float(j[1].get("pr_and_prgrp"))
        except (TypeError, ValueError):
            pass
    return {"period": (r.get("date") or "").title(), "promoters": pro, "public": pub,
            "fii": None, "dii": None, "others": round(max(0.0, 100 - pro - pub), 2),
            "promoters_prev": prev, "as_of": _now(), "source": "NSE (promoter / public only)"}


# ── NSE archives: delivery % + futures OI (batched: one parse per day file) ────

def _days_back(n_calendar: int) -> list[datetime]:
    from src.data_fetcher import _trading_days
    end = datetime.now()
    return list(reversed(_trading_days(end - timedelta(days=n_calendar), end)))


def fetch_delivery(symbols: list[str]) -> dict[str, dict]:
    import pandas as pd
    from src.data_fetcher import NSEArchiveFetcher, _CACHE_DIR

    f = NSEArchiveFetcher()
    want = set(symbols)
    rows = {s: [] for s in symbols}
    got_days = 0
    for day in _days_back(45):
        ds = day.strftime("%d%m%Y")
        raw = f._cached_get(f.EQ_URL.format(date=ds), str(Path(_CACHE_DIR) / f"eq_{ds}.csv"))
        if raw is None:
            continue
        try:
            df = pd.read_csv(StringIO(raw.decode("utf-8", errors="ignore")))
            df.columns = [c.strip().upper() for c in df.columns]
            df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
            if "SERIES" in df.columns:
                df = df[df["SERIES"].astype(str).str.strip().isin(["EQ", "BE", "BZ", "SM", "ST"])]
            df = df[df["SYMBOL"].isin(want)]
        except Exception:
            continue
        got_days += 1
        for r in df.itertuples(index=False):
            d = r._asdict()
            num = lambda k: pd.to_numeric(str(d.get(k, "")).replace(",", "").strip(), errors="coerce")
            series = str(d.get("SERIES", "")).strip()
            vol, dq, dp = num("TTL_TRD_QNTY"), num("DELIV_QTY"), num("DELIV_PER")
            if series in ("BE", "BZ") and dq != dq:
                # trade-for-trade segment: every trade settles by delivery; NSE prints "-"
                dq, dp = vol, 100.0
            rows[d["SYMBOL"]].append({"date": day.date().isoformat(), "close": num("CLOSE_PRICE"),
                                      "prev": num("PREV_CLOSE"), "vol": vol, "dq": dq, "dp": dp,
                                      "series": series})
        if got_days >= 23:
            break
    out = {}
    for s, rs in rows.items():
        rs = [r for r in rs if r["vol"] and r["vol"] == r["vol"]]   # newest first; drop NaN volume
        if not rs:
            continue

        def window(n):
            w = [r for r in rs[:n] if r["dq"] == r["dq"]]
            v = sum(r["vol"] for r in w)
            return round(sum(r["dq"] for r in w) / v * 100, 2) if v else None

        d1 = rs[0]["dp"] if rs[0]["dp"] == rs[0]["dp"] else window(1)
        m1 = window(21)
        out[s] = {"d1": d1, "w1": window(5), "m1": m1, "date": rs[0]["date"], "days": len(rs),
                  "vol_d1": rs[0]["vol"], "vol_m1_avg": round(sum(r["vol"] for r in rs[:21]) / min(21, len(rs))),
                  "spike": bool(d1 and m1 and d1 >= 1.5 * m1 and rs[0]["series"] == "EQ"),
                  "series": rs[0]["series"],
                  "close": rs[0]["close"], "prev_close": rs[0]["prev"],
                  "as_of": _now(), "source": "NSE sec_bhavdata_full"}
    return out


def fetch_oi(symbols: list[str], price_chg: dict[str, float]) -> dict[str, dict | None]:
    """Stock-futures OI summed across expiries. Symbols absent from F&O map to None."""
    import pandas as pd
    from src.data_fetcher import NSEArchiveFetcher, _CACHE_DIR

    f = NSEArchiveFetcher()
    series = {s: [] for s in symbols}
    days_seen, fno = 0, None
    for day in _days_back(45):
        fs = day.strftime("%Y%m%d")
        raw = f._cached_get(f.FO_URL.format(date=fs), str(Path(_CACHE_DIR) / f"fo_{fs}.zip"))
        if raw is None:
            continue
        try:
            with zipfile.ZipFile(BytesIO(raw)) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".csv"))
                df = pd.read_csv(StringIO(zf.read(name).decode("utf-8", errors="ignore")),
                                 usecols=lambda c: c.strip() in ("TckrSymb", "FinInstrmTp", "OpnIntrst", "ChngInOpnIntrst"))
            df.columns = [c.strip() for c in df.columns]
            df = df[df["FinInstrmTp"].astype(str).str.upper().eq("STF")]
            df["TckrSymb"] = df["TckrSymb"].astype(str).str.strip()
        except Exception:
            continue
        days_seen += 1
        if fno is None:                      # newest file decides who is in F&O at all
            fno = set(df["TckrSymb"]) & set(symbols)
            if not fno:
                break
        g = df[df["TckrSymb"].isin(fno)].groupby("TckrSymb")[["OpnIntrst", "ChngInOpnIntrst"]].sum()
        for s, r in g.iterrows():
            series[s].append((day.date().isoformat(), float(r["OpnIntrst"]), float(r["ChngInOpnIntrst"])))
        if days_seen >= 23:
            break
    out: dict[str, dict | None] = {}
    for s in symbols:
        rs = series[s]
        if fno is not None and s not in fno:
            out[s] = None
            continue
        if not rs:
            continue
        oi, chg = rs[0][1], rs[0][2]

        def vs(n):
            return round((oi / rs[n][1] - 1) * 100, 2) if len(rs) > n and rs[n][1] else None

        d1 = round(chg / (oi - chg) * 100, 2) if oi - chg else None
        pc = price_chg.get(s)
        build = None
        if pc is not None and d1 is not None:
            build = ("long buildup" if pc > 0 and d1 > 0 else "short buildup" if pc < 0 and d1 > 0
                     else "short covering" if pc > 0 and d1 < 0 else "long unwinding" if pc < 0 and d1 < 0 else None)
        out[s] = {"oi": oi, "chg": chg, "d1_pct": d1, "w1_pct": vs(5), "m1_pct": vs(min(21, len(rs) - 1)) if len(rs) > 6 else None,
                  "buildup": build, "date": rs[0][0], "as_of": _now(), "source": "NSE F&O bhavcopy (stock futures, all expiries)"}
    return out


# ── charts (yfinance) ──────────────────────────────────────────────────────────

BENCH_SYM, BENCH_YAHOO, BENCH_NAME = "NIFTY50", "^NSEI", "NIFTY 50"


def fetch_chart(sym: str, yahoo: str | None = None) -> dict | None:
    """OHLCV bars -> charts/<SYM>.json as {i5: 5-min (5 sessions), d: daily (5y), w: weekly (max)}.

    Rows are [epoch_s, open, high, low, close, volume]. One source of truth: the dashboard's
    sparklines are derived from these (chart_closes) and the explorer reads them whole.
    """
    import logging

    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    tk = yf.Ticker(yahoo or f"{sym}.NS")
    out = {}
    for key, period, interval in (("i5", "5d", "5m"), ("d", "5y", "1d"), ("w", "max", "1wk")):
        try:
            h = tk.history(period=period, interval=interval, auto_adjust=True)
        except Exception:
            h = None
        if h is None or h.empty:
            continue
        h = h.dropna(subset=["Close"])
        out[key] = [[int(ts.timestamp()), round(float(r.Open), 2), round(float(r.High), 2), round(float(r.Low), 2),
                     round(float(r.Close), 2), int(r.Volume or 0)] for ts, r in h.iterrows()]
        time.sleep(0.25)
    if not out:
        return None
    out.update(v=2, as_of=_now(), source="Yahoo Finance via yfinance", yahoo=yahoo or f"{sym}.NS")
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    (CHART_DIR / f"{sym}.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    return out


_IST = timezone(timedelta(hours=5, minutes=30))


def chart_closes(raw: dict | None) -> dict | None:
    """Compact close-only series for sparklines: d1 (last session, 5m), w1 (5 sessions, 15m),
    y1 (daily, 1y), y5 (daily, 5y, every 2nd bar), all (weekly). Plus 52W hi/lo from
    daily highs/lows and the previous session's close (the 1D baseline)."""
    if not raw:
        return None
    if raw.get("v") != 2:                      # legacy close-only file
        return {k: raw.get(k) for k in ("d1", "w1", "y1", "all", "hi52", "lo52", "as_of", "source")}
    i5, d, w = raw.get("i5") or [], raw.get("d") or [], raw.get("w") or []
    c = lambda rows: [[r[0], r[4]] for r in rows]
    out = {"as_of": raw.get("as_of"), "source": raw.get("source")}
    if i5:
        day = lambda r: datetime.fromtimestamp(r[0], _IST).date()
        last = day(i5[-1])
        out["d1"] = c([r for r in i5 if day(r) == last])
        out["w1"] = c(i5[::3] + ([i5[-1]] if (len(i5) - 1) % 3 else []))
        # previous session close: daily bar before the intraday session's date
        prev = [r for r in d if datetime.fromtimestamp(r[0], _IST).date() < last]
        out["prev_close"] = prev[-1][4] if prev else None
    if d:
        cut1 = d[-1][0] - 366 * 86400
        y = [r for r in d if r[0] >= cut1]
        out["y1"] = c(y)
        out["y5"] = c(d[::2] + ([d[-1]] if (len(d) - 1) % 2 else []))
        out["hi52"], out["lo52"] = max(r[2] for r in y), min(r[3] for r in y)
    if w:
        out["all"] = c(w)
    return out


# ── orchestration ──────────────────────────────────────────────────────────────

ALL_PARTS = ("news", "targets", "triggers", "delivery", "oi", "chart", "shareholding")


def enrich(symbols: list[str], names: dict[str, str] | None = None, parts=ALL_PARTS,
           workers: int = 4, log=print) -> dict:
    """Fetch the requested sections for `symbols`, merge into enrich.json, return a summary."""
    import requests

    names = names or {}
    symbols = sorted(set(symbols))
    if not symbols:
        return {}
    summary: dict[str, int] = {}
    cur = load()
    local = threading.local()

    def sess():
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s

    news: dict[str, dict] = {}
    if "news" in parts or "targets" in parts or "triggers" in parts:
        log(f"news: {len(symbols)} symbols")

        def one(s):
            r = {}
            try:
                if "news" in parts or "triggers" in parts:
                    r["news"] = fetch_news(s, names.get(s, s), sess())
                if "targets" in parts:
                    r["target_news"] = fetch_target_news(s, names.get(s, s), sess())
            except Exception as e:
                log(f"  news {s} failed: {type(e).__name__}")
            return s, r

        with ThreadPoolExecutor(max_workers=workers) as ex:
            got = dict(f.result() for f in as_completed([ex.submit(one, s) for s in symbols]))
        merge({s: v for s, v in got.items() if v})
        news = {s: v.get("news") or cur.get(s, {}).get("news") for s, v in got.items()}
        summary["news"] = sum(1 for v in got.values() if v.get("news", {}).get("items"))

    nse = None
    if "triggers" in parts or "shareholding" in parts:
        from src.predictor.adapters.earnings_calendar_adapter import _nse_session
        log("nse: warming session")
        nse = _nse_session()
    if "triggers" in parts:
        log("triggers: NSE event calendar + corporate actions")
        trig = fetch_triggers(symbols, news, nse)
        merge({s: {"triggers": v} for s, v in trig.items()})
        summary["triggers"] = sum(1 for v in trig.values() if v["events"])
    if "shareholding" in parts:
        need = [s for s in symbols if (cur.get(s, {}).get("shareholding") or {}).get("source", "").startswith("NSE")
                or not cur.get(s, {}).get("shareholding")]
        log(f"shareholding (NSE fallback): {len(need)} symbols without Groww data")
        sh = {}
        for s in need:
            r = fetch_nse_shareholding(s, nse)
            if r:
                sh[s] = {"shareholding": r}
            time.sleep(0.3)
        merge(sh)
        summary["shareholding"] = len(sh)

    if "delivery" in parts or "oi" in parts:
        log("delivery: NSE bhavcopy archives (cached under src/.nse_cache)")
        dl = fetch_delivery(symbols)
        if "delivery" in parts:
            merge({s: {"delivery": v} for s, v in dl.items()})
            summary["delivery"] = len(dl)
        if "oi" in parts:
            log("oi: NSE F&O bhavcopy archives")
            pc = {s: (v["close"] / v["prev_close"] - 1) * 100 for s, v in dl.items()
                  if v.get("close") and v.get("prev_close")}
            oi = fetch_oi(symbols, pc)
            merge({s: {"oi": v} for s, v in oi.items()})
            summary["oi"] = sum(1 for v in oi.values() if v)

    if "chart" in parts:
        log(f"charts: yfinance for {len(symbols)} symbols + {BENCH_NAME}")
        ok = 0
        with ThreadPoolExecutor(max_workers=2) as ex:     # gentle on Yahoo
            futs = [ex.submit(fetch_chart, s) for s in symbols] + [ex.submit(fetch_chart, BENCH_SYM, BENCH_YAHOO)]
            for f in as_completed(futs):
                try:
                    ok += bool(f.result())
                except Exception:
                    pass
        summary["chart"] = ok
    log(f"done: {summary}")
    return summary


# ── background job (server.py is single-threaded: never block a request on this) ─

_job = {"running": False, "log": [], "started": None, "finished": None, "summary": None, "symbols": []}


def job_status() -> dict:
    with _lock:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in _job.items()}


def start_job(symbols: list[str], names: dict[str, str], parts=ALL_PARTS) -> bool:
    """Run enrich() on a daemon thread. Returns False if a job is already running."""
    with _lock:
        if _job["running"]:
            return False
        _job.update(running=True, log=[], started=_now(), finished=None, summary=None, symbols=sorted(symbols))

    def log(msg):
        with _lock:
            _job["log"].append(f"{datetime.now():%H:%M:%S} {msg}")

    def run():
        try:
            s = enrich(symbols, names, parts, log=log)
        except Exception as e:
            s = {"error": f"{type(e).__name__}: {e}"}
            log(s["error"])
        with _lock:
            _job.update(running=False, finished=_now(), summary=s)

    threading.Thread(target=run, daemon=True).start()
    return True


# ── Groww imports (raw MCP tool output -> sections) ────────────────────────────

_MON = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _period_key(p: str) -> tuple:
    """"Sep '26" -> (2026, 9)."""
    m = re.match(r"([A-Za-z]{3})\s*'(\d{2})", p or "")
    return (2000 + int(m.group(2)), _MON.get(m.group(1).title(), 0)) if m else (0, 0)


def parse_groww_shareholding(content: dict, as_of: str) -> dict[str, dict]:
    """{bse_code: shareholding section} from a view='shareholding_only' response."""
    out = {}
    for _slug, v in content.items():
        cid = (v.get("groww_contract_id") or "").replace("GSTK", "")
        sp = v.get("shareholding_pattern") or {}
        if not cid or not sp:
            continue
        periods = sorted(sp, key=_period_key, reverse=True)
        tot = lambda p, k: ((sp[p].get(k) or {}).get("totalPercent"))
        cur, prev = periods[0], periods[1] if len(periods) > 1 else None
        rec = {"period": cur, "prev_period": prev, "as_of": as_of, "source": "Groww"}
        for k, name in (("promoters", "promoters"), ("fii", "fii"), ("dii", "dii"), ("public", "public"), ("others", "others")):
            rec[name] = tot(cur, k)
            rec[name + "_prev"] = tot(prev, k) if prev else None
        rec["top_holders"] = {k: [h for h in (sp[cur].get(k) or {}).get("holders", []) if h.get("allocationPercent")][:3]
                              for k in ("promoters", "fii", "dii")}
        out[cid] = rec
    return out


def parse_groww_quarterly(content: dict, as_of: str) -> dict[str, dict]:
    """{bse_code: quarterly section} from a view='financials_only' response."""
    out = {}
    for _slug, v in content.items():
        cid = (v.get("groww_contract_id") or "").replace("GSTK", "")
        q = (((v.get("financialStatement") or {}).get("Income Statement") or {}).get("quarterlyFinancial") or {})
        rev = (q.get("Revenue") or {}).get("values") or {}
        if not cid or len(rev) < 2:
            continue
        periods = sorted(rev, key=_period_key)
        last, prev = periods[-1], periods[-2]
        yago = next((p for p in periods if _period_key(p) == (_period_key(last)[0] - 1, _period_key(last)[1])), None)

        def val(item, p):
            return ((q.get(item) or {}).get("values") or {}).get(p) if p else None

        def chg(item, a, b):
            x, y = val(item, a), val(item, b)
            return round((x / y - 1) * 100, 1) if x is not None and y not in (None, 0) and y > 0 else None

        out[cid] = {
            "period": last, "prev_period": prev, "yoy_period": yago,
            "revenue": val("Revenue", last), "revenue_prev": val("Revenue", prev),
            "revenue_qoq": chg("Revenue", last, prev), "revenue_yoy": chg("Revenue", last, yago),
            "profit": val("Net Profit", last), "profit_prev": val("Net Profit", prev),
            "profit_qoq": chg("Net Profit", last, prev), "profit_yoy": chg("Net Profit", last, yago),
            "ebitda_qoq": chg("EBITDA", last, prev),
            "opm": val("Operating Profit Margin", last), "opm_prev": val("Operating Profit Margin", prev),
            "npm": val("Net Profit Margin", last), "eps": val("Earning Per Share (Diluted)", last),
            "series": {"periods": periods, "revenue": [rev[p] for p in periods],
                       "profit": [val("Net Profit", p) for p in periods]},
            "unit": "₹ Cr", "as_of": as_of, "source": "Groww",
        }
    return out
