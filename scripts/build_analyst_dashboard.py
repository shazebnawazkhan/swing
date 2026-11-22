"""Build the NSE Analyst Intelligence dashboard from a Groww MCP snapshot.

Two inputs, kept deliberately separate because they age at very different rates:

1. **Market data** -- ``data/analyst_dashboard/groww_snapshot_<date>.json``.
   Price, day change, PE/PB/ROE/EPS/div-yield/industry-PE/D-E, market cap, all
   from Groww MCP (``get_ltp`` + ``fetch_stocks_fundamental_data``). Groww MCP is
   an MCP server (not a Python API), so the snapshot is produced by Claude
   calling those tools; ``ingest`` below turns the raw tool responses into the
   snapshot format so a refresh is: call tools -> save raw JSON -> ingest -> build.

2. **Analyst inputs** -- ``data/analyst_dashboard/analyst_inputs.json``.
   Sector, broker ratings/targets, bulk deals and the thesis text. Groww MCP has
   no analyst-target or bulk-deal endpoint, so these are carried forward from the
   previous dashboard (bootstrapped with ``extract``) together with that build's
   price/upside as a ``baseline`` -- which is what lets the new build show the
   move since the last version and audit the old KPIs.

Upside is always RECOMPUTED from the live price; the stored upside from the old
dashboard is used only for the audit.

Usage::

    # one-off: pull analyst inputs + baseline out of the old dashboard
    python scripts/build_analyst_dashboard.py extract outputs/NSE_Analyst_Dashboard.html --as-of 2026-09-23

    # optional: raw Groww MCP responses -> normalized snapshot
    python scripts/build_analyst_dashboard.py ingest --ltp raw_ltp.json --fund raw_f1.json raw_f2.json \
        --out data/analyst_dashboard/groww_snapshot_2026-11-22.json

    # build (prints a KPI audit to the console)
    python scripts/build_analyst_dashboard.py build --snapshot data/analyst_dashboard/groww_snapshot_2026-11-22.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import date, datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "analyst_dashboard"
INPUTS_PATH = DATA_DIR / "analyst_inputs.json"
OUT_DIR = ROOT / "outputs"

STALE_TARGET_DAYS = 60      # broker targets older than this are flagged
OUTLIER_UPSIDE_PCT = 75     # |upside| above this is almost always a data error
MISMATCH_PP = 5             # old stored upside vs. its own target/price, pp

BULL = {"BUY", "OUTPERFORM", "OVERWEIGHT", "ADD", "ACCUMULATE"}
BEAR = {"SELL", "UNDERPERFORM", "UNDERWEIGHT", "REDUCE"}


# ─────────────────────────────────────────────────────────────────────────────
# extract: old dashboard HTML -> analyst_inputs.json
# ─────────────────────────────────────────────────────────────────────────────
def js_literal_to_json(src: str) -> str:
    """Quote bare object keys in a JS literal, leaving string contents untouched.

    A regex over the whole text is unsafe here because the thesis strings
    contain ``word:`` sequences; this walks the text and only rewrites keys that
    sit outside string literals.
    """
    out, i, n = [], 0, len(src)
    in_str: str | None = None
    while i < n:
        c = src[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(src[i + 1])
                i += 2
                continue
            if c == in_str:
                in_str = None
            i += 1
            continue
        if c in "\"'":
            in_str = c
            out.append('"' if c == "'" else c)
            i += 1
            continue
        m = re.match(r"([A-Za-z_$][\w$]*)\s*:", src[i:])
        if m and (not out or re.match(r"[\s{,]", out[-1] or " ")):
            out.append(f'"{m.group(1)}":')
            i += m.end()
            continue
        out.append(c)
        i += 1
    text = "".join(out)
    return re.sub(r",\s*([\]}])", r"\1", text)  # trailing commas


def extract_stocks_from_html(path: Path) -> list[dict]:
    html = path.read_text(encoding="utf-8")
    m = re.search(r"const\s+STOCKS\s*=\s*(\[.*?\n\]);", html, re.S)
    if not m:
        raise SystemExit(f"no `const STOCKS = [...]` block found in {path}")
    return json.loads(js_literal_to_json(m.group(1)))


def cmd_extract(args) -> None:
    stocks = extract_stocks_from_html(Path(args.html))
    rows = []
    for s in stocks:
        rows.append({
            "symbol": s["symbol"],
            "company": s.get("company"),
            "sector": s.get("sector"),
            "consensus_target": s.get("target"),
            "analyst_ratings": s.get("analystRatings") or [],
            "bulk_deals": s.get("bulkDeals") or [],
            "thesis": s.get("hypothesis"),
            "baseline": {
                "as_of": args.as_of,
                "price": s.get("price"),
                "day_change_pct": s.get("dayChange"),
                "pe": s.get("pe"),
                "roe": s.get("roe"),
                "market_cap_cr": s.get("marketCap"),
                "upside_pct": s.get("upside"),
            },
        })
    payload = {
        "source": str(Path(args.html).as_posix()),
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_as_of": args.as_of,
        "provenance_note": ("Ratings, targets, bulk deals and thesis text were carried over from "
                            "the previous dashboard (stated source: Trendlyne / broker reports / "
                            "NSE-BSE disclosures). They are NOT available from Groww MCP and have "
                            "not been re-verified."),
        "stocks": rows,
    }
    out = Path(args.out) if args.out else INPUTS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"extracted {len(rows)} stocks -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# ingest: raw Groww MCP tool responses -> normalized snapshot
# ─────────────────────────────────────────────────────────────────────────────
def gnum(v):
    """Groww display value -> float. '₹18,53,957Cr' -> 1853957, '8.69%' -> 8.69, '--'/'NA' -> None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).replace("₹", "").replace(",", "").replace("Cr", "").replace("%", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def _load_tool_json(path: Path) -> dict:
    """Accept either the bare tool JSON or Claude Code's persisted [{type:text,text:...}] wrapper."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(d, list) and d and isinstance(d[0], dict) and "text" in d[0]:
        d = json.loads(d[0]["text"])
    return d


def parse_groww_ltp(raw: dict) -> dict[str, dict]:
    """get_ltp result -> {NSE symbol: {...}} with the matching BSE code attached by company name."""
    rows = raw.get("result", raw).get("ltp_results", {})
    bse_by_name = {k[4:]: v.get("symbol") for k, v in rows.items() if k.startswith("BSE-")}
    out = {}
    for key, v in rows.items():
        if not key.startswith("NSE-") or v.get("ltp") is None:
            continue
        name = key[4:]
        out[v["symbol"]] = {
            "company": name,
            "bse_code": bse_by_name.get(name),
            "ltp": v["ltp"],
            "prev_close": v.get("prev_close"),
            "day_change_pct": v.get("day_change_perc"),
            "open": v.get("open"), "high": v.get("high"), "low": v.get("low"),
            "volume": v.get("volume"),
        }
    return out


def parse_groww_fundamentals(raw: dict) -> dict[str, dict]:
    """fetch_stocks_fundamental_data result -> {bse_code: stats} (joined to LTP via BSE code)."""
    content = raw.get("result", raw).get("content", {})
    out = {}
    for _slug, v in content.items():
        cid = (v.get("groww_contract_id") or "").replace("GSTK", "")
        st = v.get("fundamentals_stats") or {}
        if not cid or not st:
            continue
        out[cid] = {
            "market_cap_cr": gnum(st.get("marketCap")),
            "pe": gnum(st.get("peRatio")),
            "pb": gnum(st.get("pbRatio")),
            "roe": gnum(st.get("roe")),
            "eps_ttm": gnum(st.get("epsTtm")),
            "div_yield": gnum(st.get("dividendYieldInPercent")),
            "industry_pe": gnum(st.get("industryPe")),
            "book_value": gnum(st.get("bookValue")),
            "debt_to_equity": gnum(st.get("debtToEquity")),
        }
    return out


def cmd_ingest(args) -> None:
    ltp = parse_groww_ltp(_load_tool_json(Path(args.ltp)))
    fund: dict[str, dict] = {}
    for f in args.fund:
        fund.update(parse_groww_fundamentals(_load_tool_json(Path(f))))
    wanted = None
    if INPUTS_PATH.exists():
        wanted = {s["symbol"] for s in json.loads(INPUTS_PATH.read_text(encoding="utf-8"))["stocks"]}
    aliases = dict(a.split("=", 1) for a in (args.alias or []))  # dashboard_sym=groww_sym
    stocks = {}
    for sym in sorted(wanted or ltp):
        gsym = aliases.get(sym, sym)
        row = ltp.get(gsym)
        if not row:
            print(f"  ! {sym}: no NSE LTP row in the raw response")
            continue
        rec = {k: v for k, v in row.items() if k != "company"}
        if gsym != sym:
            rec["groww_symbol"] = gsym
        rec.update(fund.get(row.get("bse_code") or "", {}))
        stocks[sym] = rec
    snap = {
        "as_of": args.as_of or datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "Groww MCP: get_ltp + fetch_stocks_fundamental_data (ingested)",
        "stocks": stocks,
    }
    Path(args.out).write_text(json.dumps(snap, indent=1), encoding="utf-8")
    print(f"ingested {len(stocks)} stocks -> {args.out}")


# ─────────────────────────────────────────────────────────────────────────────
# build: merge + recompute KPIs + render
# ─────────────────────────────────────────────────────────────────────────────
def _rating_date(s: str | None) -> date | None:
    """'Sep 2026' -> 2026-09-15 (mid-month; broker dates are month-granular)."""
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%b %Y").date().replace(day=15)
    except ValueError:
        return None


def _pct(a, b):
    return None if a is None or b in (None, 0) else round((a / b - 1) * 100, 2)


def merge(snapshot: dict, inputs: dict) -> tuple[list[dict], list[str]]:
    snap_stocks = snapshot["stocks"]
    as_of = datetime.fromisoformat(snapshot["as_of"]).date()
    rows, missing = [], []
    for a in inputs["stocks"]:
        sym = a["symbol"]
        m = snap_stocks.get(sym)
        if not m:
            missing.append(sym)
            continue
        price = m["ltp"]
        base = a.get("baseline") or {}
        ratings = a.get("analyst_ratings") or []
        tgt = a.get("consensus_target")
        agency_tgts = [r["target"] for r in ratings if r.get("target")]
        agency_mean = round(statistics.mean(agency_tgts), 1) if agency_tgts else None
        dates = [d for d in (_rating_date(r.get("date")) for r in ratings) if d]
        latest = max(dates) if dates else None
        age = (as_of - latest).days if latest else None
        upside = _pct(tgt, price)

        flags = []
        if upside is not None and abs(upside) > OUTLIER_UPSIDE_PCT:
            flags.append(f"Upside {upside:+.0f}% is implausible - consensus target likely wrong")
        if tgt and agency_mean and abs(tgt / agency_mean - 1) > 0.25:
            flags.append(f"Consensus target {tgt:,.0f} disagrees with the agency mean {agency_mean:,.0f}")
        # target age is NOT a flag: when every row shares one baseline date it would
        # flag everything; it is surfaced as its own column + KPI tile instead.
        if m.get("note"):
            flags.append(m["note"])
        elif not m.get("volume"):
            flags.append("No trades in the last session (suspended / corporate action?)")
        if base.get("upside_pct") is not None and tgt and base.get("price"):
            implied = (tgt / base["price"] - 1) * 100
            if abs(implied - base["upside_pct"]) > MISMATCH_PP:
                flags.append(f"Old dashboard showed {base['upside_pct']:+.1f}% upside, but its own "
                             f"target/price implied {implied:+.1f}%")

        rows.append({
            "symbol": sym,
            "grow_symbol": m.get("groww_symbol", sym),
            "company": a.get("company") or sym,
            "sector": a.get("sector") or "Other",
            "price": price,
            "prevClose": m.get("prev_close"),
            "dayChange": m.get("day_change_pct"),
            "dayHigh": m.get("high"), "dayLow": m.get("low"),
            "volume": m.get("volume"),
            "pe": m.get("pe"), "industryPe": m.get("industry_pe"),
            "peVsInd": _pct(m.get("pe"), m.get("industry_pe")),
            "pb": m.get("pb"), "roe": m.get("roe"), "eps": m.get("eps_ttm"),
            "divYield": m.get("div_yield"), "de": m.get("debt_to_equity"),
            "bookValue": m.get("book_value"),
            "marketCap": m.get("market_cap_cr"),
            "target": tgt, "agencyMean": agency_mean,
            "upside": upside,
            "targetAgeDays": age,
            "basePrice": base.get("price"), "baseUpside": base.get("upside_pct"),
            "basePe": base.get("pe"), "baseMcap": base.get("market_cap_cr"),
            "sinceBase": _pct(price, base.get("price")),
            "analystRatings": ratings,
            "bulkDeals": a.get("bulk_deals") or [],
            "hypothesis": a.get("thesis"),
            "flags": flags,
        })
    return rows, missing


def kpis(rows: list[dict]) -> dict:
    ups = [r["upside"] for r in rows if r["upside"] is not None]
    clean = [r["upside"] for r in rows if r["upside"] is not None and abs(r["upside"]) <= OUTLIER_UPSIDE_PCT]
    since = [r["sinceBase"] for r in rows if r["sinceBase"] is not None]
    return {
        "stocks": len(rows),
        "avg_upside": round(statistics.mean(ups), 1) if ups else None,
        "avg_upside_ex_outliers": round(statistics.mean(clean), 1) if clean else None,
        "median_upside": round(statistics.median(ups), 1) if ups else None,
        "above_target": sum(1 for u in ups if u < 0),
        "rated": sum(1 for r in rows if r["analystRatings"]),
        "with_deals": sum(1 for r in rows if r["bulkDeals"]),
        "flagged": sum(1 for r in rows if r["flags"]),
        "avg_since_base": round(statistics.mean(since), 1) if since else None,
    }


def audit(rows: list[dict], inputs: dict) -> None:
    old = [s["baseline"] for s in inputs["stocks"] if s.get("baseline")]
    old_ups = [b["upside_pct"] for b in old if b.get("upside_pct") is not None]
    k = kpis(rows)
    print("\n== KPI audit ==")
    if old_ups:
        print(f"  old dashboard avg upside (stored values) : {statistics.mean(old_ups):.1f}%")
    print(f"  new avg upside (live price, all)         : {k['avg_upside']}%")
    print(f"  new avg upside (excl |upside|>{OUTLIER_UPSIDE_PCT}%)     : {k['avg_upside_ex_outliers']}%")
    print(f"  new median upside                        : {k['median_upside']}%")
    print(f"  stocks now trading above target          : {k['above_target']}")
    print(f"  avg price move since baseline            : {k['avg_since_base']}%")
    print(f"  rows with data flags                     : {k['flagged']} / {k['stocks']}")
    for r in rows:
        for f in r["flags"]:
            print(f"    {r['symbol']:<11} {f}")


def build_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("/*DATA*/null", data)


def cmd_build(args) -> None:
    snap_path = Path(args.snapshot) if args.snapshot else max(DATA_DIR.glob("groww_snapshot_*.json"))
    snapshot = json.loads(snap_path.read_text(encoding="utf-8"))
    inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    rows, missing = merge(snapshot, inputs)
    if missing:
        print(f"  ! no Groww data for: {', '.join(missing)} (omitted)")
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "snapshot": {k: snapshot.get(k) for k in ("as_of", "price_session", "source", "market_status")},
        "baseline_as_of": inputs.get("baseline_as_of"),
        "provenance_note": inputs.get("provenance_note"),
        "thresholds": {"stale_days": STALE_TARGET_DAYS, "outlier_pct": OUTLIER_UPSIDE_PCT},
        "kpis": kpis(rows),
        "stocks": rows,
    }
    stamp = datetime.fromisoformat(snapshot["as_of"]).date().isoformat()
    out = Path(args.out) if args.out else OUT_DIR / f"NSE_Analyst_Dashboard_{stamp}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(payload), encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"built {len(rows)} stocks -> {out}")
    audit(rows, inputs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="old dashboard HTML -> analyst_inputs.json")
    e.add_argument("html")
    e.add_argument("--as-of", required=True, help="date the old dashboard's prices were taken (YYYY-MM-DD)")
    e.add_argument("--out")
    e.set_defaults(fn=cmd_extract)

    i = sub.add_parser("ingest", help="raw Groww MCP responses -> snapshot JSON")
    i.add_argument("--ltp", required=True)
    i.add_argument("--fund", nargs="+", required=True)
    i.add_argument("--alias", nargs="*", help="DASHBOARD_SYM=GROWW_SYM, e.g. PIRAMALENT=PEL")
    i.add_argument("--as-of")
    i.add_argument("--out", required=True)
    i.set_defaults(fn=cmd_ingest)

    b = sub.add_parser("build", help="snapshot + analyst inputs -> HTML dashboard")
    b.add_argument("--snapshot", help="default: newest data/analyst_dashboard/groww_snapshot_*.json")
    b.add_argument("--inputs", default=str(INPUTS_PATH))
    b.add_argument("--out")
    b.set_defaults(fn=cmd_build)

    args = ap.parse_args()
    args.fn(args)


# ─────────────────────────────────────────────────────────────────────────────
# HTML template (single self-contained file; `/*DATA*/null` is replaced)
# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>NSE Analyst Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --bg:#f0f4f9;--sf:#ffffff;--sf2:#f1f5fb;--sf3:#e4ecf7;
  --bd:#d1daea;--bd2:#b0bfcf;--tx:#0f172a;--mu:#52657f;--mu2:#8fa3b8;
  --ac:#4f46e5;--ac2:#4338ca;
  --green:#15803d;--red:#dc2626;--amber:#b45309;--violet:#7c3aed;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--tx);font-family:system-ui,-apple-system,sans-serif;font-size:13px}
.header{background:linear-gradient(135deg,#fff 0%,#eef2ff 100%);border-bottom:1px solid var(--bd);box-shadow:0 2px 8px #0001;padding:12px 20px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;position:sticky;top:0;z-index:50}
.header h1{font-size:17px;font-weight:700;color:#1e293b;margin-right:auto}
.header h1 span{color:var(--ac)}
.hstat{background:#fff;border:1px solid var(--bd);border-radius:8px;padding:5px 12px;text-align:center;box-shadow:0 1px 3px #0002}
.hstat-val{font-size:15px;font-weight:700}
.hstat-lbl{font-size:10px;color:var(--mu);margin-top:1px}
.updated{font-size:10px;color:var(--mu);width:100%;text-align:right;line-height:1.5}
.layout{display:flex;height:calc(100vh - 78px);overflow:hidden}
.sidebar{width:240px;flex-shrink:0;background:var(--sf);border-right:1px solid var(--bd);overflow-y:auto;padding:12px}
.main{flex:1;overflow:hidden;display:flex;flex-direction:column;min-width:0}
.main-content{flex:1;overflow-y:auto;padding:14px}
.sl{font-size:10px;font-weight:700;color:var(--mu);text-transform:uppercase;letter-spacing:.06em;margin:14px 0 6px;padding-bottom:4px;border-bottom:1px solid var(--bd)}
.filter-group{display:flex;flex-direction:column;gap:3px}
.fb{display:flex;align-items:center;gap:7px;padding:5px 8px;border-radius:6px;cursor:pointer;color:var(--mu);font-size:12px;border:1px solid transparent}
.fb:hover{background:var(--sf2);color:var(--tx)}
.fb.active{background:var(--sf3);color:var(--ac2);border-color:var(--bd2)}
label.sl2{font-size:11px;color:var(--mu);display:block;margin-bottom:4px}
input[type=range]{width:100%;accent-color:var(--ac)}
.range-val{font-size:11px;color:var(--ac2);margin-top:2px;text-align:right}
.search-box{width:100%;background:#fff;border:1px solid var(--bd);border-radius:6px;padding:6px 10px;color:var(--tx);font-size:12px;outline:none;font-family:inherit}
.search-box:focus{border-color:var(--ac)}
.clear-btn{width:100%;margin-top:10px;padding:6px;background:none;border:1px solid var(--bd);border-radius:6px;color:var(--mu);cursor:pointer;font-size:11px}
.clear-btn:hover{background:var(--sf3);color:var(--tx)}
.banner{background:#fffbeb;border:1px solid #fcd34d;color:#78350f;border-radius:8px;padding:8px 12px;font-size:11.5px;line-height:1.5;margin-bottom:12px}
.banner b{color:#92400e}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:14px}
.kpi{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:10px 13px;position:relative;overflow:hidden;box-shadow:0 1px 4px #0001}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--c,var(--ac))}
.kpi-val{font-size:21px;font-weight:700;margin-bottom:2px;color:var(--c,var(--ac2));font-variant-numeric:tabular-nums}
.kpi-lbl{font-size:11px;color:var(--mu)}
.kpi-sub{font-size:10px;color:var(--mu2);margin-top:2px}
.charts-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}
.chart-card{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:12px;box-shadow:0 1px 4px #0001;min-width:0}
.chart-title{font-size:11px;font-weight:600;color:var(--mu);text-transform:uppercase;letter-spacing:.04em;margin-bottom:10px}
.chart-canvas-wrap{height:140px;position:relative}
.table-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:8px}
.table-title{font-size:12px;font-weight:600;color:var(--mu);text-transform:uppercase}
.result-count{font-size:11px;color:var(--mu)}
.tbl-wrap{background:var(--sf);border:1px solid var(--bd);border-radius:10px;overflow:auto;box-shadow:0 2px 8px #0001}
table{width:100%;border-collapse:collapse;font-size:12px}
thead{position:sticky;top:0;z-index:2}
th{background:#e8eef8;color:var(--mu);font-weight:600;font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;padding:9px 9px;text-align:left;border-bottom:1px solid var(--bd);white-space:nowrap;cursor:pointer;user-select:none}
th.num,td.num{text-align:right}
th:hover{color:var(--tx)}
th .sort-icon{margin-left:3px;opacity:.4;font-size:9px}
th.sorted .sort-icon{opacity:1;color:var(--ac2)}
td{padding:7px 9px;border-bottom:1px solid #d1daea55;vertical-align:middle;font-variant-numeric:tabular-nums}
tr:hover td{background:#f0f6ff;cursor:pointer}
.ticker{font-weight:700;color:#1e293b}
.company-name{font-size:10.5px;color:var(--mu);margin-top:1px}
.sector-tag{display:inline-block;font-size:9.5px;padding:1px 6px;border-radius:4px;background:var(--sf3);color:#3b5068;border:1px solid #c8d5e8;white-space:nowrap}
.pos{color:var(--green)} .neg{color:var(--red)} .neu{color:var(--mu)}
.b{font-weight:600}
.sub{font-size:10px;color:var(--mu2)}
.rpill{font-size:9.5px;padding:2px 6px;border-radius:4px;font-weight:600;border:1px solid;white-space:nowrap}
.rpill.buy{background:#dcfce7;color:#166534;border-color:#86efac}
.rpill.hold{background:#fef3c7;color:#92400e;border-color:#fde68a}
.rpill.sell{background:#fee2e2;color:#991b1b;border-color:#fca5a5}
.bulk-tag{font-size:9.5px;padding:2px 6px;border-radius:4px;border:1px solid;white-space:nowrap}
.bulk-buy{background:#dcfce7;color:#166534;border-color:#86efac}
.bulk-sell{background:#fee2e2;color:#991b1b;border-color:#fca5a5}
.na{color:#94a3b8;font-size:11px}
.flag{display:inline-block;margin-left:4px;color:#b45309;font-size:12px;cursor:help}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(15,23,42,.65);z-index:100;justify-content:center;align-items:center;padding:20px}
.modal-overlay.open{display:flex}
.modal{background:#fff;border:1px solid var(--bd);border-radius:14px;max-width:760px;width:100%;max-height:88vh;overflow-y:auto;padding:20px;box-shadow:0 20px 60px #0003}
.modal-hd{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:14px}
.modal-ticker{font-size:20px;font-weight:800}
.modal-company{font-size:12px;color:var(--mu);margin-top:2px}
.close-btn{background:none;border:none;color:var(--mu);cursor:pointer;font-size:20px;line-height:1;padding:2px}
.modal-badges{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.badge{font-size:10px;padding:3px 8px;border-radius:6px;font-weight:600;border:1px solid;background:#ede9fe;color:#4338ca;border-color:#c4b5fd}
.flag-list{background:#fffbeb;border:1px solid #fcd34d;border-radius:8px;padding:8px 12px;margin-bottom:12px;font-size:11.5px;color:#78350f;line-height:1.6}
.modal-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px}
.mstat{background:var(--sf2);border:1px solid var(--bd);border-radius:8px;padding:7px 10px}
.mstat-val{font-size:14px;font-weight:700;font-variant-numeric:tabular-nums}
.mstat-lbl{font-size:10px;color:var(--mu);margin-top:2px}
.section-hd{font-size:10px;font-weight:700;color:var(--mu);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid var(--bd)}
.mtable{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:16px}
.mtable th{font-size:10px;color:var(--mu);text-align:left;padding:5px 8px;border-bottom:1px solid var(--bd);font-weight:600;background:none;cursor:default;text-transform:none}
.mtable td{padding:6px 8px;border-bottom:1px solid #d1daea55}
.mtable tr:hover td{background:none;cursor:default}
.hypothesis{background:#eef2ff;border-radius:8px;padding:12px;font-size:12px;line-height:1.6;border-left:3px solid var(--ac)}
.modal-disclaimer{font-size:10px;color:#94a3b8;margin-top:12px;padding-top:10px;border-top:1px solid #e2e8f0;line-height:1.5}
@media(max-width:1100px){.charts-row{grid-template-columns:1fr 1fr}}
@media(max-width:768px){
  .sidebar{display:none}
  .layout{height:auto}
  .charts-row{grid-template-columns:1fr}
  .modal-grid{grid-template-columns:1fr 1fr}
  .header{padding:10px 16px}
}
</style>
</head>
<body>
<div class="header">
  <h1>NSE Stock <span>Analyst Intelligence</span> Dashboard</h1>
  <div class="hstat"><div class="hstat-val" id="h-total" style="color:#4338ca">--</div><div class="hstat-lbl">Stocks</div></div>
  <div class="hstat"><div class="hstat-val" id="h-med" style="color:#15803d">--</div><div class="hstat-lbl">Median upside</div></div>
  <div class="hstat"><div class="hstat-val" id="h-since" style="color:#4338ca">--</div><div class="hstat-lbl" id="h-since-lbl">Avg move</div></div>
  <div class="hstat"><div class="hstat-val" id="h-flag" style="color:#b45309">--</div><div class="hstat-lbl">Data flags</div></div>
  <div class="updated" id="updated"></div>
</div>

<div class="layout">
  <div class="sidebar">
    <input class="search-box" type="text" placeholder="Search symbol or company..." id="search" oninput="applyFilters()">
    <div class="sl">Sector</div>
    <div class="filter-group" id="sector-filters"></div>
    <div class="sl">Price range (&#8377;)</div>
    <label class="sl2">Min: <span id="min-price-lbl">0</span></label>
    <input type="range" id="min-price" min="0" step="50" value="0" oninput="applyFilters()">
    <label class="sl2" style="margin-top:6px">Max: <span id="max-price-lbl"></span></label>
    <input type="range" id="max-price" min="0" step="50" oninput="applyFilters()">
    <div class="sl">Minimum upside (live)</div>
    <input type="range" id="min-upside" min="-30" max="60" step="5" value="-30" oninput="applyFilters()">
    <div class="range-val" id="upside-lbl">Any</div>
    <div class="sl">Min bullish ratings</div>
    <div class="filter-group">
      <div class="fb active" data-bull="0" onclick="setF('bull',0,this)">Any</div>
      <div class="fb" data-bull="2" onclick="setF('bull',2,this)">&ge; 2 agencies</div>
      <div class="fb" data-bull="3" onclick="setF('bull',3,this)">&ge; 3 agencies</div>
    </div>
    <div class="sl">Bulk deals (as of baseline)</div>
    <div class="filter-group">
      <div class="fb active" data-deal="all" onclick="setF('deal','all',this)">All stocks</div>
      <div class="fb" data-deal="buy" onclick="setF('deal','buy',this)">Institutional buy</div>
    </div>
    <div class="sl">Data quality</div>
    <div class="filter-group">
      <div class="fb active" data-qual="all" onclick="setF('qual','all',this)">All rows</div>
      <div class="fb" data-qual="clean" onclick="setF('qual','clean',this)">Hide implausible targets</div>
      <div class="fb" data-qual="flagged" onclick="setF('qual','flagged',this)">Flagged only</div>
    </div>
    <button class="clear-btn" onclick="clearFilters()">Clear all filters</button>
  </div>

  <div class="main"><div class="main-content">
    <div class="banner" id="banner"></div>
    <div class="kpi-row" id="kpi-row"></div>
    <div class="charts-row">
      <div class="chart-card"><div class="chart-title">Sector distribution</div><div class="chart-canvas-wrap"><canvas id="c-sector"></canvas></div></div>
      <div class="chart-card"><div class="chart-title">Upside to target (live)</div><div class="chart-canvas-wrap"><canvas id="c-upside"></canvas></div></div>
      <div class="chart-card"><div class="chart-title">Rating consensus</div><div class="chart-canvas-wrap"><canvas id="c-rating"></canvas></div></div>
      <div class="chart-card"><div class="chart-title" id="c-since-title">Price move since baseline</div><div class="chart-canvas-wrap"><canvas id="c-since"></canvas></div></div>
    </div>
    <div class="table-header">
      <span class="table-title">Analyst coverage &mdash; NSE stocks</span>
      <span class="result-count" id="result-count"></span>
    </div>
    <div class="tbl-wrap"><table>
      <thead><tr>
        <th data-k="symbol">Symbol <span class="sort-icon">&#8597;</span></th>
        <th data-k="sector">Sector <span class="sort-icon">&#8597;</span></th>
        <th data-k="price" class="num">Price &#8377; <span class="sort-icon">&#8597;</span></th>
        <th data-k="dayChange" class="num">Day <span class="sort-icon">&#8597;</span></th>
        <th data-k="sinceBase" class="num" id="th-since">Since base <span class="sort-icon">&#8597;</span></th>
        <th data-k="target" class="num">Cons. target <span class="sort-icon">&#8597;</span></th>
        <th data-k="upside" class="num">Upside <span class="sort-icon">&#8597;</span></th>
        <th data-k="pe" class="num">P/E <span class="sort-icon">&#8597;</span></th>
        <th data-k="pb" class="num">P/B <span class="sort-icon">&#8597;</span></th>
        <th data-k="roe" class="num">ROE <span class="sort-icon">&#8597;</span></th>
        <th data-k="divYield" class="num">Div yld <span class="sort-icon">&#8597;</span></th>
        <th>Ratings</th>
        <th>Bulk deals</th>
        <th data-k="marketCap" class="num">Mkt cap <span class="sort-icon">&#8597;</span></th>
      </tr></thead>
      <tbody id="tbody"></tbody>
    </table></div>
  </div></div>
</div>

<div class="modal-overlay" id="modal" onclick="if(event.target.id==='modal')closeModal()">
  <div class="modal">
    <div class="modal-hd">
      <div><div class="modal-ticker" id="m-ticker"></div><div class="modal-company" id="m-company"></div></div>
      <button class="close-btn" onclick="closeModal()" aria-label="Close">&times;</button>
    </div>
    <div class="modal-badges" id="m-badges"></div>
    <div id="m-flags"></div>
    <div class="modal-grid" id="m-grid"></div>
    <div class="section-hd">Analyst ratings &amp; targets <span style="text-transform:none;font-weight:400">(upside vs live price)</span></div>
    <table class="mtable" id="m-ratings"><thead><tr><th>Agency</th><th>Rating</th><th>Target &#8377;</th><th>Upside</th><th>Date</th></tr></thead><tbody></tbody></table>
    <div class="section-hd">Bulk / block deals (carried from baseline)</div>
    <table class="mtable" id="m-bulk"><thead><tr><th>Client</th><th>Action</th><th>Qty</th><th>Price &#8377;</th><th>Value &#8377;Cr</th><th>Date</th></tr></thead><tbody></tbody></table>
    <div class="section-hd">Bullish thesis (carried from baseline)</div>
    <div class="hypothesis" id="m-thesis"></div>
    <div class="modal-disclaimer" id="m-disc"></div>
  </div>
</div>

<script>
const P = /*DATA*/null;
const STOCKS = P.stocks;
const BULL = ['BUY','OUTPERFORM','OVERWEIGHT','ADD','ACCUMULATE'];
const BEAR = ['SELL','UNDERPERFORM','UNDERWEIGHT','REDUCE'];
const OUT = P.thresholds.outlier_pct;
const st = {sector:'ALL', bull:0, deal:'all', qual:'all', sortKey:'upside', sortDir:-1};
let filtered = [];

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const inr = v => v==null ? '—' : '₹' + Number(v).toLocaleString('en-IN', {maximumFractionDigits:2});
const pct = (v,d=1) => v==null ? '—' : (v>0?'+':'') + v.toFixed(d) + '%';
const cls = v => v==null ? 'neu' : v>0 ? 'pos' : v<0 ? 'neg' : 'neu';
const num = (v,d=1) => v==null ? '<span class="na">—</span>' : v.toFixed(d);
const mcap = v => v==null ? '—' : v>=100000 ? `₹${(v/100000).toFixed(2)}L Cr` : `₹${Math.round(v).toLocaleString('en-IN')} Cr`;
const ratingCls = r => BULL.includes(r) ? 'buy' : BEAR.includes(r) ? 'sell' : 'hold';
const bullCount = s => s.analystRatings.filter(r => BULL.includes(r.rating)).length;
const implausible = s => s.upside!=null && Math.abs(s.upside) > OUT;
const fmtDate = s => s ? new Date(s).toLocaleDateString('en-IN',{day:'numeric',month:'short',year:'numeric'}) : '';
const baseLbl = P.baseline_as_of ? fmtDate(P.baseline_as_of) : 'baseline';

function init(){
  const snap = P.snapshot;
  $('updated').textContent = `Prices & fundamentals: Groww MCP, session ${fmtDate(snap.price_session || snap.as_of)} (fetched ${new Date(snap.as_of).toLocaleString('en-IN')}) · Analyst targets / bulk deals: carried from the ${baseLbl} build · Built ${new Date(P.generated).toLocaleString('en-IN')}`;
  $('th-since').firstChild.textContent = `Since ${baseLbl.replace(/ \d{4}$/,'')} `;
  $('c-since-title').textContent = `Top / bottom movers since ${baseLbl.replace(/ \d{4}$/,'')}`;
  $('h-since-lbl').textContent = `Avg move since ${baseLbl.replace(/ \d{4}$/,'')}`;
  const k = P.kpis;
  $('h-total').textContent = k.stocks;
  $('h-med').textContent = pct(k.median_upside);
  $('h-since').textContent = pct(k.avg_since_base);
  $('h-flag').textContent = k.flagged;
  $('banner').innerHTML = `<b>Read before using the targets.</b> Live prices and fundamentals were refreshed from Groww MCP. Groww has no analyst-target or bulk-deal data, so ratings, targets, deals and thesis text are carried over from the ${esc(baseLbl)} dashboard and have <b>not</b> been re-verified. Upside is recomputed against today's price. ${k.flagged} of ${k.stocks} rows carry a data flag (&#9888;) — e.g. targets that imply more than ±${OUT}% upside, or broker notes older than ${P.thresholds.stale_days} days.`;

  const maxP = Math.ceil(Math.max(...STOCKS.map(s=>s.price))/1000)*1000;
  ['min-price','max-price'].forEach(id => $(id).max = maxP);
  $('max-price').value = maxP;

  const sectors = [...new Set(STOCKS.map(s=>s.sector))].sort();
  $('sector-filters').innerHTML = `<div class="fb active" data-sector="ALL">All sectors</div>` +
    sectors.map(s=>`<div class="fb" data-sector="${esc(s)}">${esc(s)}</div>`).join('');
  $('sector-filters').querySelectorAll('.fb').forEach(b => b.onclick = () => setF('sector', b.dataset.sector, b));

  document.querySelectorAll('th[data-k]').forEach(th => th.onclick = () => sortBy(th.dataset.k));
  buildCharts();
  applyFilters();
}

function setF(key, val, btn){
  st[key] = val;
  document.querySelectorAll(`[data-${key}]`).forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}

function applyFilters(){
  const q = $('search').value.toLowerCase();
  const minP = +$('min-price').value, maxP = +$('max-price').value, minUp = +$('min-upside').value;
  $('min-price-lbl').textContent = minP.toLocaleString('en-IN');
  $('max-price-lbl').textContent = maxP.toLocaleString('en-IN');
  $('upside-lbl').textContent = minUp <= -30 ? 'Any' : `≥ ${minUp}%`;
  filtered = STOCKS.filter(s => {
    if (q && !s.symbol.toLowerCase().includes(q) && !s.company.toLowerCase().includes(q)) return false;
    if (st.sector !== 'ALL' && s.sector !== st.sector) return false;
    if (s.price < minP || s.price > maxP) return false;
    if (minUp > -30 && (s.upside == null || s.upside < minUp)) return false;
    if (bullCount(s) < st.bull) return false;
    if (st.deal === 'buy' && !s.bulkDeals.some(d => d.action === 'BUY')) return false;
    if (st.qual === 'clean' && implausible(s)) return false;
    if (st.qual === 'flagged' && !s.flags.length) return false;
    return true;
  });
  sortData(); renderTable(); renderKPIs(); updateCharts();
}

function clearFilters(){
  $('search').value = '';
  $('min-price').value = 0; $('max-price').value = $('max-price').max; $('min-upside').value = -30;
  setF('sector','ALL',document.querySelector('[data-sector="ALL"]'));
  setF('bull',0,document.querySelector('[data-bull="0"]'));
  setF('deal','all',document.querySelector('[data-deal="all"]'));
  setF('qual','all',document.querySelector('[data-qual="all"]'));
}

function sortBy(k){
  if (st.sortKey === k) st.sortDir *= -1; else { st.sortKey = k; st.sortDir = -1; }
  sortData(); renderTable();
}
function sortData(){
  const k = st.sortKey;
  document.querySelectorAll('th[data-k]').forEach(th => th.classList.toggle('sorted', th.dataset.k === k));
  filtered.sort((a,b) => {
    let av = a[k], bv = b[k];
    if (av == null) return 1; if (bv == null) return -1;   // nulls always last
    return typeof av === 'string' ? st.sortDir * av.localeCompare(bv) : st.sortDir * (av - bv);
  });
}

function renderTable(){
  $('result-count').textContent = `${filtered.length} of ${STOCKS.length} stocks`;
  $('tbody').innerHTML = filtered.map((s,i) => {
    const flag = s.flags.length ? `<span class="flag" title="${esc(s.flags.join('\n'))}">&#9888;</span>` : '';
    const ratings = s.analystRatings.slice(0,3).map(r => `<span class="rpill ${ratingCls(r.rating)}">${esc(r.agency.split(' ')[0])}: ${esc(r.rating)}</span>`).join(' ') || '<span class="na">—</span>';
    const deals = s.bulkDeals.slice(0,2).map(d => `<span class="bulk-tag ${d.action==='BUY'?'bulk-buy':'bulk-sell'}">${esc(d.action)}: ${esc(d.client.split(' ')[0])}</span>`).join(' ') || '<span class="na">—</span>';
    const peSub = s.industryPe ? `<div class="sub">ind ${s.industryPe.toFixed(1)}</div>` : '';
    const age = s.targetAgeDays != null ? `<div class="sub">${s.targetAgeDays}d old</div>` : '';
    return `<tr onclick="openModal(${i})">
      <td><div class="ticker">${esc(s.symbol)}${flag}</div><div class="company-name">${esc(s.company.length>28 ? s.company.slice(0,28)+'…' : s.company)}</div></td>
      <td><span class="sector-tag">${esc(s.sector)}</span></td>
      <td class="num b">${inr(s.price)}</td>
      <td class="num ${cls(s.dayChange)}">${pct(s.dayChange,2)}</td>
      <td class="num ${cls(s.sinceBase)}">${pct(s.sinceBase)}<div class="sub">was ${inr(s.basePrice)}</div></td>
      <td class="num" style="color:var(--ac2)">${s.target ? inr(s.target) : '<span class="na">—</span>'}${age}</td>
      <td class="num b ${implausible(s) ? '' : cls(s.upside)}" ${implausible(s) ? 'style="color:#b45309;text-decoration:line-through"' : ''}>${pct(s.upside)}</td>
      <td class="num">${num(s.pe)}${peSub}</td>
      <td class="num">${num(s.pb,2)}</td>
      <td class="num">${s.roe==null ? '<span class="na">—</span>' : s.roe.toFixed(1)+'%'}</td>
      <td class="num">${s.divYield==null ? '<span class="na">—</span>' : s.divYield.toFixed(2)+'%'}</td>
      <td>${ratings}</td>
      <td>${deals}</td>
      <td class="num">${mcap(s.marketCap)}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="14" style="text-align:center;padding:20px;color:var(--mu)">No stocks match these filters</td></tr>';
}

function median(a){ if(!a.length) return null; const s=[...a].sort((x,y)=>x-y), m=s.length>>1; return s.length%2 ? s[m] : (s[m-1]+s[m])/2; }
function mean(a){ return a.length ? a.reduce((x,y)=>x+y,0)/a.length : null; }

function renderKPIs(){
  const ups = filtered.filter(s=>s.upside!=null).map(s=>s.upside);
  const clean = filtered.filter(s=>s.upside!=null && !implausible(s)).map(s=>s.upside);
  const since = filtered.filter(s=>s.sinceBase!=null).map(s=>s.sinceBase);
  const above = ups.filter(u=>u<0).length;
  const stale = filtered.filter(s=>s.targetAgeDays!=null && s.targetAgeDays>P.thresholds.stale_days).length;
  const tiles = [
    {v: filtered.length, l:'Stocks shown', c:'#4338ca'},
    {v: pct(median(ups)), l:'Median upside (live)', sub:`mean ${pct(mean(ups))}`, c:'#15803d'},
    {v: pct(mean(clean)), l:`Avg upside excl. >±${OUT}%`, sub:`${ups.length-clean.length} implausible target(s) excluded`, c:'#15803d'},
    {v: above, l:'Trading above target', sub:'negative upside at live price', c: above ? '#dc2626' : '#52657f'},
    {v: pct(mean(since)), l:`Avg move since ${baseLbl}`, sub:`${since.filter(x=>x>0).length} up / ${since.filter(x=>x<0).length} down`, c: (mean(since)||0)>=0 ? '#15803d' : '#dc2626'},
    {v: stale, l:`Targets > ${P.thresholds.stale_days} days old`, sub:'refresh broker data', c:'#b45309'},
    {v: filtered.filter(s=>s.bulkDeals.length).length, l:'With bulk deals', sub:`as of ${baseLbl}`, c:'#7c3aed'},
  ];
  $('kpi-row').innerHTML = tiles.map(t => `<div class="kpi" style="--c:${t.c}"><div class="kpi-val">${t.v}</div><div class="kpi-lbl">${t.l}</div>${t.sub?`<div class="kpi-sub">${t.sub}</div>`:''}</div>`).join('');
}

let charts = {};
const CO = {responsive:true, maintainAspectRatio:false, animation:false,
  plugins:{legend:{display:false}, tooltip:{backgroundColor:'#1e293b', titleColor:'#f8fafc', bodyColor:'#cbd5e1'}}};
const LEG = {display:true, position:'right', labels:{color:'#52657f', font:{size:9}, boxWidth:10, padding:4}};
const AX = {ticks:{color:'#64748b', font:{size:9}}, grid:{color:'#e2e8f0'}};
const SECTORS = [...new Set(STOCKS.map(s=>s.sector))];
const BUCKETS = [[-Infinity,0,'<0%'],[0,10,'0-10'],[10,20,'10-20'],[20,30,'20-30'],[30,40,'30-40'],[40,OUT,`40-${OUT}`],[OUT,Infinity,`>${OUT}`]];
const PALETTE = ['#6366f1','#22c55e','#f59e0b','#ef4444','#e879f9','#3b82f6','#14b8a6','#f97316','#a855f7','#06b6d4','#84cc16','#ec4899','#64748b','#0ea5e9','#d946ef','#facc15'];

function ratingCounts(rows){
  const c = [0,0,0];
  rows.forEach(s => s.analystRatings.forEach(r => c[BULL.includes(r.rating)?0:BEAR.includes(r.rating)?2:1]++));
  return c;
}
function buildCharts(){
  if (typeof Chart === 'undefined') return;   // offline: table still works
  charts.sector = new Chart($('c-sector'), {type:'doughnut', data:{labels:SECTORS, datasets:[{data:[], backgroundColor:PALETTE, borderWidth:0}]}, options:{...CO, plugins:{...CO.plugins, legend:LEG}}});
  charts.upside = new Chart($('c-upside'), {type:'bar', data:{labels:BUCKETS.map(b=>b[2]), datasets:[{data:[], backgroundColor:BUCKETS.map(b=>b[1]<=0?'#fca5a5':b[0]>=OUT?'#fcd34d':'#a5b4fc'), borderWidth:0}]}, options:{...CO, scales:{x:AX, y:{...AX, ticks:{...AX.ticks, precision:0}}}}});
  charts.rating = new Chart($('c-rating'), {type:'doughnut', data:{labels:['Buy-side','Hold / neutral','Sell-side'], datasets:[{data:[], backgroundColor:['#22c55e','#f59e0b','#ef4444'], borderWidth:0}]}, options:{...CO, plugins:{...CO.plugins, legend:LEG}}});
  charts.since = new Chart($('c-since'), {type:'bar', data:{labels:[], datasets:[{data:[], backgroundColor:[], borderWidth:0}]}, options:{...CO, indexAxis:'y', scales:{x:{...AX, ticks:{...AX.ticks, callback:v=>v+'%'}}, y:{...AX, ticks:{...AX.ticks, autoSkip:false, font:{size:8}}}}}});
}
function updateCharts(){
  if (!charts.sector) return;
  charts.sector.data.datasets[0].data = SECTORS.map(sec => filtered.filter(s=>s.sector===sec).length);
  charts.upside.data.datasets[0].data = BUCKETS.map(([lo,hi]) => filtered.filter(s=>s.upside!=null && s.upside>=lo && s.upside<hi).length);
  charts.rating.data.datasets[0].data = ratingCounts(filtered);
  let sv = filtered.filter(s=>s.sinceBase!=null).sort((a,b)=>b.sinceBase-a.sinceBase);
  if (sv.length > 12) sv = [...sv.slice(0,6), ...sv.slice(-6)];   // top & bottom movers stay legible
  charts.since.data.labels = sv.map(s=>s.symbol);
  charts.since.data.datasets[0].data = sv.map(s=>s.sinceBase);
  charts.since.data.datasets[0].backgroundColor = sv.map(s=>s.sinceBase>=0?'#22c55e':'#ef4444');
  Object.values(charts).forEach(c => c.update());
}

function openModal(i){
  const s = filtered[i]; if (!s) return;
  $('m-ticker').textContent = s.symbol + (s.grow_symbol !== s.symbol ? `  (Groww/NSE: ${s.grow_symbol})` : '');
  $('m-company').textContent = s.company;
  const f1 = v => v==null ? '—' : v.toFixed(1);
  $('m-badges').innerHTML = [s.sector, `Upside ${pct(s.upside)}`, `P/E ${f1(s.pe)} vs ind ${f1(s.industryPe)}`, `ROE ${f1(s.roe)}%`].map(b=>`<span class="badge">${esc(b)}</span>`).join('');
  $('m-flags').innerHTML = s.flags.length ? `<div class="flag-list">${s.flags.map(f=>'&#9888; '+esc(f)).join('<br>')}</div>` : '';
  const g = [
    ['Price', inr(s.price), s.dayChange>=0?'#15803d':'#dc2626'],
    ['Day change', pct(s.dayChange,2), s.dayChange>=0?'#15803d':'#dc2626'],
    [`Since ${baseLbl}`, pct(s.sinceBase), (s.sinceBase||0)>=0?'#15803d':'#dc2626'],
    ['Consensus target', s.target?inr(s.target):'—', '#4338ca'],
    ['Agency mean target', s.agencyMean?inr(s.agencyMean):'—', '#4338ca'],
    ['Market cap', mcap(s.marketCap), '#0f172a'],
    ['P/E (industry)', `${s.pe??'—'} (${s.industryPe??'—'})`, '#b45309'],
    ['P/B', s.pb??'—', '#0f172a'],
    ['EPS TTM', s.eps!=null?'₹'+s.eps:'—', '#0f172a'],
    ['Book value', inr(s.bookValue), '#0f172a'],
    ['Div yield', s.divYield!=null?s.divYield+'%':'—', '#0f172a'],
    ['Debt / equity', s.de??'n/a', '#0f172a'],
  ];
  $('m-grid').innerHTML = g.map(([l,v,c]) => `<div class="mstat"><div class="mstat-val" style="color:${c}">${v}</div><div class="mstat-lbl">${l}</div></div>`).join('');
  document.querySelector('#m-ratings tbody').innerHTML = s.analystRatings.map(r => {
    const up = r.target ? (r.target/s.price-1)*100 : null;
    return `<tr><td>${esc(r.agency)}</td><td><span class="rpill ${ratingCls(r.rating)}">${esc(r.rating)}</span></td><td>${r.target?inr(r.target):'—'}</td><td class="${cls(up)}">${pct(up)}</td><td>${esc(r.date||'—')}</td></tr>`;
  }).join('') || '<tr><td colspan="5" class="na" style="text-align:center">No analyst ratings</td></tr>';
  document.querySelector('#m-bulk tbody').innerHTML = s.bulkDeals.map(d =>
    `<tr><td>${esc(d.client)}</td><td><span class="rpill ${d.action==='BUY'?'buy':'sell'}">${esc(d.action)}</span></td><td>${esc(d.qty)}</td><td>${inr(d.price)}</td><td>₹${esc(d.value)} Cr</td><td>${esc(d.date)}</td></tr>`
  ).join('') || '<tr><td colspan="6" class="na" style="text-align:center">No bulk deals recorded</td></tr>';
  $('m-thesis').textContent = s.hypothesis || 'No thesis available.';
  $('m-disc').textContent = (P.provenance_note || '') + ' Prices/fundamentals: Groww MCP. For information only; not investment advice.';
  $('modal').classList.add('open');
}
function closeModal(){ $('modal').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
