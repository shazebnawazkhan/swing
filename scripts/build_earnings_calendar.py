"""
scripts/build_earnings_calendar.py
----------------------------------
Find every NSE company declaring quarterly results in the upcoming week, enrich each
with price-relevant KPIs (revenue / profit / EPS growth, margins, valuation) and the
analyst-firm target price, and render a tabbed dashboard — one tab per date.

Outputs (both consumed by scripts/server.py at /earnings):
  outputs/earnings_calendar_latest.json   the raw payload (grouped by date)
  outputs/earnings_calendar.html          self-contained dashboard, one tab per date

Run
  python scripts/build_earnings_calendar.py                 # next 7 days
  python scripts/build_earnings_calendar.py --days 10
  python scripts/build_earnings_calendar.py --limit 40      # cap enrichment (rate-limit guard)
  python scripts/build_earnings_calendar.py --no-enrich     # calendar only, skip yfinance
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Redirected/background stdout defaults to cp1252 on Windows and chokes on non-ASCII;
# force UTF-8 so progress prints (and any arrows) never crash the run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predictor.adapters import earnings_calendar_adapter as eca  # noqa: E402

OUT_DIR = ROOT / "outputs"
STOCKS_CSV = ROOT / "data" / "stocks.csv"


def load_universe() -> tuple[dict, set, list]:
    """(company_map, halal_set, all_symbols) from stocks.csv — used to tag 'my' names."""
    if not STOCKS_CSV.exists():
        return {}, set(), []
    df = pd.read_csv(STOCKS_CSV, dtype=str).fillna("")
    company_map = dict(zip(df["stock"].str.strip(), df["company_name"].str.strip()))
    halal = set(df.loc[df["halal"].str.upper() == "Y", "stock"].str.strip())
    return company_map, halal, list(company_map)


def main():
    ap = argparse.ArgumentParser(description="Upcoming NSE quarterly-results dashboard")
    ap.add_argument("--days", type=int, default=7, help="calendar window from today (default 7)")
    ap.add_argument("--start", type=str, default=None, help="ISO start date (default today)")
    ap.add_argument("--limit", type=int, default=None, help="cap enriched companies")
    ap.add_argument("--workers", type=int, default=eca.DEFAULT_WORKERS,
                    help=f"enrichment thread-pool size (default {eca.DEFAULT_WORKERS})")
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="optional per-symbol delay inside each worker (default 0; rate is bounded by --workers)")
    ap.add_argument("--no-enrich", action="store_true", help="skip KPI/target enrichment")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    start = date.fromisoformat(args.start) if args.start else date.today()
    end = start + timedelta(days=args.days)
    verbose = not args.quiet

    company_map, halal, universe = load_universe()
    if verbose:
        print(f"Results calendar {start} -> {end}  ({len(universe)} symbols in universe)")

    payload = eca.build_calendar(
        start, end, company_map=company_map, halal=halal, universe=universe,
        limit=args.limit, throttle=(0.0 if args.no_enrich else args.throttle),
        workers=(1 if args.no_enrich else args.workers), verbose=verbose,
    )
    # --no-enrich: strip enrichment so the run is calendar-only (no yfinance).
    if args.no_enrich:
        for d in payload["dates"]:
            for r in d["rows"]:
                for k in list(r):
                    if k not in ("symbol", "company", "purpose", "source_url",
                                 "in_universe", "halal", "refs"):
                        r[k] = None

    OUT_DIR.mkdir(exist_ok=True)
    json_path = OUT_DIR / "earnings_calendar_latest.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    html = build_html(payload)
    html_path = OUT_DIR / "earnings_calendar.html"
    html_path.write_text(html, encoding="utf-8")

    c = payload["counts"]
    print(f"Wrote {json_path.relative_to(ROOT)} and {html_path.relative_to(ROOT)}")
    print(f"  {c['companies']} companies across {c['days']} day(s); "
          f"{c['in_universe']} in your universe ({c['halal']} halal)")
    if c["companies"] == 0:
        print("  NOTE: no rows — NSE calendar was blocked/empty in this environment. "
              "The dashboard shows a fetch-failure banner; re-run where NSE is reachable.")


def build_html(payload: dict) -> str:
    return _TEMPLATE.replace("/*DATA*/", json.dumps(payload))


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Upcoming Quarterly Results — NSE</title>
<style>
  :root { --bg:#0d1520; --card:#162032; --line:#1e2d40; --txt:#dce8f5; --mut:#7e92a8;
          --pos:#26d96e; --neg:#f04f5f; --watch:#f5a623; --accent:#3db8f5; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; padding:24px; }
  h1 { margin:0 0 4px; font-size:21px; } .sub { color:var(--mut); font-size:13px; margin-bottom:14px; }
  a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
  .row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }
  .kpi { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:110px; }
  .kpi .k { color:var(--mut); font-size:12px; } .kpi .v { font-size:20px; font-weight:600; margin-top:3px; }
  .banner { padding:10px 14px; border-radius:8px; margin-bottom:16px; font-size:13px;
            border:1px solid var(--watch); background:rgba(245,166,35,.08); color:var(--watch); }
  .tabs { display:flex; gap:6px; flex-wrap:wrap; border-bottom:1px solid var(--line); margin-bottom:14px; }
  .tab { padding:8px 14px; border:1px solid var(--line); border-bottom:none; border-radius:8px 8px 0 0;
         background:var(--card); color:var(--mut); cursor:pointer; font-size:13px; user-select:none; }
  .tab.active { color:var(--txt); background:#0d1826; border-color:var(--accent); font-weight:600; }
  .tab .n { display:inline-block; margin-left:6px; font-size:11px; color:var(--accent); }
  .tab .wd { display:block; font-size:11px; color:var(--mut); font-weight:400; }
  .controls { display:flex; gap:10px; margin-bottom:10px; flex-wrap:wrap; align-items:center; }
  .controls input, .controls label { font-size:13px; }
  .controls input[type=text] { background:var(--card); border:1px solid var(--line); color:var(--txt);
    border-radius:6px; padding:6px 9px; }
  .controls label { color:var(--mut); display:inline-flex; gap:5px; align-items:center; }
  .tbl-wrap { overflow:auto; border:1px solid var(--line); border-radius:10px; }
  table { border-collapse:collapse; width:100%; background:var(--card); }
  th,td { padding:8px 10px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { background:#0d1826; color:var(--mut); font-size:11px; text-transform:uppercase; letter-spacing:.03em;
       cursor:pointer; user-select:none; position:sticky; top:0; z-index:2; }
  td.l, th.l { text-align:left; }
  .pos { color:var(--pos); } .neg { color:var(--neg); } .mut { color:var(--mut); } .watch { color:var(--watch); }
  .tag { font-size:10px; font-weight:700; padding:1px 6px; border-radius:9px; margin-left:6px; }
  .tag.mine { background:rgba(61,184,245,.16); color:var(--accent); }
  .tag.halal { background:rgba(38,217,110,.16); color:var(--pos); }
  /* KPI provenance: muted by default, warmer when Yahoo had to fill in. */
  .tag.src-yf, .tag.src-mix { background:rgba(245,166,35,.14); color:var(--watch); }
  .reco { font-size:10px; font-weight:700; padding:1px 7px; border-radius:9px; text-transform:capitalize; }
  .reco.buy,.reco.strong_buy { background:rgba(38,217,110,.16); color:var(--pos); }
  .reco.hold,.reco.none { background:rgba(245,166,35,.14); color:var(--watch); }
  .reco.sell,.reco.underperform { background:rgba(240,79,95,.14); color:var(--neg); }
  .refs a { font-size:11px; margin-right:8px; }
  .empty { padding:30px; text-align:center; color:var(--mut); }
  .company { max-width:230px; overflow:hidden; text-overflow:ellipsis; display:inline-block; vertical-align:bottom; }
</style></head><body>
<h1>Upcoming Quarterly Results — NSE</h1>
<div class="sub" id="sub"></div>
<div class="row" id="kpis"></div>
<div id="banner"></div>
<div class="controls">
  <input type="text" id="search" placeholder="filter symbol / company…" oninput="render()">
  <label><input type="checkbox" id="mineonly" onchange="render()"> my universe only</label>
  <label><input type="checkbox" id="halalonly" onchange="render()"> halal only</label>
  <span class="sub" id="count" style="margin-left:auto"></span>
</div>
<div class="tabs" id="tabs"></div>
<div class="tbl-wrap"><table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div>

<script>
const D = /*DATA*/;
let activeTab = 0, sortKey = "target_upside_pct", sortAsc = false;

const COLS = [
  ["symbol","Symbol",true],["company","Company",true],["market_cap","Mkt cap",false],
  ["revenue_growth_pct","Rev gr% (ttm)",false],["earnings_growth_pct","Profit gr% (ttm)",false],
  ["fwd_rev_growth_pct","Rev gr% (fwd)",false],["fwd_eps_growth_pct","EPS gr% (fwd)",false],
  ["profit_margin_pct","Net mgn%",false],
  ["trailing_pe","PE",false],["forward_pe","Fwd PE",false],
  ["current_price","Price",false],["target_mean","Target",false],
  ["target_as_of","Tgt date",true],["target_upside_pct","Upside%",false],
  ["n_analysts","#Firms",false],["recommendation","Reco",true],["refs","References",true],
];
const esc = s => String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const signed = (v,nd=1) => v==null ? '<span class="mut">–</span>'
  : `<span class="${v>=0?'pos':'neg'}">${v>=0?'+':''}${v.toFixed(nd)}</span>`;
const daysAgo = d => { if(!d) return null; const t=Date.parse(d); return isNaN(t)?null:Math.round((Date.now()-t)/86400000); };
// Market cap in the Indian convention: 1 crore = 1e7.
const crore = v => { if(v==null) return null; const c=v/1e7;
  return c>=1e5 ? (c/1e5).toFixed(2)+"L Cr" : Math.round(c).toLocaleString("en-IN")+" Cr"; };
// Where a row's KPIs came from. stockanalysis is the norm (~96% of rows) so it stays
// unbadged; only Yahoo-touched rows get a mark, since those are the thinner/staler ones.
const SRC = {yfinance:["YF","src-yf","yfinance fallback - stockanalysis had no coverage"],
             "stockanalysis+yfinance":["SA+","src-mix","stockanalysis, with yfinance filling gaps"]};

function fmt(k,v,row){
  if(k==="symbol"){ let t=`<b>${esc(v)}</b>`;
    if(row.in_universe) t+=`<span class="tag mine">mine</span>`;
    if(row.halal) t+=`<span class="tag halal">halal</span>`;
    const s=SRC[row.kpi_source];
    if(s) t+=`<span class="tag ${s[1]}" title="${esc(s[2])}">${s[0]}</span>`;
    return t; }
  if(k==="company") return `<span class="company" title="${esc(v)}">${esc(v||'–')}</span>`;
  if(k==="market_cap"){ const c=crore(v);
    return c==null?'<span class="mut">–</span>':`<span title="${esc(String(v))}">₹${c}</span>`; }
  if(k==="profit_margin_pct"){ if(v==null) return '<span class="mut">–</span>';
    const extra=[row.gross_margin_pct!=null?`gross ${row.gross_margin_pct}%`:null,
                 row.operating_margin_pct!=null?`operating ${row.operating_margin_pct}%`:null]
                .filter(Boolean).join(" · ");
    return `<span class="${v>=0?'pos':'neg'}" title="${esc(extra||'net margin')}">${v>=0?'+':''}${v.toFixed(1)}${extra?'*':''}</span>`; }
  if(["revenue_growth_pct","earnings_growth_pct","fwd_rev_growth_pct","fwd_eps_growth_pct",
      "target_upside_pct"].includes(k)) return signed(v);
  if(k==="target_mean"){ if(v==null) return '<span class="mut">–</span>';
    const rng=(row.target_low!=null&&row.target_high!=null)?` <span class="mut">(${row.target_low}–${row.target_high})</span>`:'';
    const src={stockanalysis:'S&P Global consensus via stockanalysis.com',
               stockanalysis_quote:'stockanalysis.com quote page (no low/high band)',
               yfinance:'yfinance (fallback)'}[row.target_source]||'';
    return `<span title="${esc(src)}">${v}${rng}</span>`; }
  if(k==="target_as_of"){ const n=daysAgo(v);
    if(v==null) return row.target_source==='yfinance'?'<span class="mut" title="yfinance gives no as-of date">n/a</span>':'<span class="mut">–</span>';
    const cls=n==null?'mut':(n>90?'neg':(n>45?'watch':'mut'));
    return `<span class="${cls}" title="analyst target last updated ${esc(v)} (${n} days ago)">${esc(v)}${n>90?' ⚠':''}</span>`; }
  if(k==="recommendation"){ if(!v) return '<span class="mut">–</span>';
    return `<span class="reco ${esc(v)}">${esc(v).replace('_',' ')}</span>`; }
  if(k==="refs"){ if(!v) return ''; return `<span class="refs">`+
    Object.entries(v).map(([n,u])=>`<a href="${esc(u)}" target="_blank" rel="noopener">${esc(n)}↗</a>`).join("")+`</span>`; }
  if(v==null) return '<span class="mut">–</span>';
  if(typeof v==="number") return Number.isInteger(v)?v:v.toFixed(2);
  return esc(v);
}

function tabs(){
  document.getElementById("tabs").innerHTML = D.dates.map((d,i)=>
    `<div class="tab ${i===activeTab?'active':''}" onclick="setTab(${i})">${esc(d.label)}
       <span class="wd">${esc(d.weekday)}</span><span class="n">${d.count}</span></div>`).join("")
    || '<div class="sub">no dates</div>';
}
function setTab(i){ activeTab=i; render(); tabs(); }
function header(){
  document.getElementById("head").innerHTML = COLS.map(c=>{
    const arrow = sortKey===c[0] ? (sortAsc?" ▲":" ▼") : "";
    return `<th class="${c[2]?'l':''}" onclick="sortBy('${c[0]}')">${c[1]}${arrow}</th>`;
  }).join("");
}
function sortBy(k){ if(sortKey===k) sortAsc=!sortAsc; else {sortKey=k; sortAsc=false;} render(); }

function render(){
  header();
  const day = D.dates[activeTab];
  const body = document.getElementById("body");
  if(!day){ body.innerHTML=`<tr><td class="empty" colspan="${COLS.length}">No results scheduled.</td></tr>`;
    document.getElementById("count").textContent=""; return; }
  const q=(document.getElementById("search").value||"").toUpperCase();
  const mine=document.getElementById("mineonly").checked;
  const hal=document.getElementById("halalonly").checked;
  let rows=day.rows.filter(r=>
    (!q || r.symbol.includes(q) || (r.company||"").toUpperCase().includes(q)) &&
    (!mine || r.in_universe) && (!hal || r.halal));
  rows.sort((a,b)=>{ let x=a[sortKey],y=b[sortKey];
    if(x==null&&y==null) return 0; if(x==null) return 1; if(y==null) return -1;
    if(typeof x==="string") return sortAsc?String(x).localeCompare(y):String(y).localeCompare(x);
    return sortAsc?(x-y):(y-x); });
  body.innerHTML = rows.length ? rows.map(r=>"<tr>"+COLS.map(c=>
    `<td class="${c[2]?'l':''}">${fmt(c[0],r[c[0]],r)}</td>`).join("")+"</tr>").join("")
    : `<tr><td class="empty" colspan="${COLS.length}">No rows match the filter.</td></tr>`;
  document.getElementById("count").textContent =
    `${rows.length} of ${day.rows.length} on ${day.label}`;
}

(function init(){
  const c=D.counts||{};
  document.getElementById("sub").innerHTML =
    `results between <b>${D.start}</b> and <b>${D.end}</b> · generated ${(D.generated||'').replace('T',' ')}`;
  const kpi=(k,v,cls)=>`<div class="kpi"><div class="k">${k}</div><div class="v ${cls||''}">${v}</div></div>`;
  // KPI/target coverage is the health signal for a run: a source that starts rate-limiting
  // mid-run shows up here as a fill rate well below ~95%, not as an error.
  const all=D.dates.reduce((a,d)=>a.concat(d.rows),[]);
  const pct=f=>all.length?Math.round(all.filter(f).length*100/all.length):0;
  const fill=pct(r=>r.current_price!=null), tgt=pct(r=>r.target_mean!=null);
  document.getElementById("kpis").innerHTML =
    kpi("Companies",c.companies||0)+kpi("Trading days",c.days||0)+
    kpi("In my universe",c.in_universe||0,"")+kpi("Halal",c.halal||0,"pos")+
    kpi("KPI coverage",fill+"%",fill>=90?"pos":(fill>=70?"watch":"neg"))+
    kpi("Analyst targets",tgt+"%","mut");
  if((c.companies||0)===0){
    document.getElementById("banner").innerHTML =
      "⚠ No results fetched. NSE's calendar API blocks datacenter/headless IPs — "+
      "re-run <code>python scripts/build_earnings_calendar.py</code> from a normal network, "+
      "or pass a reachable session. KPIs come from stockanalysis.com and also need outbound access.";
  } else if(fill<70){
    document.getElementById("banner").innerHTML =
      `⚠ Only ${fill}% of rows have KPIs — a data source likely rate-limited mid-run. `+
      "Re-run the build; if it persists, lower <code>--workers</code>.";
  }
  tabs(); render();
})();
</script></body></html>"""


if __name__ == "__main__":
    main()
