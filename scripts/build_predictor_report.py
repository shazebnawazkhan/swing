"""
scripts/build_predictor_report.py
---------------------------------
Render outputs/predictor.html — the interactive next-day signal dashboard
(docs/PREDICTOR.md §9). Self-contained: reads the latest predicted_signals JSON +
both heads' model metrics.json and inlines them, so the page opens with no server.

Run
  python scripts/build_predictor_report.py
  python scripts/predict_daily.py && python scripts/build_predictor_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
HEADS = ["dir1d", "swing"]


def load_signals() -> dict:
    p = OUT_DIR / "predicted_signals_latest.json"
    if not p.exists():
        raise SystemExit("No predicted_signals_latest.json — run scripts/predict_daily.py first.")
    return json.loads(p.read_text())


def load_model_card(head: str) -> dict | None:
    ptr = MODELS_DIR / head / "latest.txt"
    if not ptr.exists():
        return None
    run_id = ptr.read_text().strip()
    mp = MODELS_DIR / head / run_id / "metrics.json"
    card = json.loads(mp.read_text()) if mp.exists() else {}
    imp_path = MODELS_DIR / head / run_id / "importance.csv"
    if imp_path.exists():
        lines = imp_path.read_text().splitlines()[1:9]
        card["importance"] = [ln.split(",")[0] for ln in lines]
    return card


def build_html(sig: dict, cards: dict) -> str:
    data_json = json.dumps({"sig": sig, "cards": cards})
    return _TEMPLATE.replace("/*DATA*/", data_json)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Next-Day Signal Predictor</title>
<style>
  :root { --bg:#0d1520; --card:#162032; --line:#1e2d40; --txt:#dce8f5; --mut:#7e92a8;
          --buy:#26d96e; --avoid:#f04f5f; --watch:#f5a623; --accent:#3db8f5; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; padding:24px; }
  h1 { margin:0 0 4px; font-size:21px; } .sub { color:var(--mut); font-size:13px; margin-bottom:16px; }
  .banner { padding:10px 14px; border-radius:8px; margin-bottom:16px; font-size:13px;
            border:1px solid var(--line); background:var(--card); }
  .banner.warn { border-color:var(--watch); color:var(--watch); }
  .banner.go { border-color:var(--buy); color:var(--buy); }
  .row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:18px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
  .card.kpi { min-width:120px; } .card .k { color:var(--mut); font-size:12px; } .card .v { font-size:20px; font-weight:600; margin-top:3px; }
  .mcard { min-width:300px; flex:1; }
  .mcard h3 { margin:0 0 8px; font-size:14px; } .mcard .tag { font-size:10px; padding:1px 6px; border-radius:4px; margin-left:8px; vertical-align:middle; }
  .tag.candidate { background:rgba(38,217,110,.15); color:var(--buy); border:1px solid var(--buy); }
  .tag.rejected { background:rgba(240,79,95,.12); color:var(--avoid); border:1px solid var(--avoid); }
  .mgrid { display:grid; grid-template-columns:repeat(4,1fr); gap:8px 14px; font-size:13px; margin:8px 0; }
  .mgrid .k { color:var(--mut); font-size:11px; } .mgrid .v { font-weight:600; }
  .feats { color:var(--mut); font-size:12px; font-family:monospace; }
  .controls { display:flex; gap:10px; margin-bottom:10px; flex-wrap:wrap; align-items:center; }
  .controls input, .controls select { background:var(--card); border:1px solid var(--line);
    color:var(--txt); border-radius:6px; padding:6px 9px; font-size:13px; }
  table { border-collapse:collapse; width:100%; background:var(--card); border:1px solid var(--line);
          border-radius:10px; overflow:hidden; }
  th,td { padding:8px 11px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { background:#0d1826; color:var(--mut); font-size:11px; text-transform:uppercase; letter-spacing:.03em;
       cursor:pointer; user-select:none; position:sticky; top:0; }
  td.l, th.l { text-align:left; }
  .badge { font-size:11px; font-weight:700; padding:1px 8px; border-radius:10px; }
  .badge.BUY { background:rgba(38,217,110,.16); color:var(--buy); }
  .badge.WATCH { background:rgba(245,166,35,.15); color:var(--watch); }
  .badge.AVOID { background:rgba(240,79,95,.14); color:var(--avoid); }
  .pos { color:var(--buy); } .neg { color:var(--avoid); } .mut { color:var(--mut); }
  .why { color:var(--mut); font-size:11px; font-family:monospace; }
  .pbar { display:inline-block; width:42px; height:6px; background:var(--line); border-radius:3px; vertical-align:middle; margin-right:5px; }
  .pbar > i { display:block; height:100%; border-radius:3px; background:var(--accent); }
</style></head><body>
<h1>Next-Day Signal Predictor</h1>
<div class="sub" id="sub"></div>
<div id="banner"></div>
<div class="row" id="kpis"></div>
<div class="row" id="models"></div>
<div class="controls">
  <input id="search" placeholder="search symbol…" oninput="render()">
  <select id="dirf" onchange="render()">
    <option value="">all directions</option><option>BUY</option><option>WATCH</option><option>AVOID</option>
  </select>
  <span class="sub" id="count"></span>
</div>
<table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table>
<script>
const D = /*DATA*/;
const sig = D.sig, cards = D.cards;
let sortKey = "rank", sortAsc = true;
const COLS = [
  ["rank","#",false],["symbol","Symbol",true],["direction","Dir",true],
  ["p_up","P(up)",false],["p_win","P(win)",false],["expected_value_pct","EV%",false],
  ["news_sent","News",false],["headline","Headline / order-win",true],
  ["ref_close","Close",false],["stop","Stop",false],["target","Target",false],
  ["sector","Sector",true],["why","Why (top features)",true],
];
const esc = s => String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function fmt(k,v,row){
  if(v==null) return "–";
  if(k==="direction") return `<span class="badge ${v}">${v}</span>`;
  if(k==="p_up"||k==="p_win"){ const pct=Math.round(v*100);
    return `<span class="pbar"><i style="width:${pct}%"></i></span>${v.toFixed(2)}`; }
  if(k==="expected_value_pct") return `<span class="${v>=0?'pos':'neg'}">${v>=0?'+':''}${v.toFixed(2)}</span>`;
  if(k==="news_sent"){ if(v==null) return '<span class="mut">–</span>';
    const cls=v>0.1?'pos':(v<-0.1?'neg':'');
    const arrow=v>0.1?'▲':(v<-0.1?'▼':'■');
    return `<span class="${cls}" title="sentiment ${v.toFixed(2)} · ${row.news_count||0} items">${arrow} ${v.toFixed(2)}</span>`; }
  if(k==="headline"){ const ow=row.order_win?`<span class="badge BUY" title="order-win event">★ OW</span> `:"";
    const age=row.news_age_days!=null?` <span class="mut">(${row.news_age_days}d)</span>`:"";
    return ow + `<span class="why">${esc((v||"").slice(0,90))}</span>` + age; }
  if(k==="why") return `<span class="why">`+ (row.why||[]).map(w=>`${w[0]}=${w[1]==null?'·':w[1]}`).join("  ")+`</span>`;
  if(typeof v==="number") return Number.isInteger(v)?v:v.toFixed(2);
  return esc(v);
}

function header(){
  document.getElementById("head").innerHTML = COLS.map(c=>{
    const arrow = sortKey===c[0] ? (sortAsc?" ▲":" ▼") : "";
    return `<th class="${c[2]?'l':''}" onclick="sortBy('${c[0]}')">${c[1]}${arrow}</th>`;
  }).join("");
}
function sortBy(k){ if(sortKey===k) sortAsc=!sortAsc; else {sortKey=k; sortAsc=true;} render(); }

function render(){
  const q=(document.getElementById("search").value||"").toUpperCase();
  const df=document.getElementById("dirf").value;
  let rows=sig.signals.filter(s=>(!q||s.symbol.includes(q))&&(!df||s.direction===df));
  rows.sort((a,b)=>{ let x=a[sortKey],y=b[sortKey];
    if(typeof x==="string"){x=x||"";y=y||"";return sortAsc?x.localeCompare(y):y.localeCompare(x);}
    return sortAsc?(x-y):(y-x); });
  document.getElementById("body").innerHTML = rows.map(r=>"<tr>"+COLS.map(c=>
    `<td class="${c[2]?'l':''}">${fmt(c[0],r[c[0]],r)}</td>`).join("")+"</tr>").join("");
  document.getElementById("count").textContent = `${rows.length} shown`;
}

function modelCard(head){
  const c=cards[head]; if(!c) return `<div class="card mcard"><h3>${head}</h3><span class="sub">not trained</span></div>`;
  const b=c.basket||{}; const delta=(c.auc-c.base_auc);
  const g=(k,v)=>`<div><div class="k">${k}</div><div class="v">${v}</div></div>`;
  return `<div class="card mcard"><h3>${head==='dir1d'?'Head A · next-day direction':'Head B · swing win (triple-barrier)'}
    <span class="tag ${c.status}">${c.status}</span></h3>
    <div class="mgrid">
      ${g("AUC",(c.auc||0).toFixed(3))}${g("vs baseline",(delta>=0?'+':'')+delta.toFixed(3))}
      ${g("Brier",(c.brier||0).toFixed(3))}${g("base rate",((c.base_rate||0)*100).toFixed(0)+'%')}
      ${g("basket PF",(b.profit_factor||0).toFixed(2))}${g("precision@N",(b.precision_at_n||0)+'%')}
      ${g("trades",b.n_trades||0)}${g("valid from",c.valid_start||'–')}
    </div>
    <div class="feats">top: ${(c.importance||[]).slice(0,6).join(", ")}</div></div>`;
}

(function init(){
  document.getElementById("sub").textContent =
    `as-of ${sig.date} · universe ${sig.universe} · models ${Object.values(sig.model_runs||{}).join(" / ")} · generated ${sig.generated}`;
  const c=sig.counts||{};
  const kpi=(k,v,cls)=>`<div class="card kpi"><div class="k">${k}</div><div class="v ${cls||''}">${v}</div></div>`;
  document.getElementById("kpis").innerHTML =
    kpi("Scored",c.scored)+kpi("BUY",c.buy,"pos")+kpi("Watch",c.watch)+kpi("Avoid",c.avoid,"neg");
  const ban=document.getElementById("banner");
  if((c.buy||0)===0){ ban.className="banner warn";
    ban.textContent="Risk-off: no high-conviction BUYs for next session — showing the ranked watchlist by P(win). The model is declining to buy in this regime."; }
  else { ban.className="banner go"; ban.textContent=`${c.buy} high-conviction BUY signal(s) for next session.`; }
  document.getElementById("models").innerHTML = HEADS_HTML();
  header(); render();
})();
function HEADS_HTML(){ return ["dir1d","swing"].map(modelCard).join(""); }
</script></body></html>"""


def main():
    sig = load_signals()
    cards = {h: load_model_card(h) for h in HEADS}
    html = build_html(sig, cards)
    out = OUT_DIR / "predictor.html"
    out.write_text(html, encoding="utf-8")
    n = len(sig.get("signals", []))
    c = sig.get("counts", {})
    print(f"Wrote {out.relative_to(ROOT)}  ({n} signals; "
          f"BUY={c.get('buy')} WATCH={c.get('watch')} AVOID={c.get('avoid')})")


if __name__ == "__main__":
    main()
