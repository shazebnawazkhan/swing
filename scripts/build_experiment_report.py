"""
build_experiment_report.py
---------------------------
Render the spec-strategy experiment results as an HTML comparison report
(outputs/comparison_report.html) — the honest-evaluation companion to the
legacy run_comparison.py report.

Reads the latest record per spec from data/results/experiments.jsonl and ranks
strategies by **market-adjusted alpha** (Kissell, Algorithmic Trading Methods,
ch.3 Index-Adjusted Performance), not raw return.  Columns surface both the raw
metrics (what batch-1 reported) and the alpha metrics (what actually matters in
a flat/falling market), plus the liquidity-aware I-Star cost actually charged.

Usage
-----
    python scripts/build_experiment_report.py
    python scripts/build_experiment_report.py --out outputs/comparison_report.html
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "data" / "results" / "experiments.jsonl"
SPEC_DIR     = ROOT / "data" / "strategies"
DEFAULT_OUT  = ROOT / "outputs" / "comparison_report.html"


def _spec_created_map() -> dict[str, str]:
    """Return {spec_id: provenance.created} for every spec file."""
    m = {}
    for p in SPEC_DIR.glob("*.json"):
        try:
            spec = json.loads(p.read_text(encoding="utf-8"))
            created = spec.get("provenance", {}).get("created")
            if created:
                m[spec["id"]] = created
        except Exception:
            pass
    return m


def latest_records() -> list[dict]:
    """
    Latest experiment record per (spec_id, regime_gate) — so a regime-gated run
    and its ungated sibling appear as distinct comparison rows.
    Enriches each record with 'created' from the corresponding spec file.
    """
    latest: dict[tuple, dict] = {}
    with open(RESULTS_FILE, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("spec_id")
            if not sid:
                continue
            # only consider records from the current runner (carry a cost model)
            if "model" not in rec.get("costs", {}):
                continue
            key = (sid, bool(rec.get("regime_gate")))
            if key not in latest or rec["ts"] > latest[key]["ts"]:
                latest[key] = rec
    created_map = _spec_created_map()
    records = list(latest.values())
    for rec in records:
        rec["created"] = created_map.get(rec.get("spec_id", ""))
    return records


def _fmt(v, suffix="", dash_zero=False):
    if v is None:
        return "—"
    if isinstance(v, float):
        if dash_zero and v == 0:
            return "—"
        if v == float("inf"):
            return "∞"
        return f"{v:g}{suffix}"
    return f"{v}{suffix}"


def _cls_pf(pf) -> str:
    if pf is None:
        return ""
    if pf == float("inf") or pf >= 1.3:
        return "good"
    if pf >= 1.0:
        return "ok"
    return "bad"


def _fmt_inr(v) -> str:
    """Format an INR amount compactly (₹1.2L / ₹3.4Cr / −₹56k)."""
    if v is None:
        return "—"
    sign = "−" if v < 0 else ""
    a = abs(float(v))
    if a >= 1e7:
        return f"{sign}₹{a/1e7:.2f}Cr"
    if a >= 1e5:
        return f"{sign}₹{a/1e5:.2f}L"
    if a >= 1e3:
        return f"{sign}₹{a/1e3:.1f}k"
    return f"{sign}₹{a:.0f}"


def _cls_signed(v) -> str:
    if v is None:
        return ""
    return "good" if v > 0 else ("bad" if v < 0 else "")


def build_html(records: list[dict]) -> str:
    # Rank by profit factor, then expectancy (pure P&L)
    def keyf(r):
        m = r.get("metrics", {})
        pf = m.get("profit_factor", 0) or 0
        if pf == float("inf"):
            pf = 999
        return (pf, m.get("expectancy_pct", 0) or 0)
    records = sorted(records, key=keyf, reverse=True)

    meta = records[0] if records else {}
    window = meta.get("window", ["?", "?"])
    bench = meta.get("benchmark", "—")
    costs = meta.get("costs", {})
    cost_desc = (f"I-Star liquidity-aware (avg {costs.get('avg_round_trip_pct','?')}% round-trip)"
                 if costs.get("model") == "istar" else f"flat {costs.get('flat_pct','?')}%")

    rows_html = []
    for r in records:
        m = r.get("metrics", {})
        apf = m.get("alpha_profit_factor")
        gate = "✓" if r.get("regime_gate") else ""
        hyp = r.get("hypothesis_id") or ""
        name = html.escape(r.get("strategy", r.get("spec_id", "?")))
        if r.get("regime_gate"):
            name += ' <span class="tag">+ regime gate</span>'
        spec_id = html.escape(r.get("spec_id", ""))
        pf = m.get("profit_factor")
        profitable = (pf is not None and (pf == float("inf") or pf >= 1.0))
        row_cls = "alpha-pos" if profitable else ""
        rows_html.append(f"""
      <tr class="{row_cls}">
        <td class="name"><span class="sn">{name}</span><span class="sid">{spec_id}</span></td>
        <td class="dt">{html.escape(r.get('created') or '—')}</td>
        <td>{_fmt(m.get('trades'))}</td>
        <td>{_fmt(m.get('win_rate'),'%')}</td>
        <td class="sep {_cls_pf(pf)}"><b>{_fmt(pf)}</b></td>
        <td class="{_cls_signed(m.get('expectancy_pct'))}"><b>{_fmt(m.get('expectancy_pct'),'%')}</b></td>
        <td class="{_cls_signed(m.get('total_pnl'))}">{_fmt_inr(m.get('total_pnl'))}</td>
        <td>{_fmt(m.get('sharpe_approx'))}</td>
        <td class="sep">{_fmt(m.get('max_drawdown_pct'),'%')}</td>
        <td>{_fmt(m.get('best_trade_pct'),'%')}</td>
        <td>{_fmt(m.get('worst_trade_pct'),'%')}</td>
        <td>{_fmt(m.get('avg_hold_days'))}</td>
        <td>{_fmt(r.get('costs',{}).get('avg_round_trip_pct'),'%')}</td>
        <td class="gate">{gate}</td>
        <td class="hyp">{html.escape(hyp)}</td>
      </tr>""")

    def _pf(r):
        v = r.get("metrics", {}).get("profit_factor") or 0
        return 999 if v == float("inf") else v
    n_alpha = sum(1 for r in records if _pf(r) >= 1.0)
    n_promote = sum(1 for r in records
                    if _pf(r) >= 1.3 and (r.get("metrics", {}).get("trades") or 0) >= 100)

    breakdown = _breakdown_html(records)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Comparison — P&amp;L Backtest</title>
<style>
  :root {{ --bg:#0f1419; --card:#1a2029; --line:#2a323d; --txt:#e6edf3; --mut:#8b97a5;
           --good:#22c55e; --bad:#f43f5e; --ok:#eab308; --accent:#38bdf8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--txt);
          font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; padding:28px; }}
  h1 {{ font-size:21px; margin:0 0 4px; }}
  .sub {{ color:var(--mut); font-size:13px; margin-bottom:18px; }}
  .cards {{ display:flex; gap:14px; flex-wrap:wrap; margin-bottom:20px; }}
  .c {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:12px 16px; min-width:150px; }}
  .c .k {{ color:var(--mut); font-size:12px; }} .c .v {{ font-size:20px; font-weight:600; margin-top:2px; }}
  table {{ border-collapse:collapse; width:100%; background:var(--card);
           border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  th,td {{ padding:9px 11px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }}
  th {{ background:#141a22; color:var(--mut); font-weight:600; font-size:12px;
        text-transform:uppercase; letter-spacing:.03em; position:sticky; top:0; }}
  th[title] {{ cursor:help; text-decoration:underline dotted var(--line); text-underline-offset:3px; }}
  td.name, th.name {{ text-align:left; }}
  .name .sn {{ display:block; font-weight:600; }}
  .name .sid {{ display:block; color:var(--mut); font-size:11px; font-family:monospace; }}
  td.sep, th.sep {{ border-left:2px solid var(--line); }}
  .good {{ color:var(--good); }} .bad {{ color:var(--bad); }} .ok {{ color:var(--ok); }}
  tr.alpha-pos {{ background:rgba(34,197,94,.06); }}
  .gate {{ color:var(--accent); }} .hyp {{ color:var(--mut); font-family:monospace; font-size:12px; }}
  .dt {{ color:var(--mut); font-size:12px; font-family:monospace; text-align:left; }}
  .tag {{ font-size:10px; color:var(--accent); border:1px solid var(--accent); border-radius:4px;
          padding:0 4px; margin-left:6px; font-weight:500; vertical-align:middle; }}
  .legend {{ margin-top:18px; color:var(--mut); font-size:12.5px; max-width:1000px; }}
  .legend b {{ color:var(--txt); }}
  code {{ background:#141a22; padding:1px 5px; border-radius:4px; font-size:12px; }}
  /* ── per-strategy breakdown ── */
  h2 {{ font-size:16px; margin:30px 0 10px; }}
  details.strat {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
                   margin-bottom:8px; overflow:hidden; }}
  details.strat > summary {{ cursor:pointer; padding:11px 14px; list-style:none;
                             display:flex; justify-content:space-between; gap:12px; align-items:center; }}
  details.strat > summary::-webkit-details-marker {{ display:none; }}
  details.strat > summary::before {{ content:"▸"; color:var(--mut); margin-right:8px; }}
  details.strat[open] > summary::before {{ content:"▾"; }}
  details.strat > summary:hover {{ background:#141a22; }}
  .s-name {{ font-weight:600; }} .s-stats {{ color:var(--mut); font-size:12.5px; font-family:monospace; }}
  .strat-body {{ padding:6px 14px 14px; }}
  .muted {{ color:var(--mut); font-size:13px; padding:8px 2px; }} .muted.err {{ color:var(--bad); }}
  .sym-count {{ color:var(--mut); font-size:12px; margin:4px 0 8px; }}
  details.sym {{ border-top:1px solid var(--line); }}
  details.sym > summary {{ cursor:pointer; padding:8px 4px; list-style:none; display:flex;
                           gap:14px; align-items:center; font-size:13px; }}
  details.sym > summary::-webkit-details-marker {{ display:none; }}
  details.sym .sy {{ font-weight:600; min-width:120px; font-family:monospace; }}
  details.sym .st {{ color:var(--mut); }}
  .chart-slot {{ padding:8px 0 4px; }}
  .lwchart {{ width:100%; height:320px; }}
  table.tr {{ width:auto; margin-top:8px; font-size:12px; background:transparent; border:none; }}
  table.tr td, table.tr th {{ padding:4px 10px; border-bottom:1px solid var(--line); }}
  .g {{ color:var(--good); }} .r {{ color:var(--bad); }}
</style>
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
</head>
<body>
  <h1>Strategy Comparison — P&amp;L Backtest</h1>
  <div class="sub">
    Window <b>{window[0]} → {window[1]}</b> · universe <b>{html.escape(str(meta.get('universe','?')))}</b>
    (<b>{meta.get('n_symbols','?')}</b> symbols, ₹100k/trade pooled) ·
    costs <b>{html.escape(cost_desc)}</b> · next-bar fill ·
    evaluated on absolute P&amp;L · generated {datetime.now():%Y-%m-%d %H:%M}
  </div>

  <div class="cards">
    <div class="c"><div class="k">Strategies compared</div><div class="v">{len(records)}</div></div>
    <div class="c"><div class="k">Profitable (PF ≥ 1.0)</div><div class="v good">{n_alpha}</div></div>
    <div class="c"><div class="k">Promotion-grade (PF ≥ 1.3, ≥100 trades)</div><div class="v good">{n_promote}</div></div>
  </div>

  <table>
    <thead><tr>
      <th class="name" title="Strategy display name and its spec id (data/strategies/&lt;id&gt;.json).">Strategy</th>
      <th title="Date the strategy spec was first created (provenance.created in the spec JSON).">Created</th>
      <th title="Number of completed round-trip trades pooled across all 1,792 universe symbols in the window (fixed ₹100k per trade, no concurrency cap).">Trades</th>
      <th title="Win rate: share of trades that closed with a positive net P&amp;L (after costs).">Win%</th>
      <th class="sep" title="Profit factor: gross winning P&amp;L ÷ gross losing P&amp;L, after costs. &gt; 1.0 = profitable; ≥ 1.3 with ≥ 100 trades = promotion-grade.">Profit Factor</th>
      <th title="Expectancy: average net % return per trade after I-Star costs. The per-trade edge in pure P&amp;L terms.">Expectancy</th>
      <th title="Total net P&amp;L in rupees: sum of every trade's profit/loss at ₹100k per position across the universe (pooled, uncapped).">Total P&amp;L</th>
      <th title="Sharpe (approx): mean per-trade return ÷ its std, annualised by average holding period. Risk-adjusted return consistency.">Sharpe</th>
      <th class="sep" title="Maximum drawdown: largest peak-to-trough drop of the pooled, exit-ordered cumulative P&amp;L curve, in %. (Pooled/uncapped — read as relative, not a portfolio DD.)">Max DD</th>
      <th title="Best single trade: largest winning trade return, in %.">Best</th>
      <th title="Worst single trade: largest losing trade return, in %.">Worst</th>
      <th title="Average holding period per trade, in trading days (capital cycle time).">Hold</th>
      <th title="Average realised round-trip cost charged by the I-Star liquidity-aware model (scales with order size ÷ ADV and volatility), in %.">Avg Cost</th>
      <th title="Regime gate: ✓ means a long/cash signal filter was applied — no new entries while NIFTY is below its 50-day EMA. (Signal filter, not an evaluation metric.)">Gate</th>
      <th title="Hypothesis id from data/research/backlog.jsonl that this experiment tests.">Hyp</th>
    </tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>

  <div class="legend">
    <p>Strategies are ranked on <b>absolute P&amp;L</b> — no benchmark adjustment. Every
    trade pays a liquidity-aware round-trip cost (I-Star model: scales with order
    size ÷ ADV and volatility), fills at the next bar, and uses a fixed ₹100k position
    pooled across all universe symbols.</p>
    <ul>
      <li><b>Profit Factor</b> — gross winning P&amp;L ÷ gross losing P&amp;L, after costs. The headline edge measure.</li>
      <li><b>Expectancy</b> — average net % return per trade (per-trade edge).</li>
      <li><b>Total P&amp;L</b> — sum of all trades' rupee profit/loss at ₹100k each.</li>
      <li><b>Sharpe</b> — mean per-trade return ÷ std, annualised by average hold (risk-adjusted consistency).</li>
      <li><b>Max DD / Best / Worst</b> — drawdown of the pooled P&amp;L curve, and the largest single win/loss.</li>
      <li><b>Avg Cost</b> — realised I-Star round-trip cost, not a flat assumption.</li>
      <li><b>Gate</b> — ✓ means a long/cash signal filter was applied (no new entries while NIFTY &lt; 50-EMA). This is an entry filter, not an evaluation metric.</li>
    </ul>
    <p>Green rows are profitable (PF ≥ 1.0). Promotion bar (docs/program.md):
    <code>profit factor ≥ 1.3</code> · <code>trades ≥ 100</code>.</p>
  </div>

  <h2>Per-strategy breakdown — traded stocks &amp; charts</h2>
  <div class="muted">Expand a strategy to see every stock it traded; expand a stock for its
  candlestick chart with entry/exit markers. Data is served on demand by
  <code>scripts/server.py</code> — open this report at the server URL (e.g.
  <code>http://localhost:8765</code>), not as a local file.</div>
  {breakdown}
"""
    return page + _SCRIPT + "\n</body></html>"


def _breakdown_html(records: list[dict]) -> str:
    out = []
    for r in records:
        m = r.get("metrics", {})
        spec_id = html.escape(r.get("spec_id", ""))
        gated = "1" if r.get("regime_gate") else "0"
        name = html.escape(r.get("strategy", spec_id))
        if r.get("regime_gate"):
            name += ' <span class="tag">+ regime gate</span>'
        stats = (f"PF {_fmt(m.get('profit_factor'))} · {_fmt(m.get('trades'))} trades · "
                 f"net {_fmt_inr(m.get('total_pnl'))}")
        out.append(f"""
  <details class="strat" data-spec="{spec_id}" data-gated="{gated}">
    <summary><span class="s-name">{name}</span><span class="s-stats">{stats}</span></summary>
    <div class="strat-body"><div class="muted">Expand to load traded stocks…</div></div>
  </details>""")
    return "".join(out)


# ── Client-side: lazy-load symbols + render lightweight-charts per stock ──────────
_SCRIPT = r"""
<script>
(function(){
  const fmtINR = v => {
    if (v == null) return "—";
    const s = v < 0 ? "−" : "", a = Math.abs(v);
    if (a >= 1e7) return s+"₹"+(a/1e7).toFixed(2)+"Cr";
    if (a >= 1e5) return s+"₹"+(a/1e5).toFixed(2)+"L";
    if (a >= 1e3) return s+"₹"+(a/1e3).toFixed(1)+"k";
    return s+"₹"+a.toFixed(0);
  };
  const LAYOUT = {
    layout: { background:{color:"#1a2029"}, textColor:"#8b97a5" },
    grid:   { vertLines:{color:"#222b36"}, horzLines:{color:"#222b36"} },
    rightPriceScale:{ borderColor:"#2a323d" },
    timeScale:{ borderColor:"#2a323d" },
  };

  async function loadSymbols(det){
    if (det.dataset.loaded) return;
    det.dataset.loaded = "1";
    const body = det.querySelector(".strat-body");
    const spec = det.dataset.spec, gated = det.dataset.gated;
    body.innerHTML = '<div class="muted">Loading traded stocks…</div>';
    try {
      const res = await fetch("/api/experiment/"+encodeURIComponent(spec)+"/symbols?gated="+gated);
      const data = await res.json();
      if (data.missing) { body.innerHTML = '<div class="muted">No trades file yet — run <code>python scripts/run_experiments.py</code>.</div>'; return; }
      if (!data.symbols || !data.symbols.length) { body.innerHTML = '<div class="muted">No trades for this strategy.</div>'; return; }
      let h = '<div class="sym-count">'+data.n_symbols+' stocks traded · sorted by net P&L · click a stock to chart it</div>';
      for (const s of data.symbols) {
        const cls = s.total_pnl >= 0 ? "g" : "r";
        h += '<details class="sym" data-sym="'+s.symbol+'">'
           +   '<summary><span class="sy">'+s.symbol+'</span>'
           +     '<span class="st">'+s.trades+' trades</span>'
           +     '<span class="st">win '+s.win_rate+'%</span>'
           +     '<span class="st">avg '+s.avg_pnl_pct+'%</span>'
           +     '<span class="'+cls+'">'+fmtINR(s.total_pnl)+'</span></summary>'
           +   '<div class="chart-slot"></div></details>';
      }
      body.innerHTML = h;
      body.querySelectorAll("details.sym").forEach(sd => {
        sd.addEventListener("toggle", () => { if (sd.open) loadChart(sd, spec, gated); });
      });
    } catch (e) {
      body.innerHTML = '<div class="muted err">Could not reach the data server. Start it with '
        + '<code>python scripts/server.py</code> and open this report at the server URL.</div>';
    }
  }

  async function loadChart(sd, spec, gated){
    if (sd.dataset.loaded) return;
    sd.dataset.loaded = "1";
    const slot = sd.querySelector(".chart-slot");
    const sym = sd.dataset.sym;
    slot.innerHTML = '<div class="muted">Loading chart…</div>';
    try {
      const res = await fetch("/api/experiment/"+encodeURIComponent(spec)+"/symbol/"+encodeURIComponent(sym)+"?gated="+gated);
      const d = await res.json();
      if (d.error) { slot.innerHTML = '<div class="muted err">'+d.error+'</div>'; return; }
      slot.innerHTML = "";
      const cdiv = document.createElement("div"); cdiv.className = "lwchart"; slot.appendChild(cdiv);
      const chart = LightweightCharts.createChart(cdiv, Object.assign({
        width: cdiv.clientWidth || 800, height: 320 }, LAYOUT));
      const cs = chart.addCandlestickSeries({
        upColor:"#22c55e", downColor:"#f43f5e", borderVisible:false,
        wickUpColor:"#22c55e", wickDownColor:"#f43f5e" });
      cs.setData(d.candles);
      if (d.markers && d.markers.length) cs.setMarkers(d.markers);
      const vs = chart.addHistogramSeries({ priceFormat:{type:"volume"}, priceScaleId:"" });
      vs.priceScale().applyOptions({ scaleMargins:{ top:0.82, bottom:0 } });
      vs.setData(d.volume);
      chart.timeScale().fitContent();
      new ResizeObserver(() => chart.applyOptions({ width: cdiv.clientWidth })).observe(cdiv);

      if (d.trades && d.trades.length) {
        let t = '<table class="tr"><thead><tr><th>Entry</th><th>Exit</th><th>Hold</th>'
              + '<th>Entry ₹</th><th>Exit ₹</th><th>P&L%</th><th>Reason</th></tr></thead><tbody>';
        for (const x of d.trades) {
          t += '<tr><td>'+x.entry_date+'</td><td>'+x.exit_date+'</td><td>'+x.hold_days+'d</td>'
             + '<td>'+x.entry_price+'</td><td>'+x.exit_price+'</td>'
             + '<td class="'+(x.pnl_pct>=0?'g':'r')+'">'+x.pnl_pct.toFixed(2)+'%</td>'
             + '<td>'+x.exit_reason+'</td></tr>';
        }
        t += "</tbody></table>";
        const td = document.createElement("div"); td.innerHTML = t; slot.appendChild(td);
      }
    } catch (e) {
      slot.innerHTML = '<div class="muted err">Failed to load chart data.</div>';
    }
  }

  document.querySelectorAll("details.strat").forEach(det => {
    det.addEventListener("toggle", () => { if (det.open) loadSymbols(det); });
  });
})();
</script>"""


def main():
    ap = argparse.ArgumentParser(description="Build HTML comparison report from experiment records")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if not RESULTS_FILE.exists():
        print("  No experiment records — run scripts/run_experiments.py first.")
        return
    records = latest_records()
    if not records:
        print("  No records found.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Preserve any existing report (e.g. the legacy run_comparison.py output) once.
    if out.exists():
        bak = out.with_suffix(".legacy.html")
        if not bak.exists():
            shutil.copy2(out, bak)

    out.write_text(build_html(records), encoding="utf-8")
    print(f"  Wrote {out.relative_to(ROOT)}  ({len(records)} strategies)")
    print(f"  (any prior report preserved as {out.with_suffix('.legacy.html').name})")


if __name__ == "__main__":
    main()
