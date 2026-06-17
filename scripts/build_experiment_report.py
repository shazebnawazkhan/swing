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
        gate = "✓" if r.get("regime_gate") else ""
        hyp = r.get("hypothesis_id") or ""
        name = html.escape(r.get("strategy", r.get("spec_id", "?")))
        if r.get("regime_gate"):
            name += ' <span class="tag">+ regime gate</span>'
        spec_id     = r.get("spec_id", "")
        spec_id_esc = html.escape(spec_id)
        gated_str   = "1" if r.get("regime_gate") else "0"
        # DOM id-safe key: spec IDs only use [a-z0-9_] so this is a no-op in practice
        row_key = spec_id.replace("-", "_").replace(".", "_") + ("_g" if r.get("regime_gate") else "_u")
        pf = m.get("profit_factor")
        profitable = (pf is not None and (pf == float("inf") or pf >= 1.0))
        row_cls = "alpha-pos" if profitable else ""
        rows_html.append(f"""
      <tr class="strat-row {row_cls}">
        <td class="name">
          <button class="exp-btn" id="expbtn-{row_key}"
                  onclick="svToggle('{spec_id_esc}','{gated_str}','{row_key}')"
                  title="Signal Validation — expand to inspect per-stock signals">&#9658;</button>
          <span class="sn">{name}</span><span class="sid">{spec_id_esc}</span>
        </td>
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
        <td>{_fmt(r.get('costs', {}).get('avg_round_trip_pct'),'%')}</td>
        <td class="gate">{gate}</td>
        <td class="hyp">{html.escape(hyp)}</td>
      </tr>
      <tr class="det-row" id="det-{row_key}" style="display:none">
        <td colspan="15" class="det-td">
          <div class="sv-panel">
            <div id="svhdr-{row_key}" class="sv-hdr"></div>
            <div id="svchips-{row_key}" class="sv-chips"></div>
            <div id="svnote-{row_key}" class="sv-note"></div>
            <div id="svchart-{row_key}" class="sv-chart"></div>
            <div id="svpnl-{row_key}" class="sv-pnl"></div>
            <div id="svtbl-{row_key}"></div>
          </div>
        </td>
      </tr>""")

    def _pf(r):
        v = r.get("metrics", {}).get("profit_factor") or 0
        return 999 if v == float("inf") else v
    n_alpha = sum(1 for r in records if _pf(r) >= 1.0)
    n_promote = sum(1 for r in records
                    if _pf(r) >= 1.3 and (r.get("metrics", {}).get("trades") or 0) >= 100)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Comparison — P&amp;L Backtest</title>
<style>
  :root {{ --bg:#0d1520; --card:#162032; --line:#1e2d40; --txt:#dce8f5; --mut:#6e849a;
           --good:#26d96e; --bad:#f04f5f; --ok:#f5a623; --accent:#3db8f5; }}
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
  th {{ background:#0d1826; color:var(--mut); font-weight:600; font-size:12px;
        text-transform:uppercase; letter-spacing:.03em; position:sticky; top:0; z-index:10; }}
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
  .muted {{ color:var(--mut); font-size:13px; padding:8px 2px; }} .muted.err {{ color:var(--bad); }}
  /* ── expand button ── */
  .exp-btn {{ background:none; border:none; cursor:pointer; color:var(--mut); font-size:12px;
              padding:0 5px 0 0; vertical-align:middle; line-height:1; transition:color .15s; }}
  .exp-btn:hover, .exp-btn.open {{ color:var(--accent); }}
  /* ── Signal Validation inline panel ── */
  tr.det-row > td.det-td {{ padding:0; border-bottom:2px solid var(--accent); }}
  .sv-panel {{ padding:14px 16px 16px; background:#0b1422; }}
  .sv-hdr {{ color:var(--mut); font-size:12px; margin-bottom:8px; }}
  .sv-chips {{ display:flex; flex-wrap:wrap; gap:5px; margin-bottom:10px;
               max-height:100px; overflow-y:auto; padding-bottom:2px; }}
  .sv-chip {{ cursor:pointer; padding:3px 9px; border-radius:12px; font-size:11px;
              font-family:monospace; white-space:nowrap; border:1px solid var(--line);
              background:var(--card); color:var(--mut); transition:all .1s; }}
  .sv-chip:hover {{ background:#253040; color:var(--txt); }}
  .sv-chip.sv-pos {{ color:#22c55e; border-color:rgba(34,197,94,.35); }}
  .sv-chip.sv-neg {{ color:#f43f5e; border-color:rgba(244,63,94,.3); }}
  .sv-chip.sv-sel {{ background:#253040; border-color:var(--accent); color:var(--accent) !important; }}
  .sv-note {{ color:var(--ok); font-size:11px; font-family:monospace; margin-bottom:6px; }}
  .sv-chart {{ width:100%; height:300px; margin-bottom:6px; }}
  .sv-pnl   {{ width:100%; height:110px; margin-bottom:10px; }}
  .sv-row-info {{ color:var(--mut); font-size:12px; margin-bottom:6px; }}
  .sv-tbl-wrap {{ overflow:auto; max-height:280px; border:1px solid var(--line); border-radius:6px; }}
  table.sv-cond-tbl {{ width:auto; border-collapse:collapse; font-size:12px; }}
  table.sv-cond-tbl th {{ background:#0f1419; color:var(--mut); padding:5px 10px;
                          position:sticky; top:0; z-index:1; border-bottom:1px solid var(--line);
                          white-space:nowrap; font-weight:500; text-transform:none; letter-spacing:0;
                          max-width:150px; overflow:hidden; text-overflow:ellipsis; }}
  table.sv-cond-tbl td {{ padding:4px 10px; border-bottom:1px solid #1e2735; white-space:nowrap; }}
  tr.sv-buy {{ background:rgba(34,197,94,.10); }}
  .sv-date {{ font-family:monospace; font-size:11px; color:var(--mut); }}
  .sv-cd {{ text-align:center; }}
  .sv-y {{ color:var(--good); font-weight:700; }}
  .sv-n {{ color:#2d3748; }}
  .sv-sig {{ font-size:11px; padding:1px 6px; border-radius:3px; font-weight:600; }}
  .sv-exec {{ background:#1d4ed8; color:#fff; }}
  .sv-skip {{ background:#374151; color:#9ca3af; }}
  .sv-exit {{ color:var(--mut); font-size:11px; }}
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
      <th class="name" title="Strategy display name and its spec id (data/strategies/&lt;id&gt;.json). Click ▶ to open the Signal Validation panel.">Strategy</th>
      <th title="Date the strategy spec was first created (provenance.created in the spec JSON).">Created</th>
      <th title="Number of completed round-trip trades pooled across all 1,792 universe symbols in the window (fixed ₹100k per trade, no concurrency cap).">Trades</th>
      <th title="Win rate: share of trades that closed with a positive net P&amp;L (after costs).">Win%</th>
      <th class="sep" title="Profit factor: gross winning P&amp;L ÷ gross losing P&amp;L, after costs. &gt; 1.0 = profitable; ≥ 1.3 with ≥ 100 trades = promotion-grade.">Profit Factor</th>
      <th title="Expectancy: average net % return per trade after I-Star costs. The per-trade edge in pure P&amp;L terms.">Expectancy</th>
      <th title="Total net P&amp;L in rupees: sum of every trade's profit/loss at ₹100k each across the universe (pooled, uncapped).">Total P&amp;L</th>
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
    <p>Click <b>&#9658;</b> next to a strategy name to open its <b>Signal Validation panel</b>.
    Select a stock to see its candlestick chart with entry/exit markers, indicator overlays, and a
    per-day conditions table showing which entry conditions fired on each signal date.
    Requires <code>scripts/server.py</code> running — open this report at
    <code>http://localhost:8765</code>, not as a local file.</p>
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
"""
    return page + _SCRIPT + "\n</body></html>"


# ── Client-side: Signal Validation with per-day conditions + chart ─────────────
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
  const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
  const LAYOUT = {
    layout: { background:{color:"#0f1c2e"}, textColor:"#6e849a" },
    grid:   { vertLines:{color:"#182436"}, horzLines:{color:"#182436"} },
    rightPriceScale:{ borderColor:"#1e2d40" },
    timeScale:{ borderColor:"#1e2d40" },
  };
  const PNL_LAYOUT = {
    layout: { background:{color:"#0b1422"}, textColor:"#6e849a" },
    grid:   { vertLines:{color:"#142030"}, horzLines:{color:"#142030"} },
    rightPriceScale:{ borderColor:"#1a2a3a", scaleMargins:{top:0.1,bottom:0.1} },
    timeScale:{ borderColor:"#1a2a3a", visible:true },
  };
  const PRICE_PAT = /^(ema|bb_up|bb_lower|hhv|llv|vwap)/i;
  const LINE_COLS = ['#3db8f5','#f5a623','#a78bfa','rgba(61,184,245,.5)',
                     'rgba(61,184,245,.35)','rgba(255,255,255,.2)','rgba(255,255,255,.14)',
                     '#26d96e','#fb923c'];

  const _svState    = {};   // row_key -> true once initialised
  const _svCharts   = {};   // row_key -> candlestick LWC chart
  const _svPnlCharts = {};  // row_key -> P&L area LWC chart

  // ── Toggle expand/collapse ──────────────────────────────────────────────────
  window.svToggle = function(specId, gated, key) {
    const row = document.getElementById('det-' + key);
    const btn = document.getElementById('expbtn-' + key);
    if (!row) return;
    const opening = (row.style.display === 'none' || !row.style.display);
    row.style.display = opening ? 'table-row' : 'none';
    if (btn) { btn.innerHTML = opening ? '&#9660;' : '&#9658;'; btn.classList.toggle('open', opening); }
    if (opening && !_svState[key]) { _svState[key] = true; _svInit(specId, gated, key); }
  };

  // ── First open: load stock list → chips ────────────────────────────────────
  async function _svInit(specId, gated, key) {
    const chipsDiv = document.getElementById('svchips-' + key);
    const hdrDiv   = document.getElementById('svhdr-'   + key);
    const chartDiv = document.getElementById('svchart-' + key);
    chipsDiv.innerHTML = '<span style="color:var(--mut);font-size:12px">Loading stocks…</span>';
    chartDiv.innerHTML = '';
    try {
      const res  = await fetch('/api/experiment/'+encodeURIComponent(specId)+'/symbols?gated='+gated);
      const data = await res.json();
      if (data.missing || !data.symbols || !data.symbols.length) {
        chipsDiv.innerHTML = '';
        chartDiv.innerHTML = '<div class="muted">No trades — run <code>scripts/run_experiments.py</code>.</div>';
        return;
      }
      hdrDiv.textContent = data.n_symbols + ' stocks traded · click a chip to inspect signals · sorted by net P&L';
      chipsDiv.innerHTML = data.symbols.map(function(s, i) {
        const cls  = s.total_pnl >= 0 ? 'sv-pos' : 'sv-neg';
        const pnl  = fmtINR(s.total_pnl);
        const wid  = ' title="'+s.trades+' trades · win '+s.win_rate+'%"';
        return '<button class="sv-chip '+cls+'" id="svchip-'+key+'-'+i+'"'
          + wid+' onclick="svChip(\''+esc(specId)+'\',\''+gated+'\',\''+key+'\','+i+',\''+esc(s.symbol)+'\')">'
          + esc(s.symbol)+' '+pnl+'</button>';
      }).join('');
      // Auto-select first chip
      _svMark(key, 0);
      _svLoad(specId, gated, key, data.symbols[0].symbol);
    } catch(e) {
      chipsDiv.innerHTML = '';
      chartDiv.innerHTML = '<div class="muted err">Server not reachable — open at <code>http://localhost:8765</code>.</div>';
    }
  }

  function _svMark(key, idx) {
    document.querySelectorAll('[id^="svchip-'+key+'-"]').forEach(b => b.classList.remove('sv-sel'));
    const c = document.getElementById('svchip-'+key+'-'+idx);
    if (c) c.classList.add('sv-sel');
  }

  window.svChip = function(specId, gated, key, idx, symbol) {
    _svMark(key, idx);
    _svLoad(specId, gated, key, symbol);
  };

  // ── Load signals for selected stock ────────────────────────────────────────
  function _svLoad(specId, gated, key, symbol) {
    const chartDiv = document.getElementById('svchart-' + key);
    const tblDiv   = document.getElementById('svtbl-'   + key);
    const noteEl   = document.getElementById('svnote-'  + key);
    if (!symbol) return;
    // Destroy old charts before clearing containers
    if (_svCharts[key])    { try { _svCharts[key].remove();    } catch(_) {} delete _svCharts[key]; }
    if (_svPnlCharts[key]) { try { _svPnlCharts[key].remove(); } catch(_) {} delete _svPnlCharts[key]; }
    chartDiv.innerHTML = '<div class="muted" style="padding:8px">Loading '+esc(symbol)+'…</div>';
    tblDiv.innerHTML   = '';
    noteEl.textContent = '';
    fetch('/api/experiment/'+encodeURIComponent(specId)+'/symbol/'+encodeURIComponent(symbol)+'/signals?gated='+gated)
      .then(function(r){ return r.json(); })
      .then(function(d) {
        if (d.error) { chartDiv.innerHTML = '<div class="muted err">'+esc(d.error)+'</div>'; return; }
        if (d.uses_rs_rank) noteEl.textContent = '⚠ rs_rank shown as 0.5 (cross-sectional — not recomputable per symbol)';
        chartDiv.innerHTML = '';
        var pnlDiv = document.getElementById('svpnl-'+key);
        if (pnlDiv) pnlDiv.innerHTML = '';
        _svRenderChart(chartDiv, d, key);
        if (pnlDiv) _svRenderPnl(pnlDiv, d, key);
        _svRenderTable(tblDiv, d);
      })
      .catch(function(e) {
        chartDiv.innerHTML = '<div class="muted err">Fetch failed: '+esc(String(e))+'</div>';
      });
  }

  // ── Candlestick + overlays + markers ───────────────────────────────────────
  function _svRenderChart(container, d, key) {
    const sd = d.signal_data.filter(function(x){ return x.open != null; });
    if (!sd.length) { container.innerHTML = '<div class="muted">No OHLCV data.</div>'; return; }

    // Create a fresh inner div — avoids LWC fighting with container styles
    const cdiv = document.createElement('div');
    cdiv.style.cssText = 'width:100%;height:320px;';
    container.appendChild(cdiv);

    // Measure after DOM insertion (getBoundingClientRect reflects actual layout)
    const w = Math.round(cdiv.getBoundingClientRect().width) || container.clientWidth || 900;
    var chart;
    try {
      chart = LightweightCharts.createChart(cdiv, Object.assign({ width: w, height: 320 }, LAYOUT));
    } catch(e) {
      container.innerHTML = '<div class="muted err">Chart error: '+esc(String(e))+'</div>';
      return;
    }
    _svCharts[key] = chart;

    const cs = chart.addCandlestickSeries({
      upColor:'#26d96e', downColor:'#f04f5f', borderVisible:false,
      wickUpColor:'#26d96e', wickDownColor:'#f04f5f' });
    cs.setData(sd.map(function(x){ return { time:x.time, open:x.open, high:x.high, low:x.low, close:x.close }; }));

    const vs = chart.addHistogramSeries({ priceFormat:{type:'volume'}, priceScaleId:'' });
    vs.priceScale().applyOptions({ scaleMargins:{ top:0.82, bottom:0 } });
    vs.setData(sd.map(function(x){
      return { time:x.time, value:x.volume||0,
               color: x.close >= x.open ? 'rgba(38,217,110,.22)' : 'rgba(240,79,95,.22)' };
    }));

    var ci = 0;
    d.indicator_cols.forEach(function(col) {
      if (!PRICE_PAT.test(col)) return;
      var pts = sd.filter(function(x){ return x[col] != null; }).map(function(x){ return { time:x.time, value:x[col] }; });
      if (!pts.length) return;
      var ls = chart.addLineSeries({ color: LINE_COLS[ci++ % LINE_COLS.length], lineWidth:1, lastValueVisible:false, priceLineVisible:false });
      ls.setData(pts);
    });

    var execEntries = new Set(d.trade_markers.map(function(t){ return t.entry_date; }));
    var markers = [];
    sd.forEach(function(x) {
      if (x.buy_signal && !execEntries.has(x.time))
        markers.push({ time:x.time, position:'belowBar', color:'#475569', shape:'circle', text:'' });
    });
    d.trade_markers.forEach(function(t) {
      if (t.entry_date) markers.push({ time:t.entry_date, position:'belowBar', color:'#3db8f5', shape:'arrowUp', text:'BUY' });
      if (t.exit_date)  markers.push({ time:t.exit_date,  position:'aboveBar',
        color: t.pnl_pct >= 0 ? '#26d96e' : '#f04f5f', shape:'arrowDown',
        text:  (t.pnl_pct >= 0 ? '+' : '') + t.pnl_pct.toFixed(1) + '%' });
    });
    markers.sort(function(a,b){ return a.time < b.time ? -1 : 1; });
    cs.setMarkers(markers);
    chart.timeScale().fitContent();
    new ResizeObserver(function(){ chart.applyOptions({ width: cdiv.getBoundingClientRect().width || cdiv.clientWidth }); }).observe(cdiv);
  }

  // ── Per-trade P&L bars (instantaneous change at each BUY exit) ─────────────
  function _svRenderPnl(container, d, key) {
    if (_svPnlCharts[key]) { try { _svPnlCharts[key].remove(); } catch(_) {} delete _svPnlCharts[key]; }
    var trades = d.trade_markers.slice().sort(function(a,b){ return a.exit_date < b.exit_date ? -1 : 1; });
    if (!trades.length) {
      container.innerHTML = '<div class="muted" style="font-size:12px;padding:4px 0">No trades for this stock.</div>';
      return;
    }

    var cdiv = document.createElement('div');
    cdiv.style.cssText = 'width:100%;height:110px;';
    container.appendChild(cdiv);
    var w = Math.round(cdiv.getBoundingClientRect().width) || container.clientWidth || 900;

    var chart;
    try {
      chart = LightweightCharts.createChart(cdiv, Object.assign({ width:w, height:110 }, PNL_LAYOUT));
    } catch(e) {
      container.innerHTML = '<div class="muted err">P&L chart error: '+esc(String(e))+'</div>';
      return;
    }
    _svPnlCharts[key] = chart;

    // One histogram bar per trade: positive bars above 0, negative bars below
    var hs = chart.addHistogramSeries({
      color: '#26d96e',
      base: 0,
      priceFormat: { type:'custom', formatter: function(p){ return (p>=0?'+':'')+p.toFixed(1)+'%'; } },
    });
    hs.setData(trades.map(function(t) {
      return {
        time:  t.exit_date,
        value: parseFloat(t.pnl_pct.toFixed(2)),
        color: t.pnl_pct >= 0 ? 'rgba(38,217,110,.80)' : 'rgba(240,79,95,.80)',
      };
    }));
    chart.timeScale().fitContent();
    new ResizeObserver(function(){ chart.applyOptions({ width: cdiv.getBoundingClientRect().width || cdiv.clientWidth }); }).observe(cdiv);
  }

  // ── Per-day conditions table ────────────────────────────────────────────────
  function _condLabel(expr) {
    return expr.replace(/@\w+/g,'').replace(/\s+/g,' ').trim().substring(0,30);
  }

  function _svRenderTable(container, d) {
    var sd     = d.signal_data;
    var defs   = d.cond_defs;
    var tDates = new Set(d.trade_markers.reduce(function(a,t){ a.push(t.entry_date, t.exit_date); return a; }, []));
    var eDates = new Set(d.trade_markers.map(function(t){ return t.entry_date; }));
    var visible = sd.filter(function(x){ return x.buy_signal || tDates.has(x.time); });
    var nBuy   = sd.filter(function(x){ return x.buy_signal; }).length;

    var h = '<div class="sv-row-info">'+nBuy+' buy signal'+(nBuy===1?'':'s')
          + ' &middot; '+d.trade_markers.length+' executed'
          + ' &mdash; showing signal &amp; trade rows only</div>';

    if (!visible.length) { container.innerHTML = h+'<div class="muted">No signals in window.</div>'; return; }

    h += '<div class="sv-tbl-wrap"><table class="sv-cond-tbl"><thead><tr><th>Date</th>'
       + defs.map(function(cd){ return '<th title="'+esc(cd.label)+'">'+esc(_condLabel(cd.label))+'</th>'; }).join('')
       + '<th>Signal</th></tr></thead><tbody>';

    visible.forEach(function(x) {
      var cls = x.buy_signal ? 'sv-buy' : '';
      h += '<tr class="'+cls+'"><td class="sv-date">'+x.time+'</td>';
      defs.forEach(function(cd) {
        h += '<td class="sv-cd">'+(x[cd.key]?'<span class="sv-y">✓</span>':'<span class="sv-n">✗</span>')+'</td>';
      });
      var sig = '';
      if (x.buy_signal && eDates.has(x.time))       sig = '<span class="sv-sig sv-exec">BUY ✓</span>';
      else if (x.buy_signal)                         sig = '<span class="sv-sig sv-skip">BUY ⊘</span>';
      else if (tDates.has(x.time) && !x.buy_signal)  sig = '<span class="sv-exit">EXIT</span>';
      h += '<td>'+sig+'</td></tr>';
    });
    h += '</tbody></table></div>';
    container.innerHTML = h;
  }

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
