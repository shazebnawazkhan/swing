"""Personal watchlists: seed from Groww, import Groww data, enrich, manage lists, render watchlist.html.

The editable copy is served by ``scripts/server.py`` at http://localhost:8765/watchlist --
list/stock edits there write ``data/watchlist/watchlist.json``; "Refresh news & data"
re-runs the enrichment in the background. Opening ``outputs/watchlist.html`` straight
from disk gives a read-only snapshot.

Usage::

    # seed the "Groww holdings" list from a saved get_equity_portfolio_holdings response
    python scripts/build_watchlist.py seed-holdings data/watchlist/groww_holdings_2026-11-22.json

    # Groww MCP tool output saved by Claude: prices/fundamentals, shareholding, quarterly results
    python scripts/build_watchlist.py import-groww --ltp data/watchlist/groww_ltp_2026-11-22.json \
        --fund data/watchlist/groww_fund_2026-11-22.json \
        --shareholding data/watchlist/groww_shareholding_2026-11-22.json \
        --financials data/watchlist/groww_financials_2026-11-22.json

    # news, target-price articles, triggers, delivery %, OI, charts (+ NSE shareholding fallback)
    python scripts/build_watchlist.py enrich                 # every symbol in every list
    python scripts/build_watchlist.py enrich E2E --parts news chart

    python scripts/build_watchlist.py refresh                # prices from stockanalysis.com
    python scripts/build_watchlist.py lists | create-list NAME | rename-list ID NAME | delete-list ID
    python scripts/build_watchlist.py add SYMBOL --list ID [--target N --stop N --note TEXT]
    python scripts/build_watchlist.py remove SYMBOL --list ID
    python scripts/build_watchlist.py build                  # static outputs/watchlist.html
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import watchlist_enrich as we  # noqa: E402
from src import watchlist_store as ws  # noqa: E402

OUT_HTML = ROOT / "outputs" / "watchlist.html"


# ── Groww imports ───────────────────────────────────────────────────────────────
def _groww_ts(raw: dict) -> str:
    """'2026-11-22 Sunday 10:09:49 PM IST' -> ISO with +05:30."""
    ts = raw.get("result_timestamp")
    try:
        dt = datetime.strptime(ts.replace(" IST", ""), "%Y-%m-%d %A %I:%M:%S %p")
        return dt.isoformat() + "+05:30"
    except (AttributeError, ValueError):
        return ws._now()


def cmd_seed_holdings(args) -> None:
    from build_analyst_dashboard import _load_tool_json

    raw = _load_tool_json(Path(args.file))
    holdings = raw.get("result", raw).get("holdings", [])
    ts = _groww_ts(raw)
    with ws._lock:
        wl = ws.load()
        lst = next((l for l in wl["watchlists"] if l["id"] == ws.HOLDINGS_ID), None)
        if lst is None:
            lst = {"id": ws.HOLDINGS_ID, "name": ws.HOLDINGS_NAME, "created": ts, "items": []}
            wl["watchlists"].insert(0, lst)
        have = {i["symbol"]: i for i in lst["items"]}
        added = refreshed = 0
        for h in holdings:
            sym = h["trading_symbol"].upper()
            holding = {"qty": h.get("quantity"), "avg_price": h.get("average_price"),
                       "isin": h.get("symbol_isin"), "as_of": ts}
            if sym in have:                    # re-seed: refresh the position, keep notes/targets
                have[sym]["holding"] = holding
                refreshed += 1
                continue
            meta = ws.universe().get(sym, {})
            lst["items"].append({
                "symbol": sym, "name": h.get("title") or meta.get("name") or sym,
                "note": "", "target": None, "stop": None, "added": ts,
                "source": "groww-holdings", "verified": True, "holding": holding,
            })
            added += 1
        held = {h["trading_symbol"].upper() for h in holdings}
        for i in lst["items"]:                 # sold since last seed: keep the row, drop the position
            if i.get("holding") and i["symbol"] not in held:
                i.pop("holding")
                i["note"] = (i.get("note") or "") + f" [no longer held as of {ts[:10]}]"
        wl["seeded_from"] = {"source": "Groww MCP get_equity_portfolio_holdings", "as_of": ts}
        ws.save(wl)
    print(f"'{ws.HOLDINGS_NAME}': {added} added, {refreshed} updated -> {ws.WATCHLIST_PATH}")


def cmd_import_groww(args) -> None:
    from build_analyst_dashboard import _load_tool_json, parse_groww_fundamentals, parse_groww_ltp

    wanted = ws.all_symbols()
    if args.ltp:
        raw_ltp = _load_tool_json(Path(args.ltp))
        ltp = parse_groww_ltp(raw_ltp)
        fund: dict = {}
        for f in args.fund or []:
            fund.update(parse_groww_fundamentals(_load_tool_json(Path(f))))
        ts = _groww_ts(raw_ltp)
        out = {}
        for sym in wanted:
            row = ltp.get(sym)
            if not row:
                print(f"  ! {sym}: not in the LTP response")
                continue
            f = fund.get(row.get("bse_code") or "", {})
            out[sym] = {
                "bse_code": row.get("bse_code"),
                "price": row["ltp"], "prev_close": row.get("prev_close"),
                "day_change_pct": row.get("day_change_pct"),
                "day_high": row.get("high"), "day_low": row.get("low"), "volume": row.get("volume"),
                # Groww reports PE/EPS 0.00 for some fresh listings: that means "not available"
                "pe": f.get("pe") or None, "industry_pe": f.get("industry_pe"), "pb": f.get("pb"),
                "roe": f.get("roe"), "eps_ttm": f.get("eps_ttm") or None, "div_yield": f.get("div_yield"),
                "debt_to_equity": f.get("debt_to_equity"), "book_value": f.get("book_value"),
                "market_cap_cr": f.get("market_cap_cr"),
                "price_source": "groww", "price_as_of": ts,
            }
        ws.merge_quotes(out)
        print(f"groww quotes: {len(out)}/{len(wanted)} symbols -> {ws.QUOTES_PATH}")

    # shareholding / quarterly responses are keyed by Groww contract id (= BSE code)
    by_bse = {q.get("bse_code"): s for s, q in ws.load_quotes().items() if q.get("bse_code")}
    for kind, files, parse, section in (("shareholding", args.shareholding, we.parse_groww_shareholding, "shareholding"),
                                        ("quarterly", args.financials, we.parse_groww_quarterly, "quarterly")):
        if not files:
            continue
        sec = {}
        for f in files:
            raw = _load_tool_json(Path(f))
            for bse, rec in parse(raw.get("result", raw).get("content", {}), _groww_ts(raw)).items():
                sym = by_bse.get(bse)
                if sym in wanted:
                    sec[sym] = {section: rec}
                elif sym is None:
                    print(f"  ! {kind}: BSE {bse} not matched to a watchlist symbol (import --ltp first)")
        we.merge(sec)
        print(f"groww {kind}: {len(sec)} symbols -> {we.ENRICH_PATH}")


def cmd_enrich(args) -> None:
    syms = [s.upper() for s in args.symbols] or ws.all_symbols()
    we.enrich(syms, ws.names_map(), tuple(args.parts) if args.parts else we.ALL_PARTS)


def cmd_refresh(args) -> None:
    syms = args.symbols or ws.all_symbols()
    got = ws.refresh_quotes(syms)
    print(f"stockanalysis refresh: {len(got)}/{len(syms)} symbols updated")


# ── list + entry commands ───────────────────────────────────────────────────────
def _run(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except ws.WatchlistError as e:
        raise SystemExit(f"error: {e}")


def cmd_lists(args) -> None:
    for l in ws.load()["watchlists"]:
        print(f"  {l['id']:<22} {l['name']:<28} {len(l['items'])} stocks")


def cmd_create_list(args) -> None:
    l = _run(ws.create_list, args.name)
    print(f"created '{l['name']}' (id {l['id']})")


def cmd_rename_list(args) -> None:
    l = _run(ws.rename_list, args.id, args.name)
    print(f"renamed {l['id']} -> '{l['name']}'")


def cmd_delete_list(args) -> None:
    n = _run(ws.delete_list, args.id)
    print(f"deleted {args.id} ({n} stocks)")


def cmd_add(args) -> None:
    it = _run(ws.add, args.symbol, list_id=args.list, note=args.note or "",
              target=args.target, stop=args.stop, force=args.force)
    print(f"added {it['symbol']} ({it['name']}); run `enrich {it['symbol']}` for news/charts/delivery")


def cmd_remove(args) -> None:
    ok = _run(ws.remove, args.list, args.symbol)
    print("removed" if ok else f"{args.symbol} was not in that list")


# ── render ──────────────────────────────────────────────────────────────────────
TEMPLATE_PATH = Path(__file__).with_name("watchlist_template.html")


def build_html(payload: dict | None = None, with_ohlc: bool = False) -> str:
    """Render the dashboard. The server leaves OHLCV out (the explorer fetches it from
    /api/watchlist/ohlc); a static snapshot embeds it so the explorer works from disk."""
    data = {"wl": payload or ws.payload(), "universe": ws.universe_list()}
    if with_ohlc:
        syms = ws.all_symbols() + [we.BENCH_SYM]
        data["ohlc"] = {s: raw for s in syms if (raw := we.load_chart(s)) and raw.get("v") == 2}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE_PATH.read_text(encoding="utf-8").replace("/*DATA*/null", blob)


def cmd_build(args) -> None:
    out = Path(args.out) if args.out else OUT_HTML
    out.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(with_ohlc=True)
    out.write_text(html, encoding="utf-8")
    wl = ws.load()
    print(f"built {len(wl['watchlists'])} lists / {len(ws.all_symbols(wl))} symbols -> {out} ({len(html)//1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("seed-holdings"); p.add_argument("file"); p.set_defaults(fn=cmd_seed_holdings)
    p = sub.add_parser("import-groww"); p.add_argument("--ltp"); p.add_argument("--fund", nargs="*")
    p.add_argument("--shareholding", nargs="*"); p.add_argument("--financials", nargs="*")
    p.set_defaults(fn=cmd_import_groww)
    p = sub.add_parser("enrich"); p.add_argument("symbols", nargs="*")
    p.add_argument("--parts", nargs="*", choices=we.ALL_PARTS); p.set_defaults(fn=cmd_enrich)
    p = sub.add_parser("refresh"); p.add_argument("symbols", nargs="*"); p.set_defaults(fn=cmd_refresh)
    p = sub.add_parser("lists"); p.set_defaults(fn=cmd_lists)
    p = sub.add_parser("create-list"); p.add_argument("name"); p.set_defaults(fn=cmd_create_list)
    p = sub.add_parser("rename-list"); p.add_argument("id"); p.add_argument("name"); p.set_defaults(fn=cmd_rename_list)
    p = sub.add_parser("delete-list"); p.add_argument("id"); p.set_defaults(fn=cmd_delete_list)
    p = sub.add_parser("add"); p.add_argument("symbol"); p.add_argument("--list")
    p.add_argument("--note"); p.add_argument("--target", type=float); p.add_argument("--stop", type=float)
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_add)
    p = sub.add_parser("remove"); p.add_argument("symbol"); p.add_argument("--list", required=True)
    p.set_defaults(fn=cmd_remove)
    p = sub.add_parser("build"); p.add_argument("--out"); p.set_defaults(fn=cmd_build)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
