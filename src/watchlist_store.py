"""
src.watchlist_store
-------------------
Local personal watchlists: multiple named lists + one shared quote cache.

    data/watchlist/watchlist.json   {"version": 2, "watchlists": [
                                        {"id", "name", "created", "items": [
                                            {symbol, name, note, target, stop, added, source,
                                             verified, holding?}]}]}
    data/watchlist/quotes.json      {SYMBOL: market data stamped with price_source/price_as_of}

A stock can sit in several lists; note/target/stop are per list (a stock can be a
"breakout" idea in one list and a long-term hold in another). Quotes are per symbol and
shared, so a refresh never touches your notes. Both files are personal (they contain your
holdings) and are git-ignored.

Quote sources
  * groww         -- written by scripts/build_watchlist.py from Groww MCP tool output
                     (Groww MCP is reachable only through Claude, not from Python).
  * stockanalysis -- fetched live by refresh_quotes(); used for stocks you add on the
                     dashboard. Price / PE / EPS / market cap / analyst target, but no day
                     change, so that cell stays blank until a Groww import.

Every network call degrades per symbol: a failure leaves the old quote in place.
A v1 file ({"items": [...]} with a per-item "group") is migrated on first load:
each group becomes a list.
"""
from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WL_DIR = ROOT / "data" / "watchlist"
WATCHLIST_PATH = WL_DIR / "watchlist.json"
QUOTES_PATH = WL_DIR / "quotes.json"
UNIVERSE_CSV = ROOT / "data" / "stocks.csv"

DEFAULT_LIST = "My watchlist"
HOLDINGS_ID, HOLDINGS_NAME = "groww-holdings", "Groww holdings"
SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&\-]{0,19}$")
EDITABLE = ("note", "target", "stop", "name")
MAX_NAME = 40

_lock = threading.RLock()


class WatchlistError(ValueError):
    """Bad input; message is safe to show in the UI."""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path: Path, data) -> None:
    """Atomic write: a crash mid-save can never leave a half-written watchlist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            # numpy scalars (from pandas) -> plain Python numbers
            json.dump(data, f, indent=1, ensure_ascii=False,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _slug(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "list"
    sid, n = base, 2
    while sid in taken:
        sid, n = f"{base}-{n}", n + 1
    return sid


def _migrate(d: dict) -> dict:
    """v1 {"items": [{..., "group"}]} -> v2 lists, one per group (order of first appearance)."""
    if "watchlists" in d:
        return d
    lists: dict[str, dict] = {}
    for it in d.get("items", []):
        g = it.pop("group", None) or DEFAULT_LIST
        if g not in lists:
            lid = HOLDINGS_ID if g == HOLDINGS_NAME else _slug(g, {l["id"] for l in lists.values()})
            lists[g] = {"id": lid, "name": g, "created": it.get("added") or _now(), "items": []}
        lists[g]["items"].append(it)
    return {"version": 2, "updated": d.get("updated"), "seeded_from": d.get("seeded_from"),
            "watchlists": list(lists.values())}


def load() -> dict:
    return _migrate(_read(WATCHLIST_PATH, {"version": 2, "watchlists": []}))


def save(wl: dict) -> None:
    wl["version"] = 2
    wl["updated"] = _now()
    _write(WATCHLIST_PATH, wl)


def load_quotes() -> dict:
    return _read(QUOTES_PATH, {})


def all_symbols(wl: dict | None = None) -> list[str]:
    wl = wl or load()
    return sorted({i["symbol"] for l in wl["watchlists"] for i in l["items"]})


_universe_cache: dict | None = None


def universe() -> dict[str, dict]:
    """NSE reference list from data/stocks.csv -> {SYMBOL: {name, sector, halal}}."""
    global _universe_cache
    if _universe_cache is None:
        out = {}
        try:
            with UNIVERSE_CSV.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    out[r["stock"].strip().upper()] = {
                        "name": (r.get("company_name") or "").strip(),
                        "sector": (r.get("sector") or "").strip(),
                        "halal": (r.get("halal") or "").strip().upper() == "Y",
                    }
        except FileNotFoundError:
            pass
        _universe_cache = out
    return _universe_cache


def _num(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise WatchlistError(f"not a number: {v!r}")


def _clean_name(name: str) -> str:
    n = " ".join((name or "").split())
    if not n:
        raise WatchlistError("list name can't be empty")
    if len(n) > MAX_NAME:
        raise WatchlistError(f"list name must be {MAX_NAME} characters or fewer")
    return n


def _find_list(wl: dict, list_id: str | None) -> dict:
    if not list_id:
        if not wl["watchlists"]:
            raise WatchlistError("create a watchlist first")
        return wl["watchlists"][0]
    for l in wl["watchlists"]:
        if l["id"] == list_id:
            return l
    raise WatchlistError(f"no watchlist with id '{list_id}'")


def _norm_sym(symbol: str) -> str:
    sym = (symbol or "").strip().upper().removesuffix(".NS")
    if not SYMBOL_RE.match(sym):
        raise WatchlistError(f"'{symbol}' doesn't look like an NSE symbol")
    return sym


# ── list management ─────────────────────────────────────────────────────────────

def create_list(name: str, list_id: str | None = None) -> dict:
    name = _clean_name(name)
    with _lock:
        wl = load()
        if any(l["name"].lower() == name.lower() for l in wl["watchlists"]):
            raise WatchlistError(f"a watchlist named '{name}' already exists")
        lst = {"id": list_id or _slug(name, {l["id"] for l in wl["watchlists"]}),
               "name": name, "created": _now(), "items": []}
        wl["watchlists"].append(lst)
        save(wl)
    return lst


def rename_list(list_id: str, name: str) -> dict:
    name = _clean_name(name)
    with _lock:
        wl = load()
        lst = _find_list(wl, list_id)
        if any(l["name"].lower() == name.lower() and l is not lst for l in wl["watchlists"]):
            raise WatchlistError(f"a watchlist named '{name}' already exists")
        lst["name"] = name
        save(wl)
    return lst


def delete_list(list_id: str) -> int:
    """Delete a list and its entries; returns how many stocks it held."""
    with _lock:
        wl = load()
        lst = _find_list(wl, list_id)
        wl["watchlists"] = [l for l in wl["watchlists"] if l is not lst]
        save(wl)
    return len(lst["items"])


def reorder_lists(ids: list[str]) -> None:
    with _lock:
        wl = load()
        pos = {i: n for n, i in enumerate(ids)}
        wl["watchlists"].sort(key=lambda l: pos.get(l["id"], len(pos)))
        save(wl)


# ── stock entries ───────────────────────────────────────────────────────────────

def add(symbol: str, list_id: str | None = None, note: str = "", target=None, stop=None,
        name: str | None = None, force: bool = False, fetch_quote: bool = True) -> dict:
    sym = _norm_sym(symbol)
    with _lock:
        lst = _find_list(load(), list_id)
        if any(i["symbol"] == sym for i in lst["items"]):
            raise WatchlistError(f"{sym} is already in '{lst['name']}'")

    meta = universe().get(sym)
    have_quote = sym in load_quotes()
    quote = refresh_quotes([sym]).get(sym) if fetch_quote else None
    if not meta and not quote and not have_quote and not force:
        raise WatchlistError(f"{sym} is not in data/stocks.csv and no quote was found - "
                             "check the symbol, or tick 'add anyway'")
    item = {
        "symbol": sym,
        "name": (name or (meta or {}).get("name") or _known_name(sym) or sym).strip(),
        "note": (note or "").strip(),
        "target": _num(target),
        "stop": _num(stop),
        "added": _now(),
        "source": "manual",
        "verified": bool(meta or quote or have_quote),
    }
    with _lock:                       # re-load: the quote fetch ran outside the lock
        wl = load()
        lst = _find_list(wl, list_id or lst["id"])
        if any(i["symbol"] == sym for i in lst["items"]):
            raise WatchlistError(f"{sym} is already in '{lst['name']}'")
        lst["items"].append(item)
        save(wl)
    return item


def _known_name(sym: str) -> str | None:
    """Name from another list's entry (e.g. a holding Groww titled) when the CSV lacks it."""
    for l in load()["watchlists"]:
        for i in l["items"]:
            if i["symbol"] == sym and i.get("name") and i["name"] != sym:
                return i["name"]
    return None


def symbol_in_any_list(sym: str) -> bool:
    return any(i["symbol"] == sym for l in load()["watchlists"] for i in l["items"])


def remove(list_id: str, symbol: str) -> bool:
    sym = (symbol or "").strip().upper()
    with _lock:
        wl = load()
        lst = _find_list(wl, list_id)
        before = len(lst["items"])
        lst["items"] = [i for i in lst["items"] if i["symbol"] != sym]
        if len(lst["items"]) == before:
            return False
        save(wl)
    return True


def update(list_id: str, symbol: str, fields: dict) -> dict:
    sym = (symbol or "").strip().upper()
    with _lock:
        wl = load()
        lst = _find_list(wl, list_id)
        item = next((i for i in lst["items"] if i["symbol"] == sym), None)
        if not item:
            raise WatchlistError(f"{sym} is not in '{lst['name']}'")
        for k in EDITABLE:
            if k in fields:
                v = fields[k]
                item[k] = _num(v) if k in ("target", "stop") else str(v or "").strip()
        save(wl)
    return item


def move(symbol: str, from_id: str, to_id: str, copy: bool = False) -> dict:
    """Move (or copy) an entry, keeping its note/target/stop/holding."""
    sym = (symbol or "").strip().upper()
    if from_id == to_id:
        raise WatchlistError("pick a different watchlist")
    with _lock:
        wl = load()
        src, dst = _find_list(wl, from_id), _find_list(wl, to_id)
        item = next((i for i in src["items"] if i["symbol"] == sym), None)
        if not item:
            raise WatchlistError(f"{sym} is not in '{src['name']}'")
        if any(i["symbol"] == sym for i in dst["items"]):
            raise WatchlistError(f"{sym} is already in '{dst['name']}'")
        new = dict(item)
        if copy:
            new["added"] = _now()
            new.pop("holding", None)      # a position lives in one place only
        else:
            src["items"] = [i for i in src["items"] if i["symbol"] != sym]
        dst["items"].append(new)
        save(wl)
    return new


# ── quotes ──────────────────────────────────────────────────────────────────────

def merge_quotes(new: dict[str, dict]) -> None:
    """Field-level merge so a stockanalysis refresh keeps Groww-only fields (P/B, ROE...)."""
    with _lock:
        q = load_quotes()
        for sym, rec in new.items():
            q[sym] = {**q.get(sym, {}), **{k: v for k, v in rec.items() if v is not None}}
        _write(QUOTES_PATH, q)


def refresh_quotes(symbols: list[str] | None = None, workers: int = 8) -> dict[str, dict]:
    """Pull price/PE/EPS/mcap/target from stockanalysis.com for `symbols` (default: all)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import requests

    from src.predictor.adapters.stockanalysis_adapter import fetch_kpis

    symbols = all_symbols() if symbols is None else symbols
    if not symbols:
        return {}
    local = threading.local()

    def one(sym):
        if not hasattr(local, "s"):
            local.s = requests.Session()
        try:
            k = fetch_kpis(sym, session=local.s, with_margins=False)
        except Exception:
            k = {}
        if not k.get("current_price"):
            return sym, None
        mc = k.get("market_cap")
        return sym, {
            "price": k["current_price"],
            "pe": k.get("trailing_pe"),
            "eps_ttm": k.get("trailing_eps"),
            "market_cap_cr": round(mc / 1e7) if mc else None,
            "analyst_target": k.get("target_mean"),
            "analyst_rating": k.get("recommendation"),
            "n_analysts": k.get("n_analysts"),
            "earnings_date": k.get("earnings_date"),
            "price_source": "stockanalysis",
            "price_as_of": _now(),
        }

    got = {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(symbols)))) as ex:
        for fut in as_completed([ex.submit(one, s) for s in symbols]):
            sym, rec = fut.result()
            if rec:
                got[sym] = rec
    if got:
        merge_quotes(got)
    return got


# ── view model ──────────────────────────────────────────────────────────────────

def payload() -> dict:
    wl, quotes, uni = load(), load_quotes(), universe()
    lists = []
    for l in wl["watchlists"]:
        items = []
        for it in l["items"]:
            meta = uni.get(it["symbol"], {})
            items.append({**it, "sector": meta.get("sector") or "", "halal": meta.get("halal")})
        lists.append({"id": l["id"], "name": l["name"], "created": l.get("created"), "items": items})
    syms = {i["symbol"] for l in lists for i in l["items"]}
    from src import watchlist_enrich as we
    enrich = we.load()
    charts = {s: c for s in syms if (c := we.chart_closes(we.load_chart(s)))}
    bench = we.chart_closes(we.load_chart(we.BENCH_SYM))
    return {
        "generated": _now(),
        "updated": wl.get("updated"),
        "seeded_from": wl.get("seeded_from"),
        "watchlists": lists,
        "quotes": {s: quotes[s] for s in syms if s in quotes},
        "enrich": {s: enrich[s] for s in syms if s in enrich},
        "charts": charts,
        "bench": {"symbol": we.BENCH_SYM, "name": we.BENCH_NAME, "chart": bench} if bench else None,
    }


def names_map() -> dict[str, str]:
    """{SYMBOL: company name} across all lists, for news queries."""
    out = {}
    for l in load()["watchlists"]:
        for i in l["items"]:
            out.setdefault(i["symbol"], i.get("name") or i["symbol"])
    return out


def universe_list() -> list[list[str]]:
    """Compact [symbol, name] pairs for the add-stock autocomplete."""
    return [[s, m["name"]] for s, m in sorted(universe().items())]
