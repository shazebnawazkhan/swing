"""
fetch_symbols.py
----------------
Downloads the complete NSE equity symbol list with enriched metadata
and writes stocks.csv.

stocks.csv columns:
  stock        – NSE ticker symbol (e.g. TATASTEEL)
  company_name – Full company name from NSE listing
  sector       – Broad sector (NSE classification, Yahoo Finance fallback)
  sub_sector   – Industry sub-group (Yahoo Finance)
  enabled      – Y / N  (all N by default; edit manually to enable)
  halal        – Y / N  (mirrors 'enabled' by default)

Data sources (in order of priority):
  1. nsearchives.nseindia.com/content/equities/EQUITY_L.csv
     → symbol + company name for all listed equities
  2. nsearchives.nseindia.com/content/indices/ind_*.csv
     → NSE sector for the ~755 Nifty Total Market stocks
  3. Yahoo Finance (yfinance, threaded)
     → sector fallback + sub_sector for all stocks

Usage:
    python fetch_symbols.py
    python fetch_symbols.py --output my_stocks.csv
    python fetch_symbols.py --skip-yf          # fast mode, no Yahoo Finance
"""

import argparse
import sys

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

# ── Constants ─────────────────────────────────────────────────────────────────

_EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
_INDEX_BASE_URL  = "https://nsearchives.nseindia.com/content/indices/"

_INDEX_FILES = [
    "ind_nifty500list.csv",
    "ind_niftytotalmarket_list.csv",
    "ind_niftymidcap150list.csv",
    "ind_niftysmallcap250list.csv",
    "ind_niftylargemidcap250list.csv",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}

_YF_WORKERS = 15   # concurrent yfinance threads


# ── Step 1: NSE equity master → symbol + company name ────────────────────────

def fetch_equity_list() -> pd.DataFrame:
    """Download EQUITY_L.csv; return DataFrame with columns [symbol, company_name]."""
    print(f"[1/3] Fetching equity list from NSE…")
    resp = requests.get(_EQUITY_LIST_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()

    text = resp.content.decode("windows-1252", errors="replace")
    df   = pd.read_csv(StringIO(text))

    sym_col  = next(c for c in df.columns if c.strip().upper() == "SYMBOL")
    name_col = next(c for c in df.columns if "NAME" in c.upper())

    out = pd.DataFrame({
        "symbol":       df[sym_col].str.strip(),
        "company_name": df[name_col].str.strip(),
    }).dropna(subset=["symbol"])

    out = out.sort_values("symbol").drop_duplicates("symbol").reset_index(drop=True)
    print(f"     {len(out):,} symbols found")
    return out


# ── Step 2: NSE index CSVs → sector map ──────────────────────────────────────

def fetch_nse_sectors() -> dict[str, str]:
    """
    Download Nifty index CSV files and build symbol → sector mapping.
    Returns {symbol: nse_industry_string}.
    """
    print(f"[2/3] Fetching NSE sector classifications from index files…")
    sector_map: dict[str, str] = {}
    session = requests.Session()

    for fname in _INDEX_FILES:
        url  = _INDEX_BASE_URL + fname
        resp = session.get(url, headers=_HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"       skip {fname} (HTTP {resp.status_code})")
            continue
        try:
            df = pd.read_csv(StringIO(resp.text))
            if "Symbol" not in df.columns or "Industry" not in df.columns:
                continue
            added = 0
            for _, row in df.iterrows():
                sym = str(row["Symbol"]).strip()
                ind = str(row["Industry"]).strip()
                if sym and ind and sym not in sector_map:
                    sector_map[sym] = ind
                    added += 1
            print(f"       {fname}: +{added} symbols")
        except Exception as exc:
            print(f"       parse error {fname}: {exc}")

    print(f"     {len(sector_map):,} symbols mapped to a sector")
    return sector_map


# ── Step 3: Yahoo Finance → sector + sub_sector ───────────────────────────────

def _yf_lookup(symbol: str) -> tuple[str, str]:
    """Fetch (sector, sub_sector) for one symbol from Yahoo Finance."""
    try:
        info = yf.Ticker(f"{symbol}.NS").info
        return (
            str(info.get("sector",   "") or ""),
            str(info.get("industry", "") or ""),
        )
    except Exception:
        return "", ""


def fetch_yf_sectors(symbols: list[str], nse_sector_map: dict[str, str]) -> dict[str, tuple[str, str]]:
    """
    Parallel yfinance lookups.
    Returns {symbol: (sector, sub_sector)}.
    Only calls Yahoo Finance for stocks missing from nse_sector_map.
    """
    missing   = [s for s in symbols if s not in nse_sector_map]
    has_nse   = [s for s in symbols if s in nse_sector_map]
    print(f"[3/3] Yahoo Finance enrichment: {len(missing):,} stocks need sector, "
          f"{len(symbols):,} need sub_sector  ({_YF_WORKERS} workers)…")

    results: dict[str, tuple[str, str]] = {}
    total   = len(symbols)
    done    = 0

    with ThreadPoolExecutor(max_workers=_YF_WORKERS) as pool:
        future_map = {pool.submit(_yf_lookup, sym): sym for sym in symbols}
        for future in as_completed(future_map):
            sym            = future_map[future]
            yf_sec, yf_sub = future.result()

            # Sector: prefer NSE classification, fall back to Yahoo Finance
            sector = nse_sector_map.get(sym) or yf_sec
            results[sym] = (sector, yf_sub)

            done += 1
            if done % 100 == 0 or done == total:
                print(f"     {done:,}/{total:,} …", end="\r", flush=True)

    print()
    filled_sector  = sum(1 for s, _ in results.values() if s)
    filled_sub     = sum(1 for _, ss in results.values() if ss)
    print(f"     Sector filled: {filled_sector:,}/{total:,}   "
          f"Sub-sector filled: {filled_sub:,}/{total:,}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch all NSE equity symbols with metadata.")
    parser.add_argument("--output",    default="data/stocks.csv", help="Output CSV (default: data/stocks.csv)")
    parser.add_argument("--skip-yf",   action="store_true",  help="Skip Yahoo Finance; use NSE data only")
    args = parser.parse_args()

    # Step 1 – equity list
    equity_df = fetch_equity_list()
    symbols   = equity_df["symbol"].tolist()

    # Step 2 – NSE sector map
    nse_sectors = fetch_nse_sectors()

    # Step 3 – Yahoo Finance (optional)
    if args.skip_yf:
        print("[3/3] Skipping Yahoo Finance (--skip-yf)")
        enriched = {sym: (nse_sectors.get(sym, ""), "") for sym in symbols}
    else:
        enriched = fetch_yf_sectors(symbols, nse_sectors)

    # Assemble final DataFrame
    rows = []
    for _, row in equity_df.iterrows():
        sym              = row["symbol"]
        sector, sub_sec  = enriched.get(sym, (nse_sectors.get(sym, ""), ""))
        rows.append({
            "stock":        sym,
            "company_name": row["company_name"],
            "sector":       sector,
            "sub_sector":   sub_sec,
            "enabled":      "N",
            "halal":        "N",
        })

    out = pd.DataFrame(rows, columns=["stock", "company_name", "sector", "sub_sector", "enabled", "halal"])
    out["sector"]     = out["sector"].fillna("")
    out["sub_sector"] = out["sub_sector"].fillna("")
    out.to_csv(args.output, index=False)
    print(f"\nWritten {len(out):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
