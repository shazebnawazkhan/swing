"""
fetch_shariah.py
----------------
Fetches Sharia compliance status for NSE-listed stocks from multiple sources
and writes data/shariah_status.csv.

Sources
  1. NSE Shariah indices  – Live constituent lists for three Nifty Shariah indices
     fetched via NSE's equity-stock-indices API (free, session-cookie only).
     Covers: NIFTY50 SHARIAH, NIFTY500 SHARIAH, NIFTY SHARIAH 25 (~199 stocks).

  2. Sector filter  – Deterministic exclusion heuristic based on AAOIFI sector
     criteria. Reads sector data from stocks.csv (populated by fetch_symbols.py).
     Excluded sectors: Financial Services, Gambling, Tobacco, Defence, Alcohol.
     Returns 0 for clearly excluded, 1 for clean sectors, NaN if sector unknown.

  Notes on sources NOT yet implemented:
    • Musaffa (musaffa.com) – requires authenticated API; their compliance data
      is loaded client-side. Sign up at musaffa.com to get an API token, then
      add it to the fetch_musaffa_status() stub below to activate this source.
    • TASIS India (tasisindia.com) – domain not currently resolvable; publishes
      quarterly compliance PDFs. Could be parsed if the domain becomes reachable.

Output columns in data/shariah_status.csv:
  stock           – NSE ticker symbol
  nse_index       – 1 if in any NSE Nifty Shariah index, else 0
  sector_filter   – 1 if sector is permissible, 0 if excluded, NaN if unknown
  shariah_score   – count of sources that call the stock compliant (max 2)

Usage:
    python scripts/fetch_shariah.py
    python scripts/fetch_shariah.py --symbols RELIANCE,INFY,TCS
    python scripts/fetch_shariah.py --output data/shariah_status.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Constants ──────────────────────────────────────────────────────────────────

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}

_NSE_HOME             = "https://www.nseindia.com/"
_NSE_INDEX_API        = "https://www.nseindia.com/api/equity-stock-indices"

# All three NSE Nifty Shariah indices (confirmed via allIndices API).
# First entry in each API response is the index summary row — skipped.
_NSE_SHARIAH_INDICES = [
    "NIFTY50 SHARIAH",
    "NIFTY500 SHARIAH",
    "NIFTY SHARIAH 25",
]

# Sectors considered non-compliant by AAOIFI / TASIS standards.
# Stocks whose sector matches these are excluded (sector_filter = 0).
# Every other known sector gets sector_filter = 1 (clean sector, passes filter).
# Unknown sector → NaN (no signal either way).
_EXCLUDED_SECTORS = {
    "Financial Services",       # conventional banks, NBFCs, brokerage, insurance
}
_EXCLUDED_SUB_SECTORS = {
    "Banks",
    "Insurance",
    "Insurance—Life",
    "Insurance—Diversified",
    "Insurance—Property & Casualty",
    "Credit Services",
    "Mortgage Finance",
    "Capital Markets",
    "Financial Data & Stock Exchanges",
    "Diversified Financials",
    "Asset Management",
    "Gambling",
    "Resorts & Casinos",
    "Tobacco",
    "Alcohol",
    "Alcoholic Beverages",
    "Defense",
    "Aerospace & Defense",
    "Aerospace & Defence",
    "Adult Entertainment",
}


# ── Source 1: NSE Shariah index constituents ───────────────────────────────────

def fetch_nse_shariah_constituents() -> set[str]:
    """
    Fetch live constituent lists for all NSE Nifty Shariah indices.
    Uses the equity-stock-indices API (needs a warm-up GET to nseindia.com).
    Returns set of NSE ticker symbols present in any Shariah index.
    """
    print("[1/2] Fetching NSE Shariah index constituents via API…")
    compliant: set[str] = set()
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)

    try:
        session.get(_NSE_HOME, timeout=15)
    except Exception as exc:
        print(f"       NSE warm-up failed: {exc}")

    for index_name in _NSE_SHARIAH_INDICES:
        try:
            resp = session.get(_NSE_INDEX_API, params={"index": index_name}, timeout=15)
            if resp.status_code != 200:
                print(f"       {index_name!r}: HTTP {resp.status_code} – skipped")
                continue
            data = resp.json().get("data", [])
            # data[0] is always the index summary row (not a stock)
            stocks = [
                row["symbol"]
                for row in data[1:]
                if row.get("symbol") and row.get("series", "EQ") == "EQ"
            ]
            print(f"       {index_name}: {len(stocks)} symbols")
            compliant.update(stocks)
        except Exception as exc:
            print(f"       {index_name!r}: error – {exc}")

    print(f"     Total in NSE Shariah indices: {len(compliant)}")
    return compliant


# ── Source 2: Sector exclusion filter ─────────────────────────────────────────

def compute_sector_filter(
    symbols: list[str],
    stocks_df: pd.DataFrame | None,
) -> dict[str, int | None]:
    """
    Apply AAOIFI-aligned sector exclusion heuristic.
    Returns {symbol: 1/0/None}.
      1   – sector is clean (passes filter)
      0   – sector is on the exclusion list (fails filter)
      None – sector unknown in stocks.csv
    """
    print("[2/2] Applying sector exclusion filter…")
    result: dict[str, int | None] = {}

    if stocks_df is None:
        print("       No stocks.csv data available; all sector_filter = NaN")
        return {s: None for s in symbols}

    sector_map  = stocks_df.set_index("stock")["sector"].to_dict()  if "sector"     in stocks_df.columns else {}
    sub_sec_map = stocks_df.set_index("stock")["sub_sector"].to_dict() if "sub_sector" in stocks_df.columns else {}

    excluded = 0
    clean    = 0
    unknown  = 0
    for sym in symbols:
        sector  = str(sector_map.get(sym, "")  or "").strip()
        sub_sec = str(sub_sec_map.get(sym, "") or "").strip()

        if not sector and not sub_sec:
            result[sym] = None
            unknown += 1
        elif sector in _EXCLUDED_SECTORS or sub_sec in _EXCLUDED_SUB_SECTORS:
            result[sym] = 0
            excluded += 1
        else:
            result[sym] = 1
            clean += 1

    print(f"     clean: {clean}  excluded: {excluded}  unknown: {unknown}")
    return result


# ── Assembler ──────────────────────────────────────────────────────────────────

def build_shariah_csv(
    symbols: list[str],
    nse_compliant: set[str],
    sector_results: dict[str, int | None],
    output_path: str,
) -> None:
    rows = []
    for sym in symbols:
        nse    = 1 if sym in nse_compliant else 0
        sector = sector_results.get(sym)   # 1 / 0 / None

        score = nse
        if sector == 1:
            score += 1

        rows.append({
            "stock":         sym,
            "nse_index":     nse,
            "sector_filter": sector,
            "shariah_score": score,
        })

    df = pd.DataFrame(rows, columns=["stock", "nse_index", "sector_filter", "shariah_score"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    total       = len(df)
    any_src     = (df["shariah_score"] > 0).sum()
    both_src    = (df["shariah_score"] == 2).sum()
    nse_only    = ((df["nse_index"] == 1) & (df["sector_filter"].isna() | (df["sector_filter"] == 0))).sum()
    print(f"\nWritten {total:,} rows -> {output_path}")
    print(f"  Score 2 (NSE index + clean sector) : {both_src:,}")
    print(f"  Score 1 (one source agrees)         : {any_src - both_src:,}")
    print(f"  Score 0 (no source agrees)          : {total - any_src:,}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Sharia compliance status for NSE stocks."
    )
    parser.add_argument(
        "--input", default="data/stocks.csv",
        help="Path to stocks.csv used for symbol list + sector data (default: data/stocks.csv)",
    )
    parser.add_argument(
        "--output", default="data/shariah_status.csv",
        help="Output path (default: data/shariah_status.csv)",
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated NSE symbols to process instead of reading --input",
    )
    args = parser.parse_args()

    # ── Load symbol list and sector data ──────────────────────────────────────
    stocks_df: pd.DataFrame | None = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        print(f"Processing {len(symbols)} symbol(s) from --symbols flag")
    else:
        try:
            stocks_df = pd.read_csv(args.input)
            symbols = stocks_df["stock"].str.strip().dropna().tolist()
            print(f"Loaded {len(symbols):,} symbols from {args.input}")
        except FileNotFoundError:
            print(f"ERROR: {args.input} not found. Run fetch_symbols.py first.")
            sys.exit(1)

    # ── Source 1: NSE Shariah indices ─────────────────────────────────────────
    nse_compliant = fetch_nse_shariah_constituents()

    # ── Source 2: Sector exclusion filter ─────────────────────────────────────
    sector_results = compute_sector_filter(symbols, stocks_df)

    # ── Assemble and write output ─────────────────────────────────────────────
    build_shariah_csv(symbols, nse_compliant, sector_results, args.output)


if __name__ == "__main__":
    main()
