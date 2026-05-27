# ============================================================
# SWING SCANNER - CONFIGURATION
# ============================================================
#
# HOW TO FIND STOCKEDGE IDs:
#   Open https://web.stockedge.com, search for a stock.
#   The URL will be like: /share/tata-steel/2883
#   The number at the end is the StockEdge ID.
#
# NSE_SYMBOL is the official NSE ticker (used for data fetching).
# ============================================================

STOCKS = {
    "TATASTEEL": {
        "nse_symbol":       "TATASTEEL",
        "stockedge_id":     2883,
        "stockedge_slug":   "tata-steel",
        "has_futures":      True,
    },
    "SBIN": {
        "nse_symbol":       "SBIN",
        "stockedge_id":     3045,
        "stockedge_slug":   "state-bank-of-india",
        "has_futures":      True,
    },
    "RELIANCE": {
        "nse_symbol":       "RELIANCE",
        "stockedge_id":     2882,
        "stockedge_slug":   "reliance-industries",
        "has_futures":      True,
    },
    "INFY": {
        "nse_symbol":       "INFY",
        "stockedge_id":     1922,
        "stockedge_slug":   "infosys",
        "has_futures":      True,
    },
    "HDFCBANK": {
        "nse_symbol":       "HDFCBANK",
        "stockedge_id":     1333,
        "stockedge_slug":   "hdfc-bank",
        "has_futures":      True,
    },
    "PCJEWELLER": {
        "nse_symbol":       "PCJEWELLER",
        "stockedge_id":     7943,
        "stockedge_slug":   "pc-jeweller",
        "has_futures":      False,
    },
    "PCJ": {
        "nse_symbol":       "PCJ",
        "stockedge_id":     0,       # update from web.stockedge.com URL
        "stockedge_slug":   "pcj",
        "has_futures":      False,
    },
    "KALYANJIL": {
        "nse_symbol":       "KALYANKJIL",   # actual NSE ticker for Kalyan Jewellers
        "stockedge_id":     0,       # update from web.stockedge.com URL
        "stockedge_slug":   "kalyan-jewellers",
        "has_futures":      False,
    },
    "TBZ": {
        "nse_symbol":       "TBZ",
        "stockedge_id":     0,       # update from web.stockedge.com URL
        "stockedge_slug":   "tbz",
        "has_futures":      False,
    },
}

# ─── Capital & Trade Management ───────────────────────────
CAPITAL            = 100_000   # Total capital (INR)
POSITION_SIZE_PCT  = 20.0      # % of capital per trade (max 5 concurrent)
STOP_LOSS_PCT      = 5.0       # Stop loss from entry
TARGET_PCT         = 10.0      # Profit target from entry
MAX_HOLD_DAYS      = 10        # Max trading days to hold

# ─── Backtest ─────────────────────────────────────────────
BACKTEST_DAYS      = 145       # Calendar days to backtest (covers Jan 1 2026 → present)
BACKTEST_START_DATE = "2026-01-01"  # Hard floor; overrides BACKTEST_DAYS if set
DATA_BUFFER_DAYS   = 50        # Extra days fetched for indicator warm-up

# ─── Strategy ─────────────────────────────────────────────
SUPPORT_LOOKBACK_DAYS   = 30   # Look-back for support level detection
SUPPORT_PROXIMITY_PCT   = 2.0  # Within X% of support = "near support"
BREAKOUT_LOOKBACK_DAYS  = 20   # Look-back for breakout detection
BREAKOUT_BUFFER_PCT     = 0.5  # Within X% below recent high = breakout zone

# When True, a stock without futures data will be skipped.
# When False, signals are generated on 3/4 conditions (OI skipped).
REQUIRE_FUTURES_OI = True

# ─── StockEdge (optional Selenium scraper) ────────────────
USE_STOCKEDGE      = False     # Set True to enable Selenium-based scraping
STOCKEDGE_EMAIL    = ""        # Your StockEdge login email
STOCKEDGE_PASSWORD = ""        # Your StockEdge login password
