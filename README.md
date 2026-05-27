# Swing Scanner — Delivery + OI Strategy

Scans NSE stocks for a delivery-volume + futures open-interest based BUY signal, backtests over the last 90 days, and flags candidates for the next trading day.

## Strategy

A BUY signal is generated only when **all four** conditions are true on the same day:

| # | Condition |
|---|-----------|
| 1 | Delivery quantity **and** delivery % are higher than the previous day |
| 2 | Closing price is above the day's VWAP |
| 3 | Futures Open Interest is positive **and** higher than the previous day |
| 4 | Price is near a support level **or** breaking out from a recent range |

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
# Backtest all stocks + show live signals for next trading day
python swing_scanner.py

# Backtest only
python swing_scanner.py backtest

# Live signals only
python swing_scanner.py signals

# Specific stocks
python swing_scanner.py backtest TATASTEEL SBIN
python swing_scanner.py signals  INFY RELIANCE
```

## Configuration — `config.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STOCKS` | 6 stocks | Add/remove stocks; set `has_futures: False` for stocks not in F&O |
| `CAPITAL` | ₹1,00,000 | Total capital for backtest simulation |
| `POSITION_SIZE_PCT` | 20% | Capital allocated per trade |
| `STOP_LOSS_PCT` | 5% | Exit if price falls this much from entry |
| `TARGET_PCT` | 10% | Exit if price rises this much from entry |
| `MAX_HOLD_DAYS` | 10 | Force-exit after this many trading days |
| `BACKTEST_DAYS` | 90 | Calendar days of history to test |
| `REQUIRE_FUTURES_OI` | `True` | Set `False` to allow signals without OI data |

### Adding a stock

```python
# config.py
STOCKS = {
    "WIPRO": {
        "nse_symbol":     "WIPRO",
        "stockedge_id":   3787,          # from web.stockedge.com URL
        "stockedge_slug": "wipro",
        "has_futures":    True,
    },
    ...
}
```

> **Finding the StockEdge ID:** open `https://web.stockedge.com`, search for a stock — the number in the URL (`/share/wipro/3787`) is the ID.

## Data Sources

Data is pulled in priority order; each level is a fallback for the previous:

1. **NSE Archive files** *(default, no login required)* — `nsearchives.nseindia.com` daily bhav-copy CSVs; contains delivery qty/%, VWAP, and futures OI. Files are cached in `.nse_cache/` after the first download.
2. **StockEdge** *(optional, requires login)* — Selenium-based scraper. Enable via `config.py`:
   ```python
   USE_STOCKEDGE      = True
   STOCKEDGE_EMAIL    = "you@email.com"
   STOCKEDGE_PASSWORD = "yourpassword"
   ```
   Then install: `pip install selenium webdriver-manager`
3. **yfinance** *(last resort)* — provides OHLCV only; delivery % unavailable, so condition 1 will not fire.

## Output

**Backtest summary** (example):
```
  SYMBOL       TRADES  WINS   WIN%    TOTAL P&L  RETURN%      FINAL CAP
------------------------------------------------------------------------
  TATASTEEL         1     0   0.0%      -₹49.40    -0.1%        ₹99,951
  INFY              1     1 100.0%     +₹673.50    +0.7%       ₹100,674
  ...
```

**Live signal** (when conditions are met):
```
  >>> INFY
     Signal date     : 2026-05-26
     Suggested entry : 2026-05-27  (next trading day)
     Entry price     : ₹1,310.00
     Stop loss       : ₹1,244.50  (-5%)
     Target          : ₹1,441.00  (+10%)
     Conditions      : 4/4
```
