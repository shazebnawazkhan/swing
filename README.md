# Swing Scanner — Delivery + OI Strategy

Scans NSE stocks for a delivery-volume + futures open-interest based BUY signal, backtests over the last 90 days, and flags candidates for the next trading day.

## Research pipeline (new)

The repo is growing into an adaptive multi-market system — see `docs/DESIGN.md` (architecture), `docs/program.md` (autoresearch loop) and `docs/strategy_format.md` (declarative strategy specs).

```bash
# 1. Build/refresh the halal-universe data store (date-major, ~35s warm)
python scripts/bulk_fetch.py --universe halal --start 2025-03-01

# 2. Run every strategy spec in data/strategies/ over the past year
python scripts/run_experiments.py

# Results: outputs/experiments_summary.csv + data/results/experiments.jsonl
```

## Upcoming results calendar

Find every NSE company declaring quarterly results in the coming week and preview the
numbers that move a stock around its result — revenue / profit / EPS growth, margins,
valuation, and the analyst-firm target price — in a dashboard with **one tab per date**.

```bash
# Fetch the calendar + enrich (writes outputs/earnings_calendar.html + _latest.json)
python scripts/build_earnings_calendar.py --days 7

python scripts/build_earnings_calendar.py --limit 40     # cap enrichment (rate-limit guard)
python scripts/build_earnings_calendar.py --no-enrich     # calendar only, skip yfinance

# Then view it (served at /earnings)
python scripts/server.py    # → http://localhost:8765/earnings
```

| Layer | Source | Notes |
|-------|--------|-------|
| Results calendar | NSE forthcoming board-meetings / event-calendar API | Purpose filtered to financial results; yfinance earnings-dates fallback if NSE is blocked |
| KPIs & growth | yfinance `.info` + earnings estimates | Revenue / profit / EPS-YoY growth, margins, forward PE |
| Analyst targets | yfinance analyst price targets | Mean + low–high range, number of firms, consensus recommendation, upside vs current |
| Reference links | Per company | NSE quote, Yahoo Finance estimates, Screener.in — every number is traceable |

Backend lives in `src/predictor/adapters/earnings_calendar_adapter.py` (graceful per-symbol
degradation, same as the news/fundamentals adapters). Full-market coverage; rows in your
`stocks.csv` universe are tagged **mine** / **halal**.

## Data model

Everything under `data/` falls into two classes: **reference inputs** (small, human/curated, git-tracked) and **generated stores** (large, regenerable, git-ignored). The scanner and research loop read the first and (re)build the second.

```
data/
├── stocks.csv                 reference  master symbol list (2.3k NSE stocks)
├── fetch_universe.csv         reference  active fetch universe (enabled subset)
├── shariah_status.csv         reference  halal screen per symbol
├── .archive_holidays.json     cache      NSE non-trading dates
├── .bulk_manifest.json        cache      trading dates already bulk-fetched
├── .musaffa_cache.json        cache      Musaffa halal lookups {symbol: 0|1}
├── strategies/*.json          reference  declarative strategy specs (one per file)
├── research/backlog.jsonl     ledger     hypothesis backlog (one hypothesis/line)
├── results/experiments.jsonl  ledger     experiment run log (one run/line, append-only)
├── fetched/*.parquet          generated  per-symbol daily OHLCV+delivery+OI bars
├── features/                  generated  ML feature panel + adapter caches
│   ├── panel_<date>.parquet          the wide daily feature/label matrix
│   ├── feature_manifest.json         per-column {source, lag_bars, kind}
│   ├── fundamentals/earnings.parquet EPS estimates vs reported
│   ├── intraday/<date>.parquet       intraday minute bars
│   └── news/<date>.parquet           per-symbol news sentiment snapshot
└── results/trades/*.parquet   generated  per-strategy backtest trade blotter
```

### Reference inputs (tracked)

**`stocks.csv` / `fetch_universe.csv`** — the symbol universe.

| Column | Type | Notes |
|--------|------|-------|
| `stock` | str | NSE symbol (primary key) |
| `company_name` | str | Full name |
| `sector`, `sub_sector` | str | Classification |
| `enabled` | Y/N | Included in the active fetch universe |
| `halal` | Y/N | Passes the Shariah screen |

**`shariah_status.csv`** — screen breakdown keyed by `stock`: `nse_index` (0/1 index member), `sector_filter` (0/1 permissible sector), `shariah_score` (0/1 final verdict).

**`strategies/*.json`** — one declarative spec per file (schema `strategy-spec/v1`, see `docs/strategy_format.md`):

| Field | Meaning |
|-------|---------|
| `id`, `name`, `description` | Identity |
| `markets`, `regimes` | Applicable market / regime tags |
| `params` | Named tunables referenced as `@name` in rules |
| `indicators` | Derived series `{fn, on, window}` |
| `entry` | `all`/`any` boolean rule lists over columns & indicators |
| `exit` | `stop_loss_pct`, `target_pct`, `max_hold_days` |
| `param_grid` | Sweep values for optimization |
| `provenance` | `created`, `hypothesis_id`, `parent_id`, `experiments[]` |

### Research ledgers (tracked, JSONL)

**`research/backlog.jsonl`** — one hypothesis per line: `id` (`h_NNN`), `created`, `category`, `statement`, `test_plan{action, specs[], universe, window_days}`, `expected_metric`, `status` (`open`/`rejected`/…), `result_exp_ids[]`, `priority`, `verdict_note`.

**`results/experiments.jsonl`** — append-only run log, one experiment per line: `id` (`exp_<ts>_<spec>`), `ts`, `kind`, `spec_id`, `params`, `universe`, `n_symbols`, `window`, `costs`, `exit`, a `metrics{}` block (`trades`, `win_rate`, `profit_factor`, `sharpe_approx`, `max_drawdown_pct`, `expectancy_pct`, …), `hypothesis_id`, `verdict`, `note`, `runtime_s`.

### Generated stores (git-ignored, regenerable)

**`fetched/<SYMBOL>.parquet`** — per-symbol daily bars, the raw substrate: `symbol, date, open, high, low, close, total_volume, delivery_qty, delivery_pct, vwap, oi, oi_change`. Index files `_NIFTY.parquet` and `_UNIVERSE_INDEX.parquet` hold `date, close` only. Rebuilt by `scripts/bulk_fetch.py`.

**`features/panel_<date>.parquet`** — the wide daily feature/label matrix (~0.5M rows) keyed by `date, symbol`. Feature columns (price/volume/delivery/OI/cross-sectional): `ret_{1,3,5,10,20}`, `ema_gap_{20,50,200}`, `rsi_{14,2}`, `atr_pct_14`, `bb_pctb_20`, `dist_120_high`, `gap_pct`, `vol_z_20`, `above_ema200`, `delivery_pct`, `deliv_vs_avg`, `deliv_z_20`, `vwap_dev`, `oi_change`, `rs_rank_60`, `ret_20_xs`, `deliv_pct_xs`, `mkt_ret_20`, `mkt_above_ema50`, `rel_ret_20`, `dow`. Labels: `y_dir1d`, `y_swing`, `swing_w`, `fwd_ret_pct`. `feature_manifest.json` records each column's `{source, lag_bars, kind}` for leakage auditing.

**`features/{fundamentals,intraday,news}/`** — adapter caches for the predictor: `fundamentals/earnings.parquet` (`symbol, report_date, eps_est, reported_eps, surprise_pct`); `intraday/<date>.parquet` (minute bars `symbol, datetime, OHLCV`); `news/<date>.parquet` (`symbol, news_sent, news_count, order_win_flag, days_since_news, top_headline, news_age_days, date`).

**`results/trades/<spec_id>.parquet`** — per-strategy trade blotter from the last backtest: `symbol, entry_date, exit_date, entry_price, exit_price, shares, hold_days, cost_pct, pnl_pct, gross_pnl, exit_reason`.

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
