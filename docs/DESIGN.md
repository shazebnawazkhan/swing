# Adaptive Multi-Market Swing Trading System — Design Document

**Audience:** A Claude Code agent running **Sonnet 4.6** that will build this system incrementally.
**Authored by:** Fable 5 (architecture + trading-domain guidance).
**Repo:** `D:\SNK\codes\repos\swing` — an existing, working NSE swing scanner that this system **extends, never rewrites**.

---

## 0. How to use this document (read this first, Sonnet)

1. Build **one phase at a time**, in order. Each phase has explicit *deliverables*, *contracts*, and *acceptance tests*. Do not start a phase until the previous phase's acceptance test passes.
2. **Never rewrite working modules.** Everything in §1 is load-bearing. Extend via the existing plugin patterns (strategy registry, fetcher fallback chain, config module). Additive changes only unless a phase explicitly says otherwise.
3. **Verify by running, not by reading.** Every phase ends with a command you run and a measurable result. If it fails, fix it before moving on.
4. Follow the **token-budget rules in §9** strictly. They are part of the spec, not a suggestion.
5. When a contract in §7 conflicts with your instinct, the contract wins. Contracts are what make the self-improvement loop (§8) possible — a machine must be able to compare every experiment ever run.

---

## 1. Inventory: what already exists (REUSE, do not rebuild)

| Asset | Path | What it gives you |
|---|---|---|
| Strategy ABC + plugin registry | `src/strategies/base.py`, `src/strategies/__init__.py` | `Strategy.generate_signals(df) -> df` contract, `@register_strategy`, `StrategyParams` dataclass (JSON-serialisable params), `TradeRecord`, `BacktestResult` |
| 3 working strategies | `src/strategies/{delivery_oi,volume_ema,ema_bb}.py` | Reference implementations of the contract |
| Backtest engine | `src/backtester.py` | Trade simulation (SL/target/max-hold), `BacktestResult`, `compare()`, `summary_by_strategy()` |
| Fast engine | `scripts/run_comparison.py` (`FastBacktester`, `_batch_precompute`) + `src/fast_indicators.py` | Numba/NumPy vectorised EMA/rolling/min + batch trade simulation; perf already profiled in `docs/perf_profile.md` |
| Data fetchers | `src/data_fetcher.py` | `NSEArchiveFetcher` (bhav copy w/ delivery + OI), `NSEApiFetcher`, yfinance fallback, `StockEdgeScraper`, `DataFetcher` orchestrator with fallback chain |
| Decoupled incremental fetch | `scripts/fetch_data.py` | Per-symbol parquet store at `data/fetched/<SYMBOL>.parquet`, incremental updates, threaded workers, status/benchmark reporting |
| Universe + screens | `data/stocks.csv` (full NSE universe), `data/shariah_status.csv` + `scripts/fetch_shariah.py` | Universe selection and an ethical/halal filter — treat as a reusable **universe filter** pattern |
| Reporting | `scripts/run_comparison.py` (`build_report`), `outputs/dashboard.html`, `scripts/server.py` | HTML comparison report, trades CSV, dashboard server |
| Raw NSE cache | `.nse_cache/*.csv` | ~8 months of daily NSE bhav copies already downloaded — backtests run offline |
| Config | `src/config.py` | Capital, SL/target/hold, backtest window. Risk params deliberately live here, **outside** strategies, so strategies compare under identical risk rules — keep that separation |

**Architecture principle inherited from this codebase:** *signal generation* (strategy), *trade management* (backtester/config), and *data acquisition* (fetchers) are decoupled. Every new component must respect this.

---

## 2. Target system overview

```
                 ┌───────────────────────────────────────────────┐
                 │              SELF-IMPROVEMENT LOOP             │
                 │  experiments.jsonl ► runner ► leaderboard.json │
                 │        ▲ hypotheses          │ promote/demote  │
                 └────────┼─────────────────────┼─────────────────┘
                          │                     ▼
┌──────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ DATA HUB │──►│ CONTEXT ENGINE   │──►│ STRATEGY LAB      │──►│ PORTFOLIO/RISK  │
│ prices,  │   │ regime detector, │   │ registry + params,│   │ sizing, exposure│
│ macro,   │   │ sector strength, │   │ walk-forward opt, │   │ caps, FX hedge, │
│ news, FX │   │ news sentiment   │   │ backtest+metrics  │   │ signal output   │
└──────────┘   └──────────────────┘   └──────────────────┘   └─────────────────┘
  parquet           context.json         results store          signals + report
```

Three markets, one pipeline: **NSE (India)** via existing archive fetcher; **US + global equities and FX** via a new yfinance-based adapter writing the *same parquet bar schema*. Strategies never know where data came from.

---

## 3. Phase plan

> Estimated effort assumes Sonnet 4.6 working in focused sessions. Each phase is sized to fit comfortably in one session with context to spare.

### Phase 0 — Metrics upgrade (foundation for everything)
The current `BacktestResult` lacks the metrics needed to rank strategies credibly.

**Deliverables**
- `src/metrics.py`: pure functions taking a trade list + equity curve → dict of:
  `sharpe_ratio` (annualised, daily returns, rf=6.5% INR / 4% USD configurable), `sortino_ratio`, `calmar_ratio`, `profit_factor`, `expectancy_pct`, `max_drawdown_pct`, `max_drawdown_days`, `exposure_pct` (fraction of days in market), `cagr_pct`.
- Extend `BacktestResult` with these fields (defaults `0.0` so existing constructors don't break).
- **Transaction costs**: add to `config.py` → `BROKERAGE_PCT = 0.03`, `STT_PCT = 0.025` (sell side), `SLIPPAGE_PCT = 0.05`. Apply in `Backtester._simulate` and the fast path in `fast_indicators.simulate_trades`. *India delivery round-trip realistic total ≈ 0.2–0.3%; this single change will materially lower every win rate — that is correct and desired.*
- Entry at **next day's open** (currently entries fill at signal-day close — lookahead bias). Add `ENTRY_AT_NEXT_OPEN = True` to config; implement in both simulators.

**Acceptance:** `python -m scripts.run_comparison` (or current invocation) completes; report shows the new columns; a hand-checked Sharpe on one symbol matches a manual pandas computation within 1%.

### Phase 1 — Data hub: global markets, macro, news
**Deliverables**
- `src/adapters/yf_adapter.py`: fetch any yfinance ticker (US stocks, `^NSEI`, `^GSPC`, FX pairs `USDINR=X`, `EURUSD=X`, commodities `GC=F`) → normalise to the **bar schema** (§7.1) → write `data/fetched/<MARKET>/<SYMBOL>.parquet`. Reuse the incremental-update pattern from `scripts/fetch_data.py` (read last cached date, fetch delta only).
- `src/adapters/macro_adapter.py`: FRED CSV endpoint (no key needed: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>`) for `FEDFUNDS, CPIAUCSL, DGS10, DTWEXBGS, VIXCLS`; India series (repo rate, CPI) from a small **manually-maintained CSV** `data/macro/india_manual.csv` (RBI has no clean free API — do not waste effort scraping; flag rows older than 45 days as stale).
- `src/adapters/news_adapter.py`: Google News RSS (`https://news.google.com/rss/search?q=<query>`) per symbol/sector — store headline, date, source in `data/news/<date>.parquet`. **Sentiment = keyword scoring** (wordlists in `src/adapters/sentiment_words.py`: order-win/upgrade/expansion vs. probe/default/downgrade) — *not* an LLM call per headline (cost rule §9). Order-win detection is a keyword class (`wins order|bags order|order book|LOI|contract worth`), giving requirement "company order win books" as a news-derived boolean feature.
- `scripts/fetch_all.py`: one CLI orchestrating NSE + global + macro + news fetch with `--status` mode (copy the UX of `fetch_data.py`).

**Acceptance:** `python scripts/fetch_all.py --market us --symbols AAPL MSFT` then `--status` shows fresh parquet; a backtest of `ema_bb` on AAPL runs unmodified — proving market-agnostic strategies.

### Phase 2 — Context engine: regime + sector strength
This is what makes the system *adaptive* (requirement 4).

**Deliverables — `src/context.py`**
- `detect_regime(index_df) -> str` per market index (NIFTY, S&P 500): classify each day into `bull / bear / sideways / high_vol` using: 50-EMA vs 200-EMA direction, ADX(14) < 20 ⇒ sideways, realised 20-day vol > 80th percentile ⇒ high_vol. Pure pandas/numpy, reuse `fast_indicators`.
- `sector_strength(market) -> DataFrame`: 20/60-day relative strength of sector indices vs market index (NSE sector indices via yfinance: `^CNXAUTO`, `^CNXIT`, `^CNXPHARMA`, etc.; US via sector ETFs XLK/XLF/XLE/…). Output: ranked sectors with RS score.
- `macro_bias(market) -> float` in [-1, +1]: simple rule blend (rates trending down +, CPI falling +, VIX > 25 −, DXY rising − for EM).
- `build_context() -> dict` → writes `outputs/context.json` (schema §7.3). Everything downstream reads this file; nothing recomputes context.

**Acceptance:** `python -m src.context` prints today's regime per market + top-3 sectors; spot-check against reality (you can verify NIFTY's 50/200 EMA from the cached data itself).

### Phase 3 — Strategy lab: regime-aware strategies + walk-forward optimisation
**Deliverables**
- New strategies (each ~60 lines, follow `ema_bb.py` as the template):
  - `mean_reversion_bb.py` — *sideways regime*: buy lower-band touch with RSI(2) < 10, exit mid-band.
  - `pullback_trend.py` — *bull regime*: 50-EMA uptrend + pullback to 20-EMA + bullish close.
  - `sector_momentum.py` — *bull regime*: only fires for symbols in top-2 RS sectors (reads `context.json`).
  - `news_catalyst.py` — order-win/sentiment spike + volume confirmation (reads news features).
- `RegimeGate` wrapper (in `src/strategies/regime_gate.py`): wraps any strategy, zeroes `buy_signal` when current regime ∉ strategy's declared `params.regimes`. Add `regimes: list[str]` to `StrategyParams` (default `["bull","bear","sideways","high_vol"]` = always on, so existing strategies are unaffected).
- `src/optimizer.py` — **walk-forward** parameter search:
  - Split history: train 6 months → test 2 months, rolling.
  - Grid/random search over `StrategyParams` fields (each strategy declares `param_grid: dict` class attribute).
  - Score = test-window Sharpe; **reject any param set whose test Sharpe < 50% of train Sharpe** (overfit guard).
  - Persist every evaluation to the results store (§7.4) — never lose an experiment.

**Acceptance:** `python scripts/optimize.py --strategy "EMA BB" --symbols-from-cache --quick` completes a small walk-forward and writes results; comparison report now shows regime-gated variants beating ungated ones in at least one regime slice.

### Phase 4 — Portfolio & risk engine
**Deliverables — `src/portfolio.py`**
- Position sizing: `risk_per_trade = 0.5–1%` of equity, `shares = risk_amount / (entry − stop)` (replaces flat 20% sizing; keep the old mode behind a config flag for comparability).
- Caps: max 5 concurrent positions, max 25% per sector, max 60% per market (global outlook = forced diversification across NSE/US), max portfolio heat (sum of open risk) 5%.
- FX awareness: INR-base equity curve; US positions marked to market via `USDINR=X` daily; report currency P&L separately.
- ATR-based stops as an alternative to fixed % (config flag `STOP_MODE = "atr" | "pct"`; ATR(14) × 2).
- Optional universe filters chainable: `--halal` (existing `shariah_status.csv`), `--min-liquidity` (avg turnover), `--fno-only`.
- Output: `outputs/signals_<date>.json` — ranked, sized, risk-checked orders for the next session (schema §7.5). This is the system's daily product.

**Acceptance:** portfolio backtest over cached history runs both markets simultaneously and the equity curve in the report respects every cap (assert in code: write `tests/test_portfolio_caps.py`).

### Phase 5 — Evaluation harness & leaderboard
**Deliverables**
- `src/results_store.py`: append-only JSONL at `data/results/experiments.jsonl` (schema §7.4) + `outputs/leaderboard.json`: best strategy per (market × regime), with its params, walk-forward Sharpe, max DD, profit factor, last-evaluated date.
- Promotion rules (encode exactly): a strategy variant is **live** for a regime if walk-forward test Sharpe ≥ 1.0, max DD ≤ 15%, ≥ 30 test trades, and it beat the incumbent for 2 consecutive evaluations. Demote when trailing 60-day live Sharpe < 0 or DD breaches 1.5× its backtest DD.
- `scripts/evaluate.py`: re-runs the full strategy × regime matrix on latest data, updates leaderboard, prints a one-screen diff ("promoted X, demoted Y").

**Acceptance:** run `scripts/evaluate.py` twice; second run is incremental (skips unchanged evaluations via a hash of `(strategy, params, data_end_date)`), confirming the store prevents recomputation.

### Phase 6 — Self-improvement loop (autoresearch)
See §8 for full design. Deliverables: `scripts/research.py`, `data/research/backlog.jsonl`, `docs/research/` notes, `outputs/system_health.json`.

**Acceptance:** `python scripts/research.py --one-cycle` picks the top backlog hypothesis, runs it, appends a research note, updates backlog statuses, and records cycle runtime in system_health.

### Phase 7 — Ops: scheduling, dashboard, paper trading
- Extend `outputs/dashboard.html` + `scripts/server.py`: leaderboard, today's signals, regime banner, equity curve, system health sparkline.
- Daily pipeline `scripts/daily.py`: fetch → context → signals → evaluate (weekly) → research (weekly) — designed to be run by Windows Task Scheduler or a Claude Code scheduled agent (`/schedule`).
- **Paper-trade ledger** `data/paper_trades.jsonl`: every emitted signal is recorded with its eventual outcome (filled at next open from data). This creates the live-vs-backtest gap measurement that feeds §8. *Live execution (e.g. Groww MCP for NSE) is explicitly out of scope until paper Sharpe ≥ backtest Sharpe × 0.7 for 60 days.*

---

## 4. Strategy design notes (domain guidance from Fable)

These priors are encoded so the agent doesn't rediscover them expensively:

- **Swing horizon is 2–15 days.** Daily bars suffice; do not build intraday infrastructure.
- **Delivery % + OI (the existing edge) is India-specific alpha** — it has no US equivalent; keep it NSE-gated. The transferable strategies are trend-pullback, BB mean-reversion, sector momentum, news catalyst.
- **Regime gating beats parameter tuning.** Most strategy "failure" is running a trend system in a chop. Expect Phase 2+3 to add more Sharpe than any optimisation.
- **Sideways markets:** mean reversion with tight time stops (3–5 days). **Trending:** pullback entries, trail with ATR, let MAX_HOLD extend to 15–20 days. Encode as regime-conditional config overrides, not new strategies.
- **FX for a swing system is a risk overlay, not an alpha source initially.** Hedge logic: if USDINR 20-day trend is against your USD exposure and exposure > 30% of equity, halve new US entries. Trade FX directly only after the loop (§8) demonstrates an edge.
- **Costs and the next-open fill (Phase 0) are the most important "features" in this whole document.** A system that is honest about costs and fills self-improves; one that isn't optimises noise.
- **Expect realistic numbers:** good swing systems show Sharpe 1.0–1.8 and 45–55% win rate with payoff > 1.5 after costs. If an experiment shows Sharpe > 3, hunt for the bug (lookahead, survivorship, fill-at-close) before celebrating.

---

## 5. Data source catalog (free-first; cost is a tracked metric)

| Data | Source | Cost | Notes |
|---|---|---|---|
| NSE EOD + delivery + OI | NSE archives (existing) | free | already cached in `.nse_cache/` |
| US/global equities, indices, FX, commodities | yfinance | free | rate-limit friendly: batch via `yf.download(tickers=...)`, throttle 2s between batches |
| US macro | FRED csv endpoint | free, no key | series list in Phase 1 |
| India macro | manual CSV (`data/macro/india_manual.csv`) | free | refresh monthly by hand; staleness flagged |
| News / order wins / political | Google News RSS, keyword-scored | free | no LLM calls in the pipeline |
| NSE corporate announcements | `nseindia.com/api/corporate-announcements` (reuse `NSEApiFetcher` session/headers) | free | order wins, results dates — best-effort, wrap in try/except, never a hard dependency |
| Shariah screen | existing `fetch_shariah.py` | free | already built |

Every adapter must record `(source, rows_fetched, seconds, bytes)` per run into `outputs/system_health.json` — this is the "cost to gather data" metric the improvement loop optimises.

---

## 6. Repository layout (target)

```
src/
  adapters/            # NEW: yf_adapter, macro_adapter, news_adapter, sentiment_words
  strategies/          # existing + new strategy modules + regime_gate
  backtester.py        # existing (extended with costs/next-open in Phase 0)
  metrics.py           # NEW Phase 0
  context.py           # NEW Phase 2
  optimizer.py         # NEW Phase 3
  portfolio.py         # NEW Phase 4
  results_store.py     # NEW Phase 5
  fast_indicators.py   # existing — add ATR, RSI, ADX here (vectorised)
  config.py            # existing — grows new flags; stays the single risk-param home
scripts/
  fetch_all.py daily.py optimize.py evaluate.py research.py   # NEW
  fetch_data.py run_comparison.py server.py ...               # existing
data/
  fetched/{nse,us,fx}/ macro/ news/ results/ research/
docs/
  DESIGN.md (this file)  research/  perf_profile.md
outputs/
  context.json leaderboard.json signals_<date>.json system_health.json dashboard.html
tests/                 # NEW: pytest; one focused test file per phase
```

---

## 7. Contracts (stable interfaces — the spine of the system)

### 7.1 Bar schema (every price parquet, every market)
```
date (datetime64), open, high, low, close, total_volume (float)
# optional, NSE only: delivery_qty, delivery_pct, vwap, futures_oi
symbol (str), market (str: "nse"|"us"|"fx")
```
Strategies must treat optional columns as absent-tolerant (existing `delivery_oi` already shows the pattern: missing data ⇒ condition False, never raise).

### 7.2 Strategy contract — unchanged from `base.py`
`generate_signals(df) -> df` with `buy_signal: bool`. New optional class attrs: `param_grid: dict[str, list]`, `params.regimes: list[str]`, `params.markets: list[str]` (default all).

### 7.3 `outputs/context.json`
```json
{"as_of": "2026-06-12",
 "markets": {"nse": {"regime": "bull", "index_close": 0.0, "vol_pctile": 0.0,
                      "top_sectors": [["IT", 1.8], ["AUTO", 1.2]], "macro_bias": 0.3},
             "us": {...}},
 "fx": {"usdinr": {"close": 0.0, "trend_20d": "up"}}}
```

### 7.4 Experiment record (`data/results/experiments.jsonl`, one JSON per line)
```json
{"id": "exp_YYYYMMDD_HHMMSS_<slug>", "ts": "...", "kind": "walkforward|backtest|research",
 "strategy": "...", "params": {...}, "universe": "...", "market": "nse",
 "window": {"train": ["...","..."], "test": ["...","..."]},
 "metrics": {"sharpe": 0.0, "max_dd_pct": 0.0, "profit_factor": 0.0, "trades": 0, "win_rate": 0.0},
 "hypothesis_id": "h_001|null", "verdict": "promote|reject|inconclusive", "notes": ""}
```

### 7.5 Daily signal (`outputs/signals_<date>.json`)
```json
{"date": "...", "regime": {"nse": "bull"}, "signals": [
  {"symbol": "INFY", "market": "nse", "strategy": "Pullback Trend", "rank": 1,
   "entry_type": "next_open", "stop": 0.0, "target": 0.0, "shares": 0,
   "risk_inr": 0.0, "sector": "IT", "filters_passed": ["halal", "liquidity"]}]}
```

### 7.6 Hypothesis (`data/research/backlog.jsonl`)
```json
{"id": "h_001", "created": "...", "category": "alpha|risk|speed|data_cost|cycle_time",
 "statement": "ATR stops reduce max DD vs fixed 5% without hurting CAGR",
 "test_plan": "walkforward both stop modes, all strategies, NSE universe",
 "expected_metric": "max_dd_pct -20% relative, cagr_pct within ±10%",
 "status": "open|running|confirmed|rejected", "result_exp_ids": [], "priority": 1}
```

---

## 8. Self-improvement loop (the heart of requirement 7)

The loop is **deterministic infrastructure + agent-driven judgment**, cleanly split:

**Deterministic (runs without any LLM, zero token cost):** `scripts/research.py --one-cycle`
1. Pop highest-priority `open` hypothesis from backlog.
2. Execute its `test_plan` (test plans are *structured*: `{"action": "walkforward", "strategy": ..., "param_overrides": ..., "universe": ...}` — research.py is an interpreter for ~5 action types, not a code generator).
3. Compare metrics vs `expected_metric`; set status `confirmed/rejected`; append experiment records; write a markdown note to `docs/research/h_<id>.md` (template: hypothesis, method, table of metrics, verdict).
4. Update `outputs/system_health.json`: cycle runtime, backtest throughput (symbol-days/sec), data fetch cost, cache hit rate, error count, **capital cycle time** (avg days from signal → exit, the user's "cycle time of investment" metric).

**Agent-driven (you, Sonnet, on a weekly cadence or when invoked):** generate *new* hypotheses into the backlog. Sources, in priority order:
1. **The system's own telemetry** — read `system_health.json` + `leaderboard.json` + the last 5 research notes. Falling Sharpe in a regime ⇒ alpha hypothesis; rising fetch seconds ⇒ data-cost hypothesis; avg hold creeping up ⇒ cycle-time hypothesis (e.g. "add a 3-day time-stop to mean-reversion exits").
2. **Live-vs-backtest gap** from the paper ledger (Phase 7) — the single best source of true hypotheses.
3. **Web research** (`WebSearch`) — *only* after telemetry-driven ideas are exhausted; cap at ~3 searches per session; distill findings into hypothesis JSON, never paste articles into the repo.

**Seed backlog (create in Phase 6 so the loop has fuel):**
- h_001 ATR vs fixed stops (risk)
- h_002 regime-gating delivery_oi to bull/sideways only (alpha)
- h_003 3-day time-stop on mean-reversion (cycle_time)
- h_004 skip news fetch for symbols with no open signal candidates (data_cost)
- h_005 batch parquet reads with `pyarrow.dataset` vs per-file (speed)
- h_006 volatility-scaled position sizing vs fixed risk (risk)
- h_007 Monday/Friday entry-day effect on NSE swing entries (alpha)

**Self-improvement also covers the codebase:** a hypothesis with `category: speed` may conclude "vectorise ADX in fast_indicators" — the research note then becomes a TODO you implement next session. The loop improves strategies *and* the system, exactly as required.

**Safety rails:** the loop may change *parameters and leaderboard status* autonomously; it may **never** change risk caps in `config.py`, delete data, or enable live trading. Those require the human.

---

## 9. Rules for the Sonnet 4.6 agent (token economy + how to excel)

Sonnet 4.6's strengths — disciplined contract-following, clean incremental diffs, reliable test-writing, fast tool use — are exactly what this build needs. Its risk is scope drift across long sessions and over-eager refactoring. These rules convert the strengths into a self-improving system at minimum token cost:

1. **Navigate by this document, not by re-reading the codebase.** §1 and §6 are your map. When you must read existing code, read the *docstring + the one function you're touching*, never whole files. `run_comparison.py` is 97KB — grep for the function you need (`Grep` with `-n`, then `Read` with offset/limit).
2. **Run scripts, read summaries.** Never `Read` parquet/CSV outputs or `.nse_cache` files. Every script you build must print a ≤20-line summary (count, date range, top-5 rows) precisely so you can verify cheaply. If a script lacks one, add it — that's part of the spec.
3. **One phase = one session = one commit.** Commit message: `Phase N: <deliverable>`. Start every new session by reading only: this file's relevant phase, `git log --oneline -5`, and the acceptance test of the previous phase (run it — trust nothing else).
4. **Write the test before the feature for contracts** (§7 schemas). A 30-line pytest that round-trips a schema costs almost nothing and prevents the most expensive failure mode: silent schema drift discovered three phases later.
5. **No LLM calls inside the pipeline.** Sentiment is wordlists; hypothesis testing is the deterministic interpreter; regime detection is math. *You* are the only intelligence in the loop, applied at design time (writing hypotheses, reading research notes) — at most a few times per week. This keeps the marginal cost of a research cycle at zero tokens.
6. **Prefer extending `fast_indicators.py`** for any new indicator (ATR, RSI, ADX) — the numba batch pattern is established; copy `rolling_mean`'s structure. Pandas-rolling in a per-symbol loop is the known perf trap (see `docs/perf_profile.md`).
7. **When uncertain between two designs, pick the one with the smaller diff** against existing code. Cleverness is reserved for hypotheses in the backlog, where it gets tested instead of trusted.
8. **Verify with the cached data first** (offline, instant, free) before any network fetch. The 8 months of `.nse_cache` supports complete Phase 0/2/3/4/5 development without one HTTP request.
9. **Windows environment:** PowerShell quirks per harness guidance; paths with backslashes; `python` from `.venv`. Test commands must work in PowerShell 5.1 (no `&&`).
10. **Stop conditions:** if an acceptance test fails twice with different fixes, write the failure into `docs/research/blockers.md` and ask the human rather than burning context on a third blind attempt.

---

## 10. Pitfalls checklist (audit every backtest against this)

- [ ] **Lookahead:** entry at next open, not signal close (Phase 0); indicators use only `shift(1)`-visible data; context.json for day T built from data ≤ T-1 close.
- [ ] **Survivorship:** `data/stocks.csv` is today's universe — fine for live scanning, biased for long backtests. Mitigation: restrict backtests to the cached window (≤ 1y) and note the bias in every research verdict; do not buy historical constituent data yet (cost rule).
- [ ] **Costs:** every simulated fill pays brokerage + STT + slippage; FX trades pay spread (~2 paise USDINR).
- [ ] **Overfit:** walk-forward only; reject test/train Sharpe < 0.5; min 30 trades for any promotion; param grids ≤ 50 combinations per run.
- [ ] **Multiple testing:** the leaderboard tracks *how many* variants were tried per strategy family; a winner among 50 siblings needs a higher bar (require Sharpe ≥ 1.2 instead of 1.0 when siblings > 20).
- [ ] **Stale data:** macro/news staleness flags propagate into context.json; a strategy must not silently trade on 45-day-old macro bias.
- [ ] **Currency:** all global P&L reported in INR base; FX translation applied daily, not at exit.

---

## 11. Definition of done (whole system)

1. `python scripts/daily.py` runs end-to-end offline-tolerant: fetch (skips gracefully if no network) → context → signals → dashboard, in < 5 minutes on cached universe.
2. Leaderboard holds a promoted strategy for ≥ 3 of the 4 regimes, each with walk-forward Sharpe ≥ 1.0 after costs.
3. `python scripts/research.py --one-cycle` is autonomous and zero-token; the backlog has ≥ 5 open hypotheses at all times (agent replenishes weekly).
4. `system_health.json` trends are visible on the dashboard: backtest throughput, fetch cost, capital cycle time, live-vs-backtest gap.
5. Paper ledger live; promotion to real execution gated on §3 Phase 7 criterion and explicit human approval.
