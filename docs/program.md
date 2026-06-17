# Autoresearch Program — Designing & Running Experiments

How this system researches, tests and refines trading strategies automatically.
Companion documents: `docs/DESIGN.md` (system architecture, §8) and
`docs/strategy_format.md` (the strategy spec format every experiment operates on).

---

## 1. Purpose

Turn strategy development from ad-hoc coding into a **closed experimental loop**:

```
 hypothesis ──► experiment design ──► run (backtest) ──► verdict ──► refine / reject
     ▲                                                                  │
     └────────────── telemetry, leaderboard, research notes ◄──────────┘
```

Two design rules make the loop cheap and trustworthy:

1. **Strategies are data** (`strategy-spec/v1` JSON). Creating, mutating or sweeping a
   strategy never requires code changes — an experiment is "evaluate these spec files
   under these conditions."
2. **Every run is recorded** (append-only `data/results/experiments.jsonl`). Negative
   results are kept forever so the loop never re-tests a dead idea.

The runner itself is deterministic Python — **no LLM calls inside the loop**. Agent
intelligence (Claude) is applied only at the edges: writing hypotheses, reading
verdicts, designing refinements.

---

## 2. Components & file map

| Component | Path | Role |
|---|---|---|
| Strategy specs | `data/strategies/*.json` | The objects under test (immutable once referenced) |
| Spec interpreter | `src/strategies/spec.py` | Turns a spec into a runnable `Strategy` |
| Bulk data store | `data/fetched/<SYMBOL>.parquet` | 1y+ OHLCV+delivery bars, built by `scripts/bulk_fetch.py` |
| Experiment runner | `scripts/run_experiments.py` | Backtests specs across a universe, computes metrics, records results |
| Metrics | `src/metrics.py` | Sharpe (per-trade approx.), profit factor, expectancy, max DD, … |
| Results store | `data/results/experiments.jsonl` | Append-only experiment records (schema §5) |
| Summary | `outputs/experiments_summary.csv` | Latest run's leaderboard-style comparison table |
| Leaderboard builder | `scripts/build_leaderboard.py` | Joins latest experiment per spec with its spec metadata + hypothesis verdict → `outputs/leaderboard.json`; called automatically at the end of `run_experiments.py` |
| Dashboard | `outputs/dashboard.html` (Strategy Lab section) | Sortable view of `outputs/leaderboard.json`, served via `scripts/server.py` |
| Hypothesis backlog | `data/research/backlog.jsonl` | Open research questions (schema §5) |
| Research notes | `docs/research/*.md` | Human-readable verdict per hypothesis |

---

## 3. The experiment lifecycle

### 3.1 Hypothesis
Every experiment starts as one line in `data/research/backlog.jsonl`:

```json
{"id": "h_008", "created": "2026-06-12", "category": "alpha",
 "statement": "High-delivery down days mark accumulation; price recovers within 10 days",
 "test_plan": {"action": "backtest_specs", "specs": ["delivery_absorption_v1"],
               "universe": "halal", "window_days": 365},
 "expected_metric": "profit_factor > 1.3 and trades >= 100",
 "status": "open", "result_exp_ids": [], "priority": 2}
```

Categories: `alpha` (returns), `risk` (drawdown), `cycle_time` (capital turnover),
`speed` (runtime), `data_cost` (fetch efficiency). The loop improves the *whole system*,
not just strategies.

### 3.2 Design
The hypothesis is materialised as one or more **spec files** (new strategy, or a child
spec refining a parent via `provenance.parent_id` + changed `params`). Design rules:

- One variable per experiment: a child spec changes **one** param family vs its parent.
- Param sweeps go in the spec's `param_grid`; the runner expands grid combinations.
- Declare the regime/market the idea should work in (`regimes`, `markets`) — a later
  regime-sliced evaluation will check that claim.

### 3.3 Run
```
python scripts/run_experiments.py                         # all specs, halal universe, 1y
python scripts/run_experiments.py --specs rsi2_meanrev_v1 # subset
python scripts/run_experiments.py --grid pullback_trend_v1  # expand its param_grid
python scripts/run_experiments.py --hypothesis h_008      # tag records with hypothesis id
```

What the runner guarantees:

- **No lookahead:** signals computed from day-T close are filled at the **next bar**
  (signal shift), and indicators only use current/past rows.
- **Costs:** every trade pays `ROUND_TRIP_COST_PCT` (config, default 0.25% — brokerage
  + STT + slippage for NSE delivery).
- **Identical risk rules** across specs unless a spec overrides `exit` — comparisons
  are apples-to-apples.
- **Reproducibility:** the experiment record embeds the spec's effective params, the
  universe id, window dates and the config cost assumptions.
- Numba-vectorised simulation (`src/fast_indicators.py`) → a full 2,000+ symbol × 1y
  run per spec costs seconds, so wide search is affordable.

### 3.4 Verdict
The runner compares metrics to `expected_metric` (when run with `--hypothesis`) and
sets the backlog row's status to `confirmed` / `rejected` / `inconclusive`. Promotion
gates (a spec becomes *leaderboard-live*):

- `trades >= 100` across the universe (statistical floor),
- `profit_factor >= 1.3` **after costs**,
- `sharpe_approx >= 1.0`,
- `max_drawdown_pct <= 15`,
- and it beats the incumbent spec for its declared regime.

Anti-overfitting bar: when a spec family has > 20 evaluated siblings (grid sweeps count),
raise the Sharpe gate to 1.2 — winners among many trials need stronger evidence.

### 3.5 Refine or retire
- **Refine:** copy the spec → bump `_v<N>` → set `parent_id` → change one thing →
  new experiment. The lineage stays queryable through `provenance`.
- **Retire:** leave the spec and its records in place; add the rejection reason to the
  research note. Dead ideas are documentation.

---

## 4. Where hypotheses come from (priority order)

1. **Own telemetry** — read `outputs/experiments_summary.csv` + recent records:
   a strategy with high win rate but profit factor < 1 ⇒ exits too early (cycle_time
   hypothesis); high Sharpe but < 30 trades ⇒ loosen filters (alpha hypothesis).
2. **Live/backtest gap** — once the paper ledger exists (DESIGN.md Phase 7), every gap
   between simulated and paper results is automatically the top-priority hypothesis.
3. **Cross-sections of existing data** — unused columns are free alpha candidates:
   `delivery_pct`, `vwap` deviation, `oi` (F&O names) are already in every parquet.
4. **External research** (web) — last resort, max ~3 searches per session; distill into
   backlog JSON, never paste articles.

Replenishment rule: keep ≥ 5 `open` hypotheses in the backlog at all times.

---

## 5. Record schemas

### Experiment record (`data/results/experiments.jsonl`, one JSON/line, append-only)
```json
{"id": "exp_20260612_153000_rsi2_meanrev_v1", "ts": "2026-06-12T15:30:00",
 "kind": "backtest", "spec_id": "rsi2_meanrev_v1", "strategy": "RSI2 Mean Reversion v1",
 "params": {"rsi_buy": 10, "trend_span": 100},
 "universe": "halal", "n_symbols": 2169, "market": "nse",
 "window": ["2025-06-12", "2026-06-12"],
 "costs": {"round_trip_pct": 0.25}, "fill": "next_bar_close",
 "metrics": {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy_pct": 0.0,
              "sharpe_approx": 0.0, "max_drawdown_pct": 0.0, "total_pnl_pct": 0.0,
              "avg_hold_days": 0.0},
 "hypothesis_id": null, "verdict": null, "runtime_s": 0.0}
```

### Hypothesis record (`data/research/backlog.jsonl`) — see §3.1.

Conventions: ids are `exp_<ts>_<spec_id>` / `h_<seq>`; timestamps ISO-8601 local;
files are append-only (corrections are new records, never edits).

---

## 6. Operating cadence

| When | What | Who |
|---|---|---|
| Daily | `bulk_fetch.py` incremental update (new bhav copies → parquets) | scheduled task |
| Per idea | design spec → `run_experiments.py` → verdict | runner (deterministic) |
| Weekly | read summary, write research note, replenish backlog, refine winners | agent session |
| Monthly | re-evaluate the full leaderboard on fresh data; demote stale winners | runner + agent |

**Safety rails:** the loop may create specs and update the leaderboard. It may **not**
change capital/risk caps in `src/config.py`, delete data or any record, or enable live
trading. Those need the human.

---

## 7. Current implementation status (2026-06-12)

- [x] Spec format + interpreter (`strategy-spec/v1`)
- [x] Bulk halal-universe data store (1 year NSE bars)
- [x] Experiment runner with costs, next-bar fill, JSONL records
- [x] First experiment batch: 10 strategy specs evaluated (see `docs/research/`)
- [x] Leaderboard builder + dashboard "Strategy Lab" section (`outputs/leaderboard.json`)
- [ ] `--grid` param sweeps wired to `param_grid`
- [ ] Walk-forward split (train/test) in the runner
- [ ] Regime-sliced metrics (needs `src/context.py`, DESIGN.md Phase 2)
- [ ] Backlog auto-verdict (`--hypothesis` flag)
- [ ] Paper-trade ledger feedback loop

---

## 8. Book-seeded research backlog (top 20 ideas)

Distilled from the library reviewed in `docs/knowledge_index.md` and filtered to
**cash-equity swing** (no futures/options). Each idea names its source, the concrete
rule/feature to implement against our `OHLCV + delivery% + (sparse) oi` parquets,
and a falsifiable `expected_metric` in the house style (P&L-first: profit factor,
expectancy %, max DD, ≥100-trade floor). Ideas are grouped: **A. Alpha signals**,
**B. Features/labelling**, **C. Risk & sizing**, **D. Validation/methodology**.
Promote a winner per §3.4; record under a new `h_0NN` id when picked up.

### A. Alpha signals (new or refined specs)

1. **TTM Squeeze breakout** *(Carter, Mastering the Trade Ch.11, p245)* — fire long
   when Bollinger Bands close back outside Keltner Channels after a squeeze (BB width
   < KC width for ≥N days), in the direction of the momentum oscillator. Refines our
   weak `bb_squeeze_breakout_v1` by adding the Keltner-containment gate.
   *Metric:* PF > 1.2, expectancy > 0.4%, trades ≥ 100.

2. **Double-bottom / Big-W reversal with measured-move target**
   *(Bulkowski Ch.7; Edwards & Magee Ch.6)* — detect W base off a 60d low, enter on
   neckline reclaim, target = pattern height, stop below the second low.
   *Metric:* win-rate > 50%, PF > 1.2, avg R:R realised ≥ 1.3.

3. **Flag / pennant continuation after a delivery-confirmed thrust**
   *(Edwards & Magee Ch.11; Bulkowski)* — a >X% impulse on ≥1.5× delivery, then a
   3–8 bar low-range consolidation, enter on consolidation-high break. Continuation
   companion to `delivery_breakout_v3`. *Metric:* PF > 1.3, trades ≥ 100, DD < 12%.

4. **Multiple-timeframe momentum alignment** *(Miner Ch.2)* — only take a daily
   momentum long when the weekly (resampled) momentum is also up. Add a `mtf_bull`
   gate feature; apply to the momentum family. *Metric:* PF improvement ≥ 0.1 vs
   parent at ≤ ½ trade count.

5. **HOLP/LOHP reversal trigger** *(Carter Ch.17, p347)* — after a multi-day
   decline, enter on the first bar that takes out the High Of the Low Period; stop at
   the LOLP. Cleaner entry than fixed oversold thresholds (cf. `oversold_snapback_v1`).
   *Metric:* win-rate > 55%, PF > 1.2.

6. **Relative-strength-vs-index long-short ranking** *(Jansen Ch.11; Lloyd Ch.1)* —
   rank the universe by RS vs the equal-weight/NIFTY benchmark; long the top decile.
   Extends our `rs_rank` feature with an explicit benchmark-relative numerator.
   *Metric:* top-decile PF > 1.3 and monotone PF across deciles (signal is ordinal).

7. **Failed-breakout fade** *(Grimes Ch.5, p118)* — when a breakout above a 20d high
   reverses back inside within 1–2 bars on rising volume, fade it (short-side skipped
   for cash; instead use it as a *veto* on long breakouts). Cuts `delivery_breakout`
   false positives. *Metric:* breakout-spec PF rises ≥ 0.1 with veto applied.

8. **OBV / Chaikin Money Flow accumulation divergence** *(Lloyd Ch.2; Successful
   Stock Signals)* — price makes a lower low while OBV/CMF makes a higher low →
   accumulation; enter on confirmation bar. Uses volume we already store.
   *Metric:* PF > 1.2, trades ≥ 100.

9. **Pullback-to-rising-MA in confirmed uptrend** *(Grimes Ch.3/Ch.6; Bulkowski
   "knots" p44)* — in an EMA50-up regime, buy the first close that reclaims the
   prior bar's high after a 3–5 bar pullback to the rising 20-EMA. Sharper trigger
   for `pullback_trend_v1` (which over-fired at 13k trades). *Metric:* trades < 3k,
   PF > 1.2.

10. **Gap-up continuation, taxonomy-filtered** *(Edwards & Magee Ch.12)* — classify
    the opening gap (breakaway vs exhaustion) using prior-trend position; only trade
    breakaway gaps that hold above the prior high to close. Refines
    `gap_up_continuation_v1`. *Metric:* win-rate > 50%, PF > 1.2.

### B. Features & labelling

11. **DVAR candlestick volatility/direction feature** *(Xie et al. Ch.5–6, p44)* —
    decompose each candle into body/upper-shadow/lower-shadow ratios and build the
    DVAR direction-of-volatility score; add as a regime/confirmation feature.
    *Metric:* adding the feature lifts a host spec's PF ≥ 0.05 (ablation test).

12. **Triple-barrier labelling for exit-grid search** *(López de Prado Ch.3)* —
    relabel every entry with profit-take / stop-loss / max-hold barriers and sweep
    the barrier multiples to find the exit grid that maximises expectancy. Directly
    informs our `exit` block defaults. *Metric:* discovered barrier set raises
    portfolio expectancy ≥ 0.1% vs current fixed exits.

13. **Meta-labelling filter on an existing spec** *(López de Prado Ch.3)* — train a
    simple classifier on `{spec signal fired}` to predict win/loss using available
    features (delivery%, rs_rank, vol regime), then trade only high-precision
    signals. *Metric:* precision-gated subset PF ≥ parent PF + 0.2 at ≥ 100 trades.

14. **Alpha-factor IC screen for new columns** *(Jansen Ch.4, p110)* — compute the
    information coefficient (rank-corr of factor vs forward N-day return) for each
    candidate feature (delivery%, OBV slope, RS, ATR%, gap%) before building a spec.
    *Metric:* shortlist features with |IC| > 0.03 and stable sign across sub-periods.

15. **Volatility-regime gate via realised ATR%** *(Candlestick Ch.7; Kissell Ch.6)* —
    suppress new entries when universe-median ATR% is in its top quartile (chop/panic).
    Complements the NIFTY EMA regime gate. *Metric:* gated variant DD falls ≥ 2pts
    with PF delta ≥ −0.02.

### C. Risk & position sizing

16. **ATR / R-multiple volatility sizing** *(Grimes Ch.9; Miner Ch.6)* — replace the
    flat ₹100k-per-trade with size = risk-budget ÷ (entry − ATR-stop), so each trade
    risks a constant fraction. *Metric:* same gross expectancy at lower DD; Sharpe up.

17. **Structural stop placement at swing/support** *(Edwards & Magee Ch.13/Ch.27;
    Aziz Ch.4)* — set stops just beyond the nearest swing low / support level instead
    of a fixed %; trail to the prior swing low as structure builds. *Metric:* PF up
    and avg-loss% down vs fixed-stop baseline on the same signals.

18. **Portfolio heat / correlation cap** *(Edwards & Magee Ch.42; Kissell Ch.10)* —
    cap simultaneous open risk and limit concurrent positions in one sector/cluster
    (cluster via Jansen Ch.13 unsupervised grouping). Moves us off the "no
    concurrency cap" pooled assumption toward a realistic book. *Metric:* portfolio-
    level DD falls with total-return delta ≥ −5%.

### D. Validation & methodology (system-level)

19. **Walk-forward + deflated/bootstrap Sharpe gate** *(López de Prado Ch.3 §
    Backtesting; Masters Ch.5)* — split each backtest into rolling train/test folds;
    promote only specs whose **test-window** PF ≥ 0.5× train and whose bootstrap PF
    CI lower bound > 1.0. Implements open hypothesis `h_018`. *Metric:* the batch-3
    promotion candidates survive; document those that don't.

20. **Monte-Carlo permutation significance test** *(Masters Ch.7, p287)* — for each
    promotion candidate, shuffle the signal/return pairing M times and compute the
    null PF distribution; require the real PF to beat the 95th percentile. Guards
    against the multiple-testing inflation our wide spec search invites (cf. §3.4
    anti-overfitting bar). *Metric:* candidate p-value < 0.05 vs permuted null.

> Sourcing note: ideas 1–10 map to Tier-1 price/structure books; 11–15 to ML feature
> & labelling references; 16–18 to risk chapters; 19–20 to validation references.
> Full chapter/page provenance is in `docs/knowledge_index.md`. Avoid futures/options
> constructs throughout — cash-equity, multi-day holds only.
