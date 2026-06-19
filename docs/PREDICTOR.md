# Next-Day Signal Predictor — Design Document

**Audience:** A Claude Code agent (Sonnet 4.6 / Opus) building this module incrementally.
**Repo:** `D:\SNK\codes\repos\swing` — extends the existing swing system; **never rewrites** it.
**Companion docs:** `docs/DESIGN.md` (system architecture & contracts), `docs/program.md`
(autoresearch loop — §12 of *this* doc adds the predictor experiment track there),
`docs/strategy_format.md` (rule-spec format the predictor's candidate generator reuses).

> **What this module is.** A *standalone machine-learning predictor* that studies the
> latest data and emits, for **next session**, a ranked BUY/SELL list across the whole
> universe — with an interactive webpage to inspect which signals fired for which stock
> under which model. It is **additive** to the rule-based Strategy Lab: the specs remain
> a candidate generator and a benchmark; the predictor is a second, independent signal
> engine governed by the same costs, metrics, results store, and safety rails.

---

## 0. Locked design decisions (do not re-litigate)

| # | Decision | Choice | Consequence |
|---|---|---|---|
| 1 | Prediction target | **Two heads**: `dir1d` (P next-day up) + `swing` (P triple-barrier win over H≈10d) | Two labels, two models, two columns on the page; a signal is *strong* when both agree |
| 2 | Role vs rule-specs | **Standalone predictor** over the full feature set across all stocks | Its own daily product; specs are a baseline, not a gate |
| 3 | Data sources | daily+delivery/OI **(core, required)**; news+macro; fundamentals+ratings/targets; intraday 1–5m; order-wins | Each is an independent adapter that **degrades gracefully** (missing feature ⇒ null, never blocks a signal) |
| 4 | Model family | **LightGBM + logistic baseline** per head now; deep learning is an autoresearch track | Robust on ~10 months of partly-sparse data; baseline proves the GBT earns its complexity |
| 5 | Validation | **Train 12→2 months ago, validate last 2 months**, with purge+embargo; rolling walk-forward is the robust upgrade (§7) | No lookahead; calibrated, out-of-sample probabilities only |

**Reality flags on data (architect for these, don't fight them):**
- Free NSE **intraday history barely exists** — yfinance gives ~60 days of 5-min for `.NS`
  tickers and is flaky; full history needs a paid/broker feed. Intraday is **optional,
  off by default**, used only for entry-timing features, never for the core daily label.
- **Fundamentals / analyst ratings / target prices / order-wins** have no clean free API —
  scraped, gap-filled, often stale. Every such feature ships with a `*_age_days` companion
  column and is **staleness-gated** before it reaches a model.

---

## 1. Reuse map — what this module stands on (do NOT rebuild)

| Need | Existing asset | How the predictor uses it |
|---|---|---|
| Price+delivery+OI bars | `data/fetched/<SYMBOL>.parquet`, `scripts/bulk_fetch.py` | Core feature source; offline, cached |
| Fast indicators | `src/fast_indicators.py` | Vectorised EMA/RSI/ATR/rolling → features; **add new indicators here, not in loops** |
| Costs | `src/costs.py` / `ROUND_TRIP_COST_PCT` | Net-of-cost economic scoring of the daily basket |
| Trade metrics | `src/metrics.py` | Profit factor, Sharpe, max DD of the predictor basket |
| Triple-barrier sim | `scripts/run_experiments.py` (numba sim) | Generates the `swing` head's labels (TP/SL/time-stop over H) |
| Rule specs | `data/strategies/*.json`, `src/strategies/spec.py` | `spec_fired_*` boolean features + a benchmark basket |
| Results store | `data/results/experiments.jsonl` | Predictor runs are experiments too (same schema, `kind:"predict"`) |
| Server + page shell | `scripts/server.py`, report builders in `scripts/` | New `/predictor` route + a new HTML page reuse the existing chart/style stack |
| Regime/benchmark | `src/benchmark.py` | `regime` + index-relative features; suppress signals in adverse regime |

**Inherited principle (keep it):** *data acquisition* ↔ *feature build* ↔ *labelling* ↔
*model* ↔ *signal emission* ↔ *UI* are decoupled stages, each with a file contract (§11).

---

## 2. Pipeline overview

```
                         ┌──────────── AUTORESEARCH (program.md §9) ───────────┐
                         │  feature/data/model hypotheses ► run ► verdict      │
                         │  ▲ telemetry (IC, AUC, basket P&L, data cost)  │    │
                         └──┼──────────────────────────────────────────────┼───┘
                            │                                              ▼
┌──────────────┐  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ DATA ADAPTERS│─►│ FEATURE STORE  │─►│  LABELLER    │─►│  MODELS      │─►│ DAILY SIGNALS│
│ daily/deliv  │  │ point-in-time  │  │ head A dir1d │  │ LGBM dir1d   │  │ rank top-N   │
│ news/macro   │  │ panel, sparse- │  │ head B swing │  │ LGBM swing   │  │ size + risk  │
│ funda/ratings│  │ tolerant, lag- │  │ (triple-     │  │ + logistic   │  │ check        │
│ intraday/OW  │  │ safe, dated    │  │  barrier)    │  │ baseline     │  │ → JSON + UI  │
└──────────────┘  └────────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
   adapters/         data/features/      labels in-mem      models/<head>/    predictor.html
```

Each stage writes a versioned artifact so any stage can be rerun/compared in isolation.

---

## 3. Data layer — adapters (each independent, graceful-degrade)

All adapters live in `src/predictor/adapters/` and obey one contract:
`fetch(symbols, start, end) -> DataFrame[date, symbol, <feature cols>, <col>_age_days]`,
write to `data/features/<source>/<...>.parquet`, and append a telemetry row
`(source, rows, seconds, bytes, stale_pct)` to `outputs/system_health.json`.

| Adapter | File | Source | Tier | Notes |
|---|---|---|---|---|
| Price/delivery/OI | reuse `bulk_fetch.py` | NSE archives (cached) | **core** | required; offline |
| Macro | `macro_adapter.py` | FRED csv + `data/macro/india_manual.csv` | core+ | staleness-flagged; DESIGN.md Phase 1 |
| News + sentiment | `news_adapter.py` | Google News RSS, keyword-scored | best-effort | no LLM calls in pipeline |
| Order-wins | `orderwin_adapter.py` | NSE corp-announcements + news keyword class | best-effort | event flags: `bags order/LOI/contract worth ₹X`; own column, not generic news |
| Fundamentals | `fundamentals_adapter.py` | screener/StockEdge scrape, quarterly | best-effort | EPS surprise, rev growth, margins; quarterly cadence |
| Ratings/targets | `ratings_adapter.py` | broker notes via news/aggregators | best-effort | `target_vs_price`, rating-change events |
| Intraday | `intraday_adapter.py` | yfinance 5m (~60d) / broker API if keyed | **optional/off** | entry-timing features only; never the daily label |

**Point-in-time discipline (the single most important rule here):** every adapter row is
stamped with the date the info was *publicly available*, not the period it describes. A
Q1 result known on 2026-05-04 is `available_date=2026-05-04`; the feature store joins on
`available_date <= T` only. Quarterly/event data is **forward-filled with an age counter**,
never back-dated. Violating this manufactures lookahead and is the #1 way this module would
silently overstate accuracy.

---

## 4. Feature store

`src/predictor/features.py` → `build_panel(asof) -> data/features/panel_<asof>.parquet`.

A long panel keyed `(date, symbol)` assembling, as-of each date:

- **Price/vol/structure** (from `fast_indicators`): returns 1/3/5/10/20d, ATR%, RSI(2/14),
  distance to 20/50/200-EMA, 52w-high proximity, Bollinger %B, gap%, volume z-score.
- **Microstructure (NSE edge):** delivery%, delivery% vs 20d avg, delivery z-score,
  vwap deviation, OI change% (F&O names), OI z-score.
- **Cross-sectional:** RS-rank vs universe (60d), sector RS, percentile of each price
  feature within the day (rank-normalised, robust to regime drift).
- **Regime/macro:** NIFTY regime label, index-relative return, VIX/DXY/rates level + slope,
  macro-bias scalar — each with `*_age_days`.
- **Event/text:** news sentiment score, order-win flag (0/1, decaying over N days),
  rating-change flag, days-since-result, EPS-surprise bucket — all staleness-gated.
- **Spec context:** `spec_fired_<id>` booleans + count of specs firing (cheap, free — the
  rule layer becomes features for the standalone model).
- **Calendar:** day-of-week, days-to-expiry, pre/post-result window.

Rules: all features use **only `<= T` data** (lag with `shift(1)` where a same-day close is
used to predict T+1); missing ⇒ `NaN` (LightGBM-native, never imputed to a fake 0 that
means "real zero"); a `feature_manifest.json` records every column, its source, and lag.

---

## 5. Labelling — the two heads

`src/predictor/labels.py`, computed only on the **train/validate window**, never at inference.

- **Head A — `dir1d`** (binary): `1` if `close[T+1] > open[T+1]` (next-session up after a
  next-open fill), else `0`. Fast, dense, every row labelled. Sanity/short-horizon head.
- **Head B — `swing`** (the economic head): for an entry filled at `open[T+1]`, run the
  existing triple-barrier sim (TP = `target_pct`, SL = `stop_loss_pct`, time-stop =
  `max_hold_days`, all from `config`) over the next H bars. Label `1` if TP hit before SL,
  `0` otherwise (time-stop/SL). Sample weight ∝ realised holding return magnitude so the
  model focuses on economically meaningful wins (López de Prado meta-label weighting).

The **page surfaces both**; the default daily signal requires **agreement** (both heads
above their calibrated thresholds) — high precision over high recall, by design.

---

## 6. Models

`src/predictor/model.py` — one estimator per head, plus a baseline:

- **Primary:** `LightGBM` classifier per head. Handles sparse/missing features natively,
  trains in seconds on the panel, exposes feature importance + SHAP. Class imbalance via
  `scale_pos_weight`; probabilities **isotonic-calibrated** on the validation fold so
  `P(win)` means what it says (essential for ranking + sizing).
- **Baseline:** `LogisticRegression` on the same features (rank-normalised) — if the GBT
  can't beat the linear baseline's validation AUC + basket P&L, the GBT is fitting noise;
  that gate is enforced, not advisory.
- **Deep learning:** *not now.* Logged as autoresearch hypothesis (h_dl) to revisit once
  ≥2 years of history + live intraday features exist. Sequence models overfit <1y data.

Artifacts per run: `models/<head>/<run_id>/{model.txt, calibrator.pkl, features.json,
metrics.json, importance.csv}`. Models are **data** — reproducible from panel + config.

---

## 7. Training / validation protocol

The user's spec — **train [12mo ago → 2mo ago], validate [last 2mo]** — is the baseline
split. Implemented with two guards that make it honest:

1. **Purge + embargo:** drop training rows whose label horizon H overlaps the validation
   window (a swing label started in the train tail "sees" validation days); embargo a
   further H days. Without this the split leaks.
2. **Rolling walk-forward (robust upgrade, autoresearch-driven):** slide the 10mo-train /
   2mo-validate window monthly across all history; report mean ± dispersion of validation
   metrics. A model promoted on one lucky fold is the classic trap (§13).

**Promotion gate (a model becomes `live` and drives the page's default basket):**
validation `ROC-AUC ≥ 0.55` **and** beats the logistic baseline, `Brier` improves vs
base-rate, **and** the top-decile-by-`P(win)` basket shows net-of-cost `profit_factor ≥ 1.3`
over ≥100 validation trades with `max_DD ≤ 15%`. Economic test outranks the statistical one.

---

## 8. Daily inference → signals

`scripts/predict_daily.py` (offline-tolerant; mirrors `daily.py` UX):

1. Build `panel_<today>` from latest available data (skips unreachable adapters, logs gaps).
2. Load each head's `live` calibrated model; score every symbol → `P_up`, `P_win`.
3. Direction: **BUY** if both heads ≥ threshold; **SELL/AVOID** if both ≤ (1−threshold);
   else **HOLD/neutral**. Rank BUY candidates by `P_win` (tie-break `P_up`).
4. Risk pass (reuse `config` + portfolio rules): next-open entry, ATR/%-stop, size by
   risk-per-trade, sector/position caps; attach `expected_value = P_win·target − (1−P_win)·stop − costs`.
5. Emit `outputs/predicted_signals_<date>.json` (§11.2) + refresh the page; append a
   `kind:"predict"` record to the results store with the model run ids used.

Each emitted signal is also written to a **paper ledger** (`data/paper_trades.jsonl`) and
its eventual outcome filled from data — the live-vs-backtest gap is the richest source of
autoresearch hypotheses (§13). **No live execution** — explicitly out of scope; human-gated.

---

## 9. The webpage (requirement 2)

New page `outputs/predictor.html` + a `/predictor` route in `scripts/server.py`, reusing the
existing LightweightCharts/style stack from the comparison report. Sections:

- **Regime banner** — today's NSE regime, macro bias, data-freshness chips (which adapters
  are fresh/stale today, so the user trusts/distrusts features knowingly).
- **Signal table** — one row per recommended stock: symbol, direction, `P_up`, `P_win`,
  agreement badge, expected value, suggested entry/stop/target/size, sector, which
  `spec_*` also fired, top-3 SHAP feature contributions (why the model likes it).
  Filter by **strategy/model head**, sort by any column, search by symbol.
- **Per-stock drill-down** — reuse the candlestick + markers panel; overlay the feature
  values and the model's probability trajectory over recent days.
- **Model card** — per head: validation AUC/Brier/PR, calibration curve, feature
  importance, train/validate window, last-trained date, baseline delta.
- **Paper-ledger scoreboard** — realised hit-rate / P&L of past emitted signals vs
  predicted `P_win` (the calibration-in-the-wild view).

Static-export friendly (open the HTML directly) **and** live via the server; data comes
from the JSON contracts, so the page never recomputes models.

---

## 10. End goals & optimization metrics (requirement 7)

The module optimises **risk-adjusted net return of the daily top-N basket**, with statistical
and operational guardrails. Tracked in `outputs/system_health.json` and on the page.

| Class | Metric | Target (v1) | Why it's here |
|---|---|---|---|
| **Primary (economic)** | Net-of-cost **profit factor** of top-N daily basket | ≥ 1.3 | The product is the basket, not the AUC |
| | **Sharpe** of the basket equity curve | ≥ 1.0 | Risk-adjusted, after costs |
| | **Max drawdown** | ≤ 15% | Capital preservation |
| | **Precision@N** (top-N BUY hit-rate) | ≥ 55% | The user acts on the top of the list |
| **Predictive quality** | Validation **ROC-AUC** / **PR-AUC** | AUC ≥ 0.55, beats baseline | Signal exists & generalises |
| | **Brier / calibration error** | < base-rate Brier | `P(win)` must be trustworthy for sizing |
| | Feature **IC** (rank-corr vs fwd return) | shortlist \|IC\| > 0.03 | Gate features before they enter a model |
| **Operational** | **Capital cycle time** (signal→exit days) | track / minimise | User's turnover objective |
| | **Data cost** (sec + bytes per daily run) | track / minimise | Autoresearch trims useless adapters |
| | **Live-vs-backtest gap** (paper vs predicted) | → 0 | The honesty metric; gates any future live step |
| | **Coverage** (signals/day, % universe scored) | stable, non-degenerate | Guards against a model that fires never/always |

**Optimization target, stated plainly:** maximise the validation-window, net-of-cost Sharpe
of the agreement-basket, subject to (calibration error < base-rate) and (coverage within
[5, 50] signals/day). Everything in §12's autoresearch backlog is a hypothesis about moving
*this* number.

---

## 11. Contracts (stable interfaces)

### 11.1 Feature panel (`data/features/panel_<asof>.parquet`)
```
date, symbol, market,                         # keys
<feature columns…>,                           # §4, NaN-tolerant
<feature>_age_days for every event/funda col, # staleness
asof (the build date)                         # provenance
```
A sibling `feature_manifest.json`: `{col: {source, lag_bars, kind, added_in_run}}`.

### 11.2 Predicted signal (`outputs/predicted_signals_<date>.json`)
```json
{"date": "2026-06-19", "model_runs": {"dir1d": "run_…", "swing": "run_…"},
 "regime": {"nse": "bull"}, "data_freshness": {"news": "fresh", "fundamentals": "stale_31d"},
 "signals": [
   {"symbol": "INFY", "market": "nse", "direction": "BUY", "rank": 1,
    "p_up": 0.63, "p_win": 0.71, "agree": true, "expected_value_pct": 1.8,
    "entry_type": "next_open", "stop": 0.0, "target": 0.0, "shares": 0, "risk_inr": 0.0,
    "sector": "IT", "specs_fired": ["rs_momentum_v1"],
    "top_features": [["delivery_z", 0.21], ["rs_rank", 0.14], ["news_sent", 0.08]]}]}
```

### 11.3 Model metrics (`models/<head>/<run_id>/metrics.json`)
```json
{"head": "swing", "run_id": "run_20260619_…", "trained": "2026-06-19",
 "train_window": ["2025-06-19","2026-04-19"], "valid_window": ["2026-04-19","2026-06-19"],
 "auc": 0.0, "pr_auc": 0.0, "brier": 0.0, "baseline_auc": 0.0,
 "basket": {"profit_factor": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0, "precision_at_n": 0.0,
            "n_trades": 0}, "status": "live|candidate|rejected", "n_features": 0}
```

### 11.4 Predictor experiment record — reuses `data/results/experiments.jsonl` with
`kind:"predict"`; `params` carries `{head, feature_set_id, model, label_cfg}`; `metrics`
carries the §11.3 numbers; `hypothesis_id` ties to the backlog. **Same store, queryable
alongside rule-spec experiments.**

---

## 12. Phase plan (one phase ≈ one session ≈ one commit `Predictor PN: …`)

Each phase ends with a **command you run** and a measurable result; don't start the next
until the prior acceptance test passes. Build on cached data first — zero network.

| Phase | Deliverable | Acceptance test |
|---|---|---|
| **P0 Scaffold + labels** | `src/predictor/` pkg; `labels.py` (both heads) on cached parquets; manifest of label distribution | `python -m src.predictor.labels --summary` prints class balance for both heads over the cached window |
| **P1 Feature store** | `features.py` building the point-in-time panel from **core** sources only; `feature_manifest.json` | `python -m src.predictor.features --asof <date> --summary` → panel shape, null%, no lookahead (spot-check a row's dates) |
| **P2 Models + split** | `model.py` LightGBM + logistic baseline; train 12→2 / validate-2mo with purge+embargo | `python scripts/train_predictor.py --head swing` writes `metrics.json`; AUC printed; GBT vs baseline delta shown |
| **P3 Daily inference** | `predict_daily.py` → `predicted_signals_<date>.json` + paper-ledger write | `python scripts/predict_daily.py --asof <cached date>` emits ranked JSON; ≤20-line summary of top signals |
| **P4 Webpage** | `predictor.html` + `/predictor` route; signal table, model card, drill-down | open page / hit route → table renders from the JSON, filter by head works, SHAP reasons show |
| **P5 Walk-forward + promotion** | rolling WF eval; promotion gate (§7) wired; `kind:"predict"` records | `python scripts/eval_predictor.py` runs WF, updates model `status`, prints promoted/rejected diff |
| **P6 Extra adapters** | news/macro/order-win/fundamentals/ratings adapters, **ablation-gated** | each adapter run shows it lifts validation AUC **or** basket PF by the gate, else stays off (recorded) |
| **P7 Intraday (optional)** | `intraday_adapter.py` + entry-timing features, off by default | best-effort fetch + one ablation; documented as optional |
| **P8 Autoresearch wiring** | predictor hypotheses in backlog; loop can run a predictor experiment end-to-end | `python scripts/research.py --one-cycle` can execute a `predict` hypothesis and record a verdict |

---

## 13. Autoresearch integration (requirement 3 & 7)

The predictor plugs into the **existing** loop in `docs/program.md` (now extended with a
"Predictor experiment track" section) — same backlog, same results store, same deterministic
runner, **no LLM calls in the loop**. New `test_plan` actions the runner learns to interpret:
`train_head`, `ablate_feature`, `add_adapter`, `sweep_label`, `walkforward_predict`.

This is exactly the "expand the horizon of data" mechanism asked for: the loop proposes a
new data source / feature / label horizon as a hypothesis, the runner trains+validates it,
the **ablation gate** (does it move validation AUC *or* basket PF past the bar?) decides
whether it stays — so the feature set grows *only* by evidence. Seed hypotheses are added to
`data/research/backlog.jsonl` and listed in `program.md`.

---

## 14. Pitfalls checklist (audit every predictor run)

- [ ] **Lookahead via labels** — labels computed only on train/validate window; features lag
      `<= T`; swing horizon **purged+embargoed** out of the train tail.
- [ ] **Point-in-time leakage** — quarterly/event data joined on `available_date`, never the
      period it describes; forward-filled with an age counter, never back-dated.
- [ ] **Calibration drift** — re-fit isotonic on each validation fold; trust `P(win)` only
      after the calibration curve is checked on-page.
- [ ] **Survivorship** — `data/stocks.csv` is today's universe; keep backtests within the
      cached window and note the bias in every verdict.
- [ ] **Costs & fills** — every basket trade pays round-trip cost and fills at next open; an
      AUC win that dies after costs is not a win.
- [ ] **Multiple testing** — wide feature/label search inflates false winners; require the
      walk-forward (not single-split) result and the baseline-beat gate before promotion.
- [ ] **Degenerate coverage** — reject a model that fires on ~0% or ~100% of the universe
      regardless of AUC; the product is a usable shortlist.
- [ ] **Stale-feature trading** — staleness flags propagate to the page; a feature past its
      max age is dropped, not silently trusted.

---

## 15. Definition of done (module)

1. `python scripts/predict_daily.py` runs end-to-end offline-tolerant on cached data in
   < 3 min, emitting a ranked, risk-checked `predicted_signals_<date>.json`.
2. Both heads have a `live` model passing the §7 promotion gate on the user's 12→2 / 2mo
   split **and** a rolling walk-forward.
3. `predictor.html` shows today's signals, per-stock reasons, both model cards, and the
   paper-ledger scoreboard — filterable by strategy/head.
4. The autoresearch loop can run a `predict` hypothesis with zero tokens and record a
   verdict; backlog holds ≥5 open predictor hypotheses.
5. `system_health.json` trends the primary metrics (basket PF/Sharpe/DD, AUC, data cost,
   live-vs-backtest gap) and they render on the page.
6. **No live execution.** Paper ledger only; real trading stays human-gated, per DESIGN.md.
