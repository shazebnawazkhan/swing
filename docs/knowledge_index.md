# Knowledge Index — Algo-Trading Library for Swing Strategy Research

A curated index into `D:\EBooks\AlgoTrading`, scoped to **cash-equity swing trading**
(multi-day holds on the NSE halal universe). It maps each usable book → chapter →
page → the concrete **algorithm / technique** that could seed a backtestable
hypothesis in this repo.

Scope rules applied:

- **Included:** technical price/volume structure, chart & candlestick patterns,
  cross-sectional/relative strength, mean reversion, breakout confirmation,
  entry/exit/position-sizing rules, feature engineering, labelling, backtest
  validation (walk-forward, permutation, bootstrap), and transaction-cost modelling.
- **Excluded / de-prioritised** (per the "avoid futures & options" instruction):
  derivatives pricing, options Greeks, futures spreads, market-making / HFT
  microstructure, and pure C++ implementation plumbing. See [§ Excluded](#excluded-or-out-of-scope) for why each was dropped.
- **Page numbers are physical PDF pages** (1-based, as the reader opens them),
  except where a book exposed only printed page numbers (noted inline). Use them as
  jump targets in any PDF viewer.

> How to use this with the autoresearch loop: each row's *Technique* is a candidate
> feature or rule. When a technique looks promising, file it as a hypothesis in
> `data/research/backlog.jsonl` and a spec in `data/strategies/`. The 20 seeded
> ideas in `docs/program.md` § "Book-seeded research backlog" already do this for the
> highest-signal rows.

---

## Tier 1 — Directly actionable for swing (price/volume structure & patterns)

### Encyclopedia of Chart Patterns — Thomas Bulkowski (3rd ed., 1315 pp)
The single most quantified pattern reference: every pattern ships with measured
break-even/failure rates and median move — i.e. ready-made priors and exit targets.

| PDF pg | Chapter / Section | Technique usable here | Swing fit |
|---|---|---|---|
| 35 | Ch.1 How to Trade Chart Patterns | Bottom-fishing, buy-the-dip, "knots & swing-trading pullbacks", throwback/pullback entries | Entry timing on existing specs |
| 44 | Ch.1 §Knots and Swing Trading Pullbacks | Pullback-after-breakout re-entry rule | Refines `delivery_breakout_v*` |
| 86 | Ch.6 Big M / Ch.7 Big W (double top/bottom) | Reversal-pattern detection + measured-move target | New mean-reversion specs |
| — | Pattern chapters (each: Tour / Identification / Statistics / Trading Tactics / Sample Trade) | Per-pattern **break-even failure rate** & **median rise/decline** statistics | Exit-target & stop priors |
| ~1180+ | Statistical summary / event patterns | Pattern performance ranked tables | Pattern selection by base rate |

### The Art and Science of Technical Analysis — Adam Grimes (424 pp)
The most "quant-compatible" discretionary book: defines structure precisely enough to code.

| PDF pg | Chapter | Technique | Swing fit |
|---|---|---|---|
| 40 | Ch.2 Market Cycle & the Four Trades | Wyckoff-style cycle phases → the four trade archetypes | Regime/state feature |
| 53 | Ch.3 On Trends | Trend definition via pullback depth & swing points; pullback entries | Trend filter, RS confirmation |
| 95 | Ch.4 On Trading Ranges | Range identification, failure tests at edges | Mean-reversion gating |
| 118 | Ch.5 Interfaces between Trends and Ranges | Breakout vs failed-breakout classification | False-breakout filter |
| 142 | Ch.6 Practical Trading Templates | Concrete entry templates (pullback, failure test, breakout) | Spec rule sources |
| 180 | Ch.7 Tools for Confirmation | Multiple-timeframe & momentum confirmation | Double-confirm features |
| 216 | Ch.8 Trade Management | Scaling, partial exits, trailing | Exit logic |
| 239 | Ch.9 Risk Management | R-multiple sizing, portfolio heat | Position sizing |
| 355 | App.B Moving Averages & MACD (deeper) | MA/MACD construction & pitfalls | Indicator hygiene |

### Technical Analysis of Stock Trends — Edwards, Magee & Bassetti (11th ed., 687 pp)
The canonical chart-pattern + Dow-theory text; Part II is explicitly tactical.

| PDF pg | Chapter | Technique | Swing fit |
|---|---|---|---|
| 90–169 | Ch.6–9 Reversal Patterns (incl. Triangles) | Head-&-shoulders, triangles, broadening tops | Reversal specs |
| 200 | Ch.11 Consolidation Formations | Flags, pennants, rectangles (continuation) | Continuation breakout |
| 220 | Ch.12 Gaps | Breakaway / runaway / exhaustion gap taxonomy | Gap-continuation spec |
| 238 | Ch.13 Support and Resistance | S/R level construction | Stop / target placement |
| 256–293 | Ch.14–15 Trendlines & Channels | Trendline & channel breaks | Trend entries |
| 402 | Ch.27 Stop Orders | Stop-placement discipline | Exit rules |
| 470 | Ch.36 Automated Trendline: the Moving Average | MA-as-trendline mechanics | MA gate |
| 540–553 | Ch.40–42 Capital use / Portfolio risk | Capital allocation, portfolio risk mgmt | Sizing & heat caps |

### Mastering the Trade — John F. Carter (3rd ed., 495 pp)
Source of the **TTM Squeeze** and several mechanical setups; Part 2 = setups.

| PDF pg | Chapter | Technique | Swing fit |
|---|---|---|---|
| 245 | Ch.11 The Squeeze | **Bollinger-inside-Keltner squeeze → momentum-fire breakout** | New squeeze spec (we already have `bb_squeeze_breakout_v1`) |
| 233 | Ch.10 Reverting to the Mean | Mean-reversion timing & profit-taking | Exit on snapbacks |
| 286 | Ch.12 Catching the Wave | Trend-following on any timeframe | Trend continuation |
| 347 | Ch.17 HOLP / LOHP | High-of-low-period / Low-of-high-period reversal triggers | Reversal entry trigger |
| 109 | Ch.5 Tools that predict the next move | Indicator toolkit & filters | Feature set |

### High Probability Trading Strategies — Robert Miner (290 pp)
Multi-timeframe momentum + structured entry/exit/sizing.

| PDF pg | Chapter | Technique | Swing fit |
|---|---|---|---|
| 23 | Ch.2 Multiple Time Frame Momentum | Dual-timeframe momentum alignment | MTF gate |
| 63 | Ch.3 Pattern Recognition for Trends & Corrections | Trend vs correction classification | Pullback entries |
| 153 | Ch.6 Entry Strategies and Position Size | Entry triggers + volatility-based sizing | Sizing rules |
| 177 | Ch.7 Exit Strategies and Trade Management | Multi-target scale-out, trailing | Exit logic |

### Successful Stock Signals — Thomas K. Lloyd (357 pp)
TA + fundamentals integration; chapters are organised by **signal type**.
(Printed page numbers shown — book has no PDF bookmarks.)

| Printed pg | Chapter | Technique | Swing fit |
|---|---|---|---|
| 1 | Ch.1 MAs + Relative Strength to Beat the Index | RS vs index, Keltner, mean reversion, breakout | **Cross-sectional RS vs benchmark** |
| ~14 | Ch.2 OBV, Accumulation/Distribution, Chaikin Money Flow, Pivot, P&F | Volume-flow accumulation signals | Delivery/flow features |
| ~? | Ch.3 Bottoms/Tops, Cycles, PPO | Cycle turning points, PPO | Reversal timing |
| ~? | Ch.4 Sell Signals | Systematic exit signals | Exit logic |
| ~? | Ch.6 Breakout Signals | Breakout confirmation | Breakout filter |
| ~? | Ch.7 RSI, Stochastic, MACD | Oscillator combos | Confirmation stack |
| ~? | Ch.9 Death Cross, Double Bottom, Bull Trap, Dead-Cat Bounce | Named reversal/trap patterns | False-signal filter |
| ~? | Ch.10 Gaps, Divergences, Breakdowns/Breakouts | Divergence detection | Momentum divergence feature |

### Candlestick Forecasting for Investments — Xie, Fan & Wang (133 pp)
Rare **academic/statistical** treatment of candlesticks (the DVAR model) — turns
candle shapes into a quantitative volatility/return forecaster.

| PDF pg | Chapter | Technique | Swing fit |
|---|---|---|---|
| 32–40 | Ch.3–4 Basic concepts & statistical properties | Candle decomposition (body/shadow) as data | Feature engineering |
| 44 | Ch.5 DVAR model | **DVAR: candlestick-based volatility/direction model** | Volatility-timing feature |
| 52 | Ch.6 Shadows in DVAR | Upper/lower shadow predictive content | Reversal feature |
| 64 | Ch.7 Market Volatility Timing | Candle-based vol regime timing | Regime gate input |
| 79–92 | Ch.8–9 Technical Range Forecasting & Spillover | Forecasting the day's range | Stop/target sizing |
| 93 | Ch.10 Stock Return Forecasting (S&P500) | Return-forecast application | Signal model template |

### Technical Analysis for Dummies — Barbara Rockefeller (387 pp)
Breadth reference for indicator definitions when seeding a new feature.

| PDF pg | Part | Technique | Swing fit |
|---|---|---|---|
| 117 | Part 2 Building Indicators from the Ground Up | Indicator construction (MAs, momentum, bands) | Feature reference |
| 173 | Part 3 Finding Patterns | Pattern catalogue | Pattern reference |
| 223 | Part 4 Dynamic Analysis | Combining indicators, confirmation | Multi-signal logic |

### Advanced Techniques in Day Trading — Andrew Aziz (385 pp)
Mostly intraday, but Ch.4–5 (S/R, price action, trade management) transfer to swing.

| PDF pg | Chapter | Technique | Swing fit (transferable) |
|---|---|---|---|
| 117 | Ch.4 Support and Resistance Levels | S/R construction & confluence | Stop/target placement |
| 156 | Ch.5 Price Action, Candlesticks, Trade Management | Candlestick reversal/continuation + trade mgmt | Entry/exit triggers |
| 309 | Ch.7 Risk and Account Management | Per-trade & account risk caps | Sizing discipline |

---

## Tier 2 — Quant methodology: features, labelling, validation, costs

### Machine Learning for Algorithmic Trading (2nd ed.) — Stefan Jansen (821 pp)
The most directly reusable quant cookbook for this repo's Python stack.

| PDF pg | Chapter | Technique | Swing fit |
|---|---|---|---|
| 110 | Ch.4 Financial Feature Engineering — Researching Alpha Factors | **Alpha-factor construction & evaluation (IC, factor decay)** | Rank/score features (rs_rank kin) |
| 150 | Ch.5 Portfolio Optimization & Performance Evaluation | Performance metrics, factor-based eval | Honest metrics |
| 176 | Ch.6 The Machine Learning Process | Train/validate/test discipline | Anti-overfit workflow |
| 250 | Ch.8 The ML4T Workflow — ML model → strategy backtest | End-to-end signal→backtest loop | Loop blueprint |
| 284 | Ch.9 Time-Series Models — Volatility & Stat-Arb | GARCH vol, cointegration pairs | Vol feature, pairs (cash) |
| 356 | Ch.11 Random Forests — Long-Short for Japanese stocks | **Cross-sectional ranking long-short** | Ranked-universe spec |
| 394 | Ch.12 Boosting Your Trading Strategy | Gradient boosting on factor sets | Meta-signal scorer |
| 436 | Ch.13 Data-Driven Risk Factors / Unsupervised | Clustering universe into risk factors | Sector/cluster neutrality |
| 754 | Alpha Factor Library (appendix) | Catalogue of coded alpha factors | Feature shopping list |

### Advances in Financial Machine Learning — Marcos López de Prado (393 pp)
The authority on **not fooling yourself** — labelling and backtest validation.

| PDF pg | Part / Chapter | Technique | Swing fit |
|---|---|---|---|
| 48 | Part 1 Data Analysis | Information-driven bars, fractional differentiation | Stationary features |
| ~70 | Ch.3 Labeling — Triple-Barrier Method | **Triple-barrier labelling (PT/SL/time) + meta-labelling** | Defines our exit grid as labels |
| 118 | Part 2 Modelling | Sample weights, sequential bootstrap | De-bias overlapping trades |
| 166 | Part 3 Backtesting | **Walk-forward, combinatorial purged CV, deflated Sharpe, backtest overfitting (PBO)** | Validation gate (h_018) |
| 274 | Part 4 Useful Financial Features | Structural breaks, entropy, microstructural features | New features |

### Testing and Tuning Market Trading Systems — Timothy Masters (325 pp)
Concise, math-first treatment of overfitting and forward-performance estimation. (C++, but the algorithms are language-agnostic.)

| PDF pg | Chapter | Technique | Swing fit |
|---|---|---|---|
| 19 | Ch.2 Pre-optimization Issues | Data hygiene, bias sources | Setup discipline |
| 42 | Ch.3 Optimization Issues | Parameter optimisation pitfalls | param_grid sweeps |
| 97 | Ch.4 Post-optimization Issues | Selection bias correction | Anti-overfit gate |
| 127 | Ch.5 Estimating Future Performance I — Unbiased Trade Simulation | **Unbiased OOS simulation** | Walk-forward runner |
| 198 | Ch.6 Estimating Future Performance II — Trade Analysis | Drawdown/return distribution analysis | Risk metrics |
| 287 | Ch.7 Permutation Tests | **Monte-Carlo permutation significance test** | Is the edge real? |

### Learn Algorithmic Trading — Donadio & Ghosh (378 pp)
Approachable bridge from TA rules to a Python backtester.

| PDF pg | Chapter | Technique | Swing fit |
|---|---|---|---|
| 52 | Ch.2 Deciphering Markets with Technical Analysis | Coded TA indicators | Feature impl reference |
| 115 | Ch.4 Classical Strategies Driven by Human Intuition | Momentum, mean-reversion, breakout coded | Spec templates |
| 144 | Ch.5 Sophisticated Algorithmic Strategies | Stat-arb, regression strategies | Advanced specs |
| 197 | Ch.6 Managing Risk of Algorithmic Strategies | Risk caps, drawdown control | Risk layer |
| 301 | Ch.9 Creating a Backtester in Python | Event-driven backtest design | Runner reference |

### Advanced Algorithmic Trading — Michael Halls-Moore (517 pp)
Bayesian + time-series + ML, applied to trading.

| PDF pg | Part | Technique | Swing fit |
|---|---|---|---|
| 84 | III Time Series Analysis | ARIMA, GARCH, cointegration, state-space/Kalman | Vol & pairs features |
| 230 | IV Statistical Machine Learning | Supervised models for signals | Signal scorer |
| 362 | V Quantitative Trading Techniques | Full strategy construction incl. mean-reversion & momentum | Spec templates |

### Successful Algorithmic Trading — Michael Halls-Moore (208 pp)
Lean end-to-end systems primer.

| PDF pg | Part | Technique | Swing fit |
|---|---|---|---|
| 22 | II Trading Systems | Strategy taxonomy & backtest mechanics | Runner concepts |
| 88 | IV Modelling | Forecasting & strategy modelling | Signal design |
| 116 | V Performance and Risk Management | Sharpe, drawdown, position sizing | Metrics & risk |

### Algorithmic Trading Methods (2nd ed.) — Robert Kissell (614 pp)
Already partially mined for the **I-Star cost model**. The execution/cost machinery.

| PDF pg | Chapter | Technique | Swing fit |
|---|---|---|---|
| 82 | Ch.3 Transaction Costs | Cost taxonomy & TCA | `src/costs.py` basis |
| 124 | Ch.4 Market Impact Models | **Square-root market-impact law** | I-Star cost (implemented) |
| 258 | Ch.10 Estimating I-Star Model Parameters | I-Star calibration | Cost calibration (h-future) |
| 294 | Ch.11 Risk, Volatility, and Factor Models | Vol & factor risk models | Risk features |
| 326 | Ch.12 Volume Forecasting Techniques | ADV / volume forecasting | Liquidity feature (_adv_inr) |

### The Science of Algorithmic Trading and Portfolio Management — Robert Kissell (474 pp)
Companion/predecessor; same cost spine plus portfolio construction.

| PDF pg | Chapter | Technique | Swing fit |
|---|---|---|---|
| 93 | Ch.3 Algorithmic Transaction Cost Analysis | Index-adjusted performance metric, TCA | Cost-aware eval |
| 135 | Ch.4 Market Impact Models | Impact model derivation | Cost model |
| 198 | Ch.6 Price Volatility | Volatility estimation methods | Vol feature |
| 240 | Ch.7 Advanced Forecasting Techniques | Return/vol forecasting | Signal models |
| 335 | Ch.10 Portfolio Construction | Constraint-aware portfolio build | Multi-name sizing |
| 365 | Ch.11 Quantitative Portfolio Management | Factor/risk-budgeted portfolios | Allocation layer |

---

## Excluded or out of scope

| Book | Reason |
|---|---|
| A Complete Guide to the Futures Market (Schwager) | Futures & options spreads — out of scope (cash equity only). Note: its TA/trend chapters overlap with included books. |
| Advanced positioning, flow & sentiment in commodity markets (Keenan) | Commodities / F&O flow; no usable bookmarks; out of scope. |
| Algorithmic and High-Frequency Trading (Cartea et al.) | HFT / market-making microstructure; sub-second horizon — irrelevant to multi-day swing. |
| Options and Derivatives Programming in C++ (Oliveira) | Options pricing + C++ plumbing. |
| Advanced Quantitative Finance with C++ (Pena); C++ for Quantitative Finance (Halls-Moore) | Derivatives-pricing C++ implementation. |
| Algorithmic Trading with Interactive Brokers (Scarpino) | Broker-API plumbing, not strategy. |
| Machine Learning in Finance (Dixon/Halperin/Bilokon) | Theory-heavy, derivatives/RL pricing focus; low direct swing yield. |
| How to Day Trade for a Living; Mastering the Trade Part 2 intraday plays | Intraday horizon (kept only the transferable swing/structure chapters above). |
| Trade like a Stock Market Wizard (Nauvall, 80 pp) | Thin introductory primer; concepts fully covered by Tier-1 books. |
| Preqin PME report, SC12_submission, executive briefing | Not trading-strategy material. |

---

*Generated 2026-06-16 from PDF bookmark/TOC extraction (PyMuPDF). Page numbers are
physical PDF pages unless marked "printed". Re-run the extraction if the library
changes.*
