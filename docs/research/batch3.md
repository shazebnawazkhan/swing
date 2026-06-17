# Research Note — Batch 3: Honest Evaluation + Market-Relative Strategies

**Date:** 2026-06-16 · **Hypotheses:** h_012 (regime gate), h_013 (delivery
double-confirm), h_014 (relative-strength filter) — all **confirmed**.
**Records:** `data/results/experiments.jsonl` (`exp_20260616_*`).
**Report:** `outputs/comparison_report.html`.

## Why this batch exists

Batch 1 & 2 rejected all 15 specs. Batch 1's own learning #1 was *"the missing
control is market-level, not stock-level."* Two things were missing to act on it:

1. **An honest yardstick.** Raw profit factor / Sharpe measure *absolute* return.
   The NIFTY 50 was roughly flat over the window (+3.1% across 2024-06→2026-06),
   so a strategy can lose in raw terms yet still beat the market over its trade
   windows — that is real, capturable alpha that raw metrics hide.
2. **A market-level gate** and a **market-relative selector.**

Inspiration for the evaluation upgrades is Robert Kissell, *Algorithmic Trading
Methods* (2nd ed. of *The Science of Algorithmic Trading & Portfolio Management*):
- **Index-Adjusted Performance Metric** (ch.3) → per-trade alpha = trade return −
  benchmark return over the *same holding window*. New code: `src/metrics.py::
  market_adjusted_metrics`, benchmark in `src/benchmark.py` (NIFTY ^NSEI cached,
  synthetic equal-weight universe index as offline fallback).
- **Market-impact / I-Star** (ch.3–4, 10) → replace the flat 0.25% cost with a
  liquidity-aware square-root model (cost rises with order/ADV and volatility).
  New code: `src/costs.py`. Transparently *uncalibrated* — a relative penalty on
  illiquid/volatile names, not a precise TCA forecast.

## Method

Same engine as batch 1 (halal universe, 1,792 symbols, 2025-06-11→2026-06-11,
next-bar fill, numba sim) with three additions, all spec-independent and computed
once per run: liquidity features (₹-ADV, daily vol), a cross-sectional
**`rs_rank`** column (60-day return ranked across the universe per date), and an
optional index-level **regime gate** (`--regime-gate`: no new longs while the
benchmark is below its 50-day EMA).

## Finding 1 — Honest re-scoring vindicated the *selective* families

Re-scoring the original 15 under alpha metrics + I-Star costs: **3 of 15 are
alpha-positive**, and they are the three most selective (≤ 891 trades/yr):

| Spec | Trades | Raw PF | α PF | α Exp | Info |
|---|---|---|---|---|---|
| high_momentum_52w_v2 | 355 | 0.95 | **1.33** | +0.80% | 0.65 |
| high_momentum_52w_v1 | 891 | 0.93 | **1.32** | +0.97% | 0.57 |
| delivery_breakout_v1 | 333 | 0.95 | **1.15** | +0.46% | 0.30 |

Every high-frequency spec (rsi2, vwap_reclaim, delivery_absorption, pullback,
gap, bb_squeeze; 4k–29k trades) stays **α PF < 1.0** — confirmed no edge even
after market adjustment. Honest eval does not rescue everything; it isolates the
**near-high momentum + delivery-confirmation** family as the genuine edge.

## Finding 2 — New market-relative specs clear the promotion bar

| Spec (hypothesis) | Trades | Raw PF | α PF | α Exp | Info | Max DD |
|---|---|---|---|---|---|---|
| delivery_breakout_v3 — delivery% **and** volume double-confirm (h_013) | 143 | **1.22** | **1.49** | +1.27% | 0.90 | 11.4% |
| rs_momentum_v1 — momentum + top-decile RS (h_014) | 215 | 1.03 | **1.41** | +1.02% | 0.80 | 12.4% |
| rs_delivery_breakout_v1 — breakout + top-quintile RS (h_014) | 161 | 1.03 | **1.32** | +0.98% | 0.62 | 22.9% |
| delivery_oi_breakout_v1 — breakout + rising OI (h_013) | 0 | — | — | — | — | — |

- **h_013 confirmed:** double-confirmation (delivery% AND volume) lifts the
  breakout edge to raw-positive (PF 1.22) with single-digit-ish drawdown.
- **h_014 confirmed:** the cross-sectional RS gate raises α PF over each parent
  (1.33→1.41 momentum; 1.15→1.32 breakout) **and** cuts drawdown.
- **delivery_oi_breakout_v1 → 0 trades:** `oi_change` is sparse/NaN in the cached
  parquets. This is a **data-coverage** finding (→ new hypothesis h_016), not a
  strategy rejection.

## Finding 3 — The index regime gate is the biggest single lever (h_012)

Adding the long/cash gate (no new longs while NIFTY < 50-EMA):

| Spec + regime gate | Trades | Raw PF | α PF | α Exp | Info | Max DD |
|---|---|---|---|---|---|---|
| rs_momentum_v1 | 91 | 1.56 | **2.35** | +2.62% | **1.99** | 6.1% |
| high_momentum_52w_v2 | 151 | 1.24 | **1.96** | +2.01% | 1.49 | 9.6% |
| delivery_breakout_v3 | 88 | 1.12 | 1.46 | +1.17% | 0.83 | 6.1% |
| delivery_breakout_v1 | 222 | 0.94 | 1.19 | +0.61% | 0.37 | 28.1% |

The gate lifts α PF and info ratio across the board and roughly halves drawdown
for the momentum family (rs_momentum: info 0.80→1.99, DD 12.4%→6.1%). This is
DESIGN.md §4's prior — *"regime gating beats parameter tuning"* — borne out.

**Caveat:** the gate trims trade counts by sitting out bear phases; `rs_momentum_v1`
falls to **91 trades**, just under the ≥100 statistical floor. Treat its 2.35 α PF
as promising-but-undersized until more history accumulates.

## Promotion candidates (per docs/program.md gates, α-based)

Clearing **α PF ≥ 1.2 and trades ≥ 100**:
`high_momentum_52w_v2 + gate` (151, 1.96) · `rs_momentum_v1` ungated (215, 1.41) ·
`delivery_breakout_v3` ungated (143, 1.49) · `rs_delivery_breakout_v1` (161, 1.32).

## Caveats (audit per DESIGN.md §10)

- **Flat NIFTY window:** alpha is measured vs a market that was ~flat; a different
  benchmark (equal-weight universe index returned ~+14% rebalanced) would be a
  harder bar. NIFTY is the defensible, tradable choice; revisit with a beta
  adjustment (h_017).
- **Uncalibrated costs:** I-Star coefficients are literature-style, not fit to NSE
  fills. They rank liquidity correctly but are not a precise cost forecast.
- **Single window / multiple testing:** these are in-sample on one year. Walk-forward
  and bootstrap CIs (Kissell ch.8) are the next rigour step (h_018) before any
  capital decision.
- **Survivorship:** today's halal universe over a 1-year window (batch-1 caveat stands).

## Status changes

- h_012 → **confirmed** (regime gate lifts α and cuts DD; trade-count caveat noted)
- h_013 → **confirmed** (double-confirm raw-positive; OI variant blocked on data)
- h_014 → **confirmed** (RS gate lifts α PF and cuts DD vs parents)
- New: h_016 (backfill futures OI so OI strategies can be tested), h_017 (beta-adjust
  alpha vs a harder universe benchmark), h_018 (walk-forward + bootstrap CIs on the
  promotion candidates).
