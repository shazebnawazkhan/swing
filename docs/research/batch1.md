# Research Note — Batch 1 & 2: First 15 Spec-Strategy Experiments

**Date:** 2026-06-12 · **Hypotheses:** h_001–h_010 (batch 1), h_011 (batch 2 refinements)
**Setup:** halal universe (1,792 tradable symbols after liquidity/history filters),
window 2025-06-11 → 2026-06-11, 0.25% round-trip costs, next-bar fill, pooled
₹100k/trade semantics. Records: `data/results/experiments.jsonl` (`exp_20260612_*`).

## Market context (critical for reading the results)

The test year was a **bear market for this universe**: median stock **−14.8%**,
only 31% of stocks positive, only 26% above their EMA200 at window end
(`tools/universe_check.py`). Every long-only result below must be read against
that baseline — random long entries lost money this year.

## Batch 1 — ten new strategies (all REJECTED for this window)

| Spec | Trades | Win% | PF | Expectancy | Verdict |
|---|---|---|---|---|---|
| delivery_breakout_v1 | 333 | 39.9 | **0.97** | −0.12% | best of batch; ~breakeven in a bear year |
| high_momentum_52w_v1 | 891 | 40.3 | 0.93 | −0.29% | second-best; selective by construction |
| rsi2_meanrev_v1 | 7,938 | 43.8 | 0.88 | −0.28% | highest win rate; bleeds via 5% stops |
| gap_up_continuation_v1 | 7,999 | 40.0 | 0.86 | −0.38% | gaps faded in weak tape |
| ema_ribbon_cross_v1 | 7,108 | 37.1 | 0.83 | −0.58% | crossovers whipsawed |
| delivery_absorption_v1 | 29,391 | 37.7 | 0.77 | −0.73% | fires far too often; catching knives |
| pullback_trend_v1 | 13,426 | 35.9 | 0.75 | −0.73% | "uptrends" too weak (EMA20>EMA50 only) |
| vwap_reclaim_delivery_v1 | 25,170 | 36.6 | 0.72 | −0.90% | too-frequent low-quality trigger |
| oversold_snapback_v1 | 424 | 36.6 | 0.61 | −1.38% | falling knives kept falling |
| bb_squeeze_breakout_v1 | 2,001 | 28.9 | 0.56 | −1.82% | worst; bear-market breakouts fail |

## Batch 2 — trend-gated refinements (h_011: PARTIALLY CONFIRMED)

| Spec (change vs parent) | Trades | PF | ΔPF | Max DD | Verdict |
|---|---|---|---|---|---|
| high_momentum_52w_v2 (+delivery confirm) | 355 | 0.95 | +0.02 | 18.2% (was 65.8%) | best overall candidate |
| rsi2_meanrev_v2 (RSI<5, EMA200, 4d exit) | 6,112 | 0.87 | −0.01 | 141% (was 187%) | quicker exits ≈ same edge |
| pullback_trend_v2 (+EMA50>EMA200) | 12,232 | 0.76 | +0.01 | 395% | still fires too often |
| delivery_breakout_v2 (+EMA200, tight exits) | 249 | 0.82 | −0.15 | 29.5% | gate *hurt* PF — the v1 edge wasn't trend-dependent |
| gap_up_continuation_v2 (+EMA200) | 4,588 | 0.69 | −0.17 | 486% | gate hurt — late-stage uptrends gapped & failed |

**h_011 verdict:** stock-level trend gates reliably cut drawdown (2–3×) and trade
count, but do **not** flip expectancy positive in a bear year. For two specs the
EMA200 gate selected late-stage uptrends and *reduced* PF.

## Learnings → next hypotheses

1. **The missing control is market-level, not stock-level.** Stock EMA200 gates
   help risk but can't fight a falling index. Build the regime detector
   (DESIGN.md Phase 2) and test an index-level gate: no new longs when
   NIFTY < its 50-EMA (→ h_012).
2. **Selectivity correlates with survival.** The three best specs (PF ≥ 0.93)
   are the three most selective (≤ 900 trades/yr). High-frequency triggers
   (25k+ trades) are noise plus costs (→ h_014 relative-strength filter to cut
   trade counts further).
3. **Costs are decisive at this horizon.** Several specs are positive pre-cost
   (expectancy > −0.25%) — strategies must clear ~0.3%/trade to be real.
4. delivery_breakout remains the most promising *family* — consistent with the
   project's original delivery+OI edge. Refine selectivity (volume + delivery
   double-confirm), not trend gates (→ h_013).
5. Pooled max-DD% is misleading for high-frequency specs (denominator is ₹1M
   while exposure is uncapped) — portfolio-level simulation with concurrency
   caps (DESIGN.md Phase 4) needed before any capital conclusions.

## Status changes

- h_001–h_010 → **rejected** (this window; specs retained for regime-sliced re-test)
- h_011 → **inconclusive** (risk ↓ confirmed, profitability ✗)
- New: h_012 (index regime gate), h_013 (delivery double-confirm breakout),
  h_014 (60d relative-strength filter), h_015 (bear-regime cash rule = trade
  only sideways/bull regimes once context engine exists)
