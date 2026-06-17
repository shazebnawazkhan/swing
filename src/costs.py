"""
costs.py
--------
Liquidity-aware round-trip transaction cost, replacing the flat 0.25%.

Motivation (Kissell, *Algorithmic Trading Methods*, ch.3–4 & 10 — Transaction
Costs / Market Impact / I-Star):  a flat per-trade cost flatters illiquid,
volatile names — exactly the small/mid-caps a wide NSE swing scan surfaces.
Batch-1's own conclusion #3 was "costs are decisive at this horizon", so the
cost model deserves to be honest about *size relative to liquidity* and
*volatility*, the two drivers of market impact.

Model (a transparent, **uncalibrated** square-root impact law in the spirit of
Kissell's I-Star and Almgren):

    one_way_impact_pct = COEF * sigma_daily_pct * sqrt(order_value / ADV_value)
    round_trip_pct     = BASE_FEE_PCT + 2 * one_way_impact_pct      (clamped)

  · BASE_FEE_PCT  — brokerage + STT + exchange + stamp + GST, round trip (fixed).
  · sigma_daily_pct — trailing daily-return volatility at entry, in percent.
  · order_value / ADV_value — participation: how big the ₹ order is vs the
    name's average daily ₹ traded.  sqrt() is the standard concave impact shape.

Calibration honesty: COEF is a literature-style choice (not fit to NSE fills),
tuned so a liquid large-cap costs ≈ the old flat 0.25% or less, while a volatile
thinly-traded name costs materially more.  It is a *relative* penalty, not a
precise TCA forecast — documented as such in every experiment record.
"""

from __future__ import annotations

import numpy as np

# ── Parameters (single home; mirrors the config-as-risk-home convention) ──────────
BASE_FEE_PCT = 0.15        # fixed round-trip statutory + brokerage cost, %
IMPACT_COEF = 0.5          # coefficient on daily-vol × sqrt(participation), one-way
MIN_ROUND_TRIP_PCT = 0.15  # floor: never cheaper than statutory costs
MAX_ROUND_TRIP_PCT = 3.0   # cap: guards against near-zero-ADV blow-ups


def istar_round_trip_pct(
    order_value: float | np.ndarray,
    adv_value: float | np.ndarray,
    sigma_daily_pct: float | np.ndarray,
) -> float | np.ndarray:
    """
    Liquidity-aware round-trip cost in percent.  Scalar or vectorised.

    order_value     : ₹ value of the position (pos_size).
    adv_value       : ₹ average daily traded value of the name at entry.
    sigma_daily_pct : trailing daily-return volatility at entry, in percent.
    """
    adv = np.asarray(adv_value, dtype=float)
    sig = np.asarray(sigma_daily_pct, dtype=float)
    ov = np.asarray(order_value, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        participation = np.where(adv > 0, ov / adv, np.nan)
    one_way = IMPACT_COEF * sig * np.sqrt(np.clip(participation, 0, None))
    rt = BASE_FEE_PCT + 2.0 * one_way

    # NaN participation (no ADV) → treat as maximally illiquid
    rt = np.where(np.isnan(rt), MAX_ROUND_TRIP_PCT, rt)
    rt = np.clip(rt, MIN_ROUND_TRIP_PCT, MAX_ROUND_TRIP_PCT)
    return float(rt) if rt.ndim == 0 else rt
