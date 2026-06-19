"""
src.predictor.labels
---------------------
The two prediction heads (docs/PREDICTOR.md §5). Labels are computed ONLY on the
train/validate window and are strictly forward of the feature row, so a feature at
close[T] never sees its own label.

  dir1d : binary, 1 if close[T+1] > open[T+1]  (next session up after next-open fill)
  swing : binary, 1 if a next-open entry hits +target before -stop within H bars
          (triple barrier; time-stop / stop-first => 0). Carries `swing_w`, a sample
          weight ∝ |realised return|, so the model focuses on economically real wins.

Both use intraday high/low for honest barrier touches — not close-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _swing_one(df: pd.DataFrame, stop_pct: float, target_pct: float, max_hold: int):
    """
    Triple-barrier label per symbol. Entry filled at open[T+1] for a signal at T.
    Returns (label, weight, fwd_ret) arrays aligned to T (NaN where unlabelable).
    """
    n = len(df)
    openp = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)

    label = np.full(n, np.nan)
    weight = np.full(n, np.nan)
    fwd_ret = np.full(n, np.nan)

    for t in range(n - 1):
        e = t + 1                       # entry bar (next open)
        entry = openp[e]
        if not np.isfinite(entry) or entry <= 0:
            continue
        tp = entry * (1.0 + target_pct / 100.0)
        sl = entry * (1.0 - stop_pct / 100.0)
        end = min(e + max_hold, n)      # exclusive horizon end
        outcome = 0                     # default: time-stop
        exit_px = close[min(end - 1, n - 1)]
        for k in range(e, end):
            hit_sl = low[k] <= sl
            hit_tp = high[k] >= tp
            if hit_sl and hit_tp:       # ambiguous bar -> assume stop first (conservative)
                outcome, exit_px = -1, sl
                break
            if hit_sl:
                outcome, exit_px = -1, sl
                break
            if hit_tp:
                outcome, exit_px = 1, tp
                break
        label[t] = 1.0 if outcome == 1 else 0.0
        fwd_ret[t] = (exit_px - entry) / entry * 100.0
        weight[t] = abs(fwd_ret[t])
    return label, weight, fwd_ret


def add_labels(panel: pd.DataFrame, data: dict[str, pd.DataFrame],
               stop_pct: float, target_pct: float, max_hold: int) -> pd.DataFrame:
    """
    Attach `y_dir1d`, `y_swing`, `swing_w`, `fwd_ret_pct` to the feature panel.

    panel : long (date, symbol, features) from features.build_panel.
    data  : {symbol: bars_df} (must include open/high/low/close).
    """
    frames = []
    for sym, df in data.items():
        d = df.sort_values("date").reset_index(drop=True)
        openp = d["open"].to_numpy(dtype=np.float64)
        close = d["close"].to_numpy(dtype=np.float64)
        n = len(d)

        # dir1d: next-session open->close direction
        nxt_open = np.roll(openp, -1); nxt_open[-1] = np.nan
        nxt_close = np.roll(close, -1); nxt_close[-1] = np.nan
        y_dir = np.where(np.isfinite(nxt_open) & np.isfinite(nxt_close),
                         (nxt_close > nxt_open).astype(float), np.nan)

        y_swing, w_swing, fwd = _swing_one(d, stop_pct, target_pct, max_hold)
        frames.append(pd.DataFrame({
            "date": d["date"], "symbol": sym,
            "y_dir1d": y_dir, "y_swing": y_swing, "swing_w": w_swing, "fwd_ret_pct": fwd,
        }))

    lab = pd.concat(frames, ignore_index=True)
    return panel.merge(lab, on=["date", "symbol"], how="left")


def swing_label_frame(data: dict[str, pd.DataFrame],
                      stop_pct: float, target_pct: float, max_hold: int) -> pd.DataFrame:
    """
    Just the swing head's labels for a given barrier set — for the label sweep (hp_005),
    which holds features fixed and only varies the triple-barrier. Returns
    (date, symbol, y_swing, swing_w, fwd_ret_pct).
    """
    frames = []
    for sym, df in data.items():
        d = df.sort_values("date").reset_index(drop=True)
        y, w, fwd = _swing_one(d, stop_pct, target_pct, max_hold)
        frames.append(pd.DataFrame({"date": d["date"], "symbol": sym,
                                    "y_swing": y, "swing_w": w, "fwd_ret_pct": fwd}))
    return pd.concat(frames, ignore_index=True)


def label_summary(panel: pd.DataFrame) -> str:
    """One-screen class-balance / coverage summary for the acceptance test."""
    lines = []
    n = len(panel)
    lines.append(f"panel rows: {n:,}  symbols: {panel['symbol'].nunique()}  "
                 f"dates: {panel['date'].nunique()} "
                 f"({panel['date'].min().date()} → {panel['date'].max().date()})")
    for head, col in (("dir1d", "y_dir1d"), ("swing", "y_swing")):
        if col not in panel:
            continue
        s = panel[col].dropna()
        if len(s):
            lines.append(f"  {head:6s}: labelled={len(s):,} ({len(s)/n*100:.0f}%)  "
                         f"positive_rate={s.mean()*100:.1f}%")
    if "fwd_ret_pct" in panel:
        fr = panel["fwd_ret_pct"].dropna()
        if len(fr):
            lines.append(f"  swing fwd_ret%: mean={fr.mean():.2f}  median={fr.median():.2f}")
    return "\n".join(lines)
