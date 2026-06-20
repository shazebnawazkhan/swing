"""
src.predictor.model
-------------------
Train / calibrate / predict / score for one prediction head (docs/PREDICTOR.md §6–§7).

Primary model : LightGBM classifier (handles sparse/missing features natively).
Baseline      : LogisticRegression on the same features (median-imputed + scaled).
                If the GBT can't beat the baseline's AUC, it is fitting noise — that
                gate is enforced by the caller (train_predictor.py), not hidden here.
Calibration   : isotonic on the validation fold, so P(win) means what it says.

Split discipline: the time-ordered split with a PURGE gap (= label horizon) is in
`time_split`, so a training label whose forward window overlaps validation is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import lightgbm as lgb


def time_split(panel: pd.DataFrame, train_months: int = 12, valid_months: int = 2,
               purge_days: int = 15):
    """
    Time-ordered split honoring the user's spec: train [earliest .. valid_start-purge],
    validate [last `valid_months`]. `purge_days` (≈ label horizon) drops train rows whose
    forward label window would overlap validation — the standard leakage guard.

    Returns (train_mask, valid_mask, valid_start_date).
    """
    dmax = panel["date"].max()
    valid_start = dmax - pd.DateOffset(months=valid_months)
    purge_cut = valid_start - pd.Timedelta(days=purge_days)
    train_start = valid_start - pd.DateOffset(months=train_months)
    train = (panel["date"] >= train_start) & (panel["date"] < purge_cut)
    valid = panel["date"] >= valid_start
    return train, valid, valid_start


def make_masks(panel, valid_start, train_months, valid_months, purge_days):
    """Train/validate masks for one fold whose validation window starts at valid_start."""
    valid_end = valid_start + pd.DateOffset(months=valid_months)
    purge_cut = valid_start - pd.Timedelta(days=purge_days)
    train_start = valid_start - pd.DateOffset(months=train_months)
    train = (panel["date"] >= train_start) & (panel["date"] < purge_cut)
    valid = (panel["date"] >= valid_start) & (panel["date"] < valid_end)
    return train, valid, valid_end


def walk_forward_folds(panel, train_months=10, valid_months=2, slide_months=1, purge_days=15):
    """
    Rolling walk-forward folds (docs/PREDICTOR.md §7). The latest fold's validation
    window ends at the panel's last date; earlier folds slide back by `slide_months`
    while the train window still fits inside history.

    Returns a list of (valid_start, valid_end, train_mask, valid_mask), oldest first.
    """
    dmin, dmax = panel["date"].min(), panel["date"].max()
    starts, cur = [], dmax - pd.DateOffset(months=valid_months)
    while (cur - pd.DateOffset(months=train_months)) >= dmin:
        starts.append(cur)
        cur = cur - pd.DateOffset(months=slide_months)
    folds = []
    for vs in sorted(starts):
        tr, va, ve = make_masks(panel, vs, train_months, valid_months, purge_days)
        folds.append((vs, ve, tr, va))
    return folds


_LGB_PARAMS = dict(
    objective="binary", n_estimators=400, learning_rate=0.03,
    num_leaves=31, max_depth=-1, min_child_samples=80,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=5.0, reg_alpha=1.0, n_jobs=-1, verbosity=-1,
)


def train_head(panel: pd.DataFrame, feature_cols: list[str], label_col: str,
               weight_col: str | None, train_mask, valid_mask) -> dict:
    """
    Train LightGBM + logistic baseline for one head; calibrate on validation.

    Returns a dict with the fitted gbt, calibrator, baseline+scaler+imputer, the
    validation frame with predicted probabilities, and a metrics dict.
    """
    feat = panel[feature_cols]
    y = panel[label_col]
    labelled = y.notna()
    tr = train_mask & labelled
    va = valid_mask & labelled

    Xtr, ytr = feat[tr], y[tr].astype(int)
    Xva, yva = feat[va], y[va].astype(int)
    if ytr.sum() == 0 or (len(ytr) - ytr.sum()) == 0 or len(Xva) == 0:
        raise ValueError(f"degenerate split for {label_col}: "
                         f"train_pos={int(ytr.sum())}/{len(ytr)} valid={len(Xva)}")

    spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))   # class imbalance
    sw = panel.loc[tr, weight_col].to_numpy() if weight_col else None
    if sw is not None:
        sw = np.where(np.isfinite(sw) & (sw > 0), sw, 1.0)

    gbt = lgb.LGBMClassifier(scale_pos_weight=spw, **_LGB_PARAMS)
    gbt.fit(Xtr, ytr, sample_weight=sw)
    p_raw = gbt.predict_proba(Xva)[:, 1]

    # Isotonic calibration on the validation fold
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(p_raw, yva)
    p_cal = cal.transform(p_raw)

    # Logistic baseline (impute + scale; linear models can't eat NaN)
    med = Xtr.median(numeric_only=True)
    scaler = StandardScaler()
    # fillna(med) then fillna(0): an all-NaN column has a NaN median, so the 0
    # backstop is what keeps the linear baseline finite.
    Xtr_b = scaler.fit_transform(Xtr.fillna(med).fillna(0.0))
    Xva_b = scaler.transform(Xva.fillna(med).fillna(0.0))
    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    base.fit(Xtr_b, ytr)
    p_base = base.predict_proba(Xva_b)[:, 1]

    metrics = {
        "auc": _safe_auc(yva, p_cal),
        "pr_auc": _safe(lambda: average_precision_score(yva, p_cal)),
        "brier": _safe(lambda: brier_score_loss(yva, p_cal)),
        "base_auc": _safe_auc(yva, p_base),
        "base_rate": float(yva.mean()),
        "n_train": int(len(ytr)), "n_valid": int(len(yva)),
        "train_pos_rate": float(ytr.mean()), "valid_pos_rate": float(yva.mean()),
    }
    metrics["beats_baseline"] = metrics["auc"] > metrics["base_auc"]

    importance = (pd.Series(gbt.feature_importances_, index=feature_cols)
                  .sort_values(ascending=False))

    valid_df = panel.loc[va, ["date", "symbol", label_col]].copy()
    valid_df["p"] = p_cal
    if "fwd_ret_pct" in panel.columns:
        valid_df["fwd_ret_pct"] = panel.loc[va, "fwd_ret_pct"].to_numpy()

    return {
        "gbt": gbt, "calibrator": cal, "baseline": base, "scaler": scaler, "impute": med,
        "feature_cols": feature_cols, "metrics": metrics,
        "importance": importance, "valid_df": valid_df,
    }


def predict_head(artefacts: dict, panel: pd.DataFrame) -> np.ndarray:
    """Calibrated P(positive) for every row of `panel` using a trained head."""
    X = panel[artefacts["feature_cols"]]
    p_raw = artefacts["gbt"].predict_proba(X)[:, 1]
    return artefacts["calibrator"].transform(p_raw)


def basket_metrics(valid_df: pd.DataFrame, label_col: str, top_frac: float = 0.1,
                   cost_pct: float = 0.25) -> dict:
    """
    Economic test (docs/PREDICTOR.md §7): take the top-decile by predicted P each day,
    net of round-trip cost, and report the realised basket stats. Needs `fwd_ret_pct`.
    """
    if "fwd_ret_pct" not in valid_df.columns or valid_df.empty:
        return {"profit_factor": 0.0, "expectancy_pct": 0.0, "n_trades": 0,
                "win_rate": 0.0, "precision_at_n": 0.0}
    # Per-date: keep the top `top_frac` by predicted p (at least 1 name per day).
    v = valid_df.dropna(subset=["fwd_ret_pct"]).copy()
    grp = v.groupby("date")
    rank = grp["p"].rank(method="first", ascending=False)
    keep_n = grp["p"].transform("size").mul(top_frac).clip(lower=1.0)
    picks = v[rank <= keep_n]
    r = picks["fwd_ret_pct"].to_numpy(dtype=float) - cost_pct
    if len(r) == 0:
        return {"profit_factor": 0.0, "expectancy_pct": 0.0, "n_trades": 0,
                "win_rate": 0.0, "precision_at_n": 0.0}
    gw, gl = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "profit_factor": round(float(gw / gl), 3) if gl > 0 else float("inf"),
        "expectancy_pct": round(float(r.mean()), 3),
        "n_trades": int(len(r)),
        "win_rate": round(float((r > 0).mean()) * 100, 1),
        "precision_at_n": round(float(picks[label_col].mean()) * 100, 1),
    }


def portfolio_curve(valid_df: pd.DataFrame, top_frac: float = 0.1, cost_pct: float = 0.25,
                    sizing: str = "equal", kelly_b: float = 2.8,
                    kelly_cap: float = 0.25, trading_days: int = 252) -> dict:
    """
    Daily-rebalanced basket equity curve for the position-sizing test (hp_004).

    Each trading day we take the top-`top_frac` names by predicted P(win) and
    allocate capital across them, realise that day's net-of-cost return, and
    compound the daily basket returns into an out-of-sample equity curve. Both
    sizing modes see *identical* picks each day — only the weights differ — so
    any CAGR / max-DD gap isolates the effect of sizing alone.

      equal      : 1/N across the N picks (gross = 1.0, always fully invested).
      kelly      : per-name fraction f = clip(p - (1-p)/b, 0, kelly_cap), gross
                   capped at 1.0 (no leverage), remainder held flat in cash. Edge-poor
                   days shrink exposure — combines de-risking with conviction tilt.
      kelly_full : the same Kelly fractions renormalised to gross = 1.0 every day, so
                   exposure matches equal-weight and only the *tilt* toward high-P(win)
                   names remains. Isolates skill-weighting from plain de-risking.

    Picks per day match `basket_metrics`: floor(N * top_frac), at least one name.
    Returns CAGR, max_DD (%), Sharpe, total_return (%), mean exposure, n_trades.
    """
    zero = {"cagr_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0, "total_return_pct": 0.0,
            "avg_exposure": 0.0, "n_trades": 0, "n_days": 0}
    if "fwd_ret_pct" not in valid_df.columns or valid_df.empty:
        return zero
    v = valid_df.dropna(subset=["fwd_ret_pct"]).copy()
    if v.empty:
        return zero
    cost = cost_pct / 100.0

    dates, rets, exposures, n_trades = [], [], [], 0
    for date, g in v.sort_values("p", ascending=False).groupby("date", sort=True):
        keep_n = max(1, int(len(g) * top_frac))
        picks = g.head(keep_n)
        r = picks["fwd_ret_pct"].to_numpy(dtype=float) / 100.0 - cost
        if sizing in ("kelly", "kelly_full"):
            p = picks["p"].to_numpy(dtype=float)
            f = np.clip(p - (1.0 - p) / kelly_b, 0.0, kelly_cap)
            s = f.sum()
            if sizing == "kelly_full":           # renormalise to gross 1.0 (tilt only)
                w = f / s if s > 0 else np.full(len(r), 1.0 / len(r))
                exposure = 1.0
            else:                                # cap gross at 1.0, hold cash otherwise
                w = f / s if s > 1.0 else f
                exposure = min(float(s), 1.0)
        else:
            w = np.full(len(r), 1.0 / len(r))
            exposure = 1.0
        dates.append(date)
        rets.append(float((w * r).sum()))
        exposures.append(exposure)
        n_trades += len(r)

    rets = np.asarray(rets, dtype=float)
    equity = np.cumprod(1.0 + rets)
    end = float(equity[-1])
    n_days = len(rets)
    span_years = n_days / trading_days
    cagr = (end ** (1.0 / span_years) - 1.0) if (span_years > 0 and end > 0) else 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / peak)) if n_days else 0.0
    sd = rets.std()
    sharpe = float(rets.mean() / sd * np.sqrt(trading_days)) if sd > 0 else 0.0
    return {
        "cagr_pct": round(cagr * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2),
        "total_return_pct": round((end - 1.0) * 100, 2),
        "avg_exposure": round(float(np.mean(exposures)), 3),
        "n_trades": int(n_trades),
        "n_days": int(n_days),
    }


def _safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    except Exception:
        return 0.5


def _safe(fn):
    try:
        return float(fn())
    except Exception:
        return 0.0
