"""
run_experiments.py
------------------
Experiment runner for spec strategies (see docs/program.md).

For every strategy spec in data/strategies/ (or a --specs subset):
  1. generate signals per symbol over the full cached history (warm-up included)
  2. zero signals outside the backtest window, shift +1 bar (next-bar fill,
     no lookahead), simulate via the numba engine with the spec's exit rules
  3. deduct round-trip transaction costs, pool trades across the universe
  4. compute metrics (src/metrics.py) and append a record to
     data/results/experiments.jsonl; refresh outputs/experiments_summary.csv

Pooled semantics: every trade uses a fixed position size (--pos-size); metrics
aggregate all symbols' trades ordered by exit date.  This compares strategies
under identical conditions — it is not a portfolio simulation (no capital or
concurrency constraints).

Usage
-----
    python scripts/run_experiments.py                          # all specs, halal, 365d
    python scripts/run_experiments.py --specs rsi2_meanrev_v1 gap_up_continuation_v1
    python scripts/run_experiments.py --window-days 180 --note "half-year check"
    python scripts/run_experiments.py --hypothesis h_008 --specs delivery_absorption_v1
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

import src.config as cfg
from src import fast_indicators as fi
from src import costs as costmodel
from src.benchmark import load_benchmark, regime_series
from src.metrics import trade_metrics, market_adjusted_metrics
from src.strategies.spec import SPEC_DIR, SpecStrategy, load_all_specs, load_spec

sys.path.insert(0, str(ROOT / "scripts"))
from build_leaderboard import main as build_leaderboard

FETCHED_DIR  = ROOT / "data" / "fetched"
STOCKS_CSV   = ROOT / "data" / "stocks.csv"
RESULTS_DIR  = ROOT / "data" / "results"
RESULTS_FILE = RESULTS_DIR / "experiments.jsonl"
TRADES_DIR   = RESULTS_DIR / "trades"          # per-spec trade lists for the report charts
SUMMARY_CSV  = ROOT / "outputs" / "experiments_summary.csv"

MIN_ROWS  = 150     # symbols with shorter history are skipped
MIN_PRICE = 10.0    # median close floor — exclude illiquid penny stocks


# ── Universe loading ──────────────────────────────────────────────────────────

def load_universe_data(universe: str) -> dict[str, pd.DataFrame]:
    """Load per-symbol parquets for the chosen universe into memory, once."""
    stocks = pd.read_csv(STOCKS_CSV, dtype=str).fillna("")
    if universe == "halal":
        wanted = set(stocks.loc[stocks["halal"].str.upper() == "Y", "stock"].str.strip())
    else:
        wanted = set(stocks["stock"].str.strip())

    data: dict[str, pd.DataFrame] = {}
    skipped_short = skipped_penny = 0
    for p in sorted(FETCHED_DIR.glob("*.parquet")):
        if p.stem not in wanted:
            continue
        df = pd.read_parquet(p)
        if len(df) < MIN_ROWS:
            skipped_short += 1
            continue
        if df["close"].median() < MIN_PRICE:
            skipped_penny += 1
            continue
        df["date"] = pd.to_datetime(df["date"])
        data[p.stem] = df.sort_values("date").reset_index(drop=True)

    print(f"  Universe '{universe}': {len(data)} symbols loaded "
          f"(skipped: {skipped_short} short-history, {skipped_penny} sub-₹{MIN_PRICE:.0f})")
    return data


# ── Feature precompute (spec-independent; done once per run) ────────────────────

def inject_liquidity_features(data: dict[str, pd.DataFrame], window: int = 20) -> None:
    """Add trailing ₹-ADV (_adv_inr) and daily-vol% (_vol_daily_pct) columns in place."""
    for df in data.values():
        turnover = df["close"] * df["total_volume"]
        df["_adv_inr"] = turnover.rolling(window).mean()
        df["_vol_daily_pct"] = df["close"].pct_change().rolling(window).std() * 100.0


def inject_cross_sectional_rs(data: dict[str, pd.DataFrame], lookback: int = 60) -> None:
    """
    Add `rs_rank` ∈ [0,1]: each symbol's `lookback`-day return ranked cross-sectionally
    against every other symbol on the same date (1.0 = strongest in the universe).
    Enables relative-strength entry conditions (h_014) the per-symbol spec engine
    cannot compute on its own.
    """
    rets = {}
    for sym, df in data.items():
        s = df.set_index("date")["close"].pct_change(lookback)
        rets[sym] = s
    wide = pd.DataFrame(rets)                          # index=date, cols=symbols
    ranks = wide.rank(axis=1, pct=True)               # per-date cross-sectional pctile
    for sym, df in data.items():
        rr = ranks[sym].reindex(df["date"].values).to_numpy()
        df["rs_rank"] = rr


# ── Per-spec backtest ─────────────────────────────────────────────────────────

def run_spec(
    strat: SpecStrategy,
    data: dict[str, pd.DataFrame],
    window_start: pd.Timestamp,
    pos_size: float,
    cost_pct: float,
    cost_model: str = "flat",
    regime_bull: dict | None = None,
) -> tuple[pd.DataFrame, int]:
    """
    Backtest one spec across the universe. Returns (trades_df, n_symbols_traded).

    cost_model  : "flat" (cost_pct per trade) or "istar" (liquidity-aware, src/costs.py).
    regime_bull : optional {date_ordinal: bool} — when given, buy signals on non-bull
                  market days are suppressed (index-level long/cash gate, h_012).
    """
    ex = strat.exit_overrides()
    sl   = float(ex.get("stop_loss_pct", cfg.STOP_LOSS_PCT))
    tp   = float(ex.get("target_pct", cfg.TARGET_PCT))
    hold = int(ex.get("max_hold_days", cfg.MAX_HOLD_DAYS))

    rows: list[dict] = []
    syms_traded = 0

    for sym, df in data.items():
        try:
            sig_df = strat.generate_signals(df)
        except Exception:
            continue
        if "buy_signal" not in sig_df.columns:
            continue

        sig = sig_df["buy_signal"].to_numpy(dtype=bool)
        # Next-bar fill: a signal from day T's close enters at bar T+1
        sig = np.roll(sig, 1)
        sig[0] = False
        # Trade only inside the backtest window (indicators warmed up before it)
        dates = sig_df["date"].to_numpy()
        sig[dates < np.datetime64(window_start)] = False

        d_ord = sig_df["date"].map(lambda d: d.toordinal()).to_numpy(dtype=np.int32)
        # Index-level regime gate: no new longs on non-bull market days
        if regime_bull is not None and sig.any():
            keep = np.array([regime_bull.get(int(o), True) for o in d_ord], dtype=bool)
            sig &= keep
        if not sig.any():
            continue

        closes = sig_df["close"].to_numpy(dtype=np.float64)

        res = fi.simulate_trades(closes, sig, d_ord, sl, tp, hold, pos_size)
        if res is None:                       # numba unavailable — python fallback
            res = _simulate_py(closes, sig, sl, tp, hold)
        if len(res["entry_idx"]) == 0:
            continue

        adv_arr = sig_df["_adv_inr"].to_numpy(dtype=float) if "_adv_inr" in sig_df else None
        vol_arr = sig_df["_vol_daily_pct"].to_numpy(dtype=float) if "_vol_daily_pct" in sig_df else None

        syms_traded += 1
        date_strs = sig_df["date"].dt.strftime("%Y-%m-%d").to_numpy()
        for k in range(len(res["entry_idx"])):
            entry_p = float(res["entry_prices"][k])
            shares = int(pos_size / entry_p) if entry_p > 0 else 0
            ei = int(res["entry_idx"][k])
            if cost_model == "istar" and adv_arr is not None and vol_arr is not None:
                rt_cost = costmodel.istar_round_trip_pct(
                    pos_size,
                    adv_arr[ei] if np.isfinite(adv_arr[ei]) else 0.0,
                    vol_arr[ei] if np.isfinite(vol_arr[ei]) else 3.0,
                )
            else:
                rt_cost = cost_pct
            net_pct = float(res["pnl_pcts"][k]) - rt_cost
            rows.append({
                "symbol":      sym,
                "entry_date":  date_strs[ei],
                "exit_date":   date_strs[int(res["exit_idx"][k])],
                "entry_price": round(entry_p, 2),
                "exit_price":  round(float(res["exit_prices"][k]), 2),
                "shares":      shares,
                "hold_days":   int(res["hold_days"][k]),
                "cost_pct":    round(float(rt_cost), 3),
                "pnl_pct":     round(net_pct, 3),
                "gross_pnl":   round(shares * entry_p * net_pct / 100.0, 2),
                "exit_reason": res["exit_codes"][k],
            })

    return pd.DataFrame(rows), syms_traded


def _simulate_py(closes: np.ndarray, sig: np.ndarray,
                 sl: float, tp: float, max_hold: int) -> dict:
    """Pure-Python mirror of fast_indicators._simulate_nb (used when numba is absent)."""
    ei, xi, ep, xp, hd, pp = [], [], [], [], [], []
    codes: list[str] = []
    in_trade, entry_p, entry_i, hold = False, 0.0, 0, 0
    n = len(closes)
    for i in range(n):
        c = closes[i]
        if np.isnan(c):
            continue
        if in_trade:
            hold += 1
            pct = (c - entry_p) / entry_p * 100.0
            reason = None
            if pct <= -sl:
                reason = "STOP_LOSS"
            elif pct >= tp:
                reason = "TARGET_HIT"
            elif hold >= max_hold:
                reason = "MAX_HOLD"
            if reason:
                ei.append(entry_i); xi.append(i); ep.append(entry_p)
                xp.append(c); hd.append(hold); pp.append(pct); codes.append(reason)
                in_trade = False
        elif sig[i] and c > 0:
            in_trade, entry_p, entry_i, hold = True, c, i, 0
    if in_trade:
        c = closes[-1] if not np.isnan(closes[-1]) else entry_p
        ei.append(entry_i); xi.append(n - 1); ep.append(entry_p); xp.append(c)
        hd.append(hold); pp.append((c - entry_p) / entry_p * 100.0)
        codes.append("OPEN_AT_END")
    return {
        "entry_idx": np.array(ei, dtype=int), "exit_idx": np.array(xi, dtype=int),
        "entry_prices": np.array(ep), "exit_prices": np.array(xp),
        "hold_days": np.array(hd, dtype=int), "pnl_pcts": np.array(pp),
        "exit_codes": codes,
    }


# ── Record / report ───────────────────────────────────────────────────────────

def append_record(record: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    cols = ["spec_id", "trades", "win_rate", "profit_factor", "expectancy_pct",
            "total_pnl", "sharpe_approx", "max_drawdown_pct", "avg_hold_days"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values(["profit_factor", "expectancy_pct"], ascending=False)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n  Summary → {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"  Records → {RESULTS_FILE.relative_to(ROOT)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run spec-strategy experiments")
    ap.add_argument("--specs", nargs="+", help="spec ids (default: all in data/strategies)")
    ap.add_argument("--universe", choices=["halal", "all"], default="halal")
    ap.add_argument("--window-days", type=int, default=365,
                    help="calendar days to trade (default 365); earlier data = warm-up")
    ap.add_argument("--pos-size", type=float, default=100_000.0)
    ap.add_argument("--cost-model", choices=["flat", "istar"], default="istar",
                    help="flat 0.25%% (legacy) or liquidity-aware I-Star (src/costs.py)")
    ap.add_argument("--regime-gate", action="store_true",
                    help="suppress new longs on non-bull market days (benchmark < 50-EMA)")
    ap.add_argument("--benchmark", choices=["nifty", "universe"], default="nifty")
    ap.add_argument("--hypothesis", default=None, help="hypothesis id to tag records with")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    if args.specs:
        specs = [load_spec(SPEC_DIR / f"{sid}.json") for sid in args.specs]
    else:
        specs = load_all_specs()
    if not specs:
        print("  No specs found in data/strategies/ — nothing to run.")
        return

    print("=" * 62)
    print("  EXPERIMENT RUNNER")
    print("=" * 62)
    print(f"  Specs    : {[s['id'] for s in specs]}")
    print(f"  Engine   : {fi.backend()}")

    data = load_universe_data(args.universe)
    if not data:
        print("  No data — run scripts/bulk_fetch.py first.")
        return

    # Spec-independent features (computed once for the whole run)
    inject_liquidity_features(data)
    inject_cross_sectional_rs(data)

    benchmark, bench_src = load_benchmark(prefer=args.benchmark)
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    regime_bull = None
    if args.regime_gate:
        reg = regime_series(benchmark, span=50)
        regime_bull = {pd.Timestamp(d).toordinal(): bool(v) for d, v in reg.items()}

    data_end = max(df["date"].iloc[-1] for df in data.values())
    window_start = data_end - timedelta(days=args.window_days)
    cost = float(getattr(cfg, "ROUND_TRIP_COST_PCT", 0.25))
    cost_desc = (f"I-Star liquidity-aware (base {costmodel.BASE_FEE_PCT}%)"
                 if args.cost_model == "istar" else f"flat {cost}%")
    print(f"  Window   : {window_start:%Y-%m-%d} → {data_end:%Y-%m-%d}  ·  pos ₹{args.pos_size:,.0f}")
    print(f"  Costs    : {cost_desc}")
    print(f"  Benchmark: {bench_src}"
          + ("  ·  regime gate ON (long/cash)" if args.regime_gate else ""))

    summary_rows = []
    for spec in specs:
        t0 = time.perf_counter()
        strat = SpecStrategy(spec)
        trades, n_syms = run_spec(strat, data, window_start, args.pos_size, cost,
                                  cost_model=args.cost_model, regime_bull=regime_bull)
        m = trade_metrics(trades, float(cfg.CAPITAL))
        am = market_adjusted_metrics(trades, benchmark)
        m = {**m, **am}
        avg_cost = round(float(trades["cost_pct"].mean()), 3) if len(trades) else 0.0

        # Persist per-spec trades so the report can draw per-stock charts (served by server.py)
        suffix = "__gated" if args.regime_gate else ""
        trades_path = TRADES_DIR / f"{spec['id']}{suffix}.parquet"
        TRADES_DIR.mkdir(parents=True, exist_ok=True)
        if len(trades):
            trades.to_parquet(trades_path, index=False)
        elif trades_path.exists():
            trades_path.unlink()                # stale file from a previous run

        runtime = round(time.perf_counter() - t0, 1)

        record = {
            "id": f"exp_{datetime.now():%Y%m%d_%H%M%S}_{spec['id']}",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "kind": "backtest",
            "spec_id": spec["id"],
            "strategy": spec["name"],
            "params": strat.effective_params(),
            "universe": args.universe,
            "n_symbols": len(data),
            "n_symbols_traded": n_syms,
            "market": "nse",
            "window": [f"{window_start:%Y-%m-%d}", f"{data_end:%Y-%m-%d}"],
            "costs": {"model": args.cost_model, "flat_pct": cost, "avg_round_trip_pct": avg_cost},
            "benchmark": bench_src,
            "regime_gate": bool(args.regime_gate),
            "fill": "next_bar_close",
            "exit": strat.exit_overrides(),
            "metrics": m,
            "hypothesis_id": args.hypothesis,
            "verdict": None,
            "note": args.note,
            "runtime_s": runtime,
        }
        append_record(record)
        summary_rows.append({"spec_id": spec["id"], **m})
        print(f"  {spec['id']:<32} {m['trades']:>5} tr  "
              f"wr {m['win_rate']:>5.1f}%  pf {m['profit_factor']:>5}  "
              f"exp {m['expectancy_pct']:>6}%  dd {m['max_drawdown_pct']:>6}%  "
              f"({runtime}s)")

    print_summary(summary_rows)
    build_leaderboard()


if __name__ == "__main__":
    main()
