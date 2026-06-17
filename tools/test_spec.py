"""Smoke test for SpecStrategy (run: python tools/test_spec.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.strategies.spec import SpecStrategy, validate_spec

spec = {
    "schema": "strategy-spec/v1", "id": "test_v1", "name": "Test",
    "params": {"rsi_buy": 30},
    "indicators": {
        "rsi2": {"fn": "rsi", "period": 2},
        "ema10": {"fn": "ema", "span": 10},
        "del_prev": {"fn": "shift", "on": "delivery_qty", "n": 1},
    },
    "entry": {"all": ["rsi2 < @rsi_buy", "close > ema10",
                      "delivery_qty > del_prev"]},
    "exit": {"stop_loss_pct": 4, "target_pct": 8, "max_hold_days": 6},
}
print("validate:", validate_spec(spec))

n = 60
rng = np.random.default_rng(1)
close = 100 + np.cumsum(rng.normal(0.1, 1, n))
df = pd.DataFrame({
    "date": pd.date_range("2026-01-01", periods=n),
    "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
    "total_volume": rng.integers(100_000, 1_000_000, n).astype(float),
    "delivery_qty": rng.integers(10_000, 100_000, n).astype(float),
})

s = SpecStrategy(spec)
out = s.generate_signals(df)
print("signals:", int(out.buy_signal.sum()), "of", len(out),
      "| exits:", s.exit_overrides())

s2 = SpecStrategy(spec, {"rsi_buy": 80})
print("override signals:", int(s2.generate_signals(df).buy_signal.sum()))

out3 = SpecStrategy(spec).generate_signals(df.drop(columns=["delivery_qty"]))
print("missing-col signals:", int(out3.buy_signal.sum()), "(expected 0)")

bad = {"schema": "strategy-spec/v1", "id": "x", "name": "x",
       "entry": {"all": ["foo > @nope"]},
       "indicators": {"z": {"fn": "wat"}}, "exit": {"bogus": 1}}
errs = validate_spec(bad)
print("bad-spec errors:", len(errs), "(expected 3)")
