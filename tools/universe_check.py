"""Quick market-context check: 1y return distribution of the fetched universe."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

rets, above_ema200 = [], 0
files = sorted((ROOT / "data" / "fetched").glob("*.parquet"))
for p in files:
    df = pd.read_parquet(p, columns=["date", "close"])
    if len(df) < 260:
        continue
    df = df.sort_values("date")
    c = df["close"].to_numpy()
    # window ≈ last 252 trading days
    start = c[-252] if len(c) >= 252 else c[0]
    if start > 0 and not np.isnan(start) and not np.isnan(c[-1]):
        rets.append((c[-1] / start - 1) * 100)
    ema200 = pd.Series(c).ewm(span=200, adjust=False).mean().iloc[-1]
    if c[-1] > ema200:
        above_ema200 += 1

r = np.array(rets)
print(f"symbols          : {len(r)}")
print(f"mean 1y return   : {r.mean():+.1f}%")
print(f"median 1y return : {np.median(r):+.1f}%")
print(f"% positive       : {(r > 0).mean() * 100:.0f}%")
print(f"quartiles        : p25 {np.percentile(r, 25):+.1f}%  p75 {np.percentile(r, 75):+.1f}%")
print(f"above EMA200 now : {above_ema200}/{len(files)} ({above_ema200 / len(files) * 100:.0f}%)")
