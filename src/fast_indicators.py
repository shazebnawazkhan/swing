"""
fast_indicators.py
------------------
Backend-agnostic accelerated indicators and a GIL-free simulation loop.

Priority chain (first available wins at each level):
  1. CuPy (CUDA)   – GPU-parallel batch indicator computation
  2. Numba JIT     – AVX2/AVX-512 auto-vectorised via LLVM; nogil=True lets
                     threads run simulations truly in parallel
  3. Bottleneck    – SSE2-optimised move_mean / move_std (2–4× faster than pandas)
  4. NumPy         – always available; AVX via underlying BLAS/MKL/OpenBLAS

Public API
----------
  backend()                              -> str   (description of active backends)
  ema(arr, span)                         -> ndarray
  rolling_mean(arr, window)              -> ndarray
  rolling_std(arr, window, ddof=0)       -> ndarray
  rolling_min(arr, window)               -> ndarray
  batch_ema(matrix, span)                -> ndarray  shape (N, T)
  batch_rolling_mean(matrix, window)     -> ndarray  shape (N, T)
  simulate_trades(closes, signals, dates_ord,
                  sl_pct, tp_pct, max_hold, pos_size)
      -> dict with arrays: entry_idx, exit_idx, entry_prices, exit_prices,
                           hold_days, pnl_pcts, gross_pnls, exit_codes
         exit_codes: 0=STOP_LOSS  1=TARGET_HIT  2=MAX_HOLD  3=OPEN_AT_END
         Returns None if Numba unavailable (caller uses Python fallback).
"""

import numpy as np

# ── Backend Detection ─────────────────────────────────────────────────────────

_parts: list[str] = []

# 1. CuPy / CUDA
try:
    import cupy as _cp
    import cupy.cuda.runtime as _curt
    _dev_props = _curt.getDeviceProperties(0)
    _gpu_name  = _dev_props["name"].decode()
    _gpu_mem   = _dev_props["totalGlobalMem"] / 1e9
    _parts.append(f"cuda:{_gpu_name}({_gpu_mem:.1f}GB)")
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False

# 2. Numba JIT (AVX via LLVM auto-vectorisation)
try:
    import numba as _numba
    from numba import njit as _njit, prange as _prange
    _n_threads = _numba.get_num_threads()
    _parts.append(f"numba-{_numba.__version__}({_n_threads}t)")
    _HAS_NUMBA = True
except Exception:
    _njit = _prange = None
    _HAS_NUMBA = False

# 3. Bottleneck (SSE2 rolling ops)
try:
    import bottleneck as _bn
    _parts.append(f"bottleneck-{_bn.__version__}")
    _HAS_BN = True
except Exception:
    _bn = None
    _HAS_BN = False

if not _parts:
    _parts.append("numpy")


def backend() -> str:
    return " + ".join(_parts)


# ── Single-series indicator functions ─────────────────────────────────────────

def ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average; alpha = 2/(span+1)."""
    a = arr.astype(np.float64)
    if _HAS_NUMBA:
        return _ema_nb(a, span)
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(a), dtype=np.float64)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1.0 - alpha) * out[i - 1]
    return out


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean; NaN for the first ``window-1`` elements."""
    a = arr.astype(np.float64)
    if _HAS_BN:
        return _bn.move_mean(a, window=window, min_count=window)
    # Cumsum trick: O(n), BLAS-accelerated (AVX via numpy)
    cs  = np.cumsum(np.concatenate([[0.0], a]))
    out = np.full(len(a), np.nan)
    out[window - 1:] = (cs[window:] - cs[:-window]) / window
    return out


def rolling_std(arr: np.ndarray, window: int, ddof: int = 0) -> np.ndarray:
    """Rolling std; NaN for the first ``window-1`` elements."""
    a = arr.astype(np.float64)
    if _HAS_BN:
        return _bn.move_std(a, window=window, min_count=window, ddof=ddof)
    rm  = rolling_mean(a, window)
    rm2 = rolling_mean(a ** 2, window)
    var = np.maximum(rm2 - rm ** 2, 0.0)
    if ddof == 1:
        var = var * window / (window - 1)
    return np.sqrt(var)


def rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum; NaN for the first ``window-1`` elements."""
    a = arr.astype(np.float64)
    if _HAS_BN:
        return _bn.move_min(a, window=window, min_count=window)
    if _HAS_NUMBA:
        return _rolling_min_nb(a, window)
    # pandas fallback
    import pandas as pd
    return pd.Series(a).rolling(window).min().to_numpy()


# ── Batch (N-stock) indicator functions ───────────────────────────────────────
# These operate on a (N_stocks, N_time) matrix for maximum throughput.
# Stocks are processed in parallel via CUDA threads or Numba prange.

def batch_ema(matrix: np.ndarray, span: int) -> np.ndarray:
    """
    EMA for every row of ``matrix`` (shape N×T).
    CUDA: all rows computed simultaneously.
    Numba prange: rows split across CPU threads.
    """
    m = matrix.astype(np.float64)
    if _HAS_CUPY:
        return _batch_ema_cuda(m, span)
    if _HAS_NUMBA:
        return _batch_ema_nb(m, span)
    return np.stack([ema(row, span) for row in m])


def batch_rolling_mean(matrix: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean for every row of ``matrix`` (shape N×T)."""
    m = matrix.astype(np.float64)
    if _HAS_CUPY:
        return _batch_rolling_mean_cuda(m, window)
    if _HAS_NUMBA:
        return _batch_rolling_mean_nb(m, window)
    return np.stack([rolling_mean(row, window) for row in m])


# ── Trade simulation ───────────────────────────────────────────────────────────

def simulate_trades(
    closes:    np.ndarray,
    signals:   np.ndarray,
    dates_ord: np.ndarray,
    sl_pct:    float,
    tp_pct:    float,
    max_hold:  int,
    pos_size:  float,
) -> dict | None:
    """
    Simulate trades from a buy-signal array.

    Parameters
    ----------
    closes     : float64 array of closing prices
    signals    : bool array – True on buy-signal days
    dates_ord  : int32 array – date.toordinal() for each row
    sl_pct     : stop-loss percentage (e.g. 5.0)
    tp_pct     : take-profit percentage (e.g. 10.0)
    max_hold   : max bars to hold
    pos_size   : capital allocated per trade (INR)

    Returns
    -------
    dict with numpy arrays, or None if Numba unavailable (use Python fallback).
    """
    if not _HAS_NUMBA:
        return None

    ei, xi, ep, xp, hd, pp, gp, ec = _simulate_nb(
        closes.astype(np.float64),
        signals.astype(np.bool_),
        dates_ord.astype(np.int32),
        float(sl_pct), float(tp_pct), int(max_hold), float(pos_size),
    )
    _CODE = {0: "STOP_LOSS", 1: "TARGET_HIT", 2: "MAX_HOLD", 3: "OPEN_AT_END"}
    return dict(
        entry_idx    = ei,
        exit_idx     = xi,
        entry_prices = ep,
        exit_prices  = xp,
        hold_days    = hd,
        pnl_pcts     = pp,
        gross_pnls   = gp,
        exit_codes   = [_CODE.get(int(c), "UNKNOWN") for c in ec],
    )


# ── Numba JIT implementations ─────────────────────────────────────────────────
# Compiled once and cached in __pycache__; subsequent imports load instantly.

if _HAS_NUMBA:
    @_njit(cache=True, fastmath=True)
    def _ema_nb(arr, span):
        alpha = 2.0 / (span + 1.0)
        out = np.empty(len(arr), dtype=np.float64)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
        return out

    @_njit(parallel=True, cache=True, fastmath=True)
    def _batch_ema_nb(matrix, span):
        n, t = matrix.shape
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(matrix)
        for i in _prange(n):            # parallel over stocks
            out[i, 0] = matrix[i, 0]
            for j in range(1, t):
                out[i, j] = alpha * matrix[i, j] + (1.0 - alpha) * out[i, j - 1]
        return out

    @_njit(parallel=True, cache=True, fastmath=True)
    def _batch_rolling_mean_nb(matrix, window):
        n, t = matrix.shape
        out = np.full((n, t), np.nan)
        for i in _prange(n):
            s = 0.0
            for j in range(t):
                s += matrix[i, j]
                if j >= window:
                    s -= matrix[i, j - window]
                if j >= window - 1:
                    out[i, j] = s / window
        return out

    @_njit(cache=True, fastmath=True)
    def _rolling_min_nb(arr, window):
        n = len(arr)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            mn = arr[i - window + 1]
            for j in range(i - window + 2, i + 1):
                if arr[j] < mn:
                    mn = arr[j]
            out[i] = mn
        return out

    @_njit(cache=True, fastmath=True, nogil=True)   # nogil → threads run concurrently
    def _simulate_nb(closes, signals, dates_ord, sl_pct, tp_pct, max_hold, pos_size):
        n     = len(closes)
        max_t = n // 2 + 2
        ei  = np.empty(max_t, dtype=np.int32)
        xi  = np.empty(max_t, dtype=np.int32)
        ep  = np.empty(max_t, dtype=np.float64)
        xp  = np.empty(max_t, dtype=np.float64)
        hd  = np.empty(max_t, dtype=np.int32)
        pp  = np.empty(max_t, dtype=np.float64)
        gp  = np.empty(max_t, dtype=np.float64)
        ec  = np.empty(max_t, dtype=np.int8)

        nt        = 0
        in_trade  = False
        entry_p   = 0.0
        entry_i   = 0
        hold      = 0
        shares    = 0

        for i in range(n):
            c = closes[i]
            if np.isnan(c):
                continue

            if in_trade:
                hold += 1
                pct = (c - entry_p) / entry_p * 100.0
                code = np.int8(-1)
                if pct <= -sl_pct:
                    code = np.int8(0)
                elif pct >= tp_pct:
                    code = np.int8(1)
                elif hold >= max_hold:
                    code = np.int8(2)

                if code >= 0:
                    ei[nt] = entry_i;   xi[nt] = i
                    ep[nt] = entry_p;   xp[nt] = c
                    hd[nt] = hold;      pp[nt] = pct
                    gp[nt] = shares * (c - entry_p)
                    ec[nt] = code;      nt += 1
                    in_trade = False

            elif signals[i]:
                shares = int(pos_size / c) if c > 0.0 else 0
                if shares == 0:
                    continue  # can't afford even 1 share; skip signal
                entry_p = c
                entry_i = i
                in_trade = True
                hold = 0

        if in_trade:
            last_c = closes[n - 1]
            if np.isnan(last_c):
                last_c = entry_p
            pct = (last_c - entry_p) / entry_p * 100.0 if entry_p > 0.0 else 0.0
            ei[nt] = entry_i;   xi[nt] = n - 1
            ep[nt] = entry_p;   xp[nt] = last_c
            hd[nt] = hold;      pp[nt] = pct
            gp[nt] = shares * (last_c - entry_p)
            ec[nt] = np.int8(3); nt += 1

        return ei[:nt], xi[:nt], ep[:nt], xp[:nt], hd[:nt], pp[:nt], gp[:nt], ec[:nt]


# ── CUDA (CuPy) batch implementations ────────────────────────────────────────
# Per-timestep ops run as vectorised CUDA kernels over N stocks simultaneously.

if _HAS_CUPY:
    def _batch_ema_cuda(matrix: np.ndarray, span: int) -> np.ndarray:
        alpha = 2.0 / (span + 1.0)
        g   = _cp.asarray(matrix, dtype=_cp.float64)
        out = _cp.empty_like(g)
        out[:, 0] = g[:, 0]
        for t in range(1, g.shape[1]):          # parallel over N stocks per step
            out[:, t] = alpha * g[:, t] + (1.0 - alpha) * out[:, t - 1]
        return _cp.asnumpy(out)

    def _batch_rolling_mean_cuda(matrix: np.ndarray, window: int) -> np.ndarray:
        g   = _cp.asarray(matrix, dtype=_cp.float64)
        n, T = g.shape
        out = _cp.full((n, T), _cp.nan)
        cs  = _cp.concatenate([_cp.zeros((n, 1)), _cp.cumsum(g, axis=1)], axis=1)
        if T >= window:
            out[:, window - 1:] = (cs[:, window:] - cs[:, :T - window + 1]) / window
        return _cp.asnumpy(out)
