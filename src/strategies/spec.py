"""
strategies/spec.py
------------------
Declarative strategy specs: a trading strategy expressed as a JSON document
instead of Python code.

Why: the autoresearch loop (docs/program.md) must be able to create, mutate
and store strategies as data.  A spec file fully describes indicators, entry
logic, exit rules and tunable params; SpecStrategy interprets it through the
exact same Strategy contract as the hand-written strategies.

Spec files live in data/strategies/*.json.
Format reference: docs/strategy_format.md.

Usage
-----
    from src.strategies.spec import load_spec, load_all_specs, SpecStrategy

    strat = SpecStrategy(load_spec("data/strategies/rsi2_meanrev_v1.json"))
    sig_df = strat.generate_signals(df)          # standard Strategy contract
    overrides = strat.exit_overrides()           # {"stop_loss_pct": 5, ...}
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Strategy, StrategyParams

SCHEMA_VERSION = "strategy-spec/v1"
SPEC_DIR = Path(__file__).resolve().parents[2] / "data" / "strategies"


# ── Indicator library ──────────────────────────────────────────────────────────
# Each function takes (df, args) and returns a Series aligned to df.index.
# "on" selects the source column (default "close").  Numeric args may be
# given as "@param" references resolved against the spec's params block.

def _src(df: pd.DataFrame, args: dict) -> pd.Series:
    return df[args.get("on", "close")]


def _rsi(df, args):
    s, period = _src(df, args), int(args.get("period", 14))
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df, args):
    period = int(args.get("period", 14))
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _bb_mid(df, args):
    return _src(df, args).rolling(int(args.get("window", 20))).mean()


def _bb_std(df, args):
    return _src(df, args).rolling(int(args.get("window", 20))).std()


_INDICATORS = {
    "sma":      lambda df, a: _src(df, a).rolling(int(a["window"])).mean(),
    "ema":      lambda df, a: _src(df, a).ewm(span=int(a["span"]), adjust=False).mean(),
    "std":      lambda df, a: _src(df, a).rolling(int(a["window"])).std(),
    "hhv":      lambda df, a: _src(df, a).rolling(int(a["window"])).max(),
    "llv":      lambda df, a: _src(df, a).rolling(int(a["window"])).min(),
    "shift":    lambda df, a: _src(df, a).shift(int(a.get("n", 1))),
    "roc":      lambda df, a: _src(df, a).pct_change(int(a["period"])) * 100,
    "rsi":      _rsi,
    "atr":      _atr,
    "bb_upper": lambda df, a: _bb_mid(df, a) + float(a.get("k", 2)) * _bb_std(df, a),
    "bb_lower": lambda df, a: _bb_mid(df, a) - float(a.get("k", 2)) * _bb_std(df, a),
    "bb_width": lambda df, a: (2 * float(a.get("k", 2)) * _bb_std(df, a))
                              / _bb_mid(df, a) * 100,
}

# Exit keys a spec may override (Backtester/config provide the defaults)
_EXIT_KEYS = {"stop_loss_pct", "target_pct", "max_hold_days"}

_PARAM_RE = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)")


# ── Param resolution ───────────────────────────────────────────────────────────

def _resolve(value, params: dict):
    """Resolve a literal or '@name' parameter reference to its value."""
    if isinstance(value, str) and value.startswith("@"):
        name = value[1:]
        if name not in params:
            raise KeyError(f"Spec references unknown param '@{name}'")
        return params[name]
    return value


def _subst_expr(expr: str, params: dict) -> str:
    """Replace every @name token in a condition expression with its value."""
    def rep(m):
        name = m.group(1)
        if name not in params:
            raise KeyError(f"Condition references unknown param '@{name}'")
        return repr(params[name])
    return _PARAM_RE.sub(rep, expr)


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_spec(spec: dict) -> list[str]:
    """Return a list of problems; empty list means the spec is valid."""
    errs: list[str] = []
    if spec.get("schema") != SCHEMA_VERSION:
        errs.append(f"schema must be '{SCHEMA_VERSION}'")
    for key in ("id", "name", "entry"):
        if not spec.get(key):
            errs.append(f"missing required field '{key}'")

    params = spec.get("params", {})
    for col, ind in spec.get("indicators", {}).items():
        fn = ind.get("fn")
        if fn not in _INDICATORS:
            errs.append(f"indicator '{col}': unknown fn '{fn}' "
                        f"(known: {sorted(_INDICATORS)})")
        for k, v in ind.items():
            if isinstance(v, str) and v.startswith("@") and v[1:] not in params:
                errs.append(f"indicator '{col}': unknown param ref '{v}'")

    entry = spec.get("entry", {})
    if isinstance(entry, dict):
        if not entry.get("all") and not entry.get("any"):
            errs.append("entry must contain a non-empty 'all' and/or 'any' list")
        for expr in (entry.get("all") or []) + (entry.get("any") or []):
            for ref in _PARAM_RE.findall(expr):
                if ref not in params:
                    errs.append(f"entry expr '{expr}': unknown param ref '@{ref}'")
    else:
        errs.append("entry must be an object with 'all'/'any' lists")

    for k, v in spec.get("exit", {}).items():
        if k not in _EXIT_KEYS:
            errs.append(f"exit: unknown key '{k}' (allowed: {sorted(_EXIT_KEYS)})")
        elif not isinstance(v, (int, float)):
            errs.append(f"exit.{k} must be numeric")
    return errs


# ── Spec loading ───────────────────────────────────────────────────────────────

def load_spec(path: str | Path) -> dict:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    errs = validate_spec(spec)
    if errs:
        raise ValueError(f"Invalid spec {path}:\n  " + "\n  ".join(errs))
    return spec


def load_all_specs(spec_dir: str | Path = SPEC_DIR) -> list[dict]:
    """Load every *.json spec in spec_dir, sorted by id."""
    specs = [load_spec(p) for p in sorted(Path(spec_dir).glob("*.json"))]
    return sorted(specs, key=lambda s: s["id"])


# ── SpecStrategy ───────────────────────────────────────────────────────────────

class SpecStrategy(Strategy):
    """
    Interprets a strategy-spec/v1 dict through the standard Strategy contract.

    Indicator columns are added under their spec-declared names, every entry
    condition becomes a ``cond_<i>`` boolean column (for diagnostics), and
    ``buy_signal`` / ``conditions_met`` follow the base-class convention.
    """

    def __init__(self, spec: dict, param_overrides: dict | None = None):
        errs = validate_spec(spec)
        if errs:
            raise ValueError("Invalid spec:\n  " + "\n  ".join(errs))
        self.spec = spec
        self._params = dict(spec.get("params", {}))
        if param_overrides:
            unknown = set(param_overrides) - set(self._params)
            if unknown:
                raise KeyError(f"Overrides for unknown params: {sorted(unknown)}")
            self._params.update(param_overrides)
        self.params = StrategyParams(
            name=spec["name"], description=spec.get("description", "")
        )

    # ── Strategy contract ─────────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col, ind in self.spec.get("indicators", {}).items():
            args = {k: _resolve(v, self._params)
                    for k, v in ind.items() if k != "fn"}
            try:
                df[col] = _INDICATORS[ind["fn"]](df, args)
            except KeyError:
                # Source column absent for this symbol → indicator all-NaN,
                # downstream conditions become False (never raise).
                df[col] = np.nan

        entry = self.spec["entry"]
        conds: list[pd.Series] = []

        all_mask = pd.Series(True, index=df.index)
        for expr in entry.get("all") or []:
            c = self._eval(df, expr)
            conds.append(c)
            all_mask &= c

        if entry.get("any"):
            any_mask = pd.Series(False, index=df.index)
            for expr in entry["any"]:
                c = self._eval(df, expr)
                conds.append(c)
                any_mask |= c
            all_mask &= any_mask

        for i, c in enumerate(conds):
            df[f"cond_{i}"] = c
        df["conditions_met"] = sum(c.astype(int) for c in conds)
        df["buy_signal"] = all_mask
        return df

    def _eval(self, df: pd.DataFrame, expr: str) -> pd.Series:
        resolved = _subst_expr(expr, self._params)
        try:
            out = df.eval(resolved, engine="python")
        except Exception:
            return pd.Series(False, index=df.index)
        if not isinstance(out, pd.Series):
            out = pd.Series(bool(out), index=df.index)
        return out.fillna(False).astype(bool)

    # ── Spec extras ───────────────────────────────────────────────────────────

    def exit_overrides(self) -> dict:
        """SL/target/hold overrides declared by the spec (may be empty)."""
        return dict(self.spec.get("exit", {}))

    @property
    def spec_id(self) -> str:
        return self.spec["id"]

    def effective_params(self) -> dict:
        return dict(self._params)

    def __repr__(self) -> str:
        return f"SpecStrategy(id={self.spec_id!r}, name={self.name!r})"
