# Strategy Spec Format — `strategy-spec/v1`

A trading strategy is stored as **one JSON file** in `data/strategies/`. The file is the
single source of truth: indicators, entry logic, exit rules, tunable parameters and
provenance. `src/strategies/spec.py` (`SpecStrategy`) interprets a spec through the same
`Strategy` contract as hand-written strategies, so specs work with every backtester in
the repo.

**Why a data format instead of code:** the autoresearch loop (see `docs/program.md`)
creates, mutates and compares strategies programmatically. A new strategy or a parameter
variant is a new JSON file — no code change, no redeploy, fully diffable in git.

## Full example

```json
{
  "schema": "strategy-spec/v1",
  "id": "rsi2_meanrev_v1",
  "name": "RSI2 Mean Reversion v1",
  "description": "Buy short-term oversold dips inside a medium-term uptrend.",
  "markets": ["nse"],
  "regimes": ["bull", "sideways"],
  "params":  { "rsi_buy": 10, "trend_span": 100 },
  "indicators": {
    "rsi2":   { "fn": "rsi", "on": "close", "period": 2 },
    "ema100": { "fn": "ema", "on": "close", "span": "@trend_span" }
  },
  "entry": {
    "all": [ "rsi2 < @rsi_buy", "close > ema100" ],
    "any": []
  },
  "exit":  { "stop_loss_pct": 5.0, "target_pct": 8.0, "max_hold_days": 7 },
  "param_grid": { "rsi_buy": [5, 10, 15] },
  "provenance": {
    "created": "2026-06-12",
    "hypothesis_id": null,
    "parent_id": null,
    "experiments": []
  }
}
```

## Field reference

| Field | Required | Meaning |
|---|---|---|
| `schema` | yes | Must be `"strategy-spec/v1"`. |
| `id` | yes | Unique slug, `snake_case`, ends with `_v<N>`. A *refinement* is a **new file** with bumped version and `provenance.parent_id` set — specs are immutable once an experiment references them. |
| `name` | yes | Human-readable display name (unique; used by the strategy registry). |
| `description` | no | One or two sentences: the market behaviour being exploited. |
| `markets` | no | `["nse"]`, `["us"]`, … Documentation + universe filtering. Default: all. |
| `regimes` | no | Regimes where the strategy may fire (`bull/bear/sideways/high_vol`). Default: all. |
| `params` | no | Named tunables. Referenced as `@name` in indicator args and entry expressions. |
| `indicators` | no | Map of *output column name* → indicator definition (see below). |
| `entry` | yes | `all`: list of expressions that must ALL be true; `any`: at least one true. Both given ⇒ `ALL(all) AND ANY(any)`. |
| `exit` | no | Overrides for `stop_loss_pct`, `target_pct`, `max_hold_days`. Missing keys fall back to `src/config.py`. |
| `param_grid` | no | Value lists per param for the optimizer / autoresearch sweeps. |
| `provenance` | no | `created`, `hypothesis_id`, `parent_id` (spec this was refined from), `experiments` (experiment ids that evaluated this spec). |

## Indicators

`{ "fn": "<name>", "on": "<source column>", ...args }` — `on` defaults to `"close"`.
Any numeric arg accepts a `"@param"` reference.

| fn | args | Output |
|---|---|---|
| `sma` | `window` | Simple moving average |
| `ema` | `span` | Exponential moving average |
| `std` | `window` | Rolling standard deviation |
| `hhv` / `llv` | `window` | Rolling highest / lowest value |
| `shift` | `n` (default 1) | Column lagged by n days (use for "previous day" logic) |
| `roc` | `period` | % rate of change over `period` days |
| `rsi` | `period` (default 14) | Wilder RSI |
| `atr` | `period` (default 14) | Average True Range (uses high/low/close) |
| `bb_upper` / `bb_lower` | `window` (20), `k` (2) | Bollinger bands |
| `bb_width` | `window`, `k` | Band width as % of middle band |

Add new indicator functions in `src/strategies/spec.py::_INDICATORS` (one lambda or
small function each), then document them here.

## Entry expressions

Evaluated per-row with `DataFrame.eval` (python engine) over the bar columns
(`open, high, low, close, total_volume, delivery_qty, delivery_pct, vwap, oi, …`)
plus every declared indicator column.

- Supported: comparisons, arithmetic, `&` `|` `~`, parentheses, `abs()`.
- **No lookahead:** expressions see only the current row of already-computed series.
  For prior-day values declare a `shift` indicator (e.g. `del_prev` = shift of
  `delivery_qty`) — never reference future rows.
- A condition referencing a missing column (e.g. `delivery_pct` on non-NSE data)
  evaluates to **False**, never raises — matching the base `Strategy` contract.
- NaN comparisons are False (warm-up rows produce no signals).

## Storage & lifecycle

```
data/strategies/<id>.json        # one spec per file, immutable after first experiment
docs/strategy_format.md          # this reference
src/strategies/spec.py           # interpreter + validator
```

Validate without running: `python -c "from src.strategies.spec import load_all_specs; print([s['id'] for s in load_all_specs()])"`

Lifecycle: **draft** (file created) → **evaluated** (referenced by experiments in
`data/results/experiments.jsonl`) → **promoted/rejected** (leaderboard verdict) →
**refined** (child spec with `parent_id`). Rejected specs stay on disk — negative
results prevent the research loop from re-testing dead ideas.
