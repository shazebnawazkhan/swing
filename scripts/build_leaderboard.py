"""
build_leaderboard.py
---------------------
Builds outputs/leaderboard.json from data/results/experiments.jsonl,
data/strategies/*.json and data/research/backlog.jsonl.

For every strategy spec, picks its most recent backtest record, joins it
with the spec's description/lineage and the hypothesis that tested it, and
writes a single JSON file consumed by the "Strategy Lab" section of
outputs/dashboard.html.

Usage
-----
    python scripts/build_leaderboard.py
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_FILE = ROOT / "data" / "results" / "experiments.jsonl"
SPEC_DIR     = ROOT / "data" / "strategies"
BACKLOG_FILE = ROOT / "data" / "research" / "backlog.jsonl"
OUT_FILE     = ROOT / "outputs" / "leaderboard.json"

MARKET_NOTE = (
    "2025-06-11 -> 2026-06-11 was a bear year for the halal universe: "
    "median stock -14.8%, only 31% positive, only 26% above EMA200 "
    "(see docs/research/batch1.md)."
)

_FAMILY_RE = re.compile(r"^(.*)_v(\d+)$")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    experiments = _read_jsonl(RESULTS_FILE)
    backlog = _read_jsonl(BACKLOG_FILE)

    # Latest backtest record per spec_id (later timestamp wins).
    latest: dict[str, dict] = {}
    for rec in experiments:
        if rec.get("kind") != "backtest":
            continue
        sid = rec["spec_id"]
        if sid not in latest or rec["ts"] > latest[sid]["ts"]:
            latest[sid] = rec

    # exp_id -> hypothesis record that cites it.
    hyp_by_exp: dict[str, dict] = {}
    for h in backlog:
        for exp_id in h.get("result_exp_ids", []):
            hyp_by_exp[exp_id] = h

    rows = []
    for spec_path in sorted(SPEC_DIR.glob("*.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        sid = spec["id"]
        rec = latest.get(sid)
        prov = spec.get("provenance", {})

        m = _FAMILY_RE.match(sid)
        family, generation = (m.group(1), f"v{m.group(2)}") if m else (sid, "v1")

        row = {
            "spec_id": sid,
            "name": spec.get("name", sid),
            "description": spec.get("description", ""),
            "family": family,
            "generation": generation,
            "parent_id": prov.get("parent_id"),
            "created": prov.get("created"),
            "regimes": spec.get("regimes", []),
            "hypothesis_id": prov.get("hypothesis_id"),
            "hypothesis_status": None,
            "verdict_note": None,
            "exp_id": None,
            "ts": None,
            "window": None,
            "metrics": None,
        }
        if rec:
            row["exp_id"] = rec["id"]
            row["ts"] = rec["ts"]
            row["window"] = rec.get("window")
            row["metrics"] = rec["metrics"]
            hyp = hyp_by_exp.get(rec["id"])
            if hyp:
                row["hypothesis_id"] = hyp["id"]
                row["hypothesis_status"] = hyp["status"]
                row["verdict_note"] = hyp.get("verdict_note")
        rows.append(row)

    open_hyps = [
        {
            "id": h["id"], "category": h["category"], "statement": h["statement"],
            "priority": h.get("priority"), "status": h["status"],
        }
        for h in backlog if h["status"] == "open"
    ]

    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "market_note": MARKET_NOTE,
        "n_specs": len(rows),
        "n_experiments": len(experiments),
        "strategies": rows,
        "open_hypotheses": open_hyps,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {OUT_FILE.relative_to(ROOT)}  "
          f"({len(rows)} strategies, {len(open_hyps)} open hypotheses)")


if __name__ == "__main__":
    main()
