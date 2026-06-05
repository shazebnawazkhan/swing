"""
server.py
---------
Lightweight local server for the Swing Scanner comparison report.
Serves the HTML report and exposes a JSON API so the browser UI can
update config and re-run the backtest without touching the terminal.

Usage
-----
    python scripts/server.py
    python scripts/server.py --port 8765
    python scripts/server.py --no-browser
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # repo root

# ── Shared run state (protected by _lock) ─────────────────────────────────────
_lock        = threading.Lock()
_run_status  = "idle"   # idle | running | done | error
_run_log: list[str] = []
_run_started = 0.0


# ── config.py helpers ─────────────────────────────────────────────────────────

def _read_capital() -> int:
    text = (ROOT / "src" / "config.py").read_text(encoding="utf-8")
    m = re.search(r"^CAPITAL\s*=\s*([\d_]+)", text, re.MULTILINE)
    return int(m.group(1).replace("_", "")) if m else 100_000


def _write_capital(value: int) -> None:
    path = ROOT / "src" / "config.py"
    text = path.read_text(encoding="utf-8")
    # Format with underscores for readability (e.g. 500_000)
    formatted = f"{value:_}"
    text = re.sub(
        r"^(CAPITAL\s*=\s*)[\d_]+",
        lambda m: m.group(1) + formatted,
        text, flags=re.MULTILINE,
    )
    path.write_text(text, encoding="utf-8")


# ── Background runner ─────────────────────────────────────────────────────────

def _run_comparison(strategy: str | None = None) -> None:
    global _run_status, _run_log, _run_started
    with _lock:
        _run_status  = "running"
        _run_log     = []
        _run_started = time.time()

    cmd = [sys.executable, str(ROOT / "scripts" / "run_comparison.py")]
    if strategy:
        cmd += ["--strategy", strategy]

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            with _lock:
                _run_log.append(line.rstrip())
        proc.wait()
        with _lock:
            _run_status = "done" if proc.returncode == 0 else "error"
    except Exception as exc:
        with _lock:
            _run_log.append(f"ERROR: {exc}")
            _run_status = "error"


# ── HTTP handler ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress noisy default request logging

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(data))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0]

        if p in ("/", "/report"):
            rpt = ROOT / "outputs" / "comparison_report.html"
            if rpt.exists():
                self._file(rpt, "text/html; charset=utf-8")
            else:
                body = b"<h2>No report yet &mdash; click Re-run Backtest in the UI.</h2>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)

        elif p == "/api/config":
            self._json({"capital": _read_capital()})

        elif p == "/api/status":
            with _lock:
                self._json({
                    "status":  _run_status,
                    "log":     list(_run_log[-80:]),   # last 80 lines
                    "elapsed": round(time.time() - _run_started, 1) if _run_started else 0,
                })

        elif p.startswith("/outputs/"):
            fp = ROOT / p.lstrip("/")
            if fp.exists() and fp.is_file():
                ct = "text/html; charset=utf-8" if fp.suffix == ".html" else "text/plain"
                self._file(fp, ct)
            else:
                self._json({"error": "not found"}, 404)

        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length) or b"{}")
        p      = self.path.split("?")[0]

        if p == "/api/config":
            cap = body.get("capital")
            if not isinstance(cap, int) or cap < 1_000:
                return self._json({"error": "capital must be an integer ≥ 1000"}, 400)
            _write_capital(cap)
            self._json({"ok": True, "capital": cap})

        elif p == "/api/run":
            with _lock:
                if _run_status == "running":
                    return self._json({"error": "already running"}, 409)
            strategy = body.get("strategy") or None
            threading.Thread(target=_run_comparison, args=(strategy,), daemon=True).start()
            self._json({"ok": True})

        else:
            self._json({"error": "not found"}, 404)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Swing Scanner UI server")
    parser.add_argument("--port",       type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open the browser")
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url    = f"http://localhost:{args.port}"
    print()
    print("=" * 52)
    print("  Swing Scanner  —  UI Server")
    print("=" * 52)
    print(f"  URL    : {url}")
    print(f"  Capital: ₹{_read_capital():,}")
    print(f"  Ctrl+C to stop")
    print()

    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
