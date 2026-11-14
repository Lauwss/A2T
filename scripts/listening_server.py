"""Local HTTP server for the blind listening test page.

Serves ``demo/listening_test.html`` at ``/`` and the exported stimulus
directory (WAV files plus ``manifest.json`` / ``answers.json``) at the root,
so the browser page can play audio and load its manifest -- opening the HTML
via ``file://`` does not work because fetch and audio decoding need http.

``POST /results`` stores a finished session as
``artifacts/listening_test/listening_results_<timestamp>.json``.

The server binds to 127.0.0.1 only and serves files from the export
directory by bare filename; nothing else on disk is reachable.

Example
-------
    .venv\\Scripts\\python.exe scripts/listening_server.py --dir artifacts/probe_test --open
"""

from __future__ import annotations

import argparse
import json
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_RESULTS_BYTES = 1_000_000
DATA_SUFFIXES = (".wav", ".json")


def make_handler(html_path: Path, data_dir: Path, results_dir: Path) -> type:
    """Build a request handler bound to fixed, whitelisted locations."""

    class ListeningHandler(BaseHTTPRequestHandler):
        server_version = "A2TListening/1.0"

        def log_message(self, fmt: str, *args: object) -> None:  # keep the console quiet
            pass

        # ------------------------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

        def _send_data_file(self, name: str, ctype: str) -> None:
            safe = Path(name).name  # strip any directory parts
            path = data_dir / safe
            if path.suffix.lower() not in DATA_SUFFIXES or not path.is_file():
                self._send_json(404, {"error": f"not found: {safe}"})
                return
            self._send(200, path.read_bytes(), ctype)

        # ------------------------------------------------------------------
        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route in ("/", "/index.html"):
                self._send(200, html_path.read_bytes(), "text/html; charset=utf-8")
            elif route == "/manifest.json":
                self._send_data_file("manifest.json", "application/json")
            elif route == "/answers.json":
                self._send_data_file("answers.json", "application/json")
            elif route.endswith(".wav"):
                self._send_data_file(route.lstrip("/"), "audio/wav")
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/results":
                self._send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_RESULTS_BYTES:
                self._send_json(400, {"error": "bad content length"})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": f"invalid JSON: {exc}"})
                return
            results_dir.mkdir(parents=True, exist_ok=True)
            name = f"listening_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
            out_path = results_dir / name
            suffix = 1
            while out_path.exists():
                suffix += 1
                out_path = results_dir / f"listening_results_{time.strftime('%Y%m%d_%H%M%S')}_{suffix}.json"
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"results saved: {out_path}")
            self._send_json(200, {"saved": name})

    return ListeningHandler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=REPO_ROOT / "artifacts" / "probe_test",
                        help="stimulus export directory (probe_test.py --export)")
    parser.add_argument("--html", type=Path, default=REPO_ROOT / "demo" / "listening_test.html")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "artifacts" / "listening_test")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = parser.parse_args(argv)
    if not args.dir.is_dir():
        parser.error(f"--dir does not exist: {args.dir}")
    if not args.html.is_file():
        parser.error(f"--html does not exist: {args.html}")
    if args.port < 1024 or args.port > 65535:
        parser.error("--port must be in 1024..65535")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    handler = make_handler(args.html.resolve(), args.dir.resolve(), args.results_dir.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving {args.html.name} + {args.dir} at {url}")
    print("Ctrl+C stops the server.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
