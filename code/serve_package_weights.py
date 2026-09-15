"""Loopback-only, allow-listed, read-only range server for kubectl port-forward."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
from pathlib import Path
import re
import sys

API_PROXY = "--api-proxy" in sys.argv
TOKEN = sys.stdin.readline().strip() if API_PROXY else None
if API_PROXY:
    assert TOKEN and len(TOKEN) == 64

ROOT = Path("/workspace/clear-ec-research/experiments/region_candidate_20260907")
STAGE = ROOT / "experimental_stage"
FILES = {k: v for k, v in json.loads((ROOT / "stage_manifest.json").read_text())["files"].items()
         if k.startswith("model/") and "anchor_quality" not in k}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if API_PROXY and not hmac.compare_digest(self.headers.get("X-Clearec-Key", ""), TOKEN):
            self.send_error(403)
            return
        name = self.path.removeprefix("/")
        if name not in FILES:
            self.send_error(404)
            return
        path = STAGE / name
        size = FILES[name]["bytes"]
        assert path.stat().st_size == size
        start, end = 0, size - 1
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
            if not match:
                self.send_error(416)
                return
            start = int(match[1])
            end = min(int(match[2]), end) if match[2] else end
            if not 0 <= start <= end < size:
                self.send_error(416)
                return
        self.send_response(206 if requested else 200)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = handle.read(min(1024**2, remaining))
                    if not block:
                        raise IOError("Unexpected EOF")
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass


ThreadingHTTPServer(("0.0.0.0" if API_PROXY else "127.0.0.1", 8765), Handler).serve_forever()
