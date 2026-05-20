"""Flask server for receiving and serving encrypted chunks.

Endpoints (all require Authorization: Bearer <INNOUT_API_KEY>):
  POST /upload
    Form fields: session_id, part (zero-padded 3 digits), total_parts
    File field:  file (binary chunk data)

  GET /manifest/<session_id>
    Returns: {"session_id": "...", "parts": ["000", "001", ...]}

  GET /download/<session_id>/<part>
    Returns: binary chunk data

  GET /
    Returns: server status (no session list exposed)

Security:
  - Bearer token auth on every endpoint (INNOUT_API_KEY env var)
  - session_id validated as UUID, part validated as 3-digit string
  - Per-IP rate limiting via Flask-Limiter
  - 2 GB per-chunk upload cap

Usage:
  INNOUT_API_KEY=<secret> uv run innout-server --store /data/chunks --port 8000
"""

from __future__ import annotations

import argparse
import hmac
import os
import re
from pathlib import Path

from flask import Flask, jsonify, request, send_file, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_PART_RE = re.compile(r"^\d{3}$")

# 2 GB hard ceiling per chunk; adjust if you need bigger slabs
_MAX_CHUNK_BYTES = 2 * 1024 * 1024 * 1024


def _validate_session_id(value: str | None) -> str:
    if not value or not _UUID_RE.match(value):
        abort(400, "session_id must be a UUID")
    return value


def _validate_part(value: str | None) -> str:
    if not value or not _PART_RE.match(value):
        abort(400, "part must be a 3-digit string (e.g. 000)")
    return value


def create_app(store_dir: str | Path, api_key: str | None = None) -> Flask:
    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)

    resolved_key = api_key or os.environ.get("INNOUT_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "INNOUT_API_KEY is required. "
            "Pass --api-key or set the INNOUT_API_KEY environment variable."
        )

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = _MAX_CHUNK_BYTES

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per minute"],
        storage_uri="memory://",
    )

    def _require_auth() -> None:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            abort(401, "Authorization header required: Bearer <api-key>")
        token = auth[len("Bearer "):]
        # Constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(token, resolved_key):
            abort(403, "Invalid API key")

    @app.get("/")
    def index():
        _require_auth()
        return jsonify({
            "status": "innout-server running",
            "endpoints": {
                "upload": "POST /upload",
                "manifest": "GET /manifest/<session_id>",
                "download": "GET /download/<session_id>/<part>",
            },
        })

    @app.post("/upload")
    @limiter.limit("60 per minute")
    def upload():
        _require_auth()
        session_id = _validate_session_id(request.form.get("session_id"))
        part = _validate_part(request.form.get("part"))

        if "file" not in request.files:
            abort(400, "Missing file field")

        session_dir = store / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        dest = session_dir / f"part{part}"
        request.files["file"].save(dest)

        return jsonify({"status": "ok", "session_id": session_id, "part": part})

    @app.get("/manifest/<session_id>")
    def manifest(session_id: str):
        _require_auth()
        session_id = _validate_session_id(session_id)

        session_dir = store / session_id
        if not session_dir.is_dir():
            abort(404, "Session not found")

        parts = sorted(
            p.name.removeprefix("part")
            for p in session_dir.iterdir()
            if p.name.startswith("part") and _PART_RE.match(p.name.removeprefix("part"))
        )
        return jsonify({"session_id": session_id, "parts": parts})

    @app.get("/download/<session_id>/<part>")
    def download(session_id: str, part: str):
        _require_auth()
        session_id = _validate_session_id(session_id)
        part = _validate_part(part)

        chunk = store / session_id / f"part{part}"
        if not chunk.is_file():
            abort(404, "Chunk not found")
        return send_file(chunk, mimetype="application/octet-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(prog="innout-server")
    parser.add_argument("--store", default="/tmp/innout-store", metavar="<dir>",
                        help="Directory to store uploaded chunks (default: /tmp/innout-store)")
    parser.add_argument("--port", type=int, default=8000, metavar="<port>",
                        help="Port to listen on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", metavar="<host>",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--api-key", default=None, metavar="<key>",
                        help="API key (overrides INNOUT_API_KEY env var)")
    parser.add_argument("--workers", type=int, default=4, metavar="<n>",
                        help="Number of gunicorn worker processes (default: 4)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    app = create_app(args.store, api_key=args.api_key)

    from gunicorn.app.base import BaseApplication

    class _App(BaseApplication):
        def load_config(self):
            self.cfg.set("bind", f"{args.host}:{args.port}")
            self.cfg.set("workers", args.workers)

        def load(self):
            return app

    print(f"Starting innout-server on {args.host}:{args.port} "
          f"(gunicorn, {args.workers} workers), store={args.store}")
    _App().run()


if __name__ == "__main__":
    main()
