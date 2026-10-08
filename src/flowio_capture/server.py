"""Localhost JSON endpoint so a Flow.Io plugin in any language can report events.

    POST http://127.0.0.1:8765/event
    Content-Type: application/json
    X-Capture-Token: <token>
    {"event": "cell_changed", "workspace_ref": "ws-1", "row": "FITC", "col": "PE-A",
     "new_value": 0.18, "event_id": "optional-unique-id"}

A body may also be a list of events. They are applied in order, stopping at
the first failure; the response reports how many were applied so the client
resends only the rest (``event_id`` makes resends idempotent anyway):

    200 {"ok": true,  "applied": n, "results": [...]}
    400 {"ok": false, "applied": k, "failed_index": k, "error": "...", "results": [...]}

Hardening: binds to loopback only; requires a token (taken from
$FLOWIO_CAPTURE_TOKEN or generated into ``<root>/.server_token``, mode
0600); requires Content-Type application/json and rejects any request with
an Origin header, so a web page in the user's browser cannot post events;
checks Host to block DNS rebinding.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from .bridge import FlowIoBridge
from .schema import SchemaError

MAX_BODY = 4 * 1024 * 1024
TOKEN_ENV = "FLOWIO_CAPTURE_TOKEN"


def make_server(bridge: FlowIoBridge, token: str, host: str = "127.0.0.1",
                port: int = 8765) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("capture server must bind to loopback")
    if not token:
        raise ValueError("a token is required")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the host app's console quiet
            pass

        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _host_ok(self) -> bool:
            port_ = self.server.server_address[1]
            allowed = {f"127.0.0.1:{port_}", f"localhost:{port_}", f"[::1]:{port_}"}
            return self.headers.get("Host", "") in allowed

        def do_GET(self):
            if self.path == "/health" and self._host_ok():
                self._reply(200, {"ok": True})
            else:
                self._reply(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            try:
                self._post()
            except Exception as ex:  # never drop the connection without a reply
                traceback.print_exc(file=sys.stderr)
                try:
                    self._reply(500, {"ok": False, "error": f"internal error: {type(ex).__name__}"})
                except Exception:
                    pass

        def _post(self):
            if self.path != "/event":
                return self._reply(404, {"ok": False, "error": "not found"})
            if self.headers.get("Origin") is not None or not self._host_ok():
                return self._reply(403, {"ok": False, "error": "forbidden"})
            if not hmac.compare_digest(self.headers.get("X-Capture-Token", ""), token):
                return self._reply(401, {"ok": False, "error": "bad token"})
            ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return self._reply(415, {"ok": False, "error": "Content-Type must be application/json"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n <= 0 or n > MAX_BODY:
                return self._reply(400, {"ok": False, "error": "bad body size"})
            try:
                payload = json.loads(self.rfile.read(n))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._reply(400, {"ok": False, "error": "invalid JSON", "applied": 0})
            events = payload if isinstance(payload, list) else [payload]
            results = []
            for i, ev in enumerate(events):
                try:
                    results.append(bridge.handle(ev))
                except (SchemaError, RuntimeError, ValueError, KeyError) as ex:
                    return self._reply(400, {"ok": False, "applied": i, "failed_index": i,
                                             "error": str(ex), "results": results})
            self._reply(200, {"ok": True, "applied": len(results), "results": results})

    return ThreadingHTTPServer((host, port), Handler)


def load_or_create_token(root: os.PathLike) -> str:
    env = os.environ.get(TOKEN_ENV)
    if env:
        return env
    path = Path(root) / ".server_token"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(secrets.token_urlsafe(32))
    return path.read_text().strip()


def serve(root: str, port: int = 8765, token: Optional[str] = None) -> None:
    bridge = FlowIoBridge(root)
    token = token or load_or_create_token(root)
    srv = make_server(bridge, token=token, port=port)
    print(f"flowio-capture listening on http://127.0.0.1:{port}/event  (store: {root}; "
          f"token in ${TOKEN_ENV} or {Path(root) / '.server_token'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close_all("server_stopped")
        srv.server_close()
