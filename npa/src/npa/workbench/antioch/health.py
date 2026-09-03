"""Local HTTP transport for fail-closed Antioch state probes."""

from __future__ import annotations

import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class StateHealthServer:
    """Expose fixed readiness/liveness checks without kubelet exec RPCs."""

    def __init__(self, *, port: int, checks: dict[str, Callable[[], bool]]) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("health port is invalid")
        if set(checks) != {"/ready", "/live"}:
            raise ValueError("health server requires ready and live checks")
        self._checks = checks
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                check = owner._checks.get(self.path)
                healthy = False
                if check is not None:
                    try:
                        healthy = bool(check())
                    except (OSError, TypeError, ValueError):
                        healthy = False
                status = 200 if healthy else (404 if check is None else 503)
                body = b"ok\n" if healthy else b"not ready\n"
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"antioch-state-health-{port}",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def close(self) -> None:
        if self._started:
            self._server.shutdown()
        self._server.server_close()
        if self._started:
            self._thread.join(timeout=5)
            self._started = False
