"""One-session-at-a-time TCP relay for Antioch's authenticated port tunnel."""

from __future__ import annotations

import socket
import threading
from types import TracebackType


class ReversePolicyRelay:
    """Pair a local tunnel backend with the hosted OpenPI WebSocket client."""

    BACKEND_BIND_HOST = "0.0.0.0"  # noqa: S104 - authenticated declared port
    FRONTEND_BIND_HOST = "127.0.0.1"

    def __init__(self, *, backend_port: int, frontend_port: int) -> None:
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._listeners: list[socket.socket] = []

    def __enter__(self) -> ReversePolicyRelay:
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("private policy relay did not become ready")
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        for listener in self._listeners:
            listener.close()
        if self._thread is not None:
            self._thread.join(10)

    def _listen(self, host: str, port: int) -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        listener.settimeout(1)
        self._listeners.append(listener)
        return listener

    def _accept(self, listener: socket.socket) -> socket.socket | None:
        while not self._stop.is_set():
            try:
                connection, _address = listener.accept()
                connection.settimeout(None)
                return connection
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return None
                raise
        return None

    def _serve(self) -> None:
        # The authenticated Antioch port tunnel enters through the service
        # network interface, while the policy frontend must remain accessible
        # only to the scenario process in this container.
        backend_listener = self._listen(self.BACKEND_BIND_HOST, self.backend_port)
        frontend_listener = self._listen(self.FRONTEND_BIND_HOST, self.frontend_port)
        self._ready.set()
        while not self._stop.is_set():
            backend = self._accept(backend_listener)
            if backend is None:
                return
            frontend = self._accept(frontend_listener)
            if frontend is None:
                backend.close()
                return
            self._pipe_pair(frontend, backend)

    @staticmethod
    def _pipe_pair(left: socket.socket, right: socket.socket) -> int:
        done = threading.Event()
        byte_count = 0
        byte_count_lock = threading.Lock()

        def pump(source: socket.socket, target: socket.socket) -> None:
            nonlocal byte_count
            try:
                while not done.is_set():
                    chunk = source.recv(1024 * 1024)
                    if not chunk:
                        return
                    target.sendall(chunk)
                    with byte_count_lock:
                        byte_count += len(chunk)
            except OSError:
                return
            finally:
                done.set()
                try:
                    target.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        workers = [
            threading.Thread(target=pump, args=(left, right), daemon=True),
            threading.Thread(target=pump, args=(right, left), daemon=True),
        ]
        for worker in workers:
            worker.start()
        done.wait()
        left.close()
        right.close()
        for worker in workers:
            worker.join(1)
        return byte_count
