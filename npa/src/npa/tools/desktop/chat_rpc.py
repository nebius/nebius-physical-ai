"""Keep a private, persistent JSON-RPC connection to the shared Codex runtime."""

from collections import deque
from concurrent.futures import Future
import json
import threading
import uuid

from websockets.sync.client import unix_connect
from websockets.exceptions import ConnectionClosed


class CodexConnection:
    """Share one Codex connection across authenticated browser clients.

    Args:
        socket: Private app-server Unix socket.
        on_notification: Optional callback for shared runtime notifications.
    Returns:
        None.
    Raises:
        OSError: The proxy cannot start.
        RuntimeError: Initialization fails.
    """

    def __init__(self, socket, on_notification=None):
        # Codex's Unix listener rejects a permessage-deflate negotiation.
        self.connection = unix_connect(
            socket, uri="ws://localhost", max_size=None, compression=None
        )
        self.closed = threading.Event()
        self.instance = str(uuid.uuid4())
        self.condition = threading.Condition()
        self.write_lock = threading.Lock()
        self.futures = {}
        self.requests = {}
        self.events = deque(maxlen=2048)
        self.sequence = 0
        self.counter = 0
        self.alive = True
        self.on_notification = on_notification
        threading.Thread(target=self._read, daemon=True).start()
        self.call(
            "initialize",
            {
                "clientInfo": {"name": "npa_desktop_chat", "version": "1.0.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        self._write({"method": "initialized", "params": {}})

    def _write(self, message):
        with self.write_lock:
            if not self.alive:
                raise RuntimeError("Codex connection closed; reconnecting is required.")
            self.connection.send(json.dumps(message))

    def call(self, method, params=None):
        """Call Codex without retrying mutations whose outcome might be unknown.

        Args:
            method: App-server method.
            params: Method parameters.
        Returns:
            The app-server result.
        Raises:
            RuntimeError: Codex reports a failure or disconnects.
        """
        with self.condition:
            self.counter += 1
            identifier = self.counter
            future = self.futures[identifier] = Future()
        try:
            self._write({"id": identifier, "method": method, "params": params or {}})
            return future.result()
        finally:
            with self.condition:
                self.futures.pop(identifier, None)

    def _read(self):
        try:
            for line in self.connection:
                self._receive(json.loads(line))
        except (OSError, ValueError, ConnectionClosed):
            pass
        finally:
            with self.condition:
                self.alive = False
                for future in self.futures.values():
                    if not future.done():
                        future.set_exception(
                            RuntimeError("Codex disconnected; refresh before retrying.")
                        )
                self.condition.notify_all()
                self.closed.set()

    def _receive(self, message):
        with self.condition:
            if "method" not in message:
                future = self.futures.get(message.get("id"))
                if future is not None and not future.done():
                    if "error" in message:
                        future.set_exception(RuntimeError(message["error"]["message"]))
                    else:
                        future.set_result(message.get("result", {}))
                return
            if "id" in message:
                self.requests[str(message["id"])] = message
            if message["method"] == "serverRequest/resolved":
                self.requests.pop(str(message["params"]["requestId"]), None)
            self.sequence += 1
            self.events.append((self.sequence, message))
            if self.on_notification is not None:
                self.on_notification(message)
            self.condition.notify_all()

    def answer(self, identifier, result):
        """Answer an outstanding request after an explicit browser action.

        Args:
            identifier: Outstanding server request ID.
            result: Validated answer object.
        Returns:
            None.
        Raises:
            ValueError: Another client already answered.
        """
        with self.condition:
            request = self.requests.get(str(identifier))
            if request is None:
                raise ValueError("This request has already been resolved.")
            self._write({"id": request["id"], "result": result})
            self.requests.pop(str(identifier), None)

    def updates(self, after):
        """Wait for notifications, retaining a cursor across browser reconnects.

        Args:
            after: Last received event sequence.
        Returns:
            A cursor, event list, and history-reset indicator.
        Raises:
            None.
        """
        with self.condition:
            self.condition.wait_for(lambda: self.sequence > after or not self.alive, 20)
            reset = bool(self.events and after < self.events[0][0] - 1)
            events = [event for sequence, event in self.events if sequence > after]
            return self.sequence, events, reset

    def pending(self):
        """Snapshot outstanding requests.

        Args:
            None.
        Returns:
            Requests awaiting explicit answers.
        Raises:
            None.
        """
        with self.condition:
            return list(self.requests.values())
