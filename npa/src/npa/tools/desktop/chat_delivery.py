"""Remember mutation outcomes across reconnects without replaying uncertain sends."""

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import uuid


class Deliveries:
    """Persist send identities and outcomes in private runtime state.

    Args:
        path: Private SQLite journal path.
    Returns:
        None.
    Raises:
        OSError: Private storage is unavailable.
        sqlite3.Error: The journal cannot be opened.
    """

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        path.chmod(0o600)
        self.lock = threading.Lock()
        self.inflight = set()
        self.errors = {}
        self.database = sqlite3.connect(path, check_same_thread=False)
        self.database.execute(
            "CREATE TABLE IF NOT EXISTS deliveries "
            "(id TEXT PRIMARY KEY, digest TEXT NOT NULL, result TEXT)"
        )
        self.database.execute(
            "CREATE TABLE IF NOT EXISTS delivery_threads "
            "(id TEXT PRIMARY KEY, thread_id TEXT NOT NULL)"
        )
        self.database.commit()

    def execute(self, identifier, body, action):
        """Run a send once, returning its saved outcome on a retry.

        Args:
            identifier: Browser-generated UUID retained until delivery is known.
            body: Send content, stored only as a digest.
            action: Callable that performs the mutation.
        Returns:
            The original successful result.
        Raises:
            ValueError: An identity is invalid or reused for different content.
            RuntimeError: An earlier send has an uncertain outcome.
        """
        digest = self._digest(identifier, body)
        cached = self._reserve(identifier, digest)
        if cached is not None:
            return json.loads(cached)
        result = action()
        with self.lock, self.database:
            self.database.execute(
                "UPDATE deliveries SET result=? WHERE id=?",
                (json.dumps(result), identifier),
            )
        return result

    @staticmethod
    def _digest(identifier, body):
        if not isinstance(identifier, str):
            raise ValueError("A send identity must be a UUID string.")
        uuid.UUID(identifier)
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def submit(self, identifier, body, action):
        """Reserve a send durably and acknowledge it before waiting for Codex.

        Args:
            identifier: Stable browser-generated message UUID.
            body: Original message, including its conversation identity.
            action: Callable performing the mutation exactly once.
        Returns:
            A receipt describing pending, complete, or uncertain delivery.
        Raises:
            ValueError: The identity is invalid or reused for another message.
            sqlite3.Error: The durable reservation cannot be written.
        """
        digest = self._digest(identifier, body)
        with self.lock, self.database:
            row = self.database.execute(
                "SELECT digest FROM deliveries WHERE id=?", (identifier,)
            ).fetchone()
            if row and row[0] != digest:
                raise ValueError("This send identity belongs to another message.")
            self.database.execute(
                "INSERT OR IGNORE INTO delivery_threads VALUES (?, ?)",
                (identifier, body["id"]),
            )
            if not row:
                self.database.execute(
                    "INSERT INTO deliveries(id, digest) VALUES (?, ?)",
                    (identifier, digest),
                )
                self.inflight.add(identifier)
        if not row:
            threading.Thread(
                target=self._complete, args=(identifier, action), daemon=True
            ).start()
        return self.status(identifier)

    def _complete(self, identifier, action):
        try:
            self.confirm(identifier, action())
        except Exception as error:
            # An owner may have accepted the prompt before its reply was lost.
            with self.lock:
                self.errors[identifier] = str(error)
        finally:
            with self.lock:
                self.inflight.discard(identifier)

    def confirm(self, identifier, result):
        """Record a response or an exact message identity observed in history.

        Args:
            identifier: Previously reserved message UUID.
            result: Confirmed delivery outcome.
        Returns:
            None.
        Raises:
            sqlite3.Error: The confirmation cannot be persisted.
        """
        with self.lock, self.database:
            self.database.execute(
                "UPDATE deliveries SET result=? WHERE id=? AND result IS NULL",
                (json.dumps(result), identifier),
            )

    def status(self, identifier):
        """Read a receipt without retrying its mutation.

        Args:
            identifier: Browser-generated message UUID.
        Returns:
            Missing, pending, complete, or uncertain state and known outcome.
        Raises:
            ValueError: The identity is invalid.
            sqlite3.Error: The journal cannot be read.
        """
        self._digest(identifier, {})
        with self.lock:
            row = self.database.execute(
                "SELECT d.result, t.thread_id FROM deliveries d "
                "LEFT JOIN delivery_threads t ON d.id=t.id WHERE d.id=?",
                (identifier,),
            ).fetchone()
            if row is None:
                return {"state": "missing"}
            if row[0] is not None:
                return {"state": "complete", "result": json.loads(row[0])}
            return {
                "state": "pending" if identifier in self.inflight else "uncertain",
                "threadId": row[1],
                "error": self.errors.get(identifier),
            }

    def _reserve(self, identifier, digest):
        with self.lock, self.database:
            row = self.database.execute(
                "SELECT digest, result FROM deliveries WHERE id=?", (identifier,)
            ).fetchone()
            if row:
                if row[0] != digest:
                    raise ValueError("This send identity belongs to another message.")
                if row[1] is None:
                    raise RuntimeError(
                        "Delivery is uncertain. Check the conversation before starting a new send."
                    )
                return row[1]
            self.database.execute(
                "INSERT INTO deliveries(id, digest) VALUES (?, ?)", (identifier, digest)
            )
        return None
