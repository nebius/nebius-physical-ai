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
        self.database = sqlite3.connect(path, check_same_thread=False)
        self.database.execute(
            "CREATE TABLE IF NOT EXISTS deliveries "
            "(id TEXT PRIMARY KEY, digest TEXT NOT NULL, result TEXT)"
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
        if not isinstance(identifier, str):
            raise ValueError("A send identity must be a UUID string.")
        uuid.UUID(identifier)
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
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
