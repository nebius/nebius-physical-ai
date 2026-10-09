"""Persist authenticated run ownership, idempotency, lifecycle, and audit records."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from .errors import ConflictError, RunNotFoundError
from .models import Actor, SubmitRequest


class TeamLedger:
    """Store server-issued ownership independently of user-editable workflow artifacts.

    Args:
        root: Private directory holding the authoritative SQLite database.
    Returns:
        A TeamLedger instance.
    Raises:
        OSError, sqlite3.Error: Private ledger initialization fails.
    """

    def __init__(self, root: Path):
        """Open durable private team state.

        Args:
            root: Persistent server-owned directory.
        Returns:
            None.
        Raises:
            OSError, sqlite3.Error: State cannot be initialized.
        """
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        self.path = root / "team.sqlite3"
        descriptor = os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(descriptor)
        with self._transaction() as db:
            db.executescript(_SCHEMA)

    def create(
        self, actor: Actor, request: SubmitRequest, binding: dict
    ) -> tuple[dict, bool]:
        """Create one run or return its identical authenticated retry.

        Args:
            actor, request: Verified submitter and validated request.
            binding: Immutable, non-secret execution allocation snapshot.
        Returns:
            Run record and whether this call created it.
        Raises:
            ConflictError: An idempotency key was reused for different work.
        """
        body = json.dumps(request.model_dump(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(body.encode()).hexdigest()
        identity = (
            actor.issuer,
            actor.subject,
            request.workspace,
            request.idempotency_key,
        )
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM runs WHERE issuer=? AND subject=? AND workspace=? AND request_key=?",
                identity,
            ).fetchone()
            if row is not None:
                if row["request_hash"] != digest:
                    raise ConflictError(
                        "idempotency key already belongs to a different submission"
                    )
                return dict(row), False
            return self._insert_run(db, actor, request, binding, digest), True

    def _insert_run(self, db, actor, request, binding, digest):
        run_id = "run-" + uuid.uuid4().hex
        db.execute(
            "INSERT INTO runs (id,issuer,subject,workspace,cluster,request_key,request_hash,workflow,binding,status) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                actor.issuer,
                actor.subject,
                request.workspace,
                request.cluster,
                request.idempotency_key,
                digest,
                json.dumps(request.workflow),
                json.dumps(binding),
                "accepted",
            ),
        )
        self._audit(db, actor, "submit", run_id, {})
        return dict(db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone())

    def get(self, run_id: str) -> dict:
        """Read one run after the caller's authorization layer checks ownership.

        Args:
            run_id: Server-issued run identifier.
        Returns:
            Internal run record.
        Raises:
            RunNotFoundError: The run does not exist.
        """
        with self._transaction() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFoundError("run not found")
        return dict(row)

    def list_owned(self, actor: Actor, workspace: str) -> list[dict]:
        """List records owned by one verified external identity.

        Args:
            actor, workspace: Verified identity and authorized workspace.
        Returns:
            The actor's runs in creation order.
        Raises:
            sqlite3.Error: Durable state is unavailable.
        """
        with self._transaction() as db:
            rows = db.execute(
                "SELECT * FROM runs WHERE issuer=? AND subject=? AND workspace=? ORDER BY created_at,id",
                (actor.issuer, actor.subject, workspace),
            ).fetchall()
        return [dict(row) for row in rows]

    def transition(self, run_id: str, expected: tuple[str, ...], status: str) -> bool:
        """Atomically claim one lifecycle transition.

        Args:
            run_id, expected, status: Run and permissible source/destination states.
        Returns:
            Whether the transition won the concurrent claim.
        Raises:
            sqlite3.Error: Durable state is unavailable.
        """
        with self._transaction() as db:
            current = db.execute(
                "SELECT status FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if current is None or current["status"] not in expected:
                return False
            changed = db.execute(
                "UPDATE runs SET status=?,updated_at=CURRENT_TIMESTAMP WHERE id=? AND status=?",
                (status, run_id, current["status"]),
            ).rowcount
        return bool(changed)

    def audit(self, actor: Actor, action: str, run_id: str, detail: dict | None = None):
        """Record a server-authorized operation without token or credential values.

        Args:
            actor, action, run_id, detail: Verified actor and non-secret event fields.
        Returns:
            None.
        Raises:
            sqlite3.Error: The audit event cannot be persisted.
        """
        with self._transaction() as db:
            self._audit(db, actor, action, run_id, detail or {})

    def recover(self) -> None:
        """Mark interrupted workers for explicit reconciliation rather than relaunch.

        Args:
            None.
        Returns:
            None.
        Raises:
            sqlite3.Error: Durable state is unavailable.
        """
        with self._transaction() as db:
            db.execute(
                "UPDATE runs SET status='recovery_required' WHERE status IN ('running','cancelling','accepted')"
            )

    def begin_wave(self, run_id: str, name: str) -> dict:
        """Record a launch intent before a request can reach SkyPilot.

        Args:
            run_id, name: Trusted parent run and engine-generated wave name.
        Returns:
            New immutable launch identity.
        Raises:
            ConflictError: Run is no longer active or this wave already has an intent.
        """
        with self._transaction() as db:
            row = db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None or row["status"] != "running":
                raise ConflictError("run is not accepting new launches")
            if db.execute(
                "SELECT id FROM waves WHERE run_id=? AND name=?", (run_id, name)
            ).fetchone():
                raise ConflictError(
                    "wave already has a launch intent; reconcile it before retry"
                )
            wave_id = "wave-" + uuid.uuid4().hex
            db.execute(
                "INSERT INTO waves (id,run_id,name) VALUES (?,?,?)",
                (wave_id, run_id, name),
            )
            return dict(
                db.execute("SELECT * FROM waves WHERE id=?", (wave_id,)).fetchone()
            )

    def record_wave(
        self, wave_id: str, *, request_id: str = "", job_ids: tuple[int, ...] = ()
    ):
        """Attach exact scheduler acknowledgements to an existing launch intent.

        Args:
            wave_id, request_id, job_ids: Trusted scheduler identifiers.
        Returns:
            None.
        Raises:
            ConflictError: An existing scheduler identity would be replaced.
        """
        with self._transaction() as db:
            row = db.execute("SELECT * FROM waves WHERE id=?", (wave_id,)).fetchone()
            if row is None:
                raise ConflictError("wave intent is missing")
            for column, value, empty, query in (
                (
                    "request_id",
                    request_id,
                    "",
                    "UPDATE waves SET request_id=? WHERE id=?",
                ),
                (
                    "job_ids",
                    json.dumps(list(job_ids)),
                    "[]",
                    "UPDATE waves SET job_ids=? WHERE id=?",
                ),
            ):
                if value == empty:
                    continue
                if row[column] not in (empty, value):
                    raise ConflictError("wave scheduler identity cannot be replaced")
                db.execute(query, (value, wave_id))

    def waves(self, run_id: str) -> list[dict]:
        """Read scheduler identities associated with one authorized run.

        Args:
            run_id: Server-issued parent run.
        Returns:
            Exact recorded wave identities.
        Raises:
            sqlite3.Error: Durable state is unavailable.
        """
        with self._transaction() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM waves WHERE run_id=? ORDER BY rowid", (run_id,)
                )
            ]

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _audit(self, db, actor, action, run_id, detail):
        db.execute(
            "INSERT INTO audit (issuer,subject,action,run_id,detail) VALUES (?,?,?,?,?)",
            (
                actor.issuer,
                actor.subject,
                action,
                run_id,
                json.dumps(detail, sort_keys=True),
            ),
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
 id TEXT PRIMARY KEY, issuer TEXT NOT NULL, subject TEXT NOT NULL,
 workspace TEXT NOT NULL, cluster TEXT NOT NULL, request_key TEXT NOT NULL,
 request_hash TEXT NOT NULL, workflow TEXT NOT NULL, binding TEXT NOT NULL,
 status TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(issuer,subject,workspace,request_key)
);
CREATE TABLE IF NOT EXISTS audit (
 sequence INTEGER PRIMARY KEY AUTOINCREMENT, issuer TEXT NOT NULL,
 subject TEXT NOT NULL, action TEXT NOT NULL, run_id TEXT NOT NULL,
 detail TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS waves (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), name TEXT NOT NULL,
 request_id TEXT NOT NULL DEFAULT '', job_ids TEXT NOT NULL DEFAULT '[]',
 UNIQUE(run_id,name)
);
"""
