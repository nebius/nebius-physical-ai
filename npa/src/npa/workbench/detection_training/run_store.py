"""Transactional run records on a service-owned persistent volume.

One service process owns the volume. The process lock is held for its lifetime;
after a crash the next owner can truthfully mark unfinished work interrupted.
SQLite transactions serialize training threads and independent status readers.
"""

from __future__ import annotations

import fcntl
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import StatusResponse

TERMINAL = {"completed", "failed", "interrupted"}


class RunStoreError(RuntimeError):
    """The durable service state cannot safely be read or changed."""


def default_state_dir() -> Path:
    configured = os.environ.get("DETECTION_TRAINING_STATE_DIR")
    root = Path(os.environ.get("NPA_CONFIG_DIR") or Path.home() / ".npa")
    return Path(configured) if configured else root / "detection-training"


class RunStore:
    def __init__(self, directory: str | Path | None = None, *, scope: str = ""):
        self.directory = Path(directory) if directory is not None else default_state_dir()
        self.scope = scope.rstrip("/")
        self._owner_file = None

    @contextmanager
    def connection(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.directory / "runs.sqlite3"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(descriptor)
        # sqlite uses a private journal alongside the database; the parent is private.
        connection = sqlite3.connect(path, timeout=30)
        os.chmod(path, 0o600)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
            connection.commit()
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT value FROM metadata WHERE key='scope'").fetchone()
            if row is None:
                connection.execute("INSERT INTO metadata VALUES ('scope', ?)", (self.scope,))
            elif row[0] != self.scope:
                raise RunStoreError("run store belongs to a different output scope")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def start(self) -> None:
        """Acquire sole process ownership before recovering previous work."""
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        owner_file = (self.directory / "owner.lock").open("a+")
        try:
            fcntl.flock(owner_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner_file.close()
            raise RunStoreError("another detection service process owns this state directory; use one worker") from exc
        self._owner_file = owner_file
        try:
            with self.connection() as connection:
                for run_id, record in connection.execute("SELECT run_id, record FROM runs").fetchall():
                    status = StatusResponse.model_validate_json(record)
                    if status.status not in TERMINAL:
                        updated = status.model_copy(update={
                            "status": "interrupted",
                            "error": "service process stopped before a terminal result was committed; automatic resume is unavailable",
                            "updated_at": _now(), "revision": status.revision + 1,
                        })
                        connection.execute("UPDATE runs SET record=? WHERE run_id=?", (updated.model_dump_json(), run_id))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._owner_file is not None:
            fcntl.flock(self._owner_file.fileno(), fcntl.LOCK_UN)
            self._owner_file.close()
            self._owner_file = None

    def create(self, status: StatusResponse) -> StatusResponse:
        now = _now()
        status = status.model_copy(update={"created_at": now, "updated_at": now, "revision": 1})
        with self.connection() as connection:
            connection.execute("INSERT INTO runs VALUES (?, ?)", (status.run_id, status.model_dump_json()))
        return status

    def get(self, run_id: str) -> StatusResponse:
        with self.connection() as connection:
            row = connection.execute("SELECT record FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return StatusResponse.model_validate_json(row[0])

    def list(self) -> list[StatusResponse]:
        with self.connection() as connection:
            rows = connection.execute("SELECT record FROM runs ORDER BY rowid").fetchall()
        return [StatusResponse.model_validate_json(row[0]) for row in rows]

    def update(self, run_id: str, **changes: Any) -> StatusResponse:
        with self.connection() as connection:
            row = connection.execute("SELECT record FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            current = StatusResponse.model_validate_json(row[0])
            if current.status in TERMINAL:
                raise RunStoreError("terminal run records cannot be overwritten")
            epochs = changes.get("epochs_completed", current.epochs_completed)
            if epochs < current.epochs_completed or epochs > current.total_epochs:
                raise RunStoreError("run progress must be monotonic and within the requested epochs")
            updated = StatusResponse.model_validate({
                **current.model_dump(), **changes, "updated_at": _now(), "revision": current.revision + 1,
            })
            if updated.status == "completed":
                checkpoints = [artifact for artifact in updated.artifacts if artifact.role == "checkpoint" and artifact.epoch == updated.total_epochs]
                metrics = [artifact for artifact in updated.artifacts if artifact.role == "training_metrics"]
                train_complete = epochs == updated.total_epochs and bool(checkpoints) and bool(metrics)
                eval_complete = updated.evaluation is not None and any(artifact.role == "evaluation_metrics" for artifact in updated.artifacts)
                if not (train_complete if updated.kind == "train" else eval_complete) or any(
                    not artifact.exists or not artifact.integrity_verified for artifact in updated.artifacts
                ):
                    raise RunStoreError("completed runs require their result and verified checkpoint/metrics artifacts")
            connection.execute("UPDATE runs SET record=? WHERE run_id=?", (updated.model_dump_json(), run_id))
        return updated


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
