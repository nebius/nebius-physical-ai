"""Persist task ownership and tool receipts independently of graph checkpoints."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

from .config import private_directory

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
 id TEXT PRIMARY KEY, profile TEXT NOT NULL, policy TEXT NOT NULL,
 request TEXT NOT NULL, goal TEXT NOT NULL, status TEXT NOT NULL,
 paused INTEGER NOT NULL DEFAULT 0, result TEXT NOT NULL DEFAULT '',
 error TEXT NOT NULL DEFAULT '', route TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS profiles (name TEXT PRIMARY KEY, paused INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS workers (name TEXT PRIMARY KEY, pid INTEGER NOT NULL, seen REAL NOT NULL);
CREATE TABLE IF NOT EXISTS calls (
 task_id TEXT NOT NULL, call_id TEXT NOT NULL, digest TEXT NOT NULL,
 status TEXT NOT NULL, result TEXT, PRIMARY KEY(task_id,call_id));
CREATE TABLE IF NOT EXISTS events (
 sequence INTEGER PRIMARY KEY, task_id TEXT NOT NULL, at REAL NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS originals (
 task_id TEXT NOT NULL, path TEXT NOT NULL, content TEXT, PRIMARY KEY(task_id,path));
"""


class UncertainOperation(RuntimeError):
    """An interrupted invocation needs operator reconciliation before it can proceed.

    Args: message: Diagnostic without credential values.
    Returns: Exception instance.
    Raises: None.
    """


class TaskStore:
    """A single-host SQLite task queue and durable side-effect journal.

    Args: directory: Private state directory outside source workspaces.
    Returns: Store with independent transactional connections per operation.
    Raises: ValueError, OSError, sqlite3.Error: Storage is unsafe or unavailable.
    """

    def __init__(self, directory: Path):
        self.directory = private_directory(directory)
        self.path = self.directory / "tasks.sqlite"
        _private_file(self.path)
        with self._connection() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                yield connection
        finally:
            connection.close()

    def _get(self, task_id):
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError("unknown task")
        return _task(row)

    def _list(self):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY created DESC"
            ).fetchall()
        return [_task(row) for row in rows]

    def _submit(self, task_id, profile, policy, request, goal, route):
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is not None:
                if row["request"] != request:
                    raise ValueError("task id already belongs to a different request")
                return _task(row)
            connection.execute(
                "INSERT INTO tasks(id,profile,policy,request,goal,status,route,created) "
                "VALUES(?,?,?,?,?,'queued',?,?)",
                (
                    task_id,
                    profile,
                    policy,
                    request,
                    goal,
                    json.dumps(route),
                    time.time(),
                ),
            )
        return self._get(task_id)

    def _next(self, profile):
        with self._connection() as connection:
            paused = connection.execute(
                "SELECT paused FROM profiles WHERE name=?", (profile,)
            ).fetchone()
            if paused and paused[0]:
                return None
            row = connection.execute(
                "SELECT * FROM tasks WHERE profile=? "
                "AND status IN ('queued','running','needs_attention') ORDER BY created LIMIT 1",
                (profile,),
            ).fetchone()
        if row and (row["paused"] or row["status"] == "needs_attention"):
            return None
        return _task(row) if row else None

    def _heartbeat(self, profile):
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO workers VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET pid=excluded.pid,seen=excluded.seen",
                (profile, os.getpid(), time.time()),
            )

    def _workers(self):
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM workers").fetchall()
        return {row["name"]: dict(row) for row in rows}

    def _update(self, task_id, status, *, result="", error=""):
        with self._connection() as connection:
            connection.execute(
                "UPDATE tasks SET status=?,result=?,error=? WHERE id=? AND status<>'cancelled'",
                (status, result, error, task_id),
            )

    def _pause_task(self, task_id, paused):
        task = self._get(task_id)
        if not paused and task["status"] == "needs_attention":
            raise ValueError("reconcile the task before resuming")
        with self._connection() as connection:
            connection.execute(
                "UPDATE tasks SET paused=? WHERE id=?", (int(paused), task_id)
            )
        return self._get(task_id)

    def _pause_profile(self, name, paused):
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO profiles VALUES(?,?) ON CONFLICT(name) DO UPDATE SET paused=excluded.paused",
                (name, int(paused)),
            )

    def _paused_profiles(self):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT name FROM profiles WHERE paused=1"
            ).fetchall()
        return {row[0] for row in rows}

    def _event(self, task_id, body):
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO events(task_id,at,body) VALUES(?,?,?)",
                (task_id, time.time(), json.dumps(body)),
            )

    def _events(self, task_id):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE task_id=? ORDER BY sequence", (task_id,)
            ).fetchall()
        return [
            {"sequence": row["sequence"], "at": row["at"], **json.loads(row["body"])}
            for row in rows
        ]

    def _begin_call(self, task_id, call_id, invocation):
        digest = hashlib.sha256(
            json.dumps(invocation, sort_keys=True).encode()
        ).hexdigest()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM calls WHERE task_id=? AND call_id=?", (task_id, call_id)
            ).fetchone()
            if row:
                if row["digest"] != digest:
                    raise ValueError("tool call identity changed")
                if row["status"] != "completed":
                    raise UncertainOperation(
                        f"Reconcile interrupted tool call {call_id}"
                    )
                return json.loads(row["result"])
            connection.execute(
                "INSERT INTO calls VALUES(?,?,?,'started',NULL)",
                (task_id, call_id, digest),
            )
        return None

    def _finish_call(self, task_id, call_id, result):
        with self._connection() as connection:
            connection.execute(
                "UPDATE calls SET status='completed',result=? WHERE task_id=? AND call_id=?",
                (json.dumps(result), task_id, call_id),
            )

    def _calls(self, task_id):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT call_id,status,digest FROM calls WHERE task_id=?", (task_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def _reconcile(self, task_id, call_id, result, retry):
        if self._get(task_id)["status"] != "needs_attention":
            raise ValueError("task is not awaiting reconciliation")
        with self._connection() as connection:
            pending = connection.execute(
                "SELECT call_id FROM calls WHERE task_id=? AND status='started'",
                (task_id,),
            ).fetchall()
            if pending and call_id not in {row[0] for row in pending}:
                raise ValueError("provide the interrupted call id")
            if pending and retry:
                connection.execute(
                    "DELETE FROM calls WHERE task_id=? AND call_id=?",
                    (task_id, call_id),
                )
            elif pending:
                connection.execute(
                    "UPDATE calls SET status='completed',result=? WHERE task_id=? AND call_id=?",
                    (json.dumps(result), task_id, call_id),
                )
            connection.execute(
                "UPDATE tasks SET status='queued',error='' WHERE id=?", (task_id,)
            )
        self._event(
            task_id, {"type": "reconciliation", "call_id": call_id, "retry": retry}
        )

    def _original(self, task_id, path, content):
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO originals VALUES(?,?,?)",
                (task_id, path, content),
            )

    def _originals(self, task_id):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT path,content FROM originals WHERE task_id=? ORDER BY path",
                (task_id,),
            ).fetchall()
        return [dict(row) for row in rows]


def _task(row):
    result = dict(row)
    result["route"] = json.loads(result["route"])
    result["paused"] = bool(result["paused"])
    return result


def _private_file(path):
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise ValueError("runtime files must be private and singly linked")
    finally:
        os.close(descriptor)
