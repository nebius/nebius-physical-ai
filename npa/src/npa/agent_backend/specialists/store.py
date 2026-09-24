"""Persist task ownership and tool receipts independently of graph checkpoints."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import time

from .config import private_directory
from .call_policy import _digest
from .storage_errors import StorageFailure

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
CREATE TABLE IF NOT EXISTS call_policies (
 task_id TEXT NOT NULL, call_id TEXT NOT NULL, classification TEXT NOT NULL,
 PRIMARY KEY(task_id,call_id));
CREATE TRIGGER IF NOT EXISTS immutable_call_policy
 BEFORE UPDATE ON call_policies BEGIN SELECT RAISE(ABORT, 'immutable call policy'); END;
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
    Raises: ValueError, OSError, StorageFailure: Storage is unsafe or unavailable.
    """

    def __init__(self, directory: Path):
        self.directory = private_directory(directory)
        self.path = self.directory / "tasks.sqlite"
        _private_file(self.path)
        with self._connection() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connection(self, phase="task_journal"):
        connection = None
        try:
            connection = sqlite3.connect(self.path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            with connection:
                yield connection
        except sqlite3.Error as error:
            raise StorageFailure(phase, error) from None
        finally:
            if connection is not None:
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
        with self._connection("worker_heartbeat") as connection:
            connection.execute(
                "INSERT INTO workers VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET pid=excluded.pid,seen=excluded.seen",
                (profile, os.getpid(), time.time()),
            )

    def _model_route(self, task_id, selection):
        with self._connection("model_routing") as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT route FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError("unknown task")
            route = {**json.loads(row[0]), "model_selection": selection}
            connection.execute(
                "UPDATE tasks SET route=? WHERE id=?", (json.dumps(route), task_id)
            )
            connection.execute(
                "INSERT INTO events(task_id,at,body) VALUES(?,?,?)",
                (
                    task_id,
                    time.time(),
                    json.dumps({"type": "model_routing", **selection}),
                ),
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

    def _begin_call(self, task_id, call_id, invocation, *, classification=None):
        digest = _digest(invocation)
        with self._connection("tool_start") as connection:
            connection.execute("BEGIN IMMEDIATE")
            _check_classification(connection, task_id, call_id, classification, digest)
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
            if classification is not None:
                _record_classification(connection, task_id, call_id, classification)
        return None

    def _finish_call(self, task_id, call_id, result):
        with self._connection("tool_receipt") as connection:
            changed = connection.execute(
                "UPDATE calls SET status='completed',result=? WHERE task_id=? AND call_id=? AND status='started'",
                (json.dumps(result), task_id, call_id),
            )
            if changed.rowcount != 1:
                raise UncertainOperation("tool receipt is already resolved or unknown")

    def _calls(self, task_id):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT c.call_id,c.status,c.digest,p.classification FROM calls c "
                "LEFT JOIN call_policies p USING(task_id,call_id) WHERE c.task_id=?",
                (task_id,),
            ).fetchall()
        return [_call(row) for row in rows]

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


def _check_classification(connection, task_id, call_id, classification, digest):
    previous = connection.execute(
        "SELECT classification FROM call_policies WHERE task_id=? AND call_id=?",
        (task_id, call_id),
    ).fetchone()
    if classification is not None and classification.get("digest") != digest:
        raise ValueError("call classification does not match invocation")
    if previous and json.loads(previous[0]) != classification:
        raise ValueError("original call classification changed")


def _record_classification(connection, task_id, call_id, classification):
    connection.execute(
        "INSERT OR IGNORE INTO call_policies VALUES(?,?,?)",
        (task_id, call_id, json.dumps(classification, sort_keys=True)),
    )
    connection.execute(
        "INSERT INTO events(task_id,at,body) VALUES(?,?,?)",
        (
            task_id,
            time.time(),
            json.dumps(
                {
                    "type": "tool_started",
                    "call_id": call_id,
                    "classification": classification,
                }
            ),
        ),
    )


def _call(row):
    result = dict(row)
    result["classification"] = (
        json.loads(result["classification"]) if result["classification"] else None
    )
    return result


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
