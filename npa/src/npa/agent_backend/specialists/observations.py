"""Dismiss lost observation results atomically without replaying or approving effects."""

from __future__ import annotations

import json
import time

from .call_policy import _require_observation
from .config import fingerprint
from .store import _call


def _profile_calls(connection, profile):
    rows = connection.execute(
        "SELECT c.*,p.classification FROM calls c JOIN tasks t ON c.task_id=t.id "
        "LEFT JOIN call_policies p ON c.task_id=p.task_id AND c.call_id=p.call_id "
        "WHERE t.profile=?",
        (profile.name,),
    ).fetchall()
    return [_call(row) for row in rows]


def _boundary(connection, profile, task_id):
    tasks = connection.execute(
        "SELECT * FROM tasks WHERE profile=?", (profile.name,)
    ).fetchall()
    target = next((task for task in tasks if task["id"] == task_id), None)
    if target is None or target["status"] != "needs_attention":
        raise ValueError("dismissal requires a task awaiting attention")
    if target["policy"] != fingerprint(profile):
        raise ValueError("task policy changed; restore its original policy")
    if any(task["status"] in {"queued", "running"} for task in tasks):
        raise ValueError("profile still has active or queued work")
    return target


def _lost_result(call):
    return {
        "ok": False,
        "error_type": "InterruptedObservation",
        "error": "Observation result was lost; no successful result is available",
        "operation": call["classification"]["operation"],
        "outcome": "failed",
        "replayed": False,
        "dismissed": True,
    }


def _dismiss(store, profile, task_id, call_id):
    with store._connection("observation_dismissal") as connection:
        connection.execute("BEGIN IMMEDIATE")
        task = _boundary(connection, profile, task_id)
        calls = _profile_calls(connection, profile)
        selected = next(
            (
                call
                for call in calls
                if call["task_id"] == task_id and call["call_id"] == call_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("unknown interrupted call")
        _require_observation(selected, profile)
        for call in calls:
            if call["status"] != "completed":
                _require_observation(call, profile)
        result = _lost_result(selected)
        if selected["status"] == "completed":
            if json.loads(selected["result"]) != result:
                raise ValueError("call already has a different result")
            return result
        if selected["status"] != "started":
            raise ValueError("call is not interrupted")
        _write_dismissal(connection, task, selected, result)
        return result


def _write_dismissal(connection, task, call, result):
    connection.execute(
        "UPDATE calls SET status='completed',result=? WHERE task_id=? AND call_id=? AND status='started'",
        (json.dumps(result), task["id"], call["call_id"]),
    )
    event = {
        "type": "interrupted_observation_dismissed",
        "call_id": call["call_id"],
        "classification": call["classification"],
        "previous_status": call["status"],
        "previous_error": task["error"],
        "result": result,
        "task_requeued": False,
        "remote_workloads_cancelled": False,
    }
    connection.execute(
        "INSERT INTO events(task_id,at,body) VALUES(?,?,?)",
        (task["id"], time.time(), json.dumps(event)),
    )
