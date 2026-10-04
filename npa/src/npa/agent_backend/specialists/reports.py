"""Build compact evidence handoffs and wait for actionable task states without inference."""

from __future__ import annotations

import hashlib
import json
import math
import threading

from .tools import WorkbenchTools
from .config import fingerprint

_ATTENTION = {"completed", "needs_attention", "cancelled"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _changes(store, task, profile):
    tools = WorkbenchTools(profile, store, task["id"])
    recorded = {
        event["result"]["path"]: event["result"].get("sha256")
        for event in task["events"]
        if event.get("type") == "tool"
        and event.get("name") == "edit_file"
        and event.get("result", {}).get("ok") is True
    }
    changes = []
    for original in store._originals(task["id"]):
        path = original["path"]
        try:
            current = hashlib.sha256(
                tools._path(path, write=True).read_bytes()
            ).hexdigest()
        except (OSError, ValueError):
            current = None
        changes.append(
            {
                "path": path,
                "sha256": current,
                "matches_last_edit": current is not None
                and current == recorded.get(path),
                "expected_sha256": recorded.get(path),
                "reason": "unreadable"
                if current is None
                else (
                    "matches_recorded_edit"
                    if current == recorded.get(path)
                    else "source_changed"
                ),
            }
        )
    return changes


def _operation_receipt(event):
    result = event["result"]
    return {
        "call_id": event["call_id"],
        "sequence": event["sequence"],
        "operation": result.get("operation"),
        "ok": result.get("ok") is True,
        "returncode": result.get("returncode"),
        "receipt_sha256": _digest(result),
        "observation": result.get("observation"),
        "stdout_sha256": hashlib.sha256(result.get("stdout", "").encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.get("stderr", "").encode()).hexdigest(),
    }


def _checks(task, required):
    latest, failures = {}, []
    for event in task["events"]:
        if event.get("type") != "tool":
            continue
        result = event.get("result", {})
        if event.get("name") == "edit_file" and result.get("ok") is True:
            latest.clear()
        if event.get("name") == "run_operation":
            receipt = _operation_receipt(event)
            latest[result.get("operation")] = receipt
            if (
                not receipt["ok"]
                and result.get("observation", {}).get("status") != "pending"
            ):
                failures.append(receipt)
    checks = {name: latest.get(name) for name in required}
    passed = all(
        item and item["ok"] and item["returncode"] == 0 for item in checks.values()
    )
    return checks, passed, failures


def _task_report(team, task_id):
    report = task_report_for_store(team.config, team.store, task_id)
    return {**report, "model_completed": report["status_completed"]}


def task_report_for_store(config, store, task_id):
    """Summarize worker or coordinator receipts without inferring model completion.

    Args: config: Original team grants. store: Exact task journal to inspect.
        task_id: Existing task identity in that journal.
    Returns: Compact current-source evidence; status_completed alone proves no checks.
    Raises: KeyError, ValueError, OSError: Task, policy or retained storage is invalid.
    """
    task = {
        **store._get(task_id),
        "events": store._events(task_id),
        "calls": store._calls(task_id),
    }
    return _report(store, config.profile(task["profile"]), task)


def _report(store, profile, task):
    changes = _changes(store, task, profile)
    checks, passed, failures = _checks(task, profile.required_operations)
    uncertain = [call for call in task["calls"] if call["status"] != "completed"]
    source_matches = all(item["matches_last_edit"] for item in changes)
    policy_matches = task["policy"] == fingerprint(profile)
    return {
        "task_id": task["id"],
        "specialist": task["profile"],
        "status": task["status"],
        "paused": task["paused"] or profile.name in store._paused_profiles(),
        "status_completed": task["status"] == "completed",
        "result": task["result"][:2000],
        "result_truncated": len(task["result"]) > 2000,
        "result_sha256": hashlib.sha256(task["result"].encode()).hexdigest(),
        "error": task["error"][:512],
        "changes": changes,
        "source_matches_last_edit": source_matches,
        "source_binding": "recorded_edits_only",
        "policy_matches": policy_matches,
        "required_operations": checks,
        "required_operations_passed": bool(checks)
        and passed
        and source_matches
        and policy_matches
        and not uncertain,
        "recent_failures": failures[-3:],
        "failure_count": len(failures),
        "failures_omitted": max(0, len(failures) - 3),
        "uncertain_calls": uncertain,
        "route": task["route"],
    }


def _wait_for_attention(team, task_ids, poll_interval, stop_event):
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must contain distinct tasks")
    if (
        isinstance(poll_interval, bool)
        or not math.isfinite(poll_interval)
        or poll_interval <= 0
    ):
        raise ValueError("poll_interval must be positive and finite")
    stopped = stop_event if stop_event is not None else threading.Event()
    while True:
        paused = team.store._paused_profiles()
        tasks = {identity: team.store._get(identity) for identity in task_ids}
        attention = [
            identity
            for identity, task in tasks.items()
            if task["status"] in _ATTENTION
            or task["paused"]
            or task["profile"] in paused
        ]
        if attention or stopped.is_set():
            return {
                "tasks": {
                    identity: team.task_report(identity) for identity in task_ids
                },
                "attention_task_ids": attention,
                "interrupted": stopped.is_set(),
            }
        stopped.wait(poll_interval)
