"""Separate coordinator phases and overlapping worker activity from retained receipts."""

from collections import Counter
import hashlib
import json
import math


def _read(path):
    return json.loads(path.read_text())


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("timing evidence must contain finite nonnegative numbers")
    return value


def _interval(start, end):
    start, end = _number(start), _number(end)
    if end < start:
        raise ValueError("timing evidence moved backwards")
    return start, end


def _elapsed(start, end):
    start, end = _interval(start, end)
    return end - start


def _occupied(intervals):
    result, until = 0.0, 0.0
    for start, end in sorted(intervals):
        result += max(0.0, end - max(start, until))
        until = max(until, end)
    return result


def _durations(intervals, missing=0):
    return {
        "complete": missing == 0,
        "recorded_intervals": len(intervals),
        "unmatched_events": missing,
        "sum_seconds": sum(end - start for start, end in intervals),
        "occupied_wall_seconds": _occupied(intervals),
    }


def _model_intervals(receipts):
    intervals, missing = [], 0
    for task in receipts.values():
        started = None
        for event in task["events"]:
            if event["type"] == "context_view":
                missing += started is not None
                started = event["at"]
            elif event["type"] in {"model", "provider_failure"}:
                if started is None:
                    missing += 1
                else:
                    intervals.append(_interval(started, event["at"]))
                started = None
        missing += started is not None
    return _durations(intervals, missing)


def _tool_pairs(receipts):
    pairs, missing = [], 0
    for task in receipts.values():
        started = {}
        for event in task["events"]:
            identity = event.get("call_id")
            if event["type"] == "tool_started":
                if identity in started:
                    raise ValueError("duplicate tool start in timing evidence")
                started[identity] = event
            elif event["type"] == "tool":
                beginning = started.pop(identity, None)
                if beginning is None:
                    missing += 1
                else:
                    pairs.append((beginning, event))
        missing += len(started)
    return pairs, missing


def _tool_durations(receipts):
    pairs, missing = _tool_pairs(receipts)
    intervals = [_interval(start["at"], end["at"]) for start, end in pairs]
    return _durations(intervals, missing)


def _assignment(arguments):
    value = [arguments.get(name) for name in ("specialist", "task_id", "goal")]
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError("delegation receipt lacks its exact assignment")
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def _delegations(receipts):
    pairs, missing = _tool_pairs(receipts)
    counts, pending, intervals = Counter(), set(), []
    for start, end in pairs:
        if start.get("name") != "delegate":
            continue
        identity = _assignment(start["arguments"])
        result = end["result"]
        intervals.append(_interval(start["at"], end["at"]))
        if (
            result.get("status") == "busy"
            and result.get("submission_attempted") is False
            and result.get("safe_to_retry") is True
        ):
            counts["safe_busy_refusals"] += 1
            pending.add(identity)
        elif result.get("status") in {"queued", "running", "completed"}:
            counts["accepted"] += 1
            counts["same_assignment_retries_accepted"] += identity in pending
            pending.discard(identity)
        else:
            counts["uncertain_or_rejected"] += 1
    return {
        **_durations(intervals, missing),
        **{
            key: counts[key]
            for key in (
                "safe_busy_refusals",
                "accepted",
                "same_assignment_retries_accepted",
                "uncertain_or_rejected",
            )
        },
        "unresolved_busy_assignments": len(pending),
    }


def _coordinator(directory):
    turns = _read(directory / "coordinator-config.json")["turns"]
    intervals = [_interval(row["started_epoch"], row["ended_epoch"]) for row in turns]
    phases = Counter(row["phase"] for row in turns)
    if set(phases) - {"work", "delegate", "review"}:
        raise ValueError("unknown coordinator phase")
    waiting = directory / "coordination.json"
    events = _read(waiting)["events"] if waiting.exists() else []
    if any(row["phase"] != "host_wait" or row["model_calls"] != 0 for row in events):
        raise ValueError("host-wait evidence contains unexpected model activity")
    waits = [_interval(row["started_epoch"], row["ended_epoch"]) for row in events]
    if _occupied(intervals + waits) < _occupied(intervals) + _occupied(waits) - 0.001:
        raise ValueError("coordinator turns overlap claimed model-free waiting")
    return {
        "turns": len(turns),
        "turns_by_phase": dict(phases),
        "turn_process_seconds": _occupied(intervals),
        "host_wait_seconds": _occupied(waits),
        "host_wait_model_calls": 0,
        "basis": "Coordinator process intervals include its tool calls; host waits may overlap worker inference and native execution.",
    }


def _routing(usage):
    responses = usage["router"]["responses"]
    attempted = [row for row in responses if row.get("api_call_attempted") is not False]
    complete = all("latency_seconds" in row for row in attempted)
    return {
        "attempts": len(attempted),
        "complete": complete,
        "sum_request_seconds": sum(_number(row["latency_seconds"]) for row in attempted)
        if complete
        else None,
        "basis": "Sum of recorded classifier request durations; overlaps delegation and worker activity.",
    }


def _lanes(receipts):
    result = {}
    names = ("physics", "publication", "adapter")
    if any(task["profile"] not in names for task in receipts.values()):
        raise ValueError("unknown benchmark lane")
    for name in names:
        tasks = {key: task for key, task in receipts.items() if task["profile"] == name}
        spans, first_calls = [], []
        for task in tasks.values():
            events = task["events"]
            if events:
                spans.append(
                    _interval(task["created"], max(event["at"] for event in events))
                )
            starts = [
                event["at"] for event in events if event["type"] == "context_view"
            ]
            if starts:
                first_calls.append(_elapsed(task["created"], min(starts)))
        result[name] = {
            "submitted_tasks": len(tasks),
            "observed_task_spans": _durations(spans),
            "queue_to_first_model_seconds": first_calls,
            "model_intervals": _model_intervals(tasks),
            "tool_intervals": _tool_durations(tasks),
            "basis": "Task spans end at the last journal event, not an independently observed process exit. Empty lanes mean no specialist task was submitted.",
        }
    return result


def _observed(directory, outcome):
    run = directory / "run"
    workers = _read(run / "task-receipts.json")
    coordinator = _read(run / "coordinator-receipts.json")
    execution = _read(run / "execution.json")
    agent = _number(outcome["agent_tool_seconds"])
    total = _number(outcome["end_to_end_seconds"])
    _interval(agent, total)
    return {
        "schema": "npa.specialists.repair-benchmark.timing.v1",
        "coordinator_process_and_workers_seconds": agent,
        "combined_verification_seconds": total - agent,
        "worker_shutdown_seconds": _elapsed(
            execution["coordinator_seconds"], execution["agent_tool_seconds"]
        ),
        "coordinator": _coordinator(run),
        "delegation": _delegations({"supervisor": coordinator["supervisor"]}),
        "routing": _routing(_read(run / "usage.json")),
        "specialist_model_intervals": _model_intervals(workers),
        "specialist_lanes": _lanes(workers),
        "specialist_tool_intervals": _tool_durations(workers),
        "coordinator_tool_intervals": _tool_durations(
            {key: value for key, value in coordinator.items() if key != "supervisor"}
        ),
        "model_timing_basis": "From context-view journaling to response journaling, including request preparation and validation; not provider-only latency or time to first token.",
        "overlap_policy": "Only process-and-workers plus combined verification partition total elapsed time. All other fields are nested or overlapping and must not be added to that total.",
    }


def _timing(directory, outcome):
    try:
        result = _observed(directory, outcome)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {
            "complete": False,
            "reason": "Missing or invalid timing evidence; do not infer zero overhead.",
            "error_type": type(error).__name__,
        }
    result["complete"] = all(
        row["complete"]
        for row in result.values()
        if isinstance(row, dict) and "complete" in row
    )
    return result
