"""SkyPilot Python API bridge executed inside the isolated SkyPilot venv."""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import io
import json
import sys
from typing import Any


def main() -> None:
    if len(sys.argv) != 2:
        _fail("usage: python -m npa.burst._sky_api <launch|queue|logs>")
    action = sys.argv[1]
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if action == "launch":
            result = _launch(payload)
        elif action == "queue":
            result = _queue(payload)
        elif action == "logs":
            result = _logs(payload)
        else:
            _fail(f"unknown action: {action}")
    except Exception as exc:  # noqa: BLE001 - bridge reports errors to parent process.
        _fail(str(exc))
    sys.stdout.write(json.dumps(result, default=_json_default, sort_keys=True) + "\n")


def _launch(payload: dict[str, Any]) -> dict[str, Any]:
    import sky

    yaml_path = str(payload["yaml_path"])
    name = str(payload["name"])
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        task = sky.Task.from_yaml(yaml_path)
        request_id = sky.jobs.launch(task, name=name, _need_confirmation=False)
        launched = sky.stream_and_get(request_id)
    job_ids = []
    if isinstance(launched, tuple) and launched:
        first = launched[0]
        if isinstance(first, list):
            job_ids = first
        elif first is not None:
            job_ids = [first]
    return {"job_ids": [int(job_id) for job_id in job_ids], "output": stream.getvalue()}


def _queue(payload: dict[str, Any]) -> dict[str, Any]:
    import sky

    job_id = int(payload["job_id"])
    refresh = bool(payload.get("refresh", True))
    skip_finished = bool(payload.get("skip_finished", False))
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        request_id = sky.jobs.queue(
            refresh=refresh,
            skip_finished=skip_finished,
            all_users=True,
            job_ids=[job_id],
            version=2,
        )
        queued = sky.get(request_id)
    records = queued[0] if isinstance(queued, tuple) else queued
    if not isinstance(records, list):
        records = []
    return {"records": records, "output": stream.getvalue()}


def _logs(payload: dict[str, Any]) -> dict[str, Any]:
    import sky

    job_id = int(payload["job_id"])
    follow = bool(payload.get("follow", False))
    refresh = bool(payload.get("refresh", True))
    tail = payload.get("tail")
    if tail is not None:
        tail = int(tail)
    stream = io.StringIO()
    status_stream = io.StringIO()
    with contextlib.redirect_stdout(status_stream), contextlib.redirect_stderr(status_stream):
        exit_code = sky.jobs.tail_logs(
            job_id=job_id,
            follow=follow,
            refresh=refresh,
            tail=tail,
            output_stream=stream,
        )
    return {
        "logs": stream.getvalue(),
        "output": status_stream.getvalue(),
        "exit_code": exit_code,
    }


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
