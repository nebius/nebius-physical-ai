"""Call pinned SkyPilot SDK operations in its separate Python environment."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import sys
from urllib.request import ProxyHandler, build_opener


def main() -> None:
    """Execute one private, server-generated SkyPilot operation through JSON stdin.

    Args:
        None; reads a server-owned request from stdin.
    Returns:
        None; emits one JSON response without credentials or backend tracebacks.
    Raises:
        SystemExit: SkyPilot is unavailable or an operation fails.
    """
    try:
        if importlib.metadata.version("skypilot") != "0.12.2":
            raise ValueError("SkyPilot version mismatch")
        endpoint = os.environ["SKYPILOT_API_SERVER_ENDPOINT"].rstrip("/")
        with build_opener(ProxyHandler({})).open(
            endpoint + "/api/health", timeout=10
        ) as response:
            if response.status != 200:
                raise ValueError("SkyPilot server unavailable")
        request = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            result = _execute(request)
        print(json.dumps(result))
    except Exception:
        # Backend tracebacks can contain credentials and private cluster addresses.
        print(
            json.dumps({"error": "SkyPilot operation failed; reconcile before retry"})
        )
        raise SystemExit(1)


def _execute(request):
    import sky
    from sky.utils import dag_utils

    # The pinned managed-job SDK omits its client-context decorator on several
    # methods. This public read-only call establishes client configuration
    # forwarding before queue/cancel/logs, including the allocation workspace.
    sky.api_info()
    operation = request["operation"]
    if operation == "launch":
        dag = dag_utils.load_dag_from_yaml_str(request["yaml"])
        return {"request_id": sky.jobs.launch(dag, name=request["name"])}
    if operation == "result":
        result = sky.get(request["request_id"])
        return {"job_ids": result[0]}
    if operation == "queue":
        try:
            rows = sky.get(
                sky.jobs.queue(
                    refresh=False, skip_finished=False, job_ids=request.get("job_ids")
                )
            )
        except sky.exceptions.ClusterNotUpError:
            # The adapter still requires every recorded job to be returned.
            # Missing controller state cannot authorize replay or cancellation.
            return {"jobs": []}
        return {"jobs": [_job_row(row) for row in rows]}
    if operation == "cancel":
        sky.get(sky.jobs.cancel(job_ids=request["job_ids"]))
        return {"cancel_requested": True}
    if operation == "logs":
        paths = sky.jobs.download_logs(
            name=None,
            job_id=request["job_id"],
            refresh=False,
            controller=False,
            local_dir=request["directory"],
        )
        return {"paths": paths}
    raise ValueError("unsupported SkyPilot bridge operation")


def _job_row(row):
    if hasattr(row, "model_dump"):
        row = row.model_dump()
    result = {
        key: row.get(key) for key in ("job_id", "job_name", "task_id", "task_name")
    }
    status = row.get("status", "UNKNOWN")
    result["status"] = getattr(status, "value", str(status))
    return result


if __name__ == "__main__":
    main()
