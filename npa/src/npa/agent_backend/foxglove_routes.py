"""`/api/foxglove/*` routes for the agent backend (shipped module).

Uploaded to ``/opt/npa-agent/agent_backend/`` at bootstrap and imported by
``backend.py`` — the same "shipped, not inlined" mechanism as ``memory``,
``retrieval`` and ``trace``. Keeping the routes here (instead of inside the
backend f-string) keeps ``agent.py`` reviewable and lets the routes be tested
directly with a FastAPI ``TestClient`` and injected fakes.

Everything the routes need from the backend is injected through
:class:`FoxgloveDeps`, so this module has no import-time dependency on the agent
VM's filesystem, session state, or object storage.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:  # agent VM: /opt/npa-agent is on sys.path and the package is flat there
    from agent_backend.foxglove import (
        convert_run_request,
        converted_recording_update,
        foxglove_status_payload,
        is_foxglove_artifact,
        live_source_update,
        prune_published,
    )
except ImportError:  # repo / local tests
    from npa.agent_backend.foxglove import (
        convert_run_request,
        converted_recording_update,
        foxglove_status_payload,
        is_foxglove_artifact,
        live_source_update,
        prune_published,
    )


@dataclass
class FoxgloveDeps:
    """Everything the Foxglove routes need from the surrounding backend."""

    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]
    record_run: Callable[[dict, dict], Any]
    foxglove_config: Callable[..., dict]
    load_artifact: Callable[[dict], Any]
    convert_run: Callable[..., Any]
    now_iso: Callable[[], str]
    validate_run_id: Callable[[str], str]
    data_dir: Path
    runs_dir: Path
    keep_published: int = 3
    set_live_url: Callable[[str], None] | None = None


def _sim_viz(state: dict) -> dict:
    value = state.get("sim_viz")
    return value if isinstance(value, dict) else {}


def register_foxglove_routes(app: Any, deps: FoxgloveDeps, http_error: Any) -> None:
    """Register the Foxglove viewer routes on ``app``.

    ``http_error`` is the backend's ``HTTPException`` class (injected so this
    module does not import FastAPI at module import time on the agent VM).
    """

    @app.get("/foxglove/config")
    def foxglove_config_route() -> dict:
        # Everything the UI needs to mount an MCAP viewer. "available" is False
        # (with a reason) when neither backend can render, so the pane can say so
        # instead of showing an empty viewer.
        return deps.foxglove_config()

    @app.get("/foxglove/status")
    def foxglove_status_route() -> dict:
        state = deps.load_state()
        return foxglove_status_payload(deps.foxglove_config(state), _sim_viz(state))

    @app.post("/foxglove/load-artifact")
    def foxglove_load_artifact_route(payload: dict | None = None) -> Any:
        # Thin wrapper over the shared artifact loader so the Foxglove pane and
        # the artifact browser publish through exactly one code path.
        body = payload if isinstance(payload, dict) else {}
        key_hint = str(body.get("key") or body.get("s3_uri") or "")
        if key_hint and not is_foxglove_artifact(key_hint):
            raise http_error(
                status_code=400,
                detail="Foxglove opens .mcap, .bag, .db3, .ulg and .ulog recordings",
            )
        result = deps.load_artifact(body)
        if isinstance(result, dict) and result.get("ok"):
            state = deps.load_state()
            result["foxglove"] = foxglove_status_payload(
                deps.foxglove_config(state), _sim_viz(state)
            )
        return result

    @app.post("/foxglove/convert-run")
    def foxglove_convert_run_route(payload: dict | None = None) -> dict:
        # Convert the active run's downloaded artifacts into a real MCAP and load
        # it into the viewer (same code path as `npa workbench foxglove`).
        state = deps.load_state()
        sim_viz = _sim_viz(state)
        request = convert_run_request(payload, sim_viz)
        run_id = request["run_id"]
        if not run_id:
            raise http_error(status_code=400, detail="no active run to convert")
        try:
            run_id = deps.validate_run_id(run_id)
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            raise http_error(status_code=400, detail=str(exc))
        source_dir = deps.runs_dir / run_id
        if not source_dir.is_dir():
            raise http_error(
                status_code=404,
                detail=(
                    f"no local artifacts for run {run_id}; load the run's artifacts "
                    "first so there are frames/metrics/logs to convert"
                ),
            )
        name = f"{secrets.token_hex(8)}-{run_id}.mcap"
        output_path = deps.data_dir / name
        try:
            deps.data_dir.mkdir(parents=True, exist_ok=True)
            summary = deps.convert_run(
                input_path=source_dir,
                output_path=output_path,
                fps=request["fps"],
                max_frames=request["max_frames"],
                run_id=run_id,
            ).to_dict()
            output_path.chmod(0o644)
        except Exception as exc:  # noqa: BLE001 - conversion is operator-triggered
            raise http_error(status_code=422, detail=f"MCAP conversion failed: {exc}")
        prune_published(deps.data_dir, keep=deps.keep_published)
        sim_viz = converted_recording_update(
            sim_viz, run_id=run_id, name=name, summary=summary, now=deps.now_iso()
        )
        state["sim_viz"] = sim_viz
        deps.record_run(state, sim_viz)
        deps.save_state(state)
        return {
            "ok": True,
            "summary": summary,
            "sim_viz": sim_viz,
            "foxglove": foxglove_status_payload(deps.foxglove_config(state), sim_viz),
        }

    @app.post("/foxglove/live")
    def foxglove_live_route(payload: dict | None = None) -> dict:
        # Point the embedded viewer at a live Foxglove/ROS-bridge WebSocket.
        state = deps.load_state()
        result = live_source_update(payload, _sim_viz(state), now=deps.now_iso())
        if result is None:
            raise http_error(
                status_code=400,
                detail=(
                    "Provide a public ws:// or wss:// URL (loopback, private, "
                    "link-local and metadata targets are refused)"
                ),
            )
        source, sim_viz = result
        # Session state, not process environment: the URL belongs to this agent
        # session and must survive a backend restart with the rest of the state.
        sim_viz["foxglove_live_url"] = source["url"]
        sim_viz["foxglove_live_protocol"] = source["protocol"]
        state["sim_viz"] = sim_viz
        deps.save_state(state)
        if deps.set_live_url is not None:
            deps.set_live_url(source["url"])
        return {
            "ok": True,
            "data_source": source,
            "foxglove": foxglove_status_payload(deps.foxglove_config(state), sim_viz),
        }
