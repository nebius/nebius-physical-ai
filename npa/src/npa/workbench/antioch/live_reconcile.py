"""Reconcile an accepted Antioch live run after the foreground CLI detaches."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import yaml

from .vendor_cli import AntiochCli

LIVE_PHASES = {"prepared", "assigned", "running", "finishing"}
LIVE_SESSION_STATES = {"created", "preparing", "waiting", "starting", "running"}
NO_ACTIVE_RUN = 3


class AntiochLiveReconcileError(RuntimeError):
    """The supported run inventory could not identify one exact live run."""


def _session_runtime_snapshot(
    cli: AntiochCli,
    *,
    runtime: Path,
    project_id: str,
    require_ready: bool,
) -> dict[str, Any]:
    """Validate the current supported session and its simulator service."""

    session = cli.session_status(runtime)
    session_id = str(session.get("session_id") or "")
    state = str(session.get("state") or "").lower()
    access_phase = str(session.get("access_phase") or "").lower()
    if not session_id or session.get("project_id") != project_id:
        raise AntiochLiveReconcileError(
            "current Antioch session does not match the exact project"
        )
    if state not in LIVE_SESSION_STATES:
        raise AntiochLiveReconcileError("current Antioch session is not live")

    board = cli.service_ps(runtime, service="sim")
    if board.get("session_id") != session_id:
        raise AntiochLiveReconcileError(
            "Antioch service status belongs to a different session"
        )
    board_state = str(board.get("state") or "").lower()
    services = board.get("services")
    if not isinstance(services, list):
        raise AntiochLiveReconcileError("Antioch service status is malformed")
    matching = [
        item
        for item in services
        if isinstance(item, dict) and item.get("service") == "sim"
    ]
    if len(matching) != 1:
        raise AntiochLiveReconcileError(
            "Antioch session does not expose one exact simulator service"
        )
    service = matching[0]
    service_state = str(service.get("state") or "").lower()
    ready = bool(
        state == "running"
        and board_state == "running"
        and access_phase == "ready"
        and service_state == "running"
        and service.get("process_healthy") is True
        and service.get("session_ready") is True
    )
    if require_ready and not ready:
        raise AntiochLiveReconcileError("Antioch simulator session is not ready")
    return {
        "session_id": session_id,
        "session_state": state,
        "session_access_phase": access_phase,
        "service_state": service_state,
        "service_process_healthy": service.get("process_healthy") is True,
        "session_ready": service.get("session_ready") is True,
        "session_observed_at": time.time(),
    }


def _project_id(runtime: Path) -> str:
    manifest = yaml.safe_load((runtime / "antioch.yaml").read_text(encoding="utf-8"))
    project_id = str((manifest or {}).get("id") or "").strip()
    if not project_id or project_id == "replace-at-runtime":
        raise AntiochLiveReconcileError("runtime project identity is unavailable")
    return project_id


def _active_run_snapshot(
    cli: AntiochCli,
    *,
    runtime: Path,
    project_id: str,
    scenario: str = "openpi_droid_live",
    require_stream_owner: bool = False,
) -> dict[str, Any] | None:
    rows = cli.list_for_project(runtime, kind="scenario", project_id=project_id)
    candidates = {
        str(row["scenario_run_id"]): row
        for row in rows
        if row.get("scenario") == scenario
        and row.get("phase") in LIVE_PHASES
        and row.get("scenario_run_id")
    }
    if len(candidates) > 1:
        raise AntiochLiveReconcileError(
            "multiple exact live runs are active; refusing ambiguous adoption"
        )
    selected = next(iter(candidates.values()), None)
    if selected is None:
        return None
    if require_stream_owner and str(selected.get("phase") or "") != "running":
        return None
    session = _session_runtime_snapshot(
        cli,
        runtime=runtime,
        project_id=project_id,
        require_ready=require_stream_owner,
    )
    selected_session_id = str(selected.get("session_id") or "")
    if not selected_session_id:
        if require_stream_owner:
            return None
    elif selected_session_id != session["session_id"]:
        raise AntiochLiveReconcileError(
            "active scenario belongs to a different Antioch session"
        )
    return {
        **selected,
        **session,
        "stream_state": str(selected.get("phase") or ""),
    }


def _active_run(
    cli: AntiochCli,
    *,
    runtime: Path,
    project_id: str,
    scenario: str = "openpi_droid_live",
    require_stream_owner: bool = False,
) -> dict[str, Any] | None:
    """Compatibility wrapper returning one exact supported ownership snapshot."""

    return _active_run_snapshot(
        cli,
        runtime=runtime,
        project_id=project_id,
        scenario=scenario,
        require_stream_owner=require_stream_owner,
    )


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def reconcile_active(
    *,
    cli_path: Path,
    runtime: Path,
    stop_file: Path,
    state_path: Path,
    scenario: str = "openpi_droid_live",
    poll_seconds: float = 5.0,
    owner_identity: str = "",
    session_id: str = "",
) -> bool:
    """Wait on one exact accepted run; return False when there is none."""

    cli = AntiochCli(cli_path)
    project_id = _project_id(runtime)
    active = _active_run_snapshot(
        cli, runtime=runtime, project_id=project_id, scenario=scenario
    )
    if active is None:
        return False
    remote_id = str(active["scenario_run_id"])
    _write_state(
        state_path,
        {
            "schema": "npa.workbench.antioch-live-active.v2",
            "schema_version": 2,
            "owner_identity": owner_identity,
            "session_id": session_id,
            "scenario": scenario,
            "scenario_run_id": remote_id,
            "stream_state": active.get("stream_state"),
            "heartbeat_unix": time.time(),
            "status": "reconciled",
        },
    )
    print("NPA_ANTIOCH_RECONCILED_ACTIVE", flush=True)
    while True:
        if stop_file.exists():
            cli.cancel(runtime, kind="scenario", remote_id=remote_id)
        current = _active_run_snapshot(
            cli, runtime=runtime, project_id=project_id, scenario=scenario
        )
        if current is None:
            _write_state(
                state_path,
                {
                    "schema": "npa.workbench.antioch-live-active.v2",
                    "schema_version": 2,
                    "owner_identity": owner_identity,
                    "session_id": session_id,
                    "scenario": scenario,
                    "scenario_run_id": remote_id,
                    "heartbeat_unix": time.time(),
                    "status": "terminal",
                },
            )
            return True
        current_id = str(current["scenario_run_id"])
        if current_id != remote_id:
            raise AntiochLiveReconcileError(
                "the active live run changed during reconciliation"
            )
        _write_state(
            state_path,
            {
                "schema": "npa.workbench.antioch-live-active.v2",
                "schema_version": 2,
                "owner_identity": owner_identity,
                "session_id": session_id,
                "scenario": scenario,
                "scenario_run_id": remote_id,
                "stream_state": current.get("stream_state"),
                "heartbeat_unix": time.time(),
                "status": "reconciled",
            },
        )
        time.sleep(poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--scenario", default="openpi_droid_live")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--owner-identity", default="")
    parser.add_argument("--session-id", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adopted = reconcile_active(
        cli_path=Path(args.cli),
        runtime=Path(args.runtime),
        stop_file=Path(args.stop_file),
        state_path=Path(args.state_path),
        scenario=args.scenario,
        poll_seconds=args.poll_seconds,
        owner_identity=args.owner_identity,
        session_id=args.session_id,
    )
    return 0 if adopted else NO_ACTIVE_RUN


if __name__ == "__main__":
    raise SystemExit(main())
