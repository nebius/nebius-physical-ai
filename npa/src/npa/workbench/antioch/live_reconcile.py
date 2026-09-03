"""Reconcile an accepted Antioch live run after the foreground CLI detaches."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import yaml

from .vendor_cli import AntiochCli

LIVE_PHASES = {"queued", "booting", "running"}
TERMINAL_STREAM_STATES = {"failed", "stopped", "idle"}
NO_ACTIVE_RUN = 3
DAEMON_OBSERVATION_MAX_AGE_SECONDS = 30.0


class AntiochLiveReconcileError(RuntimeError):
    """The supported run inventory could not identify one exact live run."""


def _timestamp_seconds(value: object) -> float:
    """Normalize the timestamp encodings exposed by the structured CLI."""

    if isinstance(value, bool):
        raise AntiochLiveReconcileError("daemon observation timestamp is malformed")
    if isinstance(value, (int, float)):
        resolved = float(value)
        # Antioch JSON timestamps are currently epoch microseconds.  Retain
        # seconds for compatibility with older structured clients.
        return resolved / 1_000_000.0 if resolved > 100_000_000_000 else resolved
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
                UTC
            ).timestamp()
        except ValueError as exc:
            raise AntiochLiveReconcileError(
                "daemon observation timestamp is malformed"
            ) from exc
    raise AntiochLiveReconcileError("daemon observation timestamp is unavailable")


def _daemon_runtime_snapshot(
    machine: dict[str, Any],
    *,
    now: float | None = None,
    max_age_seconds: float = DAEMON_OBSERVATION_MAX_AGE_SECONDS,
    require_session_owner: bool = True,
) -> dict[str, Any]:
    """Validate Rome and direct-daemon liveness from supported status JSON."""

    observed_now = time.time() if now is None else now
    runtime = machine.get("runtime")
    rome = machine.get("runtime_status")
    if not isinstance(runtime, dict) or not isinstance(rome, dict):
        raise AntiochLiveReconcileError("daemon runtime status is unavailable")
    if machine.get("daemon_error"):
        raise AntiochLiveReconcileError("direct daemon status is unhealthy")
    if str(rome.get("guest_state") or "").lower() != "healthy":
        raise AntiochLiveReconcileError("Rome daemon liveness is unhealthy")
    if rome.get("guest_failure_started_at") is not None:
        raise AntiochLiveReconcileError("Rome daemon liveness failure is active")

    direct_observed_at = _timestamp_seconds(runtime.get("observed_at"))
    rome_observed_at = _timestamp_seconds(rome.get("guest_observed_at"))
    direct_age = observed_now - direct_observed_at
    rome_age = observed_now - rome_observed_at
    if not (-5.0 <= direct_age <= max_age_seconds):
        raise AntiochLiveReconcileError("direct daemon observation is stale")
    if not (-5.0 <= rome_age <= max_age_seconds):
        raise AntiochLiveReconcileError("Rome daemon liveness observation is stale")

    direct_stream = runtime.get("stream")
    rome_observation = rome.get("observation")
    rome_stream = (
        rome_observation.get("stream")
        if isinstance(rome_observation, dict)
        else None
    )
    if not isinstance(direct_stream, dict) or not isinstance(rome_stream, dict):
        raise AntiochLiveReconcileError("daemon stream status is malformed")
    direct_run_id = str(direct_stream.get("scenario_run_id") or "")
    rome_run_id = str(rome_stream.get("scenario_run_id") or "")
    if direct_run_id != rome_run_id:
        raise AntiochLiveReconcileError(
            "Rome and direct daemon stream owners disagree"
        )

    leases = runtime.get("leases")
    if not isinstance(leases, list):
        raise AntiochLiveReconcileError("daemon lease status is malformed")
    lease_kinds = [
        (str(item.get("kind") or ""), str(item.get("label") or ""))
        for item in leases
        if isinstance(item, dict)
    ]
    scenario_sessions = sum(
        kind == "session" and label == "antioch scenario run"
        for kind, label in lease_kinds
    )
    process_leases = sum(kind == "process" for kind, _label in lease_kinds)
    stream_leases = sum(kind == "stream" for kind, _label in lease_kinds)
    if require_session_owner and direct_run_id and (
        scenario_sessions != 1 or process_leases < 1 or stream_leases != 1
    ):
        raise AntiochLiveReconcileError(
            "exact vendor process/session/stream lease ownership is unhealthy"
        )
    return {
        "stream": direct_stream,
        "rome_stream": rome_stream,
        "guest_state": "healthy",
        "direct_observed_at": direct_observed_at,
        "rome_observed_at": rome_observed_at,
        "scenario_session_leases": scenario_sessions,
        "process_leases": process_leases,
        "stream_leases": stream_leases,
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
    machine = cli.machine_status(runtime, project_id=project_id)
    daemon = _daemon_runtime_snapshot(machine)
    stream = daemon["stream"]
    stream_run_id = str(stream.get("scenario_run_id") or "")
    stream_state = str(stream.get("state") or "").lower()
    if stream_run_id and stream_state not in TERMINAL_STREAM_STATES:
        selected = candidates.get(stream_run_id)
        if selected is None:
            raise AntiochLiveReconcileError(
                "active stream owner is absent from the exact project run inventory"
            )
        return {
            **selected,
            "scenario_run_id": stream_run_id,
            "stream_state": stream_state,
            "daemon_guest_state": daemon["guest_state"],
            "daemon_observed_at": daemon["direct_observed_at"],
            "rome_guest_observed_at": daemon["rome_observed_at"],
            "scenario_session_leases": daemon["scenario_session_leases"],
            "process_leases": daemon["process_leases"],
            "stream_leases": daemon["stream_leases"],
        }
    if len(candidates) > 1:
        raise AntiochLiveReconcileError(
            "multiple exact live runs are active; refusing ambiguous adoption"
        )
    if require_stream_owner:
        return None
    selected = next(iter(candidates.values()), None)
    if selected is None:
        return None
    return {**selected, "stream_state": stream_state or "unavailable"}


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
