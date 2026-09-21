"""PID-1 controller for the cluster-native Antioch live adapter pod.

The supported Antioch named route is created in this pod. A sibling relay
container shares the pod network namespace and connects to the route only on
localhost, so the operator VM never carries frame or action traffic.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from .live import (
    AntiochLiveError,
    _cancel_remote_live_runs,
    _private_bundle_matches,
    _runtime_source_matches,
    _stage_private_bundle,
    _stage_project,
    _stage_runtime_source,
    _validate_bundle,
)
from .health import StateHealthServer
from .live_reconcile import AntiochLiveReconcileError, _active_run_snapshot
from .runtime import ensure_runtime
from .vendor_cli import AntiochCli, AntiochCliError


SCHEMA = "npa.workbench.antioch-cluster-live.v4"
SCHEMA_VERSION = 4
RELAY_SCHEMA_VERSION = 2
DEFAULT_CONTROLLER_MAX_AGE_SECONDS = 30.0
DEFAULT_RELAY_MAX_AGE_SECONDS = 150.0
STATE_READ_ATTEMPTS = 3
SESSION_POLL_SECONDS = 5.0
SESSION_ABSENCE_THRESHOLD = 3
SESSION_ERROR_THRESHOLD = 3
SESSION_STARTUP_GRACE_SECONDS = 600.0
RESTAGE_INTERVAL_SECONDS = 60.0
RECOVERY_BACKOFF_SECONDS = (2.0, 5.0, 10.0, 20.0, 30.0)
COMPLETION_POLL_ATTEMPTS = 12
COMPLETION_POLL_SECONDS = 5.0
REQUIRED_POC_CHECKS = frozenset(
    {
        "exterior_observations_advancing_nonblack",
        "wrist_observations_advancing_nonblack",
        "pi05_responses_finite_15x8",
    }
)
REQUIRED_PICKUP_CHECKS = REQUIRED_POC_CHECKS | {
    "policy_views_exposure_and_contrast", "initial_target_resolved_both_views",
    "policy_evidence_complete", "policy_actions_executed", "episode_completed",
    "end_effector_approached_cube", "bilateral_gripper_contact", "cube_lift_held",
}
_METRIC_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_CAMERA_REJECTION_VIEWS = frozenset({"exterior", "wrist", "pair"})
_CAMERA_REJECTION_REASONS = frozenset(
    {
        "missing",
        "wrong_shape",
        "non_finite",
        "blank",
        "flat",
        "overexposed",
        "target_unresolved",
        "gripper_out_of_frame",
        "low_dynamic_range",
        "cube_not_visible",
        "cube_out_of_frame",
        "stale",
        "not_distinct",
    }
)


def _sanitized_metric_line(line: bytes) -> str:
    """Return only fixed numeric metrics or a typed camera rejection event."""

    marker = b"NPA_OPENPI_METRICS "
    if marker not in line:
        rejection_marker = b"NPA_OPENPI_CAMERA_REJECT "
        if rejection_marker not in line:
            return ""
        values: dict[str, str] = {}
        for raw in line.split(rejection_marker, 1)[1].split():
            key, separator, value = raw.partition(b"=")
            if not separator:
                return ""
            try:
                values[key.decode("ascii")] = value.decode("ascii")
            except UnicodeDecodeError:
                return ""
        if set(values) != {
            "view",
            "reason",
            "render_sequence",
            "exterior_red_cube_pixels",
            "pair_difference",
        }:
            return ""
        if (
            values["view"] not in _CAMERA_REJECTION_VIEWS
            or values["reason"] not in _CAMERA_REJECTION_REASONS
        ):
            return ""
        try:
            numeric = (
                float(values["render_sequence"]),
                float(values["exterior_red_cube_pixels"]),
                float(values["pair_difference"]),
            )
        except ValueError:
            return ""
        if not all(math.isfinite(value) for value in numeric):
            return ""
        return (
            "NPA_OPENPI_CAMERA_REJECT "
            f"view={values['view']} reason={values['reason']} "
            f"render_sequence={values['render_sequence']} "
            f"exterior_red_cube_pixels={values['exterior_red_cube_pixels']} "
            f"pair_difference={values['pair_difference']}"
        )
    fields: list[str] = []
    for raw in line.split(marker, 1)[1].split():
        key, separator, value = raw.partition(b"=")
        try:
            key_text = key.decode("ascii")
            value_text = value.decode("ascii")
            number = float(value_text)
        except (UnicodeDecodeError, ValueError):
            return ""
        if (
            not separator
            or not _METRIC_KEY.fullmatch(key_text)
            or not math.isfinite(number)
        ):
            return ""
        fields.append(f"{key_text}={value_text}")
    return f"NPA_OPENPI_METRICS {' '.join(fields)}" if fields else ""


def _scenario_command(
    executable: Path, scenario: str, timeout_seconds: int,
    initial_posture: str, camera_mounts: str,
) -> list[str]:
    """Keep public experiment choices typed and separate from private bundles."""
    if initial_posture not in {"pregrasp", "droid"}:
        raise ValueError("Unsupported initial_posture")
    if camera_mounts not in {"native_wide", "droid_reference", "task_view"}:
        raise ValueError("Unsupported camera_mounts")
    command = [str(executable), "scenario", "run", "--scenario", scenario,
               "--timeout", str(timeout_seconds), "--stream", "--verbose"]
    if scenario == "openpi_franka_pickup_v3":
        command.extend(["--set", f"initial_posture={initial_posture}",
                        "--set", f"camera_mounts={camera_mounts}"])
    elif (initial_posture, camera_mounts) != ("pregrasp", "native_wide"):
        raise ValueError("Pickup parameters require the pickup scenario")
    return command


@dataclass
class VendorStreamProcess:
    """One directly-owned foreground Antioch stream client and its drain."""

    process: subprocess.Popen[bytes]
    started_monotonic: float
    last_output_monotonic: float
    output_bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _drain: threading.Thread | None = field(default=None, repr=False)

    @classmethod
    def start(
        cls,
        *,
        executable: Path,
        runtime: Path,
        scenario: str,
        timeout_seconds: int,
        initial_posture: str = "pregrasp",
        camera_mounts: str = "native_wide",
    ) -> "VendorStreamProcess":
        started = time.monotonic()
        process = subprocess.Popen(
            _scenario_command(executable, scenario, timeout_seconds,
                              initial_posture, camera_mounts),
            cwd=runtime,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        owned = cls(
            process=process,
            started_monotonic=started,
            last_output_monotonic=started,
        )
        drain = threading.Thread(
            target=owned._drain_output,
            name="antioch-vendor-output-drain",
            daemon=True,
        )
        owned._drain = drain
        drain.start()
        return owned

    def _drain_output(self) -> None:
        stream = self.process.stdout
        if stream is None:
            return
        while True:
            raw_line = stream.readline(65_537)
            if not raw_line:
                return
            with self._lock:
                self.output_bytes += len(raw_line)
                self.last_output_monotonic = time.monotonic()
            if len(raw_line) > 65_536 and not raw_line.endswith(b"\n"):
                continue
            line = _sanitized_metric_line(raw_line.rstrip(b"\r\n"))
            if line:
                print(line, flush=True)

    def output_snapshot(self, *, now: float | None = None) -> tuple[int, float]:
        with self._lock:
            observed_now = time.monotonic() if now is None else now
            return self.output_bytes, observed_now - self.last_output_monotonic

    def exit_snapshot(self) -> tuple[str, int | None]:
        code = self.process.poll()
        if code is None:
            return "running", None
        if code < 0:
            return "signal", code
        if code == 0:
            return "completed", code
        return "nonzero", code

    def terminate(self) -> None:
        _terminate_process_group(self.process)
        if self._drain is not None:
            self._drain.join(timeout=5)


@dataclass
class VendorPortProcess:
    """Directly own the current CLI's foreground named-route bridge."""

    process: subprocess.Popen[bytes]

    @classmethod
    def start(
        cls,
        *,
        cli: AntiochCli,
        executable: Path,
        runtime: Path,
    ) -> "VendorPortProcess":
        process = subprocess.Popen(
            [
                str(executable),
                *cli.service_ports_args(
                    service="sim",
                    route="policy-relay",
                    host="127.0.0.1",
                    port=18_444,
                ),
            ],
            cwd=runtime,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return cls(process=process)

    def exit_snapshot(self) -> tuple[str, int | None]:
        code = self.process.poll()
        if code is None:
            return "running", None
        if code < 0:
            return "signal", code
        if code == 0:
            return "completed", code
        return "nonzero", code

    def terminate(self) -> None:
        _terminate_process_group(self.process)


def _write_state(path: Path, **values: Any) -> None:
    state = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "published_unix": time.time(),
        **values,
    }
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, (json.dumps(state, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    os.chmod(path, 0o600)


@contextlib.contextmanager
def _recovery_heartbeat(
    path: Path,
    *,
    interval_seconds: float = 5.0,
    **values: Any,
) -> Iterator[None]:
    """Keep PID-1 liveness fresh while supported recovery calls block.

    Recovery never satisfies the readiness probe: this republishes only the
    explicit ``recovering`` state.  It prevents Kubernetes from killing PID 1
    while bounded vendor cancellation, service repair, or restaging is still
    making progress.
    """

    stopped = threading.Event()
    failure: list[Exception] = []

    def publish() -> None:
        while not stopped.wait(interval_seconds):
            try:
                _write_state(path, **values)
            except Exception as exc:  # pragma: no cover - surfaced below
                failure.append(exc)
                return

    heartbeat = threading.Thread(
        target=publish,
        name="antioch-recovery-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    try:
        yield
    finally:
        stopped.set()
        heartbeat.join(timeout=max(1.0, interval_seconds * 2.0))
    if heartbeat.is_alive():
        raise RuntimeError("recovery heartbeat did not stop")
    if failure:
        raise failure[0]


def _read_state(
    path: Path, *, attempts: int = STATE_READ_ATTEMPTS, delay_seconds: float = 0.05
) -> dict[str, Any]:
    """Read one atomically-published state with bounded transient recovery."""

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise TypeError("state is not an object")
            return parsed
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


def _state_ready(
    state: dict[str, Any],
    *,
    component: str,
    expected_owner_identity: str,
    max_age_seconds: float,
    now: float | None = None,
) -> bool:
    expected_schema_version = (
        SCHEMA_VERSION if component.startswith("controller") else RELAY_SCHEMA_VERSION
    )
    if int(state.get("schema_version") or 0) != expected_schema_version:
        return False
    if str(state.get("owner_identity") or "") != expected_owner_identity:
        return False
    observed_now = time.time() if now is None else now

    def vendor_session_ready() -> bool:
        session_observed = state.get("session_observed_at")
        if not isinstance(session_observed, (int, float)) or isinstance(
            session_observed, bool
        ):
            return False
        return bool(
            state.get("vendor_process_status") == "running"
            and state.get("route_process_status") == "running"
            and state.get("antioch_session_state") == "running"
            and state.get("antioch_session_access_phase") == "ready"
            and state.get("sim_process_healthy") is True
            and state.get("antioch_session_ready") is True
            and str(state.get("antioch_session_id") or "")
            and int(state.get("controller_pid") or 0) == 1
            and int(state.get("vendor_parent_pid") or 0) == 1
            and int(state.get("vendor_pid") or 0) > 1
            and state.get("vendor_process_group_isolated") is True
            and int(state.get("route_parent_pid") or 0) == 1
            and int(state.get("route_pid") or 0) > 1
            and state.get("route_process_group_isolated") is True
            and -5.0 <= observed_now - float(session_observed) <= max_age_seconds
        )

    if component == "controller-liveness":
        published = state.get("published_unix")
        published_age = (
            observed_now - float(published)
            if isinstance(published, (int, float)) and not isinstance(published, bool)
            else max_age_seconds + 1
        )
        if not (-5.0 <= published_age <= max_age_seconds):
            return False
        if state.get("status") == "running":
            return vendor_session_ready()
        return state.get("status") in {"starting", "recovering", "completed", "stopped"}
    heartbeat = state.get("heartbeat_unix")
    if not isinstance(heartbeat, (int, float)) or isinstance(heartbeat, bool):
        return False
    age = observed_now - float(heartbeat)
    if age < -5.0 or age > max_age_seconds:
        return False
    if component == "controller":
        return bool(
            state.get("status") == "running"
            and state.get("session_status") == "owned"
            and str(state.get("scenario_run_id") or "")
            and str(state.get("session_id") or "")
            and vendor_session_ready()
        )
    if component == "relay-liveness":
        return state.get("status") in {
            "starting",
            "connecting_simulation",
            "connecting_policy",
            "connected",
            "reconnecting",
            "stopped",
        }
    return state.get("status") == "connected"


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _supervisor_recovery_reason(
    *,
    child_dead: bool,
    last_owned_heartbeat: float,
    consecutive_absence: int,
    consecutive_errors: int,
    age_seconds: float,
    startup_age_seconds: float,
    max_age_seconds: float,
) -> str:
    """Classify only converged loss as replacement-worthy."""

    if child_dead:
        return "controller_child_exit"
    if (
        last_owned_heartbeat
        and consecutive_absence >= SESSION_ABSENCE_THRESHOLD
        and age_seconds > max_age_seconds
    ):
        return "session_owner_absent"
    if (
        not last_owned_heartbeat
        and startup_age_seconds >= SESSION_STARTUP_GRACE_SECONDS
    ):
        return "session_owner_startup_timeout"
    if consecutive_errors >= SESSION_ERROR_THRESHOLD and age_seconds > max_age_seconds:
        return "session_state_unreadable"
    return ""


def _integer_result(results: dict[str, Any], name: str, *, minimum: int) -> int:
    value = results.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise AntiochLiveError(f"completed Antioch proof has invalid {name}")
    return value


def _completed_poc_evidence(
    record: dict[str, Any], *, scenario: str, scenario_run_id: str,
    initial_posture: str | None = None, camera_mounts: str | None = None,
) -> dict[str, Any]:
    """Validate the durable terminal record before retiring live compute."""

    if (
        record.get("scenario_run_id") != scenario_run_id
        or record.get("scenario") != scenario
        or record.get("phase") != "completed"
        or record.get("outcome") != "passed"
    ):
        raise AntiochLiveError(
            "Antioch proof did not persist as the expected passed run"
        )
    results = record.get("results")
    if not isinstance(results, dict):
        raise AntiochLiveError("completed Antioch proof has no structured results")
    round_trips = _integer_result(results, "policy_round_trips", minimum=2)
    exterior_count = _integer_result(results, "exterior_observation_count", minimum=2)
    wrist_count = _integer_result(results, "wrist_observation_count", minimum=2)
    first_sequence = _integer_result(
        results, "first_accepted_render_sequence", minimum=1
    )
    last_sequence = _integer_result(
        results, "last_accepted_render_sequence", minimum=first_sequence + 1
    )
    if (
        results.get("communication_proof_complete") is not True
        or results.get("action_values_finite") is not True
        or results.get("action_shape") != [15, 8]
    ):
        raise AntiochLiveError("completed Antioch proof has invalid policy evidence")
    checks = results.get("checks")
    if not isinstance(checks, list):
        raise AntiochLiveError("completed Antioch proof has no recorded checks")
    passed = {
        item.get("criterion")
        for item in checks
        if isinstance(item, dict) and item.get("passed") is True
    }
    if not REQUIRED_POC_CHECKS.issubset(passed):
        raise AntiochLiveError(
            "completed Antioch proof is missing required passed checks"
        )
    if scenario == "openpi_franka_pickup_v3":
        _validate_pickup_record(record, results, passed)
        _validate_pickup_configuration(results, initial_posture, camera_mounts)
    evidence = {
        "scenario_run_id": scenario_run_id,
        "run_phase": "completed",
        "run_outcome": "passed",
        "communication_verified": True,
        "policy_round_trips": round_trips,
        "exterior_observation_count": exterior_count,
        "wrist_observation_count": wrist_count,
        "first_accepted_render_sequence": first_sequence,
        "last_accepted_render_sequence": last_sequence,
        "action_horizon": 15,
        "action_dimension": 8,
    }
    if scenario == "openpi_franka_pickup_v3":
        evidence["pickup_verified"] = True
    return evidence


def _validate_pickup_configuration(results, initial_posture, camera_mounts) -> None:
    """Prevent a passed record from qualifying a different requested experiment."""
    for result_name, expected in (("initial_arm_posture", initial_posture),
                                  ("policy_camera_mounts", camera_mounts)):
        if expected is not None and results.get(result_name) != expected:
            raise AntiochLiveError(f"completed pickup does not match requested {result_name}")
    if initial_posture is not None and results.get("post_reset_controller") != "openpi_policy_only":
        raise AntiochLiveError("completed pickup does not establish policy-only control")


def _validate_pickup_record(record, results, passed) -> None:
    """Reject communication-only or incomplete evidence for the pickup identity."""
    if not REQUIRED_PICKUP_CHECKS.issubset(passed):
        raise AntiochLiveError("completed pickup lacks required measured checks")
    if (results.get("episode_objective") != "pickup"
            or results.get("termination_reason") != "pickup_complete"
            or results.get("pickup_success") is not True):
        raise AntiochLiveError("completed pickup has no physical task success")
    for name, minimum in (("end_effector_approach_m", 0.05),
                          ("maximum_cube_lift_m", 0.05), ("pickup_hold_seconds", 1.0)):
        value = results.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < minimum):
            raise AntiochLiveError(f"completed pickup has invalid {name}")
    _integer_result(results, "completed_action_chunks", minimum=2)
    _integer_result(results, "safe_targets_applied", minimum=10)
    _integer_result(results, "gripper_contact_samples", minimum=1)
    _validate_camera_quality_record(results)
    artifacts = record.get("artifacts")
    artifact = artifacts.get("policy-evidence.zip") if isinstance(artifacts, dict) else None
    if (not isinstance(artifact, dict) or not artifact.get("size_bytes")
            or artifact.get("sha256") != results.get("policy_evidence_sha256")
            or not re.fullmatch(r"[a-f0-9]{64}", str(results.get("policy_evidence_sha256", "")))):
        raise AntiochLiveError("completed pickup lacks its matching control evidence archive")


def _validate_camera_quality_record(results) -> None:
    for view in ("exterior", "wrist"):
        for suffix, lower, upper in (
            ("luminance_mean_max", 5.0, 220.0),
            ("near_white_fraction_max", 0.0, 0.60),
            ("dynamic_range_min", 32.0, 255.0),
        ):
            value = results.get(f"{view}_{suffix}")
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not lower <= value <= upper):
                raise AntiochLiveError("completed pickup has invalid camera quality")


def _require_single_attempt(scenario: str, reason: str) -> None:
    """Keep a finite evaluation failure from silently becoming another scenario."""
    if reason and scenario in {"openpi_franka_mk8s_live_v2", "openpi_franka_pickup_v3"}:
        raise AntiochLiveError(
            "finite Antioch evaluation lost its owner; automatic resubmission is disabled"
        )


def _wait_for_completed_poc_record(
    cli: AntiochCli,
    *,
    runtime: Path,
    scenario: str,
    scenario_run_id: str,
    initial_posture: str | None = None,
    camera_mounts: str | None = None,
) -> dict[str, Any]:
    """Wait briefly for the clean foreground exit to become durable."""

    if not scenario_run_id:
        raise AntiochLiveError("clean Antioch exit had no observed owned run identity")
    for attempt in range(COMPLETION_POLL_ATTEMPTS):
        record = cli.show(runtime, kind="scenario", remote_id=scenario_run_id)
        if record.get("phase") == "completed":
            return _completed_poc_evidence(
                record, scenario=scenario, scenario_run_id=scenario_run_id,
                initial_posture=initial_posture, camera_mounts=camera_mounts,
            )
        if attempt + 1 < COMPLETION_POLL_ATTEMPTS:
            time.sleep(COMPLETION_POLL_SECONDS)
    raise AntiochLiveError("clean Antioch exit did not persist a terminal record")


def _private_value(path: Path, *, label: str) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o077:
        raise AntiochLiveError(f"private {label} file is unavailable")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise AntiochLiveError(f"private {label} file is empty")
    return value


def _start_cluster_service(
    cli: AntiochCli,
    *,
    runtime: Path,
    project_id: str,
    scenario: str,
) -> str:
    """Build one revision and start its project session fail-closed.

    The current platform owns compute through a project-scoped session.  First
    prove this exact scenario absent, then let ``session new`` replace only an
    idle prior session. Structured retryable failures use capped backoff;
    identity, authentication, and active-work conflicts stay terminal.
    """

    _cancel_remote_live_runs(
        cli,
        runtime=runtime,
        project_id=project_id,
        scenario=scenario,
        attempts=5,
    )
    retryable_failures = 0
    while True:
        try:
            built = cli.project_build(runtime, service="sim")
            revision_id = str(built.get("revision_id") or "")
            if not revision_id:
                raise AntiochCliError(
                    "Antioch project build did not return a revision",
                    error_type="malformed_cli_output",
                )
            session = cli.session_new(runtime, revision=revision_id)
            if not str(session.get("session_id") or ""):
                raise AntiochCliError(
                    "Antioch session startup did not return a session",
                    error_type="malformed_cli_output",
                )
            return str(session["session_id"])
        except AntiochCliError as exc:
            if not exc.retryable:
                raise
            delay = RECOVERY_BACKOFF_SECONDS[
                min(retryable_failures, len(RECOVERY_BACKOFF_SECONDS) - 1)
            ]
            retryable_failures += 1
            time.sleep(delay)


def _release_owned_session(
    cli: AntiochCli,
    *,
    runtime: Path,
    project_id: str,
    session_id: str,
) -> dict[str, Any]:
    """Release only the exact interactive session created by this controller."""

    current = cli.session_status(runtime)
    if (
        current.get("project_id") != project_id
        or str(current.get("session_id") or "") != session_id
    ):
        raise AntiochLiveError(
            "current Antioch session no longer matches controller ownership"
        )
    released = cli.session_release(runtime)
    if str(released.get("session_id") or "") != session_id:
        raise AntiochLiveError(
            "Antioch released a session that did not match controller ownership"
        )
    return released


def _launch_vendor_successor(
    cli: AntiochCli,
    *,
    executable: Path,
    runtime: Path,
    project_id: str,
    scenario: str,
    timeout_seconds: int,
    initial_posture: str = "pregrasp",
    camera_mounts: str = "native_wide",
) -> VendorStreamProcess:
    """Prove exact absence before starting one foreground successor."""

    _cancel_remote_live_runs(
        cli,
        runtime=runtime,
        project_id=project_id,
        scenario=scenario,
        attempts=60,
    )
    return VendorStreamProcess.start(
        executable=executable,
        runtime=runtime,
        scenario=scenario,
        timeout_seconds=timeout_seconds,
        initial_posture=initial_posture,
        camera_mounts=camera_mounts,
    )


def run_cluster(args: argparse.Namespace) -> NoReturn:
    stop_file = Path(args.stop_file)
    # emptyDir survives container restarts. Clearing a predecessor's stop
    # marker can replay a completed proof or undo an operator's shutdown.
    if stop_file.exists() or stop_file.is_symlink():
        raise AntiochLiveError(
            "persisted stop marker forbids restart; use a fresh adapter identity"
        )
    private_root = Path(args.private_root)
    bundle = private_root / "live-bundle"
    _validate_bundle(bundle)
    project_id = _private_value(private_root / "project-id", label="project identity")
    accepted = _private_value(private_root / "antioch-terms", label="terms acceptance")
    if accepted != "YES":
        raise AntiochLiveError(
            "Antioch terms acceptance is not the exact required value"
        )
    os.environ["NPA_ANTIOCH_ACCEPT_TERMS"] = accepted
    os.environ["ANTIOCH_CONFIG_DIR"] = str(private_root / "antioch-config")

    state_path = Path(args.state_path)
    root = Path(args.runtime_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    runtime = root / f"runtime-{uuid.uuid4().hex}"
    _stage_project(Path(args.source).resolve(), runtime, project_id)
    session_id = uuid.uuid4().hex
    cli_path = ensure_runtime()
    cli = AntiochCli(cli_path, config_dir=str(private_root / "antioch-config"))
    vendor: VendorStreamProcess | None = None
    port_bridge: VendorPortProcess | None = None
    antioch_session_id = ""
    stopping = False
    cleanup_complete = False
    failed = False
    cleanup_error: AntiochCliError | AntiochLiveError | None = None
    completion_evidence: dict[str, Any] = {}
    health: StateHealthServer | None = None

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        if cleanup_complete:
            raise SystemExit(0)
        stopping = True
        stop_file.touch(mode=0o600, exist_ok=True)
        if vendor is not None and vendor.process.poll() is None:
            vendor.terminate()
        if port_bridge is not None and port_bridge.process.poll() is None:
            port_bridge.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    service_started = False
    try:
        _write_state(
            state_path,
            status="starting",
            session_status="awaiting_owner",
            owner_identity=args.owner_identity,
            session_id=session_id,
            scenario=args.scenario,
            heartbeat_unix=0.0,
        )
        health = StateHealthServer(
            port=args.health_port,
            checks={
                "/ready": lambda: (
                    probe(
                        state_path,
                        component="controller",
                        expected_owner_identity=args.owner_identity,
                        max_age_seconds=30.0,
                    )
                    == 0
                ),
                "/live": lambda: (
                    probe(
                        state_path,
                        component="controller-liveness",
                        expected_owner_identity=args.owner_identity,
                        max_age_seconds=180.0,
                    )
                    == 0
                ),
            },
        )
        health.start()
        startup_state = {
            "status": "starting",
            "session_status": "starting_session",
            "owner_identity": args.owner_identity,
            "session_id": session_id,
            "scenario": args.scenario,
            "heartbeat_unix": 0.0,
        }
        with _recovery_heartbeat(state_path, **startup_state):
            antioch_session_id = _start_cluster_service(
                cli,
                runtime=runtime,
                project_id=project_id,
                scenario=args.scenario,
            )
        service_started = True
        _stage_runtime_source(cli, runtime=runtime)
        _stage_private_bundle(cli, runtime=runtime, client_bundle=bundle)
        port_bridge = VendorPortProcess.start(
            cli=cli,
            executable=Path(cli_path),
            runtime=runtime,
        )
        # A predecessor foreground client cannot be adopted across a pod
        # lifecycle: its exact stream ownership belongs to that process.
        # Reconcile and cancel only this project's exact live run,
        # prove stable absence, then create one cluster-owned successor.
        vendor = _launch_vendor_successor(
            cli,
            executable=Path(cli_path),
            runtime=runtime,
            project_id=project_id,
            scenario=args.scenario,
            timeout_seconds=args.scenario_timeout_seconds,
            initial_posture=args.initial_posture,
            camera_mounts=args.camera_mounts,
        )
        supervisor_started = time.monotonic()
        last_restage = supervisor_started
        last_owned_heartbeat = 0.0
        last_run_id = ""
        consecutive_absence = 0
        consecutive_errors = 0
        recoveries = 0
        while not stopping:
            if stop_file.exists() and not stopping:
                request_stop(signal.SIGTERM, None)
                break
            vendor_exit_class, _vendor_exit_code = vendor.exit_snapshot()
            route_exit_class, _route_exit_code = port_bridge.exit_snapshot()
            if vendor_exit_class == "completed":
                completion_evidence = _wait_for_completed_poc_record(
                    cli,
                    runtime=runtime,
                    scenario=args.scenario,
                    scenario_run_id=last_run_id,
                    initial_posture=args.initial_posture,
                    camera_mounts=args.camera_mounts,
                )
                _write_state(
                    state_path,
                    status="completed",
                    session_status="retiring",
                    owner_identity=args.owner_identity,
                    session_id=session_id,
                    scenario=args.scenario,
                    heartbeat_unix=time.time(),
                    **completion_evidence,
                )
                break
            child_dead = vendor_exit_class != "running" or route_exit_class != "running"
            recovery_reason = _supervisor_recovery_reason(
                child_dead=child_dead,
                last_owned_heartbeat=last_owned_heartbeat,
                consecutive_absence=consecutive_absence,
                consecutive_errors=consecutive_errors,
                age_seconds=(
                    time.time() - last_owned_heartbeat
                    if last_owned_heartbeat
                    else time.monotonic() - supervisor_started
                ),
                startup_age_seconds=time.monotonic() - supervisor_started,
                max_age_seconds=args.session_max_age_seconds,
            )
            if not child_dead:
                try:
                    active = _active_run_snapshot(
                        cli,
                        runtime=runtime,
                        project_id=project_id,
                        scenario=args.scenario,
                        require_stream_owner=True,
                    )
                    consecutive_errors = 0
                    if active is None:
                        consecutive_absence += 1
                        recovery_reason = _supervisor_recovery_reason(
                            child_dead=False,
                            last_owned_heartbeat=last_owned_heartbeat,
                            consecutive_absence=consecutive_absence,
                            consecutive_errors=consecutive_errors,
                            age_seconds=(
                                time.time() - last_owned_heartbeat
                                if last_owned_heartbeat
                                else time.monotonic() - supervisor_started
                            ),
                            startup_age_seconds=time.monotonic() - supervisor_started,
                            max_age_seconds=args.session_max_age_seconds,
                        )
                        _write_state(
                            state_path,
                            status=("degraded" if last_owned_heartbeat else "starting"),
                            session_status="owner_absent",
                            owner_identity=args.owner_identity,
                            session_id=session_id,
                            scenario=args.scenario,
                            scenario_run_id=last_run_id,
                            heartbeat_unix=last_owned_heartbeat,
                            recoveries=recoveries,
                            vendor_process_status=vendor.exit_snapshot()[0],
                            route_process_status=port_bridge.exit_snapshot()[0],
                        )
                    else:
                        if str(active["session_id"]) != antioch_session_id:
                            raise AntiochLiveReconcileError(
                                "active scenario does not belong to the controller session"
                            )
                        consecutive_absence = 0
                        last_owned_heartbeat = time.time()
                        last_run_id = str(active["scenario_run_id"])
                        _write_state(
                            state_path,
                            status="running",
                            session_status="owned",
                            owner_identity=args.owner_identity,
                            session_id=session_id,
                            scenario=args.scenario,
                            scenario_run_id=last_run_id,
                            run_phase=str(active.get("phase") or ""),
                            stream_state=str(active.get("stream_state") or ""),
                            heartbeat_unix=last_owned_heartbeat,
                            recoveries=recoveries,
                            vendor_process_status=vendor.exit_snapshot()[0],
                            route_process_status=port_bridge.exit_snapshot()[0],
                            vendor_output_bytes=vendor.output_snapshot()[0],
                            vendor_output_age_seconds=round(
                                vendor.output_snapshot()[1], 3
                            ),
                            controller_pid=os.getpid(),
                            vendor_pid=vendor.process.pid,
                            vendor_parent_pid=os.getpid(),
                            vendor_process_group_isolated=(
                                os.getpgid(vendor.process.pid) == vendor.process.pid
                            ),
                            route_pid=port_bridge.process.pid,
                            route_parent_pid=os.getpid(),
                            route_process_group_isolated=(
                                os.getpgid(port_bridge.process.pid)
                                == port_bridge.process.pid
                            ),
                            antioch_session_id=str(active["session_id"]),
                            antioch_session_state=str(active["session_state"]),
                            antioch_session_access_phase=str(
                                active["session_access_phase"]
                            ),
                            sim_service_state=str(active["service_state"]),
                            sim_process_healthy=bool(active["service_process_healthy"]),
                            antioch_session_ready=bool(active["session_ready"]),
                            session_observed_at=float(active["session_observed_at"]),
                            transport="same-pod-antioch-named-route-double-wss",
                            dev_vm_in_data_path=False,
                        )
                except (
                    AntiochCliError,
                    AntiochLiveError,
                    AntiochLiveReconcileError,
                    RuntimeError,
                ) as exc:
                    consecutive_errors += 1
                    age = (
                        time.time() - last_owned_heartbeat
                        if last_owned_heartbeat
                        else time.monotonic() - supervisor_started
                    )
                    recovery_reason = _supervisor_recovery_reason(
                        child_dead=False,
                        last_owned_heartbeat=last_owned_heartbeat,
                        consecutive_absence=consecutive_absence,
                        consecutive_errors=consecutive_errors,
                        age_seconds=age,
                        startup_age_seconds=time.monotonic() - supervisor_started,
                        max_age_seconds=args.session_max_age_seconds,
                    )
                    _write_state(
                        state_path,
                        status="degraded",
                        session_status="unreadable",
                        owner_identity=args.owner_identity,
                        session_id=session_id,
                        scenario=args.scenario,
                        scenario_run_id=last_run_id,
                        heartbeat_unix=last_owned_heartbeat,
                        error_type=type(exc).__name__,
                        recoveries=recoveries,
                        vendor_process_status=vendor.exit_snapshot()[0],
                        route_process_status=port_bridge.exit_snapshot()[0],
                    )
            _require_single_attempt(args.scenario, recovery_reason)
            if recovery_reason:
                recoveries += 1
                exit_class, exit_code = vendor.exit_snapshot()
                recovery_state = {
                    "status": "recovering",
                    "session_status": "replacing_supervisor",
                    "owner_identity": args.owner_identity,
                    "session_id": session_id,
                    "scenario": args.scenario,
                    "scenario_run_id": last_run_id,
                    "heartbeat_unix": last_owned_heartbeat,
                    "recovery_reason": recovery_reason,
                    "recoveries": recoveries,
                    "vendor_exit_class": exit_class,
                    "vendor_exit_code": exit_code,
                }
                _write_state(state_path, **recovery_state)
                with _recovery_heartbeat(state_path, **recovery_state):
                    vendor.terminate()
                    port_bridge.terminate()
                    _cancel_remote_live_runs(
                        cli,
                        runtime=runtime,
                        project_id=project_id,
                        scenario=args.scenario,
                        attempts=60,
                    )
                    try:
                        cli.service_exec(runtime, "sim", ["/bin/true"])
                    except AntiochCliError:
                        antioch_session_id = _start_cluster_service(
                            cli,
                            runtime=runtime,
                            project_id=project_id,
                            scenario=args.scenario,
                        )
                    _stage_runtime_source(cli, runtime=runtime)
                    _stage_private_bundle(cli, runtime=runtime, client_bundle=bundle)
                    port_bridge = VendorPortProcess.start(
                        cli=cli,
                        executable=Path(cli_path),
                        runtime=runtime,
                    )
                    delay = RECOVERY_BACKOFF_SECONDS[
                        min(recoveries - 1, len(RECOVERY_BACKOFF_SECONDS) - 1)
                    ]
                    if stop_file.exists():
                        break
                    time.sleep(delay)
                    vendor = _launch_vendor_successor(
                        cli,
                        executable=Path(cli_path),
                        runtime=runtime,
                        project_id=project_id,
                        scenario=args.scenario,
                        timeout_seconds=args.scenario_timeout_seconds,
                        initial_posture=args.initial_posture,
                        camera_mounts=args.camera_mounts,
                    )
                supervisor_started = time.monotonic()
                last_restage = supervisor_started
                consecutive_absence = 0
                consecutive_errors = 0
                last_owned_heartbeat = 0.0
                last_run_id = ""
            elif time.monotonic() - last_restage >= RESTAGE_INTERVAL_SECONDS:
                # Finite supported probes detect a recycled service or missing
                # bytes without retaining a remote exec process.
                try:
                    cli.service_exec(runtime, "sim", ["/bin/true"])
                    source_matches = _runtime_source_matches(cli, runtime=runtime)
                    bundle_matches = _private_bundle_matches(
                        cli, runtime=runtime, client_bundle=bundle
                    )
                except AntiochCliError:
                    port_bridge.terminate()
                    antioch_session_id = _start_cluster_service(
                        cli,
                        runtime=runtime,
                        project_id=project_id,
                        scenario=args.scenario,
                    )
                    _stage_runtime_source(cli, runtime=runtime)
                    _stage_private_bundle(cli, runtime=runtime, client_bundle=bundle)
                    port_bridge = VendorPortProcess.start(
                        cli=cli,
                        executable=Path(cli_path),
                        runtime=runtime,
                    )
                else:
                    if not source_matches:
                        _stage_runtime_source(cli, runtime=runtime)
                    if not bundle_matches:
                        _stage_private_bundle(
                            cli, runtime=runtime, client_bundle=bundle
                        )
                last_restage = time.monotonic()
            time.sleep(args.session_poll_seconds)
    except Exception as exc:
        failed = True
        _write_state(
            state_path,
            status="failed",
            session_status="failed",
            owner_identity=args.owner_identity,
            session_id=session_id,
            scenario=args.scenario,
            heartbeat_unix=time.time(),
            error_type=type(exc).__name__,
        )
        raise
    finally:
        stop_file.touch(mode=0o600, exist_ok=True)
        if vendor is not None and vendor.process.poll() is None:
            vendor.terminate()
        if port_bridge is not None and port_bridge.process.poll() is None:
            port_bridge.terminate()
        if service_started:
            try:
                _cancel_remote_live_runs(
                    cli,
                    runtime=runtime,
                    project_id=project_id,
                    scenario=args.scenario,
                    attempts=5,
                )
                _release_owned_session(
                    cli,
                    runtime=runtime,
                    project_id=project_id,
                    session_id=antioch_session_id,
                )
            except (AntiochCliError, AntiochLiveError) as exc:
                cleanup_error = exc
                _write_state(
                    state_path,
                    status="cleanup_failed",
                    session_status="cleanup_failed",
                    owner_identity=args.owner_identity,
                    session_id=session_id,
                    scenario=args.scenario,
                    heartbeat_unix=time.time(),
                    error_type=type(exc).__name__,
                )
        terminal_status = (
            "cleanup_failed"
            if cleanup_error is not None
            else "failed"
            if failed
            else "stopped"
        )
        _write_state(
            state_path,
            status=terminal_status,
            session_status=terminal_status,
            owner_identity=args.owner_identity,
            session_id=session_id,
            scenario=args.scenario,
            heartbeat_unix=time.time(),
            **completion_evidence,
        )
        if failed or cleanup_error is not None:
            if health is not None:
                health.close()
            if cleanup_error is not None and not failed:
                raise cleanup_error
        else:
            # Keep the terminal evidence owned by this exact PID until the
            # observer scales the Deployment. Exiting here lets restartPolicy
            # Always replace ``stopped`` with a fresh ``starting`` state before
            # stop_cluster's next poll, wedging safe scale-down.
            cleanup_complete = True
            while True:
                time.sleep(60)
                _write_state(
                    state_path,
                    status="stopped",
                    session_status="terminal",
                    owner_identity=args.owner_identity,
                    session_id=session_id,
                    scenario=args.scenario,
                    heartbeat_unix=time.time(),
                    **completion_evidence,
                )
    raise AssertionError("Antioch controller reached an impossible exit path")


def probe(
    path: Path,
    *,
    component: str,
    expected_owner_identity: str,
    max_age_seconds: float,
) -> int:
    try:
        state = _read_state(path)
    except (OSError, TypeError, json.JSONDecodeError):
        return 1
    return (
        0
        if _state_ready(
            state,
            component=component,
            expected_owner_identity=expected_owner_identity,
            max_age_seconds=max_age_seconds,
        )
        else 1
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--source", default="/opt/npa/antioch-openpi-live")
    run.add_argument("--private-root", default="/run/npa-antioch-private")
    run.add_argument("--runtime-root", default="/var/lib/npa-antioch-live")
    run.add_argument("--state-path", default="/var/run/npa-antioch/controller.json")
    run.add_argument("--stop-file", default="/var/run/npa-antioch/stop")
    run.add_argument("--scenario", default="openpi_franka_pickup_v3")
    run.add_argument("--scenario-timeout-seconds", type=int, default=14_400)
    run.add_argument("--initial-posture", choices=("pregrasp", "droid"), default="pregrasp")
    run.add_argument(
        "--camera-mounts", choices=("native_wide", "droid_reference", "task_view"), default="native_wide"
    )
    run.add_argument("--owner-identity", required=True)
    run.add_argument("--health-port", type=int, default=18_080)
    run.add_argument("--session-poll-seconds", type=float, default=SESSION_POLL_SECONDS)
    run.add_argument(
        "--session-max-age-seconds",
        type=float,
        default=DEFAULT_CONTROLLER_MAX_AGE_SECONDS,
    )
    check = subparsers.add_parser("probe")
    check.add_argument("--state-path", required=True)
    check.add_argument(
        "--component",
        choices=("controller", "controller-liveness", "relay", "relay-liveness"),
        required=True,
    )
    check.add_argument("--expected-owner-identity", required=True)
    check.add_argument("--max-age-seconds", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "probe":
        return probe(
            Path(args.state_path),
            component=args.component,
            expected_owner_identity=args.expected_owner_identity,
            max_age_seconds=args.max_age_seconds,
        )
    return run_cluster(args)


if __name__ == "__main__":
    raise SystemExit(main())
