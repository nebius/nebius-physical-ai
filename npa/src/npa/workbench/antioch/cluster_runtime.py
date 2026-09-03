"""PID-1 controller for the cluster-native Antioch live adapter pod.

The supported Antioch service tunnel is created in this pod.  A sibling relay
container shares the pod network namespace and connects to the tunnel only on
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
from typing import Any

from .live import (
    AntiochLiveError,
    _cancel_remote_live_runs,
    _stage_private_bundle,
    _stage_project,
    _stage_runtime_source,
    _validate_bundle,
)
from .health import StateHealthServer
from .live_reconcile import AntiochLiveReconcileError, _active_run_snapshot
from .runtime import ensure_runtime
from .vendor_cli import AntiochCli, AntiochCliError


SCHEMA = "npa.workbench.antioch-cluster-live.v3"
SCHEMA_VERSION = 3
RELAY_SCHEMA_VERSION = 2
DEFAULT_CONTROLLER_MAX_AGE_SECONDS = 30.0
DEFAULT_RELAY_MAX_AGE_SECONDS = 150.0
STATE_READ_ATTEMPTS = 3
DAEMON_POLL_SECONDS = 5.0
DAEMON_ABSENCE_THRESHOLD = 3
DAEMON_ERROR_THRESHOLD = 3
DAEMON_STARTUP_GRACE_SECONDS = 600.0
ASSIGNMENT_BOUND_MARKER = "SSH is already bound to another local client"
RESTAGE_INTERVAL_SECONDS = 60.0
RECOVERY_BACKOFF_SECONDS = (2.0, 5.0, 10.0, 20.0, 30.0)
_METRIC_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_CAMERA_REJECTION_VIEWS = frozenset({"exterior", "wrist", "pair"})
_CAMERA_REJECTION_REASONS = frozenset(
    {
        "missing",
        "wrong_shape",
        "non_finite",
        "blank",
        "flat",
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
    ) -> "VendorStreamProcess":
        started = time.monotonic()
        process = subprocess.Popen(
            [
                str(executable),
                "scenario",
                "run",
                "--scenario",
                scenario,
                "--timeout",
                str(timeout_seconds),
                "--stream",
                "--verbose",
            ],
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
        daemon_observed = state.get("daemon_observed_at")
        rome_observed = state.get("rome_guest_observed_at")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (daemon_observed, rome_observed)
        ):
            return False
        return bool(
            state.get("vendor_process_status") == "running"
            and state.get("daemon_guest_state") == "healthy"
            and int(state.get("controller_pid") or 0) == 1
            and int(state.get("vendor_parent_pid") or 0) == 1
            and int(state.get("vendor_pid") or 0) > 1
            and state.get("vendor_process_group_isolated") is True
            and int(state.get("scenario_session_leases") or 0) == 1
            and int(state.get("process_leases") or 0) >= 1
            and int(state.get("stream_leases") or 0) == 1
            and -5.0 <= observed_now - float(daemon_observed) <= max_age_seconds
            and -5.0 <= observed_now - float(rome_observed) <= max_age_seconds
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
        return state.get("status") in {"starting", "recovering", "stopped"}
    heartbeat = state.get("heartbeat_unix")
    if not isinstance(heartbeat, (int, float)) or isinstance(heartbeat, bool):
        return False
    age = observed_now - float(heartbeat)
    if age < -5.0 or age > max_age_seconds:
        return False
    if component == "controller":
        return bool(
            state.get("status") == "running"
            and state.get("daemon_status") == "owned"
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
        and consecutive_absence >= DAEMON_ABSENCE_THRESHOLD
        and age_seconds > max_age_seconds
    ):
        return "daemon_owner_absent"
    if not last_owned_heartbeat and startup_age_seconds >= DAEMON_STARTUP_GRACE_SECONDS:
        return "daemon_owner_startup_timeout"
    if consecutive_errors >= DAEMON_ERROR_THRESHOLD and age_seconds > max_age_seconds:
        return "daemon_state_unreadable"
    return ""


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
) -> bool:
    """Start the service, recovering typed transient control-plane failures.

    A provider machine can outlive the Kubernetes pod whose local SSH client
    owned it.  The supported service command then fails deterministically
    before any new stream is dispatched.  Cancel the exact project's live run
    first and release that exact assignment.  Structured retryable failures use
    capped backoff until the control plane recovers; fatal failures stay
    fail-closed.
    """

    released_assignment = False
    retryable_failures = 0
    while True:
        try:
            cli.services_build(runtime, service="sim")
            cli.services_up(runtime)
            return released_assignment
        except AntiochCliError as exc:
            if ASSIGNMENT_BOUND_MARKER in str(exc) and not released_assignment:
                _cancel_remote_live_runs(
                    cli,
                    runtime=runtime,
                    project_id=project_id,
                    scenario=scenario,
                    attempts=5,
                )
                cli.machine_release(runtime, project_id=project_id)
                released_assignment = True
                retryable_failures = 0
                continue
            if not exc.retryable:
                raise
            delay = RECOVERY_BACKOFF_SECONDS[
                min(retryable_failures, len(RECOVERY_BACKOFF_SECONDS) - 1)
            ]
            retryable_failures += 1
            time.sleep(delay)


def _launch_vendor_successor(
    cli: AntiochCli,
    *,
    executable: Path,
    runtime: Path,
    project_id: str,
    scenario: str,
    timeout_seconds: int,
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
    )


def run_cluster(args: argparse.Namespace) -> int:
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
    stop_file = Path(args.stop_file)
    # A container restart keeps the pod's emptyDir. A prior SIGTERM path may
    # have left its stop marker behind; the new owner session must not inherit it.
    stop_file.unlink(missing_ok=True)
    session_id = uuid.uuid4().hex
    cli_path = ensure_runtime()
    cli = AntiochCli(cli_path, config_dir=str(private_root / "antioch-config"))
    vendor: VendorStreamProcess | None = None
    stopping = False
    cleanup_complete = False
    failed = False
    cleanup_error: AntiochCliError | AntiochLiveError | None = None
    health: StateHealthServer | None = None

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        if cleanup_complete:
            raise SystemExit(0)
        stopping = True
        stop_file.touch(mode=0o600, exist_ok=True)
        if vendor is not None and vendor.process.poll() is None:
            vendor.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    service_started = False
    try:
        _write_state(
            state_path,
            status="starting",
            daemon_status="awaiting_owner",
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
            "daemon_status": "starting_service",
            "owner_identity": args.owner_identity,
            "session_id": session_id,
            "scenario": args.scenario,
            "heartbeat_unix": 0.0,
        }
        with _recovery_heartbeat(state_path, **startup_state):
            _start_cluster_service(
                cli,
                runtime=runtime,
                project_id=project_id,
                scenario=args.scenario,
            )
        service_started = True
        _stage_runtime_source(cli, runtime=runtime)
        _stage_private_bundle(cli, runtime=runtime, client_bundle=bundle)
        # A predecessor foreground client cannot be adopted across a pod
        # lifecycle: its exact daemon session heartbeat belongs to that
        # process. Reconcile and cancel only this project's exact live run,
        # prove stable absence, then create one cluster-owned successor.
        vendor = _launch_vendor_successor(
            cli,
            executable=Path(cli_path),
            runtime=runtime,
            project_id=project_id,
            scenario=args.scenario,
            timeout_seconds=args.scenario_timeout_seconds,
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
            child_dead = vendor.process.poll() is not None
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
                max_age_seconds=args.daemon_max_age_seconds,
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
                            max_age_seconds=args.daemon_max_age_seconds,
                        )
                        _write_state(
                            state_path,
                            status=("degraded" if last_owned_heartbeat else "starting"),
                            daemon_status="owner_absent",
                            owner_identity=args.owner_identity,
                            session_id=session_id,
                            scenario=args.scenario,
                            scenario_run_id=last_run_id,
                            heartbeat_unix=last_owned_heartbeat,
                            recoveries=recoveries,
                            vendor_process_status=vendor.exit_snapshot()[0],
                        )
                    else:
                        consecutive_absence = 0
                        last_owned_heartbeat = time.time()
                        last_run_id = str(active["scenario_run_id"])
                        _write_state(
                            state_path,
                            status="running",
                            daemon_status="owned",
                            owner_identity=args.owner_identity,
                            session_id=session_id,
                            scenario=args.scenario,
                            scenario_run_id=last_run_id,
                            run_phase=str(active.get("phase") or ""),
                            stream_state=str(active.get("stream_state") or ""),
                            heartbeat_unix=last_owned_heartbeat,
                            recoveries=recoveries,
                            vendor_process_status=vendor.exit_snapshot()[0],
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
                            daemon_guest_state=str(
                                active.get("daemon_guest_state") or ""
                            ),
                            daemon_observed_at=float(active["daemon_observed_at"]),
                            rome_guest_observed_at=float(
                                active["rome_guest_observed_at"]
                            ),
                            scenario_session_leases=int(
                                active["scenario_session_leases"]
                            ),
                            process_leases=int(active["process_leases"]),
                            stream_leases=int(active["stream_leases"]),
                            transport="same-pod-antioch-tunnel-double-wss",
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
                        max_age_seconds=args.daemon_max_age_seconds,
                    )
                    _write_state(
                        state_path,
                        status="degraded",
                        daemon_status="unreadable",
                        owner_identity=args.owner_identity,
                        session_id=session_id,
                        scenario=args.scenario,
                        scenario_run_id=last_run_id,
                        heartbeat_unix=last_owned_heartbeat,
                        error_type=type(exc).__name__,
                        recoveries=recoveries,
                        vendor_process_status=vendor.exit_snapshot()[0],
                    )
            if recovery_reason:
                recoveries += 1
                exit_class, exit_code = vendor.exit_snapshot()
                recovery_state = {
                    "status": "recovering",
                    "daemon_status": "replacing_supervisor",
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
                    _cancel_remote_live_runs(
                        cli,
                        runtime=runtime,
                        project_id=project_id,
                        scenario=args.scenario,
                        attempts=60,
                    )
                    try:
                        cli.services_exec(runtime, "sim", ["/bin/true"])
                    except AntiochCliError:
                        try:
                            cli.services_up(runtime)
                        except AntiochCliError:
                            cli.services_build(runtime, service="sim")
                            cli.services_up(runtime)
                    _stage_runtime_source(cli, runtime=runtime)
                    _stage_private_bundle(cli, runtime=runtime, client_bundle=bundle)
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
                    )
                supervisor_started = time.monotonic()
                last_restage = supervisor_started
                consecutive_absence = 0
                consecutive_errors = 0
                last_owned_heartbeat = 0.0
                last_run_id = ""
            elif time.monotonic() - last_restage >= RESTAGE_INTERVAL_SECONDS:
                # Finite supported probes ensure a recycled service is rebuilt
                # and atomically restaged without an exec-held daemon.
                try:
                    cli.services_exec(runtime, "sim", ["/bin/true"])
                except AntiochCliError:
                    try:
                        cli.services_up(runtime)
                    except AntiochCliError:
                        cli.services_build(runtime, service="sim")
                        cli.services_up(runtime)
                    _stage_runtime_source(cli, runtime=runtime)
                    _stage_private_bundle(cli, runtime=runtime, client_bundle=bundle)
                last_restage = time.monotonic()
            time.sleep(args.daemon_poll_seconds)
    except Exception as exc:
        failed = True
        _write_state(
            state_path,
            status="failed",
            daemon_status="failed",
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
        if service_started:
            try:
                _cancel_remote_live_runs(
                    cli,
                    runtime=runtime,
                    project_id=project_id,
                    scenario=args.scenario,
                    attempts=5,
                )
                cli.services_down(runtime)
            except (AntiochCliError, AntiochLiveError) as exc:
                cleanup_error = exc
                _write_state(
                    state_path,
                    status="cleanup_failed",
                    daemon_status="cleanup_failed",
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
            daemon_status=terminal_status,
            owner_identity=args.owner_identity,
            session_id=session_id,
            scenario=args.scenario,
            heartbeat_unix=time.time(),
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
                    daemon_status="terminal",
                    owner_identity=args.owner_identity,
                    session_id=session_id,
                    scenario=args.scenario,
                    heartbeat_unix=time.time(),
                )


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
    run.add_argument("--scenario", default="openpi_franka_mk8s_live_v2")
    run.add_argument("--scenario-timeout-seconds", type=int, default=14_400)
    run.add_argument("--owner-identity", required=True)
    run.add_argument("--health-port", type=int, default=18_080)
    run.add_argument("--daemon-poll-seconds", type=float, default=DAEMON_POLL_SECONDS)
    run.add_argument(
        "--daemon-max-age-seconds",
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
