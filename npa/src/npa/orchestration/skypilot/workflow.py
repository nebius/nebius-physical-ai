"""Workflow submission helpers for NPA's SkyPilot orchestration layer."""

from __future__ import annotations

import re
import hashlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from npa.orchestration.skypilot._bin import (
    SkyBin,
    SkyPilotConfigError,
    SkyPilotNotInstalledError,
    SkyPilotVersionError,
    ensure_skypilot_version,
    resolve_config,
)
from npa.orchestration.skypilot.cleanup import sky_environment
from npa.orchestration.skypilot.controller import (
    DEFAULT_CONTROLLER_BACKEND,
    ControllerBackend,
    apply_controller_override,
)
from npa.orchestration.skypilot.diagnostics import (
    SkyPilotDiagnosis,
    diagnose_skypilot_output,
)
from npa.orchestration.skypilot.json_output import (
    is_verified_empty_queue_result,
    parse_single_json_document,
    queue_rows_from_output,
)
from npa.orchestration.skypilot.launch_transaction import (
    ControllerState,
    EvidenceState,
    FailureCategory,
    KubectlApiProbe,
    LaunchState,
    LaunchTransactionError,
    LaunchTransactionResult,
    ReconciliationEvidence,
    ReconciliationState,
    RecoveryPolicy,
    StabilityPolicy,
    classify_failure,
    logical_launch_identity,
    run_launch_transaction,
    wait_for_api_stability,
)
from npa.orchestration.skypilot.workflow_state import redact_text


JOBS_CONTROLLER_PREFIX = "sky-jobs-controller-"
HEALTHY_CONTROLLER_STATUS = "UP"
# A managed-jobs controller may be UP or autostopped (STOPPED). Both are safe to
# launch against: `sky jobs launch` restarts a STOPPED controller on demand.
# Only genuinely transient states (e.g. INIT / provisioning) should block a
# concurrent launch. Treating STOPPED as unhealthy made a stale/autostopped
# controller wait for a status ("UP") it never reaches without a launch, so the
# preflight burned the whole timeout and failed a submit that would have worked.
READY_CONTROLLER_STATUSES = frozenset({HEALTHY_CONTROLLER_STATUS, "STOPPED"})


def _stable_sky_cwd(isolated_config_dir: Path | None) -> str:
    """Return a durable directory to run the ``sky`` CLI from.

    SkyPilot runs a long-lived local API server daemon that performs all
    provisioning and file sync, and it inherits the working directory of the
    process that first starts it. NPA workflows frequently execute from
    short-lived temp directories (e.g. ``tempfile.mkdtemp``). If the daemon
    auto-starts with such a cwd and that directory is later cleaned up, every
    later operation fails with ``getcwd() failed: No such file or directory``
    and ``rsync`` exits with code 3 ("Failed to set up SkyPilot runtime").
    Pinning sky invocations to a durable directory keeps the daemon's cwd valid.
    """

    for candidate in (isolated_config_dir, Path.home()):
        if candidate is None:
            continue
        try:
            path = Path(candidate)
            if path.is_dir():
                return str(path)
        except OSError:
            continue
    return str(Path.home())


@dataclass
class WorkflowResult:
    """Result of submitting or querying a SkyPilot managed workflow."""

    status: str
    job_id: str = ""
    log_paths: dict[str, str] = field(default_factory=dict)
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    submitted_yaml_path: str = ""
    error: str = ""
    launch_transaction: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.error


@dataclass(frozen=True)
class ManagedJobEvidence:
    """One exact managed-job lookup through NPA's pinned SkyPilot runtime."""

    outcome: str
    job_id: str = ""
    status: str = ""
    task_rows: tuple[dict[str, Any], ...] = ()
    error: str = ""


class SkyPilotSubmitError(RuntimeError):
    """Raised when a SkyPilot workflow cannot be submitted."""

    def __init__(
        self,
        message: str,
        *,
        transaction: LaunchTransactionResult | None = None,
    ) -> None:
        super().__init__(message)
        self.transaction = transaction


class _SkyPilotLaunchCommandError(RuntimeError):
    """Preserve launch command evidence for the central failure classifier."""

    def __init__(
        self,
        message: str,
        result: subprocess.CompletedProcess[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result


def _controller_region_from_infra(
    infra: str,
    controller_backend: ControllerBackend,
) -> str | None:
    """Derive a Kubernetes controller context from a ``k8s/<context>`` infra.

    Co-locating the managed-jobs controller with the job's kube context keeps
    the controller in the same region as its tasks, so launch-time bucket mounts
    resolve against the matching object-storage endpoint instead of falling back
    to a different region. Returns ``None`` for non-Kubernetes backends or when
    ``infra`` does not name a context.
    """

    if controller_backend != "kubernetes":
        return None
    value = (infra or "").strip()
    for prefix in ("k8s/", "kubernetes/"):
        if value.startswith(prefix):
            context = value[len(prefix) :].strip()
            return context or None
    return None


def _selected_kube_context(
    infra: str,
    *,
    env: Mapping[str, str],
    controller_backend: ControllerBackend,
) -> str:
    """Resolve the exact context SkyPilot will use without changing it."""

    explicit = _controller_region_from_infra(infra, controller_backend)
    if explicit:
        return explicit
    if controller_backend != "kubernetes":
        return ""
    kubectl = shutil.which("kubectl")
    if kubectl is None:
        return ""
    try:
        result = subprocess.run(
            [kubectl, "config", "current-context"],
            env=dict(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def submit_workflow(
    yaml_path: Path,
    run_id: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND,
    infra: str = "",
    secret_envs: Sequence[str] | None = None,
    require_controller_up: bool = False,
    extra_env: Mapping[str, str] | None = None,
    timeout: int = 1800,
    controller_preflight_timeout: int = 300,
    controller_preflight_interval: float = 15.0,
    stream_output: bool = True,
    echo: Callable[[str], None] | None = None,
    logical_launch_id: str = "",
    transaction_recorder: Callable[[dict[str, Any]], None] | None = None,
    stability_policy: StabilityPolicy = StabilityPolicy(),
    recovery_policy: RecoveryPolicy = RecoveryPolicy(),
    stability_probe: Callable[[], Any] | None = None,
    transaction_clock: Callable[[], float] = time.monotonic,
    transaction_sleeper: Callable[[float], None] = time.sleep,
    transaction_random: Callable[[], float] | None = None,
    launch_lock_root: Path | None = None,
) -> WorkflowResult:
    """Submit a SkyPilot YAML through NPA's controller convention."""

    yaml_path = Path(yaml_path)
    submission_dir: Path | None = None
    owned_submission_dir: Path | None = None
    prepared_yaml: Path | None = None
    streamer: _LaunchStreamer | None = None
    try:
        runtime_config = resolve_config(
            sky_bin=sky_bin,
            global_config_path=config_path,
            isolated_config_dir=isolated_config_dir,
        )
        docs = _load_yaml_documents(yaml_path)
        if not docs:
            raise ValueError("SkyPilot YAML is empty")
        submission_dir = _submission_dir(run_id, runtime_config.isolated_config_dir)
        if runtime_config.isolated_config_dir is None:
            owned_submission_dir = submission_dir
        prepared_yaml = submission_dir / "workflow.yaml"
        shutil.copy2(yaml_path, prepared_yaml)
        # The task YAML can carry registry/docker auth + S3 creds; keep it owner-only.
        _chmod_owner_only(prepared_yaml)
        sky_executable = str(ensure_skypilot_version(runtime_config.sky_bin))
        global_config = apply_controller_override(
            _load_base_config(runtime_config.global_config_path),
            controller_backend=controller_backend,
            controller_region=_controller_region_from_infra(infra, controller_backend),
        )
        generated_config_path = submission_dir / "skypilot-config.yaml"
        generated_config_path.write_text(
            yaml.safe_dump(global_config, sort_keys=False), encoding="utf-8"
        )
        _chmod_owner_only(generated_config_path)
        env = sky_environment(runtime_config.isolated_config_dir)
        for key, value in (extra_env or {}).items():
            if value:
                env[key] = value
        env["SKYPILOT_GLOBAL_CONFIG"] = str(generated_config_path)

        cmd = [
            sky_executable,
            "jobs",
            "launch",
            "--name",
            run_id,
            "--detach-run",
            "--yes",
            str(prepared_yaml),
        ]
        if infra:
            cmd[-1:-1] = ["--infra", infra]
        for secret_name in secret_envs or ():
            if env.get(secret_name):
                cmd[-1:-1] = ["--secret", secret_name]
        stable_cwd = _stable_sky_cwd(runtime_config.isolated_config_dir)
        controller_state = _wait_for_healthy_jobs_controller(
            sky_executable,
            env=env,
            timeout=controller_preflight_timeout,
            interval=controller_preflight_interval,
            require_existing=require_controller_up,
            cwd=stable_cwd,
        )
        streamer = (
            _LaunchStreamer(
                echo or _default_launch_echo,
                optional_nebius_profile=_nebius_profile_is_optional(
                    docs,
                    controller_backend=controller_backend,
                    infra=infra,
                ),
            )
            if stream_output
            else None
        )
        selected_context = _selected_kube_context(
            infra,
            env=env,
            controller_backend=controller_backend,
        )
        readiness_probe = stability_probe
        if readiness_probe is None and controller_backend == "kubernetes":
            readiness_probe = KubectlApiProbe(
                env=env,
                context=selected_context,
                clock=transaction_clock,
            )

        def _readiness():
            if controller_backend != "kubernetes":
                from npa.orchestration.skypilot.launch_transaction import StabilityResult

                return StabilityResult(EvidenceState.READY, FailureCategory.NONE)
            assert readiness_probe is not None
            return wait_for_api_stability(
                readiness_probe,
                policy=stability_policy,
                clock=transaction_clock,
                sleeper=transaction_sleeper,
                progress=echo or _default_launch_echo,
            )

        def _reconcile() -> ReconciliationEvidence:
            return _reconcile_managed_job_env(
                run_id,
                env=env,
                sky_executable=sky_executable,
                cwd=stable_cwd,
            )

        def _launch() -> tuple[subprocess.CompletedProcess[str], list[SkyPilotDiagnosis]]:
            try:
                launch_result, diagnoses = _run_launch(
                    cmd,
                    env=env,
                    cwd=stable_cwd,
                    timeout=timeout,
                    log_dir=submission_dir,
                    streamer=streamer,
                )
            except subprocess.TimeoutExpired as exc:
                message = f"sky jobs launch timed out after {timeout}s"
                for diagnosis in streamer.diagnoses if streamer is not None else ():
                    message = f"{message}\n{diagnosis.render()}"
                raise _SkyPilotLaunchCommandError(message) from exc
            if launch_result.returncode != 0:
                raise _SkyPilotLaunchCommandError(
                    _format_submit_error(cmd, launch_result, streamed=diagnoses),
                    launch_result,
                )
            return launch_result, diagnoses

        def _classify(exc: BaseException) -> tuple[EvidenceState, FailureCategory]:
            command_result = getattr(exc, "result", None)
            return classify_failure(
                phase="launch",
                stdout=str(getattr(command_result, "stdout", "") or ""),
                stderr=str(getattr(command_result, "stderr", "") or ""),
                exception=exc,
            )

        identity = logical_launch_id or logical_launch_identity(
            str(runtime_config.isolated_config_dir or "default"),
            selected_context,
            run_id,
            hashlib.sha256(prepared_yaml.read_bytes()).hexdigest(),
        )
        random_source = transaction_random
        if random_source is None:
            import random as _random

            random_source = _random.random

        def _record_with_controller(payload: dict[str, Any]) -> None:
            if transaction_recorder is None:
                return
            enriched = dict(payload)
            enriched["controller"] = {
                "state": controller_state.value,
                "selected_context": selected_context,
            }
            transaction_recorder(enriched)

        try:
            transaction = run_launch_transaction(
                logical_id=identity,
                readiness=_readiness,
                launch=_launch,
                reconcile=_reconcile,
                classify_launch_error=_classify,
                recovery_policy=recovery_policy,
                lock_root=launch_lock_root,
                clock=transaction_clock,
                sleeper=transaction_sleeper,
                random_source=random_source,
                record=_record_with_controller,
                progress=echo or _default_launch_echo,
            )
        except LaunchTransactionError as exc:
            exc.result.controller = {
                "state": controller_state.value,
                "selected_context": selected_context,
            }
            if exc.result.state in {
                LaunchState.INDETERMINATE,
                LaunchState.TRANSIENT_API_FAILURE,
                LaunchState.INTERRUPTED,
            }:
                exc.result.operator_remedy = (
                    f"Re-run the identical submit arguments with `--resume-run {run_id}` "
                    "for this exact logical launch; do not choose a new run ID or cancel "
                    "by name."
                )
            message = str(exc)
            if exc.result.operator_remedy and exc.result.operator_remedy not in message:
                message = f"{message}\n{exc.result.operator_remedy}"
            raise SkyPilotSubmitError(message, transaction=exc.result) from exc
        transaction.controller = {
            "state": controller_state.value,
            "selected_context": selected_context,
        }
        launch_pair = transaction.launch_result
        result = launch_pair[0] if isinstance(launch_pair, tuple) else None
        job_id = transaction.job_id
        return WorkflowResult(
            # Preserve the public result contract; adoption is exposed through
            # launch_transaction.state and the human reconciliation message.
            status="SUBMITTED",
            job_id=job_id,
            log_paths={
                "submission_dir": str(submission_dir),
                "config": str(generated_config_path),
            },
            returncode=result.returncode if result is not None else 0,
            stdout=result.stdout if result is not None else "",
            stderr=result.stderr if result is not None else "",
            submitted_yaml_path=str(prepared_yaml),
            launch_transaction=transaction.to_dict(),
        )
    except SkyPilotSubmitError:
        _cleanup_owned_submission_dir(owned_submission_dir)
        raise
    except (
        OSError,
        ValueError,
        yaml.YAMLError,
        SkyPilotConfigError,
        SkyPilotNotInstalledError,
        SkyPilotVersionError,
    ) as exc:
        _cleanup_owned_submission_dir(owned_submission_dir)
        raise SkyPilotSubmitError(
            f"SkyPilot workflow submission failed: {exc}"
        ) from exc


def workflow_status(
    job_id: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND,
    timeout: int = 300,
) -> WorkflowResult:
    """Query a SkyPilot managed job status via `sky jobs queue`."""

    del controller_backend
    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [
        str(ensure_skypilot_version(runtime_config.sky_bin)),
        "jobs",
        "queue",
        "--all",
        "--output",
        "json",
    ]
    if runtime_config.global_config_path is not None:
        cmd[3:3] = ["--config", str(runtime_config.global_config_path)]
    result = subprocess.run(
        cmd,
        env=sky_environment(runtime_config.isolated_config_dir),
        cwd=_stable_sky_cwd(runtime_config.isolated_config_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if is_verified_empty_queue_result(result):
        return WorkflowResult(
            status="ABSENT",
            job_id=job_id,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    if result.returncode != 0:
        return WorkflowResult(
            status="UNKNOWN",
            job_id=job_id,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.stderr.strip() or result.stdout.strip(),
        )

    status = _status_from_queue_payload(result.stdout, job_id)
    return WorkflowResult(
        status=status or "UNKNOWN",
        job_id=job_id,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def workflow_task_statuses(
    job_id: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    timeout: int = 300,
) -> list[dict[str, Any]]:
    """Return per-task rows for a managed job (pipeline tasks or JobGroup members).

    Each row carries the timing fields SkyPilot records per task
    (``submitted_at`` / ``start_at`` / ``end_at``), which is how a JobGroup can be
    shown to have run its members *concurrently* and how a barrier state can be
    shown to have started only after its predecessors finished.
    """

    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [
        str(ensure_skypilot_version(runtime_config.sky_bin)),
        "jobs",
        "queue",
        "--all",
        "--output",
        "json",
    ]
    result = subprocess.run(
        cmd,
        env=sky_environment(runtime_config.isolated_config_dir),
        cwd=_stable_sky_cwd(runtime_config.isolated_config_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        return []
    return parse_task_statuses(result.stdout, job_id)


def workflow_controller_logs(
    job_id: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Fetch bounded managed-jobs controller evidence without following logs."""

    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [
        str(ensure_skypilot_version(runtime_config.sky_bin)),
        "jobs",
        "logs",
        "--controller",
        str(job_id),
        "--no-follow",
    ]
    return subprocess.run(
        cmd,
        env=sky_environment(runtime_config.isolated_config_dir),
        cwd=_stable_sky_cwd(runtime_config.isolated_config_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def find_job_ids_by_name(
    job_name: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    timeout: int = 300,
) -> list[str]:
    """Return the managed-job ids whose ``job_name`` matches, newest first.

    Used to verify (or recover) the job id parsed from ``sky jobs launch`` output.
    Trusting the parsed number alone is unsafe: a flaky API server can leave stale
    text in the stream, and polling the wrong id makes the driver abandon a job that
    is still running — observed live, with four GPUs left burning.
    """

    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    result = subprocess.run(
        [
            str(ensure_skypilot_version(runtime_config.sky_bin)),
            "jobs",
            "queue",
            "--all",
            "--output",
            "json",
        ],
        env=sky_environment(runtime_config.isolated_config_dir),
        cwd=_stable_sky_cwd(runtime_config.isolated_config_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        return []
    return parse_job_ids_by_name(result.stdout, job_name)


def lookup_managed_job(
    job_name: str,
    *,
    job_id: str = "",
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    timeout: int = 300,
) -> ManagedJobEvidence:
    """Authoritatively find one exact job name/id without fuzzy matching.

    A failed provider/auth/CLI request is ``unavailable``. Only a successful,
    parseable ``jobs queue --all --output json`` response can return ``absent``.
    """

    try:
        runtime_config = resolve_config(
            sky_bin=sky_bin,
            global_config_path=config_path,
            isolated_config_dir=isolated_config_dir,
        )
        cmd = [
            str(ensure_skypilot_version(runtime_config.sky_bin)),
            "jobs",
            "queue",
            "--all",
            "--output",
            "json",
        ]
        if runtime_config.global_config_path is not None:
            cmd[3:3] = ["--config", str(runtime_config.global_config_path)]
        result = subprocess.run(
            cmd,
            env=sky_environment(runtime_config.isolated_config_dir),
            cwd=_stable_sky_cwd(runtime_config.isolated_config_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - callers must distinguish unavailability
        return ManagedJobEvidence("unavailable", error=redact_text(str(exc)))
    if is_verified_empty_queue_result(result):
        jobs: list[dict[str, Any]] = []
    elif result.returncode != 0:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"exit {result.returncode}"
        )
        return ManagedJobEvidence("unavailable", error=redact_text(detail))
    else:
        parsed_jobs = queue_rows_from_output(result.stdout)
        if parsed_jobs is None:
            return ManagedJobEvidence(
                "unavailable",
                error="SkyPilot queue returned malformed, ambiguous, or schema-invalid JSON",
            )
        jobs = parsed_jobs

    wanted_id = str(job_id or "").strip()
    matching_ids: set[int] = set()
    declared_job_names: set[str] = set()
    for row in jobs:
        if not isinstance(row, dict):
            continue
        raw_id = str(row.get("job_id") or row.get("id") or "")
        row_name = str(row.get("job_name") or row.get("name") or "")
        if wanted_id:
            if raw_id != wanted_id:
                continue
            declared_name = str(row.get("job_name") or "")
            if declared_name:
                declared_job_names.add(declared_name)
        elif row_name != job_name:
            continue
        if raw_id.isdigit():
            matching_ids.add(int(raw_id))
    if wanted_id and declared_job_names and job_name not in declared_job_names:
        names = ", ".join(sorted(declared_job_names))
        return ManagedJobEvidence(
            "unavailable",
            error=(
                f"recorded SkyPilot job {wanted_id} belongs to {names!r}, "
                f"not exact run {job_name!r}"
            ),
        )
    if not matching_ids:
        return ManagedJobEvidence("absent")
    if not wanted_id and len(matching_ids) > 1:
        rendered = ", ".join(str(value) for value in sorted(matching_ids))
        return ManagedJobEvidence(
            "unavailable",
            error=(
                f"exact managed-job name {job_name!r} is ambiguous; matching IDs: "
                f"{rendered}. Supply durable run state with an immutable job ID."
            ),
        )
    selected = str(max(matching_ids))
    rows = tuple(parse_task_statuses(result.stdout, selected))
    return ManagedJobEvidence(
        "found",
        job_id=selected,
        status=_status_from_queue_payload(result.stdout, selected) or "UNKNOWN",
        task_rows=rows,
    )


def _reconcile_managed_job_env(
    job_name: str,
    *,
    env: Mapping[str, str],
    sky_executable: str,
    cwd: str | None,
    timeout: int = 60,
) -> ReconciliationEvidence:
    """Reconcile one exact name through the same SkyPilot runtime as launch."""

    try:
        result = subprocess.run(
            [sky_executable, "jobs", "queue", "--all", "--output", "json"],
            env=dict(env),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (KeyboardInterrupt, InterruptedError):
        raise
    except BaseException as exc:
        return ReconciliationEvidence(
            ReconciliationState.UNAVAILABLE, error=redact_text(str(exc))
        )
    if is_verified_empty_queue_result(result):
        rows: list[dict[str, Any]] = []
    elif result.returncode != 0:
        detail = _command_detail(result)
        state, _category = classify_failure(
            phase="reconciliation", stdout=result.stdout, stderr=result.stderr
        )
        return ReconciliationEvidence(
            ReconciliationState.UNAVAILABLE
            if state is not EvidenceState.AMBIGUOUS
            else ReconciliationState.AMBIGUOUS,
            error=redact_text(detail),
        )
    else:
        parsed_rows = queue_rows_from_output(result.stdout)
        if parsed_rows is None:
            return ReconciliationEvidence(
                ReconciliationState.AMBIGUOUS,
                error="SkyPilot queue returned malformed, ambiguous, or schema-invalid JSON",
            )
        rows = parsed_rows
    matching: set[str] = set()
    statuses: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("job_name") or row.get("name") or "") != job_name:
            continue
        job_id = str(row.get("job_id") or row.get("id") or "")
        if job_id.isdigit():
            matching.add(job_id)
            statuses[job_id] = str(row.get("status") or "UNKNOWN").upper()
    if not matching:
        return ReconciliationEvidence(ReconciliationState.ABSENT)
    if len(matching) != 1:
        return ReconciliationEvidence(
            ReconciliationState.AMBIGUOUS,
            error=(
                f"exact managed-job name {job_name!r} maps to multiple immutable IDs: "
                + ", ".join(sorted(matching, key=int))
            ),
        )
    selected = next(iter(matching))
    return ReconciliationEvidence(
        ReconciliationState.FOUND,
        job_id=selected,
        status=statuses.get(selected, "UNKNOWN"),
    )


def parse_job_ids_by_name(output: str, job_name: str) -> list[str]:
    """Extract managed-job ids for ``job_name`` from queue JSON, newest first."""

    jobs = queue_rows_from_output(output)
    if jobs is None:
        return []
    ids: list[int] = []
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        if str(job.get("job_name") or "") != job_name:
            continue
        raw = str(job.get("job_id") or job.get("id") or "")
        if raw.isdigit() and int(raw) not in ids:
            ids.append(int(raw))
    return [str(value) for value in sorted(ids, reverse=True)]


def parse_task_statuses(output: str, job_id: str) -> list[dict[str, Any]]:
    """Extract per-task rows for ``job_id`` from ``sky jobs queue --output json``."""

    jobs = queue_rows_from_output(output)
    if jobs is None:
        return []
    rows: list[dict[str, Any]] = []
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        current_id = str(job.get("job_id") or job.get("id") or "")
        if current_id != str(job_id):
            continue
        row = {
            "job_id": current_id,
            "task_id": job.get("task_id"),
            "task_name": job.get("task_name") or job.get("job_name") or "",
            "status": str(job.get("status") or "").upper(),
            "submitted_at": job.get("submitted_at"),
            "start_at": job.get("start_at"),
            "end_at": job.get("end_at"),
            "is_job_group": job.get("is_job_group"),
            "execution": job.get("execution"),
            "cluster_name": job.get("cluster_name_on_cloud")
            or job.get("current_cluster_name")
            or "",
        }
        # Preserve scheduler/recovery evidence without binding to one SkyPilot
        # release's field spelling.  The actionable status projector consumes
        # these normalized names and keeps the raw status separately.
        for target, aliases in {
            "retry_count": ("retry_count", "recovery_count", "num_restarts", "attempt"),
            "last_progress_at": ("last_progress_at", "last_transition_at"),
            "last_updated_at": ("last_updated_at", "updated_at"),
            "failure_reason": ("failure_reason", "failure_message", "error"),
        }.items():
            for alias in aliases:
                if job.get(alias) not in (None, ""):
                    row[target] = job[alias]
                    break
        rows.append(row)
    rows.sort(key=lambda row: (row.get("task_id") is None, row.get("task_id") or 0))
    return rows


class _LaunchStreamer:
    """Tee ``sky jobs launch`` output to the operator while it is still running.

    SkyPilot buffers nothing useful into its exit status: a controller stuck in a
    retry loop simply never returns. Capturing stdout/stderr to a pipe therefore
    leaves the operator staring at a blank terminal for the full submit timeout.
    Writing to files and tailing them keeps ``subprocess.run`` (so callers and
    tests that stub it are unaffected) while still showing progress live.
    """

    _POLL_SECONDS = 0.25

    def __init__(
        self,
        echo: Callable[[str], None],
        *,
        optional_nebius_profile: bool = False,
    ) -> None:
        self._echo = echo
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._paths: list[Path] = []
        self._offsets: dict[Path, int] = {}
        self._pending: dict[Path, str] = {}
        self.diagnoses: list[SkyPilotDiagnosis] = []
        self._optional_nebius_profile = optional_nebius_profile
        self._profile_notice_emitted = False

    def watch(self, paths: Sequence[Path]) -> None:
        self._paths = list(paths)
        self._offsets = {path: 0 for path in self._paths}
        self._pending = {path: "" for path in self._paths}
        self._thread = threading.Thread(
            target=self._run, name="sky-launch-stream", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._drain()
        for path, pending in self._pending.items():
            if pending.strip():
                self._emit(pending)
            self._pending[path] = ""

    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain()
            self._stop.wait(self._POLL_SECONDS)

    def _drain(self) -> None:
        for path in self._paths:
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(self._offsets[path])
                    chunk = handle.read()
                    self._offsets[path] = handle.tell()
            except OSError:
                continue
            if not chunk:
                continue
            buffered = self._pending[path] + chunk
            *lines, self._pending[path] = buffered.split("\n")
            for line in lines:
                self._emit(line)

    def _emit(self, line: str) -> None:
        text = line.rstrip()
        if self._optional_nebius_profile and "Unable to create Nebius profile" in text:
            if not self._profile_notice_emitted:
                self._echo(
                    "npa: optional SkyPilot Nebius provider-profile creation was skipped; "
                    "the active Kubernetes-controller/context execution path remains selected."
                )
                self._profile_notice_emitted = True
            return
        if text:
            self._echo(text)
        diagnosis = diagnose_skypilot_output(text)
        if diagnosis is not None and all(
            existing.code != diagnosis.code for existing in self.diagnoses
        ):
            self.diagnoses.append(diagnosis)
            self._echo(f"npa: detected {diagnosis.code}. {diagnosis.render()}")


def _nebius_profile_is_optional(
    documents: Sequence[Mapping[str, Any]],
    *,
    controller_backend: ControllerBackend,
    infra: str,
) -> bool:
    """Whether a Nebius provider profile is outside the selected execution path.

    A Kubernetes jobs controller alone is not proof: it can orchestrate tasks on
    the Nebius VM provider.  The profile is optional only when ``--infra`` pins a
    Kubernetes context, or every explicit task cloud is Kubernetes.  Missing or
    mixed cloud declarations fail closed so a real provider-auth failure remains
    visible and fatal.
    """

    if controller_backend != "kubernetes":
        return False
    selected = str(infra or "").strip().lower()
    if selected.startswith(("k8s/", "kubernetes/")):
        return True
    clouds: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() == "cloud" and isinstance(item, str):
                    clouds.append(item.strip().lower())
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    for document in documents:
        collect(document.get("resources"))
    return bool(clouds) and all(cloud in {"kubernetes", "k8s"} for cloud in clouds)


def _default_launch_echo(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _run_launch(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: str | None,
    timeout: int,
    log_dir: Path,
    streamer: _LaunchStreamer | None,
) -> tuple[subprocess.CompletedProcess[str], list[SkyPilotDiagnosis]]:
    """Run ``sky jobs launch``, streaming output when a streamer is supplied."""

    if streamer is None:
        result = subprocess.run(
            list(cmd),
            env=dict(env),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return result, []

    out_path = log_dir / "sky-launch.stdout.log"
    err_path = log_dir / "sky-launch.stderr.log"
    with (
        out_path.open("w", encoding="utf-8") as out_handle,
        err_path.open("w", encoding="utf-8") as err_handle,
    ):
        streamer.watch([out_path, err_path])
        try:
            result = subprocess.run(
                list(cmd),
                env=dict(env),
                cwd=cwd,
                text=True,
                stdout=out_handle,
                stderr=err_handle,
                timeout=timeout,
                check=False,
            )
        finally:
            streamer.stop()
    # A stubbed subprocess.run returns captured text directly and never touches
    # the files; prefer whichever side is actually populated.
    stdout = result.stdout or _read_text(out_path)
    stderr = result.stderr or _read_text(err_path)
    return (
        subprocess.CompletedProcess(list(cmd), result.returncode, stdout, stderr),
        list(streamer.diagnoses),
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _wait_for_healthy_jobs_controller(
    sky_executable: str,
    *,
    env: dict[str, str],
    timeout: int,
    interval: float,
    require_existing: bool = False,
    cwd: str | None = None,
) -> ControllerState:
    """Block launch while an existing managed-jobs controller is not ready."""

    deadline = time.monotonic() + max(timeout, 0)
    last_summary = "no jobs-controller found" if require_existing else ""
    unhealthy: list[tuple[str, str]] = []
    while True:
        result = subprocess.run(
            [sky_executable, "status", "--refresh", "--output", "json"],
            env=env,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=min(max(timeout, 1), 300),
            check=False,
        )
        if result.returncode != 0:
            detail = _command_detail(result)
            raise SkyPilotSubmitError(
                f"SkyPilot controller health check failed: {detail}"
                + _controller_health_remedy(detail)
            )
        controllers = _jobs_controller_statuses(result.stdout)
        if require_existing and not controllers:
            last_summary = "no jobs-controller found"
            unhealthy = []
        else:
            unhealthy = [
                (name, status)
                for name, status in controllers
                if status.upper() not in READY_CONTROLLER_STATUSES
            ]
            if not unhealthy:
                if not controllers:
                    return ControllerState.ABSENT
                statuses = {status.upper() for _name, status in controllers}
                return (
                    ControllerState.UP
                    if HEALTHY_CONTROLLER_STATUS in statuses
                    else ControllerState.STOPPED
                )
            last_summary = ", ".join(
                f"{name}={status or 'UNKNOWN'}" for name, status in unhealthy
            )
        if time.monotonic() >= deadline:
            # Name the controller that is actually unhealthy: pointing `sky down`
            # at the first of several cached controllers (or at a placeholder when
            # none exists) tears down the wrong thing.
            stuck = [name for name, _status in unhealthy]
            remedy = (
                f" If it is stuck/stale, tear it down with `sky down {stuck[0]}` "
                "(it is recreated on the next launch)."
                if stuck
                else " A launch creates one, so retry the submit; if it keeps failing, "
                "check `sky check` and your kube context."
            )
            raise SkyPilotSubmitError(
                f"SkyPilot jobs controller not healthy before launch: {last_summary}.{remedy}"
            )
        time.sleep(max(interval, 0.1))


# Signals that `sky status` failed because a *cached* controller still points at
# infrastructure that is gone (a kubeconfig from another setup, a deleted
# cluster, a renamed context) rather than because of a transient error.
_STALE_CONTROLLER_SIGNALS = (
    "kubeconfig",
    "kube-config",
    "not found in kubeconfig",
    "invalid kube-context",
    "kubernetes context",
    "credentials not found",
    "cachedclusterunavailable",
    "clusterstatusfetchingerror",
)
# Transport/filesystem errors that also describe unrelated failures — an API
# server that is not running, a missing ~/.sky config. They only point at a stale
# controller when the same message also names a controller or a kube context;
# alone they would send the operator to `sky down` for something else entirely.
_AMBIGUOUS_STALE_SIGNALS = (
    "no such file or directory",
    "unable to connect",
    "connection refused",
)
_CONTROLLER_CONTEXT_SIGNALS = (
    "kubeconfig",
    "kube-config",
    "kube-context",
    "kubernetes",
    "sky-jobs-controller",
)
_KUBECONFIG_PATH_RE = re.compile(r"[\w./~-]*kubeconfig[\w./-]*")


def _referenced_kubeconfig_path(detail: str) -> str:
    """Return the kubeconfig *path* named in *detail*, or "".

    SkyPilot's message mentions the bare word "kubeconfig" several times before
    the actual file, so pick the longest match that looks like a path.
    """
    candidates = [
        match
        for match in _KUBECONFIG_PATH_RE.findall(str(detail or ""))
        if "/" in match
    ]
    return max(candidates, key=len) if candidates else ""


def _looks_like_stale_controller(lowered: str) -> bool:
    if any(signal in lowered for signal in _STALE_CONTROLLER_SIGNALS):
        return True
    return any(signal in lowered for signal in _AMBIGUOUS_STALE_SIGNALS) and any(
        signal in lowered for signal in _CONTROLLER_CONTEXT_SIGNALS
    )


def _controller_health_remedy(detail: str) -> str:
    """Return actionable remediation text for a failed controller health check.

    A stale managed-jobs controller cached from an unrelated NPA setup (pointing
    at a kubeconfig that no longer exists) surfaced only as a raw `sky status`
    stack trace, with nothing telling the operator that the fix is to purge the
    controller or repoint KUBECONFIG.
    """

    lowered = str(detail or "").lower()
    if not _looks_like_stale_controller(lowered):
        return ""
    lines = [
        "",
        "",
        "This usually means a cached SkyPilot managed-jobs controller "
        "(sky-jobs-controller-*) still points at infrastructure from another "
        "setup. To recover:",
    ]
    referenced = _referenced_kubeconfig_path(detail)
    if referenced:
        lines.append(
            f"  - the referenced kubeconfig is {referenced}; provision or "
            "restore that cluster (`npa cluster up` / `npa provision-if-absent "
            "--project <alias>`), or point KUBECONFIG at the cluster you want."
        )
    else:
        lines.append(
            "  - point KUBECONFIG at the cluster you want, or provision one "
            "with `npa cluster up` / `npa provision-if-absent --project <alias>`."
        )
    lines.extend(
        [
            "  - list what SkyPilot has cached: `sky status -r` (refresh) or "
            "`sky status` (SkyPilot 0.12 has no `--all` for `sky status`).",
            "  - purge the stale controller: `sky down sky-jobs-controller-<id>` "
            "(it is recreated on the next launch).",
            "  - if `sky status -r` still reports nothing useful, the API server is "
            "holding the stale state: `sky api stop && sky api start`, then retry.",
            "  - target a specific context explicitly with "
            "`npa workbench workflow submit ... --infra k8s/<context>`.",
        ]
    )
    return "\n".join(lines)


def _jobs_controller_statuses(output: str) -> list[tuple[str, str]]:
    payload = _json_payload_from_output(output)
    if payload is None:
        raise SkyPilotSubmitError(
            "SkyPilot controller health check returned non-json output"
        )
    raw_clusters: Any
    if isinstance(payload, list):
        raw_clusters = payload
    elif isinstance(payload, dict):
        raw_clusters = payload.get("clusters", payload.get("jobs", []))
    else:
        raw_clusters = []
    clusters = raw_clusters if isinstance(raw_clusters, list) else []
    controllers = []
    for cluster in clusters or []:
        if not isinstance(cluster, dict):
            continue
        name = str(cluster.get("name") or cluster.get("cluster") or "")
        if not name.startswith(JOBS_CONTROLLER_PREFIX):
            continue
        status = str(cluster.get("status") or cluster.get("cluster_status") or "")
        controllers.append((name, status.upper()))
    return controllers


def _json_payload_from_output(output: str) -> Any | None:
    return parse_single_json_document(output)


def _load_yaml_documents(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        docs = [doc for doc in yaml.safe_load_all(handle) if doc is not None]
    if not all(isinstance(doc, dict) for doc in docs):
        raise ValueError("SkyPilot YAML documents must be mappings")
    return docs


def _load_base_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"SkyPilot global config must be a mapping: {config_path}")
    data.pop("name", None)
    return data


def _chmod_owner_only(path: Path, *, is_dir: bool = False) -> None:
    """Best-effort ``chmod`` to owner-only (0700 dir / 0600 file).

    The submission dir holds the rendered task YAML and generated SkyPilot config,
    which can carry registry/docker auth and S3 creds. ``write_text`` /
    ``mkdir`` honor the umask (commonly world-readable), so tighten explicitly.
    """
    try:
        path.chmod(0o700 if is_dir else 0o600)
    except OSError:  # pragma: no cover - unusual filesystems (e.g. mounted FAT)
        pass


def _submission_dir(run_id: str, isolated_config_dir: Path | None) -> Path:
    if isolated_config_dir is None:
        # Successful submissions return this path for debugging; exception paths
        # remove it via _cleanup_owned_submission_dir.
        root = Path(tempfile.mkdtemp(prefix=f"npa-skypilot-{run_id}-"))
    else:
        root = Path(isolated_config_dir) / "submissions" / run_id
        root.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    _chmod_owner_only(root, is_dir=True)
    return root


def _verified_job_id(
    parsed: str,
    job_name: str,
    *,
    env: dict[str, str],
    sky_executable: str,
    cwd: str | None,
) -> str:
    """Cross-check the scraped job id against the job NAME we just launched.

    Costs one extra `sky jobs queue` call per submit. That is deliberate — polling the
    wrong job is how a running GPU job gets abandoned — but it is bounded (short
    timeout) and can be disabled with ``NPA_SKYPILOT_VERIFY_JOB_ID=0`` for callers that
    submit in a tight loop and accept the risk.

    ``sky jobs launch`` streams from the API server, and a flaky/restarting server can
    leave a previous request's ``Job submitted, ID: N`` in the output. Callers then poll
    somebody else's job: observed live twice — a runtime wave declared CANCELLED while
    its real job kept four GPUs busy, and a live e2e case reported FAILED by reading the
    *previous* spec's job. The name is authoritative; the parsed id is only trusted when
    the queue agrees.
    """

    import os as _os

    if str(_os.environ.get("NPA_SKYPILOT_VERIFY_JOB_ID", "1")).strip().lower() in {
        "0",
        "false",
        "no",
    }:
        return parsed
    try:
        result = subprocess.run(
            [sky_executable, "jobs", "queue", "--all", "--output", "json"],
            env=env,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            return parsed
        ids = parse_job_ids_by_name(result.stdout, job_name)
    except Exception:  # noqa: BLE001 - never fail a successful submit over a lookup
        return parsed
    if not ids:
        return parsed
    if parsed and parsed in ids:
        return parsed
    return ids[0]


def _parse_job_id(output: str) -> str:
    for pattern in (r"Job submitted,\s*ID:\s*([0-9]+)", r"Managed Job ID:\s*([0-9]+)"):
        match = re.search(pattern, output)
        if match:
            return match.group(1)
    return ""


def _cleanup_owned_submission_dir(path: Path | None) -> None:
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)


def _format_submit_error(
    cmd: Sequence[str],
    result: subprocess.CompletedProcess[str],
    *,
    streamed: Sequence[SkyPilotDiagnosis] = (),
) -> str:
    detail = _command_detail(result)
    prefix = (
        "SkyPilot auth failure during jobs launch"
        if _looks_like_auth_error(detail)
        else "sky jobs launch failed"
    )
    # `sky status --refresh` exits 0 while merely *warning* about clusters it
    # cannot refresh, so a controller cached against a dead kubeconfig gets past
    # the health check and fails here instead (CachedClusterUnavailable). Attach
    # the same NPA-level recovery steps.
    pod_config_remedy = _pod_config_error_remedy(detail)
    message = (
        f"{prefix}: {' '.join(cmd)}: {detail}"
        + pod_config_remedy
        + _controller_health_remedy(detail)
    )
    diagnoses = list(streamed)
    diagnosis = diagnose_skypilot_output(
        f"{result.stdout or ''}\n{result.stderr or ''}"
    )
    if diagnosis is not None and all(item.code != diagnosis.code for item in diagnoses):
        diagnoses.append(diagnosis)
    for item in diagnoses:
        # The pod_config case already has a dedicated remedy above; do not say it twice.
        if item.code == "kubernetes_client_pod_config" and pod_config_remedy:
            continue
        message = f"{message}\n{item.render()}"
    return message


def _looks_like_pod_config_error(detail: str) -> bool:
    """Detect the SkyPilot/kubernetes-client pod_config type-resolution failure.

    A too-new ``kubernetes`` client makes SkyPilot 0.12.x fail to render the jobs
    controller pod with ``Invalid pod_config … No module named
    'kubernetes.client.models.dict[str, str]'``. The controller retries this
    *forever* (it is neither a quota nor a capacity problem), so a submit hangs
    with almost no feedback.
    """
    normalized = detail.lower()
    if "invalid pod_config" in normalized:
        return True
    if "kubernetes.client.models" in normalized:
        return True
    return "no module named" in normalized and "kubernetes" in normalized


def _pod_config_error_remedy(detail: str) -> str:
    if not _looks_like_pod_config_error(detail):
        return ""
    return (
        "\n\nThis is a SkyPilot/kubernetes-client incompatibility (the pod_config "
        "type resolver fails with 'No module named kubernetes.client.models…'), not "
        "a cluster/quota problem — the jobs controller retries it indefinitely, so a "
        "submit appears to hang. Rebuild the isolated SkyPilot runtime so its "
        "kubernetes client matches SkyPilot: `npa skypilot uninstall && npa skypilot "
        "bootstrap` (or `pip install 'kubernetes<31'` into ~/.npa/skypilot-venv), then "
        "re-submit."
    )


def _command_detail(result: subprocess.CompletedProcess[str]) -> str:
    return result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"


def _looks_like_auth_error(detail: str) -> bool:
    normalized = detail.lower()
    return any(
        token in normalized
        for token in (
            "auth",
            "credential",
            "unauthorized",
            "forbidden",
            "permission denied",
            "401",
            "403",
        )
    )


def _status_from_queue_payload(output: str, job_id: str) -> str:
    jobs = queue_rows_from_output(output)
    if jobs is None:
        return ""
    statuses = []
    for job in jobs or []:
        current_id = str(job.get("job_id") or job.get("id") or "")
        if current_id == str(job_id):
            status = str(job.get("status", "")).upper()
            if status:
                statuses.append(status)
    if not statuses:
        return ""
    for status in statuses:
        if status.startswith("FAILED") or status == "CANCELLED":
            return status
    if all(status == "SUCCEEDED" for status in statuses):
        return "SUCCEEDED"
    for status in ("RUNNING", "RECOVERING", "STARTING", "PENDING", "CANCELLING"):
        if status in statuses:
            return status
    return statuses[0]
