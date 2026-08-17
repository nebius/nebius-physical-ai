"""Explicit cleanup helpers for NPA SkyPilot workflows."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from npa.orchestration.skypilot._bin import (
    SkyBin,
    ensure_skypilot_version,
    resolve_config,
)
from npa.orchestration.skypilot.controller import (
    DEFAULT_CONTROLLER_BACKEND,
    ControllerBackend,
)
from npa.orchestration.skypilot.json_output import (
    is_verified_empty_queue_result,
    verified_structured_queue_rows,
)
from npa.orchestration.skypilot.workflow_state import redact_text


@dataclass
class CleanupResult:
    """Result of an explicit SkyPilot cleanup operation."""

    resources_removed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    outcome: str = "cleaned"
    identity_source: str = "live_configuration"
    receipt_id: str = ""
    verified: bool = False
    no_op: bool = False
    project_alias: str = ""
    project_id: str = ""
    cluster_id: str = ""
    context: str = ""
    remote_absence_verified: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: "CleanupResult") -> None:
        self.resources_removed.extend(other.resources_removed)
        self.errors.extend(other.errors)
        self.commands.extend(other.commands)


NONTERMINAL_JOB_STATUSES = {
    "PENDING",
    "STARTING",
    "RUNNING",
    "RECOVERING",
    "CANCELLING",
}
JOBS_CONTROLLER_PATTERN = "sky-jobs-controller-*"
RUN_ID_MIN_LENGTH = 12
_RUN_ID_ALLOWED_RE = re.compile(r"^[A-Za-z0-9-]+$")

# `sky jobs cancel` only *schedules* cancellation: the controller still reports the
# job as CANCELLING for a while afterwards, and `sky down` on the controller refuses
# to run while any managed job is non-terminal. Cancelling and immediately tearing
# down therefore fails with the error below even though the operator did exactly
# what the message asks for. Wait for the cancellation to land instead.
DEFAULT_JOB_DRAIN_TIMEOUT_SECONDS = 300
DEFAULT_JOB_DRAIN_INTERVAL_SECONDS = 5.0
_IN_PROGRESS_JOBS_MARKERS = ("in-progress managed jobs", "in progress managed jobs")


class InvalidRunIdError(ValueError):
    """Raised when a run id is unsafe for SkyPilot cleanup matching."""


class JobQueueUnreadableError(RuntimeError):
    """Raised when managed-job absence cannot be authoritatively verified."""


@dataclass(frozen=True)
class JobQueueSnapshot:
    """Verified or unreadable state returned by the shared queue inventory."""

    state: str
    jobs: tuple[dict[str, Any], ...] = ()
    detail: str = ""

    @property
    def readable(self) -> bool:
        return self.state in {"verified_empty", "verified_jobs"}


def sky_down(
    cluster_name: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    timeout: int = 900,
) -> CleanupResult:
    """Run `sky down --yes` for a cluster or SkyPilot glob pattern."""

    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [
        str(ensure_skypilot_version(runtime_config.sky_bin)),
        "down",
        "--yes",
        cluster_name,
    ]
    result = _run(
        cmd,
        isolated_config_dir=runtime_config.isolated_config_dir,
        config_path=runtime_config.global_config_path,
        timeout=timeout,
    )
    cleanup = CleanupResult(commands=[cmd])
    if result.returncode == 0:
        cleanup.resources_removed.append(cluster_name)
    elif _is_conclusive_absence_error(_command_detail(result)):
        cleanup.resources_removed.append(f"{cluster_name}:already-absent")
    else:
        cleanup.errors.append(_format_command_error(cmd, result))
    return cleanup


def cleanup_jobs_controller(
    *,
    project: str = "",
    context: str = "",
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    job_drain_timeout: int = DEFAULT_JOB_DRAIN_TIMEOUT_SECONDS,
    receipt: str = "",
    project_id: str = "",
    cluster_id: str = "",
    cluster_name: str = "",
) -> CleanupResult:
    """Transactionally remove the controller for one verified NPA cluster.

    Identity comes only from explicit project/context arguments or the selected
    NPA project and its exact saved cluster record.  Remote deletion runs against
    a clone of SkyPilot state; the real local metadata is changed only after an
    independent Kubernetes absence check succeeds.
    """

    cleanup = CleanupResult(
        receipt_id=receipt,
        identity_source=(
            f"receipt:{receipt}"
            if receipt
            else "explicit_exact_arguments"
            if project_id or cluster_id or cluster_name
            else "live_configuration"
        ),
    )
    from npa.cluster.identity import (
        ClusterIdentityError,
        resolve_verified_cluster_identity,
    )
    from npa.teardown_receipts import record_teardown_event

    if receipt:
        from npa.cleanup_identity import CleanupIdentityError, resolve_cleanup_identity
        from npa.clients.config import resolve_environment

        try:
            configured = resolve_environment(project) if project else None
        except (OSError, RuntimeError, ValueError) as exc:
            cleanup.errors.append(
                f"live controller configuration is unreadable; no mutation was attempted: {exc}"
            )
            cleanup.outcome = "unsafe"
            return cleanup
        local_cluster = None
        if context:
            try:
                from npa.cluster.state import load_cluster_state

                local_cluster = load_cluster_state(context)
            except (OSError, RuntimeError, ValueError):
                local_cluster = None
        live_identity = {
            "project_alias": project,
            "project_id": str(getattr(configured, "project_id", "") or ""),
            "tenant_id": str(getattr(configured, "tenant_id", "") or ""),
            "region": str(getattr(configured, "region", "") or ""),
            "context": context,
            "cluster_id": str(getattr(local_cluster, "cluster_id", "") or ""),
            "cluster_name": str(getattr(local_cluster, "name", "") or ""),
            "kubeconfig_path": str(getattr(local_cluster, "kubeconfig_path", "") or ""),
        }

        try:
            receipt_identity = resolve_cleanup_identity(
                explicit={
                    "project_alias": project,
                    "project_id": project_id,
                    "context": context,
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                },
                receipt_id=receipt,
                live=live_identity,
                phase="controller",
                resource=context,
            )
        except CleanupIdentityError as exc:
            cleanup.errors.append(str(exc))
            cleanup.outcome = "unsafe"
            return cleanup
        cleanup.identity_source = receipt_identity.source
        cleanup.project_alias = str(receipt_identity.get("project_alias") or "")
        cleanup.project_id = str(receipt_identity.get("project_id") or "")
        cleanup.cluster_id = str(receipt_identity.get("cluster_id") or "")
        cleanup.context = str(
            receipt_identity.get("context")
            or receipt_identity.get("controller_context")
            or ""
        )
        if receipt_identity.receipt_is_terminal:
            cleanup.outcome = "already_absent"
            cleanup.verified = True
            cleanup.no_op = True
            return cleanup
        try:
            cluster_receipt_identity = resolve_cleanup_identity(
                explicit={
                    "project_alias": project,
                    "project_id": project_id,
                    "context": context,
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                },
                receipt_id=receipt,
                live=live_identity,
                phase="cluster",
                resource=context,
            )
        except CleanupIdentityError as exc:
            cleanup.errors.append(str(exc))
            cleanup.outcome = "unsafe"
            return cleanup
        if cluster_receipt_identity.receipt_is_terminal:
            cleanup.project_alias = str(
                cluster_receipt_identity.get("project_alias") or cleanup.project_alias
            )
            cleanup.project_id = str(
                cluster_receipt_identity.get("project_id") or cleanup.project_id
            )
            cleanup.cluster_id = str(
                cluster_receipt_identity.get("cluster_id") or cleanup.cluster_id
            )
            cleanup.context = str(
                cluster_receipt_identity.get("context") or cleanup.context
            )
            cleanup.outcome = "already_absent"
            cleanup.verified = True
            cleanup.no_op = True
            return cleanup

    try:
        identity = resolve_verified_cluster_identity(
            project=project,
            context=context,
            receipt=receipt,
            project_id=project_id,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
        )
    except ClusterIdentityError as exc:
        cleanup.errors.append(str(exc))
        try:
            record_teardown_event(
                phase="controller",
                resource=context or "unresolved-controller",
                terminal_state="verification_failed",
                project_alias=project,
                context=context,
                precheck={"identity_verified": False},
                action={"kind": "none"},
                verification={"remote_state": "not_inspected"},
                errors=[str(exc)],
            )
        except (OSError, RuntimeError, ValueError):
            pass
        return cleanup

    identity_fields = identity.receipt_identity()
    cleanup.project_alias = str(identity.project_alias or "")
    cleanup.project_id = str(identity.project_id or "")
    cleanup.cluster_id = str(identity.cluster_id or "")
    cleanup.context = str(identity.context or "")
    try:
        record_teardown_event(
            phase="controller",
            resource=identity.context,
            terminal_state="in_progress",
            project_alias=identity.project_alias,
            project_id=identity.project_id,
            context=identity.context,
            identity=identity.receipt_identity(),
            precheck={"identity_verified": True, **identity_fields},
            action={"kind": "inspect_remote_controller"},
            verification={"remote_state": "pending"},
        )
    except (OSError, RuntimeError, ValueError) as exc:
        cleanup.errors.append(
            f"controller transaction receipt could not be started; no remote or "
            f"local mutation was attempted: {exc}"
        )
        return cleanup

    if identity.cluster_absent:
        cleanup.outcome = "already_absent"
        cleanup.verified = True
        cleanup.no_op = True
        _record_controller_result(identity, cleanup, "verified_absent")
        return cleanup

    remote_pods: list[tuple[str, str, str]] = []
    if not identity.cluster_absent:
        remote_pods, remote_error = _kubernetes_controller_pods(
            kubeconfig=identity.kubeconfig,
            context=identity.context,
        )
        if remote_error:
            cleanup.errors.append(
                "exact-context controller inspection failed; local SkyPilot state "
                f"was preserved: {remote_error}"
            )
            _record_controller_result(identity, cleanup, "verification_failed")
            return cleanup

    controller_clusters, status_error = _jobs_controller_clusters(
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
        refresh=False,
        env_extra={"KUBECONFIG": str(identity.kubeconfig)},
    )
    if status_error:
        cleanup.errors.append(status_error)
        _record_controller_result(identity, cleanup, "verification_failed")
        return cleanup
    remote_names = {item[2] for item in remote_pods if item[2]}
    context_clusters = [
        item
        for item in controller_clusters
        if _controller_belongs_to_context(item, identity.context)
        or _cluster_name(item) in remote_names
    ]
    if remote_names and not context_clusters:
        cleanup.errors.append(
            "The verified context contains controller pod(s) "
            + ", ".join(sorted(remote_names))
            + ", but SkyPilot has no matching exact-context metadata. Local state "
            "was preserved. Restore the selected project's SkyPilot state and retry."
        )
        _record_controller_result(identity, cleanup, "verification_failed")
        return cleanup
    controller_clusters = context_clusters
    # Unrelated controller rows are deliberately ignored; they are neither
    # targets nor cleanup results for this exact project/context transaction.
    if controller_clusters:
        try:
            pending = _nonterminal_job_ids(
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                sky_bin=sky_bin,
            )
            if pending:
                wait_for_jobs_terminal(
                    pending,
                    isolated_config_dir=isolated_config_dir,
                    config_path=config_path,
                    sky_bin=sky_bin,
                    timeout=job_drain_timeout,
                )
        except JobQueueUnreadableError as exc:
            cleanup.errors.append(
                f"managed-job queue is unreadable/unverified; controller and "
                f"local SkyPilot state were preserved: {exc}"
            )
            _record_controller_result(identity, cleanup, "verification_failed")
            return cleanup
    if not remote_pods:
        # The exact context (or the provider itself) proves remote absence.  It is
        # now safe to converge matching local metadata.
        if not _record_remote_controller_absence(identity, cleanup):
            return cleanup
        for controller_cluster in controller_clusters:
            cleanup.extend(
                _down_jobs_controller(
                    _cluster_name(controller_cluster),
                    isolated_config_dir=isolated_config_dir,
                    config_path=config_path,
                    sky_bin=sky_bin,
                    job_drain_timeout=job_drain_timeout,
                    env_extra={"KUBECONFIG": str(identity.kubeconfig)},
                )
            )
        _verify_local_controller_metadata_removed(
            identity,
            controller_clusters,
            cleanup,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
        )
        cleanup.verified = cleanup.ok
        _record_controller_result(
            identity,
            cleanup,
            "verified_absent" if cleanup.ok else "verification_failed",
            remote_pods=[],
        )
        return cleanup

    with _cloned_skypilot_state(isolated_config_dir) as remote_state:
        for controller_cluster in controller_clusters:
            controller_name = _cluster_name(controller_cluster)
            remote_result = _down_jobs_controller(
                controller_name,
                isolated_config_dir=remote_state,
                config_path=config_path,
                sky_bin=sky_bin,
                job_drain_timeout=job_drain_timeout,
                env_extra={"KUBECONFIG": str(identity.kubeconfig)},
            )
            cleanup.commands.extend(remote_result.commands)
            if remote_result.errors:
                cleanup.errors.extend(remote_result.errors)
                _record_controller_result(
                    identity, cleanup, "verification_failed", remote_pods=remote_pods
                )
                return cleanup

    remaining, verify_error = _wait_for_controller_pods_absent(
        remote_names,
        kubeconfig=identity.kubeconfig,
        context=identity.context,
    )
    if verify_error or remaining:
        detail = verify_error or ", ".join(
            f"{namespace}/{pod}" for namespace, pod, _name in remaining
        )
        cleanup.errors.append(
            "remote controller absence was not proven; real local SkyPilot state "
            f"was preserved: {detail}"
        )
        _record_controller_result(
            identity, cleanup, "verification_failed", remote_pods=remaining
        )
        return cleanup

    # Only this post-verification call is allowed to mutate the real SkyPilot
    # cache/metadata.  Its remote target is already authoritatively absent.
    if not _record_remote_controller_absence(identity, cleanup):
        return cleanup
    for controller_cluster in controller_clusters:
        cleanup.extend(
            _down_jobs_controller(
                _cluster_name(controller_cluster),
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                sky_bin=sky_bin,
                job_drain_timeout=job_drain_timeout,
                env_extra={"KUBECONFIG": str(identity.kubeconfig)},
            )
        )
    _verify_local_controller_metadata_removed(
        identity,
        controller_clusters,
        cleanup,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )
    cleanup.verified = cleanup.ok
    _record_controller_result(
        identity,
        cleanup,
        "verified_deleted" if cleanup.ok else "verification_failed",
        remote_pods=[],
    )
    return cleanup


def _record_remote_controller_absence(identity: Any, cleanup: CleanupResult) -> bool:
    """Durably checkpoint remote absence before real local state may change."""

    from npa.teardown_receipts import record_teardown_event

    try:
        record_teardown_event(
            phase="controller",
            resource=identity.context,
            terminal_state="remote_absent_local_pending",
            project_alias=identity.project_alias,
            project_id=identity.project_id,
            context=identity.context,
            precheck={"identity_verified": True, **identity.receipt_identity()},
            action={"kind": "checkpoint_before_local_state_removal"},
            verification={"remote_controller_pods": [], "remote_absence": True},
        )
    except (OSError, RuntimeError, ValueError) as exc:
        cleanup.errors.append(
            "remote controller absence was proven, but its durable checkpoint "
            f"failed; real local SkyPilot state was preserved: {exc}"
        )
        return False
    cleanup.remote_absence_verified = True
    return True


def _verify_local_controller_metadata_removed(
    identity: Any,
    targets: Sequence[dict[str, Any]],
    cleanup: CleanupResult,
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> None:
    """Verify that the post-remote local convergence removed only target rows."""

    target_names = {_cluster_name(item) for item in targets}
    if not target_names or cleanup.errors:
        return
    rows, error = _jobs_controller_clusters(
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
        refresh=False,
        env_extra={"KUBECONFIG": str(identity.kubeconfig)},
    )
    if error:
        cleanup.errors.append(
            "remote absence was proven, but local controller metadata could not be "
            f"verified after convergence: {error}"
        )
        return
    remaining = sorted(target_names & {_cluster_name(item) for item in rows})
    if remaining:
        cleanup.errors.append(
            "remote absence was proven, but local SkyPilot metadata still lists: "
            + ", ".join(remaining)
            + ". Retry the same exact project/context transaction."
        )


def _record_controller_result(
    identity: Any,
    cleanup: CleanupResult,
    state: str,
    *,
    remote_pods: Sequence[tuple[str, str, str]] = (),
) -> None:
    from npa.teardown_receipts import record_teardown_event

    try:
        record_teardown_event(
            phase="controller",
            resource=identity.context,
            terminal_state=state,
            project_alias=identity.project_alias,
            project_id=identity.project_id,
            context=identity.context,
            identity=identity.receipt_identity(),
            precheck={"identity_verified": True, **identity.receipt_identity()},
            action={"commands": cleanup.commands},
            verification={
                "remote_controller_pods": [
                    f"{namespace}/{pod}" for namespace, pod, _name in remote_pods
                ],
                "local_state_removed_after_remote_absence": (
                    state in {"verified_absent", "verified_deleted"}
                ),
            },
            errors=cleanup.errors,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        cleanup.errors.append(
            f"controller teardown receipt could not be written: {exc}"
        )


def cleanup_workflow(
    cluster_or_job_id: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
) -> CleanupResult:
    """Cancel a managed job ID or tear down a SkyPilot cluster name."""

    if cluster_or_job_id.isdigit():
        return _cancel_job(
            cluster_or_job_id,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
        )
    return sky_down(
        cluster_or_job_id,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )


def cleanup_launched_workflow(
    job_id: str,
    run_id: str,
    *,
    cluster: str = "",
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    job_drain_timeout: int = DEFAULT_JOB_DRAIN_TIMEOUT_SECONDS,
    job_name: str = "",
    teardown_cluster: bool = True,
) -> CleanupResult:
    """Cancel one manifest-proven managed job and best-effort tear down its cluster.

    The shared jobs controller is deliberately never removed here.  This helper
    uses the same pinned NPA SkyPilot resolver for dry-run/status/cancel paths and
    continues safe cluster cleanup even when cancellation or draining reports an
    error.
    """

    cleaned_job_id = str(job_id or "").strip()
    cleaned_run_id = str(run_id or "").strip()
    if not cleaned_job_id:
        raise ValueError("A manifest-proven SkyPilot managed-job ID is required.")
    if not cleaned_run_id or not _RUN_ID_ALLOWED_RE.fullmatch(cleaned_run_id):
        raise InvalidRunIdError(
            "run id must contain only letters, numbers, and hyphens for exact cleanup"
        )
    exact_job_name = str(job_name or cleaned_run_id).strip()
    cleanup = _cancel_job(
        cleaned_job_id,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )
    if cleanup.errors:
        convergence = _verify_managed_job_convergence(
            cleaned_job_id,
            exact_job_name,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
        )
        if convergence == "terminal":
            cleanup.errors.clear()
            cleanup.resources_removed.append(f"job:{cleaned_job_id}:already-terminal")
        elif convergence == "absent":
            cleanup.errors.clear()
            cleanup.resources_removed.append(f"job:{cleaned_job_id}:already-absent")
        elif convergence.startswith("unavailable:"):
            cleanup.errors.append(
                f"managed job {cleaned_job_id} could not be re-verified after the "
                f"cancel failure: {convergence.removeprefix('unavailable:')}"
            )
    try:
        drained, still_running = wait_for_jobs_terminal(
            [cleaned_job_id],
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
            timeout=job_drain_timeout,
        )
    except JobQueueUnreadableError as exc:
        cleanup.errors.append(
            "managed-job queue is unreadable/unverified after cancellation; "
            f"the run cluster was preserved: {exc}"
        )
        return cleanup
    if not drained:
        cleanup.errors.append(
            "managed job(s) "
            + ", ".join(still_running)
            + f" were still non-terminal {job_drain_timeout}s after cancel"
        )
    elif not cleanup.errors:
        convergence = _verify_managed_job_convergence(
            cleaned_job_id,
            exact_job_name,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
        )
        if convergence not in {"terminal", "absent"}:
            detail = convergence.removeprefix("unavailable:")
            cleanup.errors.append(
                f"managed job {cleaned_job_id} cancellation could not be verified as "
                f"terminal/absent: {detail or convergence}"
            )
    if cleanup.errors:
        # `sky down` updates local cluster/controller handles as part of remote
        # teardown. Preserve those recovery handles until the queue proves the
        # exact job terminal or absent.
        return cleanup
    if teardown_cluster:
        cleanup.extend(
            sky_down(
                cluster or cleaned_run_id,
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                sky_bin=sky_bin,
            )
        )
    return cleanup


def cleanup_launched_workflows(
    jobs: Sequence[tuple[str, str]],
    run_id: str,
    *,
    cluster: str = "",
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    job_drain_timeout: int = DEFAULT_JOB_DRAIN_TIMEOUT_SECONDS,
) -> CleanupResult:
    """Cancel every exact managed-job record, continuing after partial failures."""

    cleanup = CleanupResult()
    seen: set[str] = set()
    targets = [
        (str(job_id or "").strip(), str(job_name or "").strip())
        for job_id, job_name in jobs
        if str(job_id or "").strip()
    ]
    for job_id, exact_name in targets:
        if job_id in seen:
            continue
        seen.add(job_id)
        try:
            result = cleanup_launched_workflow(
                job_id,
                run_id,
                cluster="",
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                sky_bin=sky_bin,
                job_drain_timeout=job_drain_timeout,
                job_name=exact_name,
                teardown_cluster=False,
            )
        except (OSError, ValueError) as exc:
            cleanup.errors.append(
                f"managed job {job_id} cleanup could not start: {type(exc).__name__}: {exc}"
            )
            continue
        cleanup.extend(result)
    if cleanup.errors:
        # All jobs share this run cluster. One unverified drain makes cluster
        # teardown unsafe even when other exact jobs converged successfully.
        return cleanup
    cleanup.extend(
        sky_down(
            cluster or run_id,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
        )
    )
    return cleanup


def _verify_managed_job_convergence(
    job_id: str,
    job_name: str,
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> str:
    """Return terminal/absent, preserving provider/auth lookup failures."""

    from npa.orchestration.skypilot.workflow import lookup_managed_job

    evidence = lookup_managed_job(
        job_name,
        job_id=job_id,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )
    if evidence.outcome == "absent":
        return "absent"
    if evidence.outcome == "unavailable":
        return f"unavailable:{evidence.error or 'provider unavailable'}"
    status = str(evidence.status or "").strip().upper()
    if status and status not in NONTERMINAL_JOB_STATUSES and status != "UNKNOWN":
        return "terminal"
    return status or "UNKNOWN"


def cleanup_all_for_run(
    run_id: str,
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    also_teardown_controller: bool = False,
    job_drain_timeout: int = DEFAULT_JOB_DRAIN_TIMEOUT_SECONDS,
) -> CleanupResult:
    """Cancel jobs and tear down this run's clusters.

    The SkyPilot managed-jobs controller is shared operator state. It is left in
    place by default; pass ``also_teardown_controller=True`` only when no other
    SkyPilot-managed work depends on it.
    """

    _validate_run_id(run_id)
    cleanup = CleanupResult()
    cancelled: list[str] = []
    try:
        matching_jobs = _matching_jobs(
            run_id,
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
        )
    except JobQueueUnreadableError as exc:
        cleanup.errors.append(
            "managed-job queue is unreadable/unverified; run clusters and local "
            f"SkyPilot state were preserved: {exc}"
        )
        return cleanup
    for job in matching_jobs:
        if str(job.get("status", "")).upper() in NONTERMINAL_JOB_STATUSES:
            job_id = str(job.get("job_id") or job.get("id"))
            cleanup.extend(
                _cancel_job(
                    job_id,
                    isolated_config_dir=isolated_config_dir,
                    config_path=config_path,
                    sky_bin=sky_bin,
                )
            )
            cancelled.append(job_id)

    # `sky jobs cancel` returns as soon as cancellation is scheduled, so tearing
    # down immediately races the controller.
    if cancelled:
        try:
            drained, still_running = wait_for_jobs_terminal(
                cancelled,
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                sky_bin=sky_bin,
                timeout=job_drain_timeout,
            )
        except JobQueueUnreadableError as exc:
            cleanup.errors.append(
                "managed-job queue became unreadable/unverified after cancellation; "
                f"run clusters and local SkyPilot state were preserved: {exc}"
            )
            return cleanup
        if not drained:
            cleanup.errors.append(
                "managed job(s) "
                + ", ".join(still_running)
                + f" were still non-terminal {job_drain_timeout}s after cancel; "
                "run clusters and their local recovery handles were preserved until "
                "the queue verifies those jobs terminal or absent."
            )
            return cleanup

    for pattern in cluster_name_patterns_for_run(run_id):
        cleanup.extend(
            sky_down(
                pattern,
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                sky_bin=sky_bin,
            )
        )
    if also_teardown_controller:
        cleanup.extend(
            cleanup_jobs_controller(
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                sky_bin=sky_bin,
            )
        )
    return cleanup


def cluster_name_patterns_for_run(run_id: str) -> list[str]:
    """Return boundary-aware SkyPilot cluster-name globs for a validated run id."""

    _validate_run_id(run_id)
    tag = run_tag(run_id)
    patterns = [tag, f"{tag}-*"]
    sanitized = _sanitize_name(run_id)
    if sanitized and sanitized != tag:
        patterns.extend([sanitized, f"{sanitized}-*"])
    return list(dict.fromkeys(patterns))


def run_tag(run_id: str, *, max_length: int = 32) -> str:
    """Return a Kubernetes/SkyPilot-safe short tag for cluster/task names."""

    sanitized = _sanitize_name(run_id)
    if len(sanitized) <= max_length:
        return sanitized
    return sanitized[-max_length:].strip("-") or sanitized[:max_length].strip("-")


@contextmanager
def skypilot_workflow(
    *,
    run_id: str,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND,
) -> Iterator["_SkyPilotWorkflow"]:
    """Context manager that guarantees run-scoped SkyPilot cleanup.

    The shared managed-jobs controller is not torn down on context exit.
    """

    workflow = _SkyPilotWorkflow(
        run_id=run_id,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
        controller_backend=controller_backend,
    )
    try:
        yield workflow
    finally:
        workflow.cleanup()


@dataclass
class _SkyPilotWorkflow:
    run_id: str
    isolated_config_dir: Path | None = None
    config_path: Path | None = None
    sky_bin: SkyBin = None
    controller_backend: ControllerBackend = DEFAULT_CONTROLLER_BACKEND
    cleanup_result: CleanupResult | None = None

    def submit(self, yaml_path: Path):
        from npa.orchestration.skypilot.workflow import submit_workflow

        return submit_workflow(
            yaml_path,
            self.run_id,
            isolated_config_dir=self.isolated_config_dir,
            sky_bin=self.sky_bin,
            controller_backend=self.controller_backend,
        )

    def cleanup(self) -> CleanupResult:
        self.cleanup_result = cleanup_all_for_run(
            self.run_id,
            isolated_config_dir=self.isolated_config_dir,
            config_path=self.config_path,
            sky_bin=self.sky_bin,
        )
        return self.cleanup_result


def looks_like_in_progress_jobs_error(detail: str) -> bool:
    """Whether ``sky down`` refused because managed jobs are still non-terminal."""

    lowered = str(detail or "").lower()
    return any(marker in lowered for marker in _IN_PROGRESS_JOBS_MARKERS)


def wait_for_jobs_terminal(
    job_ids: Sequence[str],
    *,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: SkyBin = None,
    timeout: int = DEFAULT_JOB_DRAIN_TIMEOUT_SECONDS,
    interval: float = DEFAULT_JOB_DRAIN_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, list[str]]:
    """Block until the given managed jobs are terminal.

    Returns ``(drained, still_running)``. An unreadable queue raises
    :class:`JobQueueUnreadableError`; callers must preserve recoverable state.
    """

    wanted = {str(job_id).strip() for job_id in job_ids if str(job_id).strip()}
    if not wanted:
        return True, []
    deadline = time.monotonic() + max(timeout, 0)
    still_running: list[str] = []
    while True:
        snapshot = _all_jobs(
            isolated_config_dir=isolated_config_dir,
            config_path=config_path,
            sky_bin=sky_bin,
        )
        still_running = [
            job_id
            for job_id, status in _job_statuses(snapshot.jobs).items()
            if job_id in wanted and status in NONTERMINAL_JOB_STATUSES
        ]
        if not still_running:
            return True, []
        if time.monotonic() >= deadline:
            return False, sorted(still_running)
        sleep(max(interval, 0.1))


def _job_statuses(jobs: Sequence[dict[str, Any]]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for job in jobs or []:
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("job_id") or job.get("id") or "").strip()
        if not job_id:
            continue
        status = str(job.get("status") or "").upper()
        # A job group reports one row per task; the job is only terminal once
        # every one of its rows is.
        if job_id in statuses and statuses[job_id] in NONTERMINAL_JOB_STATUSES:
            continue
        statuses[job_id] = status
    return statuses


def _nonterminal_job_ids(
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> list[str]:
    snapshot = _all_jobs(
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )
    return sorted(
        job_id
        for job_id, status in _job_statuses(snapshot.jobs).items()
        if status in NONTERMINAL_JOB_STATUSES
    )


def _all_jobs(
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> JobQueueSnapshot:
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
    try:
        result = _run(
            cmd,
            isolated_config_dir=runtime_config.isolated_config_dir,
            config_path=runtime_config.global_config_path,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise JobQueueUnreadableError(
            "managed-job queue command failed: " + redact_text(str(exc))
        ) from exc
    if is_verified_empty_queue_result(result):
        return JobQueueSnapshot(state="verified_empty")
    combined = " ".join(
        str(value or "").lower() for value in (result.stdout, result.stderr)
    )
    if "no in-progress managed jobs" in combined:
        raise JobQueueUnreadableError(
            "managed-job queue mixed a benign-empty marker with unexpected output"
        )
    if result.returncode != 0:
        detail = _command_detail(result)
        raise JobQueueUnreadableError(
            "managed-job queue command was rejected or unreachable: "
            + redact_text(detail)
        )
    jobs = verified_structured_queue_rows(result)
    if jobs is None:
        raise JobQueueUnreadableError(
            "managed-job queue returned empty, malformed, ambiguous, or "
            "schema-invalid JSON"
        )
    return JobQueueSnapshot(
        state="verified_jobs" if jobs else "verified_empty",
        jobs=tuple(jobs),
    )


def _is_verified_empty_queue_message(
    result: subprocess.CompletedProcess[str],
) -> bool:
    """Recognize only pinned SkyPilot's complete benign empty-queue response.

    SkyPilot 0.12.2 has emitted this diagnostic on either stream and with both
    zero and one return codes.  Requiring the entire non-empty output to equal a
    known sentence prevents an auth/transport failure that merely mentions the
    sentence from becoming false absence proof.
    """

    return is_verified_empty_queue_result(result)


def _matching_jobs(
    run_id: str,
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> list[dict[str, Any]]:
    jobs = _all_jobs(
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    ).jobs
    patterns = {run_id, run_tag(run_id), _sanitize_name(run_id)}

    def name_matches(value: object) -> bool:
        normalized = str(value or "").strip()
        return any(
            pattern and (normalized == pattern or normalized.startswith(pattern + "-"))
            for pattern in patterns
        )

    matched = []
    for job in jobs or []:
        names_match = any(
            name_matches(job.get(key)) for key in ("name", "job_name", "task")
        )
        exact_id = str(job.get("job_id") or job.get("id") or "").strip()
        if names_match or exact_id in patterns:
            matched.append(job)
    return matched


def _validate_run_id(run_id: str) -> None:
    value = str(run_id)
    if len(value) < RUN_ID_MIN_LENGTH:
        raise InvalidRunIdError(
            f"SkyPilot run_id must be at least {RUN_ID_MIN_LENGTH} characters "
            "before cleanup can derive cluster-name patterns."
        )
    if not _RUN_ID_ALLOWED_RE.fullmatch(value):
        raise InvalidRunIdError(
            "SkyPilot run_id may contain only ASCII letters, digits, and hyphens. "
            "Use a long timestamp or UUID-style suffix and avoid glob, shell, or "
            "Kubernetes-unsafe characters."
        )


def _cancel_job(
    job_id: str,
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> CleanupResult:
    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [
        str(ensure_skypilot_version(runtime_config.sky_bin)),
        "jobs",
        "cancel",
        "--yes",
        job_id,
    ]
    result = _run(
        cmd,
        isolated_config_dir=runtime_config.isolated_config_dir,
        config_path=runtime_config.global_config_path,
        timeout=300,
    )
    cleanup = CleanupResult(commands=[cmd])
    if result.returncode == 0:
        cleanup.resources_removed.append(f"job:{job_id}")
    else:
        cleanup.errors.append(_format_command_error(cmd, result))
    return cleanup


def _jobs_controller_names(
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
) -> tuple[list[str], str]:
    controller_clusters, status_error = _jobs_controller_clusters(
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )
    if status_error:
        return [], status_error
    return [_cluster_name(cluster) for cluster in controller_clusters], ""


def _jobs_controller_clusters(
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
    refresh: bool = True,
    env_extra: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [str(ensure_skypilot_version(runtime_config.sky_bin)), "status"]
    if refresh:
        cmd.append("--refresh")
    cmd.extend(["--output", "json"])
    result = _run(
        cmd,
        isolated_config_dir=runtime_config.isolated_config_dir,
        config_path=runtime_config.global_config_path,
        timeout=300,
        env_extra=env_extra,
    )
    if result.returncode != 0:
        return [], _format_command_error(cmd, result)
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return [], f"{' '.join(cmd)} returned non-json output"
    clusters = payload if isinstance(payload, list) else payload.get("clusters", [])
    controllers = []
    for cluster in clusters or []:
        if not isinstance(cluster, dict):
            continue
        name = _cluster_name(cluster)
        if name.startswith("sky-jobs-controller-"):
            controllers.append(cluster)
    deduped: dict[str, dict[str, Any]] = {}
    for controller in controllers:
        deduped.setdefault(_cluster_name(controller), controller)
    return list(deduped.values()), ""


def _cluster_name(cluster: dict[str, Any]) -> str:
    return str(cluster.get("name") or cluster.get("cluster") or "")


def _down_jobs_controller(
    controller_name: str,
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    sky_bin: SkyBin,
    job_drain_timeout: int = DEFAULT_JOB_DRAIN_TIMEOUT_SECONDS,
    env_extra: dict[str, str] | None = None,
) -> CleanupResult:
    runtime_config = resolve_config(
        sky_bin=sky_bin,
        global_config_path=config_path,
        isolated_config_dir=isolated_config_dir,
    )
    cmd = [
        str(ensure_skypilot_version(runtime_config.sky_bin)),
        "down",
        "--yes",
        controller_name,
    ]
    cleanup = CleanupResult(commands=[cmd])
    result = _run(
        cmd,
        isolated_config_dir=runtime_config.isolated_config_dir,
        config_path=runtime_config.global_config_path,
        timeout=900,
        input_text="delete\n",
        env_extra=env_extra,
    )
    if result.returncode != 0 and looks_like_in_progress_jobs_error(
        _combined_output(result)
    ):
        # A job that finished cancelling between our poll and this call still
        # trips the guard; drain once more and retry rather than reporting a
        # failure the operator can only fix by waiting and rerunning.
        try:
            pending = _nonterminal_job_ids(
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                sky_bin=sky_bin,
            )
            drained, still_running = wait_for_jobs_terminal(
                pending,
                isolated_config_dir=isolated_config_dir,
                config_path=config_path,
                sky_bin=sky_bin,
                timeout=job_drain_timeout,
            )
        except JobQueueUnreadableError as exc:
            cleanup.errors.append(
                "managed-job queue is unreadable/unverified; controller teardown "
                f"was not retried and local SkyPilot state must be preserved: {exc}"
            )
            return cleanup
        if drained:
            cleanup.commands.append(cmd)
            result = _run(
                cmd,
                isolated_config_dir=runtime_config.isolated_config_dir,
                config_path=runtime_config.global_config_path,
                timeout=900,
                input_text="delete\n",
                env_extra=env_extra,
            )
        elif still_running:
            cleanup.errors.append(
                f"`sky down {controller_name}` refuses while managed job(s) "
                + ", ".join(still_running)
                + " are still non-terminal, and they did not finish cancelling within "
                f"{job_drain_timeout}s. Suggested action: `sky jobs cancel -a`, wait for "
                "`sky jobs queue --all` to show them terminal, then rerun teardown."
            )
            return cleanup
    if result.returncode == 0:
        cleanup.resources_removed.append(controller_name)
    elif _is_conclusive_absence_error(_command_detail(result)):
        cleanup.resources_removed.append(f"{controller_name}:already-absent")
    else:
        cleanup.errors.append(_format_command_error(cmd, result))
    return cleanup


def _is_kubernetes_controller(cluster: dict[str, Any]) -> bool:
    return "kubernetes" in json.dumps(cluster, sort_keys=True).lower()


def _controller_belongs_to_context(cluster: dict[str, Any], context: str) -> bool:
    """Require exact context evidence; a generic Kubernetes row is ambiguous."""

    expected = str(context or "").strip().lower()
    if not expected:
        return False

    def scalar_values(value: object) -> Iterator[str]:
        if isinstance(value, dict):
            for item in value.values():
                yield from scalar_values(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from scalar_values(item)
        elif isinstance(value, str):
            yield value.strip().lower()

    exact_values = {
        expected,
        f"k8s/{expected}",
        f"kubernetes/{expected}",
    }
    return any(value in exact_values for value in scalar_values(cluster))


@contextmanager
def _cloned_skypilot_state(source_root: Path | None) -> Iterator[Path]:
    """Yield an isolated clone so remote deletion cannot erase real metadata."""

    source_env = sky_environment(source_root)
    source_home = Path(source_env.get("HOME") or Path.home())
    with tempfile.TemporaryDirectory(prefix="npa-controller-transaction-") as raw:
        clone_root = Path(raw)
        clone_home = clone_root / "home"
        clone_home.mkdir(parents=True, exist_ok=True)
        source_sky = source_home / ".sky"
        if source_sky.is_dir() and not source_sky.is_symlink():
            shutil.copytree(source_sky, clone_home / ".sky", symlinks=True)
        yield clone_root


def _controller_name_from_pod(item: dict[str, Any]) -> str:
    raw_metadata = item.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    raw_labels = metadata.get("labels")
    labels: dict[str, Any] = raw_labels if isinstance(raw_labels, dict) else {}
    searchable = [
        str(labels.get("skypilot-cluster") or ""),
        str(labels.get("ray.io/cluster") or ""),
        str(labels.get("app.kubernetes.io/instance") or ""),
        str(metadata.get("name") or ""),
    ]
    for value in searchable:
        match = re.search(r"(sky-jobs-controller-[a-zA-Z0-9-]+)", value)
        if match:
            return match.group(1).removesuffix("-ray-head")
    return ""


def _kubernetes_controller_pods(
    *,
    kubeconfig: Path,
    context: str,
) -> tuple[list[tuple[str, str, str]], str]:
    from npa.cluster.drain import _noninteractive_kubeconfig_env

    cmd = [
        "kubectl",
        "--context",
        context,
        "get",
        "pods",
        "--all-namespaces",
        "-o",
        "json",
    ]
    with _noninteractive_kubeconfig_env(str(kubeconfig)) as (env, issue):
        if issue is not None:
            return [], issue.summary
        try:
            result = subprocess.run(
                cmd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return [], f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return [], _format_command_error(cmd, result)
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return [], "kubectl get pods returned non-json output"
    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    items = raw_items if isinstance(raw_items, list) else []
    matches: list[tuple[str, str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        controller = _controller_name_from_pod(item)
        if not controller:
            continue
        raw_metadata = item.get("metadata")
        metadata: dict[str, Any] = (
            raw_metadata if isinstance(raw_metadata, dict) else {}
        )
        matches.append(
            (
                str(metadata.get("namespace") or "default"),
                str(metadata.get("name") or ""),
                controller,
            )
        )
    return matches, ""


def _wait_for_controller_pods_absent(
    controller_names: set[str],
    *,
    kubeconfig: Path,
    context: str,
    timeout: int = 900,
    interval: float = 5.0,
) -> tuple[list[tuple[str, str, str]], str]:
    deadline = time.monotonic() + timeout
    while True:
        pods, error = _kubernetes_controller_pods(
            kubeconfig=kubeconfig, context=context
        )
        if error:
            return [], error
        relevant = [item for item in pods if item[2] in controller_names]
        if not relevant:
            return [], ""
        if time.monotonic() >= deadline:
            return relevant, f"controller pods remained after {timeout}s"
        time.sleep(interval)


def _run(
    cmd: list[str],
    *,
    isolated_config_dir: Path | None,
    config_path: Path | None,
    timeout: int,
    input_text: str | None = None,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    effective_cmd = list(cmd)
    if config_path is not None and "--config" not in effective_cmd:
        command_name_index = (
            2 if len(effective_cmd) > 1 and effective_cmd[1] == "jobs" else 1
        )
        effective_cmd[command_name_index + 1 : command_name_index + 1] = [
            "--config",
            str(config_path),
        ]
    env = sky_environment(isolated_config_dir)
    env.update(env_extra or {})
    from npa.progress import WaitProgress

    operation = " ".join([Path(effective_cmd[0]).name, *effective_cmd[1:3]]).replace(
        " --config", ""
    )
    progress = WaitProgress(f"SkyPilot subprocess ({operation})")
    progress.start(f"attempt=1 timeout={timeout}s")
    stop = threading.Event()

    def report_wait() -> None:
        while not stop.wait(progress.interval):
            progress.tick("attempt=1 state=running")

    reporter = threading.Thread(
        target=report_wait, name="npa-skypilot-progress", daemon=True
    )
    reporter.start()
    outcome = "failed"
    try:
        result = subprocess.run(
            effective_cmd,
            env=env,
            text=True,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        outcome = "completed" if result.returncode == 0 else "failed"
        return result
    except subprocess.TimeoutExpired:
        outcome = "timed_out"
        raise
    finally:
        stop.set()
        reporter.join(timeout=1)
        progress.finish(outcome, "attempt=1")


def sky_environment(isolated_config_dir: Path | None = None) -> dict[str, str]:
    """Return an environment that keeps SkyPilot state inside a run directory."""

    env = os.environ.copy()
    if isolated_config_dir is None:
        return env
    root = Path(isolated_config_dir)
    home = root / "home"
    runtime = root / "sky-runtime"
    home.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    # Kubernetes kubeconfigs commonly use the Nebius CLI as an exec credential
    # plugin.  Isolating HOME for SkyPilot must not make that exact provider
    # identity disappear: link the already-selected operator configuration into
    # the owner-only run home without copying credentials into journals/state.
    provider_home = Path(env.get("HOME") or "").expanduser()
    provider_config = provider_home / ".nebius"
    isolated_provider_config = home / ".nebius"
    if (
        provider_home != home
        and provider_config.is_dir()
        and not isolated_provider_config.exists()
        and not isolated_provider_config.is_symlink()
    ):
        isolated_provider_config.symlink_to(provider_config, target_is_directory=True)
    env["HOME"] = str(home)
    env["SKY_RUNTIME_DIR"] = str(runtime)
    env["PYTHONUNBUFFERED"] = "1"
    repo_src = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = str(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    sanitized = re.sub(r"-+", "-", sanitized)
    return sanitized


def _format_command_error(
    cmd: list[str], result: subprocess.CompletedProcess[str]
) -> str:
    detail = _combined_output(result) or f"exit {result.returncode}"
    return f"{' '.join(cmd)} failed: {detail}"


def _command_detail(result: subprocess.CompletedProcess[str]) -> str:
    return _combined_output(result) or f"exit {result.returncode}"


def _is_conclusive_absence_error(detail: str) -> bool:
    """Recognize an exact absent resource without swallowing auth/API failures."""

    lowered = " ".join(str(detail or "").lower().split())
    if any(
        marker in lowered
        for marker in (
            "unauthorized",
            "unauthenticated",
            "permission denied",
            "forbidden",
            "timed out",
            "timeout",
            "connection refused",
            "unavailable",
        )
    ):
        return False
    return any(
        marker in lowered
        for marker in (
            "not found",
            "notfound",
            "does not exist",
            "already absent",
            "no cluster found",
        )
    )


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or "").strip() or (result.stdout or "").strip()
