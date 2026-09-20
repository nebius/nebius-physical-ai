"""Derived cleanup receipt for an NCore qualification run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Sequence

from npa.errors import NpaError
from npa.workbench.nurec.s3_probe import _prefix, _snapshot


CLEANUP_FORMAT = "npa_ncore_qualification_cleanup_v2"
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,252}")


class NcoreQualificationCleanupError(NpaError):
    """Owned qualification resources were not proven terminal or absent."""


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _safe(value: str, label: str) -> str:
    value = str(value).strip()
    if _SAFE.fullmatch(value) is None:
        raise NcoreQualificationCleanupError(f"{label} is invalid")
    return value


def _run(
    command: Sequence[str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NcoreQualificationCleanupError(
            "local cleanup observation failed"
        ) from exc


def _remove_local(
    *,
    docker_bin: str,
    image: str,
    builder: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> list[list[str]]:
    commands = [
        [docker_bin, "image", "rm", "--force", image],
        [docker_bin, "buildx", "rm", builder],
    ]
    for command in commands:
        result = _run(command, runner)
        detail = (result.stdout + "\n" + result.stderr).lower()
        if result.returncode != 0 and not any(
            marker in detail
            for marker in ("no such image", "not found", "does not exist", "no builder")
        ):
            raise NcoreQualificationCleanupError("local runtime cleanup failed")
    checks = [
        [docker_bin, "image", "inspect", image],
        [docker_bin, "buildx", "inspect", builder],
    ]
    for command in checks:
        result = _run(command, runner)
        detail = (result.stdout + "\n" + result.stderr).lower()
        if result.returncode == 0 or not any(
            marker in detail
            for marker in ("no such image", "not found", "does not exist", "no builder")
        ):
            raise NcoreQualificationCleanupError("local runtime absence was not proven")
    return [*commands, *checks]


def _active_job_pods(
    *,
    kubectl_bin: str,
    context: str,
    namespace: str,
    jobs: set[tuple[str, str]],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> int:
    result = _run(
        [
            kubectl_bin,
            "--context",
            context,
            "--namespace",
            namespace,
            "get",
            "pods",
            "--output",
            "json",
        ],
        runner,
    )
    if result.returncode != 0:
        raise NcoreQualificationCleanupError(
            "Kubernetes orphan inventory was unavailable"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise NcoreQualificationCleanupError(
            "Kubernetes orphan inventory was invalid"
        ) from exc
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise NcoreQualificationCleanupError("Kubernetes orphan inventory was invalid")
    active = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata")
        status = item.get("status")
        if not isinstance(metadata, dict) or not isinstance(status, dict):
            continue
        annotations = metadata.get("annotations")
        if (
            isinstance(annotations, dict)
            and (
                str(annotations.get("skypilot-managed-job-id") or ""),
                str(annotations.get("skypilot-managed-job-name") or ""),
            )
            in jobs
            and status.get("phase") not in {"Succeeded", "Failed"}
        ):
            active += 1
    return active


def _workflow_jobs(path: Path, run_id: str) -> tuple[list[tuple[str, str]], str]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_uid != os.getuid()
        or path.stat().st_nlink != 1
        or path.stat().st_mode & 0o077
    ):
        raise NcoreQualificationCleanupError("workflow status evidence is missing")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise NcoreQualificationCleanupError(
            "workflow status evidence is invalid"
        ) from exc
    stages = payload.get("stages") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("run_id") != run_id
        or not isinstance(stages, dict)
    ):
        raise NcoreQualificationCleanupError("workflow status run binding differs")
    expected_states = {"reconstruct", "render", "visualize", "finalize"}
    observed_states = {
        str(stage.get("workflow_state") or "")
        for stage in stages.values()
        if isinstance(stage, dict)
    }
    if observed_states != expected_states:
        raise NcoreQualificationCleanupError(
            "workflow status does not contain the complete qualification graph"
        )
    jobs: dict[str, str] = {}
    for stage in stages.values():
        if not isinstance(stage, dict) or stage.get("job_attribution") == "ambiguous":
            raise NcoreQualificationCleanupError(
                "workflow status contains ambiguous job attribution"
            )
        attempts = stage.get("managed_job_attempts")
        if not isinstance(attempts, list):
            attempts = [
                {
                    "job_id": stage.get("managed_job_id"),
                    "job_name": stage.get("job_name"),
                }
            ]
        before_count = len(jobs)
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise NcoreQualificationCleanupError(
                    "workflow status job evidence is invalid"
                )
            job_id = str(attempt.get("job_id") or "").strip()
            job_name = str(attempt.get("job_name") or "").strip()
            if not job_id:
                continue
            if not job_id.isdigit() or int(job_id) < 1:
                raise NcoreQualificationCleanupError(
                    "workflow status managed job ID is invalid"
                )
            job_name = _safe(job_name, "managed job name")
            if job_id in jobs and jobs[job_id] != job_name:
                raise NcoreQualificationCleanupError(
                    "workflow status managed job identity is inconsistent"
                )
            jobs[job_id] = job_name
        if len(jobs) == before_count:
            raise NcoreQualificationCleanupError(
                "workflow status stage has no exact managed job"
            )
    if not jobs:
        raise NcoreQualificationCleanupError("workflow status contains no managed jobs")
    return sorted(jobs.items(), key=lambda item: int(item[0])), hashlib.sha256(
        raw
    ).hexdigest()


def cleanup_qualification(
    *,
    run_id: str,
    workflow_status_path: Path,
    context: str,
    namespace: str,
    storage_prefix: str,
    local_image: str,
    builder: str,
    build_receipt_path: Path,
    source_sha: str,
    output_path: Path,
    isolated_config_dir: Path | None = None,
    config_path: Path | None = None,
    sky_bin: str | None = None,
    docker_bin: str = "docker",
    kubectl_bin: str = "kubectl",
    storage_client: Any = None,
    process_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    workflow_cleaner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Cancel, await terminality, down owned compute, and prove no active pod."""
    from npa.clients.storage import StorageClient
    from npa.orchestration.skypilot.cleanup import cleanup_launched_workflows

    run_id = _safe(run_id, "run ID")
    jobs, workflow_status_sha256 = _workflow_jobs(workflow_status_path, run_id)
    context = _safe(context, "context")
    namespace = _safe(namespace, "namespace")
    builder = _safe(builder, "builder")
    if (
        not local_image
        or local_image.startswith("-")
        or any(character.isspace() for character in local_image)
    ):
        raise NcoreQualificationCleanupError("local image reference is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise NcoreQualificationCleanupError("source SHA is invalid")
    try:
        build = json.loads(build_receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NcoreQualificationCleanupError("build receipt is invalid") from exc
    argv = build.get("argv") if isinstance(build, dict) else None
    if (
        not isinstance(build, dict)
        or build.get("schema") != "npa.ncore.committed-oci-build.v1"
        or build.get("source_sha") != source_sha
        or not isinstance(argv, list)
        or "--oci-output" not in argv
        or any(value in {"--push", "push", "publish"} for value in argv)
    ):
        raise NcoreQualificationCleanupError(
            "build receipt does not prove a local-only OCI route"
        )
    cleaner = workflow_cleaner or cleanup_launched_workflows
    cleanup = cleaner(
        jobs,
        run_id,
        isolated_config_dir=isolated_config_dir,
        config_path=config_path,
        sky_bin=sky_bin,
    )
    if cleanup.errors:
        raise NcoreQualificationCleanupError(
            "managed-job cancellation or cluster teardown did not converge"
        )
    command_tokens = [list(map(str, command)) for command in cleanup.commands]
    cancel_indexes = [
        index
        for index, command in enumerate(command_tokens)
        if "jobs" in command and "cancel" in command
    ]
    down_indexes = [
        index for index, command in enumerate(command_tokens) if "down" in command
    ]
    if (
        len(cancel_indexes) != len(jobs)
        or len(down_indexes) != 1
        or max(cancel_indexes) >= down_indexes[0]
    ):
        raise NcoreQualificationCleanupError(
            "cleanup did not prove cancel-before-destroy ordering"
        )
    active_pods = _active_job_pods(
        kubectl_bin=kubectl_bin,
        context=context,
        namespace=namespace,
        jobs=set(jobs),
        runner=process_runner,
    )
    if active_pods:
        raise NcoreQualificationCleanupError(
            "exact managed job still has active Kubernetes pods"
        )
    local_commands = _remove_local(
        docker_bin=docker_bin,
        image=local_image,
        builder=builder,
        runner=process_runner,
    )
    bucket, prefix = _prefix(storage_prefix)
    inventory = _snapshot(
        storage_client or StorageClient.from_environment(), bucket, prefix
    )
    if not inventory:
        raise NcoreQualificationCleanupError(
            "retained qualification evidence prefix is empty"
        )
    receipt = {
        "format": CLEANUP_FORMAT,
        "status": "pass",
        "run_id_sha256": hashlib.sha256(run_id.encode()).hexdigest(),
        "workflow_status_sha256": workflow_status_sha256,
        "managed_jobs": len(jobs),
        "managed_job_identities_sha256": _sha(jobs),
        "jobs_terminal_or_absent": True,
        "cancel_before_destroy": True,
        "active_job_pods": 0,
        "controller_disposition": "retained_not_owned",
        "storage_disposition": "retained_declared",
        "storage_inventory_sha256": _sha(inventory),
        "storage_objects": len(inventory),
        "registry_disposition": "not_created_local_oci_route",
        "build_receipt_sha256": hashlib.sha256(
            build_receipt_path.read_bytes()
        ).hexdigest(),
        "local_runtime_disposition": "removed",
        "workflow_commands_sha256": _sha(command_tokens),
        "local_commands_sha256": _sha(local_commands),
        "orphan_count": 0,
    }
    _write_private(output_path, receipt)
    return receipt
