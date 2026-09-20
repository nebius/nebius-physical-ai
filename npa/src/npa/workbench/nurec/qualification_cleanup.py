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


CLEANUP_FORMAT = "npa_ncore_qualification_cleanup_v1"
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
    job_name: str,
    job_id: str,
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
            and annotations.get("skypilot-managed-job-name") == job_name
            and str(annotations.get("skypilot-managed-job-id") or "") == job_id
            and status.get("phase") not in {"Succeeded", "Failed"}
        ):
            active += 1
    return active


def cleanup_qualification(
    *,
    run_id: str,
    job_id: str,
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
    from npa.orchestration.skypilot.cleanup import cleanup_launched_workflow

    run_id = _safe(run_id, "run ID")
    job_id = str(job_id).strip()
    if not job_id.isdigit() or int(job_id) < 1:
        raise NcoreQualificationCleanupError("managed job ID is invalid")
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
        build.get("schema") != "npa.ncore.committed-oci-build.v1"
        or build.get("source_sha") != source_sha
        or not isinstance(argv, list)
        or "--oci-output" not in argv
        or any(value in {"--push", "push", "publish"} for value in argv)
    ):
        raise NcoreQualificationCleanupError(
            "build receipt does not prove a local-only OCI route"
        )
    cleaner = workflow_cleaner or cleanup_launched_workflow
    cleanup = cleaner(
        job_id,
        run_id,
        job_name=run_id,
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
        len(cancel_indexes) != 1
        or len(down_indexes) != 1
        or cancel_indexes[0] >= down_indexes[0]
    ):
        raise NcoreQualificationCleanupError(
            "cleanup did not prove cancel-before-destroy ordering"
        )
    active_pods = _active_job_pods(
        kubectl_bin=kubectl_bin,
        context=context,
        namespace=namespace,
        job_name=run_id,
        job_id=job_id,
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
        "managed_job_id_sha256": hashlib.sha256(job_id.encode()).hexdigest(),
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
