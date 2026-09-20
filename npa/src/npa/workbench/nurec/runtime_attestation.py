"""Control-plane observation of NRE pod image and GPU allocation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Sequence

from npa.errors import NpaError


STAGE_FORMAT = "npa_nurec_kubernetes_runtime_stage_v2"
BUNDLE_FORMAT = "npa_nurec_runtime_attestation_v3"
STAGES = ("reconstruct", "render")
GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,252}")
_JOB_ID = re.compile(r"[1-9][0-9]*")
_GPU_LABELS = {
    "NVIDIARTXPRO6000BLACKWELLSERVEREDITION",
    "RTXPRO6000BLACKWELLSERVEREDITION",
}


class NurecRuntimeAttestationError(NpaError):
    """Authoritative Kubernetes runtime evidence was absent or inconsistent."""


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_fresh_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _json_command(
    command: Sequence[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    try:
        result = runner(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NurecRuntimeAttestationError(
            "Kubernetes control-plane observation failed"
        ) from exc
    if result.returncode != 0:
        raise NurecRuntimeAttestationError(
            "Kubernetes control-plane observation failed"
        )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise NurecRuntimeAttestationError(
            "Kubernetes control-plane response is not JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise NurecRuntimeAttestationError(
            "Kubernetes control-plane response is not an object"
        )
    return payload


def _exact_name(value: str, label: str) -> str:
    value = str(value).strip()
    if _SAFE_NAME.fullmatch(value) is None:
        raise NurecRuntimeAttestationError(f"{label} is invalid")
    return value


def _expected_digest(image: str) -> str:
    match = re.fullmatch(r"[^@\s]+@(sha256:[0-9a-f]{64})", str(image).strip())
    if match is None:
        raise NurecRuntimeAttestationError(
            "expected image must be an immutable canonical digest"
        )
    return match.group(1)


def _observed_digest(image_id: Any) -> str:
    value = str(image_id or "").strip()
    matches = re.findall(r"sha256:[0-9a-f]{64}", value)
    if len(matches) != 1:
        raise NurecRuntimeAttestationError(
            "pod status does not contain one immutable image digest"
        )
    return matches[0]


def _gpu_name(node: dict[str, Any]) -> str:
    labels = node.get("metadata", {}).get("labels", {})
    if not isinstance(labels, dict):
        raise NurecRuntimeAttestationError("node GPU labels are missing")
    candidates = [
        labels.get("nvidia.com/gpu.product"),
        labels.get("skypilot.co/accelerator"),
    ]
    normalized = {
        re.sub(r"[^A-Z0-9]", "", str(value).upper()) for value in candidates if value
    }
    if normalized.isdisjoint(_GPU_LABELS):
        raise NurecRuntimeAttestationError(
            "node does not advertise the exact RTX PRO 6000 Blackwell Server Edition GPU"
        )
    return GPU_NAME


def _job_id(value: str) -> str:
    value = str(value).strip()
    if _JOB_ID.fullmatch(value) is None:
        raise NurecRuntimeAttestationError("managed job ID is invalid")
    return value


def _identity_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _bound_pod(
    pod: dict[str, Any],
    *,
    stage: str,
    managed_job_name: str,
    managed_job_id: str,
) -> bool:
    metadata = pod.get("metadata")
    if not isinstance(metadata, dict):
        return False
    annotations = metadata.get("annotations")
    labels = metadata.get("labels")
    if not isinstance(annotations, dict) or not isinstance(labels, dict):
        return False
    cluster_name = str(labels.get("skypilot-cluster-name") or "")
    return (
        annotations.get("skypilot-managed-job-name") == managed_job_name
        and str(annotations.get("skypilot-managed-job-id") or "") == managed_job_id
        and cluster_name.startswith(f"{stage}-{managed_job_id}-")
    )


def _discover_pod(
    prefix: Sequence[str],
    *,
    stage: str,
    managed_job_name: str,
    managed_job_id: str,
    container_name: str,
    max_wait_seconds: float,
    poll_seconds: float,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    monotonic: Callable[[], float],
    sleeper: Callable[[float], None],
) -> dict[str, Any]:
    if max_wait_seconds < 0 or poll_seconds <= 0:
        raise NurecRuntimeAttestationError("runtime observation timing is invalid")
    deadline = monotonic() + max_wait_seconds if max_wait_seconds else None
    while True:
        listed = _json_command(
            [*prefix, "get", "pods", "--output", "json"], runner=runner
        )
        items = listed.get("items")
        if not isinstance(items, list):
            raise NurecRuntimeAttestationError("Kubernetes pod list is invalid")
        matches = [
            item
            for item in items
            if isinstance(item, dict)
            and _bound_pod(
                item,
                stage=stage,
                managed_job_name=managed_job_name,
                managed_job_id=managed_job_id,
            )
        ]
        if len(matches) > 1:
            raise NurecRuntimeAttestationError(
                "managed job stage maps to multiple Kubernetes pods"
            )
        observable = []
        for item in matches:
            spec = item.get("spec")
            status = item.get("status")
            statuses = (
                status.get("containerStatuses") if isinstance(status, dict) else None
            )
            if (
                isinstance(spec, dict)
                and isinstance(spec.get("nodeName"), str)
                and spec["nodeName"]
                and isinstance(statuses, list)
                and any(
                    isinstance(container, dict)
                    and container.get("name") == container_name
                    and isinstance(container.get("state"), dict)
                    for container in statuses
                )
            ):
                observable.append(item)
        if len(observable) > 1:
            raise NurecRuntimeAttestationError(
                "managed job stage maps to multiple observable Kubernetes pods"
            )
        if observable:
            return observable[0]
        if matches and str(matches[0].get("status", {}).get("phase") or "") in {
            "Failed",
            "Succeeded",
        }:
            return matches[0]
        if deadline is not None and monotonic() >= deadline:
            raise NurecRuntimeAttestationError(
                "managed job stage pod was not observed before the deadline"
            )
        sleeper(poll_seconds)


def observe_kubernetes_stage(
    *,
    stage: str,
    pod_name: str = "",
    namespace: str,
    container_name: str,
    expected_image: str,
    managed_job_name: str,
    managed_job_id: str,
    output_path: Path,
    context: str,
    max_wait_seconds: float = 0,
    poll_seconds: float = 5,
    kubectl_bin: str = "kubectl",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Query exact pod/node objects and write a sanitized immutable stage receipt."""
    if stage not in STAGES:
        raise NurecRuntimeAttestationError("stage must be reconstruct or render")
    pod_name = _exact_name(pod_name, "pod name") if pod_name else ""
    namespace = _exact_name(namespace, "namespace")
    container_name = _exact_name(container_name, "container name")
    managed_job_name = _exact_name(managed_job_name, "managed job name")
    managed_job_id = _job_id(managed_job_id)
    context = _exact_name(context, "context")
    expected_digest = _expected_digest(expected_image)
    prefix = [kubectl_bin, "--context", context]
    prefix.extend(["--namespace", namespace])
    pod = (
        _json_command(
            [*prefix, "get", "pod", pod_name, "--output", "json"], runner=runner
        )
        if pod_name
        else _discover_pod(
            prefix,
            stage=stage,
            managed_job_name=managed_job_name,
            managed_job_id=managed_job_id,
            container_name=container_name,
            max_wait_seconds=max_wait_seconds,
            poll_seconds=poll_seconds,
            runner=runner,
            monotonic=monotonic,
            sleeper=sleeper,
        )
    )
    spec = pod.get("spec")
    status = pod.get("status")
    metadata = pod.get("metadata")
    if not all(isinstance(value, dict) for value in (spec, status, metadata)):
        raise NurecRuntimeAttestationError("pod object is incomplete")
    if not _bound_pod(
        pod,
        stage=stage,
        managed_job_name=managed_job_name,
        managed_job_id=managed_job_id,
    ):
        raise NurecRuntimeAttestationError(
            "pod is not bound to the exact managed job and stage"
        )
    containers = spec.get("containers")
    statuses = status.get("containerStatuses")
    if not isinstance(containers, list) or not isinstance(statuses, list):
        raise NurecRuntimeAttestationError("pod container evidence is missing")
    matching_specs = [
        item
        for item in containers
        if isinstance(item, dict) and item.get("name") == container_name
    ]
    matching_statuses = [
        item
        for item in statuses
        if isinstance(item, dict) and item.get("name") == container_name
    ]
    if len(matching_specs) != 1 or len(matching_statuses) != 1:
        raise NurecRuntimeAttestationError(
            "pod does not contain the exact expected container"
        )
    container = matching_specs[0]
    container_status = matching_statuses[0]
    if container.get("image") != expected_image:
        raise NurecRuntimeAttestationError(
            "pod requested image differs from the expected digest"
        )
    observed_digest = _observed_digest(container_status.get("imageID"))
    if observed_digest != expected_digest:
        raise NurecRuntimeAttestationError(
            "pod observed image digest differs from the expected digest"
        )
    limits = container.get("resources", {}).get("limits", {})
    if not isinstance(limits, dict) or str(limits.get("nvidia.com/gpu")) != "1":
        raise NurecRuntimeAttestationError("pod does not have an exact one-GPU limit")
    state = container_status.get("state")
    if not isinstance(state, dict):
        raise NurecRuntimeAttestationError("pod container state is missing")
    running = isinstance(state.get("running"), dict)
    terminated = state.get("terminated")
    terminal_ok = isinstance(terminated, dict) and terminated.get("exitCode") == 0
    if not running and not terminal_ok:
        raise NurecRuntimeAttestationError(
            "pod container was neither running nor successfully terminated"
        )
    node_name = spec.get("nodeName")
    pod_uid = metadata.get("uid")
    if not isinstance(node_name, str) or not node_name or not isinstance(pod_uid, str):
        raise NurecRuntimeAttestationError("pod scheduling identity is missing")
    node = _json_command(
        [*prefix, "get", "node", node_name, "--output", "json"], runner=runner
    )
    node_uid = node.get("metadata", {}).get("uid")
    if not isinstance(node_uid, str) or not node_uid:
        raise NurecRuntimeAttestationError("node identity is missing")
    cluster_name = str(metadata["labels"]["skypilot-cluster-name"])
    relevant = {
        "pod_uid": pod_uid,
        "node_uid": node_uid,
        "managed_job_name": managed_job_name,
        "managed_job_id": managed_job_id,
        "cluster_name": cluster_name,
        "context": context,
        "namespace": namespace,
        "container_name": container_name,
        "requested_image": container.get("image"),
        "observed_image_id": container_status.get("imageID"),
        "gpu_limit": limits.get("nvidia.com/gpu"),
        "container_state": "running" if running else "terminated_zero",
    }
    receipt = {
        "format": STAGE_FORMAT,
        "status": "pass",
        "source": "kubernetes_control_plane",
        "stage": stage,
        "requested_image": expected_image,
        "observed_image_digest": observed_digest,
        "gpu_names": [_gpu_name(node)],
        "gpu_count": 1,
        "resource_identity_sha256": hashlib.sha256(
            f"{pod_uid}\0{node_uid}".encode()
        ).hexdigest(),
        "pod_identity_sha256": _identity_sha(pod_uid),
        "managed_job_name_sha256": _identity_sha(managed_job_name),
        "managed_job_id_sha256": _identity_sha(managed_job_id),
        "task_cluster_sha256": _identity_sha(cluster_name),
        "context_sha256": _identity_sha(context),
        "namespace_sha256": _identity_sha(namespace),
        "control_plane_record_sha256": _sha(relevant),
        "container_state": relevant["container_state"],
    }
    _write_fresh_private(output_path, receipt)
    return receipt


def _load_stage(path: Path, expected_stage: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NurecRuntimeAttestationError("stage receipt is unreadable") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("format") != STAGE_FORMAT
        or payload.get("status") != "pass"
        or payload.get("source") != "kubernetes_control_plane"
        or payload.get("stage") != expected_stage
    ):
        raise NurecRuntimeAttestationError("stage receipt contract differs")
    return payload


def bundle_runtime_attestations(
    *,
    reconstruct_path: Path,
    render_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind independently observed reconstruct and render pods into one receipt."""
    stages = {
        "reconstruct": _load_stage(reconstruct_path, "reconstruct"),
        "render": _load_stage(render_path, "render"),
    }
    images = {stage["requested_image"] for stage in stages.values()}
    digests = {stage["observed_image_digest"] for stage in stages.values()}
    gpu_names = {tuple(stage.get("gpu_names", [])) for stage in stages.values()}
    binding_fields = (
        "managed_job_name_sha256",
        "managed_job_id_sha256",
        "context_sha256",
        "namespace_sha256",
    )
    bindings = {
        field: {stage.get(field) for stage in stages.values()}
        for field in binding_fields
    }
    pod_identities = {stage.get("pod_identity_sha256") for stage in stages.values()}
    task_identities = {stage.get("task_cluster_sha256") for stage in stages.values()}
    if (
        len(images) != 1
        or len(digests) != 1
        or len(gpu_names) != 1
        or any(len(values) != 1 for values in bindings.values())
        or len(pod_identities) != len(stages)
        or len(task_identities) != len(stages)
    ):
        raise NurecRuntimeAttestationError(
            "reconstruct and render runtime identities differ"
        )
    receipt = {
        "format": BUNDLE_FORMAT,
        "status": "pass",
        "source": "kubernetes_control_plane",
        "requested_image": next(iter(images)),
        "observed_image_digest": next(iter(digests)),
        "gpu_names": list(next(iter(gpu_names))),
        "gpu_count": 1,
        **{field: next(iter(values)) for field, values in bindings.items()},
        "stages": {
            name: {
                **payload,
                "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, payload, path in (
                ("reconstruct", stages["reconstruct"], reconstruct_path),
                ("render", stages["render"], render_path),
            )
        },
    }
    _write_fresh_private(output_path, receipt)
    return receipt
