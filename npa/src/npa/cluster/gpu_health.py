"""Fail-closed post-deploy health validation for NPA mk8s GPU clusters."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from npa.cluster.gpu_driver import gpus_per_node

DEFAULT_STABILIZATION_SECONDS = 120
DEFAULT_POLL_SECONDS = 10
DEFAULT_CUDA_SMOKE_IMAGE = (
    "nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda12.5.0-ubuntu22.04"
)
_FABRIC_SUCCESS = frozenset({"complete", "completed", "success", "successful"})

CaptureFn = Callable[..., Any]


class GpuHealthError(RuntimeError):
    """Raised when GPU health cannot stabilize or the CUDA smoke fails."""

    def __init__(self, message: str, *, evidence: dict[str, Any] | None = None):
        super().__init__(message)
        self.evidence = evidence or {}


@dataclass(frozen=True)
class GpuHealthConfig:
    """Requested topology and validation policy for one cluster."""

    expected_nodes: int
    expected_gpu_nodes: int
    gpu_preset: str
    gpu_platform: str
    driver_mode: str
    nvswitch: bool = False
    stabilization_seconds: int = DEFAULT_STABILIZATION_SECONDS
    poll_seconds: int = DEFAULT_POLL_SECONDS
    timeout_seconds: int = 3600
    cuda_smoke: bool = True
    cuda_smoke_image: str = DEFAULT_CUDA_SMOKE_IMAGE

    @property
    def expected_gpus(self) -> int:
        return self.expected_gpu_nodes * gpus_per_node(self.gpu_preset)

    def validate(self) -> None:
        if self.expected_nodes < 0 or self.expected_gpu_nodes < 0:
            raise ValueError("expected node counts cannot be negative")
        if self.expected_gpu_nodes > self.expected_nodes:
            raise ValueError("expected GPU nodes cannot exceed expected total nodes")
        if self.expected_gpu_nodes and self.expected_gpus <= 0:
            raise ValueError(
                f"GPU preset {self.gpu_preset!r} does not encode a positive GPU count"
            )
        if self.driver_mode not in {"managed-image", "operator"}:
            raise ValueError(
                "GPU health requires an effective driver mode of managed-image or operator"
            )
        if self.stabilization_seconds < 0 or self.poll_seconds <= 0:
            raise ValueError("GPU health stabilization must be >= 0 and polling > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("GPU health timeout must be positive")
        if self.cuda_smoke and not self.cuda_smoke_image.strip():
            raise ValueError("CUDA smoke image cannot be empty when smoke is enabled")


def _run_json(
    capture: CaptureFn,
    kubectl_bin: str,
    kubeconfig_path: Path,
    args: list[str],
) -> dict[str, Any]:
    env = os.environ.copy()
    env["KUBECONFIG"] = str(kubeconfig_path)
    result = capture([kubectl_bin, *args], env=env, check=False)
    if getattr(result, "returncode", 0) != 0:
        detail = str(getattr(result, "stderr", "") or getattr(result, "stdout", ""))
        raise GpuHealthError(
            f"kubectl {' '.join(args)} failed ({getattr(result, 'returncode', '?')}): "
            f"{detail.strip()[-1000:]}"
        )
    try:
        payload = json.loads(getattr(result, "stdout", "") or "{}")
    except json.JSONDecodeError as exc:
        raise GpuHealthError(f"kubectl {' '.join(args)} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise GpuHealthError(f"kubectl {' '.join(args)} returned a non-object")
    return payload


def _ready(node: dict[str, Any]) -> bool:
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in (node.get("status") or {}).get("conditions", [])
        if isinstance(condition, dict)
    )


def _node_name(node: dict[str, Any]) -> str:
    return str((node.get("metadata") or {}).get("name") or "")


def _boot_id(node: dict[str, Any]) -> str:
    return str(((node.get("status") or {}).get("nodeInfo") or {}).get("bootID") or "")


def _allocatable_gpus(node: dict[str, Any]) -> int:
    raw = ((node.get("status") or {}).get("allocatable") or {}).get("nvidia.com/gpu", 0)
    try:
        return int(str(raw or "0"))
    except ValueError:
        return 0


def _canonical_platform(value: str) -> str:
    value = str(value or "").strip().lower()
    return value.removesuffix("-a")


def _is_requested_gpu_node(node: dict[str, Any], platform: str) -> bool:
    labels = (node.get("metadata") or {}).get("labels") or {}
    instance_type = str(labels.get("node.kubernetes.io/instance-type") or "")
    platform_match = bool(
        platform
        and instance_type
        and _canonical_platform(instance_type) == _canonical_platform(platform)
    )
    return platform_match or _allocatable_gpus(node) > 0


def _node_condition_errors(node: dict[str, Any], *, nvswitch: bool) -> list[str]:
    name = _node_name(node) or "<unnamed>"
    errors: list[str] = []
    conditions = (node.get("status") or {}).get("conditions") or []
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        condition_type = str(condition.get("type") or "")
        status = str(condition.get("status") or "")
        if condition_type == "NebiusGPUError" and status == "True":
            detail = str(condition.get("message") or condition.get("reason") or "")
            errors.append(
                f"{name}: NebiusGPUError=True{': ' + detail if detail else ''}"
            )
        if nvswitch and "fabric" in condition_type.lower() and status != "True":
            detail = str(condition.get("reason") or condition.get("message") or status)
            errors.append(f"{name}: {condition_type} is not ready ({detail})")
    return errors


def _fabric_metadata_errors(node: dict[str, Any], *, nvswitch: bool) -> list[str]:
    if not nvswitch:
        return []
    metadata = node.get("metadata") or {}
    observed: list[tuple[str, str]] = []
    for section_name in ("labels", "annotations"):
        section = metadata.get(section_name) or {}
        for key, value in section.items():
            normalized = str(key).lower().replace("_", "-")
            if "fabric" not in normalized or not (
                "state" in normalized or "status" in normalized
            ):
                continue
            observed.append((str(key), str(value)))
    name = _node_name(node) or "<unnamed>"
    return [
        f"{name}: exposed fabric field {key}={value!r} is not Completed/Success"
        for key, value in observed
        if value.strip().lower() not in _FABRIC_SUCCESS
    ]


def _pod_errors(pods: list[dict[str, Any]], namespace: str) -> list[str]:
    if not pods:
        return [f"{namespace}: no component pods found"]
    errors: list[str] = []
    for pod in pods:
        metadata = pod.get("metadata") or {}
        status = pod.get("status") or {}
        name = str(metadata.get("name") or "<unnamed>")
        phase = str(status.get("phase") or "Unknown")
        if metadata.get("deletionTimestamp"):
            errors.append(f"{namespace}/{name}: terminating")
            continue
        if phase == "Succeeded":
            continue
        container_statuses = status.get("containerStatuses") or []
        if (
            phase != "Running"
            or not container_statuses
            or not all(bool(item.get("ready")) for item in container_statuses)
        ):
            waiting = [
                str(
                    ((item.get("state") or {}).get("waiting") or {}).get("reason") or ""
                )
                for item in container_statuses
                if not item.get("ready")
            ]
            detail = ", ".join(filter(None, waiting))
            errors.append(
                f"{namespace}/{name}: phase={phase}"
                + (f" ({detail})" if detail else "")
            )
    return errors


def probe_gpu_health(
    capture: CaptureFn,
    *,
    kubectl_bin: str,
    kubeconfig_path: Path,
    config: GpuHealthConfig,
) -> dict[str, Any]:
    """Capture one Kubernetes health snapshot and its actionable errors."""

    config.validate()
    nodes_payload = _run_json(
        capture, kubectl_bin, kubeconfig_path, ["get", "nodes", "-o", "json"]
    )
    nodes = [item for item in nodes_payload.get("items", []) if isinstance(item, dict)]
    gpu_nodes = [
        node for node in nodes if _is_requested_gpu_node(node, config.gpu_platform)
    ]
    ready_nodes = [node for node in nodes if _ready(node)]
    errors: list[str] = []
    if len(nodes) < config.expected_nodes:
        errors.append(
            f"expected at least {config.expected_nodes} nodes, found {len(nodes)}"
        )
    not_ready = [_node_name(node) or "<unnamed>" for node in nodes if not _ready(node)]
    if not_ready:
        errors.append("nodes are not Ready: " + ", ".join(sorted(not_ready)))
    if len(gpu_nodes) != config.expected_gpu_nodes:
        errors.append(
            f"expected {config.expected_gpu_nodes} GPU nodes for platform "
            f"{config.gpu_platform}, found {len(gpu_nodes)}"
        )
    total_gpus = sum(_allocatable_gpus(node) for node in gpu_nodes)
    if total_gpus != config.expected_gpus:
        errors.append(
            f"expected {config.expected_gpus} nvidia.com/gpu allocatable from "
            f"{config.expected_gpu_nodes}x{gpus_per_node(config.gpu_preset)}, "
            f"found {total_gpus}"
        )
    for node in gpu_nodes:
        errors.extend(_node_condition_errors(node, nvswitch=config.nvswitch))
        errors.extend(_fabric_metadata_errors(node, nvswitch=config.nvswitch))
    missing_boot_ids = [
        _node_name(node) or "<unnamed>" for node in nodes if not _boot_id(node)
    ]
    if missing_boot_ids:
        errors.append(
            "nodes have no observable boot ID: " + ", ".join(missing_boot_ids)
        )

    namespace = (
        "nvidia-device-plugin"
        if config.driver_mode == "managed-image"
        else "gpu-operator"
    )
    pods_payload = _run_json(
        capture,
        kubectl_bin,
        kubeconfig_path,
        ["get", "pods", "-n", namespace, "-o", "json"],
    )
    pods = [item for item in pods_payload.get("items", []) if isinstance(item, dict)]
    errors.extend(_pod_errors(pods, namespace))
    return {
        "observed_at_monotonic": time.monotonic(),
        "expected_nodes": config.expected_nodes,
        "observed_nodes": len(nodes),
        "ready_nodes": len(ready_nodes),
        "expected_gpu_nodes": config.expected_gpu_nodes,
        "gpu_nodes": sorted(_node_name(node) for node in gpu_nodes),
        "expected_gpus": config.expected_gpus,
        "total_gpus": total_gpus,
        "driver_mode": config.driver_mode,
        "component_namespace": namespace,
        "boot_ids": {_node_name(node): _boot_id(node) for node in nodes},
        "errors": errors,
    }


def _write_evidence(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fabric_errors_from_nvidia_smi(output: str) -> list[str]:
    """Return failures only when ``nvidia-smi -q`` exposes fabric state/status."""

    errors: list[str] = []
    blocks = re.findall(
        r"(?ms)^\s*Fabric\s*$\n(?P<body>(?:^\s{4,}.*$\n?){1,24})", output
    )
    for body in blocks:
        state = re.search(r"(?mi)^\s*State\s*:\s*(.+?)\s*$", body)
        status = re.search(r"(?mi)^\s*Status\s*:\s*(.+?)\s*$", body)
        if state and state.group(1).strip().lower() not in _FABRIC_SUCCESS:
            errors.append(f"NVSwitch Fabric State={state.group(1).strip()!r}")
        if status and status.group(1).strip().lower() not in _FABRIC_SUCCESS:
            errors.append(f"NVSwitch Fabric Status={status.group(1).strip()!r}")
    return errors


def _cuda_smoke_on_node(
    capture: CaptureFn,
    *,
    kubectl_bin: str,
    kubeconfig_path: Path,
    node_name: str,
    image: str,
    timeout_seconds: int,
    nvswitch: bool,
    sleep_fn: Callable[[float], None],
    monotonic_fn: Callable[[], float],
) -> dict[str, Any]:
    digest = hashlib.sha256(node_name.encode()).hexdigest()[:10]
    pod_name = f"npa-gpu-health-{digest}"
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": "default",
            "labels": {"app.kubernetes.io/managed-by": "npa-gpu-health"},
        },
        "spec": {
            "restartPolicy": "Never",
            "nodeName": node_name,
            "tolerations": [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }
            ],
            "containers": [
                {
                    "name": "vectoradd",
                    "image": image,
                    "command": ["/bin/bash", "-c"],
                    "args": ["/cuda-samples/vectorAdd && nvidia-smi -q"],
                    "resources": {"limits": {"nvidia.com/gpu": 1}},
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                }
            ],
        },
    }
    env = os.environ.copy()
    env["KUBECONFIG"] = str(kubeconfig_path)
    create = capture(
        [kubectl_bin, "apply", "-f", "-"],
        env=env,
        input_text=json.dumps(manifest),
        check=False,
    )
    if getattr(create, "returncode", 0) != 0:
        detail = str(getattr(create, "stderr", "") or getattr(create, "stdout", ""))
        raise GpuHealthError(
            f"CUDA vectorAdd pod {pod_name} could not be created on {node_name}: "
            f"{detail.strip()[-1000:]}"
        )
    deadline = monotonic_fn() + timeout_seconds
    phase = ""
    failure_detail = ""
    try:
        while monotonic_fn() <= deadline:
            pod = _run_json(
                capture,
                kubectl_bin,
                kubeconfig_path,
                ["get", "pod", pod_name, "-n", "default", "-o", "json"],
            )
            phase = str((pod.get("status") or {}).get("phase") or "")
            if phase in {"Succeeded", "Failed"}:
                statuses = (pod.get("status") or {}).get("containerStatuses") or []
                failure_detail = ", ".join(
                    str(
                        ((item.get("state") or {}).get("terminated") or {}).get(
                            "reason"
                        )
                        or ""
                    )
                    for item in statuses
                    if ((item.get("state") or {}).get("terminated") or {}).get(
                        "exitCode", 0
                    )
                    != 0
                )
                break
            sleep_fn(min(2.0, max(0.0, deadline - monotonic_fn())))
        logs = capture(
            [kubectl_bin, "logs", pod_name, "-n", "default"],
            env=env,
            check=False,
        )
        output = str(getattr(logs, "stdout", "") or "")
        if phase != "Succeeded" or getattr(logs, "returncode", 0) != 0:
            raise GpuHealthError(
                f"CUDA vectorAdd failed on {node_name}: phase={phase or 'timeout'}"
                + (f" ({failure_detail})" if failure_detail else "")
                + (f"; logs: {output[-1000:]}" if output else "")
            )
        if "Test PASSED" not in output:
            raise GpuHealthError(
                f"CUDA vectorAdd on {node_name} exited successfully without "
                "required 'Test PASSED' evidence"
            )
        fabric_errors = _fabric_errors_from_nvidia_smi(output) if nvswitch else []
        if fabric_errors:
            raise GpuHealthError(f"{node_name}: " + "; ".join(fabric_errors))
        return {
            "node": node_name,
            "pod": pod_name,
            "phase": phase,
            "vectoradd": "passed",
            "fabric": "success" if nvswitch and "Fabric" in output else "not-exposed",
        }
    finally:
        capture(
            [
                kubectl_bin,
                "delete",
                "pod",
                pod_name,
                "-n",
                "default",
                "--ignore-not-found=true",
                "--wait=false",
            ],
            env=env,
            check=False,
        )


def validate_gpu_health(
    capture: CaptureFn,
    *,
    kubectl_bin: str,
    kubeconfig_path: Path,
    config: GpuHealthConfig,
    evidence_path: Path | None = None,
    on_status: Callable[[str], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Wait for healthy stable GPU nodes, then run vectorAdd on every GPU node."""

    config.validate()
    deadline = monotonic_fn() + config.timeout_seconds
    stable_since: float | None = None
    baseline_boot_ids: dict[str, str] | None = None
    observations: list[dict[str, Any]] = []
    fatal_error = ""
    stabilized = False
    last_errors: list[str] = []
    final_snapshot: dict[str, Any] = {}

    while monotonic_fn() <= deadline:
        try:
            snapshot = probe_gpu_health(
                capture,
                kubectl_bin=kubectl_bin,
                kubeconfig_path=kubeconfig_path,
                config=config,
            )
        except GpuHealthError as exc:
            snapshot = {"errors": [str(exc)], "boot_ids": {}, "gpu_nodes": []}
        observations.append(snapshot)
        final_snapshot = snapshot
        current_boot_ids = snapshot.get("boot_ids") or {}
        complete_topology = (
            snapshot.get("observed_nodes", 0) >= config.expected_nodes
            and len(snapshot.get("gpu_nodes") or []) == config.expected_gpu_nodes
            and bool(current_boot_ids)
            and all(current_boot_ids.values())
        )
        if complete_topology and baseline_boot_ids is None:
            baseline_boot_ids = dict(current_boot_ids)
        elif complete_topology and baseline_boot_ids != current_boot_ids:
            fatal_error = (
                "node identity/boot IDs changed during GPU stabilization; managed "
                "remediation or a reboot loop is still active"
            )
            break

        last_errors = [str(item) for item in snapshot.get("errors") or []]
        if last_errors:
            stable_since = None
            if on_status:
                on_status("GPU health pending: " + "; ".join(last_errors))
        else:
            now = monotonic_fn()
            if stable_since is None:
                stable_since = now
            stable_for = now - stable_since
            if on_status:
                on_status(
                    f"GPU health stable for {int(stable_for)}/"
                    f"{config.stabilization_seconds}s"
                )
            if stable_for >= config.stabilization_seconds:
                stabilized = True
                break
        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            break
        sleep_fn(min(float(config.poll_seconds), remaining))

    report: dict[str, Any] = {
        "status": "failed",
        "config": asdict(config),
        "observations": observations,
        "final_snapshot": final_snapshot,
        "cuda_smokes": [],
    }
    if fatal_error or last_errors or stable_since is None or not stabilized:
        message = fatal_error or (
            "GPU health did not remain healthy for the requested "
            f"{config.stabilization_seconds}s stabilization interval before timeout"
            + (": " + "; ".join(last_errors) if last_errors else "")
        )
        report["error"] = message
        _write_evidence(evidence_path, report)
        raise GpuHealthError(message, evidence=report)

    try:
        if config.cuda_smoke:
            for node_name in final_snapshot.get("gpu_nodes") or []:
                remaining = max(1, int(deadline - monotonic_fn()))
                report["cuda_smokes"].append(
                    _cuda_smoke_on_node(
                        capture,
                        kubectl_bin=kubectl_bin,
                        kubeconfig_path=kubeconfig_path,
                        node_name=node_name,
                        image=config.cuda_smoke_image,
                        timeout_seconds=remaining,
                        nvswitch=config.nvswitch,
                        sleep_fn=sleep_fn,
                        monotonic_fn=monotonic_fn,
                    )
                )
    except GpuHealthError as exc:
        report["error"] = str(exc)
        _write_evidence(evidence_path, report)
        raise GpuHealthError(str(exc), evidence=report) from exc

    report["status"] = "healthy"
    report["stabilization_seconds"] = config.stabilization_seconds
    _write_evidence(evidence_path, report)
    return report
