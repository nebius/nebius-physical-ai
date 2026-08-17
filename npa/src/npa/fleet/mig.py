"""Supported NVIDIA MIG policy for declarative NPA fleets.

The first supported fleet MIG target is deliberately narrow: true hardware MIG
on the RTX PRO 6000 Blackwell Server Edition.  Keeping the compatibility and
expected-resource contract here gives the spec parser, Terraform renderer, and
live verifier one fail-closed source of truth.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Final

RTX_PRO_6000_PLATFORM: Final = "gpu-rtx6000"
RTX_PRO_6000_PCI_DEVICE_ID: Final = "0x2BB5"
RTX_PRO_6000_PRESET: Final = "1gpu-24vcpu-218gb"
RTX_PRO_6000_BOOT_DISK_GIB: Final = 128
RTX_PRO_6000_VBIOS: Final = frozenset({"98.02.67.00.0A", "98.02.8D.00.01"})
RTX_PRO_6000_MEMORY_MIB: Final = 97887
MIG_KUBERNETES_VERSION: Final = "1.34"

GPU_OPERATOR_VERSION: Final = "v26.3.3"
GPU_DRIVER_VERSION: Final = "580.173.02"
GPU_DEVICE_PLUGIN_VERSION: Final = "v0.19.3"
GPU_GFD_VERSION: Final = "v0.19.3"
GPU_MIG_MANAGER_VERSION: Final = "v0.14.2"
GPU_CONTAINER_TOOLKIT_VERSION: Final = "v1.19.1"
GPU_CUDA_VERSION: Final = "13.0"
_CLEANUP_TIMEOUT_SECONDS: Final = 30.0

SUPPORTED_MIG_STRATEGY: Final = "mixed"
SUPPORTED_MIG_CONFIG: Final = "all-balanced"

# Resource counts advertised by one RTX PRO 6000 after the official
# all-balanced (PCI device 0x2BB5) geometry converges.  Whole-GPU capacity and
# allocatable must both remain zero under the mixed strategy.
RTX_PRO_6000_ALL_BALANCED_RESOURCES: Final[dict[str, int]] = {
    "nvidia.com/gpu": 0,
    "nvidia.com/mig-1g.24gb": 2,
    "nvidia.com/mig-2g.48gb": 1,
}


class MigSpecError(ValueError):
    """Raised when a fleet MIG block is unsupported or internally inconsistent."""


class MigVerificationError(RuntimeError):
    """Raised when live MIG state is unreadable or does not converge."""


@dataclass(frozen=True)
class MigNodeStatus:
    """Sanitized expected-state result for one Kubernetes GPU node."""

    name: str
    product: str
    config: str
    config_state: str
    capacity: dict[str, int]
    allocatable: dict[str, int]
    schedulable: bool
    ready: bool


@dataclass(frozen=True)
class MigHardwareStatus:
    """Hardware identity and GI/CI evidence collected from one driver pod."""

    node: str
    product: str
    pci_device_id: str
    vbios_version: str
    driver_version: str
    memory_mib: int
    mig_current: str
    mig_pending: str
    gpu_uuid: str
    mig_uuids: tuple[str, ...]
    mig_profiles: tuple[str, ...] = ()
    cuda_version: str = ""


@dataclass(frozen=True)
class MigVerificationReport:
    """A serializable, credential-free live MIG readiness report."""

    ready: bool
    nodes: tuple[MigNodeStatus, ...]
    errors: tuple[str, ...]
    operator_state: str
    operator_version: str
    hardware: tuple[MigHardwareStatus, ...] = ()
    cuda_smoke: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "operator_state": self.operator_state,
            "operator_version": self.operator_version,
            "nodes": [
                {
                    "name": node.name,
                    "product": node.product,
                    "config": node.config,
                    "config_state": node.config_state,
                    "capacity": node.capacity,
                    "allocatable": node.allocatable,
                    "schedulable": node.schedulable,
                    "ready": node.ready,
                }
                for node in self.nodes
            ],
            "hardware": [
                {
                    "node": item.node,
                    "product": item.product,
                    "pci_device_id": item.pci_device_id,
                    "vbios_version": item.vbios_version,
                    "driver_version": item.driver_version,
                    "memory_mib": item.memory_mib,
                    "mig_current": item.mig_current,
                    "mig_pending": item.mig_pending,
                    "gpu_uuid": item.gpu_uuid,
                    "mig_uuids": list(item.mig_uuids),
                    "mig_profiles": list(item.mig_profiles),
                    "cuda_version": item.cuda_version,
                }
                for item in self.hardware
            ],
            "cuda_smoke": self.cuda_smoke,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class MigSpec:
    """Hardware MIG configuration for a cluster's GPU node pool.

    Omitting ``mig`` from a fleet spec retains the historical whole-GPU path.
    A present block is enabled by default so the compact form
    ``mig: {strategy: mixed, config: all-balanced}`` is unambiguous.
    Component versions are not user-overridable: NPA pins the compatibility
    tuple above and the Terraform layer asserts the same values.
    """

    enabled: bool = True
    strategy: str = SUPPORTED_MIG_STRATEGY
    config: str = SUPPORTED_MIG_CONFIG

    def validate(
        self,
        *,
        platform: str,
        preset: str,
        gpu_nodes: int,
        capacity_block_group: str,
        disk_size_gib: int,
        k8s_version: str,
    ) -> None:
        """Fail closed unless this is the tested RTX PRO 6000 MIG tuple."""

        if not self.enabled:
            if self.strategy != "none" or self.config:
                raise MigSpecError(
                    "disabled MIG requires strategy 'none' and an empty config"
                )
            return
        if platform != RTX_PRO_6000_PLATFORM:
            raise MigSpecError(
                "hardware MIG is currently supported only for RTX PRO 6000 "
                f"Blackwell Server Edition ({RTX_PRO_6000_PLATFORM}); got {platform!r}"
            )
        if preset != RTX_PRO_6000_PRESET:
            raise MigSpecError(
                "RTX PRO 6000 MIG requires the tested one-GPU preset "
                f"{RTX_PRO_6000_PRESET!r}; got {preset!r}"
            )
        if gpu_nodes <= 0:
            raise MigSpecError(
                "RTX PRO 6000 MIG requires a positive GPU worker count; "
                f"got {gpu_nodes}"
            )
        if gpu_nodes != 2:
            raise MigSpecError(
                f"RTX PRO 6000 MIG requires exactly two GPU workers; got {gpu_nodes}"
            )
        if not capacity_block_group:
            raise MigSpecError(
                "RTX PRO 6000 MIG requires gpu_nodes.capacity_block_group so "
                "reserved-capacity placement is STRICT and cannot fall back to PAYG"
            )
        if disk_size_gib not in {0, RTX_PRO_6000_BOOT_DISK_GIB}:
            raise MigSpecError(
                "RTX PRO 6000 MIG requires the quota-safe 128 GiB GPU boot disk; "
                f"got {disk_size_gib} GiB"
            )
        if self.strategy != SUPPORTED_MIG_STRATEGY:
            raise MigSpecError(
                f"RTX PRO 6000 MIG strategy must be {SUPPORTED_MIG_STRATEGY!r}; "
                f"got {self.strategy!r}"
            )
        if self.config != SUPPORTED_MIG_CONFIG:
            raise MigSpecError(
                f"RTX PRO 6000 MIG config must be {SUPPORTED_MIG_CONFIG!r}; "
                f"got {self.config!r}"
            )
        if k8s_version and k8s_version != MIG_KUBERNETES_VERSION:
            raise MigSpecError(
                "RTX PRO 6000 MIG requires Kubernetes "
                f"{MIG_KUBERNETES_VERSION}; got {k8s_version!r}"
            )

    def expected_resources_per_gpu(self) -> dict[str, int]:
        """Return the exact verifier contract for one physical GPU."""

        if not self.enabled:
            return {"nvidia.com/gpu": 1}
        return dict(RTX_PRO_6000_ALL_BALANCED_RESOURCES)

    def expected_resources_per_node(self, *, gpus_per_node: int) -> dict[str, int]:
        """Return exact kubelet resources for one node of the selected preset."""

        if gpus_per_node <= 0:
            raise MigSpecError("gpus_per_node must be positive")
        return {
            resource: count * gpus_per_node
            for resource, count in self.expected_resources_per_gpu().items()
        }


def mig_spec_from_mapping(data: Any) -> MigSpec | None:
    """Parse an optional ``mig`` mapping without permissive bool coercion."""

    if data is None:
        return None
    if not isinstance(data, dict):
        raise MigSpecError("mig must be a mapping")
    unknown = sorted(set(data) - {"enabled", "strategy", "config"})
    if unknown:
        raise MigSpecError(f"mig contains unsupported field(s): {', '.join(unknown)}")
    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise MigSpecError("mig.enabled must be a boolean")
    default_strategy = SUPPORTED_MIG_STRATEGY if enabled else "none"
    default_config = SUPPORTED_MIG_CONFIG if enabled else ""
    strategy = data.get("strategy", default_strategy)
    config = data.get("config", default_config)
    if not isinstance(strategy, str) or not isinstance(config, str):
        raise MigSpecError("mig.strategy and mig.config must be strings")
    return MigSpec(
        enabled=enabled,
        strategy=strategy.strip().lower(),
        config=config.strip().lower(),
    )


def _quantity(mapping: Any, resource: str) -> int:
    """Parse one Kubernetes integer extended-resource quantity fail-closed."""

    if not isinstance(mapping, dict) or resource not in mapping:
        return 0
    raw = mapping[resource]
    try:
        value = int(str(raw))
    except (TypeError, ValueError) as exc:
        raise MigVerificationError(
            f"resource {resource!r} has non-integer quantity {raw!r}"
        ) from exc
    if value < 0:
        raise MigVerificationError(
            f"resource {resource!r} has negative quantity {value}"
        )
    return value


def _kubectl_env() -> dict[str, str]:
    """Avoid stale ambient IAM tokens shadowing kubeconfig exec credentials."""

    env = os.environ.copy()
    reuse = env.get("NPA_REUSE_IAM_TOKEN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not reuse:
        for name in (
            "NEBIUS_IAM_TOKEN",
            "NPA_NEBIUS_IAM_TOKEN",
            "NEBIUS_IAM_TOKEN_FILE",
        ):
            env.pop(name, None)
    return env


def inspect_mig_state(
    nodes_payload: dict[str, Any],
    cluster_policy_payload: dict[str, Any],
    daemonsets_payload: dict[str, Any],
    operator_deployment_payload: dict[str, Any] | None = None,
    *,
    expected_nodes: int,
) -> MigVerificationReport:
    """Validate one atomic Kubernetes snapshot against the pinned MIG contract.

    Both capacity and allocatable are checked independently. In particular,
    ``nvidia.com/gpu`` capacity 1 with allocatable 0 is a hard failure rather
    than a transient success.
    """

    errors: list[str] = []
    node_statuses: list[MigNodeStatus] = []
    raw_nodes = nodes_payload.get("items")
    if not isinstance(raw_nodes, list):
        raise MigVerificationError("Kubernetes node response has no list 'items'")
    gpu_nodes = [
        node
        for node in raw_nodes
        if isinstance(node, dict)
        and str(
            node.get("metadata", {}).get("labels", {}).get("nvidia.com/gpu.present", "")
        ).lower()
        == "true"
    ]
    if len(gpu_nodes) != expected_nodes:
        errors.append(
            f"expected exactly {expected_nodes} NVIDIA GPU node(s), found {len(gpu_nodes)}"
        )

    expected_resources = RTX_PRO_6000_ALL_BALANCED_RESOURCES
    for raw in sorted(
        gpu_nodes, key=lambda item: str(item.get("metadata", {}).get("name", ""))
    ):
        metadata = raw.get("metadata", {})
        labels = metadata.get("labels", {})
        status = raw.get("status", {})
        spec = raw.get("spec", {})
        name = str(metadata.get("name") or "<unnamed>")
        product = str(labels.get("nvidia.com/gpu.product") or "")
        config = str(labels.get("nvidia.com/mig.config") or "")
        config_state = str(labels.get("nvidia.com/mig.config.state") or "")
        strategy = str(labels.get("nvidia.com/mig.strategy") or "")
        capacity = {
            resource: _quantity(status.get("capacity"), resource)
            for resource in expected_resources
        }
        allocatable = {
            resource: _quantity(status.get("allocatable"), resource)
            for resource in expected_resources
        }
        conditions = status.get("conditions", [])
        ready = any(
            isinstance(condition, dict)
            and condition.get("type") == "Ready"
            and condition.get("status") == "True"
            for condition in conditions
        )
        schedulable = not bool(spec.get("unschedulable", False))
        node_statuses.append(
            MigNodeStatus(
                name=name,
                product=product,
                config=config,
                config_state=config_state,
                capacity=capacity,
                allocatable=allocatable,
                schedulable=schedulable,
                ready=ready,
            )
        )
        if not ready:
            errors.append(f"node {name}: Kubernetes Ready condition is not True")
        if not schedulable:
            errors.append(f"node {name}: node is cordoned/unschedulable")
        taints = spec.get("taints") or []
        if not isinstance(taints, list):
            errors.append(f"node {name}: Kubernetes taints are malformed")
        elif any(
            isinstance(taint, dict)
            and taint.get("key") == "nvidia.com/gpu"
            and taint.get("value") == "mig-not-ready"
            and taint.get("effect") == "NoSchedule"
            for taint in taints
        ):
            errors.append(
                f"node {name}: stale nvidia.com/gpu=mig-not-ready:NoSchedule taint "
                "blocks MIG workloads"
            )
        expected_product = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
        if product != expected_product:
            errors.append(
                f"node {name}: expected product {expected_product!r}, got {product!r}"
            )
        if strategy != SUPPORTED_MIG_STRATEGY:
            errors.append(
                f"node {name}: nvidia.com/mig.strategy={strategy!r}, expected 'mixed'"
            )
        if config != SUPPORTED_MIG_CONFIG or config_state != "success":
            errors.append(
                f"node {name}: MIG config/state is {config!r}/{config_state!r}, "
                "expected 'all-balanced'/'success'"
            )
        for resource, expected in expected_resources.items():
            cap = capacity[resource]
            alloc = allocatable[resource]
            if cap != expected or alloc != expected:
                errors.append(
                    f"node {name}: {resource} capacity/allocatable={cap}/{alloc}, "
                    f"expected {expected}/{expected}"
                )

    policies = cluster_policy_payload.get("items")
    if not isinstance(policies, list) or len(policies) != 1:
        errors.append("expected exactly one NVIDIA ClusterPolicy")
        policy: dict[str, Any] = {}
    else:
        policy = policies[0] if isinstance(policies[0], dict) else {}
    policy_spec = policy.get("spec", {})
    policy_status = policy.get("status", {})
    policy_labels = policy.get("metadata", {}).get("labels", {})
    operator_state = str(policy_status.get("state") or "")
    operator_version = str(policy_labels.get("app.kubernetes.io/version") or "")
    if operator_state != "ready":
        errors.append(
            f"NVIDIA ClusterPolicy state is {operator_state!r}, expected 'ready'"
        )
    pin_checks = {
        "GPU Operator": (operator_version, GPU_OPERATOR_VERSION),
        "driver": (
            str(policy_spec.get("driver", {}).get("version") or ""),
            GPU_DRIVER_VERSION,
        ),
        "device plugin": (
            str(policy_spec.get("devicePlugin", {}).get("version") or ""),
            GPU_DEVICE_PLUGIN_VERSION,
        ),
        "GFD": (str(policy_spec.get("gfd", {}).get("version") or ""), GPU_GFD_VERSION),
        "MIG Manager": (
            str(policy_spec.get("migManager", {}).get("version") or ""),
            GPU_MIG_MANAGER_VERSION,
        ),
    }
    for component, (actual, expected) in pin_checks.items():
        if actual != expected:
            errors.append(f"{component} version is {actual!r}, expected {expected!r}")
    if str(policy_spec.get("mig", {}).get("strategy") or "") != "mixed":
        errors.append("NVIDIA ClusterPolicy MIG strategy is not 'mixed'")
    if policy_spec.get("cdi", {}).get("enabled") is not True:
        errors.append("NVIDIA ClusterPolicy CDI is not enabled")

    required_daemonsets = {
        "nvidia-driver-daemonset": GPU_DRIVER_VERSION,
        "nvidia-mig-manager": GPU_MIG_MANAGER_VERSION,
        "gpu-feature-discovery": GPU_GFD_VERSION,
        "nvidia-device-plugin-daemonset": GPU_DEVICE_PLUGIN_VERSION,
        "nvidia-container-toolkit-daemonset": GPU_CONTAINER_TOOLKIT_VERSION,
    }
    ds_items = daemonsets_payload.get("items")
    if not isinstance(ds_items, list):
        raise MigVerificationError("Kubernetes DaemonSet response has no list 'items'")
    daemonsets = {
        str(ds.get("metadata", {}).get("name") or ""): ds
        for ds in ds_items
        if isinstance(ds, dict)
    }
    for name, version in required_daemonsets.items():
        ds = daemonsets.get(name)
        if not ds:
            errors.append(f"required DaemonSet {name!r} is absent")
            continue
        metadata = ds.get("metadata", {})
        status = ds.get("status", {})
        generation = int(metadata.get("generation") or 0)
        observed = int(status.get("observedGeneration") or 0)
        desired = int(status.get("desiredNumberScheduled") or 0)
        current = int(status.get("currentNumberScheduled") or 0)
        updated = int(status.get("updatedNumberScheduled") or 0)
        ready = int(status.get("numberReady") or 0)
        available = int(status.get("numberAvailable") or 0)
        if generation <= 0 or observed != generation:
            errors.append(
                f"DaemonSet {name}: generation/observed={generation}/{observed}, "
                "expected an observed current generation"
            )
        if (
            desired != expected_nodes
            or current != expected_nodes
            or updated != expected_nodes
            or ready != expected_nodes
            or available != expected_nodes
        ):
            errors.append(
                f"DaemonSet {name}: desired/current/updated/ready/available="
                f"{desired}/{current}/{updated}/{ready}/{available}, expected "
                f"{expected_nodes}/{expected_nodes}/{expected_nodes}/"
                f"{expected_nodes}/{expected_nodes}"
            )
        images = [
            str(container.get("image") or "")
            for container in ds.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
            if isinstance(container, dict)
        ]
        if not any(version in image for image in images):
            errors.append(
                f"DaemonSet {name}: no container image contains pinned version {version!r}"
            )

    if operator_deployment_payload is not None:
        metadata = operator_deployment_payload.get("metadata", {})
        spec = operator_deployment_payload.get("spec", {})
        status = operator_deployment_payload.get("status", {})
        generation = int(metadata.get("generation") or 0)
        observed = int(status.get("observedGeneration") or 0)
        desired = int(spec.get("replicas") or 1)
        updated = int(status.get("updatedReplicas") or 0)
        ready = int(status.get("readyReplicas") or 0)
        available = int(status.get("availableReplicas") or 0)
        if generation <= 0 or observed != generation:
            errors.append(
                "GPU Operator Deployment generation has not been observed: "
                f"generation/observed={generation}/{observed}"
            )
        if desired != 1 or updated != 1 or ready != 1 or available != 1:
            errors.append(
                "GPU Operator Deployment desired/updated/ready/available="
                f"{desired}/{updated}/{ready}/{available}, expected 1/1/1/1"
            )

    return MigVerificationReport(
        ready=not errors,
        nodes=tuple(node_statuses),
        errors=tuple(errors),
        operator_state=operator_state,
        operator_version=operator_version,
    )


def _kubectl_json(
    kubectl_bin: str,
    kubeconfig: Path,
    args: list[str],
    *,
    timeout_seconds: float | None = None,
    allow_not_found: bool = False,
) -> dict[str, Any]:
    command = [kubectl_bin, "--kubeconfig", str(kubeconfig), *args, "-o", "json"]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=_kubectl_env(),
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MigVerificationError(
            f"could not execute kubectl for {' '.join(args)}: {type(exc).__name__}"
        ) from exc
    if result.returncode != 0:
        if allow_not_found and "notfound" in result.stderr.replace(" ", "").lower():
            return {}
        # Provider stderr can include endpoints or auth details; retain only the
        # exit status and operation, never copy it into diagnostics.
        raise MigVerificationError(
            f"kubectl {' '.join(args)} failed with exit status {result.returncode}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MigVerificationError(
            f"kubectl {' '.join(args)} returned invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise MigVerificationError(
            f"kubectl {' '.join(args)} returned a non-object JSON payload"
        )
    return payload


def _kubectl_text(
    kubectl_bin: str,
    kubeconfig: Path,
    args: list[str],
    *,
    timeout_seconds: float | None = None,
) -> str:
    command = [kubectl_bin, "--kubeconfig", str(kubeconfig), *args]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=_kubectl_env(),
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MigVerificationError(
            f"could not execute kubectl for {' '.join(args[:4])}: {type(exc).__name__}"
        ) from exc
    if result.returncode != 0:
        raise MigVerificationError(
            f"kubectl {' '.join(args[:4])} failed with exit status {result.returncode}"
        )
    return result.stdout


def _remaining_timeout(
    deadline: float | None, monotonic_fn: Callable[[], float]
) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - monotonic_fn()
    if remaining <= 0:
        raise MigVerificationError("MIG verification deadline expired")
    return max(0.001, remaining)


def _collect_hardware(
    kubectl_bin: str,
    kubeconfig: Path,
    *,
    expected_nodes: int,
    timeout_seconds: float | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> tuple[tuple[MigHardwareStatus, ...], tuple[str, ...]]:
    deadline = monotonic_fn() + timeout_seconds if timeout_seconds is not None else None
    pods_payload = _kubectl_json(
        kubectl_bin,
        kubeconfig,
        [
            "get",
            "pods",
            "-n",
            "gpu-operator",
            "-l",
            "app=nvidia-driver-daemonset",
        ],
        timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
    )
    pods = pods_payload.get("items")
    if not isinstance(pods, list):
        raise MigVerificationError("driver Pod response has no list 'items'")
    running = [
        pod
        for pod in pods
        if isinstance(pod, dict)
        and pod.get("status", {}).get("phase") == "Running"
        and pod.get("spec", {}).get("nodeName")
    ]
    errors: list[str] = []
    if len(running) != expected_nodes:
        errors.append(
            f"expected {expected_nodes} running NVIDIA driver pod(s), found {len(running)}"
        )
    hardware: list[MigHardwareStatus] = []
    all_mig_uuids: list[str] = []
    query = (
        "name,pci.device_id,vbios_version,driver_version,memory.total,"
        "mig.mode.current,mig.mode.pending,uuid"
    )
    for pod in sorted(
        running, key=lambda item: str(item.get("spec", {}).get("nodeName", ""))
    ):
        pod_name = str(pod.get("metadata", {}).get("name") or "")
        node = str(pod.get("spec", {}).get("nodeName") or "")
        try:
            output = _kubectl_text(
                kubectl_bin,
                kubeconfig,
                [
                    "exec",
                    "-n",
                    "gpu-operator",
                    pod_name,
                    "-c",
                    "nvidia-driver-ctr",
                    "--",
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
            ).strip()
        except MigVerificationError:
            errors.append(
                f"node {node}: driver pod hardware inspection is temporarily unavailable"
            )
            continue
        rows = [row.strip() for row in output.splitlines() if row.strip()]
        if len(rows) != 1:
            errors.append(
                f"node {node}: expected one physical GPU query row, got {len(rows)}"
            )
            continue
        fields = [field.strip() for field in rows[0].split(",")]
        if len(fields) != 8:
            errors.append(f"node {node}: unexpected nvidia-smi identity field count")
            continue
        product, pci, vbios, driver, memory, mig_current, mig_pending, gpu_uuid = fields
        try:
            memory_mib = int(memory)
        except ValueError:
            errors.append(f"node {node}: invalid GPU memory quantity {memory!r}")
            continue
        try:
            list_output = _kubectl_text(
                kubectl_bin,
                kubeconfig,
                [
                    "exec",
                    "-n",
                    "gpu-operator",
                    pod_name,
                    "-c",
                    "nvidia-driver-ctr",
                    "--",
                    "nvidia-smi",
                    "-L",
                ],
                timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
            )
        except MigVerificationError:
            errors.append(
                f"node {node}: MIG device enumeration is temporarily unavailable"
            )
            continue
        mig_uuids = tuple(sorted(set(re.findall(r"MIG-[0-9a-fA-F-]+", list_output))))
        mig_profiles = tuple(
            sorted(re.findall(r"MIG\s+([0-9]+g\.[0-9]+gb)\s+Device", list_output))
        )
        try:
            summary_output = _kubectl_text(
                kubectl_bin,
                kubeconfig,
                [
                    "exec",
                    "-n",
                    "gpu-operator",
                    pod_name,
                    "-c",
                    "nvidia-driver-ctr",
                    "--",
                    "nvidia-smi",
                ],
                timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
            )
        except MigVerificationError:
            errors.append(
                f"node {node}: CUDA compatibility inspection is temporarily unavailable"
            )
            continue
        cuda_match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", summary_output)
        cuda_version = cuda_match.group(1) if cuda_match else ""
        all_mig_uuids.extend(mig_uuids)
        item = MigHardwareStatus(
            node=node,
            product=product,
            pci_device_id=pci,
            vbios_version=vbios,
            driver_version=driver,
            memory_mib=memory_mib,
            mig_current=mig_current,
            mig_pending=mig_pending,
            gpu_uuid=gpu_uuid,
            mig_uuids=mig_uuids,
            mig_profiles=mig_profiles,
            cuda_version=cuda_version,
        )
        hardware.append(item)
        expected_product = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        if product != expected_product:
            errors.append(f"node {node}: hardware product is {product!r}")
        if not pci.upper().startswith("0X2BB5"):
            errors.append(f"node {node}: PCI device is {pci!r}, expected 0x2BB5")
        if vbios not in RTX_PRO_6000_VBIOS:
            errors.append(
                f"node {node}: unsupported vBIOS {vbios!r}; expected one of "
                f"{sorted(RTX_PRO_6000_VBIOS)!r}"
            )
        if driver != GPU_DRIVER_VERSION:
            errors.append(
                f"node {node}: driver is {driver!r}, expected {GPU_DRIVER_VERSION!r}"
            )
        if memory_mib != RTX_PRO_6000_MEMORY_MIB:
            errors.append(
                f"node {node}: memory is {memory_mib} MiB, expected {RTX_PRO_6000_MEMORY_MIB} MiB"
            )
        if mig_current != "Enabled" or mig_pending != "Enabled":
            errors.append(
                f"node {node}: MIG current/pending={mig_current!r}/{mig_pending!r}, expected Enabled/Enabled"
            )
        if len(mig_uuids) != 3:
            errors.append(
                f"node {node}: found {len(mig_uuids)} MIG UUID(s), expected 3"
            )
        expected_profiles = ("1g.24gb", "1g.24gb", "2g.48gb")
        if mig_profiles != expected_profiles:
            errors.append(
                f"node {node}: MIG GI/CI profiles are {mig_profiles!r}, "
                f"expected {expected_profiles!r}"
            )
        if cuda_version != GPU_CUDA_VERSION:
            errors.append(
                f"node {node}: CUDA compatibility is {cuda_version!r}, "
                f"expected {GPU_CUDA_VERSION!r}"
            )
    if len(all_mig_uuids) != len(set(all_mig_uuids)):
        errors.append("MIG UUIDs are not distinct across GPU nodes")
    return tuple(hardware), tuple(errors)


def verify_mig_cluster(
    *,
    kubectl_bin: str,
    kubeconfig: Path,
    expected_nodes: int,
    timeout_seconds: float | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> MigVerificationReport:
    """Collect and validate one live MIG control-plane snapshot."""

    deadline = monotonic_fn() + timeout_seconds if timeout_seconds is not None else None
    report = inspect_mig_state(
        _kubectl_json(
            kubectl_bin,
            kubeconfig,
            ["get", "nodes"],
            timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
        ),
        _kubectl_json(
            kubectl_bin,
            kubeconfig,
            ["get", "clusterpolicies.nvidia.com"],
            timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
        ),
        _kubectl_json(
            kubectl_bin,
            kubeconfig,
            ["get", "daemonsets", "-n", "gpu-operator"],
            timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
        ),
        _kubectl_json(
            kubectl_bin,
            kubeconfig,
            ["get", "deployment", "gpu-operator", "-n", "gpu-operator"],
            timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
        ),
        expected_nodes=expected_nodes,
    )
    hardware, hardware_errors = _collect_hardware(
        kubectl_bin,
        kubeconfig,
        expected_nodes=expected_nodes,
        timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
        monotonic_fn=monotonic_fn,
    )
    errors = (*report.errors, *hardware_errors)
    return MigVerificationReport(
        ready=not errors,
        nodes=report.nodes,
        errors=errors,
        operator_state=report.operator_state,
        operator_version=report.operator_version,
        hardware=hardware,
    )


def _restart_discovery_operands(
    kubectl_bin: str,
    kubeconfig: Path,
    *,
    deadline: float | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> None:
    """Reconcile GFD before device-plugin registration, never simultaneously."""

    base = [kubectl_bin, "--kubeconfig", str(kubeconfig), "rollout"]
    for operation, daemonset in (
        ("restart", "daemonset/gpu-feature-discovery"),
        ("status", "daemonset/gpu-feature-discovery"),
        ("restart", "daemonset/nvidia-device-plugin-daemonset"),
        ("status", "daemonset/nvidia-device-plugin-daemonset"),
    ):
        try:
            result = subprocess.run(
                [*base, operation, daemonset, "-n", "gpu-operator"],
                check=False,
                capture_output=True,
                text=True,
                env=_kubectl_env(),
                timeout=_remaining_timeout(deadline, monotonic_fn),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MigVerificationError(
                f"could not {operation} NVIDIA "
                f"{daemonset.removeprefix('daemonset/')} before the MIG "
                f"deployment health deadline ({type(exc).__name__})"
            ) from exc
        if result.returncode != 0:
            raise MigVerificationError(
                f"could not {operation} NVIDIA {daemonset.removeprefix('daemonset/')} "
                "during ordered stale-resource reconciliation "
                f"(kubectl exit status {result.returncode})"
            )


def _reconcile_stale_mig_taints(
    kubectl_bin: str,
    kubeconfig: Path,
    *,
    deadline: float | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> int:
    """Remove only the obsolete MIG-readiness taint from successful GPU nodes."""

    nodes = _kubectl_json(
        kubectl_bin,
        kubeconfig,
        ["get", "nodes"],
        timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
    )
    removed = 0
    for node in nodes.get("items", []):
        if not isinstance(node, dict):
            continue
        metadata = node.get("metadata", {})
        labels = metadata.get("labels", {})
        if (
            str(labels.get("nvidia.com/gpu.present", "")).lower() != "true"
            or labels.get("nvidia.com/mig.config") != SUPPORTED_MIG_CONFIG
            or labels.get("nvidia.com/mig.config.state") != "success"
        ):
            continue
        name = str(metadata.get("name") or "")
        taints = node.get("spec", {}).get("taints") or []
        if not name or not isinstance(taints, list):
            continue
        indexes = [
            index
            for index, taint in enumerate(taints)
            if isinstance(taint, dict)
            and taint.get("key") == "nvidia.com/gpu"
            and taint.get("value") == "mig-not-ready"
            and taint.get("effect") == "NoSchedule"
        ]
        for index in reversed(indexes):
            patch = json.dumps(
                [
                    {
                        "op": "test",
                        "path": f"/spec/taints/{index}/key",
                        "value": "nvidia.com/gpu",
                    },
                    {
                        "op": "test",
                        "path": f"/spec/taints/{index}/value",
                        "value": "mig-not-ready",
                    },
                    {
                        "op": "test",
                        "path": f"/spec/taints/{index}/effect",
                        "value": "NoSchedule",
                    },
                    {"op": "remove", "path": f"/spec/taints/{index}"},
                ],
                separators=(",", ":"),
            )
            try:
                result = subprocess.run(
                    [
                        kubectl_bin,
                        "--kubeconfig",
                        str(kubeconfig),
                        "patch",
                        "node",
                        name,
                        "--type=json",
                        "-p",
                        patch,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=_kubectl_env(),
                    timeout=_remaining_timeout(deadline, monotonic_fn),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise MigVerificationError(
                    f"could not remove the stale MIG readiness taint from node "
                    f"{name} before the deployment health deadline "
                    f"({type(exc).__name__})"
                ) from exc
            if result.returncode != 0:
                raise MigVerificationError(
                    f"could not remove the stale MIG readiness taint from node {name} "
                    f"(kubectl exit status {result.returncode})"
                )
            removed += 1
    return removed


def _active_gpu_workloads(pods_payload: dict[str, Any]) -> tuple[str, ...]:
    """Return running non-Operator pods that explicitly hold NVIDIA resources."""

    items = pods_payload.get("items")
    if not isinstance(items, list):
        raise MigVerificationError("Kubernetes Pod response has no list 'items'")
    active: list[str] = []
    for pod in items:
        if not isinstance(pod, dict) or pod.get("status", {}).get("phase") != "Running":
            continue
        metadata = pod.get("metadata", {})
        namespace = str(metadata.get("namespace") or "default")
        if namespace == "gpu-operator":
            continue
        containers = [
            *pod.get("spec", {}).get("initContainers", []),
            *pod.get("spec", {}).get("containers", []),
            *pod.get("spec", {}).get("ephemeralContainers", []),
        ]
        holds_gpu = any(
            any(
                str(resource).startswith("nvidia.com/")
                and _quantity(values, resource) > 0
                for resource in values
            )
            for container in containers
            if isinstance(container, dict)
            for values in (
                container.get("resources", {}).get("requests", {}),
                container.get("resources", {}).get("limits", {}),
            )
            if isinstance(values, dict)
        )
        if holds_gpu:
            active.append(f"{namespace}/{metadata.get('name') or '<unnamed>'}")
    return tuple(sorted(set(active)))


def _reconcile_ondelete_driver(
    kubectl_bin: str,
    kubeconfig: Path,
    *,
    deadline: float,
    sleep_fn: Callable[[float], None],
    monotonic_fn: Callable[[], float],
) -> None:
    """Replace OnDelete driver pods one node at a time after a workload-free gate."""

    all_pods = _kubectl_json(
        kubectl_bin,
        kubeconfig,
        ["get", "pods", "-A"],
        timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
    )
    active = _active_gpu_workloads(all_pods)
    if active:
        raise MigVerificationError(
            "NVIDIA driver OnDelete replacement is blocked by active GPU workload(s): "
            + ", ".join(active)
            + "; delete those workloads explicitly, then rerun fleet deploy or "
            "fleet verify-mig --wait --reconcile"
        )
    driver_pods = [
        pod
        for pod in all_pods.get("items", [])
        if isinstance(pod, dict)
        and pod.get("metadata", {}).get("namespace") == "gpu-operator"
        and pod.get("metadata", {}).get("labels", {}).get("app")
        == "nvidia-driver-daemonset"
        and pod.get("spec", {}).get("nodeName")
    ]
    if not driver_pods:
        raise MigVerificationError(
            "NVIDIA driver DaemonSet update is pending but no driver pods were found"
        )
    for pod in sorted(driver_pods, key=lambda item: item["spec"]["nodeName"]):
        metadata = pod.get("metadata", {})
        old_name = str(metadata.get("name") or "")
        old_uid = str(metadata.get("uid") or "")
        node = str(pod.get("spec", {}).get("nodeName") or "")
        node_payload = _kubectl_json(
            kubectl_bin,
            kubeconfig,
            ["get", "node", node],
            timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
        )
        cordoned_here = not bool(node_payload.get("spec", {}).get("unschedulable"))
        primary_error: BaseException | None = None
        try:
            if cordoned_here:
                try:
                    cordon = subprocess.run(
                        [
                            kubectl_bin,
                            "--kubeconfig",
                            str(kubeconfig),
                            "cordon",
                            node,
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        env=_kubectl_env(),
                        timeout=_remaining_timeout(deadline, monotonic_fn),
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    raise MigVerificationError(
                        f"could not cordon node {node} before the MIG deployment "
                        f"health deadline ({type(exc).__name__})"
                    ) from exc
                if cordon.returncode != 0:
                    raise MigVerificationError(
                        f"could not cordon node {node} before NVIDIA driver "
                        f"replacement (kubectl exit status {cordon.returncode})"
                    )
            # Close the scheduling race between the global workload check and
            # the cordon. NPA never deletes an application workload implicitly.
            active_after_cordon = _active_gpu_workloads(
                _kubectl_json(
                    kubectl_bin,
                    kubeconfig,
                    ["get", "pods", "-A"],
                    timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
                )
            )
            if active_after_cordon:
                raise MigVerificationError(
                    "NVIDIA driver OnDelete replacement is blocked by active GPU "
                    "workload(s): "
                    + ", ".join(active_after_cordon)
                    + "; delete those workloads explicitly, then rerun fleet deploy "
                    "or fleet verify-mig --wait --reconcile"
                )
            _replace_driver_pod(
                kubectl_bin=kubectl_bin,
                kubeconfig=kubeconfig,
                pod_name=old_name,
                pod_uid=old_uid,
                node=node,
                deadline=deadline,
                sleep_fn=sleep_fn,
                monotonic_fn=monotonic_fn,
            )
        except BaseException as exc:
            primary_error = exc
        uncordon = None
        uncordon_error = ""
        if cordoned_here:
            try:
                uncordon = subprocess.run(
                    [kubectl_bin, "--kubeconfig", str(kubeconfig), "uncordon", node],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=_kubectl_env(),
                    timeout=_CLEANUP_TIMEOUT_SECONDS,
                )
                if uncordon.returncode != 0:
                    uncordon_error = f"kubectl exit status {uncordon.returncode}"
            except (OSError, subprocess.SubprocessError) as exc:
                uncordon_error = type(exc).__name__
        if primary_error is not None:
            if uncordon_error:
                raise MigVerificationError(
                    f"{primary_error}; additionally could not uncordon node {node} "
                    f"({uncordon_error})"
                ) from primary_error
            raise primary_error
        if uncordon_error:
            raise MigVerificationError(
                f"NVIDIA driver replacement completed but node {node} could not be "
                f"uncordoned ({uncordon_error})"
            )


def _replace_driver_pod(
    *,
    kubectl_bin: str,
    kubeconfig: Path,
    pod_name: str,
    pod_uid: str,
    node: str,
    deadline: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> None:
    """Delete one OnDelete driver pod and wait for its ready replacement."""

    if monotonic_fn() >= deadline:
        raise MigVerificationError(
            f"MIG deployment health deadline expired before driver pod {pod_name} "
            f"on node {node} could be replaced"
        )
    command = [
        kubectl_bin,
        "--kubeconfig",
        str(kubeconfig),
        "delete",
        "pod",
        pod_name,
        "-n",
        "gpu-operator",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=_kubectl_env(),
            timeout=max(0.001, deadline - monotonic_fn()),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MigVerificationError(
            f"could not delete NVIDIA driver pod {pod_name} on node {node} before "
            f"the MIG deployment health deadline ({type(exc).__name__})"
        ) from exc
    if result.returncode != 0:
        raise MigVerificationError(
            f"could not replace NVIDIA driver pod on node {node} "
            f"(kubectl exit status {result.returncode})"
        )
    while monotonic_fn() < deadline:
        try:
            replacement_payload = _kubectl_json(
                kubectl_bin,
                kubeconfig,
                [
                    "get",
                    "pods",
                    "-n",
                    "gpu-operator",
                    "-l",
                    "app=nvidia-driver-daemonset",
                ],
                timeout_seconds=max(0.001, deadline - monotonic_fn()),
            )
        except MigVerificationError as exc:
            if monotonic_fn() >= deadline:
                raise MigVerificationError(
                    "timed out querying the replacement NVIDIA driver pod on node "
                    f"{node} before the MIG deployment health deadline"
                ) from exc
            raise
        replacement = next(
            (
                candidate
                for candidate in replacement_payload.get("items", [])
                if isinstance(candidate, dict)
                and candidate.get("spec", {}).get("nodeName") == node
                and str(candidate.get("metadata", {}).get("uid") or "") != pod_uid
                and candidate.get("status", {}).get("phase") == "Running"
                and candidate.get("status", {}).get("containerStatuses")
                and all(
                    bool(status.get("ready"))
                    for status in candidate["status"]["containerStatuses"]
                    if isinstance(status, dict)
                )
            ),
            None,
        )
        if replacement is not None:
            return
        sleep_fn(min(10.0, max(0.0, deadline - monotonic_fn())))
    raise MigVerificationError(
        "timed out waiting for the replacement NVIDIA driver pod on node "
        f"{node} to become Running and ready before the MIG deployment health "
        "deadline; the node will be uncordoned and the next deploy can resume "
        "reconciliation"
    )


def _ensure_device_plugin_mig_gate(
    kubectl_bin: str,
    kubeconfig: Path,
    *,
    deadline: float | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> bool:
    """Keep the device plugin off a node until MIG Manager reports success.

    Without this affinity the mixed-strategy plugin can briefly register a whole
    GPU after a Blackwell driver reload, before MIG Manager has recreated the
    GI/CI geometry. Kubelet retains that obsolete capacity even after the plugin
    switches to MIG resources. The Operator preserves this pod-spec field when
    reconciling its DaemonSet.

    Kubernetes strategic merge replaces ``nodeSelectorTerms`` because that list
    has no merge key. To preserve every Operator/platform placement constraint,
    NPA reads the current terms and AND-appends the MIG-success expression to
    each one. Terms are OR-ed, so adding a separate term would bypass the gate.
    """

    try:
        daemonset = _kubectl_json(
            kubectl_bin,
            kubeconfig,
            ["get", "daemonset/nvidia-device-plugin-daemonset", "-n", "gpu-operator"],
            timeout_seconds=_remaining_timeout(deadline, monotonic_fn),
            allow_not_found=True,
        )
    except MigVerificationError as exc:
        if "notfound" in str(exc).replace(" ", "").lower():
            return False
        raise
    if not daemonset:
        return False
    resource_version = str(daemonset.get("metadata", {}).get("resourceVersion") or "")
    if not resource_version:
        raise MigVerificationError(
            "NVIDIA device plugin DaemonSet has no resourceVersion; refusing an "
            "unconditional node-affinity replacement"
        )
    required = {
        "key": "nvidia.com/mig.config.state",
        "operator": "In",
        "values": ["success"],
    }
    affinity = (
        daemonset.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("affinity", {})
    )
    existing_terms = (
        affinity.get("nodeAffinity", {})
        .get("requiredDuringSchedulingIgnoredDuringExecution", {})
        .get("nodeSelectorTerms", [])
    )
    if not isinstance(existing_terms, list):
        raise MigVerificationError(
            "NVIDIA device plugin node affinity has non-list nodeSelectorTerms"
        )
    terms = existing_terms or [{"matchExpressions": []}]
    gated_terms: list[dict[str, Any]] = []
    for raw_term in terms:
        if not isinstance(raw_term, dict):
            raise MigVerificationError(
                "NVIDIA device plugin node affinity contains an invalid selector term"
            )
        term = dict(raw_term)
        expressions = term.get("matchExpressions", [])
        if not isinstance(expressions, list):
            raise MigVerificationError(
                "NVIDIA device plugin node affinity contains invalid matchExpressions"
            )
        term["matchExpressions"] = [
            expression
            for expression in expressions
            if not (
                isinstance(expression, dict)
                and expression.get("key") == required["key"]
            )
        ] + [required]
        gated_terms.append(term)

    patch = json.dumps(
        {
            # nodeSelectorTerms has no strategic-merge key, so this patch must
            # replace the list reconstructed above. The resourceVersion makes
            # that replacement optimistic: a concurrent Operator/user update
            # conflicts instead of being overwritten.
            "metadata": {"resourceVersion": resource_version},
            "spec": {
                "template": {
                    "spec": {
                        "affinity": {
                            "nodeAffinity": {
                                "requiredDuringSchedulingIgnoredDuringExecution": {
                                    "nodeSelectorTerms": gated_terms
                                }
                            }
                        }
                    }
                }
            },
        },
        separators=(",", ":"),
    )
    command = [
        kubectl_bin,
        "--kubeconfig",
        str(kubeconfig),
        "patch",
        "daemonset/nvidia-device-plugin-daemonset",
        "-n",
        "gpu-operator",
        "--type=strategic",
        "-p",
        patch,
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=_kubectl_env(),
            timeout=_remaining_timeout(deadline, monotonic_fn),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MigVerificationError(
            "could not install the MIG-success scheduling gate on the NVIDIA "
            f"device plugin ({type(exc).__name__})"
        ) from exc
    if result.returncode == 0:
        return True
    raise MigVerificationError(
        "could not install the MIG-success scheduling gate on the NVIDIA device "
        f"plugin (kubectl exit status {result.returncode})"
    )


def _cleanup_mig_cuda_smoke_pod(
    *,
    kubectl_bin: str,
    kubeconfig: Path,
    pod_name: str,
    deadline: float,
    monotonic_fn: Callable[[], float],
) -> str:
    """Delete a smoke pod within the caller's deadline; return a safe error."""

    cleanup_timeout = deadline - monotonic_fn()
    if cleanup_timeout <= 0:
        return "deployment health deadline expired before pod deletion"
    try:
        cleanup = subprocess.run(
            [
                kubectl_bin,
                "--kubeconfig",
                str(kubeconfig),
                "delete",
                "pod",
                pod_name,
                "-n",
                "default",
                "--ignore-not-found=true",
                "--wait=true",
                f"--timeout={max(1, int(cleanup_timeout))}s",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_kubectl_env(),
            timeout=max(0.001, cleanup_timeout),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return type(exc).__name__
    if cleanup.returncode != 0:
        return f"kubectl exit status {cleanup.returncode}"
    return ""


def _run_mig_cuda_smoke(
    *,
    kubectl_bin: str,
    kubeconfig: Path,
    image: str,
    deadline: float,
    sleep_fn: Callable[[float], None],
    monotonic_fn: Callable[[], float],
) -> dict[str, Any]:
    """Run one representative CUDA workload on an advertised hardware MIG slice."""

    now = monotonic_fn()
    remaining = deadline - now
    cleanup_reserve = min(_CLEANUP_TIMEOUT_SECONDS, max(1.0, remaining / 10.0))
    work_deadline = deadline - cleanup_reserve
    if now >= work_deadline:
        raise MigVerificationError(
            "MIG readiness converged but too little time remained before the "
            "deployment health deadline to run and clean up the required CUDA smoke"
        )
    resource = next(
        resource
        for resource, count in RTX_PRO_6000_ALL_BALANCED_RESOURCES.items()
        if resource.startswith("nvidia.com/mig-") and count > 0
    )
    pod_name = f"npa-mig-cuda-smoke-{uuid.uuid4().hex[:10]}"
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": "default",
            "labels": {"app.kubernetes.io/managed-by": "npa-fleet-mig"},
        },
        "spec": {
            "restartPolicy": "Never",
            "runtimeClassName": "nvidia",
            "activeDeadlineSeconds": max(1, int(work_deadline - now)),
            "terminationGracePeriodSeconds": 5,
            "containers": [
                {
                    "name": "vectoradd",
                    "image": image,
                    "command": ["/bin/bash", "-c"],
                    "args": [
                        "set -eu; /cuda-samples/vectorAdd; "
                        "printf 'NPA_MIG_VISIBLE=%s\\n' "
                        '"${NVIDIA_VISIBLE_DEVICES:-}"; nvidia-smi -L; nvidia-smi'
                    ],
                    "resources": {
                        "requests": {resource: 1},
                        "limits": {resource: 1},
                    },
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                }
            ],
        },
    }
    try:
        create = subprocess.run(
            [kubectl_bin, "--kubeconfig", str(kubeconfig), "apply", "-f", "-"],
            input=json.dumps(manifest),
            check=False,
            capture_output=True,
            text=True,
            env=_kubectl_env(),
            timeout=_remaining_timeout(work_deadline, monotonic_fn),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        cleanup_error = _cleanup_mig_cuda_smoke_pod(
            kubectl_bin=kubectl_bin,
            kubeconfig=kubeconfig,
            pod_name=pod_name,
            deadline=deadline,
            monotonic_fn=monotonic_fn,
        )
        raise MigVerificationError(
            "could not create the representative MIG CUDA smoke pod before the "
            f"deployment health deadline ({type(exc).__name__})"
            + (
                f"; additionally pod cleanup failed ({cleanup_error}); delete "
                f"default/{pod_name} before retrying"
                if cleanup_error
                else ""
            )
        ) from exc
    if create.returncode != 0:
        cleanup_error = _cleanup_mig_cuda_smoke_pod(
            kubectl_bin=kubectl_bin,
            kubeconfig=kubeconfig,
            pod_name=pod_name,
            deadline=deadline,
            monotonic_fn=monotonic_fn,
        )
        raise MigVerificationError(
            f"could not create representative MIG CUDA smoke pod for {resource} "
            f"(kubectl exit status {create.returncode})"
            + (
                f"; additionally pod cleanup failed ({cleanup_error}); delete "
                f"default/{pod_name} before retrying"
                if cleanup_error
                else ""
            )
        )
    phase = ""
    node = ""
    failure_detail = ""
    try:
        while monotonic_fn() < work_deadline:
            pod = _kubectl_json(
                kubectl_bin,
                kubeconfig,
                ["get", "pod", pod_name, "-n", "default"],
                timeout_seconds=_remaining_timeout(work_deadline, monotonic_fn),
            )
            status = pod.get("status", {})
            phase = str(status.get("phase") or "")
            node = str(pod.get("spec", {}).get("nodeName") or "")
            if phase in {"Succeeded", "Failed"}:
                terminated = [
                    item.get("state", {}).get("terminated", {})
                    for item in status.get("containerStatuses", [])
                    if isinstance(item, dict)
                ]
                failure_detail = ", ".join(
                    str(item.get("reason") or f"exit {item.get('exitCode')}")
                    for item in terminated
                    if item.get("exitCode", 0) != 0
                )
                break
            sleep_fn(min(2.0, max(0.0, work_deadline - monotonic_fn())))

        try:
            logs = subprocess.run(
                [
                    kubectl_bin,
                    "--kubeconfig",
                    str(kubeconfig),
                    "logs",
                    pod_name,
                    "-n",
                    "default",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=_kubectl_env(),
                timeout=_remaining_timeout(work_deadline, monotonic_fn),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MigVerificationError(
                "could not collect representative MIG CUDA smoke logs before the "
                f"deployment health deadline ({type(exc).__name__})"
            ) from exc
        output = logs.stdout or ""
        if phase != "Succeeded" or logs.returncode != 0:
            detail = output[-1000:] or (logs.stderr or "")[-1000:]
            raise MigVerificationError(
                "representative MIG CUDA smoke failed: "
                f"resource={resource}, node={node or '<unscheduled>'}, "
                f"phase={phase or 'timeout'}"
                + (f", reason={failure_detail}" if failure_detail else "")
                + (f"; logs: {detail}" if detail else "")
            )
        if "Test PASSED" not in output:
            raise MigVerificationError(
                "representative MIG CUDA smoke exited successfully without required "
                "'Test PASSED' evidence"
            )
        profile = resource.removeprefix("nvidia.com/mig-")
        visible = re.search(r"(?m)^NPA_MIG_VISIBLE=([^\r\n]*)$", output)
        visible_value = visible.group(1).strip() if visible else ""
        instances = re.findall(
            r"(?im)^\s*MIG\s+(\S+)\s+Device\s+\d+:.*"
            r"UUID:\s*(MIG-[0-9A-Fa-f-]{16,})\)",
            output,
        )
        if len(instances) != 1:
            raise MigVerificationError(
                "representative MIG CUDA smoke did not expose exactly one hardware "
                f"MIG instance in-container (observed {len(instances)}); refusing "
                "whole-GPU or ambiguous allocation"
            )
        observed_profile, identity = instances[0]
        if observed_profile.lower() != profile.lower():
            raise MigVerificationError(
                "representative MIG CUDA smoke did not report its allocated "
                f"{identity} as the requested {profile} profile in-container"
            )
        # The device plugin may inject the allocation through CDI. In that mode
        # NVIDIA_VISIBLE_DEVICES is deliberately ``void`` and the container's
        # restricted nvidia-smi view is authoritative. In legacy envvar mode
        # the exact MIG UUID must agree with that view.
        if visible_value != "void" and visible_value != identity:
            raise MigVerificationError(
                "representative MIG CUDA smoke has inconsistent allocated-device "
                "identity: NVIDIA_VISIBLE_DEVICES is neither the sole in-container "
                "MIG UUID nor the CDI sentinel 'void'"
            )
        memory = re.search(r"\d+MiB\s*/\s*(\d+)MiB", output)
        memory_mib = int(memory.group(1)) if memory else 0
        expected_memory = {
            "1g.24gb": (23_000, 25_000),
            "2g.48gb": (47_000, 50_000),
        }[profile]
        if not expected_memory[0] <= memory_mib <= expected_memory[1]:
            raise MigVerificationError(
                "representative MIG CUDA smoke did not report framebuffer memory "
                f"for the requested {profile} profile (observed {memory_mib} MiB)"
            )
        return {
            "resource": resource,
            "profile": profile,
            "node": node,
            "pod": pod_name,
            "phase": phase,
            "vectoradd": "passed",
            "mig_uuid": identity,
            "memory_mib": memory_mib,
        }
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_error = _cleanup_mig_cuda_smoke_pod(
            kubectl_bin=kubectl_bin,
            kubeconfig=kubeconfig,
            pod_name=pod_name,
            deadline=deadline,
            monotonic_fn=monotonic_fn,
        )
        if cleanup_error and primary_error is not None:
            raise MigVerificationError(
                f"{primary_error}; additionally representative MIG CUDA smoke pod "
                f"cleanup failed ({cleanup_error}); delete default/{pod_name} before "
                "retrying"
            ) from primary_error
        if cleanup_error:
            raise MigVerificationError(
                "representative MIG CUDA smoke passed but its pod cleanup failed "
                f"({cleanup_error}); delete default/{pod_name} before retrying"
            )


def wait_for_mig_ready(
    *,
    kubectl_bin: str,
    kubeconfig: Path,
    expected_nodes: int,
    reconcile: bool = True,
    timeout_seconds: int = 3600,
    cuda_smoke_image: str | None = None,
    on_status: Callable[[str], None] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    monotonic_fn: Callable[[], float] | None = None,
) -> MigVerificationReport:
    """Wait for two stable exact snapshots, repairing the known discovery race.

    Authentication/API failures raise immediately; ordinary Operator convergence
    is bounded by the cluster's configured GPU health timeout.
    Once geometry and labels are successful, stale kubelet resources trigger a
    single ordered GFD/device-plugin restart. The exact obsolete MIG readiness
    taint may be removed from an otherwise successful GPU node; arbitrary node
    taints and node status are never patched.
    """

    if timeout_seconds <= 0:
        raise ValueError("MIG readiness timeout must be positive")
    sleep_fn = sleep_fn or time.sleep
    monotonic_fn = monotonic_fn or time.monotonic
    deadline = monotonic_fn() + timeout_seconds
    stable = 0
    restarted = False
    driver_reconciled = False
    taints_reconciled = False
    plugin_gate_installed = not reconcile
    previous_errors: tuple[str, ...] | None = None
    while stable < 2:
        if monotonic_fn() >= deadline:
            detail = "; ".join(previous_errors or ()) or "no exact snapshot observed"
            raise MigVerificationError(
                f"MIG readiness did not converge within {timeout_seconds}s: {detail}"
            )
        if not plugin_gate_installed:
            plugin_gate_installed = _ensure_device_plugin_mig_gate(
                kubectl_bin,
                kubeconfig,
                deadline=deadline,
                monotonic_fn=monotonic_fn,
            )
            if plugin_gate_installed and on_status:
                on_status("NVIDIA device plugin is gated on mig.config.state=success")
        report = verify_mig_cluster(
            kubectl_bin=kubectl_bin,
            kubeconfig=kubeconfig,
            expected_nodes=expected_nodes,
            timeout_seconds=max(0.001, deadline - monotonic_fn()),
            monotonic_fn=monotonic_fn,
        )
        if report.ready:
            stable += 1
            if on_status:
                on_status(f"MIG readiness snapshot {stable}/2 is exact")
        else:
            stable = 0
            repeated_errors = report.errors == previous_errors
            if report.errors != previous_errors and on_status:
                on_status("MIG not ready: " + "; ".join(report.errors))
            previous_errors = report.errors
            immutable_hardware_errors = tuple(
                error
                for error in report.errors
                if any(
                    marker in error
                    for marker in (
                        "hardware product is",
                        "PCI device is",
                        "unsupported vBIOS",
                        "memory is",
                    )
                )
            )
            if repeated_errors and immutable_hardware_errors:
                raise MigVerificationError(
                    "MIG compatibility cannot converge on this hardware: "
                    + "; ".join(immutable_hardware_errors)
                )
            driver_update_pending = False
            for error in report.errors:
                match = re.match(
                    r"DaemonSet nvidia-driver-daemonset: "
                    r"desired/current/updated/ready/available="
                    r"(\d+)/(\d+)/(\d+)/(\d+)/(\d+),",
                    error,
                )
                if match:
                    desired, current, updated, ready, available = map(
                        int, match.groups()
                    )
                    driver_update_pending = (
                        desired == current == ready == available == expected_nodes
                        and updated != expected_nodes
                    )
                    break
            if reconcile and driver_update_pending and not driver_reconciled:
                if on_status:
                    on_status(
                        "NVIDIA OnDelete driver update is pending; checking for "
                        "active GPU workloads before one-node-at-a-time replacement"
                    )
                _reconcile_ondelete_driver(
                    kubectl_bin,
                    kubeconfig,
                    deadline=deadline,
                    sleep_fn=sleep_fn,
                    monotonic_fn=monotonic_fn,
                )
                driver_reconciled = True
                continue
            stale_taint_only = bool(report.errors) and all(
                "stale nvidia.com/gpu=mig-not-ready:NoSchedule taint" in error
                for error in report.errors
            )
            if reconcile and stale_taint_only and not taints_reconciled:
                if on_status:
                    on_status(
                        "MIG is ready but a stale replacement-node readiness taint "
                        "blocks scheduling; removing only that exact taint"
                    )
                _reconcile_stale_mig_taints(
                    kubectl_bin,
                    kubeconfig,
                    deadline=deadline,
                    monotonic_fn=monotonic_fn,
                )
                taints_reconciled = True
                continue
            resource_only = bool(report.errors) and all(
                "capacity/allocatable=" in error for error in report.errors
            )
            labels_success = bool(report.nodes) and all(
                node.config == SUPPORTED_MIG_CONFIG and node.config_state == "success"
                for node in report.nodes
            )
            if reconcile and resource_only and labels_success and not restarted:
                if on_status:
                    on_status(
                        "MIG geometry is ready but kubelet resources are stale; "
                        "restarting GFD and the NVIDIA device plugin once"
                    )
                _restart_discovery_operands(
                    kubectl_bin,
                    kubeconfig,
                    deadline=deadline,
                    monotonic_fn=monotonic_fn,
                )
                restarted = True
        if stable < 2:
            sleep_fn(min(10.0, max(0.0, deadline - monotonic_fn())))
    if cuda_smoke_image:
        if on_status:
            on_status("running representative hardware MIG CUDA smoke")
        smoke = _run_mig_cuda_smoke(
            kubectl_bin=kubectl_bin,
            kubeconfig=kubeconfig,
            image=cuda_smoke_image,
            deadline=deadline,
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
        )
        report = replace(report, cuda_smoke=smoke)
    return report
