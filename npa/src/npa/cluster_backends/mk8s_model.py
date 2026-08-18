"""Backend-owned Managed Kubernetes desired-state boundary."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from npa.cluster.gpu_driver import (
    DEFAULT_MANAGED_DRIVER_PRESET,
    GpuDriverStrategyError,
    resolve_gpu_driver_strategy,
)
from npa.cluster.gpu_health import (
    DEFAULT_CUDA_SMOKE_IMAGE,
    DEFAULT_STABILIZATION_SECONDS,
)
from npa.cluster_backends.mig import (
    MIG_KUBERNETES_VERSION,
    RTX_PRO_6000_BOOT_DISK_GIB,
    MigSpec,
)

_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


@dataclass(frozen=True)
class MK8sNodePool:
    count: int = 0
    platform: str = "cpu-d3"
    preset: str = "16vcpu-64gb"
    disk_size_gib: int = 0
    capacity_block_group: str = ""
    preemptible: bool = False

    @classmethod
    def from_surface(cls, pool: Any | None) -> "MK8sNodePool | None":
        if pool is None or isinstance(pool, cls):
            return pool
        return cls(**{name: getattr(pool, name) for name in cls.__dataclass_fields__})

    def is_gpu(self) -> bool:
        return self.platform.startswith("gpu-")


@dataclass(frozen=True)
class MK8sExecutionScope:
    fleet_name: str
    tenant_id: str = ""
    region: str = ""
    project_prefix: str = ""

    @property
    def name(self) -> str:
        return self.fleet_name


@dataclass(frozen=True)
class MK8sProjectIdentity:
    project_key: str
    project_id: str = ""
    project_name: str = ""
    expected_provider_name: str = ""

    def key(self) -> str:
        return self.project_key

    @property
    def name(self) -> str:
        return self.project_name

    def display_name(self, _prefix: str) -> str:
        return self.expected_provider_name


@dataclass(frozen=True)
class MK8sDesired:
    """Pure mk8s desired state, with no fleet or soperator envelope fields."""

    name: str
    k8s_version: str = ""
    cpu_nodes: MK8sNodePool | None = None
    gpu_nodes: MK8sNodePool | None = None
    enable_gpu_cluster: bool | None = None
    infiniband_fabric: str = ""
    enable_filestore: bool = False
    existing_filestore: str = ""
    filestore_disk_size_gibibytes: int = 1024
    gpu_disk_size_gib: int = 1023
    subnet_id: str = ""
    filestore_mount_path: str = "/mnt/data"
    filestore_mount_tag: str = "data"
    gpu_driver_mode: str = "auto"
    managed_driver_preset: str = DEFAULT_MANAGED_DRIVER_PRESET
    allow_unsafe_nvswitch_operator: bool = False
    gpu_health_stabilization_seconds: int = DEFAULT_STABILIZATION_SECONDS
    gpu_health_timeout_minutes: int = 60
    gpu_cuda_smoke: bool = True
    gpu_cuda_smoke_image: str = DEFAULT_CUDA_SMOKE_IMAGE
    mig: MigSpec | None = None
    allow_control_plane_only: bool = False

    @classmethod
    def from_surface(cls, cluster: Any) -> "MK8sDesired":
        if isinstance(cluster, cls):
            cluster.validate()
            return cluster
        cluster.validate()
        if cluster.backend_name() != "mk8s":
            raise ValueError("mk8s backend received a non-mk8s desired state")
        if getattr(cluster, "soperator", None) is not None:
            raise ValueError(
                "mk8s desired state cannot contain soperator configuration"
            )
        values = {name: getattr(cluster, name) for name in cls.__dataclass_fields__}
        values["cpu_nodes"] = MK8sNodePool.from_surface(values["cpu_nodes"])
        values["gpu_nodes"] = MK8sNodePool.from_surface(values["gpu_nodes"])
        desired = cls(**values)
        desired.validate()
        return desired

    def validate(self) -> None:
        if not _DNS_LABEL.fullmatch(self.name):
            raise ValueError(
                f"mk8s cluster name must be a lowercase DNS-1123 label: {self.name!r}"
            )
        if self.cpu_count() < 0 or self.gpu_count() < 0:
            raise ValueError("mk8s node counts cannot be negative")
        if (
            self.cpu_count() <= 0
            and self.gpu_count() <= 0
            and not (self.mig and self.mig.enabled)
            and not self.allow_control_plane_only
        ):
            raise ValueError("mk8s desired state needs at least one CPU or GPU node")
        for label, pool in (
            ("cpu_nodes", self.cpu_nodes),
            ("gpu_nodes", self.gpu_nodes),
        ):
            if pool is None:
                continue
            if pool.disk_size_gib < 0:
                raise ValueError(f"mk8s {label} disk size cannot be negative")
            if pool.count > 0 and (not pool.platform or not pool.preset):
                raise ValueError(
                    f"mk8s {label} requires platform and preset when count is positive"
                )
            if label == "cpu_nodes" and pool.count > 0 and pool.is_gpu():
                raise ValueError("mk8s cpu_nodes cannot select a GPU platform")
            if label == "gpu_nodes" and pool.count > 0 and not pool.is_gpu():
                raise ValueError("mk8s gpu_nodes must select a GPU platform")
        if self.cpu_nodes and self.cpu_nodes.capacity_block_group:
            raise ValueError(
                "mk8s capacity_block_group is supported only for gpu_nodes"
            )
        if (
            self.gpu_nodes
            and self.gpu_nodes.capacity_block_group
            and self.gpu_nodes.preemptible
        ):
            raise ValueError("strict reserved gpu_nodes cannot also be preemptible")
        gpu = self.gpu_nodes
        if gpu and gpu.count > 0:
            gpu_prefix = gpu.preset.split("gpu-", 1)[0] if gpu.preset else ""
            if not gpu_prefix.isdigit() or int(gpu_prefix) <= 0:
                raise ValueError("mk8s gpu_nodes preset must include a GPU count")
        if gpu and gpu.capacity_block_group:
            gpu_prefix = gpu.preset.split("gpu-", 1)[0] if gpu.preset else ""
            if gpu.count <= 0 or not gpu.is_gpu() or not gpu_prefix.isdigit():
                raise ValueError(
                    "mk8s strict capacity_block_group requires a GPU node pool "
                    "with a GPU-count preset"
                )
        if self.resolved_enable_gpu_cluster():
            if not (gpu and gpu.preset.startswith("8gpu-")):
                raise ValueError("mk8s enable_gpu_cluster requires an 8-GPU preset")
            if not self.infiniband_fabric:
                raise ValueError("mk8s enable_gpu_cluster requires infiniband_fabric")
        if self.enable_filestore:
            if self.filestore_disk_size_gibibytes <= 0:
                raise ValueError("mk8s filestore size must be positive")
            if not self.filestore_mount_path.startswith("/"):
                raise ValueError("mk8s filestore mount path must be absolute")
            if not self.filestore_mount_tag or any(
                char.isspace() or char == "," for char in self.filestore_mount_tag
            ):
                raise ValueError(
                    "mk8s filestore mount tag must not contain whitespace or commas"
                )
        if gpu and gpu.count > 0 and self.resolved_gpu_disk_size_gib() <= 0:
            raise ValueError("mk8s GPU disk size must be positive")
        if self.mig is not None:
            self.mig.validate(
                platform=gpu.platform if gpu else "",
                preset=gpu.preset if gpu else "",
                gpu_nodes=gpu.count if gpu else 0,
                capacity_block_group=gpu.capacity_block_group if gpu else "",
                disk_size_gib=gpu.disk_size_gib if gpu else 0,
                k8s_version=self.k8s_version,
            )
            if self.mig.enabled and self.gpu_driver_mode == "managed-image":
                raise ValueError(
                    "RTX PRO 6000 MIG requires the GPU Operator driver path"
                )
            if self.mig.enabled and not self.gpu_cuda_smoke:
                raise ValueError("RTX PRO 6000 MIG requires gpu_cuda_smoke=true")
        try:
            resolve_gpu_driver_strategy(
                gpu_nodes=self.gpu_count(),
                platform=gpu.platform if gpu else "",
                preset=gpu.preset if gpu else "",
                mode=self.resolved_gpu_driver_mode(),
                managed_driver_preset=self.managed_driver_preset,
                enable_gpu_cluster=self.resolved_enable_gpu_cluster(),
                allow_unsafe_nvswitch_operator=self.allow_unsafe_nvswitch_operator,
            )
        except GpuDriverStrategyError as exc:
            raise ValueError(str(exc)) from exc
        if self.gpu_health_stabilization_seconds < 0:
            raise ValueError("mk8s GPU health stabilization cannot be negative")
        if self.gpu_health_timeout_minutes <= 0:
            raise ValueError("mk8s GPU health timeout must be positive")
        if self.gpu_cuda_smoke and not self.gpu_cuda_smoke_image.strip():
            raise ValueError("mk8s CUDA smoke image cannot be empty when enabled")

    def backend_name(self) -> str:
        return "mk8s"

    def resolved_k8s_version(self) -> str:
        if self.mig and self.mig.enabled:
            return self.k8s_version or MIG_KUBERNETES_VERSION
        return self.k8s_version

    def resolved_gpu_disk_size_gib(self) -> int:
        gpu = self.gpu_nodes
        if gpu and gpu.disk_size_gib > 0:
            return gpu.disk_size_gib
        if self.mig and self.mig.enabled:
            return RTX_PRO_6000_BOOT_DISK_GIB
        return self.gpu_disk_size_gib

    def resolved_gpu_driver_mode(self) -> str:
        return "operator" if self.mig and self.mig.enabled else self.gpu_driver_mode

    def resolved_enable_gpu_cluster(self) -> bool:
        if self.enable_gpu_cluster is not None:
            return self.enable_gpu_cluster
        gpu = self.gpu_nodes
        return bool(gpu and gpu.count > 0 and gpu.preset.startswith("8gpu-"))

    def gpu_count(self) -> int:
        return self.gpu_nodes.count if self.gpu_nodes else 0

    def cpu_count(self) -> int:
        return self.cpu_nodes.count if self.cpu_nodes else 0


def as_mk8s_desired(value: Any) -> MK8sDesired:
    return MK8sDesired.from_surface(value)
