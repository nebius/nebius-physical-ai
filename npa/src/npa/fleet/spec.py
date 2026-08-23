"""Declarative spec for an npa-managed **fleet** of Kubernetes clusters.

apiVersion: ``npa.fleet/v0.0.1``

A fleet deploys one *or many* managed Kubernetes clusters across one *or many*
projects in a single Nebius tenant, wrapping the public
``nebius/nebius-solutions-library`` ``k8s-training`` recipe (the same recipe the
single-cluster ``npa cluster up`` uses). The spec is intentionally small but
composable:

* A ``defaults`` cluster profile is merged under every cluster, so deploying the
  **same** cluster profile across N projects is just N project entries with no
  per-cluster overrides ("identical" fleets).
* Any project may override the profile per cluster and/or declare several
  clusters ("custom" fleets), and a single spec may freely **mix** identical and
  custom projects.
* Projects may reference an existing ``project_id`` *or* be created on demand
  under the tenant (name = ``project_prefix`` + the entry's ``name``) using the
  ``nebius`` CLI.
* ``profile`` names the ``~/.nebius`` profile the run authenticates as, so one
  workstation can deploy fleets into several tenants without switching the
  machine-wide active profile.

No tenant/project IDs are baked in here -- they are resolved from the spec,
``~/.npa``/``~/.nebius`` config, or explicit CLI arguments, keeping this module
public-repo safe.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from npa.cluster.gpu_driver import (
    DEFAULT_MANAGED_DRIVER_PRESET,
    GpuDriverStrategyError,
    is_fabric_capable_topology,
    resolve_gpu_driver_strategy,
)
from npa.cluster.gpu_health import (
    DEFAULT_CUDA_SMOKE_IMAGE,
    DEFAULT_STABILIZATION_SECONDS,
)
from npa.fleet.mig import (
    MIG_KUBERNETES_VERSION,
    RTX_PRO_6000_BOOT_DISK_GIB,
    MigSpec,
    MigSpecError,
    mig_spec_from_mapping,
)
from npa.soperator.spec import (
    SoperatorSpec,
    SoperatorSpecError,
    spec_from_mapping as soperator_spec_from_mapping,
)

API_VERSION = "npa.fleet/v0.0.1"

_MK8S_ENVELOPE_FIELDS = {
    "name",
    "k8s_version",
    "cpu_nodes",
    "gpu_nodes",
    "enable_gpu_cluster",
    "infiniband_fabric",
    "enable_filestore",
    "existing_filestore",
    "filestore_disk_size_gibibytes",
    "gpu_disk_size_gib",
    "subnet_id",
    "filestore_mount_path",
    "filestore_mount_tag",
    "filesystem_csi_chart_repository",
    "gpu_driver_mode",
    "managed_driver_preset",
    "allow_unsafe_nvswitch_operator",
    "gpu_health_stabilization_seconds",
    "gpu_health_timeout_minutes",
    "gpu_cuda_smoke",
    "gpu_cuda_smoke_image",
    "mig",
}


class FleetSpecError(ValueError):
    """Raised when a fleet spec is missing required fields or is malformed."""


def _slug(value: str) -> str:
    out = "".join(
        ch if (ch.isalnum() or ch == "-") else "-" for ch in value.strip().lower()
    )
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def _is_dns_name(value: str) -> bool:
    # DNS-1123 label: lowercase alphanumeric/dash, start alphanumeric, <= 63 chars.
    return (
        bool(value)
        and len(value) <= 63
        and value == _slug(value)
        and value[0].isalnum()
    )


@dataclass
class NodePoolSpec:
    """A single node group (CPU-only or GPU) inside a cluster."""

    count: int = 0
    platform: str = "cpu-d3"
    preset: str = "16vcpu-64gb"
    disk_size_gib: int = 0  # 0 -> let the recipe/tfvars default apply
    # Optional capacity block group for GPU nodes. Fleet renders this as a
    # STRICT reservation policy, so the node group can never fall back to
    # ordinary on-demand capacity when the reservation is unavailable.
    capacity_block_group: str = ""
    preemptible: bool = False

    def is_gpu(self) -> bool:
        return self.platform.startswith("gpu-")


@dataclass
class ClusterSpec:
    """One managed Kubernetes cluster (a single k8s-training deployment)."""

    name: str = "cluster"
    k8s_version: str = ""  # empty -> backend default
    cpu_nodes: NodePoolSpec | None = None
    gpu_nodes: NodePoolSpec | None = None
    # None -> auto (True only for 8-GPU presets; the recipe rejects clustering on
    # single-GPU presets such as RTX PRO 6000 1-GPU).
    enable_gpu_cluster: bool | None = None
    infiniband_fabric: str = ""
    # Off by default: a shared filesystem provisions a 1 TiB NETWORK_SSD per
    # cluster and consumes tenant compute.filesystem.count/size quota. Opt in
    # per cluster when a shared FS + CSI default StorageClass is actually needed.
    enable_filestore: bool = False
    existing_filestore: str = ""
    filestore_disk_size_gibibytes: int = 1024
    gpu_disk_size_gib: int = 1023
    subnet_id: str = ""
    # Keep these explicit in the fleet contract: the filesystem attachment and
    # cloud-init fstab entry must agree on one stable virtiofs tag and mount.
    filestore_mount_path: str = "/mnt/data"
    filestore_mount_tag: str = "data"
    # Runtime/operator-supplied chart source. Keep private registry endpoints
    # out of committed specs and never surface this value in plan JSON.
    filesystem_csi_chart_repository: str = ""
    # Stable cross-path GPU driver contract. Auto selects Nebius's managed
    # driver-full node image for every requested GPU pool when the active recipe
    # supports it; operator is the explicit legacy/debug escape hatch.
    gpu_driver_mode: str = "auto"
    managed_driver_preset: str = DEFAULT_MANAGED_DRIVER_PRESET
    allow_unsafe_nvswitch_operator: bool = False
    gpu_health_stabilization_seconds: int = DEFAULT_STABILIZATION_SECONDS
    gpu_health_timeout_minutes: int = 60
    gpu_cuda_smoke: bool = True
    gpu_cuda_smoke_image: str = DEFAULT_CUDA_SMOKE_IMAGE
    # Declared last to preserve positional compatibility for SDK callers.
    # None (or an explicitly disabled block) retains the historical whole-GPU
    # behavior; enabled MIG is validated against the pinned RTX PRO tuple.
    mig: MigSpec | None = None
    # Backend selection is additive and declared last for positional SDK
    # compatibility. Missing/empty retains the historical mk8s meaning.
    backend: str = "mk8s"
    # Only populated for ``backend: soperator`` targets. Keeping the native
    # SoperatorSpec preserves that backend's complete, independently validated
    # desired-state contract instead of flattening both backends into one blob.
    soperator: SoperatorSpec | None = None
    # Parsing provenance used only to keep legacy plan JSON byte-compatible.
    backend_explicit: bool = field(default=False, repr=False, compare=False)
    # Standalone cluster up historically permits a control-plane-only topology;
    # fleet-created clusters remain required to have workers.
    allow_control_plane_only: bool = field(default=False, repr=False, compare=False)

    def backend_name(self) -> str:
        return self.backend.strip().lower() or "mk8s"

    def resolved_k8s_version(self) -> str:
        """Pin the tested Kubernetes version whenever hardware MIG is enabled."""

        if self.mig and self.mig.enabled:
            return self.k8s_version or MIG_KUBERNETES_VERSION
        return self.k8s_version

    def resolved_gpu_disk_size_gib(self) -> int:
        """Return the boot-disk size rendered and charged to quota preflight."""

        gpu = self.gpu_nodes
        if gpu and gpu.disk_size_gib > 0:
            return gpu.disk_size_gib
        if self.mig and self.mig.enabled:
            return RTX_PRO_6000_BOOT_DISK_GIB
        return self.gpu_disk_size_gib

    def resolved_gpu_driver_mode(self) -> str:
        """Use the in-cluster pinned driver path required by hardware MIG."""

        if self.mig and self.mig.enabled:
            return "operator"
        return self.gpu_driver_mode

    def resolved_enable_gpu_cluster(self) -> bool:
        if self.enable_gpu_cluster is not None:
            return self.enable_gpu_cluster
        gpu = self.gpu_nodes
        return bool(
            gpu
            and gpu.count > 0
            and is_fabric_capable_topology(
                platform=gpu.platform,
                preset=gpu.preset,
            )
        )

    def gpu_count(self) -> int:
        return self.gpu_nodes.count if self.gpu_nodes else 0

    def cpu_count(self) -> int:
        return self.cpu_nodes.count if self.cpu_nodes else 0

    def validate(self) -> None:
        backend = self.backend_name()
        if backend not in {"mk8s", "soperator"}:
            raise FleetSpecError(
                f"cluster {self.name!r}: unsupported backend {self.backend!r}; "
                "expected 'mk8s' or 'soperator'"
            )
        if backend == "soperator":
            if self.soperator is None:
                raise FleetSpecError(
                    f"cluster {self.name!r}: backend 'soperator' requires a "
                    "'soperator' configuration mapping"
                )
            if self.soperator.name != self.name:
                raise FleetSpecError(
                    f"cluster {self.name!r}: soperator.name must match the fleet "
                    "target name"
                )
            try:
                self.soperator.validate()
            except SoperatorSpecError as exc:
                raise FleetSpecError(f"cluster {self.name!r}: {exc}") from exc
            return
        if self.soperator is not None:
            raise FleetSpecError(
                f"cluster {self.name!r}: 'soperator' configuration is unsupported "
                "for backend 'mk8s'"
            )
        if not _is_dns_name(self.name):
            raise FleetSpecError(
                f"cluster name must be a lowercase DNS-1123 label: {self.name!r}"
            )
        mig_enabled = bool(self.mig and self.mig.enabled)
        if (
            self.cpu_count() <= 0
            and self.gpu_count() <= 0
            and not mig_enabled
            and not self.allow_control_plane_only
        ):
            raise FleetSpecError(
                f"cluster {self.name!r}: needs at least one CPU or GPU node"
            )
        for kind, pool in (
            ("cpu_nodes", self.cpu_nodes),
            ("gpu_nodes", self.gpu_nodes),
        ):
            if pool and pool.count < 0:
                raise FleetSpecError(
                    f"cluster {self.name!r}: {kind}.count cannot be negative"
                )
            if pool and pool.disk_size_gib < 0:
                raise FleetSpecError(
                    f"cluster {self.name!r}: {kind}.disk_size_gib cannot be negative"
                )
        if self.cpu_nodes and self.cpu_nodes.capacity_block_group:
            raise FleetSpecError(
                f"cluster {self.name!r}: capacity_block_group is only valid for gpu_nodes"
            )
        gpu = self.gpu_nodes
        if gpu and gpu.count > 0:
            if not gpu.is_gpu():
                raise FleetSpecError(
                    f"cluster {self.name!r}: gpu_nodes.platform must start with 'gpu-'"
                )
            gpu_prefix = gpu.preset.split("gpu-", 1)[0] if gpu.preset else ""
            if not gpu_prefix.isdigit() or int(gpu_prefix) <= 0:
                raise FleetSpecError(
                    f"cluster {self.name!r}: gpu_nodes.preset must include a positive GPU count"
                )
        if gpu and gpu.capacity_block_group and not mig_enabled:
            if gpu.count <= 0 or not gpu.is_gpu():
                raise FleetSpecError(
                    f"cluster {self.name!r}: capacity_block_group requires a GPU node pool"
                )
            if not gpu.preset or not gpu.preset.split("gpu-", 1)[0].isdigit():
                raise FleetSpecError(
                    f"cluster {self.name!r}: capacity_block_group requires a GPU-count preset"
                )
        if gpu and gpu.capacity_block_group and gpu.preemptible:
            raise FleetSpecError(
                f"cluster {self.name!r}: strict reserved gpu_nodes cannot also be preemptible"
            )
        if self.resolved_enable_gpu_cluster():
            gpu = self.gpu_nodes
            if not (
                gpu
                and is_fabric_capable_topology(
                    platform=gpu.platform,
                    preset=gpu.preset,
                )
            ):
                preset = gpu.preset if gpu else ""
                raise FleetSpecError(
                    f"cluster {self.name!r}: enable_gpu_cluster requires a "
                    f"fabric-capable 8-GPU SXM/NVL preset (got {preset!r} if any)"
                )
            if not self.infiniband_fabric:
                raise FleetSpecError(
                    f"cluster {self.name!r}: enable_gpu_cluster requires "
                    "'infiniband_fabric'"
                )
        if self.enable_filestore:
            if self.filestore_disk_size_gibibytes <= 0:
                raise FleetSpecError(
                    f"cluster {self.name!r}: filestore_disk_size_gibibytes must be positive"
                )
            if not self.filestore_mount_path.startswith("/"):
                raise FleetSpecError(
                    f"cluster {self.name!r}: filestore_mount_path must be absolute"
                )
            if not self.filestore_mount_tag or any(
                ch.isspace() or ch == "," for ch in self.filestore_mount_tag
            ):
                raise FleetSpecError(
                    f"cluster {self.name!r}: filestore_mount_tag must be a non-empty "
                    "value without whitespace or commas"
                )
            if any(ch.isspace() for ch in self.filesystem_csi_chart_repository):
                raise FleetSpecError(
                    f"cluster {self.name!r}: filesystem_csi_chart_repository must "
                    "not contain whitespace"
                )
        if (
            gpu
            and gpu.count > 0
            and gpu.disk_size_gib == 0
            and self.gpu_disk_size_gib <= 0
        ):
            raise FleetSpecError(
                f"cluster {self.name!r}: gpu_disk_size_gib must be positive"
            )
        if self.mig:
            try:
                self.mig.validate(
                    platform=gpu.platform if gpu else "",
                    preset=gpu.preset if gpu else "",
                    gpu_nodes=gpu.count if gpu else 0,
                    capacity_block_group=gpu.capacity_block_group if gpu else "",
                    disk_size_gib=gpu.disk_size_gib if gpu else 0,
                    k8s_version=self.k8s_version,
                )
            except MigSpecError as exc:
                raise FleetSpecError(f"cluster {self.name!r}: {exc}") from exc
            if self.mig.enabled and self.gpu_driver_mode == "managed-image":
                raise FleetSpecError(
                    f"cluster {self.name!r}: RTX PRO 6000 MIG requires the pinned "
                    "GPU Operator driver path; gpu_driver_mode='managed-image' is "
                    "incompatible"
                )
            if self.mig.enabled and not self.gpu_cuda_smoke:
                raise FleetSpecError(
                    f"cluster {self.name!r}: RTX PRO 6000 MIG requires "
                    "gpu_cuda_smoke=true so deploy verifies a real MIG allocation"
                )
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
            raise FleetSpecError(f"cluster {self.name!r}: {exc}") from exc
        if self.gpu_health_stabilization_seconds < 0:
            raise FleetSpecError(
                f"cluster {self.name!r}: gpu_health_stabilization_seconds cannot be negative"
            )
        if self.gpu_health_timeout_minutes <= 0:
            raise FleetSpecError(
                f"cluster {self.name!r}: gpu_health_timeout_minutes must be positive"
            )
        if self.gpu_cuda_smoke and not self.gpu_cuda_smoke_image.strip():
            raise FleetSpecError(
                f"cluster {self.name!r}: gpu_cuda_smoke_image cannot be empty when enabled"
            )


@dataclass
class ProjectSpec:
    """A deployment target: an existing project id or a create-on-demand name."""

    name: str = ""  # logical name; project display name = <prefix><name>
    project_id: str = ""  # existing project id (used verbatim when set)
    region: str = ""  # per-project region override
    clusters: list[ClusterSpec] = field(default_factory=list)

    def key(self) -> str:
        """Stable local key for install/state dirs (never used as a cloud name)."""

        return _slug(self.name or self.project_id or "project")

    def display_name(self, prefix: str) -> str:
        """Project display name used when looking up / creating the project."""

        if self.name:
            if prefix and not self.name.startswith(prefix):
                return f"{prefix}{self.name}"
            return self.name
        return self.project_id

    def validate(self) -> None:
        if not self.name and not self.project_id:
            raise FleetSpecError("each project needs a 'name' or 'project_id'")
        if self.name and not _is_dns_name(_slug(self.name)):
            raise FleetSpecError(f"project name is not a valid label: {self.name!r}")
        if not self.clusters:
            raise FleetSpecError(
                f"project {self.name or self.project_id!r}: no clusters resolved"
            )
        seen: set[str] = set()
        for cluster in self.clusters:
            cluster.validate()
            if cluster.name in seen:
                raise FleetSpecError(
                    f"project {self.name or self.project_id!r}: duplicate cluster "
                    f"name {cluster.name!r}"
                )
            seen.add(cluster.name)


@dataclass
class FleetSpec:
    """A full fleet: many clusters across many projects in one tenant."""

    name: str
    tenant_id: str = ""  # resolved from ~/.npa / ~/.nebius when empty
    region: str = ""
    project_prefix: str = ""
    ssh_public_key: str = ""
    projects: list[ProjectSpec] = field(default_factory=list)
    # Declared last so adding it cannot shift the positional order of the
    # pre-existing fields for SDK callers. The ~/.nebius profile every nebius CLI
    # call runs under; empty -> the machine's active profile. Needed because a
    # Nebius service account is single-tenant, so deploying into another tenant
    # means selecting that tenant's profile rather than mutating the active one.
    profile: str = ""

    def validate(self) -> None:
        if not _is_dns_name(self.name):
            raise FleetSpecError(
                f"fleet name must be a lowercase DNS-1123 label: {self.name!r}"
            )
        if not self.projects:
            raise FleetSpecError("at least one project is required")
        keys: set[str] = set()
        soperator_names: set[str] = set()
        for project in self.projects:
            project.validate()
            key = project.key()
            if key in keys:
                raise FleetSpecError(f"duplicate project key: {key!r}")
            keys.add(key)
            for cluster in project.clusters:
                if cluster.backend_name() != "soperator":
                    continue
                if cluster.name in soperator_names:
                    raise FleetSpecError(
                        f"duplicate soperator cluster name {cluster.name!r} across fleet; "
                        "soperator physical/context identity must be fleet-wide unique"
                    )
                soperator_names.add(cluster.name)

    def cluster_targets(self) -> list[tuple[ProjectSpec, ClusterSpec]]:
        """Flatten the fleet into ``(project, cluster)`` deployment targets."""

        return [(p, c) for p in self.projects for c in p.clusters]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` deep-merged with ``override`` (override wins)."""

    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        # ``mig`` is an atomic policy block. This allows a cluster to disable an
        # enabled default with ``mig: {enabled: false}`` without inheriting the
        # default strategy/config into an invalid hybrid policy.
        if (
            key != "mig"
            and isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _node_pool_from(
    data: dict[str, Any] | None, *, default_platform: str
) -> NodePoolSpec | None:
    if not data:
        return None
    return NodePoolSpec(
        count=int(data.get("count", 0)),
        platform=str(data.get("platform", default_platform)),
        preset=str(data.get("preset", "")),
        disk_size_gib=int(data.get("disk_size_gib", 0) or 0),
        capacity_block_group=str(data.get("capacity_block_group", "") or "").strip(),
        preemptible=bool(data.get("preemptible", False)),
    )


def _cluster_from(
    data: dict[str, Any], *, backend_explicit: bool = False
) -> ClusterSpec:
    enable_gpu = data.get("enable_gpu_cluster", None)
    try:
        mig = mig_spec_from_mapping(data.get("mig"))
    except MigSpecError as exc:
        raise FleetSpecError(str(exc)) from exc
    return ClusterSpec(
        name=_slug(str(data.get("name", "cluster"))) or "cluster",
        k8s_version=str(data.get("k8s_version", "") or ""),
        cpu_nodes=_node_pool_from(data.get("cpu_nodes"), default_platform="cpu-d3"),
        gpu_nodes=_node_pool_from(
            data.get("gpu_nodes"), default_platform="gpu-rtx6000"
        ),
        enable_gpu_cluster=None if enable_gpu is None else bool(enable_gpu),
        infiniband_fabric=str(data.get("infiniband_fabric", "") or ""),
        enable_filestore=bool(data.get("enable_filestore", False)),
        existing_filestore=str(data.get("existing_filestore", "") or ""),
        filestore_disk_size_gibibytes=int(
            data.get("filestore_disk_size_gibibytes", 1024)
        ),
        gpu_disk_size_gib=int(data.get("gpu_disk_size_gib", 1023)),
        subnet_id=str(data.get("subnet_id", "") or ""),
        filestore_mount_path=str(data.get("filestore_mount_path", "/mnt/data") or ""),
        filestore_mount_tag=str(data.get("filestore_mount_tag", "data") or ""),
        filesystem_csi_chart_repository=str(
            data.get("filesystem_csi_chart_repository", "") or ""
        ).strip(),
        gpu_driver_mode=str(data.get("gpu_driver_mode", "auto") or "auto"),
        managed_driver_preset=str(
            data.get("managed_driver_preset", DEFAULT_MANAGED_DRIVER_PRESET)
            or DEFAULT_MANAGED_DRIVER_PRESET
        ),
        allow_unsafe_nvswitch_operator=bool(
            data.get("allow_unsafe_nvswitch_operator", False)
        ),
        gpu_health_stabilization_seconds=int(
            data.get("gpu_health_stabilization_seconds", DEFAULT_STABILIZATION_SECONDS)
        ),
        gpu_health_timeout_minutes=int(data.get("gpu_health_timeout_minutes", 60)),
        gpu_cuda_smoke=bool(data.get("gpu_cuda_smoke", True)),
        gpu_cuda_smoke_image=str(
            data.get("gpu_cuda_smoke_image", DEFAULT_CUDA_SMOKE_IMAGE)
            or DEFAULT_CUDA_SMOKE_IMAGE
        ),
        mig=mig,
        backend="mk8s",
        backend_explicit=backend_explicit,
    )


def _soperator_cluster_from(data: dict[str, Any]) -> ClusterSpec:
    """Build one fleet target around the native Soperator desired state."""

    name = _slug(str(data.get("name", "cluster"))) or "cluster"
    raw = data.get("soperator")
    if not isinstance(raw, dict):
        raise FleetSpecError(
            f"cluster {name!r}: backend 'soperator' requires a 'soperator' mapping"
        )
    mapping = copy.deepcopy(raw)
    # Fleet, not a backend payload, owns target identity. Allowing an embedded
    # identity to win would deploy into one project while recording another.
    identity_fields = sorted(
        key
        for key in ("tenant_id", "project_id", "region", "subnet_id")
        if str(mapping.get(key, "") or "").strip()
    )
    if identity_fields:
        raise FleetSpecError(
            f"cluster {name!r}: fleet soperator configuration must not set "
            f"target identity field(s) {', '.join(identity_fields)}; use the "
            "fleet/project envelope"
        )
    allowed = {
        "name",
        "workers",
        "control_plane",
        "root_login_ssh_public_key",
        "ssh_public_keys",
        "accounting",
        "slurm_rest_enabled",
        "telemetry",
        "use_default_apparmor_profile",
        "jail_size_gib",
        "slurm_operator_version",
        "k8s_version",
        "node_group_version",
    }
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise FleetSpecError(
            f"cluster {name!r}: unsupported soperator field(s): {', '.join(unknown)}"
        )
    worker_allowed = {
        "name",
        "platform",
        "preset",
        "size",
        "boot_disk_gib",
        "fabric",
        "preemptible",
        "docker_cache",
        "docker_cache_gib",
        "docker_cache_disk_type",
        "capacity_block_group",
        "capacity_block_group_name",
    }
    workers = mapping.get("workers") or []
    if isinstance(workers, list):
        for index, worker in enumerate(workers):
            if isinstance(worker, dict):
                unknown_worker = sorted(set(worker) - worker_allowed)
                if unknown_worker:
                    raise FleetSpecError(
                        f"cluster {name!r}: soperator.workers[{index}] has "
                        f"unsupported field(s): {', '.join(unknown_worker)}"
                    )
    control = mapping.get("control_plane") or {}
    if not isinstance(control, dict):
        raise FleetSpecError(
            f"cluster {name!r}: soperator.control_plane must be a mapping"
        )
    unknown_control = sorted(
        set(control) - {"system", "controller", "accounting", "login"}
    )
    if unknown_control:
        raise FleetSpecError(
            f"cluster {name!r}: soperator.control_plane has unsupported field(s): "
            f"{', '.join(unknown_control)}"
        )
    for role, role_config in control.items():
        if not isinstance(role_config, dict):
            raise FleetSpecError(
                f"cluster {name!r}: soperator.control_plane.{role} must be a mapping"
            )
        role_allowed = {"preset"}
        if role == "system":
            role_allowed |= {"min_size", "max_size"}
        unknown_role = sorted(set(role_config) - role_allowed)
        if unknown_role:
            raise FleetSpecError(
                f"cluster {name!r}: soperator.control_plane.{role} has unsupported "
                f"field(s): {', '.join(unknown_role)}"
            )
    embedded_name = str(mapping.get("name", "") or "")
    if embedded_name and embedded_name != name:
        raise FleetSpecError(
            f"cluster {name!r}: soperator.name {embedded_name!r} must match the "
            "fleet target name"
        )
    mapping["name"] = name
    mapping["apiVersion"] = "npa.soperator/v0.0.1"
    try:
        desired = soperator_spec_from_mapping(mapping)
    except SoperatorSpecError as exc:
        raise FleetSpecError(f"cluster {name!r}: {exc}") from exc
    return ClusterSpec(
        name=name,
        backend="soperator",
        soperator=desired,
        backend_explicit=True,
    )


def spec_from_mapping(data: dict[str, Any]) -> FleetSpec:
    """Build a :class:`FleetSpec` from a parsed YAML/JSON mapping.

    ``defaults`` is deep-merged under each cluster so identical fleets need no
    per-cluster fields, while custom clusters override only what they change.
    """

    if not isinstance(data, dict):
        raise FleetSpecError("spec must be a mapping")
    api = str(data.get("apiVersion", API_VERSION))
    if api != API_VERSION:
        raise FleetSpecError(f"unsupported apiVersion {api!r}; expected {API_VERSION}")

    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise FleetSpecError("defaults must be a mapping")

    raw_projects = data.get("projects") or []
    if not isinstance(raw_projects, list) or not raw_projects:
        raise FleetSpecError("projects must be a non-empty list")

    projects: list[ProjectSpec] = []
    for entry in raw_projects:
        if not isinstance(entry, dict):
            raise FleetSpecError("each project must be a mapping")
        raw_clusters = entry.get("clusters")
        cluster_mappings: list[dict[str, Any]]
        if raw_clusters is None:
            # No clusters listed -> one cluster from the pure defaults profile.
            cluster_mappings = [{}]
        elif isinstance(raw_clusters, list) and raw_clusters:
            cluster_mappings = [c if isinstance(c, dict) else {} for c in raw_clusters]
        else:
            raise FleetSpecError("project 'clusters' must be a non-empty list when set")

        clusters: list[ClusterSpec] = []
        multi = len(cluster_mappings) > 1
        for idx, raw in enumerate(cluster_mappings):
            requested_backend = (
                str(raw.get("backend", defaults.get("backend", "mk8s")) or "mk8s")
                .strip()
                .lower()
            )
            if requested_backend == "soperator":
                # Backend envelopes are deliberately strict. Legacy flat fields
                # remain accepted only for the implicit mk8s path; accepting
                # them beside a soperator block would silently ignore a typo or
                # apply an option to the wrong backend.
                unexpected = sorted(set(raw) - {"name", "backend", "soperator"})
                if unexpected:
                    raise FleetSpecError(
                        f"soperator fleet cluster has unsupported mk8s/flat field(s): "
                        f"{', '.join(unexpected)}; place backend-specific settings "
                        "under 'soperator'"
                    )
                default_soperator = defaults.get("soperator") or {}
                if not isinstance(default_soperator, dict):
                    raise FleetSpecError("defaults.soperator must be a mapping")
                raw_soperator = raw.get("soperator") or {}
                if not isinstance(raw_soperator, dict):
                    raise FleetSpecError("cluster.soperator must be a mapping")
                merged = {
                    "backend": "soperator",
                    "name": raw.get("name") or defaults.get("name"),
                    "soperator": _deep_merge(default_soperator, raw_soperator),
                }
            elif requested_backend == "mk8s":
                raw_mk8s = raw.get("mk8s")
                if raw_mk8s is not None:
                    if not isinstance(raw_mk8s, dict):
                        raise FleetSpecError("cluster.mk8s must be a mapping")
                    unknown_mk8s = sorted(set(raw_mk8s) - _MK8S_ENVELOPE_FIELDS)
                    if unknown_mk8s:
                        raise FleetSpecError(
                            "cluster.mk8s has unsupported field(s): "
                            + ", ".join(unknown_mk8s)
                        )
                    for pool_name in ("cpu_nodes", "gpu_nodes"):
                        pool = raw_mk8s.get(pool_name)
                        if pool is not None and not isinstance(pool, dict):
                            raise FleetSpecError(
                                f"cluster.mk8s.{pool_name} must be a mapping"
                            )
                        unknown_pool = sorted(
                            set(pool or {})
                            - {
                                "count",
                                "platform",
                                "preset",
                                "disk_size_gib",
                                "capacity_block_group",
                                "preemptible",
                            }
                        )
                        if unknown_pool:
                            raise FleetSpecError(
                                f"cluster.mk8s.{pool_name} has unsupported field(s): "
                                + ", ".join(unknown_pool)
                            )
                    unexpected = sorted(set(raw) - {"name", "backend", "mk8s"})
                    if unexpected:
                        raise FleetSpecError(
                            f"mk8s fleet cluster mixes the 'mk8s' envelope with "
                            f"flat field(s): {', '.join(unexpected)}"
                        )
                    default_mk8s = defaults.get("mk8s") or {
                        key: value
                        for key, value in defaults.items()
                        if key not in {"backend", "soperator", "mk8s"}
                    }
                    if not isinstance(default_mk8s, dict):
                        raise FleetSpecError("defaults.mk8s must be a mapping")
                    merged = _deep_merge(default_mk8s, raw_mk8s)
                    merged["name"] = raw.get("name") or merged.get("name")
                else:
                    # Historical v0.0.1 mapping: preserve its normalized
                    # semantic resolution and permissive forward fields.
                    legacy_defaults = {
                        key: value
                        for key, value in defaults.items()
                        if key not in {"backend", "soperator", "mk8s"}
                    }
                    merged = _deep_merge(legacy_defaults, raw)
            else:
                raise FleetSpecError(
                    f"unsupported cluster backend {requested_backend!r}; expected "
                    "'mk8s' or 'soperator'"
                )
            # Derive a stable default cluster name when unset.
            if not merged.get("name"):
                merged["name"] = f"cluster-{idx}" if multi else "cluster"
            clusters.append(
                _soperator_cluster_from(merged)
                if requested_backend == "soperator"
                else _cluster_from(
                    merged, backend_explicit=("backend" in raw or "mk8s" in raw)
                )
            )

        projects.append(
            ProjectSpec(
                name=str(entry.get("name", "") or ""),
                project_id=str(entry.get("project_id", "") or ""),
                region=str(entry.get("region", "") or ""),
                clusters=clusters,
            )
        )

    return FleetSpec(
        name=_slug(str(data.get("name", ""))),
        tenant_id=str(data.get("tenant_id", "") or ""),
        region=str(data.get("region", "") or ""),
        project_prefix=str(data.get("project_prefix", "") or ""),
        ssh_public_key=str(data.get("ssh_public_key", "") or ""),
        projects=projects,
        profile=str(data.get("profile", "") or ""),
    )


def load_spec(path: str | Path) -> FleetSpec:
    """Load and validate a fleet spec from a YAML file."""

    text = Path(path).expanduser().read_text()
    data = yaml.safe_load(text) or {}
    spec = spec_from_mapping(data)
    spec.validate()
    return spec
