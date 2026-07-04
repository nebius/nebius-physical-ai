"""Declarative spec for an npa-managed soperator cluster.

apiVersion: ``npa.soperator/v0.0.1``

The spec is intentionally small: it captures the control-plane sizing plus a
list of heterogeneous worker pools. Each worker pool maps to one
``slurm_nodeset_workers`` entry in the solutions-library recipe, so a single
cluster can mix presets (e.g. a CPU pool and a GPU pool) and enable a
node-local Docker/Enroot image cache per pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

API_VERSION = "npa.soperator/v0.0.1"

# Minimum sufficient cpu-d3 presets per soperator node role (from the recipe's
# available_resources sufficiency map): system>=8, controller>=4, login>=16.
_MIN_PRESET = {
    "system": "8vcpu-32gb",
    "controller": "4vcpu-16gb",
    "login": "16vcpu-64gb",
}


class SoperatorSpecError(ValueError):
    """Raised when a soperator spec is missing required fields or malformed."""


@dataclass
class WorkerPoolSpec:
    """One worker node pool. Maps to a single ``slurm_nodeset_workers`` entry."""

    name: str
    platform: str = "cpu-d3"
    preset: str = "8vcpu-32gb"
    size: int = 1
    boot_disk_gib: int = 512
    # InfiniBand fabric id -- REQUIRED by the recipe for GPU presets that
    # support clustering (8-GPU SXM). Leave empty for CPU pools.
    fabric: str = ""
    preemptible: bool = False
    # Node-local Docker/Enroot image cache disk (the reason multi-GB GPU tool
    # images don't thrash the boot disk). Enables node_local_image_disk.
    docker_cache: bool = False
    docker_cache_gib: int = 372  # must be divisible by 93 for IO_M3 (keep IO_M3 quota modest)
    docker_cache_disk_type: str = "NETWORK_SSD_IO_M3"

    def is_gpu(self) -> bool:
        return self.platform.startswith("gpu-")

    def validate(self) -> None:
        if not self.name or not self.name.replace("-", "").isalnum():
            raise SoperatorSpecError(
                f"worker pool name must be alphanumeric/dash: {self.name!r}"
            )
        if self.size < 1:
            raise SoperatorSpecError(f"worker pool {self.name}: size must be >= 1")
        if self.is_gpu() and not self.fabric:
            # 1-GPU SXM presets cannot join a fabric; the recipe requires a
            # fabric for any GPU preset, so GPU pools must be fabric-capable
            # (8-GPU SXM) and supply the region fabric id.
            raise SoperatorSpecError(
                f"worker pool {self.name}: GPU preset {self.preset!r} requires a "
                "non-empty 'fabric' (region InfiniBand fabric id). 1-GPU presets "
                "cannot cluster; use an 8-GPU SXM preset for GPU workers."
            )
        if not self.is_gpu() and self.fabric:
            raise SoperatorSpecError(
                f"worker pool {self.name}: CPU preset must not set 'fabric'"
            )
        if self.docker_cache and self.docker_cache_gib % 93 != 0:
            raise SoperatorSpecError(
                f"worker pool {self.name}: docker_cache_gib must be divisible by 93 "
                f"(got {self.docker_cache_gib})"
            )
        if self.boot_disk_gib < 512:
            raise SoperatorSpecError(
                f"worker pool {self.name}: boot_disk_gib must be >= 512 (recipe rule)"
            )


@dataclass
class SoperatorSpec:
    """A full soperator cluster spec."""

    name: str
    region: str = ""  # resolved from ~/.npa config when empty
    tenant_id: str = ""
    project_id: str = ""
    subnet_id: str = ""
    ssh_public_keys: list[str] = field(default_factory=list)

    # Control-plane sizing (minimal defaults that fit the sufficiency map).
    system_min_size: int = 3  # recipe minimum is 3
    system_preset: str = _MIN_PRESET["system"]
    controller_preset: str = _MIN_PRESET["controller"]
    login_preset: str = _MIN_PRESET["login"]

    workers: list[WorkerPoolSpec] = field(default_factory=list)

    # Toggles that keep the deploy small and working out of the box.
    accounting: bool = False
    telemetry: bool = False
    # Custom AppArmor profile is not loaded by SPO in the 4.1.0 stable build;
    # unconfined (use_default_apparmor_profile=false) keeps login/worker sshd
    # starting. Default off for reliability; opt in if your build loads it.
    use_default_apparmor_profile: bool = False
    jail_size_gib: int = 512
    slurm_operator_version: str = "4.1.0"

    def validate(self) -> None:
        if not self.name or not self.name.replace("-", "").isalnum():
            raise SoperatorSpecError(f"cluster name must be alphanumeric/dash: {self.name!r}")
        if self.system_min_size < 3:
            raise SoperatorSpecError("system_min_size must be >= 3 (recipe rule)")
        if not self.workers:
            raise SoperatorSpecError("at least one worker pool is required")
        seen: set[str] = set()
        for pool in self.workers:
            pool.validate()
            if pool.name in seen:
                raise SoperatorSpecError(f"duplicate worker pool name: {pool.name}")
            seen.add(pool.name)


def spec_from_mapping(data: dict[str, Any]) -> SoperatorSpec:
    """Build a :class:`SoperatorSpec` from a parsed YAML/JSON mapping."""

    if not isinstance(data, dict):
        raise SoperatorSpecError("spec must be a mapping")
    api = str(data.get("apiVersion", API_VERSION))
    if api != API_VERSION:
        raise SoperatorSpecError(f"unsupported apiVersion {api!r}; expected {API_VERSION}")

    raw_workers = data.get("workers") or []
    if not isinstance(raw_workers, list):
        raise SoperatorSpecError("workers must be a list")
    workers: list[WorkerPoolSpec] = []
    for entry in raw_workers:
        if not isinstance(entry, dict):
            raise SoperatorSpecError("each worker pool must be a mapping")
        workers.append(
            WorkerPoolSpec(
                name=str(entry.get("name", "")),
                platform=str(entry.get("platform", "cpu-d3")),
                preset=str(entry.get("preset", "8vcpu-32gb")),
                size=int(entry.get("size", 1)),
                boot_disk_gib=int(entry.get("boot_disk_gib", 512)),
                fabric=str(entry.get("fabric", "")),
                preemptible=bool(entry.get("preemptible", False)),
                docker_cache=bool(entry.get("docker_cache", False)),
                docker_cache_gib=int(entry.get("docker_cache_gib", 372)),
                docker_cache_disk_type=str(
                    entry.get("docker_cache_disk_type", "NETWORK_SSD_IO_M3")
                ),
            )
        )

    control = data.get("control_plane") or {}
    system = control.get("system") or {}
    spec = SoperatorSpec(
        name=str(data.get("name", "")),
        region=str(data.get("region", "")),
        tenant_id=str(data.get("tenant_id", "")),
        project_id=str(data.get("project_id", "")),
        subnet_id=str(data.get("subnet_id", "")),
        ssh_public_keys=list(data.get("ssh_public_keys") or []),
        system_min_size=int(system.get("min_size", 3)),
        system_preset=str(system.get("preset", _MIN_PRESET["system"])),
        controller_preset=str((control.get("controller") or {}).get("preset", _MIN_PRESET["controller"])),
        login_preset=str((control.get("login") or {}).get("preset", _MIN_PRESET["login"])),
        workers=workers,
        accounting=bool(data.get("accounting", False)),
        telemetry=bool(data.get("telemetry", False)),
        use_default_apparmor_profile=bool(data.get("use_default_apparmor_profile", False)),
        jail_size_gib=int(data.get("jail_size_gib", 512)),
        slurm_operator_version=str(data.get("slurm_operator_version", "4.1.0")),
    )
    return spec


def load_spec(path: str | Path) -> SoperatorSpec:
    """Load and validate a soperator spec from a YAML file."""

    text = Path(path).expanduser().read_text()
    data = yaml.safe_load(text) or {}
    spec = spec_from_mapping(data)
    spec.validate()
    return spec
