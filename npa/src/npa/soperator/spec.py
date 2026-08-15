"""Declarative spec for an npa-managed soperator cluster.

apiVersion: ``npa.soperator/v0.0.1``

The spec is intentionally small: it captures the control-plane sizing plus a
list of heterogeneous worker pools. Each worker pool maps to one
``slurm_nodeset_workers`` entry in the solutions-library recipe, so a single
cluster can mix presets (e.g. a CPU pool and a GPU pool) and enable a
node-local Docker/Enroot image cache per pool.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml

API_VERSION = "npa.soperator/v0.0.1"
DEFAULT_SOLUTIONS_LIBRARY_REF = "7046fb3c68314a940cdb47ff5c4fd23c01a6711e"
DEFAULT_SLURM_OPERATOR_VERSION = "4.1.6"
DEFAULT_K8S_VERSION = "1.34"
DEFAULT_NODE_GROUP_VERSION = "72"

# Mirrors modules/sizing_tier/main.tf at DEFAULT_SOLUTIONS_LIBRARY_REF. Keeping
# this small table in NPA lets an unsafe explicit override fail before cloning,
# Terraform initialization, or any provider mutation. Omitted presets are still
# resolved by the pinned upstream module, which remains the source of truth.
_SIZING_BOUNDARIES = ((10, "XS"), (100, "S"), (500, "M"), (2000, "L"))
_MIN_PRESET_BY_ROLE_AND_TIER = {
    "system": {
        "XS": "16vcpu-64gb",
        "S": "16vcpu-64gb",
        "M": "16vcpu-64gb",
        "L": "32vcpu-128gb",
        "XL": "64vcpu-256gb",
    },
    "controller": {tier: "16vcpu-64gb" for tier in ("XS", "S", "M", "L", "XL")},
    "accounting": {
        "XS": "8vcpu-32gb",
        "S": "8vcpu-32gb",
        "M": "8vcpu-32gb",
        "L": "16vcpu-64gb",
        "XL": "32vcpu-128gb",
    },
    "login": {tier: "16vcpu-64gb" for tier in ("XS", "S", "M", "L", "XL")},
}
_CPU_D3_PRESET_RANK = {
    "4vcpu-16gb": 4,
    "8vcpu-32gb": 8,
    "16vcpu-64gb": 16,
    "32vcpu-128gb": 32,
    "64vcpu-256gb": 64,
    "128vcpu-512gb": 128,
}
_OPERATOR_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_VERIFIED_OPERATOR_VERSIONS = frozenset({"4.1.6", "4.1.7"})
_VERIFIED_USERNS_OPERATOR_VERSIONS = frozenset({DEFAULT_SLURM_OPERATOR_VERSION})
_SSH_KEY_TYPE_RE = re.compile(
    r"^(?:ssh-(?:ed25519|rsa)|ecdsa-sha2-nistp(?:256|384|521)|"
    r"sk-(?:ssh-ed25519|ecdsa-sha2-nistp256)@openssh\.com)$"
)


class SoperatorSpecError(ValueError):
    """Raised when a soperator spec is missing required fields or malformed."""


def sizing_tier_for_worker_count(worker_count: int) -> str:
    """Return the pinned upstream sizing tier for *worker_count*."""

    for upper_bound, tier in _SIZING_BOUNDARIES:
        if worker_count < upper_bound:
            return tier
    return "XL"


def _validate_role_preset(role: str, preset: str | None, tier: str) -> None:
    """Reject explicit cpu-d3 presets smaller than the pinned tier requires."""

    if preset is None:
        return
    actual_rank = _CPU_D3_PRESET_RANK.get(preset)
    if actual_rank is None:
        accepted = ", ".join(_CPU_D3_PRESET_RANK)
        raise SoperatorSpecError(
            f"control_plane.{role}.preset {preset!r} is not a supported cpu-d3 "
            f"preset; expected one of: {accepted}"
        )
    required = _MIN_PRESET_BY_ROLE_AND_TIER[role][tier]
    if actual_rank < _CPU_D3_PRESET_RANK[required]:
        raise SoperatorSpecError(
            f"control_plane.{role}.preset {preset!r} is insufficient for {tier} "
            f"sizing ({role} requires at least {required}); omit the preset to "
            "use the pinned upstream sizing tier"
        )


def validate_ssh_public_key_record(value: str) -> str:
    """Validate and normalize one OpenSSH public-key record.

    Options, blank records, private-key blocks, and multiple newline-delimited
    keys are rejected. The optional comment may contain spaces.
    """

    normalized = value.strip()
    if not normalized or "\n" in normalized or "\r" in normalized:
        raise SoperatorSpecError(
            "root login SSH key must be exactly one non-empty public-key record"
        )
    fields = normalized.split(maxsplit=2)
    if len(fields) < 2 or not _SSH_KEY_TYPE_RE.fullmatch(fields[0]):
        raise SoperatorSpecError(
            "root login SSH key must start with a supported OpenSSH key type "
            "and contain one base64 key blob"
        )
    try:
        decoded = base64.b64decode(fields[1], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SoperatorSpecError("root login SSH key has an invalid base64 blob") from exc
    if not decoded:
        raise SoperatorSpecError("root login SSH key has an empty key blob")
    return normalized


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
    # Reserved-capacity selectors are runtime inputs. ``capacity_block_group``
    # matches the fleet contract and accepts an immutable group ID;
    # ``capacity_block_group_name`` is resolved to exactly one tenant-owned
    # group before Terraform is rendered. Live selector values must not be
    # committed in public examples.
    capacity_block_group: str = ""
    capacity_block_group_name: str = ""
    # Populated only by the provider preflight for name-based selectors. It is
    # deliberately absent from YAML parsing and public plan/status output.
    resolved_capacity_block_group_id: str = field(
        default="", repr=False, compare=False
    )

    def is_gpu(self) -> bool:
        return self.platform.startswith("gpu-")

    def capacity_mode(self) -> str:
        """Return the public worker-capacity mode without exposing selectors."""

        if self.capacity_block_group or self.capacity_block_group_name:
            return "reserved"
        return "preemptible" if self.preemptible else "on-demand"

    def reservation_selector_kind(self) -> str:
        """Return the configured selector kind, never its private value."""

        if self.capacity_block_group:
            return "id"
        if self.capacity_block_group_name:
            return "name"
        return ""

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
        if self.capacity_block_group and self.capacity_block_group_name:
            raise SoperatorSpecError(
                f"worker pool {self.name}: set only one of capacity_block_group "
                "or capacity_block_group_name"
            )
        if self.capacity_mode() == "reserved" and not self.is_gpu():
            raise SoperatorSpecError(
                f"worker pool {self.name}: reserved capacity selectors are valid "
                "only for GPU worker pools"
            )
        if self.capacity_mode() == "reserved" and self.preemptible:
            raise SoperatorSpecError(
                f"worker pool {self.name}: reserved capacity is mutually exclusive "
                "with preemptible=true; disable preemptibility to use a capacity block"
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
    # Existing positional field retained as a one-record compatibility alias.
    ssh_public_keys: list[str] = field(default_factory=list)

    # Omitted system/controller/accounting presets are derived from the pinned
    # upstream XS..XL sizing tier. Explicit values are checked against that tier.
    system_min_size: int = 3  # recipe minimum is 3
    system_preset: str | None = None
    controller_preset: str | None = None
    login_preset: str = "16vcpu-64gb"

    workers: list[WorkerPoolSpec] = field(default_factory=list)

    # Toggles that keep the deploy small and working out of the box.
    accounting: bool = False
    telemetry: bool = False
    # Custom AppArmor profile is not loaded by the verified chart contract;
    # unconfined (use_default_apparmor_profile=false) keeps login/worker sshd
    # starting. Default off for reliability; opt in if your build loads it.
    use_default_apparmor_profile: bool = False
    jail_size_gib: int = 512
    slurm_operator_version: str = DEFAULT_SLURM_OPERATOR_VERSION
    # These defaults are part of the immutable solutions-library contract.
    k8s_version: str = DEFAULT_K8S_VERSION
    node_group_version: str = DEFAULT_NODE_GROUP_VERSION

    # New fields are declared last so existing positional SDK construction keeps
    # its historical meaning. The canonical key name makes the root grant clear.
    root_login_ssh_public_key: str = ""
    # Compatibility migration: the old NPA renderer fixed max_size to min_size.
    # Omission now preserves the pinned upstream autoscaling ceiling of 24 while
    # explicit values remain available for operator-controlled capacity/cost.
    system_max_size: int | None = None
    accounting_preset: str | None = None
    # ``None`` preserves the historical default (follow accounting) while an
    # explicit bool makes the public contract independent. The pinned 4.1.6
    # operator still cannot reconcile REST without accounting; NPA therefore
    # performs its own direct GPU creation check when accounting is disabled.
    slurm_rest_enabled: bool | None = None

    def explicit_root_login_ssh_public_key(self) -> str:
        """Return the canonical or legacy explicit root-login key, if supplied."""

        if self.root_login_ssh_public_key:
            return self.root_login_ssh_public_key
        return self.ssh_public_keys[0] if self.ssh_public_keys else ""

    def effective_slurm_rest_enabled(self) -> bool:
        """Resolve the backward-compatible REST default independently."""

        if self.slurm_rest_enabled is None:
            return self.accounting
        return self.slurm_rest_enabled

    def effective_system_max_size(self) -> int:
        """Return the rendered autoscaling ceiling (visible in plans/results)."""

        if self.system_max_size is not None:
            return self.system_max_size
        return max(self.system_min_size, 24)

    def validate(self) -> None:
        if not self.name or not self.name.replace("-", "").isalnum():
            raise SoperatorSpecError(f"cluster name must be alphanumeric/dash: {self.name!r}")
        if self.system_min_size < 3:
            raise SoperatorSpecError("system_min_size must be >= 3 (recipe rule)")
        if self.system_max_size is not None and self.system_max_size < self.system_min_size:
            raise SoperatorSpecError(
                "control_plane.system.max_size must be >= min_size"
            )
        if not self.workers:
            raise SoperatorSpecError("at least one worker pool is required")
        if not self.k8s_version or not self.node_group_version:
            raise SoperatorSpecError(
                "k8s_version and node_group_version must both be non-empty"
            )
        if not _OPERATOR_VERSION_RE.fullmatch(self.slurm_operator_version):
            raise SoperatorSpecError(
                "slurm_operator_version must be an explicit semantic version "
                "such as '4.1.6'"
            )
        if self.slurm_operator_version not in _VERIFIED_OPERATOR_VERSIONS:
            supported = ", ".join(sorted(_VERIFIED_OPERATOR_VERSIONS))
            if self.slurm_operator_version == "4.1.0":
                raise SoperatorSpecError(
                    "slurm_operator_version 4.1.0 is no longer verified; replace it "
                    f"with {DEFAULT_SLURM_OPERATOR_VERSION} for the default unconfined "
                    "Enroot/Pyxis setup. Version 4.1.7 is accepted only with "
                    "use_default_apparmor_profile=true after separately validating that "
                    "profile on the nodes; 4.1.0 is not safe to re-enable"
                )
            raise SoperatorSpecError(
                "slurm_operator_version is outside the pinned runtime contract; "
                f"use {DEFAULT_SLURM_OPERATOR_VERSION} for the default unconfined "
                f"Enroot/Pyxis setup; verified overrides are: {supported} (4.1.7 "
                "requires use_default_apparmor_profile=true)"
            )
        if (
            not self.use_default_apparmor_profile
            and self.slurm_operator_version not in _VERIFIED_USERNS_OPERATOR_VERSIONS
        ):
            raise SoperatorSpecError(
                "the unconfined Enroot/Pyxis user-namespace override is verified only "
                f"with slurm_operator_version {DEFAULT_SLURM_OPERATOR_VERSION}; use "
                "that version or explicitly enable use_default_apparmor_profile for "
                "a separately validated chart"
            )
        if self.root_login_ssh_public_key and self.ssh_public_keys:
            raise SoperatorSpecError(
                "set only root_login_ssh_public_key; ssh_public_keys is its legacy alias"
            )
        if len(self.ssh_public_keys) > 1:
            raise SoperatorSpecError(
                "ssh_public_keys accepts exactly one root-login public-key record; "
                "use root_login_ssh_public_key"
            )
        explicit_root_key = self.explicit_root_login_ssh_public_key()
        if explicit_root_key:
            validate_ssh_public_key_record(explicit_root_key)
        seen: set[str] = set()
        for pool in self.workers:
            pool.validate()
            if pool.name in seen:
                raise SoperatorSpecError(f"duplicate worker pool name: {pool.name}")
            seen.add(pool.name)

        worker_count = sum(pool.size for pool in self.workers)
        tier = sizing_tier_for_worker_count(worker_count)
        _validate_role_preset("system", self.system_preset, tier)
        _validate_role_preset("controller", self.controller_preset, tier)
        if self.accounting:
            _validate_role_preset("accounting", self.accounting_preset, tier)
        _validate_role_preset("login", self.login_preset, tier)
        if self.effective_slurm_rest_enabled() and not self.accounting:
            raise SoperatorSpecError(
                "slurm_rest_enabled=true requires accounting=true with the pinned "
                f"Slurm operator {self.slurm_operator_version} runtime contract; the "
                "verified controller skips REST reconciliation "
                "without an accounting database. GPU creation checks do not require "
                "accounting because NPA runs them directly through the login-node jail"
            )


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
                capacity_block_group=str(
                    entry.get("capacity_block_group", "") or ""
                ).strip(),
                capacity_block_group_name=str(
                    entry.get("capacity_block_group_name", "") or ""
                ).strip(),
            )
        )

    control = data.get("control_plane") or {}
    system = control.get("system") or {}
    controller = control.get("controller") or {}
    accounting_control = control.get("accounting") or {}
    login = control.get("login") or {}

    legacy_keys = data.get("ssh_public_keys") or []
    if not isinstance(legacy_keys, list):
        raise SoperatorSpecError("ssh_public_keys must be a list")
    rest_value = data.get("slurm_rest_enabled")
    if rest_value is not None and not isinstance(rest_value, bool):
        raise SoperatorSpecError("slurm_rest_enabled must be a boolean when set")
    spec = SoperatorSpec(
        name=str(data.get("name", "")),
        region=str(data.get("region", "")),
        tenant_id=str(data.get("tenant_id", "")),
        project_id=str(data.get("project_id", "")),
        subnet_id=str(data.get("subnet_id", "")),
        root_login_ssh_public_key=str(data.get("root_login_ssh_public_key", "") or ""),
        ssh_public_keys=[str(key) for key in legacy_keys],
        system_min_size=int(system.get("min_size", 3)),
        system_max_size=(
            int(system["max_size"]) if system.get("max_size") is not None else None
        ),
        system_preset=(str(system["preset"]) if system.get("preset") is not None else None),
        controller_preset=(
            str(controller["preset"]) if controller.get("preset") is not None else None
        ),
        accounting_preset=(
            str(accounting_control["preset"])
            if accounting_control.get("preset") is not None
            else None
        ),
        login_preset=str(login.get("preset", "16vcpu-64gb")),
        workers=workers,
        accounting=bool(data.get("accounting", False)),
        slurm_rest_enabled=(
            rest_value if rest_value is not None else None
        ),
        telemetry=bool(data.get("telemetry", False)),
        use_default_apparmor_profile=bool(data.get("use_default_apparmor_profile", False)),
        jail_size_gib=int(data.get("jail_size_gib", 512)),
        slurm_operator_version=str(
            data.get("slurm_operator_version", DEFAULT_SLURM_OPERATOR_VERSION)
        ),
        k8s_version=str(data.get("k8s_version", DEFAULT_K8S_VERSION)),
        node_group_version=str(
            data.get("node_group_version", DEFAULT_NODE_GROUP_VERSION)
        ),
    )
    return spec


def load_spec(path: str | Path) -> SoperatorSpec:
    """Load and validate a soperator spec from a YAML file."""

    text = Path(path).expanduser().read_text()
    data = yaml.safe_load(text) or {}
    spec = spec_from_mapping(data)
    spec.validate()
    return spec
