"""Configuration defaults and validation for ``npa cluster``."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace

from npa.clients.config import resolve_environment
from npa.cluster.exceptions import ClusterConfigError

DEFAULT_REGION = "eu-north1"
DEFAULT_K8S_VERSION = "1.33"
DEFAULT_NODE_PLATFORM = "cpu-e2"
DEFAULT_NODE_PRESET = "2vcpu-8gb"
DEFAULT_NODE_COUNT = 1
DEFAULT_BOOT_DISK_TYPE = "network_ssd"
DEFAULT_BOOT_DISK_SIZE_GIB = 128

SUPPORTED_REGIONS = {DEFAULT_REGION}
SUPPORTED_NODE_PRESETS = {
    "cpu-e2": {
        "2vcpu-8gb",
        "4vcpu-16gb",
        "8vcpu-32gb",
        "16vcpu-64gb",
        "32vcpu-128gb",
        "48vcpu-192gb",
        "64vcpu-256gb",
        "80vcpu-320gb",
    },
    "cpu-d3": {
        "4vcpu-16gb",
        "8vcpu-32gb",
        "16vcpu-64gb",
        "32vcpu-128gb",
        "48vcpu-192gb",
        "64vcpu-256gb",
        "96vcpu-384gb",
        "128vcpu-512gb",
    },
}

_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_K8S_VERSION_RE = re.compile(r"^\d+\.\d+$")
_PROJECT_ENV_VARS = (
    "NPA_CLUSTER_PROJECT_ID",
    "NPA_PROJECT_ID",
    "NPA_E2E_SERVERLESS_PROJECT",
    "NEBIUS_PROJECT_ID",
)


@dataclass(frozen=True)
class ClusterConfig:
    """Validated cluster deployment configuration."""

    name: str
    project_id: str = ""
    region: str = DEFAULT_REGION
    node_count: int = DEFAULT_NODE_COUNT
    node_platform: str = DEFAULT_NODE_PLATFORM
    node_preset: str = DEFAULT_NODE_PRESET
    k8s_version: str = DEFAULT_K8S_VERSION
    subnet_id: str = ""
    wait: bool = True
    timeout_minutes: int = 30
    poll_interval_seconds: float = 30.0
    boot_disk_type: str = DEFAULT_BOOT_DISK_TYPE
    boot_disk_size_gib: int = DEFAULT_BOOT_DISK_SIZE_GIB
    public_node_ip: bool = True

    def __post_init__(self) -> None:
        validate_cluster_name(self.name)
        validate_region(self.region)
        validate_node_shape(self.node_platform, self.node_preset)
        if self.node_count < 1 or self.node_count > 100:
            raise ClusterConfigError("--node-count must be between 1 and 100")
        if not _K8S_VERSION_RE.match(self.k8s_version):
            raise ClusterConfigError("--k8s-version must use <major>.<minor> format")
        if self.timeout_minutes < 1:
            raise ClusterConfigError("--timeout must be at least 1 minute")
        if self.poll_interval_seconds <= 0:
            raise ClusterConfigError("poll interval must be positive")
        if self.boot_disk_size_gib < 32:
            raise ClusterConfigError("boot disk size must be at least 32 GiB")

    def with_project_id(self, project_id: str) -> "ClusterConfig":
        return replace(self, project_id=project_id)

    def with_subnet_id(self, subnet_id: str) -> "ClusterConfig":
        return replace(self, subnet_id=subnet_id)


def validate_cluster_name(name: str) -> None:
    if not name or not _NAME_RE.match(name):
        raise ClusterConfigError(
            "cluster name must be 1-63 characters, using letters, numbers, and hyphens, "
            "and must not start or end with a hyphen"
        )


def validate_region(region: str) -> None:
    if region not in SUPPORTED_REGIONS:
        allowed = ", ".join(sorted(SUPPORTED_REGIONS))
        raise ClusterConfigError(f"unsupported region '{region}'. Supported regions: {allowed}")


def validate_node_shape(platform: str, preset: str) -> None:
    presets = SUPPORTED_NODE_PRESETS.get(platform)
    if not presets:
        allowed = ", ".join(sorted(SUPPORTED_NODE_PRESETS))
        raise ClusterConfigError(f"unsupported CPU node platform '{platform}'. Supported: {allowed}")
    if preset not in presets:
        allowed = ", ".join(sorted(presets))
        raise ClusterConfigError(
            f"unsupported preset '{preset}' for platform '{platform}'. Supported: {allowed}"
        )


def resolve_project_id(explicit_project_id: str = "") -> str:
    """Resolve the Nebius project ID from CLI, env, or saved NPA config."""

    if explicit_project_id.strip():
        return explicit_project_id.strip()
    for env_var in _PROJECT_ENV_VARS:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    env_cfg = resolve_environment()
    if env_cfg and env_cfg.project_id:
        return env_cfg.project_id
    raise ClusterConfigError(
        "Nebius project ID is required. Pass --project-id or configure a default project in ~/.npa/config.yaml."
    )


DEFAULT_CLUSTER_CONFIG = ClusterConfig(name="npa-cluster-default")
