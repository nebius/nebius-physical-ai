"""Helpers for BYOVM (bring your own VM) workbench deployments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from npa.clients.config import SSHConfig
from npa.clients.credentials import CredentialsConfig, load_credentials
from npa.clients.ssh import SSHClient, SSHError

RUNTIME_BYOVM = "byovm"
RUNTIME_CONTAINER = "container"
RUNTIME_VM = "vm"
RUNTIME_HELP = (
    "Application runtime: vm provisions a managed VM, container provisions a "
    "managed VM and runs a container, byovm skips Terraform and deploys the "
    "app to an existing SSH-accessible VM. BYOVM does not manage VM "
    "lifecycle."
)


@dataclass(frozen=True)
class BYOVMTarget:
    host: str
    user: str
    key_path: str


@dataclass(frozen=True)
class GPUInfo:
    count: int
    names: list[str]

    @property
    def primary_name(self) -> str:
        return self.names[0] if self.names else "unknown"


def runtime_value(runtime: Any) -> str:
    return str(getattr(runtime, "value", runtime))


def is_byovm_runtime(runtime: Any) -> bool:
    return runtime_value(runtime) == RUNTIME_BYOVM


def runtime_uses_container(runtime: Any) -> bool:
    return runtime_value(runtime) in {RUNTIME_CONTAINER, RUNTIME_BYOVM}


def resolve_byovm_target(
    *,
    host: str = "",
    ssh_key: str = "",
    ssh_user: str = "",
    credentials: CredentialsConfig | None = None,
    environ: Mapping[str, str] | None = None,
) -> BYOVMTarget:
    """Resolve BYOVM SSH target from CLI flags, env vars, then credentials."""
    env = environ if environ is not None else os.environ
    creds = credentials or load_credentials(environ=env)

    resolved_host = (
        host
        or env.get("NPA_BYOVM_HOST", "")
        or env.get("NPA_SSH_HOST", "")
        or creds.ssh_host
    )
    resolved_user = (
        ssh_user
        or env.get("NPA_BYOVM_SSH_USER", "")
        or env.get("NPA_SSH_USER", "")
        or creds.ssh_user
        or "ubuntu"
    )
    resolved_key = (
        ssh_key
        or env.get("NPA_BYOVM_SSH_KEY", "")
        or env.get("NPA_SSH_KEY", "")
        or creds.ssh_key_path
    )

    missing: list[str] = []
    if not resolved_host:
        missing.append("--host or NPA_BYOVM_HOST")
    if not resolved_key:
        missing.append("--ssh-key or NPA_BYOVM_SSH_KEY")
    if missing:
        raise ValueError(
            "BYOVM target is incomplete. Provide "
            + " and ".join(missing)
            + ", or configure ssh.host and ssh.key_path in ~/.npa/credentials.yaml."
        )

    return BYOVMTarget(host=resolved_host, user=resolved_user, key_path=resolved_key)


def detect_gpu_info(ssh: SSHClient) -> GPUInfo:
    """Detect GPUs with nvidia-smi on the target VM."""
    _, out, _ = ssh.run_or_raise(
        "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null"
    )
    names = [line.strip() for line in out.splitlines() if line.strip()]
    if not names:
        raise SSHError("nvidia-smi returned no GPUs on the BYOVM target")
    return GPUInfo(count=len(names), names=names)


def select_visible_devices(detected_count: int, gpu_count: int | None = None) -> tuple[int, str]:
    """Return effective GPU count and CUDA_VISIBLE_DEVICES."""
    if detected_count <= 0:
        raise ValueError("No GPUs were detected on the BYOVM target")
    effective = detected_count if gpu_count in (None, 0) else int(gpu_count)
    if effective <= 0:
        raise ValueError(f"--gpu-count must be positive, got {gpu_count}")
    if effective > detected_count:
        raise ValueError(
            f"--gpu-count {effective} exceeds detected GPU count {detected_count}"
        )
    return effective, ",".join(str(i) for i in range(effective))


def gpu_config_fields(
    info: GPUInfo | None,
    *,
    effective_count: int | None,
    visible_devices: str,
) -> dict[str, Any]:
    if info is None:
        return {}
    return {
        "gpu_platform": info.primary_name,
        "gpu_preset": f"{effective_count or info.count}gpu-byovm",
        "gpu_count": effective_count or info.count,
        "detected_gpu_count": info.count,
        "detected_gpu_names": info.names,
        "cuda_visible_devices": visible_devices,
        "managed_lifecycle": False,
    }


def gpu_env_fields(
    info: GPUInfo | None,
    *,
    effective_count: int | None,
    visible_devices: str,
) -> dict[str, str]:
    if not visible_devices:
        return {}
    env = {
        "CUDA_VISIBLE_DEVICES": visible_devices,
        "NPA_GPU_COUNT": str(effective_count or (info.count if info else "")),
    }
    if info is not None:
        env["NPA_DETECTED_GPU_COUNT"] = str(info.count)
        env["NPA_GPU_TYPE"] = info.primary_name.replace(" ", "_")
    return {k: v for k, v in env.items() if v}


def workbench_storage_outputs(
    *,
    target: BYOVMTarget,
    bucket: str = "",
    endpoint: str = "",
) -> dict[str, str]:
    return {
        "vm_ip": target.host,
        "ssh_user": target.user,
        "ssh_key_path": target.key_path,
        "storage_bucket": bucket,
        "storage_endpoint": endpoint,
    }


def apply_storage_env_vars(
    merged_vars: dict[str, str],
    *,
    explicit_vars: Mapping[str, str],
) -> None:
    """Apply storage-related environment variables unless CLI vars were explicit."""
    env_mapping = {
        "s3_bucket": "NPA_CHECKPOINT_BUCKET",
        "s3_endpoint": "AWS_ENDPOINT_URL",
        "nebius_api_key": "AWS_ACCESS_KEY_ID",
        "nebius_secret_key": "AWS_SECRET_ACCESS_KEY",
    }
    for key, env_name in env_mapping.items():
        value = os.environ.get(env_name, "")
        if value and key not in explicit_vars:
            merged_vars[key] = value


def apply_project_storage_vars(
    merged_vars: dict[str, str],
    *,
    project: str | None,
    explicit_vars: Mapping[str, str],
    warn: Any | None = None,
) -> bool:
    """Apply project-level storage settings to a BYOVM deploy var map."""
    from npa.clients.config import resolve_project_storage

    storage = resolve_project_storage(project)
    mapping = {
        "s3_bucket": storage.checkpoint_bucket,
        "s3_endpoint": storage.endpoint_url,
        "nebius_api_key": storage.aws_access_key_id,
        "nebius_secret_key": storage.aws_secret_access_key,
    }
    found = any(mapping.values())
    if not found and warn is not None:
        warn(
            f"Warning: Project {project} has no object-storage settings. "
            "S3 operations on this workbench will fail unless configured manually."
        )
    for key, value in mapping.items():
        if value and key not in explicit_vars and not merged_vars.get(key):
            merged_vars[key] = value
    return found


def ssh_config_for_target(target: BYOVMTarget, *, tokens: dict[str, str] | None = None) -> SSHConfig:
    return SSHConfig(
        host=target.host,
        user=target.user,
        key_path=target.key_path,
        tokens=tokens or {},
    )
