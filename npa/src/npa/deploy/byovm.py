"""Helpers for BYOVM (bring your own VM) workbench deployments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from npa.clients.config import SSHConfig, StorageConfig
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


def select_visible_devices(
    detected_count: int, gpu_count: int | None = None
) -> tuple[int, str]:
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


def deployment_preview_outputs() -> dict[str, str]:
    """Return non-identifying placeholders for a VM deployment preview.

    Args:
        None.

    Returns:
        Redacted connection and configured-storage placeholders.

    Raises:
        None.
    """
    return {
        "vm_ip": "<redacted>",
        "ssh_user": "<redacted>",
        "ssh_key_path": "<redacted>",
        "storage_bucket": "<configured>",
        "storage_endpoint": "<configured>",
    }


def read_existing_workbench_outputs(
    project: str,
    name: str,
    terraform_directory: str,
    use_remote_state: bool,
    deployment_vars: Mapping[str, str],
) -> dict[str, Any]:
    """Read existing VM connection outputs without selecting storage again.

    Args:
        project: Selected project alias.
        name: Selected workbench name.
        terraform_directory: Explicit Terraform working directory, if any.
        use_remote_state: Whether the managed remote-state directory is active.
        deployment_vars: Credentials used only to initialize that state backend.

    Returns:
        Existing VM connection fields, which intentionally omit storage identity.

    Raises:
        None. Unavailable Terraform outputs fall back to saved connection fields.
    """
    outputs = _terraform_connection_outputs(
        project, name, terraform_directory, use_remote_state, deployment_vars
    )
    return outputs if outputs is not None else _saved_connection_outputs(project, name)


def _terraform_connection_outputs(
    project: str,
    name: str,
    terraform_directory: str,
    use_remote_state: bool,
    deployment_vars: Mapping[str, str],
) -> dict[str, Any] | None:
    from npa.deploy import provisioner
    from npa.deploy.provisioner import ProvisionerError

    if terraform_directory:
        try:
            return provisioner.outputs(tf_dir=terraform_directory)
        except ProvisionerError:
            return None
    if not use_remote_state:
        return None
    work_dir = provisioner.working_dir_path(project, name)
    if not work_dir.exists():
        return None
    try:
        provisioner.init(
            tf_dir=str(work_dir),
            backend_config={
                "access_key": deployment_vars.get("nebius_api_key", ""),
                "secret_key": deployment_vars.get("nebius_secret_key", ""),
            },
        )
        return provisioner.outputs(tf_dir=str(work_dir))
    except ProvisionerError:
        return None


def _saved_connection_outputs(project: str, name: str) -> dict[str, Any]:
    from npa.clients.config import (
        _deep_get,
        _load_yaml,
        _resolve_project_section,
        _resolve_workbench_in_project,
    )

    try:
        config = _load_yaml()
        selected_project = _resolve_project_section(config, project)
        workbench = _resolve_workbench_in_project(selected_project, name, config)
    except Exception:
        workbench = {}
    return {
        "vm_ip": _deep_get(workbench, "ssh", "host", default=""),
        "ssh_user": _deep_get(workbench, "ssh", "user", default="ubuntu"),
        "ssh_key_path": _deep_get(
            workbench, "ssh", "key_path", default="~/.ssh/id_ed25519"
        ),
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


def _storage_value(storage: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = storage.get(key)
        if value:
            return str(value)
    return ""


def _exact_project_storage(project_id: str) -> StorageConfig | None:
    from npa.clients.project_credential_store import project_credential_record

    record = project_credential_record(project_id, migrate_legacy=False)
    if record.get("storage_selected") is False:
        raise ValueError(
            "The selected project has no selected object-storage identity."
        )
    saved = record.get("storage")
    if not isinstance(saved, Mapping) or not saved:
        return None
    storage = StorageConfig(
        checkpoint_bucket=_storage_value(
            saved, "checkpoint_bucket", "bucket", "s3_bucket"
        ),
        endpoint_url=_storage_value(saved, "endpoint_url", "endpoint", "s3_endpoint"),
        aws_access_key_id=_storage_value(
            saved, "aws_access_key_id", "access_key", "nebius_api_key"
        ),
        aws_secret_access_key=_storage_value(
            saved, "aws_secret_access_key", "secret_key", "nebius_secret_key"
        ),
    )
    if not all(
        (
            storage.checkpoint_bucket,
            storage.endpoint_url,
            storage.aws_access_key_id,
            storage.aws_secret_access_key,
        )
    ):
        raise ValueError(
            "The selected project has an incomplete exact-project object-storage "
            "identity. Reconfigure storage for this exact project."
        )
    return storage


def _storage_identity(source: Mapping[str, Any]) -> StorageConfig:
    return StorageConfig(
        checkpoint_bucket=_storage_value(
            source, "checkpoint_bucket", "bucket", "s3_bucket"
        ),
        endpoint_url=_storage_value(source, "endpoint_url", "endpoint", "s3_endpoint"),
        aws_access_key_id=_storage_value(
            source, "aws_access_key_id", "access_key", "nebius_api_key"
        ),
        aws_secret_access_key=_storage_value(
            source, "aws_secret_access_key", "secret_key", "nebius_secret_key"
        ),
    )


def _complete_storage_identity(storage: StorageConfig) -> bool:
    return all(
        (
            storage.checkpoint_bucket,
            storage.endpoint_url,
            storage.aws_access_key_id,
            storage.aws_secret_access_key,
        )
    )


def _alias_project_storage(project: str | None) -> StorageConfig:
    from npa.clients.config import _load_yaml, _resolve_project_section

    proj = _resolve_project_section(_load_yaml(), project)
    sources = [
        proj.get("object-storage"),
        proj.get("object_storage"),
        proj.get("storage"),
        proj.get("terraform_state"),
    ]
    identities: list[StorageConfig] = []
    for source in sources:
        if source in (None, {}):
            continue
        if not isinstance(source, Mapping):
            raise ValueError(
                "The selected project has an invalid object-storage identity."
            )
        identity = _storage_identity(source)
        if not _complete_storage_identity(identity):
            raise ValueError(
                "The selected project has an incomplete object-storage identity."
            )
        identities.append(identity)
    if not identities:
        raise ValueError(
            "The selected project has no complete selected object-storage identity."
        )
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError(
            "The selected project has conflicting object-storage identities."
        )
    return identities[0]


_STORAGE_VAR_NAMES = (
    "s3_bucket",
    "s3_endpoint",
    "nebius_api_key",
    "nebius_secret_key",
)


def _storage_var_mapping(storage: StorageConfig) -> dict[str, str]:
    return {
        "s3_bucket": storage.checkpoint_bucket,
        "s3_endpoint": storage.endpoint_url,
        "nebius_api_key": storage.aws_access_key_id,
        "nebius_secret_key": storage.aws_secret_access_key,
    }


def _explicit_storage_identity(explicit_vars: Mapping[str, str]) -> dict[str, str]:
    explicit = {
        key: explicit_vars[key] for key in _STORAGE_VAR_NAMES if key in explicit_vars
    }
    if not explicit:
        return {}
    missing = [key for key in _STORAGE_VAR_NAMES if not explicit.get(key)]
    if missing:
        raise ValueError(
            "Explicit object-storage overrides must provide one complete identity "
            f"(missing: {', '.join(missing)})."
        )
    return explicit


def _strict_project_storage(
    project: str | None,
    *,
    project_id: str,
    configured_project_id: str,
) -> StorageConfig:
    storage = _exact_project_storage(project_id) if project_id else None
    if storage is not None:
        return storage
    if project_id and configured_project_id and configured_project_id != project_id:
        raise ValueError(
            "The selected project has no exact storage identity, and the alias "
            "storage belongs to a different project."
        )
    if project_id:
        raise ValueError(
            "The selected project has no exact object-storage identity. "
            "Configure storage for this exact project before deploying."
        )
    return _alias_project_storage(project)


def _apply_storage_mapping(
    merged_vars: dict[str, str],
    mapping: Mapping[str, str],
    *,
    explicit_vars: Mapping[str, str],
    replace_existing: bool,
) -> bool:
    found = any(mapping.values())
    for key, value in mapping.items():
        if (
            value
            and key not in explicit_vars
            and (replace_existing or not merged_vars.get(key))
        ):
            merged_vars[key] = value
    return found


def apply_project_storage_vars(
    merged_vars: dict[str, str],
    *,
    project: str | None,
    explicit_vars: Mapping[str, str],
    warn: Any | None = None,
    require_complete: bool = False,
    project_id: str = "",
    configured_project_id: str = "",
) -> bool:
    """Apply project-level storage settings to a BYOVM deploy var map.

    Args:
        merged_vars: Deployment variables to update.
        project: Selected project alias.
        explicit_vars: Variables supplied explicitly by the caller.
        warn: Optional compatibility warning sink.
        require_complete: Whether to require one atomic identity.
        project_id: Exact selected provider project ID.
        configured_project_id: Provider project ID owning alias storage.

    Returns:
        Whether the selected resolver returned any storage settings.

    Raises:
        ValueError: Strict selection did not resolve one complete identity.
    """
    return _apply_resolved_project_storage(
        merged_vars,
        project=project,
        explicit_vars=explicit_vars,
        warn=warn,
        require_complete=require_complete,
        project_id=project_id,
        configured_project_id=configured_project_id,
    )


def _apply_resolved_project_storage(
    merged_vars: dict[str, str],
    *,
    project: str | None,
    explicit_vars: Mapping[str, str],
    warn: Any | None,
    require_complete: bool,
    project_id: str,
    configured_project_id: str,
) -> bool:
    explicit_storage = (
        _explicit_storage_identity(explicit_vars) if require_complete else {}
    )
    if explicit_storage:
        merged_vars.update(explicit_storage)
        return True
    if require_complete:
        storage = _strict_project_storage(
            project,
            project_id=project_id,
            configured_project_id=configured_project_id,
        )
    else:
        from npa.clients.config import resolve_project_storage

        storage = resolve_project_storage(
            project,
            include_shared_credentials=True,
            include_environment=True,
        )
    mapping = _storage_var_mapping(storage)
    missing = [key for key, value in mapping.items() if not value]
    if require_complete and missing:
        missing_names = ", ".join(missing)
        raise ValueError(
            "The selected project has no complete selected object-storage identity "
            f"(missing: {missing_names}). Configure storage for this exact project "
            "before deploying."
        )
    found = _apply_storage_mapping(
        merged_vars,
        mapping,
        explicit_vars=explicit_vars,
        replace_existing=require_complete,
    )
    if not found and warn is not None:
        warn(
            f"Warning: Project {project} has no object-storage settings. "
            "S3 operations on this workbench will fail unless configured manually."
        )
    return found


def ssh_config_for_target(
    target: BYOVMTarget, *, tokens: dict[str, str] | None = None
) -> SSHConfig:
    return SSHConfig(
        host=target.host,
        user=target.user,
        key_path=target.key_path,
        tokens=tokens or {},
    )
