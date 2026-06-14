"""Additive-only runtime provisioning hooks."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

from npa.clients import config as config_module
from npa.clients.config import ConfigError, EnvironmentConfig, StorageConfig
from npa.clients.nebius import ensure_bucket
from npa.cluster.state import kubeconfig_file, load_cluster_state


@dataclass
class ProvisionIfAbsentResult:
    status: str
    project: str
    cluster_name: str
    kubeconfig_path: str = ""
    context_name: str = ""
    storage_bucket: str = ""
    storage_endpoint: str = ""
    registry: str = ""
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def provision_if_absent(
    *,
    project: str | None = None,
    cluster_name: str = "npa-cluster",
    terraform_dir: Path | None = None,
    kubeconfig: Path | None = None,
    context_name: str = "",
    skip_k8s: bool = False,
    skip_s3: bool = False,
    validate: bool = True,
    sky_smoke: bool = False,
    dry_run: bool = False,
    timeout: int = 120,
) -> ProvisionIfAbsentResult:
    """Ensure configured S3 and Kubernetes exist, without teardown or mutation."""
    alias, environment, storage, registry = _resolve_project_runtime(project)
    context = context_name.strip() or cluster_name
    kubeconfig_path = kubeconfig or kubeconfig_file(context)
    actions: list[str] = []
    warnings: list[str] = []

    if skip_s3:
        actions.append("s3:skipped")
    else:
        bucket_name = _bucket_name(storage.checkpoint_bucket)
        if not bucket_name:
            warnings.append("s3 bucket is not configured")
        elif not environment.project_id:
            warnings.append("project_id is required to ensure S3")
        elif dry_run:
            actions.append(f"s3:dry-run ensure bucket {bucket_name}")
        else:
            ensure_bucket(environment.project_id, bucket_name)
            actions.append(f"s3:ensured bucket {bucket_name}")

    if skip_k8s:
        actions.append("k8s:skipped")
    elif _has_cached_kubeconfig(context, kubeconfig_path):
        actions.append(f"k8s:reused kubeconfig {kubeconfig_path}")
    elif not environment.project_id or not environment.tenant_id:
        warnings.append("project_id and tenant_id are required to ensure Kubernetes")
    elif dry_run:
        actions.append(f"k8s:dry-run terraform apply {terraform_dir or 'deploy/cluster'}")
    else:
        with _runtime_env(alias, environment, storage, registry):
            from npa.cli.cluster.terraform_lifecycle import up_cmd

            up_cmd(
                terraform_dir=terraform_dir,
                kubeconfig=kubeconfig_path,
                context_name=context,
                validate=validate,
                sky_smoke=sky_smoke,
                timeout=timeout,
            )
        actions.append(f"k8s:ensured terraform cluster {context}")

    status = "ok" if not warnings else "partial"
    return ProvisionIfAbsentResult(
        status=status,
        project=alias,
        cluster_name=cluster_name,
        kubeconfig_path=str(kubeconfig_path),
        context_name=context,
        storage_bucket=storage.checkpoint_bucket,
        storage_endpoint=storage.endpoint_url,
        registry=registry,
        actions=actions,
        warnings=warnings,
    )


def _resolve_project_runtime(
    project: str | None,
) -> tuple[str, EnvironmentConfig, StorageConfig, str]:
    yml = config_module._load_yaml()
    alias = config_module._resolved_project_name(yml, project)
    environment = config_module.resolve_environment(project) or EnvironmentConfig("", "", "")
    storage = config_module.resolve_project_storage(project)
    registry = config_module.resolve_container_registry(project)
    return alias, environment, storage, registry


def _bucket_name(uri_or_name: str) -> str:
    value = uri_or_name.strip()
    if not value:
        return ""
    if value.startswith("s3://"):
        return urlparse(value).netloc
    return value.split("/", 1)[0]


def _has_cached_kubeconfig(context: str, kubeconfig_path: Path) -> bool:
    if kubeconfig_path.exists():
        return True
    state = load_cluster_state(context)
    return bool(state and state.kubeconfig_path and Path(state.kubeconfig_path).exists())


@contextmanager
def _runtime_env(
    alias: str,
    environment: EnvironmentConfig,
    storage: StorageConfig,
    registry: str,
) -> Iterator[None]:
    yml = config_module._load_yaml()
    registry_id = ""
    try:
        proj = config_module._resolve_project_section(yml, alias)
        if isinstance(proj, dict):
            registry_id = str(proj.get("registry_id", "") or "")
    except ConfigError:
        pass

    values = {
        "NPA_PROJECT_ID": environment.project_id,
        "NPA_TENANT_ID": environment.tenant_id,
        "NPA_REGION": environment.region,
        "NPA_REGISTRY": registry,
        "NPA_REGISTRY_ID": registry_id,
        "NPA_S3_BUCKET": storage.checkpoint_bucket,
        "NPA_STORAGE_ENDPOINT": storage.endpoint_url,
        "AWS_ENDPOINT_URL": storage.endpoint_url,
        "NEBIUS_S3_ENDPOINT": storage.endpoint_url,
        "AWS_ACCESS_KEY_ID": storage.aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": storage.aws_secret_access_key,
        "TF_VAR_parent_id": environment.project_id,
        "TF_VAR_tenant_id": environment.tenant_id,
        "TF_VAR_region": environment.region,
    }
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
