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
    gpu_readiness: str = "not-requested"
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
    gpu_nodes: int = -1,
    cpu_nodes: int = -1,
    cpu_platform: str = "",
    cpu_preset: str = "",
    gpu_platform: str = "",
    gpu_preset: str = "",
    preemptible: bool | None = None,
    accelerator: str = "",
    gpu_readiness_timeout: float = 600.0,
    gpu_readiness_poll_interval: float = 10.0,
    sky_bin: str = "",
) -> ProvisionIfAbsentResult:
    """Ensure configured S3 and Kubernetes exist without deleting resources."""
    alias, environment, storage, registry = _resolve_project_runtime(project)
    context = context_name.strip() or cluster_name
    kubeconfig_path = kubeconfig or kubeconfig_file(context)
    actions: list[str] = []
    warnings: list[str] = []
    k8s_ready = False
    gpu_readiness = "not-requested"

    storage_ready = skip_s3
    storage_bucket = storage.checkpoint_bucket
    storage_endpoint = storage.endpoint_url
    if skip_s3:
        actions.append("s3:skipped")
    else:
        from npa.clients.nebius import bucket_name_for, redact_nebius_output
        from npa.clients.storage_setup import provision_storage, storage_setup_record
        from npa.clients.storage_validation import probe_storage_write

        bucket_name = _bucket_name(storage.checkpoint_bucket)
        partial = (
            storage_setup_record(environment.project_id)
            if environment.project_id
            else {}
        )
        bucket_name = bucket_name or str(partial.get("bucket_name", "") or "").strip()
        if not bucket_name and environment.project_id and environment.tenant_id:
            bucket_name = bucket_name_for(environment.tenant_id, environment.project_id)
        if not environment.project_id:
            warnings.append("project_id is required to ensure S3")
        elif not environment.tenant_id:
            warnings.append("tenant_id is required to ensure S3")
        elif dry_run:
            action = (
                "reconcile"
                if partial and partial.get("status") != "complete"
                else "ensure"
            )
            actions.append(f"s3:dry-run {action} writable bucket {bucket_name}")
            storage_ready = True
        else:
            probe = probe_storage_write(
                bucket=storage.checkpoint_bucket,
                endpoint_url=storage.endpoint_url,
                access_key_id=storage.aws_access_key_id,
                secret_access_key=storage.aws_secret_access_key,
                region=environment.region,
            )
            if probe.ok:
                storage_ready = True
                actions.append(f"s3:verified writable bucket {bucket_name}")
            else:
                try:
                    credentials, reconciled_probe = provision_storage(
                        project_id=environment.project_id,
                        tenant_id=environment.tenant_id,
                        region=environment.region,
                        bucket_name=bucket_name,
                        project_alias=alias,
                    )
                except Exception as exc:  # noqa: BLE001 - return an actionable partial result
                    warnings.append(
                        "writable S3 reconciliation failed: "
                        + redact_nebius_output(str(exc))
                    )
                    actions.append(
                        "s3:partial; resume with `npa provision-if-absent"
                        + (f" --project {alias}" if alias else "")
                        + " --skip-k8s`"
                    )
                else:
                    storage_ready = reconciled_probe.ok
                    storage_bucket = f"s3://{credentials['s3_bucket'].strip().removeprefix('s3://').strip('/')}/"
                    storage_endpoint = credentials["s3_endpoint"]
                    # The transaction committed the newly validated credentials;
                    # re-resolve so a following Terraform apply receives them.
                    storage = config_module.resolve_project_storage(alias or None)
                    actions.append(f"s3:reconciled writable bucket {bucket_name}")

    if not storage_ready and not skip_s3:
        actions.append("k8s:blocked until writable S3 is reconciled")
    elif skip_k8s:
        actions.append("k8s:skipped")
    elif _has_cached_kubeconfig(context, kubeconfig_path):
        actions.append(f"k8s:reused kubeconfig {kubeconfig_path}")
        k8s_ready = True
    elif not environment.project_id or not environment.tenant_id:
        warnings.append("project_id and tenant_id are required to ensure Kubernetes")
    elif dry_run:
        shape = ", ".join(
            [
                f"{name}={count}"
                for name, count in (("gpu_nodes", gpu_nodes), ("cpu_nodes", cpu_nodes))
                if count >= 0
            ]
            + ([f"preemptible={str(bool(preemptible)).lower()}"] if preemptible is not None else [])
            + [
                f"{name}={value.strip()}"
                for name, value in (
                    ("cpu_platform", cpu_platform),
                    ("cpu_preset", cpu_preset),
                    ("gpu_platform", gpu_platform),
                    ("gpu_preset", gpu_preset),
                )
                if value.strip()
            ]
        )
        actions.append(
            f"k8s:dry-run terraform apply {terraform_dir or 'deploy/cluster'}"
            + (f" ({shape})" if shape else "")
        )
    else:
        with _runtime_env(alias, environment, storage, registry):
            from npa.cli.cluster.terraform_lifecycle import up_cmd

            # Every parameter is passed explicitly: `up_cmd` is a Typer command,
            # so an omitted one arrives as an OptionInfo sentinel rather than its
            # default (the §23 bug), and `gpu_nodes`/`cpu_nodes` would reach the
            # Terraform override as objects.
            up_cmd(
                terraform_dir=terraform_dir,
                kubeconfig=kubeconfig_path,
                context_name=context,
                project=alias or "",
                validate=validate,
                sky_smoke=sky_smoke,
                sky_gpus="",
                capacity_block_group="",
                gpu_nodes=gpu_nodes,
                cpu_nodes=cpu_nodes,
                cpu_platform=cpu_platform,
                cpu_preset=cpu_preset,
                gpu_platform=gpu_platform,
                gpu_preset=gpu_preset,
                preemptible=preemptible,
                validation_timeout=60,
                timeout=timeout,
            )
        actions.append(f"k8s:ensured terraform cluster {context}")
        k8s_ready = True

    requested_accelerator = str(accelerator or "").strip()
    if requested_accelerator and k8s_ready and not skip_k8s and not dry_run:
        from npa.orchestration.skypilot.k8s_gpu_catalog import (
            KubernetesGpuCatalogError,
            wait_for_kubernetes_accelerators,
        )

        previous_kubeconfig = os.environ.get("KUBECONFIG")
        os.environ["KUBECONFIG"] = str(kubeconfig_path)
        try:
            wait_for_kubernetes_accelerators(
                [requested_accelerator],
                context=context,
                sky_bin=sky_bin or None,
                timeout=gpu_readiness_timeout,
                poll_interval=gpu_readiness_poll_interval,
                on_status=lambda message: actions.append(f"gpu:{message}"),
            )
        except (KubernetesGpuCatalogError, ValueError) as exc:
            gpu_readiness = "timeout"
            warnings.append(str(exc))
            actions.append("gpu:capacity preserved; resume readiness without reprovisioning")
        else:
            gpu_readiness = "ready"
            actions.append(f"gpu:SkyPilot ready for {requested_accelerator}")
        finally:
            if previous_kubeconfig is None:
                os.environ.pop("KUBECONFIG", None)
            else:
                os.environ["KUBECONFIG"] = previous_kubeconfig
    elif requested_accelerator and dry_run:
        gpu_readiness = "dry-run"
        actions.append(f"gpu:dry-run wait for SkyPilot {requested_accelerator}")

    status = "ok" if not warnings and (skip_s3 or storage_ready) else "partial"
    return ProvisionIfAbsentResult(
        status=status,
        project=alias,
        cluster_name=cluster_name,
        kubeconfig_path=str(kubeconfig_path),
        context_name=context,
        storage_bucket=storage_bucket,
        storage_endpoint=storage_endpoint,
        registry=registry,
        gpu_readiness=gpu_readiness,
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
