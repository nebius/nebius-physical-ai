"""Additive-only runtime provisioning hooks."""

from __future__ import annotations

import os
import functools
import inspect
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

from npa.clients import config as config_module
from npa.clients.config import ConfigError, EnvironmentConfig, StorageConfig
from npa.cluster.gpu_driver import DEFAULT_MANAGED_DRIVER_PRESET
from npa.cluster.gpu_health import DEFAULT_CUDA_SMOKE_IMAGE, DEFAULT_STABILIZATION_SECONDS
from npa.cluster.state import kubeconfig_file, load_cluster_state
from npa.provisioning_journal import (
    ProvisioningOperation,
    current_operation,
    emit_recovery_summary,
    operation_context,
)


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
    operation_id: str = ""
    operation_journal: str = ""
    recovery_command: str = ""
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preflight: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _provision_recovery_argv(
    arguments: dict[str, Any],
    *,
    alias: str,
    cluster_name: str,
    context: str,
    kubeconfig: Path,
) -> list[str]:
    """Serialize every effective, non-secret topology option for exact resume."""

    argv = [
        "npa",
        "provision-if-absent",
        "--cluster-name",
        cluster_name,
        "--context",
        context,
        "--kubeconfig",
        str(kubeconfig),
        "--timeout",
        str(int(arguments.get("timeout") or 120)),
        "--gpu-nodes",
        str(int(arguments.get("gpu_nodes", -1))),
        "--cpu-nodes",
        str(int(arguments.get("cpu_nodes", -1))),
        "--gpu-readiness-timeout",
        str(float(arguments.get("gpu_readiness_timeout") or 600.0)),
        "--gpu-readiness-poll-interval",
        str(float(arguments.get("gpu_readiness_poll_interval") or 10.0)),
        "--gpu-health-stabilization-seconds",
        str(
            int(
                arguments.get("gpu_health_stabilization_seconds")
                if arguments.get("gpu_health_stabilization_seconds") is not None
                else DEFAULT_STABILIZATION_SECONDS
            )
        ),
        "--output-format",
        str(arguments.get("output_format") or "text"),
    ]
    if alias:
        argv.extend(["--project", alias])
    terraform_dir = arguments.get("terraform_dir")
    if terraform_dir:
        argv.extend(["--terraform-dir", str(terraform_dir)])
    for key, flag in (
        ("cpu_platform", "--cpu-platform"),
        ("cpu_preset", "--cpu-preset"),
        ("gpu_platform", "--gpu-platform"),
        ("gpu_preset", "--gpu-preset"),
        ("gpu_driver_mode", "--gpu-driver-mode"),
        ("managed_driver_preset", "--managed-driver-preset"),
        ("gpu_cuda_smoke_image", "--gpu-cuda-smoke-image"),
        ("accelerator", "--accelerator"),
        ("sky_bin", "--sky-bin"),
    ):
        value = str(arguments.get(key) or "")
        if value:
            argv.extend([flag, value])
    for key, enabled, disabled in (
        ("validate", "--validate", "--skip-validate"),
        ("sky_smoke", "--sky-smoke", "--skip-sky-smoke"),
        ("gpu_cuda_smoke", "--gpu-cuda-smoke", "--skip-gpu-cuda-smoke"),
    ):
        argv.append(enabled if bool(arguments.get(key)) else disabled)
    if bool(arguments.get("skip_k8s")):
        argv.append("--skip-k8s")
    if bool(arguments.get("skip_s3")):
        argv.append("--skip-s3")
    if bool(arguments.get("dry_run")):
        argv.append("--dry-run")
    preemptible = arguments.get("preemptible")
    if preemptible is not None:
        argv.append("--preemptible" if bool(preemptible) else "--on-demand")
    unsafe_operator = arguments.get("allow_unsafe_nvswitch_operator")
    if unsafe_operator is not None:
        argv.append(
            "--allow-unsafe-nvswitch-operator"
            if bool(unsafe_operator)
            else "--deny-unsafe-nvswitch-operator"
        )
    return argv


def _transactional_provision(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        from npa.lifecycle_intent import forbid_destructive_provisioning

        forbid_destructive_provisioning("provision_if_absent")
        if current_operation() is not None:
            return function(*args, **kwargs)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        requested_project = bound.arguments.get("project")
        alias, environment, storage, _registry = _resolve_project_runtime(
            requested_project
        )
        cluster_name = str(bound.arguments.get("cluster_name") or "npa-cluster")
        skip_k8s = bool(bound.arguments.get("skip_k8s"))
        context = str(bound.arguments.get("context_name") or "").strip() or cluster_name
        kubeconfig = bound.arguments.get("kubeconfig") or kubeconfig_file(context)
        plan = bound.arguments.get("_resolved_plan") or _build_provision_plan(
            alias=alias,
            environment=environment,
            cluster_name=cluster_name,
            kubeconfig=Path(kubeconfig),
            context=context,
            skip_k8s=skip_k8s,
            dry_run=bool(bound.arguments.get("dry_run")),
            accelerator=str(bound.arguments.get("accelerator") or ""),
            gpu_nodes=int(bound.arguments.get("gpu_nodes", -1)),
            cpu_nodes=int(bound.arguments.get("cpu_nodes", -1)),
            cpu_platform=str(bound.arguments.get("cpu_platform") or ""),
            cpu_preset=str(bound.arguments.get("cpu_preset") or ""),
            gpu_platform=str(bound.arguments.get("gpu_platform") or ""),
            gpu_preset=str(bound.arguments.get("gpu_preset") or ""),
            preemptible=bound.arguments.get("preemptible"),
        )
        kwargs["_resolved_plan"] = plan
        if bool(bound.arguments.get("dry_run")):
            return function(*args, **kwargs)
        plan.assert_mutation_ready()
        resource_type = "storage" if skip_k8s else "cluster"
        requested_name = (
            _bucket_name(storage.checkpoint_bucket) if skip_k8s else cluster_name
        )
        resume_argv = _provision_recovery_argv(
            dict(bound.arguments),
            alias=alias,
            cluster_name=cluster_name,
            context=context,
            kubeconfig=Path(kubeconfig),
        )
        destroy_argv = (
            [
                "npa",
                "cluster",
                "down",
                "--project",
                alias,
                "--context",
                context,
                "--kubeconfig",
                str(kubeconfig),
                "--timeout",
                str(int(bound.arguments.get("timeout") or 120)),
                "--force",
            ]
            if not skip_k8s
            else []
        )
        operation = ProvisioningOperation.prepare(
            command="npa provision-if-absent",
            project_alias=alias,
            project_id=str(getattr(environment, "project_id", "") or ""),
            tenant_id=str(getattr(environment, "tenant_id", "") or ""),
            region=str(getattr(environment, "region", "") or ""),
            backend={
                "bucket": _bucket_name(storage.checkpoint_bucket),
                "endpoint": storage.endpoint_url,
                "region": str(getattr(environment, "region", "") or ""),
            },
            resource_type=resource_type,
            requested_name=requested_name or resource_type,
            ownership_source="provision-if-absent",
            resume_command="",
            destroy_command="",
            resume_argv=resume_argv,
            destroy_argv=destroy_argv,
        )
        operation.record_preflight_plan(plan.to_dict())
        with operation_context(operation):
            sys.stderr.write(
                f"Provisioning operation {operation.operation_id}: preflight complete; beginning mutation\n"
            )
            sys.stderr.flush()
            operation.transition("mutating")
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                operation.record_failure(exc)
                if str(operation.read().get("phase") or "") not in {
                    "rolled-back",
                    "destroyed",
                }:
                    operation.transition("recovery-required")
                    _rollback_owned_cluster(
                        operation,
                        project_alias=alias,
                        context=context,
                        terraform_dir=bound.arguments.get("terraform_dir"),
                        kubeconfig=bound.arguments.get("kubeconfig"),
                        timeout=int(bound.arguments.get("timeout") or 120),
                    )
                sys.stderr.write(emit_recovery_summary(operation) + "\n")
                raise
            if result.status == "partial":
                operation.transition(
                    "recovery-required",
                    error="; ".join(result.warnings) or "provisioning returned partial",
                    details={"error_type": "PartialProvisioningError"},
                )
                rolled_back = _rollback_owned_cluster(
                    operation,
                    project_alias=alias,
                    context=context,
                    terraform_dir=bound.arguments.get("terraform_dir"),
                    kubeconfig=bound.arguments.get("kubeconfig"),
                    timeout=int(bound.arguments.get("timeout") or 120),
                )
                if rolled_back:
                    result.actions.append(
                        "rollback:removed only cluster resources created by this operation"
                    )
            else:
                phase = str(operation.read().get("phase") or "")
                if phase in {"mutating", "resource-created"}:
                    operation.transition("state-durable")
                if phase not in {"rolled-back", "rollback-incomplete"}:
                    operation.commit()
            summary = operation.recovery_summary()
            result.operation_id = operation.operation_id
            result.operation_journal = str(operation.path)
            result.recovery_command = str(summary.get("resume_command") or "")
            return result

    return wrapped


@_transactional_provision
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
    gpu_driver_mode: str = "",
    managed_driver_preset: str = "",
    allow_unsafe_nvswitch_operator: bool | None = None,
    gpu_health_stabilization_seconds: int = DEFAULT_STABILIZATION_SECONDS,
    gpu_cuda_smoke: bool = True,
    gpu_cuda_smoke_image: str = DEFAULT_CUDA_SMOKE_IMAGE,
    preemptible: bool | None = None,
    accelerator: str = "",
    gpu_readiness_timeout: float = 600.0,
    gpu_readiness_poll_interval: float = 10.0,
    sky_bin: str = "",
    agent_exists: bool = False,
    output_format: str = "text",
    _resolved_plan=None,
) -> ProvisionIfAbsentResult:
    """Ensure configured S3 and Kubernetes exist without deleting resources."""
    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("provision_if_absent")
    alias, environment, storage, registry = _resolve_project_runtime(project)
    context = context_name.strip() or cluster_name
    kubeconfig_path = kubeconfig or kubeconfig_file(context)
    actions: list[str] = []
    warnings: list[str] = []
    k8s_ready = False
    gpu_readiness = "not-requested"
    plan = _resolved_plan or _build_provision_plan(
        alias=alias,
        environment=environment,
        cluster_name=cluster_name,
        kubeconfig=Path(kubeconfig_path),
        context=context,
        skip_k8s=skip_k8s,
        dry_run=dry_run,
        accelerator=accelerator,
        gpu_nodes=gpu_nodes,
        cpu_nodes=cpu_nodes,
        cpu_platform=cpu_platform,
        cpu_preset=cpu_preset,
        gpu_platform=gpu_platform,
        gpu_preset=gpu_preset,
        preemptible=preemptible,
        agent_exists=agent_exists,
    )
    topology = plan.topology
    gpu_nodes = topology.gpu_nodes
    cpu_nodes = topology.cpu_nodes
    cpu_platform = topology.cpu_platform
    cpu_preset = topology.cpu_preset
    gpu_platform = topology.gpu_platform
    gpu_preset = topology.gpu_preset
    preemptible = topology.gpu_preemptible
    actions.extend(_preflight_actions(plan))

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
                bucket=bucket_name,
                endpoint_url=storage.endpoint_url,
                access_key_id=storage.aws_access_key_id,
                secret_access_key=storage.aws_secret_access_key,
                region=environment.region,
            )
            if probe.ok:
                storage_ready = True
                actions.append(f"s3:verified writable bucket {bucket_name}")
                if probe.retained_object:
                    warnings.append(probe.summary)
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
        if validate and gpu_nodes > 0:
            from npa.cli.cluster.terraform_lifecycle import (
                _require_bin,
                _validate_cluster,
            )

            kubectl_bin = _require_bin(
                os.environ.get("NPA_KUBECTL_BIN") or "kubectl"
            )
            _validate_cluster(
                kubectl_bin,
                Path(kubeconfig_path),
                {
                    "cpu_nodes_count": cpu_nodes,
                    "gpu_nodes_count": gpu_nodes,
                    "gpu_nodes_platform": gpu_platform,
                    "gpu_nodes_preset": gpu_preset,
                    "gpu_driver_mode": gpu_driver_mode or "auto",
                    "managed_driver_preset": (
                        managed_driver_preset or DEFAULT_MANAGED_DRIVER_PRESET
                    ),
                    "allow_unsafe_nvswitch_operator": bool(
                        allow_unsafe_nvswitch_operator
                    ),
                },
                60,
                gpu_health_stabilization_seconds=(
                    gpu_health_stabilization_seconds
                ),
                gpu_cuda_smoke=gpu_cuda_smoke,
                gpu_cuda_smoke_image=gpu_cuda_smoke_image,
            )
            actions.append("k8s:validated stable GPU health and CUDA vectorAdd")
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
            + (
                [f"preemptible={str(bool(preemptible)).lower()}"]
                if preemptible is not None
                else []
            )
            + [
                f"{name}={value.strip()}"
                for name, value in (
                    ("cpu_platform", cpu_platform),
                    ("cpu_preset", cpu_preset),
                    ("gpu_platform", gpu_platform),
                    ("gpu_preset", gpu_preset),
                    ("gpu_driver_mode", gpu_driver_mode),
                    ("managed_driver_preset", managed_driver_preset),
                )
                if value.strip()
            ]
        )
        actions.append(
            f"k8s:dry-run terraform apply {terraform_dir or 'deploy/cluster'}"
            + (f" ({shape})" if shape else "")
        )
    else:
        from npa.provisioning_preflight import resolved_plan_context

        with _runtime_env(alias, environment, storage, registry), resolved_plan_context(
            plan
        ):
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
                # Provisioning owns one shared cached/fresh GPU-readiness and
                # smoke boundary below. Running smoke inside up_cmd would put
                # the fresh path ahead of label repair and readiness.
                sky_smoke=False,
                sky_gpus="",
                sky_bin=sky_bin,
                capacity_block_group="",
                gpu_nodes=gpu_nodes,
                cpu_nodes=cpu_nodes,
                cpu_platform=cpu_platform,
                cpu_preset=cpu_preset,
                gpu_platform=gpu_platform,
                gpu_preset=gpu_preset,
                gpu_driver_mode=gpu_driver_mode,
                managed_driver_preset=managed_driver_preset,
                allow_unsafe_nvswitch_operator=allow_unsafe_nvswitch_operator,
                gpu_health_stabilization_seconds=(
                    gpu_health_stabilization_seconds
                ),
                gpu_cuda_smoke=gpu_cuda_smoke,
                gpu_cuda_smoke_image=gpu_cuda_smoke_image,
                preemptible=preemptible,
                validation_timeout=60,
                timeout=timeout,
            )
        actions.append(f"k8s:ensured terraform cluster {context}")
        k8s_ready = True

    requested_accelerator = str(accelerator or "").strip()
    needs_gpu_setup = bool(requested_accelerator or sky_smoke)
    if needs_gpu_setup and k8s_ready and not skip_k8s and not dry_run:
        from npa.controller_ownership import ensure_controller_owner
        from npa.cli.cluster.terraform_lifecycle import _run_skypilot_smoke
        from npa.orchestration.skypilot.k8s_gpu_catalog import (
            wait_for_kubernetes_accelerators,
        )

        try:
            owner = ensure_controller_owner(alias, context)
            actions.append(
                f"controller:bound {owner.project_alias}/{owner.context}/{owner.cluster_id}"
            )

            def report_gpu_status(message: str) -> None:
                actions.append(f"gpu:{message}")
                sys.stderr.write(message.rstrip() + "\n")
                sys.stderr.flush()
                operation = current_operation()
                if operation is not None:
                    operation.heartbeat(details={"gpu_readiness": message})

            wait_for_kubernetes_accelerators(
                [requested_accelerator] if requested_accelerator else [],
                context=context,
                kubeconfig=kubeconfig_path,
                sky_bin=sky_bin or None,
                label_known_gpus=True,
                timeout=gpu_readiness_timeout,
                poll_interval=gpu_readiness_poll_interval,
                on_status=report_gpu_status,
            )
            if sky_smoke:
                _run_skypilot_smoke(
                    Path(kubeconfig_path),
                    context,
                    cluster_name,
                    requested_accelerator,
                    sky_bin=sky_bin,
                )
                actions.append("sky-smoke:passed")
        except Exception as exc:  # noqa: BLE001 - return a resumable partial result
            gpu_readiness = (
                "timeout" if str(exc).startswith("Timed out after ") else "failed"
            )
            warnings.append(str(exc))
            operation = current_operation()
            created_cluster = bool(
                operation
                and any(
                    isinstance(item, dict)
                    and item.get("resource_type") == "managed_kubernetes_cluster"
                    and item.get("ownership") == "created_by_this_operation"
                    for item in operation.read().get("resources", [])
                )
            )
            actions.append(
                "gpu:new cluster will be transactionally rolled back"
                if created_cluster
                else "gpu:pre-existing capacity preserved; resume readiness without reprovisioning"
            )
        else:
            gpu_readiness = "ready"
            actions.append(
                "gpu:SkyPilot ready for "
                + (requested_accelerator or "auto-detected smoke accelerator")
            )
    elif needs_gpu_setup and dry_run:
        gpu_readiness = "dry-run"
        actions.append(
            "gpu:dry-run wait for SkyPilot "
            + (requested_accelerator or "auto-detected smoke accelerator")
        )

    status = "ok" if not warnings and (skip_s3 or storage_ready) else "partial"
    if dry_run:
        status = plan.decision
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
        preflight=plan.to_dict(),
    )


def _build_provision_plan(
    *,
    alias: str,
    environment: EnvironmentConfig,
    cluster_name: str,
    kubeconfig: Path,
    context: str,
    skip_k8s: bool,
    dry_run: bool,
    accelerator: str,
    gpu_nodes: int,
    cpu_nodes: int,
    cpu_platform: str,
    cpu_preset: str,
    gpu_platform: str,
    gpu_preset: str,
    preemptible: bool | None,
    agent_exists: bool = False,
):
    from npa.provisioning_preflight import (
        build_whole_path_plan,
        discover_existing_capacity,
        resolve_topology,
    )

    cluster_exists = skip_k8s or _has_cached_kubeconfig(context, kubeconfig)
    requested = resolve_topology(
        cluster_name=cluster_name,
        accelerator=accelerator,
        agent_exists=agent_exists,
        cpu_nodes=0 if skip_k8s else cpu_nodes,
        gpu_nodes=0 if skip_k8s else gpu_nodes,
        cpu_platform=cpu_platform,
        cpu_preset=cpu_preset,
        gpu_platform=gpu_platform,
        gpu_preset=gpu_preset,
        preemptible=preemptible,
    )
    checks = []
    if skip_k8s:
        existing_cpu_nodes = existing_gpu_nodes = 0
    elif cluster_exists:
        existing_cpu_nodes = requested.cpu_nodes
        existing_gpu_nodes = requested.gpu_nodes
    else:
        existing = discover_existing_capacity(
            project_id=str(getattr(environment, "project_id", "") or ""),
            cluster_name=cluster_name,
            cpu_platform=requested.cpu_platform,
            cpu_preset=requested.cpu_preset,
            gpu_platform=requested.gpu_platform,
            gpu_preset=requested.gpu_preset,
        )
        existing_cpu_nodes = min(requested.cpu_nodes, existing.cpu_nodes)
        existing_gpu_nodes = min(requested.gpu_nodes, existing.gpu_nodes)
        checks.append(existing.check)
    if accelerator and not skip_k8s:
        from npa.controller_ownership import controller_preflight

        owner_status, owner_reason = controller_preflight(alias, context)
        from npa.provisioning_preflight import PreflightCheck

        checks.append(
            PreflightCheck(
                name="controller_ownership",
                status=owner_status,
                reason=owner_reason,
            )
        )
    topology = resolve_topology(
        cluster_name=cluster_name,
        accelerator=accelerator,
        agent_exists=agent_exists,
        cpu_nodes=requested.cpu_nodes,
        gpu_nodes=requested.gpu_nodes,
        existing_cpu_nodes=existing_cpu_nodes,
        existing_gpu_nodes=existing_gpu_nodes,
        cpu_platform=requested.cpu_platform,
        cpu_preset=requested.cpu_preset,
        gpu_platform=requested.gpu_platform,
        gpu_preset=requested.gpu_preset,
        preemptible=requested.gpu_preemptible,
    )
    return build_whole_path_plan(
        project_alias=alias,
        project_id=str(getattr(environment, "project_id", "") or ""),
        tenant_id=str(getattr(environment, "tenant_id", "") or ""),
        region=str(getattr(environment, "region", "") or ""),
        topology=topology,
        checks=checks,
        mutation=not dry_run,
    )


def resolve_provision_plan(
    *,
    project: str | None = None,
    cluster_name: str = "npa-cluster",
    kubeconfig: Path | None = None,
    context_name: str = "",
    skip_k8s: bool = False,
    accelerator: str = "",
    gpu_nodes: int = -1,
    cpu_nodes: int = -1,
    cpu_platform: str = "",
    cpu_preset: str = "",
    gpu_platform: str = "",
    gpu_preset: str = "",
    preemptible: bool | None = None,
    mutation: bool = False,
):
    """Resolve the immutable plan without creating journals or resources."""

    alias, environment, _storage, _registry = _resolve_project_runtime(project)
    context = context_name.strip() or cluster_name
    kubeconfig_path = kubeconfig or kubeconfig_file(context)
    return _build_provision_plan(
        alias=alias,
        environment=environment,
        cluster_name=cluster_name,
        kubeconfig=Path(kubeconfig_path),
        context=context,
        skip_k8s=skip_k8s,
        dry_run=not mutation,
        accelerator=accelerator,
        gpu_nodes=gpu_nodes,
        cpu_nodes=cpu_nodes,
        cpu_platform=cpu_platform,
        cpu_preset=cpu_preset,
        gpu_platform=gpu_platform,
        gpu_preset=gpu_preset,
        preemptible=preemptible,
    )


def _preflight_actions(plan) -> list[str]:
    topology = plan.topology
    actions = [
        "preflight:"
        f"{plan.decision} cpu={topology.cpu_nodes}x{topology.cpu_platform}/{topology.cpu_preset} "
        f"gpu={topology.gpu_nodes}x{topology.gpu_platform}/{topology.gpu_preset} "
        f"preemptible={str(topology.gpu_preemptible).lower()} disks={topology.required_disks} "
        f"network_ssd_bytes={topology.required_network_ssd_bytes}"
    ]
    for quota in plan.quotas:
        rendered = (
            "quota:"
            f"{quota.name}:{quota.status} required={quota.required} used={quota.used} "
            f"limit={quota.limit} available={quota.available} shortfall={quota.shortfall}"
        )
        structured = quota.to_dict()
        if structured.get("unit") == "bytes":
            rendered += (
                f" bytes required_gib={structured.get('required_gib')} "
                f"available_gib={structured.get('available_gib')} "
                f"shortfall_gib={structured.get('shortfall_gib')}"
            )
        actions.append(rendered)
    return actions


def _rollback_owned_cluster(
    operation: ProvisioningOperation,
    *,
    project_alias: str,
    context: str,
    terraform_dir: Path | None,
    kubeconfig: Path | None,
    timeout: int,
) -> bool:
    """Roll back only a cluster explicitly receipted as created by this operation."""

    payload = operation.read()
    if str(payload.get("phase") or "") == "rolled-back":
        return True
    created = [
        item
        for item in payload.get("resources", [])
        if isinstance(item, dict)
        and item.get("ownership") == "created_by_this_operation"
    ]
    owned = [
        item
        for item in created
        if item.get("resource_type") == "managed_kubernetes_cluster"
    ]
    if not created:
        operation.record_rollback(
            attempted=False,
            completed=False,
            removed=[],
            preserved=list(payload.get("resources") or []),
        )
        return False
    if not owned or not str(owned[0].get("provider_id") or ""):
        error = (
            "rollback requires the exact created managed Kubernetes cluster ID; "
            "owned resources were preserved for exact retry"
        )
        operation.record_rollback(
            attempted=True,
            completed=False,
            removed=[],
            preserved=list(payload.get("resources") or []),
            outcomes=[{**item, "outcome": "preserved_missing_cluster_identity"} for item in created],
            error=error,
        )
        operation.transition("rollback-incomplete", error=error)
        return False
    terraform_owned = [
        item
        for item in payload.get("resources", [])
        if isinstance(item, dict)
        and item.get("ownership") == "created_by_this_operation"
        and str(item.get("ownership_source") or "").startswith("terraform")
    ]
    preserved = [
        item for item in payload.get("resources", []) if item not in terraform_owned
    ]
    operation.record_rollback(
        attempted=True,
        completed=False,
        removed=[],
        preserved=list(payload.get("resources") or []),
        outcomes=[{**item, "outcome": "rollback_started"} for item in terraform_owned],
    )
    operation.transition("rolling-back")
    try:
        from npa.cli.cluster.terraform_lifecycle import down_cmd

        down_cmd(
            terraform_dir=terraform_dir,
            project=project_alias,
            receipt="",
            project_id="",
            tenant_id="",
            region="",
            cluster_id=str(owned[0].get("provider_id") or ""),
            operation_id=operation.operation_id,
            context_name=context,
            keep_local_state=False,
            force=True,
            timeout=timeout,
            kubeconfig=kubeconfig,
            output_json=False,
        )
    except BaseException as exc:
        operation.record_rollback(
            attempted=True,
            completed=False,
            removed=[],
            preserved=list(payload.get("resources") or []),
            outcomes=[
                {**item, "outcome": "rollback_failed", "error": str(exc)}
                for item in terraform_owned
            ],
            error=str(exc),
        )
        operation.transition("rollback-incomplete", error=str(exc))
        return False
    try:
        from npa.controller_ownership import clear_controller_owner

        clear_controller_owner(
            project_id=str(payload.get("project_id") or ""),
            cluster_id=str(owned[0].get("provider_id") or ""),
            context=context,
        )
    except (OSError, RuntimeError, ValueError):
        # cluster down succeeded; a non-matching/unavailable owner is preserved
        # for explicit reconciliation and is never cleared by alias alone.
        pass
    operation.record_rollback(
        attempted=True,
        completed=True,
        removed=terraform_owned,
        preserved=preserved,
        outcomes=[
            *({**item, "outcome": "removed"} for item in terraform_owned),
            *({**item, "outcome": "preserved_not_owned_by_terraform"} for item in preserved),
        ],
    )
    operation.transition("rolled-back")
    return True


def _resolve_project_runtime(
    project: str | None,
) -> tuple[str, EnvironmentConfig, StorageConfig, str]:
    yml = config_module._load_yaml()
    alias = config_module._resolved_project_name(yml, project)
    environment = config_module.resolve_environment(project) or EnvironmentConfig(
        "", "", ""
    )
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
    return bool(
        state and state.kubeconfig_path and Path(state.kubeconfig_path).exists()
    )


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
        # Consumers of NPA_S3_BUCKET pass it as the provider Bucket argument;
        # keep URI/prefix forms in checkpoint_bucket only.
        "NPA_S3_BUCKET": _bucket_name(storage.checkpoint_bucket),
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
        for key, previous_value in previous.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value
