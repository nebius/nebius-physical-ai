"""Terraform-backed ``npa cluster up`` and ``npa cluster down`` commands."""

from __future__ import annotations

import json
import functools
import inspect
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import typer

from npa.cli._typer_defaults import resolve_typer_defaults
from npa.cluster.state import (
    ClusterState,
    kubeconfig_file,
    load_cluster_state,
    save_cluster_state,
    utc_now_iso,
)
from npa.cluster.gpu_driver import (
    DEFAULT_MANAGED_DRIVER_PRESET,
    GpuDriverSelection,
    GpuDriverStrategyError,
    resolve_gpu_driver_strategy,
)
from npa.cluster.gpu_health import (
    DEFAULT_CUDA_SMOKE_IMAGE,
    DEFAULT_STABILIZATION_SECONDS,
    GpuHealthConfig,
    probe_gpu_health,
    validate_gpu_health,
)
from npa.cluster_backends import get_backend
from npa.cluster_backends.mk8s import (
    MK8sApplyRequest,
    MK8sDestroyRequest,
    MK8sExecutionScope,
    MK8sProjectIdentity,
    MK8sStatusRequest,
)
from npa.cluster_backends.mig import (
    GPU_DEVICE_PLUGIN_VERSION,
    GPU_DRIVER_VERSION,
    GPU_GFD_VERSION,
    GPU_MIG_MANAGER_VERSION,
    GPU_OPERATOR_VERSION,
    MIG_KUBERNETES_VERSION,
    MigSpec,
    wait_for_mig_ready,
)
from npa.fleet.spec import ClusterSpec, FleetSpec, NodePoolSpec, ProjectSpec
from npa.cluster_backends.mk8s_render import validate_recipe_mig_compatibility
from npa.provisioning_journal import (
    ProvisioningOperation,
    current_operation,
    emit_recovery_summary,
    list_operations,
    operation_context,
)
from npa.lifecycle_intent import OperationIntent, intent_boundary, json_stdout_contract

_DEFAULT_TERRAFORM_SUBDIR = Path("deploy") / "cluster"
_DEFAULT_SKYPILOT_BIN = Path.home() / ".npa" / "skypilot-venv" / "bin" / "sky"
_DEFAULT_FILESTORE_SIZE_GIB = 1024
_GIB = 1024**3
# deploy/cluster vendors nebius-solutions-library, whose k8s-rbac-bindings module
# declares `required_version >= 1.12.0` and whose o11y module uses `ephemeral`
# blocks (Terraform 1.10+). Terraform loads every referenced module during
# `init`, even the ones this config disables, so an older binary fails with a
# wall of "Unsupported Terraform Core version" / "Unsupported block type" errors
# from vendored files the operator never wrote. Check up front instead.
_MIN_TERRAFORM_VERSION = (1, 12, 0)


def _redacted_exception_message(prefix: str, exc: BaseException) -> str:
    from npa.clients.nebius import redact_nebius_output

    return redact_nebius_output(f"{prefix}: {type(exc).__name__}: {exc}")


def _transactional_cluster_up(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if current_operation() is not None:
            return function(*args, **kwargs)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        tf_dir = _resolve_terraform_dir(bound.arguments.get("terraform_dir"))
        tfvars = _read_tfvars(tf_dir)
        from npa.provisioning_preflight import current_resolved_plan

        inherited_plan = current_resolved_plan()
        context_arg = str(bound.arguments.get("context_name") or "").strip()
        context = context_arg or str(
            (inherited_plan.topology.cluster_name if inherited_plan else "")
            or tfvars.get("cluster_name")
            or "npa-cluster"
        )
        project = str(
            (inherited_plan.project_alias if inherited_plan else "")
            or bound.arguments.get("project")
            or ""
        ).strip()
        project_id = str(
            (inherited_plan.project_id if inherited_plan else "")
            or tfvars.get("parent_id")
            or ""
        )
        tenant_id = str(
            (inherited_plan.tenant_id if inherited_plan else "")
            or tfvars.get("tenant_id")
            or ""
        )
        region = str(
            (inherited_plan.region if inherited_plan else "")
            or tfvars.get("region")
            or ""
        )
        if not project_id:
            from npa.clients.config import resolve_environment

            saved = resolve_environment(project or None)
            if saved is not None:
                project_id = saved.project_id
                tenant_id = tenant_id or saved.tenant_id
                region = region or saved.region
        operation = ProvisioningOperation.prepare(
            command="npa cluster up",
            project_alias=project,
            project_id=project_id,
            tenant_id=tenant_id,
            region=region,
            backend={"kind": "local-state", "terraform_dir": str(tf_dir)},
            resource_type="cluster",
            requested_name=context,
            ownership_source="cluster-terraform",
            resume_command=(
                f"npa cluster up --project {project} --context {context} "
                f"--terraform-dir {tf_dir}"
            ),
            destroy_command=(
                f"npa cluster down --project {project} --context {context} "
                f"--terraform-dir {tf_dir} --force"
            ),
        )
        state_existed_before = any(
            candidate.is_file()
            for candidate in (tf_dir / "terraform.tfstate", tf_dir / "errored.tfstate")
        )
        with operation_context(operation):
            operation.transition("mutating")
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                typer.echo(
                    _redacted_exception_message("cluster up failed", exc), err=True
                )
                operation.record_failure(exc)
                for candidate in (
                    tf_dir / "errored.tfstate",
                    tf_dir / "terraform.tfstate",
                ):
                    if candidate.is_file():
                        operation.preserve_state_file(candidate, name=candidate.stem)
                rolled_back = _rollback_fresh_cluster_apply(
                    operation,
                    state_existed_before=state_existed_before,
                    terraform_dir=tf_dir,
                    project=project,
                    context=context,
                    timeout_minutes=int(bound.arguments.get("timeout") or 120),
                )
                if not rolled_back and operation.read().get("phase") not in {
                    "rollback-incomplete",
                    "rolled-back",
                }:
                    operation.transition("recovery-required")
                typer.echo(emit_recovery_summary(operation), err=True)
                raise
            state_path = tf_dir / "terraform.tfstate"
            if state_path.is_file():
                operation.preserve_state_file(state_path, name="verified-local")
            if operation.read().get("phase") == "resource-created":
                operation.transition("state-durable")
            operation.commit()
            return result

    return wrapped


def _rollback_fresh_cluster_apply(
    operation: ProvisioningOperation,
    *,
    state_existed_before: bool,
    terraform_dir: Path,
    project: str,
    context: str,
    timeout_minutes: int,
) -> bool:
    """Destroy only state first created during this failed direct apply."""

    if operation.read().get("phase") == "rolled-back":
        return True
    if state_existed_before:
        return False
    state_candidates = (
        terraform_dir / "terraform.tfstate",
        terraform_dir / "errored.tfstate",
    )
    if not any(candidate.is_file() for candidate in state_candidates):
        return False
    operation.record_resource(
        resource_type="terraform_cluster_stack",
        requested_name=context,
        ownership="created_by_this_operation",
        ownership_source="fresh-terraform-state-after-green-preflight",
        project_id=str(operation.read().get("project_id") or ""),
    )
    operation.transition("rolling-back")
    try:
        down_cmd(
            terraform_dir=terraform_dir,
            project=project,
            receipt="",
            project_id=str(operation.read().get("project_id") or ""),
            tenant_id=str(operation.read().get("tenant_id") or ""),
            region=str(operation.read().get("region") or ""),
            cluster_id="",
            operation_id="",
            context_name=context,
            keep_local_state=False,
            force=True,
            timeout=max(60, timeout_minutes * 60),
            kubeconfig=None,
            output_json=False,
        )
    except BaseException as rollback_exc:
        operation.transition("rollback-incomplete", error=str(rollback_exc))
        return False
    operation.transition("rolled-back")
    return True


@resolve_typer_defaults
@_transactional_cluster_up
def up_cmd(
    terraform_dir: Path | None = typer.Option(
        None,
        "--terraform-dir",
        help="Terraform cluster directory. Defaults to ./deploy/cluster or the repo root deploy/cluster.",
    ),
    kubeconfig: Path | None = typer.Option(
        None,
        "--kubeconfig",
        help="Kubeconfig output path. Defaults to ~/.npa/clusters/<cluster-name>/kubeconfig.",
    ),
    context_name: str = typer.Option(
        "",
        "--context",
        # `provision-if-absent` and `cluster node-group *` name the same cluster
        # with --cluster-name; accept it here so copying a command between help
        # pages does not fail on an unknown option.
        "--cluster-name",
        help="Kubeconfig context name. Defaults to the Terraform cluster name.",
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--skip-validate",
        help="Validate stable nodes, GPU capacity/fabric/driver components, CUDA vectorAdd, and the default StorageClass.",
    ),
    sky_smoke: bool = typer.Option(
        True,
        "--sky-smoke/--skip-sky-smoke",
        help="Run a SkyPilot Kubernetes GPU smoke task and clean it up through NPA.",
    ),
    sky_gpus: str = typer.Option(
        "",
        "--sky-gpus",
        help="SkyPilot GPU demand for the smoke task. Defaults to auto-detecting the first Kubernetes GPU.",
    ),
    sky_bin: str = typer.Option(
        "",
        "--sky-bin",
        help="Pinned NPA SkyPilot executable override for GPU readiness and smoke.",
    ),
    capacity_block_group: str = typer.Option(
        "",
        "--capacity-block-group",
        help=(
            "Optional private capacity block group ID for strict GPU node-group "
            "reservation selection. Equivalent to TF_VAR_capacity_block_group."
        ),
    ),
    gpu_nodes: int = typer.Option(
        -1,
        "--gpu-nodes",
        help="Number of GPU nodes (overrides tfvars/TF_VAR_gpu_nodes_count). -1 keeps the configured value.",
    ),
    cpu_nodes: int = typer.Option(
        -1,
        "--cpu-nodes",
        help="Number of CPU nodes (overrides tfvars/TF_VAR_cpu_nodes_count). -1 keeps the configured value.",
    ),
    cpu_platform: str = typer.Option(
        "",
        "--cpu-platform",
        help="CPU node platform (overrides tfvars/TF_VAR_cpu_nodes_platform).",
    ),
    cpu_preset: str = typer.Option(
        "",
        "--cpu-preset",
        help="CPU node preset (overrides tfvars/TF_VAR_cpu_nodes_preset).",
    ),
    gpu_platform: str = typer.Option(
        "",
        "--gpu-platform",
        help="GPU node platform (overrides tfvars/TF_VAR_gpu_nodes_platform).",
    ),
    gpu_preset: str = typer.Option(
        "",
        "--gpu-preset",
        help="GPU node preset (overrides tfvars/TF_VAR_gpu_nodes_preset).",
    ),
    gpu_driver_mode: str = typer.Option(
        "",
        "--gpu-driver-mode",
        help="GPU driver strategy: auto, managed-image, or operator. Empty keeps tfvars/default auto.",
    ),
    managed_driver_preset: str = typer.Option(
        "",
        "--managed-driver-preset",
        help="Nebius managed driver preset for auto/managed-image mode (default: cuda13.0).",
    ),
    allow_unsafe_nvswitch_operator: bool | None = typer.Option(
        None,
        "--allow-unsafe-nvswitch-operator/--deny-unsafe-nvswitch-operator",
        help="Explicitly acknowledge the unsafe operator/Fabric Manager ordering path on NVSwitch systems (diagnostics only).",
    ),
    gpu_health_stabilization_seconds: int = typer.Option(
        DEFAULT_STABILIZATION_SECONDS,
        "--gpu-health-stabilization-seconds",
        help="Seconds GPU nodes, boot IDs, fabric, capacity, and components must remain healthy before success.",
    ),
    gpu_cuda_smoke: bool = typer.Option(
        True,
        "--gpu-cuda-smoke/--skip-gpu-cuda-smoke",
        help="Run NVIDIA's CUDA vectorAdd smoke on every requested GPU node.",
    ),
    gpu_cuda_smoke_image: str = typer.Option(
        DEFAULT_CUDA_SMOKE_IMAGE,
        "--gpu-cuda-smoke-image",
        help="Container image for the post-deploy CUDA vectorAdd smoke.",
    ),
    mig_enabled: bool = typer.Option(
        False,
        "--mig/--no-mig",
        help="Enable the pinned RTX PRO 6000 hardware-MIG policy and exact readiness gate.",
    ),
    mig_strategy: str = typer.Option(
        "mixed", "--mig-strategy", help="Hardware-MIG strategy (validated: mixed)."
    ),
    mig_config: str = typer.Option(
        "all-balanced",
        "--mig-config",
        help="RTX PRO 6000 MIG geometry (validated: all-balanced).",
    ),
    preemptible: bool | None = typer.Option(
        None,
        "--preemptible/--on-demand",
        help=(
            "Run the GPU node group as preemptible. Preemptible capacity is often the "
            "only way to get several GPUs, but a reclaim stops the nodes mid-run -- keep "
            "CPU stages on the CPU pool. Unset keeps tfvars/TF_VAR_gpu_nodes_preemptible."
        ),
    ),
    project: str = typer.Option(
        "",
        "--project",
        help="NPA project alias whose saved project/tenant/region to use when tfvars omit them.",
    ),
    validation_timeout: int = typer.Option(
        60,
        "--validation-timeout",
        help="Post-apply Kubernetes validation timeout in minutes.",
    ),
    timeout: int = typer.Option(
        120, "--timeout", help="Terraform apply timeout in minutes."
    ),
) -> None:
    """Create or update the Terraform-managed NPA Kubernetes cluster."""

    from npa.cli.cluster.terraform_runtime import (
        isolated_terraform_data_dir,
        record_terraform_inventory,
    )

    tf_dir = _resolve_terraform_dir(terraform_dir)
    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    kubectl_bin = _require_bin(os.environ.get("NPA_KUBECTL_BIN") or "kubectl")
    _preflight_provider_lock(tf_dir)
    tfvars = _read_tfvars(tf_dir)
    from npa.provisioning_preflight import current_resolved_plan

    inherited_plan = current_resolved_plan()
    if inherited_plan is not None:
        topology = _apply_inherited_plan_tfvars(tfvars, inherited_plan)
        project = inherited_plan.project_alias
        gpu_nodes = topology.gpu_nodes
        cpu_nodes = topology.cpu_nodes
        cpu_platform = topology.cpu_platform
        cpu_preset = topology.cpu_preset
        gpu_platform = topology.gpu_platform
        gpu_preset = topology.gpu_preset
        preemptible = topology.gpu_preemptible
    explicit_context = _apply_context_cluster_name(
        tfvars,
        context_name,
        inherited_name=(
            str(inherited_plan.topology.cluster_name or "")
            if inherited_plan is not None
            else ""
        ),
    )
    # First-class node-count flags: a runbook can pick "agent XOR 2-GPU cluster"
    # under a tight compute.instance.count without editing tfvars or exporting
    # TF_VAR_*. -1 means "leave the configured value alone".
    _apply_node_count_override(tfvars, "gpu_nodes_count", gpu_nodes)
    _apply_node_count_override(tfvars, "cpu_nodes_count", cpu_nodes)
    for key, value in (
        ("cpu_nodes_platform", cpu_platform),
        ("cpu_nodes_preset", cpu_preset),
        ("gpu_nodes_platform", gpu_platform),
        ("gpu_nodes_preset", gpu_preset),
        ("gpu_driver_mode", gpu_driver_mode),
        ("managed_driver_preset", managed_driver_preset),
    ):
        _apply_string_override(tfvars, key, value)
    if preemptible is not None:
        tfvars["gpu_nodes_preemptible"] = bool(preemptible)
    if allow_unsafe_nvswitch_operator is not None:
        tfvars["allow_unsafe_nvswitch_operator"] = bool(allow_unsafe_nvswitch_operator)
    context = explicit_context or str(tfvars.get("cluster_name") or "npa-cluster")

    with isolated_terraform_data_dir(tf_dir, context) as terraform_data:
        env = _terraform_env(nebius_bin)
        env["TF_DATA_DIR"] = str(terraform_data)
        _apply_capacity_block_group_env(env, capacity_block_group)

        typer.echo(f"Terraform directory: {tf_dir}")
        typer.echo(f"Terraform data: isolated NPA scratch {terraform_data}")
        _preflight_terraform_version(terraform_bin)
        _apply_project_tf_vars(env, project, tfvars)
        _guard_tfvars_iam_token(tf_dir, tfvars)
        _apply_capacity_block_group_tfvars(tfvars, capacity_block_group)
        # MIG uses the recipe's deliberately small 128 GiB worker disk. Apply
        # that resolved desired state before whole-path disk quota accounting.
        if mig_enabled:
            tfvars["gpu_disk_size"] = 128
            tfvars["mig_enabled"] = True
        driver = _resolve_gpu_driver_selection(tfvars, env)
        typer.echo(
            "GPU driver strategy: "
            + (
                f"{driver.effective_mode} ({driver.managed_driver_preset})"
                if driver.uses_managed_image
                else driver.effective_mode
            )
        )
        if driver.unsafe_operator_acknowledged:
            typer.echo(
                "WARNING: operator mode on this NVSwitch topology explicitly "
                "re-enables the unsafe driver/Fabric Manager host-device ordering path.",
                err=True,
            )
        resolved_gpu_nodes = int(_tfvar_value(tfvars, env, "gpu_nodes_count", 1) or 0)
        resolved_cpu_nodes = int(_tfvar_value(tfvars, env, "cpu_nodes_count", 1) or 0)
        resolved_capacity = capacity_block_group.strip() or str(
            _tfvar_value(tfvars, env, "capacity_block_group", "") or ""
        )
        backend_desired = ClusterSpec(
            name=context,
            k8s_version=(
                MIG_KUBERNETES_VERSION
                if mig_enabled
                else str(_tfvar_value(tfvars, env, "k8s_version", "") or "")
            ),
            cpu_nodes=(
                NodePoolSpec(
                    count=resolved_cpu_nodes,
                    platform=str(
                        _tfvar_value(tfvars, env, "cpu_nodes_platform", "cpu-d3")
                    ),
                    preset=str(
                        _tfvar_value(tfvars, env, "cpu_nodes_preset", "8vcpu-32gb")
                    ),
                )
                if resolved_cpu_nodes
                else None
            ),
            gpu_nodes=(
                NodePoolSpec(
                    count=resolved_gpu_nodes,
                    platform=str(
                        _tfvar_value(tfvars, env, "gpu_nodes_platform", "gpu-rtx6000")
                        or "gpu-rtx6000"
                    ),
                    preset=str(
                        _tfvar_value(
                            tfvars, env, "gpu_nodes_preset", "1gpu-24vcpu-218gb"
                        )
                        or "1gpu-24vcpu-218gb"
                    ),
                    disk_size_gib=(
                        128
                        if mig_enabled
                        else int(_tfvar_value(tfvars, env, "gpu_disk_size", 0) or 0)
                    ),
                    capacity_block_group=resolved_capacity,
                    preemptible=_tfvar_bool(
                        tfvars, env, "gpu_nodes_preemptible", False
                    ),
                )
                if resolved_gpu_nodes
                else None
            ),
            enable_gpu_cluster=_tfvar_bool(tfvars, env, "enable_gpu_cluster", False),
            infiniband_fabric=str(
                _tfvar_value(tfvars, env, "infiniband_fabric", "") or ""
            ),
            enable_filestore=(
                _tfvar_bool(tfvars, env, "enable_filestore", False)
                or bool(_tfvar_value(tfvars, env, "existing_filestore", ""))
            ),
            existing_filestore=str(
                _tfvar_value(tfvars, env, "existing_filestore", "") or ""
            ),
            subnet_id=str(_tfvar_value(tfvars, env, "subnet_id", "") or ""),
            filestore_disk_size_gibibytes=int(
                _tfvar_value(tfvars, env, "filestore_disk_size_gibibytes", 1024) or 1024
            ),
            gpu_driver_mode=(
                "operator"
                if mig_enabled
                else str(_tfvar_value(tfvars, env, "gpu_driver_mode", "auto") or "auto")
            ),
            managed_driver_preset=str(
                _tfvar_value(
                    tfvars,
                    env,
                    "managed_driver_preset",
                    DEFAULT_MANAGED_DRIVER_PRESET,
                )
                or DEFAULT_MANAGED_DRIVER_PRESET
            ),
            allow_unsafe_nvswitch_operator=_tfvar_bool(
                tfvars, env, "allow_unsafe_nvswitch_operator", False
            ),
            gpu_health_stabilization_seconds=gpu_health_stabilization_seconds,
            gpu_health_timeout_minutes=validation_timeout,
            gpu_cuda_smoke=gpu_cuda_smoke,
            gpu_cuda_smoke_image=gpu_cuda_smoke_image,
            mig=(
                MigSpec(enabled=True, strategy=mig_strategy, config=mig_config)
                if mig_enabled
                else None
            ),
            allow_control_plane_only=True,
        )
        backend_desired.validate()
        recipe_dir = tf_dir / "vendor" / "nebius-solutions-library" / "k8s-training"
        if mig_enabled:
            validate_recipe_mig_compatibility(backend_desired, recipe_dir)
        from npa.cluster.state import cluster_dir

        project_id_value = str(_tfvar_value(tfvars, env, "parent_id", "") or "")
        tenant_id_value = str(_tfvar_value(tfvars, env, "tenant_id", "") or "")
        region_value = str(_tfvar_value(tfvars, env, "region", "") or "")
        legacy_state_exists = any(
            candidate.is_file()
            for candidate in (tf_dir / "terraform.tfstate", tf_dir / "errored.tfstate")
        )
        shared_recipe_available = (recipe_dir / "variables.tf").is_file() and (
            recipe_dir.parent / "modules"
        ).is_dir()
        shared_ssh_public_key = (
            _resolve_shared_ssh_public_key(tfvars, env)
            if not legacy_state_exists and shared_recipe_available
            else str(_tfvar_value(tfvars, env, "ssh_public_key", "") or "")
        )
        backend_root = cluster_dir(context) / "backend-state"
        project_spec = ProjectSpec(
            # Existing-project identity is ID-backed. Giving it a synthetic
            # name would make provider reconciliation compare that fake name
            # with the real cloud project and defeat zero-increment reuse.
            name="",
            project_id=project_id_value,
            region=region_value,
            clusters=[backend_desired],
        )
        one_target = FleetSpec(
            name="standalone",
            tenant_id=tenant_id_value,
            region=region_value,
            projects=[project_spec],
        )
        # Standalone, agent, and fleet consume one canonical desired-state,
        # capability, and materialization boundary for every mk8s topology.
        adapter_request = MK8sApplyRequest(
            recipe_dir=(
                recipe_dir if (recipe_dir / "variables.tf").is_file() else None
            ),
            nebius_bin=nebius_bin,
            tenant_id=tenant_id_value,
            region=region_value,
            provider_env=env,
            # Every fresh shared-backend apply uses the fleet capacity/quota
            # preflight, not only MIG targets. In particular, a non-MIG GPU
            # pool with a capacity block must prove the exact STRICT
            # reservation before subnet or Terraform mutation. Existing
            # legacy Terraform state keeps its established reconciliation
            # checks below instead of being reclassified as fleet state.
            provider_preflight=(
                mig_enabled or (not legacy_state_exists and shared_recipe_available)
            ),
            scope=MK8sExecutionScope(
                fleet_name=one_target.name,
                tenant_id=tenant_id_value,
                region=region_value,
                project_prefix=one_target.project_prefix,
            ),
            project=MK8sProjectIdentity(
                project_key=project_spec.key(),
                project_id=project_id_value,
                project_name=project_spec.name,
                expected_provider_name=project_spec.display_name(
                    one_target.project_prefix
                ),
            ),
            project_id=project_id_value,
            subnet_id=backend_desired.subnet_id,
            ssh_public_key=shared_ssh_public_key,
            fleet_root=backend_root,
        )
        get_backend("mk8s").preflight(backend_desired, adapter_request)
        get_backend("mk8s").materialize(backend_desired, adapter_request)
        mig_desired = backend_desired if mig_enabled else None
        _preflight_whole_path_capacity(
            tfvars, env, context=context, project_alias=project
        )
        if not legacy_state_exists and shared_recipe_available:
            if not (project_id_value and tenant_id_value and region_value):
                raise typer.BadParameter(
                    "shared mk8s apply requires resolved project, tenant, and region"
                )
            # Subnet creation is a provider mutation. Run it only after both
            # shared and whole-path capacity checks have succeeded.
            if not backend_desired.subnet_id:
                from npa.fleet.lifecycle import ensure_subnet

                resolved_subnet_id, _created_network_id = ensure_subnet(
                    nebius_bin,
                    project_id_value,
                    name_stem=context,
                    env=env,
                    network_state_path=(
                        backend_root / project_spec.key() / ".npa-fleet-network.json"
                    ),
                    on_status=lambda message: typer.echo(message, err=True),
                )
                backend_desired = replace(backend_desired, subnet_id=resolved_subnet_id)
                project_spec = replace(project_spec, clusters=[backend_desired])
                one_target = replace(one_target, projects=[project_spec])
            result = get_backend("mk8s").apply(
                backend_desired,
                MK8sApplyRequest(
                    scope=MK8sExecutionScope(
                        fleet_name=one_target.name,
                        tenant_id=tenant_id_value,
                        region=region_value,
                        project_prefix=one_target.project_prefix,
                    ),
                    project=MK8sProjectIdentity(
                        project_key=project_spec.key(),
                        project_id=project_id_value,
                        project_name=project_spec.name,
                        expected_provider_name=project_spec.display_name(
                            one_target.project_prefix
                        ),
                    ),
                    project_id=project_id_value,
                    subnet_id=backend_desired.subnet_id,
                    region=region_value,
                    tenant_id=tenant_id_value,
                    ssh_public_key=shared_ssh_public_key,
                    fleet_root=backend_root,
                    recipe_root=recipe_dir.parent,
                    terraform_bin=terraform_bin,
                    nebius_bin=nebius_bin,
                    timeout_minutes=timeout,
                    on_status=lambda message: typer.echo(message, err=True),
                    standalone_context=context,
                    standalone_kubeconfig=kubeconfig,
                    post_deploy_validation=("standalone-full" if validate else "skip"),
                    basic_validation_timeout_minutes=validation_timeout,
                    kubectl_bin=kubectl_bin,
                ),
            )
            if result.get("status") != "deployed":
                raise RuntimeError(
                    str(result.get("error") or "shared mk8s backend apply failed")
                )
            kubeconfig_path = Path(str(result["kubeconfig"]))
            typer.echo(f"Cluster ID: {result['cluster_id']}")
            typer.echo(f"Cluster name: {backend_desired.name}")
            typer.echo(f"Kubeconfig: {kubeconfig_path}")
            if validate:
                basics = result.get("cluster_basics") or {}
                typer.echo(
                    "Validation: "
                    f"{basics.get('ready_nodes', 0)} Ready nodes, "
                    f"default StorageClass "
                    f"{basics.get('default_storage_class', 'unknown')}"
                )
            else:
                typer.echo(
                    "Post-deploy validation skipped by --skip-validate; "
                    "Terraform apply and identity persistence still completed."
                )
            if sky_smoke and mig_desired is None:
                from npa.orchestration.skypilot.k8s_gpu_catalog import (
                    wait_for_kubernetes_accelerators,
                )

                wait_for_kubernetes_accelerators(
                    [sky_gpus] if sky_gpus.strip() else [],
                    context=context,
                    kubeconfig=kubeconfig_path,
                    sky_bin=sky_bin or None,
                    label_known_gpus=True,
                    on_status=lambda message: typer.echo(message, err=True),
                )
                _run_skypilot_smoke(
                    kubeconfig_path,
                    context,
                    backend_desired.name,
                    sky_gpus,
                    sky_bin=sky_bin,
                )
            return
        _terraform_init(terraform_bin, tf_dir, env)
        _guard_unmanaged_duplicate(nebius_bin, terraform_bin, tf_dir, tfvars, env)
        _preflight_filestore_quota(nebius_bin, tfvars, env)
        _preflight_gpu_capacity(nebius_bin, tfvars, env)

        apply_args = [
            terraform_bin,
            "apply",
            "-auto-approve",
            # -var beats terraform.tfvars, TF_VAR_* does not. Pass the flag value
            # explicitly so `--capacity-block-group` is not silently dropped by a
            # `capacity_block_group = ""` line in a checked-in tfvars file.
            *_capacity_block_group_var_args(capacity_block_group),
            *_string_var_args(
                "cluster_name", str(tfvars.get("cluster_name") or "npa-cluster")
            ),
            *_node_count_var_args(tfvars, "gpu_nodes_count", gpu_nodes),
            *_node_count_var_args(tfvars, "cpu_nodes_count", cpu_nodes),
            *_string_var_args("cpu_nodes_platform", cpu_platform),
            *_string_var_args("cpu_nodes_preset", cpu_preset),
            *_string_var_args("gpu_nodes_platform", gpu_platform),
            *_string_var_args("gpu_nodes_preset", gpu_preset),
            *_string_var_args("gpu_driver_mode", gpu_driver_mode),
            *_string_var_args("managed_driver_preset", managed_driver_preset),
            *(
                [
                    "-var",
                    "mig_enabled=true",
                    "-var",
                    f"mig_strategy={mig_strategy}",
                    "-var",
                    f"mig_parted_config={mig_config}",
                    "-var",
                    "gpu_driver_mode=operator",
                    "-var",
                    "gpu_disk_size=128",
                    "-var",
                    f"k8s_version={MIG_KUBERNETES_VERSION}",
                    "-var",
                    f"gpu_operator_version={GPU_OPERATOR_VERSION}",
                    "-var",
                    f"gpu_driver_version={GPU_DRIVER_VERSION}",
                    "-var",
                    f"gpu_device_plugin_version={GPU_DEVICE_PLUGIN_VERSION}",
                    "-var",
                    f"gpu_gfd_version={GPU_GFD_VERSION}",
                    "-var",
                    f"gpu_mig_manager_version={GPU_MIG_MANAGER_VERSION}",
                    "-var",
                    "gpu_mig_with_reboot=true",
                    "-var",
                    "gpu_operator_rdma_enabled=false",
                ]
                if mig_enabled
                else []
            ),
            *(
                [
                    "-var",
                    "allow_unsafe_nvswitch_operator="
                    + str(bool(allow_unsafe_nvswitch_operator)).lower(),
                ]
                if allow_unsafe_nvswitch_operator is not None
                else []
            ),
            *(
                ["-var", f"gpu_nodes_preemptible={str(bool(preemptible)).lower()}"]
                if preemptible is not None
                else []
            ),
            *_ssh_public_key_var_args(tfvars, env),
        ]
        # Once apply starts, remote backend state may exist even if the process
        # is interrupted before kubeconfig/cluster state is written.
        record_terraform_inventory(context, tf_dir)
        # Terraform prints only `Still creating...` while a node group retries, so a
        # cloud-side failure (QuotaFailure, no capacity) is invisible for as long as
        # the operator is willing to wait. Report node-group state alongside it, and
        # cancel the apply when the platform reports a refusal retrying cannot fix.
        watcher = _NodeGroupWatcher(nebius_bin, tfvars, env)
        watcher.start()
        state_existed_before_apply = any(
            candidate.is_file()
            for candidate in (
                tf_dir / "terraform.tfstate",
                tf_dir / "errored.tfstate",
            )
        )
        try:
            get_backend("mk8s").apply(
                backend_desired,
                replace(
                    adapter_request,
                    terraform_command=tuple(apply_args),
                    terraform_cwd=tf_dir,
                    terraform_env=env,
                    terraform_timeout_seconds=timeout * 60,
                    terraform_cancel_reason=lambda: watcher.fatal_reason,
                    command_runner=_run_stream,
                ),
            )
        except BaseException as exc:
            watcher.stop()
            typer.echo(
                _redacted_exception_message("terraform apply error", exc), err=True
            )
            operation = current_operation()
            rolled_back = False
            if operation is not None:
                for candidate in (
                    tf_dir / "errored.tfstate",
                    tf_dir / "terraform.tfstate",
                ):
                    if candidate.is_file():
                        operation.preserve_state_file(candidate, name=candidate.stem)
                rolled_back = _rollback_fresh_cluster_apply(
                    operation,
                    state_existed_before=state_existed_before_apply,
                    terraform_dir=tf_dir,
                    project=project,
                    context=context,
                    timeout_minutes=timeout,
                )
            if not rolled_back:
                _echo_apply_recovery(tf_dir, tfvars, isinstance(exc, KeyboardInterrupt))
            raise
        finally:
            watcher.stop()
        outputs = _terraform_outputs(terraform_bin, tf_dir, env)
        cluster = _cluster_output(outputs)
        cluster_id = str(cluster.get("id") or "")
        cluster_name = str(
            cluster.get("name") or tfvars.get("cluster_name") or "npa-cluster"
        )
        if not cluster_id:
            raise typer.BadParameter("Terraform output kube_cluster.id is empty")

        operation = current_operation()
        if operation is not None:
            operation.transition("resource-created")
            operation.record_resource(
                resource_type="managed_kubernetes_cluster",
                requested_name=cluster_name,
                provider_id=cluster_id,
                ownership="created_by_this_operation",
                ownership_source="terraform-output-and-state",
                project_id=str(
                    tfvars.get("parent_id") or env.get("TF_VAR_parent_id") or ""
                ),
            )
            output_ids = {
                "network": str(
                    (outputs.get("created_network_id") or {}).get("value") or ""
                ),
                "subnet": str(
                    (outputs.get("created_subnet_id") or {}).get("value") or ""
                ),
                "service_account": str(
                    (outputs.get("k8s_node_group_service_account_id") or {}).get(
                        "value"
                    )
                    or ""
                ),
            }
            for resource_type, requested_name in (
                ("network", f"{cluster_name}-network"),
                ("subnet", f"{cluster_name}-subnet"),
                (
                    "service_account",
                    f"{cluster_name}-k8s-node-group-sa",
                ),
            ):
                operation.record_resource(
                    resource_type=resource_type,
                    requested_name=requested_name,
                    provider_id=output_ids[resource_type],
                    ownership="created_by_this_operation",
                    ownership_source="terraform-state",
                    project_id=str(
                        tfvars.get("parent_id") or env.get("TF_VAR_parent_id") or ""
                    ),
                )
            cpu_count = int(_tfvar_value(tfvars, env, "cpu_nodes_count", 0) or 0)
            gpu_count = int(_tfvar_value(tfvars, env, "gpu_nodes_count", 0) or 0)
            node_groups = ([f"{cluster_name}-ng-cpu"] if cpu_count else []) + [
                f"{cluster_name}-ng-gpu-{index}" for index in range(gpu_count)
            ]
            for requested_name in node_groups:
                operation.record_resource(
                    resource_type="managed_kubernetes_node_group",
                    requested_name=requested_name,
                    ownership="created_by_this_operation",
                    ownership_source="terraform-state",
                    project_id=str(
                        tfvars.get("parent_id") or env.get("TF_VAR_parent_id") or ""
                    ),
                )
            local_state = tf_dir / "terraform.tfstate"
            if not local_state.is_file():
                raise typer.BadParameter(
                    "Terraform apply returned cluster outputs but no durable local "
                    "state exists; the operation journal was kept for recovery."
                )
            operation.preserve_state_file(local_state, name="verified-local")
            operation.transition("state-durable")

        kubeconfig_path = kubeconfig or kubeconfig_file(context)
        _write_kubeconfig(nebius_bin, cluster_id, kubeconfig_path, context)
        _save_terraform_cluster_state(
            tfvars,
            cluster,
            context,
            kubeconfig_path,
            env=env,
            last_seen_state="VALIDATING",
        )

        typer.echo(f"Cluster ID: {cluster_id}")
        typer.echo(f"Cluster name: {cluster_name}")
        typer.echo(f"Kubeconfig: {kubeconfig_path}")

        if mig_desired is not None:
            typer.echo("Waiting for exact two-snapshot MIG convergence...")
            verification = get_backend("mk8s").verify(
                mig_desired,
                MK8sStatusRequest(
                    kubeconfig=kubeconfig_path,
                    kubectl_bin=kubectl_bin,
                    on_status=lambda message: typer.echo(message, err=True),
                    run_capture=_run_capture,
                    mig_verifier=wait_for_mig_ready,
                    gpu_health_verifier=validate_gpu_health,
                ),
            )
            typer.echo(
                "MIG validation: "
                f"{verification['verified_nodes']} reserved workers with exact "
                "partition resources"
            )

        if validate and mig_desired is None:
            validation = _validate_cluster(
                kubectl_bin,
                kubeconfig_path,
                tfvars,
                validation_timeout,
                gpu_health_stabilization_seconds=gpu_health_stabilization_seconds,
                gpu_cuda_smoke=gpu_cuda_smoke,
                gpu_cuda_smoke_image=gpu_cuda_smoke_image,
                env=env,
            )
            typer.echo(
                "Validation: "
                f"{validation['ready_nodes']} Ready nodes, "
                f"{validation['total_gpus']} allocatable GPUs, "
                f"default StorageClass {validation['default_storage_class']}"
            )
        if sky_smoke and mig_desired is None:
            from npa.orchestration.skypilot.k8s_gpu_catalog import (
                wait_for_kubernetes_accelerators,
            )

            _check_skypilot_kubernetes(
                kubeconfig_path,
                context,
                sky_bin=sky_bin,
            )
            wait_for_kubernetes_accelerators(
                [sky_gpus] if sky_gpus.strip() else [],
                context=context,
                kubeconfig=kubeconfig_path,
                sky_bin=sky_bin or None,
                label_known_gpus=True,
                on_status=lambda message: typer.echo(message, err=True),
            )
            _run_skypilot_smoke(
                kubeconfig_path,
                context,
                cluster_name,
                sky_gpus,
                sky_bin=sky_bin,
                credentials_checked=True,
            )
        elif sky_smoke and mig_desired is not None:
            typer.echo(
                "SkyPilot whole-GPU smoke skipped: the mandatory MIG readiness "
                "gate already ran a representative MIG CUDA allocation."
            )
        _save_terraform_cluster_state(
            tfvars,
            cluster,
            context,
            kubeconfig_path,
            env=env,
            last_seen_state="RUNNING",
        )


@intent_boundary(OperationIntent.DESTROY)
@json_stdout_contract
def down_cmd(
    terraform_dir: Path | None = typer.Option(
        None,
        "--terraform-dir",
        help="Terraform cluster directory. Defaults to ./deploy/cluster or the repo root deploy/cluster.",
    ),
    project: str = typer.Option(
        "",
        "--project",
        help="NPA project alias whose saved project/tenant/region to use when tfvars omit them.",
    ),
    receipt: str = typer.Option(
        "", "--receipt", help="Opaque teardown receipt ID for alias-free recovery."
    ),
    project_id: str = typer.Option("", "--project-id", help="Exact Nebius project ID."),
    tenant_id: str = typer.Option("", "--tenant-id", help="Exact Nebius tenant ID."),
    region: str = typer.Option("", "--region", help="Exact Nebius region."),
    cluster_id: str = typer.Option(
        "", "--cluster-id", help="Exact immutable cluster ID."
    ),
    operation_id: str = typer.Option(
        "", "--operation-id", help="Exact provisioning attempt journal ID."
    ),
    context_name: str = typer.Option(
        "",
        "--context",
        help="Kubeconfig context whose local state to remove. Defaults to the Terraform cluster_name.",
    ),
    keep_local_state: bool = typer.Option(
        False,
        "--keep-local-state",
        help="Leave ~/.npa/clusters/<context>/ in place after the destroy succeeds.",
    ),
    force: bool = typer.Option(False, "--force", help="Skip confirmation."),
    timeout: int = typer.Option(
        120, "--timeout", help="Terraform destroy timeout in minutes."
    ),
    kubeconfig: Path | None = typer.Option(
        None,
        "--kubeconfig",
        help=(
            "Kubeconfig used for the non-interactive PodDisruptionBudget drain "
            "preview. Defaults to NPA's saved kubeconfig for this cluster/context, "
            "then the ambient KUBECONFIG."
        ),
    ),
    output_json: bool = typer.Option(
        False, "--json", help="Emit a machine-readable result."
    ),
) -> None:
    """Destroy the Terraform-managed NPA cluster: cloud resources and local state.

    This is the complete teardown for a cluster created by `npa cluster up` or
    `npa provision-if-absent` — the Managed Kubernetes cluster, its VPC network and
    subnet, and the local kubeconfig/state under ~/.npa/clusters/<context>/.
    (`npa cluster destroy` is the API-only path for a cluster Terraform does not
    manage; it leaves the network behind.)

    With no Terraform resource state/inventory and no NPA kubeconfig, this is a
    local no-op: it does not authenticate, initialize/download providers, inspect
    Kubernetes/RBAC, or run destroy. Real teardown uses ephemeral NPA-owned
    TF_DATA_DIR scratch and never populates deploy/cluster/.terraform.
    """

    from npa.cli.cluster.terraform_runtime import (
        has_destroy_evidence,
        isolated_terraform_data_dir,
    )

    from npa.cleanup_identity import CleanupIdentityError, resolve_cleanup_identity
    from npa.clients.config import resolve_environment

    tf_dir = _resolve_terraform_dir(terraform_dir)
    tfvars = _read_tfvars(tf_dir)
    alias = project.strip()
    saved = resolve_environment(alias) if alias else None
    live = {
        "project_alias": alias,
        "project_id": str(
            getattr(saved, "project_id", "") or tfvars.get("parent_id") or ""
        ),
        "tenant_id": str(
            getattr(saved, "tenant_id", "") or tfvars.get("tenant_id") or ""
        ),
        "region": str(getattr(saved, "region", "") or tfvars.get("region") or ""),
        "context": context_name.strip() or str(tfvars.get("cluster_name") or ""),
    }
    try:
        saved_cluster = load_cluster_state(str(live["context"] or ""))
    except Exception as exc:  # noqa: BLE001 - ownership evidence must fail closed
        raise typer.BadParameter(
            f"Local cluster ownership state is unreadable: {exc}. "
            "Nothing was deleted; repair or restore the state before retrying."
        ) from exc
    if saved_cluster is not None:
        live["project_id"] = live["project_id"] or saved_cluster.project_id
        live["region"] = live["region"] or saved_cluster.region
        live["cluster_id"] = saved_cluster.cluster_id
    try:
        cleanup_identity = resolve_cleanup_identity(
            explicit={
                "project_alias": alias,
                "project_id": project_id,
                "tenant_id": tenant_id,
                "region": region,
                "cluster_id": cluster_id,
                "operation_id": operation_id,
                "context": context_name.strip(),
                "kubeconfig_path": str(kubeconfig) if kubeconfig else "",
            },
            receipt_id=receipt,
            live=live,
            phase="cluster",
            resource=context_name.strip() or str(live.get("context") or ""),
        )
    except (CleanupIdentityError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    preview_context = str(
        cleanup_identity.get("context")
        or cleanup_identity.get("cluster_name")
        or tfvars.get("cluster_name")
        or "npa-cluster"
    )
    exact_project_id = str(cleanup_identity.get("project_id") or "")
    exact_cluster_id = str(cleanup_identity.get("cluster_id") or "")
    from npa.cluster.state import delete_cluster_state, metadata_file

    shared_metadata_path = metadata_file(preview_context)
    shared_metadata: dict[str, Any] = {}
    metadata_present = (
        shared_metadata_path.exists() or shared_metadata_path.is_symlink()
    )
    if metadata_present:
        if shared_metadata_path.is_symlink() or not shared_metadata_path.is_file():
            raise typer.BadParameter(
                "Local cluster ownership metadata is not a regular file; "
                "nothing was deleted."
            )
        try:
            candidate = json.loads(shared_metadata_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise typer.BadParameter(
                f"Local cluster ownership metadata is unreadable: {exc}. "
                "Nothing was deleted; repair or restore the metadata before retrying."
            ) from exc
        if not isinstance(candidate, dict):
            raise typer.BadParameter(
                "Local cluster ownership metadata must be a JSON object; "
                "nothing was deleted."
            )
        shared_metadata = candidate
    if saved_cluster is not None and not metadata_present:
        raise typer.BadParameter(
            "Local cluster state exists without its ownership metadata; nothing was "
            "deleted. Restore metadata before retrying."
        )
    if shared_metadata.get("managed_by") == "npa cluster shared-mk8s-backend":
        from npa.fleet.lifecycle import _reclaim_unused_project_networks

        recorded_project = str(shared_metadata.get("backend_project_id") or "")
        recorded_cluster = str(shared_metadata.get("backend_cluster_id") or "")
        if exact_project_id and exact_project_id != recorded_project:
            raise typer.BadParameter(
                "shared mk8s teardown project identity does not match local ownership"
            )
        if exact_cluster_id and exact_cluster_id != recorded_cluster:
            raise typer.BadParameter(
                "shared mk8s teardown cluster identity does not match local ownership"
            )
        backend_root = Path(
            str(shared_metadata.get("backend_state_root") or "")
        ).expanduser()
        expected_root = shared_metadata_path.parent / "backend-state"
        if (
            not backend_root.is_absolute()
            or backend_root.resolve() != expected_root.resolve()
        ):
            raise typer.BadParameter(
                "shared mk8s backend state root is missing or non-canonical"
            )
        provider_name = str(shared_metadata.get("backend_cluster_name") or "")
        project_key = str(shared_metadata.get("backend_project_key") or "")
        fleet_name = str(shared_metadata.get("backend_fleet_name") or "")
        if not (provider_name and project_key and fleet_name and recorded_project):
            raise typer.BadParameter("shared mk8s ownership metadata is incomplete")
        desired = ClusterSpec(name=provider_name, allow_control_plane_only=True)
        backend_project = ProjectSpec(
            name=project_key,
            project_id=recorded_project,
            clusters=[desired],
        )
        backend_spec = FleetSpec(
            name=fleet_name,
            tenant_id=str(shared_metadata.get("backend_tenant_id") or ""),
            region=str(shared_metadata.get("backend_region") or ""),
            projects=[backend_project],
        )
        if not force and not typer.confirm(
            f"Destroy shared-backend cluster {preview_context} ({recorded_cluster})?"
        ):
            raise typer.Abort()
        destroyed = get_backend("mk8s").destroy(
            desired,
            MK8sDestroyRequest(
                scope=MK8sExecutionScope(
                    fleet_name=backend_spec.name,
                    tenant_id=backend_spec.tenant_id,
                    region=backend_spec.region,
                    project_prefix=backend_spec.project_prefix,
                ),
                project=MK8sProjectIdentity(
                    project_key=backend_project.key(),
                    project_id=backend_project.project_id,
                    project_name=backend_project.name,
                    expected_provider_name=backend_project.display_name(
                        backend_spec.project_prefix
                    ),
                ),
                fleet_root=backend_root,
                terraform_bin=_require_bin(
                    os.environ.get("NPA_TERRAFORM_BIN") or "terraform"
                ),
                nebius_bin=_require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius"),
                profile=str(shared_metadata.get("backend_profile") or ""),
                timeout_minutes=timeout,
                on_status=lambda message: typer.echo(message, err=True),
            ),
        )
        network_results = _reclaim_unused_project_networks(
            backend_spec,
            fleet_root=backend_root,
            nebius_bin=_require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius"),
            prefix=backend_spec.project_prefix,
            only_projects=[backend_project.key()],
            profile=str(shared_metadata.get("backend_profile") or ""),
            on_status=lambda message: typer.echo(message, err=True),
        )
        network_errors = [
            str(error)
            for item in network_results
            for error in item.get("errors", [])
            if isinstance(item, dict)
        ]
        if network_errors:
            raise RuntimeError("; ".join(network_errors))
        if destroyed and destroyed.get("status") == "destroy-incomplete":
            raise RuntimeError(
                "; ".join(str(item) for item in destroyed.get("errors") or [])
            )
        if not keep_local_state:
            delete_cluster_state(preview_context)
        response = {
            "status": "destroyed",
            "backend": "mk8s",
            "cluster_id": recorded_cluster,
            "context": preview_context,
        }
        typer.echo(
            json.dumps(response) if output_json else f"Destroyed {preview_context}."
        )
        return
    if (
        metadata_present
        and shared_metadata.get("managed_by") != "npa cluster terraform"
    ):
        raise typer.BadParameter(
            "Local cluster ownership metadata does not authorize legacy Terraform "
            "destroy; nothing was deleted."
        )
    if operation_id:
        from npa.provisioning_journal import load_operation

        recovery_operations = [load_operation(operation_id)]
    else:
        recovery_operations = list_operations(
            project_alias=alias,
            project_id=exact_project_id,
            resource_type="cluster",
            requested_name=preview_context,
        )
    if exact_cluster_id:
        recovery_operations = [
            candidate
            for candidate in recovery_operations
            if any(
                isinstance(item, dict)
                and item.get("resource_type") == "managed_kubernetes_cluster"
                and str(item.get("provider_id") or "") == exact_cluster_id
                and str(item.get("project_id") or exact_project_id) == exact_project_id
                for item in candidate.read().get("resources") or []
            )
        ]
    recovery_operation = (
        recovery_operations[0] if len(recovery_operations) == 1 else None
    )
    has_evidence = has_destroy_evidence(
        tf_dir,
        preview_context,
        kubeconfig=kubeconfig,
    )
    if not has_evidence and len(recovery_operations) > 1:
        identities = sorted(
            {
                str(operation.read().get("project_id") or "(missing)")
                for operation in recovery_operations
            }
        )
        raise typer.BadParameter(
            "Cluster recovery is ambiguous across operation journals for context "
            f"{preview_context!r}: {', '.join(identities)}. Pass the exact --project; "
            "no resources were changed."
        )
    if not has_evidence and recovery_operation is not None:
        payload = recovery_operation.read()
        journal_project_id = str(payload.get("project_id") or "")
        if project:
            from npa.clients.config import resolve_environment

            saved = resolve_environment(project)
            saved_project_id = str(getattr(saved, "project_id", "") or "")
            if saved_project_id and journal_project_id != saved_project_id:
                raise typer.BadParameter(
                    "Cluster recovery project mismatch between config and operation "
                    "journal; no resources were changed."
                )
        copies = recovery_operation.state_copies()
        if copies:
            shutil.copy2(copies[0], tf_dir / "terraform.tfstate")
            (tf_dir / "terraform.tfstate").chmod(0o600)
            has_evidence = True
            typer.echo(
                "Recovered exact Terraform state from operation "
                f"{recovery_operation.operation_id}: {copies[0]}",
                err=True,
            )
        else:
            resources = [
                dict(item)
                for item in payload.get("resources") or []
                if isinstance(item, dict)
                and item.get("ownership") == "created_by_this_operation"
                and str(item.get("project_id") or "") == exact_project_id
            ]
            if not resources and str(payload.get("phase") or "") == "prepared":
                # A journal that never crossed the mutation boundary and has no
                # durable state or operation-owned inventory has nothing that may
                # be deleted.  In particular, do not call the provider with an
                # empty cluster ID: doing so both fails recovery and risks turning
                # a precise operation cleanup into name-based discovery.
                recovery_operation.transition("destroyed")
                message = (
                    f"Operation {recovery_operation.operation_id} recorded no "
                    "cluster mutation or owned resources; nothing to do. "
                    "Terraform, provider, and Kubernetes APIs were not invoked."
                )
                result_payload = {
                    **cleanup_identity.to_dict(),
                    "outcome": "already_absent",
                    "verified": True,
                    "no_op": True,
                    "state_consumers_absent": True,
                    "resources_removed": [],
                    "message": message,
                }
                if output_json:
                    typer.echo(json.dumps(result_payload, indent=2, sort_keys=True))
                else:
                    typer.echo(f"identity_source: {cleanup_identity.source}")
                    typer.echo(message)
                return
            required_types = {
                str(item.get("resource_type") or "")
                for item in resources
                if str(item.get("resource_type") or "")
                in {
                    "managed_kubernetes_cluster",
                    "network",
                    "subnet",
                    "service_account",
                }
            }
            unresolved = sorted(
                str(item.get("resource_type") or "")
                for item in resources
                if str(item.get("resource_type") or "") in required_types
                and not str(item.get("provider_id") or "").strip()
            )
            if unresolved:
                raise typer.BadParameter(
                    "The exact operation has no recoverable Terraform state and its "
                    "provider inventory lacks immutable IDs for: "
                    + ", ".join(unresolved)
                    + ". No provider mutation ran; restore preserved state or retry "
                    "with a newer operation receipt."
                )
            if not force and not typer.confirm(
                f"Destroy exact operation-owned provider inventory for {preview_context}?"
            ):
                raise typer.Exit(1)
            from npa.cluster.api import MK8sClient
            from npa.cluster.exceptions import ClusterNotFoundError
            from npa.clients.nebius import (
                NebiusError,
                delete_network,
                delete_service_account,
                delete_subnet,
            )
            from npa.teardown_receipts import record_teardown_event

            ids = {
                str(item.get("resource_type") or ""): str(item.get("provider_id") or "")
                for item in resources
                if str(item.get("provider_id") or "")
            }
            errors: list[str] = []
            removed: list[dict[str, str]] = []
            cluster_removed = False
            target_cluster = ids.get("managed_kubernetes_cluster", exact_cluster_id)
            try:
                client = MK8sClient(timeout=timeout * 60, poll_interval=30.0)
                try:
                    client.delete_cluster(target_cluster, project_id=exact_project_id)
                    client.wait_for_deleted(
                        target_cluster,
                        project_id=exact_project_id,
                        timeout_minutes=timeout,
                    )
                except ClusterNotFoundError:
                    pass
                cluster_removed = True
                removed.append(
                    {"type": "managed_kubernetes_cluster", "id": target_cluster}
                )
            except Exception as exc:  # noqa: BLE001 - retain independent phase evidence
                errors.append(
                    f"managed_kubernetes_cluster: {type(exc).__name__}: {exc}"
                )
            if cluster_removed:
                for kind, delete_fn in (
                    ("service_account", delete_service_account),
                    ("subnet", delete_subnet),
                    ("network", delete_network),
                ):
                    resource_id = ids.get(kind, "")
                    if not resource_id:
                        continue
                    try:
                        delete_fn(
                            resource_id,
                            **(
                                {"profile": str(cleanup_identity.get("profile") or "")}
                                if cleanup_identity.get("profile")
                                else {}
                            ),
                        )
                        removed.append({"type": kind, "id": resource_id})
                    except NebiusError as exc:
                        errors.append(f"{kind}: {exc}")
            record_teardown_event(
                phase="cluster",
                resource=preview_context,
                terminal_state="verified_deleted" if not errors else "partial",
                project_alias=alias,
                project_id=exact_project_id,
                context=preview_context,
                identity=cleanup_identity.values,
                precheck={
                    "identity_source": cleanup_identity.source,
                    "state_source": "operation_inventory",
                },
                action={"kind": "delete_exact_operation_inventory", "removed": removed},
                verification={
                    "state_consumers_absent": cluster_removed,
                    "errors": errors,
                },
                errors=errors,
            )
            result_payload = {
                **cleanup_identity.to_dict(),
                "outcome": "verified_deleted" if not errors else "partial",
                "verified": not errors,
                "state_consumers_absent": cluster_removed,
                "resources_removed": removed,
                "errors": errors,
            }
            if output_json:
                typer.echo(json.dumps(result_payload, indent=2, sort_keys=True))
            else:
                typer.echo(f"identity_source: {cleanup_identity.source}")
                typer.echo(
                    f"Removed {len(removed)} exact operation-owned provider resources."
                )
                for error in errors:
                    typer.echo(f"Warning: {error}", err=True)
            if errors:
                raise typer.Exit(code=2)
            if not keep_local_state:
                _clear_local_cluster_state(preview_context)
            return
    if not has_evidence:
        if cleanup_identity.receipt_is_terminal:
            payload = {
                **cleanup_identity.to_dict(),
                "outcome": "already_absent",
                "verified": True,
                "no_op": True,
                "message": f"Cluster {preview_context!r} is already absent per terminal receipt evidence.",
            }
            if output_json:
                typer.echo(json.dumps(payload, indent=2, sort_keys=True))
            else:
                typer.echo(f"identity_source: {cleanup_identity.source}")
                typer.echo(payload["message"])
            return
        if receipt or project_id or cluster_id:
            if not exact_project_id or not exact_cluster_id:
                raise typer.BadParameter(
                    "No cluster state exists. Pass --receipt containing both exact project and "
                    "cluster IDs, or pass --project-id and --cluster-id; Terraform was not invoked."
                )
            from npa.cluster.api import MK8sClient
            from npa.cluster.exceptions import ClusterNotFoundError

            try:
                MK8sClient().get_cluster(exact_cluster_id, project_id=exact_project_id)
            except ClusterNotFoundError:
                from npa.teardown_receipts import record_teardown_event

                record_teardown_event(
                    phase="cluster",
                    resource=preview_context,
                    terminal_state="verified_absent",
                    project_alias=alias,
                    project_id=exact_project_id,
                    context=preview_context,
                    identity=cleanup_identity.values,
                    precheck={"identity_source": cleanup_identity.source},
                    action={"kind": "exact_provider_check", "mutation": False},
                    verification={"provider_absence": "verified"},
                )
                message = (
                    f"Provider verified exact cluster {exact_cluster_id} is absent; "
                    "nothing to do. Terraform was not invoked."
                )
                if output_json:
                    typer.echo(
                        json.dumps(
                            {
                                **cleanup_identity.to_dict(),
                                "outcome": "already_absent",
                                "verified": True,
                                "no_op": True,
                                "message": message,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                    )
                else:
                    typer.echo(f"identity_source: {cleanup_identity.source}")
                    typer.echo(message)
                return
            except Exception as exc:
                raise typer.BadParameter(
                    f"Exact cluster verification is unresolved: {exc}. Terraform was not invoked."
                ) from exc
            raise typer.BadParameter(
                f"Exact cluster {exact_cluster_id} is present, but no complete Terraform "
                "state is available. Restore the receipt-recorded state/backend; Terraform "
                "was not invoked and nothing was deleted."
            )
        typer.echo(f"identity_source: {cleanup_identity.source}")
        typer.echo(
            f"No Terraform-managed cluster is recorded for {preview_context!r}: "
            "no Terraform resource state/inventory or NPA kubeconfig was found. "
            "Nothing to do. Terraform init/provider downloads, Nebius "
            "authentication, Kubernetes/RBAC calls, and destroy were not invoked."
        )
        return

    terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    _preflight_provider_lock(tf_dir)
    if not force and not typer.confirm(
        f"Destroy Terraform-managed cluster in {tf_dir}?"
    ):
        raise typer.Exit(1)
    confirmed_full_destroy = True
    _preflight_terraform_version(terraform_bin)
    # Prefer the kubeconfig NPA saved for this exact cluster/context instead of
    # an unrelated ambient current-context. The preview rewrites only a temporary
    # copy so its exec credential cannot launch browser authentication.
    with isolated_terraform_data_dir(tf_dir, preview_context) as terraform_data:
        env = _terraform_env(nebius_bin)
        env["TF_DATA_DIR"] = str(terraform_data)
        for key, variable in (
            ("project_id", "TF_VAR_parent_id"),
            ("tenant_id", "TF_VAR_tenant_id"),
            ("region", "TF_VAR_region"),
        ):
            tfvar_key = "parent_id" if key == "project_id" else key
            if tfvar_key not in tfvars and cleanup_identity.get(key):
                env[variable] = str(cleanup_identity.get(key))
        _apply_project_tf_vars(env, alias, tfvars)
        _guard_tfvars_iam_token(tf_dir, tfvars)
        preview_kubeconfig = kubeconfig
        preview_state_issue = None
        if preview_kubeconfig is None:
            from npa.cluster.exceptions import ClusterStateError
            from npa.cluster.state import existing_kubeconfig

            try:
                preview_kubeconfig = existing_kubeconfig(preview_context)
            except (OSError, ClusterStateError):
                from npa.cluster.drain import DrainPreviewIssue

                preview_state_issue = DrainPreviewIssue(
                    "kubeconfig",
                    "NPA's saved kubeconfig reference or cluster state could not be loaded",
                )
        verified_identity = None
        identity_error = ""
        try:
            from npa.cluster.identity import resolve_verified_cluster_identity

            verified_identity = resolve_verified_cluster_identity(
                project=project,
                context=preview_context,
                kubeconfig=preview_kubeconfig,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            identity_error = str(exc)
            typer.echo(
                "drain-policy: system PDB relaxation is disabled because the exact "
                f"NPA project/context identity was not verified: {identity_error}",
                err=True,
            )
        # The node-group watcher below shows that the drain is progressing; this names
        # *why* it is slow and safely relaxes only exact managed system add-ons.
        if preview_state_issue is not None:
            from npa.cluster.drain import describe_preview_unavailable

            typer.echo(
                f"drain-preview: {describe_preview_unavailable(preview_state_issue)}",
                err=True,
            )
        else:
            preview_inventory = _report_drain_blockers(
                preview_kubeconfig, context=preview_context
            )
        if preview_state_issue is not None:
            preview_inventory = None
        _terraform_init(terraform_bin, tf_dir, env)
        if saved_cluster is not None or exact_cluster_id:
            if not exact_cluster_id:
                raise typer.BadParameter(
                    "Legacy Terraform destroy requires an exact persisted cluster ID; "
                    "nothing was deleted."
                )
            managed_cluster_ids = _terraform_state_cluster_ids(
                terraform_bin, tf_dir, env
            )
            if managed_cluster_ids and managed_cluster_ids != {exact_cluster_id}:
                raise typer.BadParameter(
                    "Legacy Terraform state does not own exactly the persisted cluster "
                    f"ID {exact_cluster_id}; nothing was deleted."
                )
            if not managed_cluster_ids:
                if (
                    saved_cluster is None
                    or not metadata_present
                    or shared_metadata.get("managed_by") != "npa cluster terraform"
                    or saved_cluster.cluster_id != exact_cluster_id
                    or saved_cluster.project_id != exact_project_id
                    or saved_cluster.name != preview_context
                ):
                    raise typer.BadParameter(
                        "Legacy residual recovery lacks matching persisted cluster, "
                        "project, context, and Terraform ownership evidence; nothing "
                        "was deleted."
                    )
                residual_types = _verify_residual_terraform_ownership(
                    terraform_bin,
                    tf_dir,
                    env,
                    project_id=exact_project_id,
                    cluster_id=exact_cluster_id,
                )
                from npa.cluster.api import MK8sClient
                from npa.cluster.exceptions import ClusterNotFoundError

                try:
                    MK8sClient(timeout=120, poll_interval=15.0).get_cluster(
                        exact_cluster_id,
                        project_id=exact_project_id,
                    )
                except ClusterNotFoundError:
                    typer.echo(
                        "Legacy recovery: exact cluster is provider-absent; "
                        "destroying retained Terraform-owned residuals ("
                        + ", ".join(residual_types)
                        + ").",
                        err=True,
                    )
                except Exception as exc:
                    raise typer.BadParameter(
                        "Legacy recovery could not prove exact provider cluster "
                        f"absence: {exc}. Nothing was deleted."
                    ) from exc
                else:
                    raise typer.BadParameter(
                        "Legacy recovery exact cluster is still present while retained "
                        "Terraform state no longer owns it; nothing was deleted."
                    )
        # `Still destroying...` every 10s with no detail made a ~6-minute node-group
        # drain look like a hang. Report node-group state while it happens.
        watcher = _NodeGroupWatcher(nebius_bin, tfvars, env)
        watcher.start()
        pdb_report = None
        try:
            from npa.cluster.drain import relax_system_pdbs_for_full_destroy

            with relax_system_pdbs_for_full_destroy(
                preview_inventory,
                context=preview_context,
                kubeconfig=(
                    str(verified_identity.kubeconfig)
                    if verified_identity is not None
                    else str(preview_kubeconfig or "")
                ),
                confirmed_full_destroy=confirmed_full_destroy,
                identity_verified=(
                    verified_identity is not None
                    and not verified_identity.cluster_absent
                ),
            ) as pdb_report:
                if pdb_report.eligible:
                    typer.echo(
                        "drain-policy: normal eviction was attempted for verified "
                        "cluster-system blockers; temporary full-destroy relaxation: "
                        + ", ".join(pdb_report.eligible),
                        err=True,
                    )
                if pdb_report.user_or_unverified:
                    typer.echo(
                        "drain-policy: preserved user/unverified PDB blockers: "
                        + ", ".join(pdb_report.user_or_unverified),
                        err=True,
                    )
                for error in pdb_report.errors:
                    typer.echo(
                        f"drain-policy: PDB relaxation failed safely: {error}",
                        err=True,
                    )
                _run_stream(
                    [
                        terraform_bin,
                        "destroy",
                        "-auto-approve",
                        # Variable validation runs on destroy too, so the key has to resolve
                        # here as well — but a teardown must not be blocked by a machine that
                        # has no SSH key, since the value cannot affect what is destroyed.
                        *_ssh_public_key_var_args(tfvars, env, allow_placeholder=True),
                    ],
                    cwd=tf_dir,
                    env=env,
                    timeout=timeout * 60,
                )
        except BaseException as exc:
            try:
                from npa.teardown_receipts import record_teardown_event

                record_teardown_event(
                    phase="cluster",
                    resource=preview_context,
                    terminal_state="failed",
                    project_alias=(
                        verified_identity.project_alias
                        if verified_identity is not None
                        else project
                    ),
                    project_id=(
                        verified_identity.project_id
                        if verified_identity is not None
                        else str(tfvars.get("parent_id") or "")
                    ),
                    context=preview_context,
                    precheck={
                        "identity_verified": verified_identity is not None,
                        "identity_error": identity_error,
                    },
                    action={
                        "kind": "terraform_full_cluster_destroy",
                        "system_pdbs_temporarily_removed": list(
                            getattr(pdb_report, "removed", []) or []
                        ),
                        "system_pdbs_restored": list(
                            getattr(pdb_report, "restored", []) or []
                        ),
                    },
                    verification={"terraform_destroy": "failed"},
                    errors=[f"{type(exc).__name__}: {exc}"],
                    identity=cleanup_identity.values,
                )
            except (OSError, RuntimeError, ValueError) as receipt_exc:
                typer.echo(
                    f"Warning: cluster failure receipt could not be written: {receipt_exc}",
                    err=True,
                )
            raise
        finally:
            watcher.stop()
        from npa.teardown_receipts import record_teardown_event

        record_teardown_event(
            phase="cluster",
            resource=preview_context,
            terminal_state="verified_deleted",
            project_alias=(
                verified_identity.project_alias
                if verified_identity is not None
                else project
            ),
            project_id=(
                verified_identity.project_id
                if verified_identity is not None
                else str(tfvars.get("parent_id") or "")
            ),
            context=preview_context,
            precheck={
                "identity_verified": verified_identity is not None,
                "cluster_id": (
                    verified_identity.cluster_id
                    if verified_identity is not None
                    else ""
                ),
                "full_destroy_confirmed": confirmed_full_destroy,
            },
            action={
                "kind": "terraform_full_cluster_destroy",
                "normal_eviction_attempts": list(
                    getattr(pdb_report, "eviction_attempts", []) or []
                ),
                "system_pdbs_temporarily_removed": list(
                    getattr(pdb_report, "removed", []) or []
                ),
                "user_pdbs_preserved": list(
                    getattr(pdb_report, "user_or_unverified", []) or []
                ),
            },
            verification={"terraform_destroy": "completed"},
            identity=cleanup_identity.values,
        )
        if not keep_local_state:
            _clear_local_cluster_state(preview_context)
        if recovery_operation is not None:
            phase = str(recovery_operation.read().get("phase") or "")
            if phase not in {"committed", "destroyed"}:
                recovery_operation.transition("destroyed")


def _clear_local_cluster_state(context: str) -> None:
    """Remove ``~/.npa/clusters/<context>/`` after the cloud resources are gone.

    A successful destroy used to leave the state and kubeconfig behind, so
    `npa cluster list` still showed the cluster (as UNKNOWN, with a kubeconfig
    path that resolves nothing) and `npa cluster destroy` was needed purely to
    delete local files.
    """
    from npa.cluster.state import cluster_dir, delete_cluster_state

    directory = cluster_dir(context)
    if not directory.exists():
        return
    delete_cluster_state(context)
    typer.echo(f"Removed local cluster state {directory}")


def kubeconfig_cmd(
    cluster_name: str = typer.Option(
        "",
        "--cluster-name",
        help="Managed Kubernetes cluster name. Defaults to the Terraform cluster_name, else npa-cluster.",
    ),
    project_id: str = typer.Option(
        "",
        "--project-id",
        help="Nebius project id holding the cluster. Defaults to tfvars/TF_VAR_parent_id, then the configured project.",
    ),
    project: str = typer.Option(
        "", "--project", help="NPA project alias whose saved project_id to use."
    ),
    context_name: str = typer.Option(
        "", "--context", help="Kubeconfig context name. Defaults to the cluster name."
    ),
    kubeconfig: Path | None = typer.Option(
        None,
        "--kubeconfig",
        help="Kubeconfig output path. Defaults to ~/.npa/clusters/<context>/kubeconfig.",
    ),
    terraform_dir: Path | None = typer.Option(
        None, "--terraform-dir", help="Terraform cluster directory to read tfvars from."
    ),
) -> None:
    """Write a kubeconfig for a Managed Kubernetes cluster that already exists.

    An interrupted `npa cluster up` (or one provisioned elsewhere) leaves a running
    cluster with no local kubeconfig, which nothing could then use. This adopts it:
    it writes the kubeconfig and cluster state that `npa cluster status` and
    `npa workbench workflow submit --infra k8s/<context>` read.
    """
    from npa.clients.config import resolve_environment

    nebius_bin = _require_bin(os.environ.get("NPA_NEBIUS_BIN") or "nebius")
    env = _terraform_env(nebius_bin)
    tfvars: dict[str, Any] = {}
    try:
        tfvars = _read_tfvars(_resolve_terraform_dir(terraform_dir))
    except typer.BadParameter:
        # Adopting a cluster does not require a Terraform directory at all.
        tfvars = {}

    name = cluster_name.strip() or str(tfvars.get("cluster_name") or "npa-cluster")
    resolved_project = project_id.strip() or str(
        tfvars.get("parent_id") or os.environ.get("TF_VAR_parent_id") or ""
    )
    if not resolved_project:
        saved = resolve_environment(project or None)
        resolved_project = str(getattr(saved, "project_id", "") or "")
    if not resolved_project:
        raise typer.BadParameter(
            "Cannot tell which Nebius project holds the cluster. Pass --project-id "
            "<id> (or --project <alias> after `npa configure`)."
        )

    result = _run_capture(
        [
            nebius_bin,
            "mk8s",
            "cluster",
            "list",
            "--parent-id",
            resolved_project,
            "--format",
            "json",
        ],
        env=env,
    )
    matches = [
        item
        for item in (json.loads(result.stdout or "{}") or {}).get("items", [])
        if str((item.get("metadata") or {}).get("name", "")) == name
    ]
    if not matches:
        raise typer.BadParameter(
            f"No Managed Kubernetes cluster named {name!r} in project {resolved_project}. "
            f"List what exists with `nebius mk8s cluster list --parent-id {resolved_project}`."
        )
    metadata = matches[0].get("metadata") or {}
    cluster_id = str(metadata.get("id") or "")
    if not cluster_id:
        raise typer.BadParameter(f"Cluster {name!r} has no id in the Nebius response")

    context = context_name.strip() or name
    kubeconfig_path = kubeconfig or kubeconfig_file(context)
    _write_kubeconfig(nebius_bin, cluster_id, kubeconfig_path, context)
    _save_terraform_cluster_state(
        {**tfvars, "parent_id": resolved_project, "cluster_name": name},
        {"id": cluster_id, "name": name},
        context,
        kubeconfig_path,
        env=env,
    )
    typer.echo(f"Cluster ID: {cluster_id}")
    typer.echo(f"Kubeconfig: {kubeconfig_path}")
    typer.echo(f"Context: {context}")
    typer.echo(
        f"Submit against it with `--infra k8s/{context}` (npa resolves this file), "
        f"or export KUBECONFIG={kubeconfig_path} for kubectl."
    )


def _report_drain_blockers(kubeconfig: Path | None, *, context: str = ""):
    """Inspect the same cluster-wide inventory the full node drain will affect.

    This is best-effort and preview-only. Authentication, RBAC, kubeconfig and
    API failures are explained, but none blocks the Terraform destroy. NPA does
    not mutate PDBs or force-delete their protected pods.
    """

    from npa.cluster.drain import (
        drain_inventory,
        describe_drain_expectation,
        describe_preview_unavailable,
    )

    selected_kubeconfig = str(kubeconfig) if kubeconfig else ""
    inventory, issue = drain_inventory(
        kubeconfig=selected_kubeconfig,
        context=context,
    )
    if issue is not None:
        typer.echo(f"drain-preview: {describe_preview_unavailable(issue)}", err=True)
        return None
    blockers = list(inventory.blockers) if inventory is not None else []
    if not blockers:
        typer.echo(
            "drain-preview: inspected the cluster-wide node/pod/controller/PDB "
            "inventory with eviction selector and placement semantics; no PDB will "
            "deny an eviction in the observed drain.",
            err=True,
        )
        return inventory

    guidance = describe_drain_expectation(blockers)
    if guidance:
        typer.echo(f"drain-preview: {guidance}", err=True)
    return inventory


def terraform_status(terraform_dir: Path | None = None) -> dict[str, Any] | None:
    """Return Terraform cluster outputs when state exists."""

    try:
        tf_dir = _resolve_terraform_dir(terraform_dir)
        terraform_bin = _require_bin(os.environ.get("NPA_TERRAFORM_BIN") or "terraform")
        env = os.environ.copy()
        return _terraform_outputs(terraform_bin, tf_dir, env)
    except Exception:
        return None


def _resolve_terraform_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise typer.BadParameter(f"Terraform directory does not exist: {path}")
        return path
    cwd_candidate = (Path.cwd() / _DEFAULT_TERRAFORM_SUBDIR).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_root = _find_repo_root(Path.cwd())
    if repo_root is not None:
        repo_candidate = (repo_root / _DEFAULT_TERRAFORM_SUBDIR).resolve()
        if repo_candidate.exists():
            return repo_candidate
    raise typer.BadParameter("Cannot find deploy/cluster; pass --terraform-dir")


def _find_repo_root(path: Path) -> Path | None:
    for current in [path, *path.parents]:
        if (current / ".git").exists():
            return current
    return None


def _require_bin(binary: str) -> str:
    from npa.cluster_backends.process import BackendCommandError, require_bin

    try:
        return require_bin(binary)
    except BackendCommandError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _apply_project_tf_vars(
    env: dict[str, str], project: str, tfvars: dict[str, Any]
) -> None:
    """Fill TF_VAR_parent_id/tenant_id/region from ``~/.npa/config.yaml``.

    `npa provision-if-absent` exports these before calling `up`, so a cluster
    provisioned that way has no `terraform.tfvars` — and a later bare
    `npa cluster down --force` then failed with "No value for required variable",
    leaving the VPC/subnet orphaned until the operator exported them by hand.
    Values already in tfvars or the environment win.
    """
    missing = [
        key
        for key, var in (
            ("parent_id", "TF_VAR_parent_id"),
            ("tenant_id", "TF_VAR_tenant_id"),
            ("region", "TF_VAR_region"),
        )
        if key not in tfvars and not str(env.get(var, "") or "").strip()
    ]
    if not missing:
        return
    try:
        from npa.clients.config import resolve_environment

        saved = resolve_environment(project or None)
    except Exception:  # noqa: BLE001 - no saved config is a normal first run
        saved = None
    if saved is None:
        return
    resolved: list[str] = []
    for key, var, value in (
        ("parent_id", "TF_VAR_parent_id", str(getattr(saved, "project_id", "") or "")),
        ("tenant_id", "TF_VAR_tenant_id", str(getattr(saved, "tenant_id", "") or "")),
        ("region", "TF_VAR_region", str(getattr(saved, "region", "") or "")),
    ):
        if key in missing and value:
            env[var] = value
            resolved.append(f"{key}={value}")
    if resolved:
        typer.echo(
            f"Using saved project settings from ~/.npa/config.yaml: {', '.join(resolved)}"
        )


def _terraform_env(nebius_bin: str, *, profile: str = "") -> dict[str, str]:
    """Terraform env with a freshly minted ``TF_VAR_iam_token``.

    ``profile`` selects a non-default ``~/.nebius`` profile, so a caller
    targeting a different tenant mints the token for *that* principal instead of
    the machine's active profile.
    """
    from npa.cluster_backends.process import BackendCommandError, terraform_env

    try:
        return terraform_env(nebius_bin, profile=profile, capture_runner=_run_capture)
    except BackendCommandError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _run_stream(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    cancel: Callable[[], str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """CLI compatibility wrapper around the backend-neutral runner."""

    from npa.cluster_backends.process import BackendCommandError, run_stream

    try:
        return run_stream(
            args,
            cwd=cwd,
            env=env,
            timeout=timeout,
            cancel=cancel,
            capture_output=capture_output,
        )
    except BackendCommandError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _stop_process(process: subprocess.Popen[str]) -> None:
    """Compatibility alias for tests and older internal callers."""

    from npa.cluster_backends.process import _stop_process as stop_process

    stop_process(process)


def _run_capture(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    from npa.cluster_backends.process import BackendCommandError, run_capture

    try:
        return run_capture(
            args,
            cwd=cwd,
            env=env,
            timeout=timeout,
            check=check,
            input_text=input_text,
        )
    except BackendCommandError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _terraform_init(
    terraform_bin: str,
    terraform_dir: Path,
    env: dict[str, str],
) -> None:
    """Initialize in isolated data while keeping the tracked lock immutable."""

    from npa.terraform_lock import (
        TerraformLockError,
        configure_plugin_cache,
        validate_provider_lock,
    )

    try:
        validate_provider_lock(terraform_dir)
        configure_plugin_cache(
            env,
            terraform_dir,
            default_root=Path.home() / ".npa" / "terraform-plugin-cache",
        )
    except (OSError, TerraformLockError) as exc:
        raise typer.BadParameter(
            f"Terraform provider-lock/cache preflight failed: {exc}"
        ) from exc

    lock_file = terraform_dir / ".terraform.lock.hcl"
    try:
        lock_before = lock_file.read_bytes()
    except OSError:
        lock_before = None
    init_error: typer.BadParameter | None = None
    try:
        _run_stream(
            [terraform_bin, "init", "-lockfile=readonly"],
            cwd=terraform_dir,
            env=env,
            timeout=600,
            capture_output=True,
        )
    except typer.BadParameter as exc:
        init_error = exc
    try:
        lock_after = lock_file.read_bytes()
    except OSError:
        lock_after = None
    if lock_before != lock_after:
        raise typer.BadParameter(
            f"Terraform changed {lock_file} despite -lockfile=readonly. NPA stopped "
            "without running apply/destroy; restore and review the tracked lock file "
            "before retrying."
        )
    if init_error is None:
        return

    detail = str(init_error)
    lowered = detail.lower()
    checksum_mismatch = any(
        marker in lowered
        for marker in (
            "doesn't match any of the checksums",
            "does not match any of the checksums",
            "doesn't match the checksums",
            "does not match the checksums",
            "checksum mismatch",
        )
    )
    if checksum_mismatch:
        raise typer.BadParameter(
            "Terraform provider checksum verification failed: the downloaded "
            f"provider does not match the tracked checksums in {lock_file}. NPA "
            "did not modify the lock file.\nChecksum bypass is forbidden. Verify "
            "the configured provider source/release and any network mirror, then "
            "reconcile checksums in a clean reviewed checkout with `terraform "
            f"-chdir={terraform_dir} providers lock -platform=<target-platform>`; "
            "review the exact lock-file diff before committing and retrying. "
            f"Terraform detail: {detail[-3000:]}"
        )
    raise typer.BadParameter(f"Terraform init failed: {detail[-3000:]}") from init_error


def _preflight_provider_lock(terraform_dir: Path) -> str:
    """Fail before authentication/provisioning if this host lacks lock coverage."""

    from npa.terraform_lock import TerraformLockError, validate_provider_lock

    try:
        target_platform = validate_provider_lock(terraform_dir)
    except TerraformLockError as exc:
        raise typer.BadParameter(
            f"Terraform provider-lock preflight failed: {exc}"
        ) from exc
    typer.echo(f"Terraform provider lock: verified for {target_platform}")
    return target_platform


def _read_tfvars(terraform_dir: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for path in [
        terraform_dir / "terraform.tfvars",
        *sorted(terraform_dir.glob("*.auto.tfvars")),
    ]:
        if not path.exists():
            continue
        for key, raw_value in _tfvar_assignments(path.read_text(encoding="utf-8")):
            values[key] = _parse_tfvar_scalar(raw_value)
    return values


def _tfvar_assignments(document: str) -> list[tuple[str, str]]:
    """Read complete top-level HCL assignments, including multiline objects/maps."""

    assignments: list[tuple[str, str]] = []
    key = ""
    parts: list[str] = []
    depth = 0
    quote = ""
    escaped = False
    for raw_line in document.splitlines():
        line = raw_line
        if not key:
            match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
            if not match:
                continue
            key, value = match.groups()
            parts = [value]
        else:
            parts.append(line)
        for character in parts[-1]:
            if escaped:
                escaped = False
                continue
            if quote and character == "\\":
                escaped = True
                continue
            if character in {'"', "'"}:
                if not quote:
                    quote = character
                elif quote == character:
                    quote = ""
                continue
            if quote:
                continue
            if character == "#":
                break
            if character in "[{(":
                depth += 1
            elif character in "]})":
                depth -= 1
        if not quote and depth <= 0:
            value = "\n".join(parts).strip()
            value = re.sub(r"\s+#.*$", "", value).strip()
            assignments.append((key, value))
            key = ""
            parts = []
            depth = 0
    if key:
        assignments.append((key, "\n".join(parts).strip()))
    return assignments


def _apply_capacity_block_group_env(
    env: dict[str, str], capacity_block_group: str
) -> None:
    value = capacity_block_group.strip()
    if value:
        env["TF_VAR_capacity_block_group"] = value


def _capacity_block_group_var_args(capacity_block_group: str) -> list[str]:
    value = capacity_block_group.strip()
    if not value:
        return []
    return ["-var", f"capacity_block_group={value}"]


#: Node-group SSH keys, most modern first. `ssh-keygen` has defaulted to ed25519
#: for years, and the rest of the CLI (agent deploy, the tfvars example) uses it.
_SSH_PUBLIC_KEY_NAMES = ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub")


def _resolve_shared_ssh_public_key(tfvars: dict[str, Any], env: dict[str, str]) -> str:
    """Resolve legacy path-or-key tfvars into the shared recipe's key value."""

    raw = tfvars.get("ssh_public_key")
    if raw is None:
        raw = env.get("TF_VAR_ssh_public_key", "")
    document = str(raw or "").strip()
    key_match = re.search(r'\bkey\s*=\s*"([^"]+)"', document, re.DOTALL)
    if key_match:
        return key_match.group(1).strip()
    path_match = re.search(r'\bpath\s*=\s*"([^"]+)"', document, re.DOTALL)
    if path_match:
        path = Path(path_match.group(1)).expanduser()
        if not path.is_file():
            raise typer.BadParameter(f"SSH public key path does not exist: {path}")
        return path.read_text(encoding="utf-8").strip()
    if document and document.startswith(("ssh-", "ecdsa-")):
        return document
    explicit = os.environ.get("NPA_SSH_PUBLIC_KEY", "").strip()
    candidates = (
        [Path(explicit).expanduser()]
        if explicit
        else [Path.home() / ".ssh" / name for name in _SSH_PUBLIC_KEY_NAMES]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    searched = ", ".join(str(path) for path in candidates)
    raise typer.BadParameter(
        f"No SSH public key found for the cluster node groups (looked at {searched}). "
        "Create one with `ssh-keygen -t ed25519`, point NPA_SSH_PUBLIC_KEY at an "
        "existing key, or set ssh_public_key in terraform.tfvars."
    )


def _ssh_public_key_var_args(
    tfvars: dict[str, Any], env: dict[str, str], *, allow_placeholder: bool = False
) -> list[str]:
    """Pin an SSH public key that exists on this machine.

    The vendored module validates ``fileexists(ssh_public_key.path)`` against a
    default of ``~/.ssh/id_rsa.pub``, so the zero-config path
    (``npa provision-if-absent``, which passes no key) fails at plan time on any
    machine that only has an ed25519 key. Resolve one here instead; an explicit
    ``ssh_public_key`` in tfvars or ``TF_VAR_ssh_public_key`` always wins.

    ``allow_placeholder`` is for destroy: variable validation still runs, but the
    value is irrelevant to tearing resources down, and a teardown must not be
    blocked by a missing key on the machine doing the cleanup.
    """
    if (
        "ssh_public_key" in tfvars
        or str(env.get("TF_VAR_ssh_public_key", "") or "").strip()
    ):
        return []
    explicit = os.environ.get("NPA_SSH_PUBLIC_KEY", "").strip()
    candidates = (
        [Path(explicit).expanduser()]
        if explicit
        else [Path.home() / ".ssh" / name for name in _SSH_PUBLIC_KEY_NAMES]
    )
    for candidate in candidates:
        if candidate.is_file():
            return ["-var", f'ssh_public_key={{path="{candidate}"}}']
    if allow_placeholder:
        return [
            "-var",
            'ssh_public_key={key="ssh-ed25519 AAAA npa-teardown-placeholder"}',
        ]
    searched = ", ".join(str(path) for path in candidates)
    raise typer.BadParameter(
        f"No SSH public key found for the cluster node groups (looked at {searched}). "
        "Create one with `ssh-keygen -t ed25519`, point NPA_SSH_PUBLIC_KEY at an "
        "existing key, or set ssh_public_key in terraform.tfvars."
    )


def _preflight_terraform_version(terraform_bin: str) -> None:
    """Fail early when the terraform binary is older than the vendored modules need."""
    result = _run_capture([terraform_bin, "version", "-json"], check=False)
    version = ""
    if result.returncode == 0:
        try:
            version = str(
                json.loads(result.stdout or "{}").get("terraform_version") or ""
            )
        except json.JSONDecodeError:
            version = ""
    parsed = _parse_semver(version)
    if parsed is None:
        # Never block on an unparseable version; terraform itself will complain.
        return
    if parsed >= _MIN_TERRAFORM_VERSION:
        return
    minimum = ".".join(str(part) for part in _MIN_TERRAFORM_VERSION)
    raise typer.BadParameter(
        f"Terraform {version} is too old for deploy/cluster: it vendors modules that "
        f"require Terraform >= {minimum} (and use `ephemeral` blocks). "
        f"Install a newer Terraform (https://developer.hashicorp.com/terraform/install), "
        f"then re-run; point NPA_TERRAFORM_BIN at it to keep the old binary on PATH."
    )


def _parse_semver(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?", str(version or "").strip())
    if not match:
        return None
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch or 0))


def _guard_tfvars_iam_token(terraform_dir: Path, tfvars: dict[str, Any]) -> None:
    """Reject an ``iam_token`` pinned in tfvars.

    Terraform gives ``terraform.tfvars`` precedence over ``TF_VAR_*``, so a token
    left in that file (the example file used to ship a placeholder) shadows the
    fresh token ``_terraform_env`` mints — apply then fails with Unauthenticated,
    or succeeds until the pasted token expires an hour later.
    """
    if "iam_token" not in tfvars:
        return
    files = ", ".join(
        str(path.name)
        for path in [
            terraform_dir / "terraform.tfvars",
            *sorted(terraform_dir.glob("*.auto.tfvars")),
        ]
        if path.exists()
    )
    raise typer.BadParameter(
        f"Remove the `iam_token` line from {files or 'terraform.tfvars'} in {terraform_dir}: "
        "Terraform prefers tfvars over the fresh token npa mints for every run, so a "
        "pinned token (or the example's <nebius-iam-token> placeholder) breaks apply "
        "with Unauthenticated. npa supplies iam_token automatically."
    )


def _apply_capacity_block_group_tfvars(
    tfvars: dict[str, Any], capacity_block_group: str
) -> None:
    value = capacity_block_group.strip()
    if value:
        tfvars["capacity_block_group"] = value


def _parse_tfvar_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value


def _guard_unmanaged_duplicate(
    nebius_bin: str,
    terraform_bin: str,
    terraform_dir: Path,
    tfvars: dict[str, Any],
    env: dict[str, str],
) -> None:
    cluster_name = str(tfvars.get("cluster_name") or "npa-cluster")
    project_id = str(
        tfvars.get("parent_id") or os.environ.get("TF_VAR_parent_id") or ""
    )
    if not project_id:
        typer.echo(
            "Skipping duplicate cluster preflight: parent_id is not set in tfvars or env.",
            err=True,
        )
        return
    result = _run_capture(
        [
            nebius_bin,
            "mk8s",
            "cluster",
            "list",
            "--parent-id",
            project_id,
            "--format",
            "json",
        ],
        env=env,
    )
    payload = json.loads(result.stdout or "{}")
    matches = [
        item
        for item in payload.get("items", [])
        if item.get("metadata", {}).get("name") == cluster_name
    ]
    if not matches:
        return
    managed_ids = _terraform_state_cluster_ids(terraform_bin, terraform_dir, env)
    unmanaged = [
        item.get("metadata", {}).get("id")
        for item in matches
        if item.get("metadata", {}).get("id") not in managed_ids
    ]
    if unmanaged:
        ids = ", ".join(str(value) for value in unmanaged if value)
        raise typer.BadParameter(
            f"Cluster {cluster_name} already exists outside this Terraform state: {ids}. "
            f"Adopt it with `npa cluster kubeconfig --cluster-name {cluster_name}` (writes "
            "its kubeconfig and cluster state), pick another `cluster_name`, or delete it "
            f"with `npa cluster destroy --name {cluster_name} --project-id "
            f"{project_id} --force`."
        )


def _preflight_filestore_quota(
    nebius_bin: str, tfvars: dict[str, Any], env: dict[str, str]
) -> None:
    # The Terraform default is enable_filestore = false (deploy/cluster/variables.tf),
    # so the default FTUE / PAIDF cluster needs no Shared Filesystem SSD quota. Only
    # check the quota when the shared filesystem is explicitly opted into and is being
    # created (not attached via existing_filestore).
    enable_filestore = _tfvar_bool(tfvars, env, "enable_filestore", False)
    existing_filestore = str(
        _tfvar_value(tfvars, env, "existing_filestore", "") or ""
    ).strip()
    if not enable_filestore or existing_filestore:
        return
    tenant_id = str(_tfvar_value(tfvars, env, "tenant_id", "") or "").strip()
    region = str(_tfvar_value(tfvars, env, "region", "") or "").strip()
    if not tenant_id or not region:
        typer.echo(
            "Skipping shared filesystem quota preflight: tenant_id or region is not set in tfvars or env.",
            err=True,
        )
        return
    size_gib = int(
        _tfvar_value(
            tfvars, env, "filestore_disk_size_gibibytes", _DEFAULT_FILESTORE_SIZE_GIB
        )
    )
    requested_bytes = size_gib * _GIB
    quota = _quota_allowance(
        nebius_bin,
        parent_id=tenant_id,
        region=region,
        name="compute.filesystem.size.network-ssd",
        env=env,
    )
    limit = _quota_limit(quota)
    usage = _quota_usage(quota)
    available = limit - usage
    if available < requested_bytes:
        raise typer.BadParameter(
            "Shared filesystem quota is insufficient for Terraform creation: "
            f"compute.filesystem.size.network-ssd available {available} bytes, "
            f"requested {requested_bytes} bytes in {region}. "
            "Provide existing_filestore or raise Shared Filesystem SSD quota before running apply."
        )


def _apply_node_count_override(tfvars: dict[str, Any], key: str, value: int) -> None:
    """Set ``tfvars[key]`` from a ``--gpu-nodes``/``--cpu-nodes`` flag (-1 = keep)."""
    if value is not None and value >= 0:
        tfvars[key] = int(value)


def _apply_context_cluster_name(
    tfvars: dict[str, Any], context_name: str, *, inherited_name: str = ""
) -> str:
    """Make an explicit context the actual Terraform cluster resource name."""

    context = str(context_name or "").strip()
    resolved = str(inherited_name or "").strip()
    if context and resolved and context != resolved:
        raise typer.BadParameter(
            f"--context {context!r} conflicts with the resolved cluster name {resolved!r}"
        )
    if context:
        tfvars["cluster_name"] = context
    return context


def _apply_inherited_plan_tfvars(tfvars: dict[str, Any], plan: Any) -> Any:
    """Make a previously resolved plan authoritative over env and checked tfvars."""

    topology = plan.topology
    tfvars.update(
        {
            "cluster_name": topology.cluster_name,
            "parent_id": plan.project_id,
            "tenant_id": plan.tenant_id,
            "region": plan.region,
            "gpu_nodes_count": topology.gpu_nodes,
            "cpu_nodes_count": topology.cpu_nodes,
            "cpu_nodes_platform": topology.cpu_platform,
            "cpu_nodes_preset": topology.cpu_preset,
            "gpu_nodes_platform": topology.gpu_platform,
            "gpu_nodes_preset": topology.gpu_preset,
            "gpu_nodes_preemptible": topology.gpu_preemptible,
        }
    )
    return topology


def _node_count_var_args(tfvars: dict[str, Any], key: str, value: int) -> list[str]:
    """Return ``-var key=N`` when the flag was given, so it beats a tfvars line."""
    if value is not None and value >= 0:
        return ["-var", f"{key}={int(value)}"]
    return []


def _apply_string_override(tfvars: dict[str, Any], key: str, value: str) -> None:
    """Set a Terraform string variable from a non-empty CLI flag."""

    cleaned = str(value or "").strip()
    if cleaned:
        tfvars[key] = cleaned


def _string_var_args(key: str, value: str) -> list[str]:
    """Return an explicit Terraform string override for a non-empty flag."""

    cleaned = str(value or "").strip()
    return ["-var", f"{key}={cleaned}"] if cleaned else []


def _preflight_instance_count_quota(
    tfvars: dict[str, Any], env: dict[str, str]
) -> None:
    """Compatibility diagnostic using the canonical resolved topology.

    Direct ``cluster up`` uses :func:`_preflight_whole_path_capacity`; this
    retained helper shares its default resolver so callers cannot accidentally
    model a no-tfvars cluster as zero nodes.
    """
    from npa.provisioning_preflight import resolve_topology

    topology = resolve_topology(
        gpu_nodes=int(_tfvar_value(tfvars, env, "gpu_nodes_count", -1) or 0),
        cpu_nodes=int(_tfvar_value(tfvars, env, "cpu_nodes_count", -1) or 0),
    )
    gpu_nodes = topology.new_gpu_nodes
    cpu_nodes = topology.new_cpu_nodes
    required = topology.required_instances
    if required <= 0:
        return
    tenant_id = str(_tfvar_value(tfvars, env, "tenant_id", "") or "").strip()
    region = str(_tfvar_value(tfvars, env, "region", "") or "").strip()
    if not tenant_id or not region:
        return
    from npa.clients.nebius import get_compute_instance_quota

    usage, limit = get_compute_instance_quota(tenant_id, region)
    if usage is None or limit is None:
        return
    free = max(0, limit - usage)
    if required <= free:
        return
    raise typer.BadParameter(
        f"Cluster needs {required} compute instance(s) ({gpu_nodes} GPU + {cpu_nodes} CPU "
        f"node(s)), but the tenant compute.instance.count quota in {region} has only {free} "
        f"free (limit {limit}, in use {usage}). Free instances (e.g. `npa agent destroy` an "
        f"agent, or delete idle VMs), reduce --gpu-nodes/--cpu-nodes, or ask a tenant admin to "
        f"raise compute.instance.count. At compute.instance.count={limit}, an agent VM and this "
        f"cluster cannot both fit — pick agent XOR cluster, or fewer GPU nodes."
    )


def _preflight_whole_path_capacity(
    tfvars: dict[str, Any],
    env: dict[str, str],
    *,
    context: str,
    project_alias: str = "",
) -> None:
    """Gate direct Terraform apply with the same immutable cumulative model."""

    from npa.provisioning_preflight import (
        build_whole_path_plan,
        current_resolved_plan,
        discover_existing_capacity,
        resolve_topology,
    )

    inherited_plan = current_resolved_plan()
    if inherited_plan is not None:
        operation = current_operation()
        if operation is not None:
            operation.record_preflight_plan(inherited_plan.to_dict())
        try:
            inherited_plan.assert_mutation_ready()
        except Exception as exc:
            raise typer.BadParameter(str(exc)) from exc
        topology = inherited_plan.topology
        typer.echo(
            "Whole-path preflight ready: "
            f"cpu={topology.cpu_nodes}x{topology.cpu_platform}/{topology.cpu_preset}, "
            f"gpu={topology.gpu_nodes}x{topology.gpu_platform}/{topology.gpu_preset}, "
            f"preemptible={str(topology.gpu_preemptible).lower()}, "
            f"new instances={topology.required_instances}, disks={topology.required_disks}, "
            f"network-ssd={topology.required_network_ssd_bytes} bytes."
        )
        return

    gpu_nodes = int(_tfvar_value(tfvars, env, "gpu_nodes_count", 1) or 0)
    cpu_nodes = int(_tfvar_value(tfvars, env, "cpu_nodes_count", 1) or 0)
    try:
        cached = load_cluster_state(context)
    except Exception:  # noqa: BLE001 - unreadable state must not imply ownership
        cached = None
    existing = bool(
        cached
        and cached.kubeconfig_path
        and Path(cached.kubeconfig_path).expanduser().is_file()
    )
    requested = resolve_topology(
        cluster_name=str(_tfvar_value(tfvars, env, "cluster_name", context) or context),
        cpu_nodes=cpu_nodes,
        cpu_platform=str(
            _tfvar_value(tfvars, env, "cpu_nodes_platform", "cpu-d3") or ""
        ),
        cpu_preset=str(
            _tfvar_value(tfvars, env, "cpu_nodes_preset", "8vcpu-32gb") or ""
        ),
        gpu_nodes=gpu_nodes,
        gpu_platform=str(
            _tfvar_value(tfvars, env, "gpu_nodes_platform", "gpu-rtx6000") or ""
        ),
        gpu_preset=str(
            _tfvar_value(tfvars, env, "gpu_nodes_preset", "1gpu-24vcpu-218gb") or ""
        ),
        preemptible=_tfvar_bool(tfvars, env, "gpu_nodes_preemptible", False),
        cpu_disk_gib=int(_tfvar_value(tfvars, env, "cpu_disk_size", 128) or 128),
        gpu_disk_gib=int(_tfvar_value(tfvars, env, "gpu_disk_size", 1023) or 1023),
        public_node_ips=False,
    )
    checks = []
    if existing:
        existing_cpu_nodes = requested.cpu_nodes
        existing_gpu_nodes = requested.gpu_nodes
    else:
        discovered = discover_existing_capacity(
            project_id=str(_tfvar_value(tfvars, env, "parent_id", "") or ""),
            cluster_name=requested.cluster_name,
            cpu_platform=requested.cpu_platform,
            cpu_preset=requested.cpu_preset,
            gpu_platform=requested.gpu_platform,
            gpu_preset=requested.gpu_preset,
        )
        existing_cpu_nodes = min(requested.cpu_nodes, discovered.cpu_nodes)
        existing_gpu_nodes = min(requested.gpu_nodes, discovered.gpu_nodes)
        checks.append(discovered.check)
    topology = resolve_topology(
        cluster_name=requested.cluster_name,
        cpu_nodes=requested.cpu_nodes,
        existing_cpu_nodes=existing_cpu_nodes,
        cpu_platform=requested.cpu_platform,
        cpu_preset=requested.cpu_preset,
        gpu_nodes=requested.gpu_nodes,
        existing_gpu_nodes=existing_gpu_nodes,
        gpu_platform=requested.gpu_platform,
        gpu_preset=requested.gpu_preset,
        preemptible=requested.gpu_preemptible,
        cpu_disk_gib=requested.cpu_disk_gib,
        gpu_disk_gib=requested.gpu_disk_gib,
        public_node_ips=requested.public_node_ips,
    )
    plan = build_whole_path_plan(
        project_alias=project_alias,
        project_id=str(_tfvar_value(tfvars, env, "parent_id", "") or ""),
        tenant_id=str(_tfvar_value(tfvars, env, "tenant_id", "") or ""),
        region=str(_tfvar_value(tfvars, env, "region", "") or ""),
        topology=topology,
        checks=checks,
        mutation=True,
    )
    operation = current_operation()
    if operation is not None:
        operation.record_preflight_plan(plan.to_dict())
    try:
        plan.assert_mutation_ready()
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        "Whole-path preflight ready: "
        f"cpu={topology.cpu_nodes}x{topology.cpu_platform}/{topology.cpu_preset}, "
        f"gpu={topology.gpu_nodes}x{topology.gpu_platform}/{topology.gpu_preset}, "
        f"preemptible={str(topology.gpu_preemptible).lower()}, "
        f"new instances={topology.required_instances}, disks={topology.required_disks}, "
        f"network-ssd={topology.required_network_ssd_bytes} bytes."
    )


def _preflight_gpu_capacity(
    nebius_bin: str, tfvars: dict[str, Any], env: dict[str, str]
) -> None:
    """Fail before apply when the tenant's GPU quota cannot cover the node group."""
    from npa.cli.cluster.capacity import gpu_capacity_error

    gpu_nodes = int(_tfvar_value(tfvars, env, "gpu_nodes_count", 0) or 0)
    if gpu_nodes <= 0:
        return
    # MIG's STRICT capacity-block-backed pool consumes the named reservation,
    # not ordinary on-demand GPU quota. Reservation ownership/region/platform
    # and remaining capacity are validated by the shared mk8s preflight above.
    if (
        _tfvar_bool(tfvars, env, "mig_enabled", False)
        and str(_tfvar_value(tfvars, env, "capacity_block_group", "") or "").strip()
    ):
        return
    platform = str(
        _tfvar_value(tfvars, env, "gpu_nodes_platform", "gpu-rtx6000") or ""
    ).strip()
    preset = str(_tfvar_value(tfvars, env, "gpu_nodes_preset", "") or "").strip()
    tenant_id = str(_tfvar_value(tfvars, env, "tenant_id", "") or "").strip()
    region = str(_tfvar_value(tfvars, env, "region", "") or "").strip()
    if not tenant_id or not region:
        typer.echo(
            "Skipping GPU quota preflight: tenant_id or region is not set in tfvars or env.",
            err=True,
        )
        return
    preemptible = _tfvar_bool(tfvars, env, "gpu_nodes_preemptible", False)
    required = gpu_nodes * _gpus_per_node(preset)
    capture = lambda args: _run_capture(args, env=env, check=False)  # noqa: E731 - passed through
    message = gpu_capacity_error(
        capture,
        nebius_bin=nebius_bin,
        tenant_id=tenant_id,
        region=region,
        platform=platform,
        preset=preset,
        required_gpus=required,
        preemptible=preemptible,
    )
    if message:
        raise typer.BadParameter(message)
    if not preemptible and not _gpu_quota_was_readable(
        capture,
        nebius_bin=nebius_bin,
        tenant_id=tenant_id,
        region=region,
        platform=platform,
    ):
        # Skipping silently is what let an unreadable quota become a node group
        # that retries for hours with no explanation.
        typer.echo(
            f"Warning: could not read the {platform} GPU quota for {region} "
            f"(tenant {tenant_id}), so this apply is not quota-checked. If the node "
            "group never leaves PROVISIONING, the platform is refusing it — check "
            "`nebius quotas quota-allowance get-by-name --parent-id <tenant> "
            f"--region {region} --name compute.instance.gpu.<model>`.",
            err=True,
        )


def _gpu_quota_was_readable(
    capture: Any, *, nebius_bin: str, tenant_id: str, region: str, platform: str
) -> bool:
    from npa.cli.cluster.capacity import gpu_quota_headroom, gpu_quota_name

    quota_name = gpu_quota_name(platform)
    if not quota_name:
        return True
    return (
        gpu_quota_headroom(
            capture,
            nebius_bin=nebius_bin,
            tenant_id=tenant_id,
            region=region,
            quota_name=quota_name,
        )
        is not None
    )


#: Node-group status text that means the platform has *refused* the group rather
#: than being slow: waiting these out cannot succeed. Terraform keeps printing
#: `Still creating...` and retries until its own timeout (two hours by default),
#: which is what turned an unavailable GPU into an open-ended hang.
_TERMINAL_NODE_GROUP_MARKERS = (
    "quotafailure",
    "quota exceeded",
    "quota_exceeded",
    "exceeded quota",
    "out of capacity",
    "insufficient capacity",
    "no capacity",
    "capacity not available",
)


#: Status keys whose contents are diagnostics worth printing. Once one matches,
#: every string underneath it counts: Nebius reports the real reason as
#: ``status.events[].last_occurrence``, whose own key says nothing.
_DIAGNOSTIC_KEY_TOKENS = (
    "error",
    "failure",
    "message",
    "condition",
    "reason",
    "state",
    "event",
)


def _status_texts(value: Any, *, key: str = "", inherited: bool = False) -> list[str]:
    """Return diagnostic strings from a node-group status, nested ones included.

    Scanning only top-level keys missed every QuotaFailure — the message the whole
    watcher exists to surface — because it arrives inside ``events[]``.
    """
    keep = inherited or any(token in key.lower() for token in _DIAGNOSTIC_KEY_TOKENS)
    if isinstance(value, dict):
        texts: list[str] = []
        for child_key, child in value.items():
            texts.extend(_status_texts(child, key=str(child_key), inherited=keep))
        return texts
    if isinstance(value, list):
        texts = []
        for child in value:
            texts.extend(_status_texts(child, key=key, inherited=keep))
        return texts
    # Only strings: counts and timestamps under an events entry are not reasons.
    if keep and isinstance(value, str) and value.strip():
        return [value]
    return []


def terminal_node_group_failure(status: dict[str, Any]) -> str:
    """Return the refusal text when *status* shows a failure retrying cannot fix.

    Several fields can match (an event's ``type`` as well as its
    ``last_occurrence``); report the most descriptive one, since it is what the
    operator has to act on.
    """
    matches = [
        text
        for text in _status_texts(status or {})
        if any(marker in text.lower() for marker in _TERMINAL_NODE_GROUP_MARKERS)
    ]
    return max(matches, key=len) if matches else ""


class _NodeGroupWatcher:
    """Watch Managed-Kubernetes node groups while ``terraform apply`` runs.

    Terraform's own output is just `Still creating...`, so a node group that
    Nebius refuses (QuotaFailure, no capacity) looks identical to one that is
    provisioning normally — the operator waits, then interrupts, and is left with
    a half-created cluster and no reason. This prints each group's state as it
    changes and, when the platform reports a refusal that retrying cannot fix,
    cancels the apply (``on_fatal``) instead of waiting out the timeout.

    Polling is best-effort: any error inside the thread stops the watcher rather
    than affecting the apply.
    """

    def __init__(
        self,
        nebius_bin: str,
        tfvars: dict[str, Any],
        env: dict[str, str],
        *,
        interval: float = 45.0,
        on_fatal: Callable[[str], None] | None = None,
    ):
        self._nebius_bin = nebius_bin
        self._project_id = str(_tfvar_value(tfvars, env, "parent_id", "") or "").strip()
        self._cluster_name = str(tfvars.get("cluster_name") or "npa-cluster")
        self._env = env
        self._interval = interval
        self._on_fatal = on_fatal
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen: dict[str, str] = {}
        self.fatal_reason = ""

    def start(self) -> None:
        if not self._project_id:
            return
        self._thread = threading.Thread(
            target=self._run, name="npa-node-group-watch", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._poll()
            except Exception:  # noqa: BLE001 - observability must never break apply
                return

    def _poll(self) -> None:
        cluster_id = self._cluster_id()
        if not cluster_id:
            return
        result = _run_capture(
            [
                self._nebius_bin,
                "mk8s",
                "node-group",
                "list",
                "--parent-id",
                cluster_id,
                "--format",
                "json",
            ],
            env=self._env,
            check=False,
        )
        if result.returncode != 0:
            return
        for item in (json.loads(result.stdout or "{}") or {}).get("items", []):
            name = str((item.get("metadata") or {}).get("name", "") or "")
            status = item.get("status") or {}
            line = _format_node_group_status(status)
            if name and self._seen.get(name) != line:
                self._seen[name] = line
                typer.echo(f"  node group {name}: {line}", err=True)
            refusal = terminal_node_group_failure(status)
            if refusal and not self.fatal_reason:
                self.fatal_reason = f"node group {name}: {refusal}"
                typer.echo(
                    f"  Nebius refused node group {name}: {refusal}. Retrying cannot fix "
                    "this, so the apply is being cancelled instead of waiting for the "
                    "Terraform timeout.",
                    err=True,
                )
                if self._on_fatal is not None:
                    self._on_fatal(self.fatal_reason)
                self._stop.set()
                return

    def _cluster_id(self) -> str:
        result = _run_capture(
            [
                self._nebius_bin,
                "mk8s",
                "cluster",
                "list",
                "--parent-id",
                self._project_id,
                "--format",
                "json",
            ],
            env=self._env,
            check=False,
        )
        if result.returncode != 0:
            return ""
        for item in (json.loads(result.stdout or "{}") or {}).get("items", []):
            metadata = item.get("metadata") or {}
            if str(metadata.get("name", "")) == self._cluster_name:
                return str(metadata.get("id", "") or "")
        return ""


def _format_node_group_status(status: dict[str, Any]) -> str:
    """Summarize node-group status, keeping any failure text Nebius reports."""
    state = str(status.get("state", "") or "UNKNOWN")
    counts = f"{status.get('ready_node_count', '?')}/{status.get('target_node_count', '?')} ready"
    details = _node_group_status_details(status)
    return ", ".join([f"{state} ({counts})", *dict.fromkeys(details)])


def _node_group_status_details(status: dict[str, Any]) -> list[str]:
    """Render diagnostics while reconciling confirmed absence as success.

    Managed Kubernetes can race its own instance reconciliation: deletion sees
    an instance that has already disappeared, records an event whose *type* says
    ``ComputeInstanceDeletionFailed``, then completes normally. Preserve every
    genuine failure, but translate that exact type + NotFound combination into
    idempotent progress rather than alarming the operator.
    """

    details = _status_texts(
        {key: value for key, value in status.items() if key not in {"state", "events"}}
    )
    events = status.get("events")
    if not isinstance(events, list):
        details.extend(_status_texts(events, key="events"))
        return details
    for event in events:
        if isinstance(event, dict) and _instance_deletion_already_absent(event):
            details.append(
                "compute instance already absent; node-group deletion is reconciling "
                "idempotently"
            )
            continue
        details.extend(_status_texts(event, key="event"))
    return details


def _instance_deletion_already_absent(event: dict[str, Any]) -> bool:
    event_type = re.sub(r"[^a-z]", "", str(event.get("type") or "").lower())
    if event_type != "computeinstancedeletionfailed":
        return False
    detail = " ".join(_status_texts(event, key="event")).lower()
    return any(
        marker in detail for marker in ("notfound", "not found", "already absent")
    )


def _echo_apply_recovery(
    tf_dir: Path, tfvars: dict[str, Any], interrupted: bool
) -> None:
    """Say what may exist in the cloud after a failed or interrupted apply.

    An interrupted `cluster up` leaves a real cluster running with no local
    kubeconfig, which reads as "nothing happened" until the bill arrives.
    """
    cluster_name = str(tfvars.get("cluster_name") or "npa-cluster")
    reason = "interrupted" if interrupted else "failed"
    typer.echo("", err=True)
    typer.echo(
        f"terraform apply was {reason}. Cluster {cluster_name!r} may exist (partially) "
        "in the project, and no kubeconfig was written yet. Either:",
        err=True,
    )
    typer.echo(
        f"  - resume: re-run `npa cluster up --terraform-dir {tf_dir}` (idempotent; it "
        "finishes what was created and writes the kubeconfig), or",
        err=True,
    )
    typer.echo(
        f"  - tear it down: `npa cluster down --terraform-dir {tf_dir} --force`, or",
        err=True,
    )
    typer.echo(
        f"  - adopt what exists: `npa cluster kubeconfig --cluster-name {cluster_name}` "
        "writes the kubeconfig and cluster state for a cluster that is already running.",
        err=True,
    )
    typer.echo(
        "  - check what exists now: `nebius mk8s cluster list --parent-id <project-id>`.",
        err=True,
    )


def _tfvar_value(
    tfvars: dict[str, Any], env: dict[str, str], key: str, default: Any
) -> Any:
    if key in tfvars:
        return tfvars[key]
    return env.get(f"TF_VAR_{key}", default)


def _tfvar_bool(
    tfvars: dict[str, Any], env: dict[str, str], key: str, default: bool
) -> bool:
    """Read a boolean tfvar, treating ``TF_VAR_x`` strings the way Terraform does.

    ``TF_VAR_*`` values arrive as strings, and ``bool("false")`` is ``True``, so a
    documented ``TF_VAR_enable_filestore=false`` opt-out would otherwise be read as
    enabled.
    """
    value = _tfvar_value(tfvars, env, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
        return default
    return bool(value)


def _resolve_gpu_driver_selection(
    tfvars: dict[str, Any], env: dict[str, str]
) -> GpuDriverSelection:
    """Resolve direct-cluster Terraform/CLI values through the shared contract."""

    try:
        return resolve_gpu_driver_strategy(
            gpu_nodes=int(_tfvar_value(tfvars, env, "gpu_nodes_count", 1) or 0),
            platform=str(
                _tfvar_value(tfvars, env, "gpu_nodes_platform", "gpu-rtx6000") or ""
            ),
            preset=str(
                _tfvar_value(tfvars, env, "gpu_nodes_preset", "1gpu-24vcpu-218gb") or ""
            ),
            mode=str(_tfvar_value(tfvars, env, "gpu_driver_mode", "auto") or "auto"),
            managed_driver_preset=str(
                _tfvar_value(
                    tfvars,
                    env,
                    "managed_driver_preset",
                    DEFAULT_MANAGED_DRIVER_PRESET,
                )
                or DEFAULT_MANAGED_DRIVER_PRESET
            ),
            enable_gpu_cluster=_tfvar_bool(tfvars, env, "enable_gpu_cluster", False),
            allow_unsafe_nvswitch_operator=_tfvar_bool(
                tfvars, env, "allow_unsafe_nvswitch_operator", False
            ),
        )
    except GpuDriverStrategyError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _shared_filesystem_requested(tfvars: dict[str, Any], env: dict[str, str]) -> bool:
    """Whether the config asks for a shared filesystem (created or attached).

    ``deploy/cluster`` turns ``existing_filestore`` into ``enable_filestore`` for the
    vendored module, so either one means the filesystem CSI and its
    ``csi-mounted-fs-path-sc`` default StorageClass are installed.
    """
    if _tfvar_bool(tfvars, env, "enable_filestore", False):
        return True
    return bool(str(_tfvar_value(tfvars, env, "existing_filestore", "") or "").strip())


def _quota_allowance(
    nebius_bin: str,
    *,
    parent_id: str,
    region: str,
    name: str,
    env: dict[str, str],
) -> dict[str, Any]:
    result = _run_capture(
        [
            nebius_bin,
            "quotas",
            "quota-allowance",
            "get-by-name",
            "--parent-id",
            parent_id,
            "--region",
            region,
            "--name",
            name,
            "--format",
            "json",
        ],
        env=env,
    )
    return json.loads(result.stdout or "{}")


def _quota_limit(quota: dict[str, Any]) -> int:
    raw_limit = quota.get("spec", {}).get("limit")
    return int(raw_limit or 0)


def _quota_usage(quota: dict[str, Any]) -> int:
    raw_usage = quota.get("status", {}).get("usage")
    return int(raw_usage or 0)


def _terraform_state_cluster_ids(
    terraform_bin: str, terraform_dir: Path, env: dict[str, str]
) -> set[str]:
    result = _run_capture(
        [terraform_bin, "state", "pull"],
        cwd=terraform_dir,
        env=env,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return set()
    try:
        state = json.loads(result.stdout)
    except json.JSONDecodeError:
        return set()
    ids: set[str] = set()
    output_cluster_id = (
        state.get("outputs", {}).get("kube_cluster", {}).get("value", {}).get("id")
    )
    if output_cluster_id:
        ids.add(str(output_cluster_id))

    def walk(module: dict[str, Any]) -> None:
        for resource in module.get("resources", []):
            if resource.get("type") != "nebius_mk8s_v1_cluster":
                continue
            for instance in resource.get("instances", []):
                cluster_id = instance.get("attributes", {}).get("id")
                if cluster_id:
                    ids.add(str(cluster_id))
        for child in module.get("child_modules", []):
            walk(child)

    walk(state.get("values", {}).get("root_module", {}))
    for resource in state.get("resources", []):
        if resource.get("type") != "nebius_mk8s_v1_cluster":
            continue
        for instance in resource.get("instances", []):
            cluster_id = instance.get("attributes", {}).get("id")
            if cluster_id:
                ids.add(str(cluster_id))
    return ids


def _verify_residual_terraform_ownership(
    terraform_bin: str,
    terraform_dir: Path,
    env: dict[str, str],
    *,
    project_id: str,
    cluster_id: str,
) -> list[str]:
    """Validate residual-only state for an out-of-band/partial cluster delete."""

    result = _run_capture(
        [terraform_bin, "state", "pull"],
        cwd=terraform_dir,
        env=env,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise typer.BadParameter(
            "Legacy recovery requires readable retained Terraform state; nothing "
            "was deleted. Restore the state and retry."
        )
    try:
        state = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            "Legacy recovery Terraform state is malformed; nothing was deleted."
        ) from exc
    resources = state.get("resources") if isinstance(state, dict) else None
    if (
        not isinstance(resources, list)
        or not str(state.get("lineage") or "").strip()
        or not isinstance(state.get("serial"), int)
    ):
        raise typer.BadParameter(
            "Legacy recovery Terraform state lacks valid lineage/resource ownership; "
            "nothing was deleted."
        )
    project_scoped_types = {
        "nebius_compute_v1_filesystem",
        "nebius_compute_v1_gpu_cluster",
        "nebius_applications_v1alpha1_k8s_release",
        "nebius_iam_v1_service_account",
        "nebius_storage_v1_bucket",
        "nebius_vpc_v1_network",
        "nebius_vpc_v1_subnet",
    }
    cluster_scoped_types = {
        "nebius_mk8s_v1_node_group",
    }
    indirect_types = {
        "nebius_iam_v1_group_membership",
        "nebius_iam_v2_access_key",
    }
    managed_ids: dict[str, set[str]] = {}
    for resource in resources:
        if (
            not isinstance(resource, dict)
            or resource.get("mode", "managed") != "managed"
        ):
            continue
        resource_type = str(resource.get("type") or "")
        instances = resource.get("instances")
        if not resource_type or not isinstance(instances, list):
            raise typer.BadParameter(
                "Legacy recovery Terraform state contains malformed managed "
                "resource inventory; nothing was deleted."
            )
        for instance in instances:
            attributes = (
                instance.get("attributes") if isinstance(instance, dict) else None
            )
            if not isinstance(attributes, dict):
                raise typer.BadParameter(
                    "Legacy recovery Terraform state contains malformed ownership "
                    "attributes; nothing was deleted."
                )
            resource_id = str(attributes.get("id") or "").strip()
            if resource_type.startswith("nebius_") and not resource_id:
                raise typer.BadParameter(
                    "Legacy recovery Terraform state contains a Nebius resource "
                    "without an exact provider ID; nothing was deleted."
                )
            if resource_id:
                managed_ids.setdefault(resource_type, set()).add(resource_id)

    service_account_ids = managed_ids.get("nebius_iam_v1_service_account", set())
    residual_types: list[str] = []
    for resource in resources:
        if (
            not isinstance(resource, dict)
            or resource.get("mode", "managed") != "managed"
        ):
            continue
        resource_type = str(resource.get("type") or "")
        if not resource_type:
            raise typer.BadParameter(
                "Legacy recovery Terraform state contains a malformed managed "
                "resource; nothing was deleted."
            )
        instances = resource.get("instances")
        if not isinstance(instances, list):
            raise typer.BadParameter(
                "Legacy recovery Terraform state contains malformed managed "
                "instances; nothing was deleted."
            )
        if not instances:
            # Terraform retains empty blocks after count/for_each transitions.
            # They prove no live provider resource and carry no ownership to
            # validate, including an already-removed cluster block.
            continue
        if resource_type == "nebius_mk8s_v1_cluster":
            raise typer.BadParameter(
                "Legacy recovery found an unidentifiable cluster resource; nothing "
                "was deleted."
            )
        for instance in instances:
            attributes = (
                instance.get("attributes") if isinstance(instance, dict) else None
            )
            if not isinstance(attributes, dict):
                raise typer.BadParameter(
                    "Legacy recovery Terraform state contains malformed ownership "
                    "attributes; nothing was deleted."
                )
            candidate_parents = {
                str(attributes.get(key) or "").strip()
                for key in ("parent_id", "project_id", "parentId", "projectId")
                if str(attributes.get(key) or "").strip()
            }
            if resource_type in project_scoped_types and candidate_parents != {
                project_id
            }:
                raise typer.BadParameter(
                    "Legacy recovery Terraform residuals do not match the requested "
                    "project; nothing was deleted."
                )
            if resource_type in cluster_scoped_types and candidate_parents != {
                cluster_id
            }:
                raise typer.BadParameter(
                    "Legacy recovery Terraform residuals do not match the exact "
                    "persisted cluster; nothing was deleted."
                )
            if resource_type == "nebius_applications_v1alpha1_k8s_release":
                linked_cluster_id = str(
                    attributes.get("cluster_id") or attributes.get("clusterId") or ""
                ).strip()
                if linked_cluster_id != cluster_id:
                    raise typer.BadParameter(
                        "Legacy recovery application-release ownership is not "
                        "linked to the exact persisted cluster; nothing was deleted."
                    )
            if resource_type == "nebius_iam_v2_access_key":
                if len(candidate_parents) != 1 or not candidate_parents.issubset(
                    service_account_ids
                ):
                    raise typer.BadParameter(
                        "Legacy recovery access-key ownership is not linked to an "
                        "exact Terraform-owned service account; nothing was deleted."
                    )
            if resource_type == "nebius_iam_v1_group_membership":
                member = attributes.get("member")
                member_id = str(
                    attributes.get("member_id")
                    or attributes.get("memberId")
                    or (member.get("id") if isinstance(member, dict) else "")
                ).strip()
                if len(candidate_parents) != 1 or member_id not in service_account_ids:
                    raise typer.BadParameter(
                        "Legacy recovery group-membership ownership is not linked to "
                        "an exact Terraform-owned service account; nothing was deleted."
                    )
            if (
                resource_type.startswith("nebius_")
                and resource_type not in project_scoped_types
                and resource_type not in cluster_scoped_types
                and resource_type not in indirect_types
            ):
                raise typer.BadParameter(
                    f"Legacy recovery does not recognize Nebius residual type "
                    f"{resource_type!r}; nothing was deleted."
                )
        residual_types.append(resource_type)
    if not residual_types:
        raise typer.BadParameter(
            "Legacy recovery state contains no Terraform-owned residual resources; "
            "nothing was deleted."
        )
    return sorted(set(residual_types))


def _terraform_outputs(
    terraform_bin: str, terraform_dir: Path, env: dict[str, str]
) -> dict[str, Any]:
    result = _run_capture(
        [terraform_bin, "output", "-json"], cwd=terraform_dir, env=env
    )
    return json.loads(result.stdout or "{}")


def _cluster_output(outputs: dict[str, Any]) -> dict[str, Any]:
    value = outputs.get("kube_cluster", {}).get("value")
    if not isinstance(value, dict):
        raise typer.BadParameter("Terraform output kube_cluster is missing")
    return value


def _write_kubeconfig(
    nebius_bin: str, cluster_id: str, kubeconfig_path: Path, context: str
) -> None:
    kubeconfig_path.parent.mkdir(parents=True, exist_ok=True)
    _run_stream(
        [
            nebius_bin,
            "mk8s",
            "cluster",
            "get-credentials",
            "--id",
            cluster_id,
            "--force",
            "--kubeconfig",
            str(kubeconfig_path),
            "--external",
            "--context-name",
            context,
        ],
        timeout=120,
    )


def _save_terraform_cluster_state(
    tfvars: dict[str, Any],
    cluster: dict[str, Any],
    context: str,
    kubeconfig_path: Path,
    *,
    env: dict[str, str] | None = None,
    last_seen_state: str = "RUNNING",
) -> None:
    raw_endpoints = cluster.get("endpoints")
    endpoints: dict[str, Any] = (
        dict(raw_endpoints) if isinstance(raw_endpoints, dict) else {}
    )
    state = ClusterState(
        name=context,
        cluster_id=str(cluster.get("id") or ""),
        project_id=str(_tfvar_value(tfvars, env or {}, "parent_id", "") or ""),
        region=str(_tfvar_value(tfvars, env or {}, "region", "") or ""),
        node_count=int(tfvars.get("cpu_nodes_count") or 0)
        + int(tfvars.get("gpu_nodes_count") or 0),
        node_platform=str(tfvars.get("gpu_nodes_platform") or ""),
        node_preset=str(tfvars.get("gpu_nodes_preset") or ""),
        k8s_version=str(tfvars.get("k8s_version") or ""),
        subnet_id=str(tfvars.get("subnet_id") or ""),
        created_at=utc_now_iso(),
        last_seen_state=last_seen_state,
        endpoint=str(endpoints.get("public_endpoint") or ""),
        kubeconfig_path=str(kubeconfig_path),
        provider_name=str(cluster.get("name") or ""),
    )
    save_cluster_state(
        state,
        metadata={
            "managed_by": "npa cluster terraform",
            "event": (
                "gpu_health_validated"
                if last_seen_state == "RUNNING"
                else "kubeconfig_written_validation_pending"
            ),
            "updated_at": utc_now_iso(),
            "teardown": "Run `npa cluster down --terraform-dir deploy/cluster --force` when finished.",
        },
    )


def _validate_cluster(
    kubectl_bin: str,
    kubeconfig_path: Path,
    tfvars: dict[str, Any],
    timeout_minutes: int,
    *,
    gpu_health_stabilization_seconds: int = DEFAULT_STABILIZATION_SECONDS,
    gpu_cuda_smoke: bool = True,
    gpu_cuda_smoke_image: str = DEFAULT_CUDA_SMOKE_IMAGE,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    resolved_env = dict(env or os.environ)
    driver = _resolve_gpu_driver_selection(tfvars, resolved_env)
    expected_gpu_nodes = int(
        _tfvar_value(tfvars, resolved_env, "gpu_nodes_count", 1) or 0
    )
    expected_cpu_nodes = int(
        _tfvar_value(tfvars, resolved_env, "cpu_nodes_count", 1) or 0
    )
    if expected_gpu_nodes > 0:
        try:
            gpu_report = validate_gpu_health(
                _run_capture,
                kubectl_bin=kubectl_bin,
                kubeconfig_path=kubeconfig_path,
                config=GpuHealthConfig(
                    expected_nodes=expected_cpu_nodes + expected_gpu_nodes,
                    expected_gpu_nodes=expected_gpu_nodes,
                    gpu_preset=str(
                        _tfvar_value(
                            tfvars,
                            resolved_env,
                            "gpu_nodes_preset",
                            "1gpu-24vcpu-218gb",
                        )
                        or ""
                    ),
                    gpu_platform=str(
                        _tfvar_value(
                            tfvars,
                            resolved_env,
                            "gpu_nodes_platform",
                            "gpu-rtx6000",
                        )
                        or ""
                    ),
                    driver_mode=driver.effective_mode,
                    nvswitch=driver.nvswitch,
                    stabilization_seconds=gpu_health_stabilization_seconds,
                    timeout_seconds=timeout_minutes * 60,
                    cuda_smoke=gpu_cuda_smoke,
                    cuda_smoke_image=gpu_cuda_smoke_image,
                ),
                evidence_path=kubeconfig_path.parent / "gpu-health.json",
                on_status=lambda message: typer.echo(message),
            )
        except (RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        final = gpu_report["final_snapshot"]
        result = {
            "ready_nodes": final["ready_nodes"],
            "gpu_nodes": len(final["gpu_nodes"]),
            "total_gpus": final["total_gpus"],
            "driver_mode": driver.effective_mode,
            "cuda_smokes": gpu_report["cuda_smokes"],
        }
    else:
        result = {}

    deadline = time.monotonic() + timeout_minutes * 60
    last_error = ""
    while time.monotonic() <= deadline:
        try:
            once = _validate_cluster_once(
                kubectl_bin,
                kubeconfig_path,
                tfvars,
                env=resolved_env,
                skip_gpu_probe=expected_gpu_nodes > 0,
            )
            return {
                **once,
                **result,
                "default_storage_class": once["default_storage_class"],
            }
        except typer.BadParameter as exc:
            last_error = str(exc)
            typer.echo(f"Validation pending: {last_error}")
            time.sleep(30)
    raise typer.BadParameter(
        f"Cluster validation did not pass within {timeout_minutes} minutes: {last_error}"
    )


def _validate_cluster_once(
    kubectl_bin: str,
    kubeconfig_path: Path,
    tfvars: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    skip_gpu_probe: bool = False,
) -> dict[str, Any]:
    resolved_env = dict(env or os.environ)
    kubectl_env = os.environ.copy()
    kubectl_env["KUBECONFIG"] = str(kubeconfig_path)
    expected_gpu_nodes = int(
        _tfvar_value(tfvars, resolved_env, "gpu_nodes_count", 1) or 0
    )
    expected_cpu_nodes = int(
        _tfvar_value(tfvars, resolved_env, "cpu_nodes_count", 1) or 0
    )
    if expected_gpu_nodes and not skip_gpu_probe:
        driver = _resolve_gpu_driver_selection(tfvars, resolved_env)
        try:
            snapshot = probe_gpu_health(
                _run_capture,
                kubectl_bin=kubectl_bin,
                kubeconfig_path=kubeconfig_path,
                config=GpuHealthConfig(
                    expected_nodes=expected_cpu_nodes + expected_gpu_nodes,
                    expected_gpu_nodes=expected_gpu_nodes,
                    gpu_preset=str(
                        _tfvar_value(
                            tfvars,
                            resolved_env,
                            "gpu_nodes_preset",
                            "1gpu-24vcpu-218gb",
                        )
                        or ""
                    ),
                    gpu_platform=str(
                        _tfvar_value(
                            tfvars,
                            resolved_env,
                            "gpu_nodes_platform",
                            "gpu-rtx6000",
                        )
                        or ""
                    ),
                    driver_mode=driver.effective_mode,
                    nvswitch=driver.nvswitch,
                    stabilization_seconds=0,
                    cuda_smoke=False,
                ),
            )
        except (RuntimeError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        if snapshot["errors"]:
            raise typer.BadParameter("; ".join(snapshot["errors"]))
        ready_nodes = snapshot["ready_nodes"]
        gpu_node_count = len(snapshot["gpu_nodes"])
        total_gpus = snapshot["total_gpus"]
    elif expected_gpu_nodes:
        ready_nodes = expected_cpu_nodes + expected_gpu_nodes
        gpu_node_count = expected_gpu_nodes
        total_gpus = expected_gpu_nodes * _gpus_per_node(
            str(
                _tfvar_value(
                    tfvars,
                    resolved_env,
                    "gpu_nodes_preset",
                    "1gpu-24vcpu-218gb",
                )
                or ""
            )
        )
    else:
        nodes = json.loads(
            _run_capture(
                [kubectl_bin, "get", "nodes", "-o", "json"], env=kubectl_env
            ).stdout
        ).get("items", [])
        ready_nodes = sum(
            1
            for node in nodes
            if any(
                condition.get("type") == "Ready" and condition.get("status") == "True"
                for condition in (node.get("status") or {}).get("conditions", [])
            )
        )
        if len(nodes) < expected_cpu_nodes or ready_nodes != len(nodes):
            raise typer.BadParameter(
                f"Expected {expected_cpu_nodes} Ready CPU nodes, found "
                f"{ready_nodes}/{len(nodes)} Ready"
            )
        gpu_node_count = 0
        total_gpus = 0

    storage_classes = json.loads(
        _run_capture(
            [kubectl_bin, "get", "storageclass", "-o", "json"], env=kubectl_env
        ).stdout
    )
    default_sc = ""
    for item in storage_classes.get("items", []):
        annotations = item.get("metadata", {}).get("annotations", {})
        if annotations.get("storageclass.kubernetes.io/is-default-class") == "true":
            default_sc = item.get("metadata", {}).get("name", "")
            break
    # The filesystem CSI is installed only when the shared filesystem is enabled.
    # Respect environment overrides and existing-filesystem attachment semantics,
    # then validate the exact expected default on both sides of that decision.
    filestore_enabled = _shared_filesystem_requested(tfvars, resolved_env)
    expected_default_sc = (
        "csi-mounted-fs-path-sc"
        if filestore_enabled
        else str(
            tfvars.get("previous_default_storage_class_name")
            or "compute-csi-default-sc"
        )
    )
    if default_sc != expected_default_sc:
        raise typer.BadParameter(
            f"Expected default StorageClass {expected_default_sc}, found {default_sc}"
        )
    return {
        "ready_nodes": ready_nodes,
        "gpu_nodes": gpu_node_count,
        "total_gpus": total_gpus,
        "default_storage_class": default_sc,
    }


def _gpus_per_node(preset: str) -> int:
    match = re.match(r"^(\d+)gpu-", preset)
    return int(match.group(1)) if match else 0


def _skypilot_context(
    kubeconfig_path: Path,
    context: str,
    *,
    sky_bin: str = "",
) -> tuple[str, dict[str, str], str]:
    executable = (
        sky_bin or os.environ.get("NPA_SKYPILOT_BIN") or str(_DEFAULT_SKYPILOT_BIN)
    )
    sky = _require_bin(executable)
    env = os.environ.copy()
    env["KUBECONFIG"] = str(kubeconfig_path)
    from npa.orchestration.skypilot.k8s_gpu_catalog import (
        exact_kubernetes_context_config,
    )

    config_override = exact_kubernetes_context_config(context)
    return sky, env, config_override


def _check_skypilot_kubernetes(
    kubeconfig_path: Path,
    context: str,
    *,
    sky_bin: str = "",
) -> tuple[str, dict[str, str], str]:
    """Enable and verify SkyPilot against the exact Kubernetes context."""

    sky, env, config_override = _skypilot_context(
        kubeconfig_path, context, sky_bin=sky_bin
    )
    # SkyPilot auto-starts a long-lived local API server and that daemon inherits
    # the CLI process's cwd.  Keep it on the durable cluster state directory: a
    # deleted Terraform/temp cwd later makes every rsync fail with getcwd(2).
    sky_cwd = kubeconfig_path.parent
    check_result = _run_stream(
        [
            sky,
            "check",
            "--config",
            config_override,
            "kubernetes",
        ],
        cwd=sky_cwd,
        env=env,
        timeout=300,
        capture_output=True,
    )
    plain_check = re.sub(
        r"\x1b\[[0-?]*[ -/]*[@-~]",
        "",
        "\n".join((check_result.stdout or "", check_result.stderr or "")),
    )
    if not re.search(r"\bKubernetes:\s+enabled\b", plain_check, flags=re.IGNORECASE):
        raise RuntimeError(
            "SkyPilot returned success without enabling the exact Kubernetes context"
        )
    typer.echo(f"SkyPilot Kubernetes credentials verified for context {context!r}.")
    return sky, env, config_override


def _run_skypilot_smoke(
    kubeconfig_path: Path,
    context: str,
    cluster_name: str,
    sky_gpus: str,
    *,
    sky_bin: str = "",
    credentials_checked: bool = False,
) -> None:
    if credentials_checked:
        sky, env, config_override = _skypilot_context(
            kubeconfig_path, context, sky_bin=sky_bin
        )
    else:
        sky, env, config_override = _check_skypilot_kubernetes(
            kubeconfig_path, context, sky_bin=sky_bin
        )
    infra = f"k8s/{context}"
    sky_cwd = kubeconfig_path.parent
    accelerator = sky_gpus.strip() or _detect_skypilot_gpu(
        sky, infra, env, config_override=config_override, cwd=sky_cwd
    )
    smoke_name = _sky_cluster_name(cluster_name)
    try:
        _run_stream(
            [
                sky,
                "launch",
                "--config",
                config_override,
                "-c",
                smoke_name,
                "--infra",
                infra,
                "--gpus",
                accelerator,
                "-y",
                "nvidia-smi",
            ],
            cwd=sky_cwd,
            env=env,
            timeout=1800,
        )
    finally:
        _run_stream(
            [sky, "down", "--config", config_override, "--yes", smoke_name],
            cwd=sky_cwd,
            env=env,
            timeout=600,
        )
        _wait_for_sky_down(
            sky,
            smoke_name,
            env,
            config_override=config_override,
            cwd=sky_cwd,
        )
    typer.echo(f"SkyPilot smoke passed and {smoke_name} was removed.")


def _detect_skypilot_gpu(
    sky: str,
    infra: str,
    env: dict[str, str],
    *,
    config_override: str = "",
    cwd: Path | None = None,
) -> str:
    cmd = [sky, "show-gpus", "--infra", infra, "--all"]
    if config_override:
        cmd[2:2] = ["--config", config_override]
    result = _run_capture(cmd, cwd=cwd, env=env, timeout=300)
    for line in result.stdout.splitlines():
        if "RTX" not in line.upper() or "6000" not in line:
            continue
        columns = [column for column in re.split(r"\s{2,}", line.strip()) if column]
        if columns:
            return f"{columns[0]}:1"
    raise typer.BadParameter(
        "Unable to auto-detect a Kubernetes GPU for SkyPilot; pass --sky-gpus"
    )


def _sky_cluster_name(cluster_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9-]+", "-", cluster_name).strip("-").lower()
    return f"{normalized[:40]}-sky-smoke"


def _wait_for_sky_down(
    sky: str,
    cluster_name: str,
    env: dict[str, str],
    *,
    config_override: str = "",
    cwd: Path | None = None,
) -> None:
    for _ in range(30):
        cmd = [sky, "status", "--refresh"]
        if config_override:
            cmd[2:2] = ["--config", config_override]
        result = _run_capture(cmd, cwd=cwd, env=env, timeout=120, check=False)
        if cluster_name not in result.stdout:
            return
        time.sleep(10)
    raise typer.BadParameter(
        f"SkyPilot cluster {cluster_name} still appears in sky status"
    )
