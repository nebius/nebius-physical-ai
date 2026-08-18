"""Managed Kubernetes implementation of the shared cluster backend contract."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable

from npa.cluster.gpu_driver import resolve_gpu_driver_strategy
from npa.cluster_backends.base import BackendCapabilities, MaterializedPlan
from npa.cluster_backends.mk8s_model import (
    MK8sDesired,
    MK8sExecutionScope,
    MK8sProjectIdentity,
    as_mk8s_desired,
)
from npa.cluster_backends.mk8s_render import render_tfvars


@dataclass(frozen=True)
class MK8sApplyRequest:
    """Typed execution boundary supplied by standalone or fleet orchestration."""

    ssh_public_key: str = ""
    recipe_dir: Path | None = None
    scope: MK8sExecutionScope | None = None
    project: MK8sProjectIdentity | None = None
    project_id: str = ""
    project_created: bool = False
    subnet_id: str = ""
    region: str = ""
    tenant_id: str = ""
    fleet_root: Path | None = None
    recipe_root: Path | None = None
    terraform_bin: str = ""
    nebius_bin: str = ""
    profile: str = ""
    timeout_minutes: int = 120
    on_status: Callable[[str], None] | None = None
    log_path: Path | None = None
    provider_env: dict[str, str] | None = None
    provider_preflight: bool = False
    # Legacy-state compatibility only. New standalone targets use the native
    # one-target request below; existing deploy/cluster state must continue to
    # reconcile in place rather than being silently orphaned.
    terraform_command: tuple[str, ...] = ()
    terraform_cwd: Path | None = None
    terraform_env: dict[str, str] | None = None
    terraform_timeout_seconds: int = 0
    terraform_cancel_reason: Callable[[], str | None] | None = None
    command_runner: Callable[..., Any] | None = None
    standalone_context: str = ""
    standalone_kubeconfig: Path | None = None
    # Fleet historically validates GPU/MIG targets as part of deploy, while the
    # standalone CLI additionally promises CPU node-count and default
    # StorageClass validation. Keep that surface policy explicit instead of
    # allowing the standalone ``--validate`` flag to leak into fleet semantics.
    post_deploy_validation: str = "fleet"
    basic_validation_timeout_minutes: int = 30
    kubectl_bin: str = ""


@dataclass(frozen=True)
class MK8sStatusRequest:
    state: dict[str, Any] | None = None
    install_dir: Path | None = None
    kubeconfig: Path | None = None
    kubectl_bin: str = ""
    evidence_path: Path | None = None
    on_status: Callable[[str], None] | None = None
    run_capture: Callable[..., Any] | None = None
    mig_verifier: Callable[..., Any] | None = None
    gpu_health_verifier: Callable[..., Any] | None = None
    validation_policy: str = "fleet"
    basic_validation_timeout_minutes: int = 30


@dataclass(frozen=True)
class MK8sDestroyRequest:
    scope: MK8sExecutionScope
    project: MK8sProjectIdentity
    fleet_root: Path
    terraform_bin: str
    nebius_bin: str
    profile: str = ""
    timeout_minutes: int = 120
    on_status: Callable[[str], None] | None = None
    log_path: Path | None = None


def desired_state(cluster: MK8sDesired) -> dict[str, Any]:
    """Canonical mk8s desired state used by every planning surface."""

    cluster = as_mk8s_desired(cluster)
    gpu = cluster.gpu_nodes
    driver = resolve_gpu_driver_strategy(
        gpu_nodes=cluster.gpu_count(),
        platform=gpu.platform if gpu else "",
        preset=gpu.preset if gpu else "",
        mode=cluster.resolved_gpu_driver_mode(),
        managed_driver_preset=cluster.managed_driver_preset,
        enable_gpu_cluster=cluster.resolved_enable_gpu_cluster(),
        allow_unsafe_nvswitch_operator=cluster.allow_unsafe_nvswitch_operator,
    )
    return {
        "backend": "mk8s",
        "name": cluster.name,
        "cpu_nodes": cluster.cpu_count(),
        "cpu_platform": cluster.cpu_nodes.platform if cluster.cpu_nodes else "",
        "cpu_preset": cluster.cpu_nodes.preset if cluster.cpu_nodes else "",
        "gpu_nodes": cluster.gpu_count(),
        "gpu_platform": gpu.platform if gpu else "",
        "gpu_preset": gpu.preset if gpu else "",
        "gpu_reservation": (
            "strict" if gpu and gpu.capacity_block_group else "on-demand"
        ),
        "gpu_preemptible": bool(gpu and gpu.preemptible),
        "enable_gpu_cluster": cluster.resolved_enable_gpu_cluster(),
        "gpu_driver_mode": driver.effective_mode,
        "managed_driver_preset": (
            driver.managed_driver_preset if driver.uses_managed_image else None
        ),
        "unsafe_nvswitch_operator": driver.unsafe_operator_acknowledged,
        "gpu_health_stabilization_seconds": cluster.gpu_health_stabilization_seconds,
        "gpu_health_timeout_minutes": cluster.gpu_health_timeout_minutes,
        "gpu_cuda_smoke": cluster.gpu_cuda_smoke,
        "gpu_cuda_smoke_image": cluster.gpu_cuda_smoke_image,
        "enable_filestore": cluster.enable_filestore,
        "filestore_disk_size_gibibytes": cluster.filestore_disk_size_gibibytes,
        "filestore_mount_path": cluster.filestore_mount_path,
        "filestore_mount_tag": cluster.filestore_mount_tag,
        "k8s_version": cluster.resolved_k8s_version() or "backend-default",
        "mig": (
            {"strategy": cluster.mig.strategy, "config": cluster.mig.config}
            if cluster.mig and cluster.mig.enabled
            else None
        ),
    }


class MK8sBackend:
    name = "mk8s"
    capabilities = BackendCapabilities(
        backend=name,
        supports_mig=True,
        supports_shared_filestore=True,
        supports_slurm=False,
        supports_capacity_blocks=True,
        supports_cuda_verification=True,
    )

    def validate(self, desired: MK8sDesired) -> None:
        as_mk8s_desired(desired)

    def plan(self, desired: MK8sDesired) -> dict[str, Any]:
        return desired_state(desired)

    def preflight(
        self, desired: MK8sDesired, request: MK8sApplyRequest
    ) -> dict[str, Any]:
        desired = as_mk8s_desired(desired)
        result: dict[str, Any] = {
            "backend": self.name,
            "required": True,
            "provider_mutation": False,
        }
        if request.provider_preflight:
            if not (
                request.nebius_bin
                and request.tenant_id
                and request.region
                and request.provider_env is not None
            ):
                raise ValueError(
                    "mk8s provider preflight requires nebius_bin, tenant_id, "
                    "region, and provider_env"
                )
            from npa.cluster_backends.mk8s_execution import is_verified_unchanged_target
            from npa.cluster_backends.process import run_capture
            from npa.cluster_backends.quotas import preflight_region, shortfall_message

            if (
                request.project is not None
                and request.fleet_root is not None
                and request.scope is not None
                and is_verified_unchanged_target(
                    project=request.project,
                    cluster=desired,
                    prefix=request.scope.project_prefix,
                    tenant_id=request.tenant_id,
                    region=request.region,
                    ssh_public_key=request.ssh_public_key,
                    fleet_root=request.fleet_root,
                    nebius_bin=request.nebius_bin,
                    profile=request.profile,
                    env=request.provider_env,
                )
            ):
                result["capacity_quota"] = "provider-verified-zero-increment"
                result["incremental_demand"] = 0
                return result
            shortfalls = preflight_region(
                nebius_bin=request.nebius_bin,
                tenant_id=request.tenant_id,
                region=request.region,
                clusters=[desired],
                env=request.provider_env,
                profile=request.profile,
                run_capture=run_capture,
                nebius_argv=lambda binary, profile: (
                    [binary, "--profile", profile] if profile else [binary]
                ),
                on_status=request.on_status,
            )
            if shortfalls:
                raise ValueError(shortfall_message(shortfalls, request.tenant_id))
            result["capacity_quota"] = "passed"
        return result

    def materialize(
        self, desired: MK8sDesired, request: MK8sApplyRequest
    ) -> MaterializedPlan:
        desired = as_mk8s_desired(desired)
        rendered = render_tfvars(
            desired,
            ssh_public_key=request.ssh_public_key,
            recipe_dir=request.recipe_dir,
        )
        return MaterializedPlan(
            backend=self.name,
            desired_state=desired_state(desired),
            deployment_inputs={"terraform_tfvars": rendered},
        )

    def apply(self, desired: MK8sDesired, request: MK8sApplyRequest) -> dict[str, Any]:
        desired = as_mk8s_desired(desired)
        if request.terraform_command:
            if request.terraform_cwd is None or request.terraform_env is None:
                raise ValueError(
                    "mk8s Terraform apply requires terraform_cwd and terraform_env"
                )
            from npa.cluster_backends.process import run_stream

            (request.command_runner or run_stream)(
                list(request.terraform_command),
                cwd=request.terraform_cwd,
                env=request.terraform_env,
                timeout=request.terraform_timeout_seconds,
                cancel=request.terraform_cancel_reason,
            )
            return {
                "backend": self.name,
                "cluster_name": desired.name,
                "status": "applied",
            }
        required = {
            "scope": request.scope,
            "project": request.project,
            "fleet_root": request.fleet_root,
            "recipe_root": request.recipe_root,
            "terraform_bin": request.terraform_bin,
            "nebius_bin": request.nebius_bin,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "mk8s apply requires native execution field(s): " + ", ".join(missing)
            )
        from npa.cluster_backends.mk8s_execution import deploy_cluster

        if request.standalone_context:
            assert request.fleet_root is not None
            request.fleet_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        result = deploy_cluster(
            spec=request.scope,
            project=request.project,
            cluster=desired,
            project_id=request.project_id,
            project_created=request.project_created,
            subnet_id=request.subnet_id,
            region=request.region,
            tenant_id=request.tenant_id,
            ssh_public_key=request.ssh_public_key,
            fleet_root=request.fleet_root,
            recipe_root=request.recipe_root,
            terraform_bin=request.terraform_bin,
            nebius_bin=request.nebius_bin,
            profile=request.profile,
            timeout_minutes=request.timeout_minutes,
            on_status=request.on_status,
            log_path=request.log_path,
            validation_policy=request.post_deploy_validation,
            basic_validation_timeout_minutes=request.basic_validation_timeout_minutes,
            kubectl_bin=request.kubectl_bin,
        )
        if request.standalone_context and result.get("status") == "deployed":
            result = self._adopt_standalone_result(desired, request, result)
        return {"backend": self.name, **result}

    def _adopt_standalone_result(
        self,
        desired: MK8sDesired,
        request: MK8sApplyRequest,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Adopt a one-target backend result under the legacy CLI context."""

        import yaml

        from npa.cluster.state import (
            ClusterState,
            delete_cluster_state,
            kubeconfig_file,
            save_cluster_state,
            utc_now_iso,
        )

        context = request.standalone_context
        source_context = str(result.get("kube_context") or "")
        source = Path(str(result.get("kubeconfig") or ""))
        target = request.standalone_kubeconfig or kubeconfig_file(context)
        if not source.is_file():
            raise RuntimeError("shared mk8s apply returned no durable kubeconfig")
        payload = yaml.safe_load(source.read_text()) or {}
        for item in payload.get("contexts") or []:
            if isinstance(item, dict) and item.get("name") == source_context:
                item["name"] = context
        if payload.get("current-context") == source_context:
            payload["current-context"] = context
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)
                temporary = Path(handle.name)
            temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        primary = desired.gpu_nodes or desired.cpu_nodes
        state = ClusterState(
            name=context,
            cluster_id=str(result.get("cluster_id") or ""),
            project_id=request.project_id,
            region=request.region,
            node_count=desired.cpu_count() + desired.gpu_count(),
            node_platform=primary.platform if primary else "",
            node_preset=primary.preset if primary else "",
            k8s_version=desired.k8s_version,
            subnet_id=request.subnet_id,
            created_at=utc_now_iso(),
            last_seen_state="RUNNING",
            node_group_id=str(result.get("node_group_id") or ""),
            endpoint=str(result.get("endpoint") or ""),
            kubeconfig_path=str(target),
            provider_name=desired.name,
        )
        assert request.scope is not None
        assert request.project is not None
        assert request.fleet_root is not None
        save_cluster_state(
            state,
            metadata={
                "managed_by": "npa cluster shared-mk8s-backend",
                "backend": "mk8s",
                "backend_state_root": str(request.fleet_root.resolve()),
                "backend_fleet_name": request.scope.fleet_name,
                "backend_project_key": request.project.key(),
                "backend_cluster_name": desired.name,
                "backend_project_id": request.project_id,
                "backend_cluster_id": state.cluster_id,
                "backend_tenant_id": request.tenant_id,
                "backend_region": request.region,
                "backend_profile": request.profile,
            },
        )
        if source_context and source_context != context:
            delete_cluster_state(source_context)
        return {
            **result,
            "kube_context": context,
            "kubeconfig": str(target),
            "ownership": "standalone-shared-mk8s-backend",
        }

    def status(
        self, desired: MK8sDesired, request: MK8sStatusRequest
    ) -> dict[str, Any]:
        desired = as_mk8s_desired(desired)
        state = dict(request.state or {})
        state.setdefault("cluster_name", desired.name)
        state.setdefault("status", "unknown")
        return {"backend": self.name, **state}

    def verify(
        self, desired: MK8sDesired, request: MK8sStatusRequest
    ) -> dict[str, Any]:
        desired = as_mk8s_desired(desired)
        if request.kubeconfig is None:
            return self.status(desired, request)
        from npa.cluster_backends.mk8s_execution import verify_cluster
        from npa.cluster_backends.process import run_capture
        from npa.cluster.gpu_health import validate_gpu_health
        from npa.cluster_backends.mig import wait_for_mig_ready

        return verify_cluster(
            cluster=desired,
            kubeconfig=request.kubeconfig,
            kubectl_bin=request.kubectl_bin,
            evidence_path=request.evidence_path,
            on_status=request.on_status,
            run_capture=request.run_capture or run_capture,
            mig_verifier=request.mig_verifier or wait_for_mig_ready,
            gpu_health_verifier=request.gpu_health_verifier or validate_gpu_health,
            validation_policy=request.validation_policy,
            basic_validation_timeout_seconds=(
                request.basic_validation_timeout_minutes * 60
            ),
        )

    def destroy(
        self, desired: MK8sDesired, request: MK8sDestroyRequest
    ) -> dict[str, Any] | None:
        desired = as_mk8s_desired(desired)
        from npa.cluster_backends.mk8s_execution import destroy_cluster

        result = destroy_cluster(
            spec=request.scope,
            project=request.project,
            cluster=desired,
            fleet_root=request.fleet_root,
            terraform_bin=request.terraform_bin,
            nebius_bin=request.nebius_bin,
            profile=request.profile,
            timeout_minutes=request.timeout_minutes,
            on_status=request.on_status,
            log_path=request.log_path,
        )
        if result is None:
            return None
        return {"backend": self.name, **result}
