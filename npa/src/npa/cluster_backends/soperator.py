"""Soperator implementation of the shared cluster backend contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from npa.cluster_backends.base import BackendCapabilities, MaterializedPlan
from npa.soperator.lifecycle import DEFAULT_GPU_CREATION_CHECK_TIMEOUT_SECONDS
from npa.soperator.spec import DEFAULT_SOLUTIONS_LIBRARY_REF, SoperatorSpec
from npa.soperator.tfvars import render_tfvars


@dataclass(frozen=True)
class SoperatorApplyRequest:
    terraform_dir: Path | None = None
    work_root: Path | None = None
    solutions_library_ref: str = DEFAULT_SOLUTIONS_LIBRARY_REF
    root_login_ssh_public_key_file: Path | None = None
    project: str | None = None
    timeout_minutes: int = 90
    gpu_creation_check_timeout_seconds: int = DEFAULT_GPU_CREATION_CHECK_TIMEOUT_SECONDS
    apply_fixes: bool = True
    source_preflight_only: bool = False
    stream_terraform_output: bool = True
    on_status: Callable[[str], None] | None = None
    profile: str = ""
    provider_preflight: bool = False
    provider_nebius_bin: str = ""
    provider_tenant_id: str = ""
    provider_project_id: str = ""
    provider_region: str = ""
    provider_install_dir: Path | None = None
    provider_work_root: Path | None = None
    provider_env: dict[str, str] | None = None


@dataclass(frozen=True)
class SoperatorStatusRequest:
    terraform_dir: Path | None = None
    work_root: Path | None = None


@dataclass(frozen=True)
class SoperatorDestroyRequest:
    terraform_dir: Path | None = None
    work_root: Path | None = None
    solutions_library_ref: str = DEFAULT_SOLUTIONS_LIBRARY_REF
    project: str | None = None
    timeout_minutes: int = 90
    source_preflight_only: bool = False
    on_status: Callable[[str], None] | None = None
    profile: str = ""


class SoperatorBackend:
    name = "soperator"
    capabilities = BackendCapabilities(
        backend=name,
        supports_mig=False,
        supports_shared_filestore=False,
        supports_slurm=True,
        supports_capacity_blocks=True,
        supports_cuda_verification=True,
    )

    def validate(self, desired: SoperatorSpec) -> None:
        desired.validate()

    def plan(self, desired: SoperatorSpec) -> dict[str, Any]:
        from npa.soperator import lifecycle

        return lifecycle.plan_cluster(desired)

    def preflight(
        self, desired: SoperatorSpec, request: SoperatorApplyRequest
    ) -> dict[str, Any]:
        self.validate(desired)
        result = {
            "backend": self.name,
            "reservation_preflight": self.plan(desired)["reservation_preflight"],
            "provider_mutation": False,
        }
        if request.provider_preflight:
            from npa.soperator.lifecycle import _resolve_reserved_worker_capacity

            install_dir = request.provider_install_dir
            if request.provider_work_root is not None:
                candidates = sorted(
                    request.provider_work_root.glob(
                        "nebius-solutions-library*/soperator/installations/"
                        + desired.name
                    )
                )
                if len(candidates) > 1:
                    raise ValueError(
                        "multiple Soperator installations match the persisted target; "
                        "refusing ambiguous reserved-capacity credit"
                    )
                install_dir = candidates[0] if candidates else install_dir
            required = {
                "provider_nebius_bin": request.provider_nebius_bin,
                "provider_tenant_id": request.provider_tenant_id,
                "provider_project_id": request.provider_project_id,
                "provider_region": request.provider_region,
                "provider_install_dir": install_dir,
                "provider_env": request.provider_env,
            }
            missing = [
                key for key, value in required.items() if value is None or value == ""
            ]
            if missing:
                raise ValueError(
                    "soperator provider preflight requires: " + ", ".join(missing)
                )
            resolved, summary = _resolve_reserved_worker_capacity(
                desired,
                install_dir=install_dir,
                nebius_bin=request.provider_nebius_bin,
                tenant_id=request.provider_tenant_id,
                project_id=request.provider_project_id,
                region=request.provider_region,
                env=request.provider_env,
                on_status=request.on_status,
            )
            result["reservation_preflight"] = summary
            result["resolved_desired"] = resolved
        return {"backend": self.name, **result}

    def materialize(
        self, desired: SoperatorSpec, request: SoperatorApplyRequest
    ) -> MaterializedPlan:
        self.validate(desired)
        return MaterializedPlan(
            backend=self.name,
            desired_state=self.plan(desired),
            deployment_inputs={"terraform_tfvars": render_tfvars(desired)},
        )

    def apply(
        self, desired: SoperatorSpec, request: SoperatorApplyRequest
    ) -> dict[str, Any]:
        from npa.soperator import lifecycle

        self.validate(desired)
        try:
            result = lifecycle.deploy_cluster(
                desired,
                terraform_dir=request.terraform_dir,
                work_root=request.work_root,
                solutions_library_ref=request.solutions_library_ref,
                root_login_ssh_public_key_file=request.root_login_ssh_public_key_file,
                project=request.project,
                timeout_minutes=request.timeout_minutes,
                gpu_creation_check_timeout_seconds=request.gpu_creation_check_timeout_seconds,
                apply_fixes=request.apply_fixes,
                source_preflight_only=request.source_preflight_only,
                stream_terraform_output=request.stream_terraform_output,
                on_status=request.on_status,
                profile=request.profile,
            )
        except (
            lifecycle.SoperatorDeploymentValidationError,
            lifecycle.SoperatorStateCaptureError,
        ) as exc:
            exc.result["backend"] = self.name
            raise
        return {**(result or {}), "backend": self.name} if result is not None else None

    def status(
        self, desired: SoperatorSpec, request: SoperatorStatusRequest
    ) -> dict[str, Any]:
        from npa.soperator import lifecycle

        result = lifecycle.cluster_status(
            desired.name,
            terraform_dir=request.terraform_dir,
            work_root=request.work_root,
        )
        return {"backend": self.name, **result}

    def verify(
        self, desired: SoperatorSpec, request: SoperatorStatusRequest
    ) -> dict[str, Any]:
        return self.status(desired, request)

    def destroy(
        self, desired: SoperatorSpec, request: SoperatorDestroyRequest
    ) -> dict[str, Any] | None:
        from npa.soperator import lifecycle

        result = lifecycle.destroy_cluster(
            desired.name,
            terraform_dir=request.terraform_dir,
            work_root=request.work_root,
            solutions_library_ref=request.solutions_library_ref,
            project=request.project,
            timeout_minutes=request.timeout_minutes,
            source_preflight_only=request.source_preflight_only,
            on_status=request.on_status,
            profile=request.profile,
        )
        return {**result, "backend": self.name} if result is not None else None
