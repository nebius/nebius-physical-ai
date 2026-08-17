"""``deployIfAbsent`` — provision workflow infra (e.g. GPU clusters) on demand.

A resource profile in an ``npa.workflow`` spec may declare ``deployIfAbsent`` so
that ``npa workbench workflow submit`` provisions the target Kubernetes/GPU
cluster through ``npa``'s provisioning path *before* submitting, instead of
failing when the cluster is missing. This keeps operators on the ``npa`` toolchain
(never calling ``sky``/``kubectl``/terraform directly) and makes a spec
self-provisioning.

Accepted forms inside a ``resources.<profile>`` block::

    resources:
      trainer-gpu:
        cloud: kubernetes
        accelerators: RTXPRO6000:1
        deployIfAbsent: true                 # provision with config defaults

      trainer-gpu-explicit:
        cloud: kubernetes
        accelerators: RTXPRO6000:1
        deployIfAbsent:
          clusterName: npa-rtxpro-mk8s        # cluster profile / context
          context: npa-rtxpro-mk8s            # optional; defaults to clusterName
          project: default                    # optional npa project alias
          skipS3: true                        # optional; default true (k8s only)

Provisioning is idempotent: the underlying ``provision_if_absent`` reuses a
cached kubeconfig when the cluster already exists, so a present cluster is a
no-op ("reused") rather than a re-deploy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.spec import NpaWorkflowSpec, profile_num_nodes
from npa.provisioning_preflight import (
    DEFAULT_CPU_NODES,
    DEFAULT_CPU_PLATFORM,
    DEFAULT_CPU_PRESET,
    DEFAULT_GPU_NODES,
    DEFAULT_GPU_PLATFORM,
    DEFAULT_GPU_PRESET,
)

DEFAULT_CLUSTER_NAME = "npa-cluster"

# provisioner(project, cluster_name, context_name, skip_s3, dry_run) -> result object
Provisioner = Callable[..., Any]


@dataclass(frozen=True)
class DeployTarget:
    """A resource profile that should be provisioned when absent."""

    profile: str
    cluster_name: str = DEFAULT_CLUSTER_NAME
    context: str = ""
    project: str = ""
    accelerators: str = ""
    cloud: str = "kubernetes"
    skip_s3: bool = True
    cpu_nodes: int = DEFAULT_CPU_NODES
    cpu_platform: str = DEFAULT_CPU_PLATFORM
    cpu_preset: str = DEFAULT_CPU_PRESET
    gpu_nodes: int = DEFAULT_GPU_NODES
    gpu_platform: str = DEFAULT_GPU_PLATFORM
    gpu_preset: str = DEFAULT_GPU_PRESET
    preemptible: bool = False

    @property
    def resolved_context(self) -> str:
        return self.context.strip() or self.cluster_name


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def parse_deploy_targets(spec: NpaWorkflowSpec) -> list[DeployTarget]:
    """Extract deploy targets from resource profiles that set ``deployIfAbsent``."""

    targets: list[DeployTarget] = []
    resources = spec.resources if isinstance(spec.resources, Mapping) else {}
    for profile, raw in resources.items():
        if not isinstance(raw, Mapping):
            continue
        directive = raw.get("deployIfAbsent", raw.get("deploy_if_absent"))
        if directive in (None, False):
            continue

        cloud = str(raw.get("cloud") or "kubernetes").strip().lower()
        accelerators = str(raw.get("accelerators") or "").strip()
        cluster_name = DEFAULT_CLUSTER_NAME
        context = ""
        project = ""
        skip_s3 = True
        cpu_nodes = DEFAULT_CPU_NODES
        cpu_platform = DEFAULT_CPU_PLATFORM
        cpu_preset = DEFAULT_CPU_PRESET
        gpu_nodes = DEFAULT_GPU_NODES
        gpu_platform = DEFAULT_GPU_PLATFORM
        gpu_preset = DEFAULT_GPU_PRESET
        preemptible = False

        if isinstance(directive, Mapping):
            cluster_name = (
                str(
                    directive.get("clusterName")
                    or directive.get("cluster_name")
                    or cluster_name
                ).strip()
                or DEFAULT_CLUSTER_NAME
            )
            context = str(directive.get("context") or "").strip()
            project = str(directive.get("project") or "").strip()
            if "skipS3" in directive or "skip_s3" in directive:
                skip_s3 = _coerce_bool(
                    directive.get("skipS3", directive.get("skip_s3"))
                )
            cpu_nodes = int(
                directive.get("cpuNodes", directive.get("cpu_nodes", cpu_nodes))
            )
            cpu_platform = str(
                directive.get("cpuPlatform")
                or directive.get("cpu_platform")
                or cpu_platform
            ).strip()
            cpu_preset = str(
                directive.get("cpuPreset") or directive.get("cpu_preset") or cpu_preset
            ).strip()
            gpu_nodes = int(
                directive.get("gpuNodes", directive.get("gpu_nodes", gpu_nodes))
            )
            gpu_platform = str(
                directive.get("gpuPlatform")
                or directive.get("gpu_platform")
                or gpu_platform
            ).strip()
            gpu_preset = str(
                directive.get("gpuPreset") or directive.get("gpu_preset") or gpu_preset
            ).strip()
            preemptible = _coerce_bool(
                directive.get(
                    "preemptible", directive.get("gpuPreemptible", preemptible)
                )
            )
        elif not _coerce_bool(directive):
            continue

        # A gang-scheduled stage needs a cluster that can actually hold the block:
        # `num_nodes: 4` on a one-GPU-node cluster does not fail, it sits PENDING.
        # An explicit gpuNodes directive still wins when it asks for more.
        gang_nodes = profile_num_nodes(dict(raw), name=str(profile), config=spec.config)
        if accelerators and gang_nodes > gpu_nodes:
            gpu_nodes = gang_nodes

        targets.append(
            DeployTarget(
                profile=str(profile),
                cluster_name=cluster_name,
                context=context,
                project=project,
                accelerators=accelerators,
                cloud=cloud,
                skip_s3=skip_s3,
                cpu_nodes=cpu_nodes,
                cpu_platform=cpu_platform,
                cpu_preset=cpu_preset,
                gpu_nodes=gpu_nodes,
                gpu_platform=gpu_platform,
                gpu_preset=gpu_preset,
                preemptible=preemptible,
            )
        )
    return targets


def bind_deploy_targets_to_submit(
    targets: list[DeployTarget], *, project: str = "", infra: str = ""
) -> list[DeployTarget]:
    """Bind explicit submit identity before any deploy planning or mutation."""

    selected_project = str(project or "").strip()
    selected_context = ""
    raw_infra = str(infra or "").strip()
    if "/" in raw_infra:
        kind, _, candidate = raw_infra.partition("/")
        if kind.strip().lower() in {"k8s", "kubernetes"}:
            selected_context = candidate.strip()

    bound: list[DeployTarget] = []
    for target in targets:
        item = target
        if selected_project:
            item = replace(item, project=selected_project)
        if selected_context and item.cloud.strip().lower() in {"k8s", "kubernetes"}:
            item = replace(
                item,
                cluster_name=selected_context,
                context=selected_context,
            )
        bound.append(item)
    return bound


def _default_provisioner() -> Provisioner:
    # Lazy import: keeps heavy config/nebius/cluster deps out of import time and
    # out of unit tests (which inject a fake provisioner).
    from npa.provisioning import provision_if_absent

    def _provision(
        *,
        project: str | None,
        cluster_name: str,
        context_name: str,
        skip_s3: bool,
        dry_run: bool,
        accelerator: str,
        gpu_readiness_timeout: float,
        gpu_readiness_poll_interval: float,
        sky_bin: str,
        cpu_nodes: int,
        cpu_platform: str,
        cpu_preset: str,
        gpu_nodes: int,
        gpu_platform: str,
        gpu_preset: str,
        preemptible: bool,
        _resolved_plan: Any = None,
    ) -> Any:
        return provision_if_absent(
            project=project,
            cluster_name=cluster_name,
            context_name=context_name,
            skip_s3=skip_s3,
            dry_run=dry_run,
            accelerator=accelerator,
            gpu_readiness_timeout=gpu_readiness_timeout,
            gpu_readiness_poll_interval=gpu_readiness_poll_interval,
            sky_bin=sky_bin,
            cpu_nodes=cpu_nodes,
            cpu_platform=cpu_platform,
            cpu_preset=cpu_preset,
            gpu_nodes=gpu_nodes,
            gpu_platform=gpu_platform,
            gpu_preset=gpu_preset,
            preemptible=preemptible,
            _resolved_plan=_resolved_plan,
        )

    return _provision


def ensure_infra_present(
    targets: list[DeployTarget],
    *,
    dry_run: bool = False,
    provisioner: Provisioner | None = None,
    gpu_readiness_timeout: float = 600.0,
    gpu_readiness_poll_interval: float = 10.0,
    sky_bin: str = "",
    resolved_plans: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Provision each unique target cluster that declares ``deployIfAbsent``.

    Idempotent and deduplicated by resolved context: multiple GPU profiles on the
    same cluster provision it once. Returns one result record per unique context.
    """

    from npa.lifecycle_intent import forbid_destructive_provisioning

    forbid_destructive_provisioning("ensure_infra_present")

    if not targets:
        return []
    provision = provisioner or _default_provisioner()

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for target in targets:
        context = target.resolved_context
        if context in seen:
            continue
        seen.add(context)
        try:
            resolved_plan = (resolved_plans or {}).get(context)
            outcome = provision(
                project=target.project or None,
                cluster_name=target.cluster_name,
                context_name=context,
                skip_s3=target.skip_s3,
                dry_run=dry_run,
                accelerator=target.accelerators,
                gpu_readiness_timeout=gpu_readiness_timeout,
                gpu_readiness_poll_interval=gpu_readiness_poll_interval,
                sky_bin=sky_bin,
                cpu_nodes=target.cpu_nodes,
                cpu_platform=target.cpu_platform,
                cpu_preset=target.cpu_preset,
                gpu_nodes=target.gpu_nodes,
                gpu_platform=target.gpu_platform,
                gpu_preset=target.gpu_preset,
                preemptible=target.preemptible,
                _resolved_plan=resolved_plan,
            )
        except Exception as exc:  # noqa: BLE001 - surface as workflow error
            raise NpaWorkflowError(
                f"deployIfAbsent failed for resource {target.profile!r} "
                f"(cluster {target.cluster_name!r}): {exc}"
            ) from exc
        results.append(
            {
                "profile": target.profile,
                "cluster_name": target.cluster_name,
                "context": context,
                "accelerators": target.accelerators,
                "status": getattr(outcome, "status", "ok"),
                "actions": list(getattr(outcome, "actions", []) or []),
                "warnings": list(getattr(outcome, "warnings", []) or []),
                "dry_run": dry_run,
                "topology": (
                    resolved_plan.topology.to_dict()
                    if resolved_plan is not None
                    else getattr(outcome, "preflight", {}).get("topology", {})
                ),
                "quotas": (
                    [quota.to_dict() for quota in resolved_plan.quotas]
                    if resolved_plan is not None
                    else getattr(outcome, "preflight", {}).get("quotas", [])
                ),
            }
        )
    return results


def plan_infra_present(
    targets: list[DeployTarget], *, mutation: bool
) -> dict[str, Any]:
    """Resolve every unique deploy target without writing state or provisioning."""

    from npa.provisioning import resolve_provision_plan

    plans: dict[str, Any] = {}
    for target in targets:
        context = target.resolved_context
        if context in plans:
            continue
        plan = resolve_provision_plan(
            project=target.project or None,
            cluster_name=target.cluster_name,
            context_name=context,
            skip_k8s=False,
            accelerator=target.accelerators,
            cpu_nodes=target.cpu_nodes,
            cpu_platform=target.cpu_platform,
            cpu_preset=target.cpu_preset,
            gpu_nodes=target.gpu_nodes,
            gpu_platform=target.gpu_platform,
            gpu_preset=target.gpu_preset,
            preemptible=target.preemptible,
            mutation=mutation,
        )
        if mutation:
            plan.assert_mutation_ready()
        plans[context] = plan
    return plans


__all__ = [
    "DeployTarget",
    "bind_deploy_targets_to_submit",
    "ensure_infra_present",
    "plan_infra_present",
    "parse_deploy_targets",
]
