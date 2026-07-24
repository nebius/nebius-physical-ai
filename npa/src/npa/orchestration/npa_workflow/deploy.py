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

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.spec import NpaWorkflowSpec

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

        if isinstance(directive, Mapping):
            cluster_name = str(
                directive.get("clusterName") or directive.get("cluster_name") or cluster_name
            ).strip() or DEFAULT_CLUSTER_NAME
            context = str(directive.get("context") or "").strip()
            project = str(directive.get("project") or "").strip()
            if "skipS3" in directive or "skip_s3" in directive:
                skip_s3 = _coerce_bool(directive.get("skipS3", directive.get("skip_s3")))
        elif not _coerce_bool(directive):
            continue

        targets.append(
            DeployTarget(
                profile=str(profile),
                cluster_name=cluster_name,
                context=context,
                project=project,
                accelerators=accelerators,
                cloud=cloud,
                skip_s3=skip_s3,
            )
        )
    return targets


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
    ) -> Any:
        return provision_if_absent(
            project=project,
            cluster_name=cluster_name,
            context_name=context_name,
            skip_s3=skip_s3,
            dry_run=dry_run,
        )

    return _provision


def ensure_infra_present(
    targets: list[DeployTarget],
    *,
    dry_run: bool = False,
    provisioner: Provisioner | None = None,
) -> list[dict[str, Any]]:
    """Provision each unique target cluster that declares ``deployIfAbsent``.

    Idempotent and deduplicated by resolved context: multiple GPU profiles on the
    same cluster provision it once. Returns one result record per unique context.
    """

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
            outcome = provision(
                project=target.project or None,
                cluster_name=target.cluster_name,
                context_name=context,
                skip_s3=target.skip_s3,
                dry_run=dry_run,
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
            }
        )
    return results


__all__ = [
    "DeployTarget",
    "ensure_infra_present",
    "parse_deploy_targets",
]
