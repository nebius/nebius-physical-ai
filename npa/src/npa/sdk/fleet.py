"""SDK surface for deploying fleets of Managed Kubernetes clusters.

Mirrors the ``npa fleet`` CLI. Programmatic callers build a
:class:`~npa.fleet.spec.FleetSpec` (or load one from YAML) and call
:func:`deploy`, :func:`destroy`, :func:`plan`, or :func:`status`.

Example::

    from npa.sdk import fleet
    from npa.fleet.spec import FleetSpec, ProjectSpec, ClusterSpec, NodePoolSpec

    spec = FleetSpec(
        name="fleet1-test",
        region="us-central1",
        project_prefix="fleet1-test-",
        profile="",  # ~/.nebius profile / tenant to authenticate as ("" = active)
        projects=[
            ProjectSpec(name="a", clusters=[ClusterSpec(
                name="cluster",
                # Use gpu_workload_profile="rtx-rendering" for RTX/Isaac
                # rendering; it selects RTX PRO 6000 + operator graphics mounts.
                gpu_driver_mode="auto",  # managed image remains the default
                managed_driver_preset="cuda13.0",
                cpu_nodes=NodePoolSpec(count=1, platform="cpu-d3", preset="48vcpu-192gb"),
                gpu_nodes=NodePoolSpec(
                    count=1,
                    platform="gpu-rtx6000",
                    preset="1gpu-24vcpu-218gb",
                    # Set at runtime to bind this pool STRICTLY to reserved capacity:
                    capacity_block_group="",
                ),
            )]),
            ProjectSpec(name="b", clusters=[ClusterSpec(
                name="cluster",
                cpu_nodes=NodePoolSpec(count=1, platform="cpu-d3", preset="48vcpu-192gb"),
                gpu_nodes=NodePoolSpec(count=1, platform="gpu-rtx6000", preset="1gpu-24vcpu-218gb"),
            )]),
        ],
    )
    result = fleet.deploy(spec)
"""

from __future__ import annotations

from npa.fleet.lifecycle import (
    deploy_fleet as deploy,
    destroy_fleet as destroy,
    fleet_status as status,
    plan_fleet as plan,
)
from npa.fleet.mig import (
    MigSpec,
    MigVerificationReport,
    verify_mig_cluster,
    wait_for_mig_ready,
)
from npa.fleet.spec import (
    ClusterSpec,
    FleetSpec,
    NodePoolSpec,
    ObjectStorageSpec,
    ProjectSpec,
    load_spec,
    spec_from_mapping,
)


def verify_storage(spec, *, only_projects=None, only_clusters=None,
                   project_prefix=None, profile=None, evidence_dir=None) -> dict:
    """Verify every selected worker through the shared Fleet storage implementation.

    Args:
        spec: Loaded Fleet declaration.
        only_projects: Project keys or display names to include.
        only_clusters: Cluster names within selected projects.
        project_prefix: Optional project display-name prefix override.
        profile: Optional authentication profile override.
        evidence_dir: Owner-private directory for exact verification receipts.
    Returns:
        Sanitized target and worker evidence, counts, cleanup results, and hashes.
    Raises:
        StorageIdentityError: Selection or private evidence configuration is invalid.
    """
    from npa.fleet.storage_verification import verify_storage as verify

    return verify(spec, only_projects=only_projects, only_clusters=only_clusters,
                  project_prefix=project_prefix, profile=profile, evidence_dir=evidence_dir)

__all__ = [
    "deploy",
    "destroy",
    "plan",
    "status",
    "verify_storage",
    "FleetSpec",
    "ProjectSpec",
    "ClusterSpec",
    "NodePoolSpec",
    "ObjectStorageSpec",
    "MigSpec",
    "MigVerificationReport",
    "verify_mig_cluster",
    "wait_for_mig_ready",
    "load_spec",
    "spec_from_mapping",
]
