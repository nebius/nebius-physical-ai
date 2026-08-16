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
                gpu_driver_mode="auto",  # managed image; operator is opt-in
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
    ProjectSpec,
    load_spec,
    spec_from_mapping,
)

__all__ = [
    "deploy",
    "destroy",
    "plan",
    "status",
    "FleetSpec",
    "ProjectSpec",
    "ClusterSpec",
    "NodePoolSpec",
    "MigSpec",
    "MigVerificationReport",
    "verify_mig_cluster",
    "wait_for_mig_ready",
    "load_spec",
    "spec_from_mapping",
]
