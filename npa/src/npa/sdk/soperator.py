"""SDK surface for deploying soperator (Slurm-on-Kubernetes) clusters.

Mirrors the ``npa soperator`` CLI. Programmatic callers build a
:class:`~npa.soperator.spec.SoperatorSpec` (or load one from YAML) and call
:func:`deploy`, :func:`destroy`, or :func:`status`.

Example::

    from npa.sdk import soperator
    from npa.soperator.spec import SoperatorSpec, WorkerPoolSpec

    spec = SoperatorSpec(
        name="npa-soperator",
        region="us-central1",
        # Explicitly grants root SSH access to the public login node.
        root_login_ssh_public_key="ssh-ed25519 AAAA... operator",
        workers=[
            WorkerPoolSpec(name="cpu", platform="cpu-d3", preset="8vcpu-32gb", docker_cache=True),
            WorkerPoolSpec(name="gpu", platform="gpu-b200-sxm", preset="8gpu-160vcpu-1792gb",
                           size=2, fabric="us-central1-b", preemptible=True, docker_cache=True),
        ],
    )
    result = soperator.deploy(spec)
"""

from __future__ import annotations

from npa.soperator.lifecycle import (
    DeploymentValidationFailure,
    SoperatorDeploymentValidationError,
    apply_post_deploy_fixes,
    deploy_cluster as deploy,
    destroy_cluster as destroy,
    plan_cluster as plan,
)
from npa.soperator.spec import SoperatorSpec, WorkerPoolSpec, load_spec, spec_from_mapping

__all__ = [
    "deploy",
    "destroy",
    "plan",
    "apply_post_deploy_fixes",
    "DeploymentValidationFailure",
    "SoperatorDeploymentValidationError",
    "SoperatorSpec",
    "WorkerPoolSpec",
    "load_spec",
    "spec_from_mapping",
]
