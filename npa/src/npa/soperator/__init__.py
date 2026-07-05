"""Deploy Nebius soperator (Slurm-on-Kubernetes) clusters from a compact npa spec.

This package wraps the public ``nebius/nebius-solutions-library`` soperator
Terraform recipe. Callers describe a cluster with a small declarative spec
(``npa.soperator/v0.0.1``) that supports **multiple worker node pools with
different presets** and an optional per-pool **Docker/Enroot image cache disk**
(node-local ``NETWORK_SSD_IO_M3``). The spec is rendered into the recipe's
``terraform.tfvars`` and applied; post-deploy fixes make the cluster usable.

No project/tenant/registry IDs are baked in here -- they are resolved from
``~/.npa/config.yaml`` or explicit arguments, keeping this module public-safe.
"""

from npa.soperator.spec import (
    SoperatorSpec,
    WorkerPoolSpec,
    load_spec,
    spec_from_mapping,
)
from npa.soperator.tfvars import render_tfvars

__all__ = [
    "SoperatorSpec",
    "WorkerPoolSpec",
    "load_spec",
    "spec_from_mapping",
    "render_tfvars",
]
