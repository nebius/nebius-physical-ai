"""Deploy fleets of Nebius Managed Kubernetes clusters from a compact npa spec.

A **fleet** is many ``k8s-training`` clusters across many projects in one
tenant. Callers describe the fleet with a small declarative spec
(``npa.fleet/v0.0.1``) that supports:

* a shared ``defaults`` cluster profile (identical clusters across projects),
* per-project / per-cluster overrides (custom clusters), freely mixed, and
* projects referenced by id *or* created on demand under the tenant.

The recipe wrapped is the public ``nebius/nebius-solutions-library``
``k8s-training`` module -- repo-vendored by default, or a freshly cloned ref so
the fleet can consume the latest recipe changes. No project/tenant IDs are baked
in; they resolve from the spec, ``~/.npa`` / ``~/.nebius`` config, or arguments.
"""

from npa.fleet.mig import MigSpec, MigVerificationReport
from npa.fleet.spec import (
    API_VERSION,
    ClusterSpec,
    FleetSpec,
    FleetSpecError,
    NodePoolSpec,
    ProjectSpec,
    load_spec,
    spec_from_mapping,
)
from npa.fleet.tfvars import patch_provider_domain, provider_domain, render_tfvars

__all__ = [
    "API_VERSION",
    "ClusterSpec",
    "FleetSpec",
    "FleetSpecError",
    "NodePoolSpec",
    "MigSpec",
    "MigVerificationReport",
    "ProjectSpec",
    "load_spec",
    "spec_from_mapping",
    "render_tfvars",
    "provider_domain",
    "patch_provider_domain",
]
