"""Read-only Fleet project identity checks against an owner-private spec.

No projects, storage, clusters, or workload pods are created by this suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from npa.fleet.lifecycle import _nebius_cli_env, resolve_project_id
from npa.fleet.spec import FleetSpec, load_spec


pytestmark = pytest.mark.e2e


@pytest.fixture
def live_spec() -> FleetSpec:
    if not os.environ.get("NPA_FLEET_PROJECT_VERIFY_SPEC"):
        pytest.skip("supply an owner-private Fleet spec for read-only identity checks")
    spec = load_spec(Path(os.environ["NPA_FLEET_PROJECT_VERIFY_SPEC"]).expanduser())
    assert spec.profile and spec.tenant_id and spec.region
    assert spec.projects and all(project.project_id for project in spec.projects)
    return spec


@pytest.mark.parametrize("mismatch", ["", "tenant", "region"])
def test_existing_project_identity_live(live_spec: FleetSpec, mismatch: str) -> None:
    env = _nebius_cli_env()
    for key in ("NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN", "TF_VAR_iam_token"):
        env.pop(key, None)
    for project in live_spec.projects:
        kwargs = {
            "prefix": live_spec.project_prefix,
            "create": False,
            "env": env,
            "region": "wrong-region" if mismatch == "region" else project.region or live_spec.region,
            "profile": live_spec.profile,
        }
        tenant = "wrong-tenant" if mismatch == "tenant" else live_spec.tenant_id
        if mismatch:
            with pytest.raises(ValueError, match="immutable identity verification"):
                resolve_project_id("nebius", tenant, project, **kwargs)
        else:
            project_id, created = resolve_project_id("nebius", tenant, project, **kwargs)
            assert project_id == project.project_id
            assert created is False
