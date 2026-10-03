"""Read-only reservation and storage quota preflight for a planned Fleet.

The supplied spec is owner-private. Target projects need not exist yet; this
suite verifies tenant capacity only and never authorizes deployment by itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from npa.fleet.lifecycle import _nebius_cli_env, _preflight_quotas
from npa.fleet.spec import load_spec


pytestmark = pytest.mark.e2e


def test_planned_fleet_storage_and_reserved_capacity_live() -> None:
    if not os.environ.get("NPA_FLEET_QUOTA_VERIFY_SPEC"):
        pytest.skip("supply an owner-private planned Fleet spec for quota checks")
    spec = load_spec(Path(os.environ["NPA_FLEET_QUOTA_VERIFY_SPEC"]).expanduser())
    assert spec.profile and spec.tenant_id and spec.region
    clusters = {}
    storage = {}
    for project in spec.projects:
        region = project.region or spec.region
        clusters.setdefault(region, []).extend(project.clusters)
        if project.object_storage and project.object_storage.enabled:
            storage.setdefault(region, []).append(project.object_storage)
    assert storage, "this suite requires a Fleet object-storage declaration"
    env = _nebius_cli_env()
    for key in ("NEBIUS_IAM_TOKEN", "NPA_NEBIUS_IAM_TOKEN", "TF_VAR_iam_token"):
        env.pop(key, None)
    evidence = []
    _preflight_quotas(
        "nebius",
        tenant_id=spec.tenant_id,
        by_region=clusters,
        new_projects_by_region={},
        env=env,
        profile=spec.profile,
        on_status=evidence.append,
        object_storage_by_region=storage,
    )
    for region, declarations in storage.items():
        for item in declarations:
            quota = "storage.bucket.size." + item.normalized_storage_class().replace(
                "_", "-"
            )
            assert any(f"quota {quota} [{region}]" in line for line in evidence)
        assert any(
            f"quota storage.bucket.count [{region}]" in line for line in evidence
        )
