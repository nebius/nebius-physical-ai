"""Check malformed provisioning requests against an authenticated deployed agent."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.agent_live,
    pytest.mark.skipif(
        not os.environ.get("NPA_AGENT_PROVISION_BOOLEAN_LIVE_CONFIG"),
        reason="Requires private connection configuration for an isolated CPU agent.",
    ),
]


def test_deployed_provision_rejects_malformed_booleans():
    path = Path(os.environ["NPA_AGENT_PROVISION_BOOLEAN_LIVE_CONFIG"])
    assert path.stat().st_mode & 0o077 == 0, "Connection config must be owner-only"
    config = json.loads(path.read_text())
    with httpx.Client(
        base_url=config["base_url"],
        auth=(config["username"], config["password"]),
        verify=False,
    ) as client:
        assert client.get("health").json()["ok"] is True
        for route in ("infra/provision", "infra/k8s/provision", "infra/mk8s/provision"):
            for field in ("dry_run", "validate", "skip_s3", "preemptible"):
                for value in ("false", "true", "", 0, 1, 0.0, None, [], {}):
                    response = client.post(route, json={field: value})
                    assert response.status_code == 400
                    assert response.json() == {
                        "ok": False,
                        "status": "invalid",
                        "error": f"{field} must be a literal boolean",
                    }
