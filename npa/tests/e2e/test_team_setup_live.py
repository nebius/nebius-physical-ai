"""Qualify existing shared setup and retries with real operator transports only."""

import json
import os
from pathlib import Path

import pytest

from npa.workbench.team.client import TeamClient
from npa.workbench.team.setup import setup_control_plane
from npa.workbench.team.setup_models import read_setup_request
from npa.workbench.team.setup_transport import kubernetes, resource

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(0),
    pytest.mark.skipif(
        os.getenv("NPA_TEAM_SETUP_LIVE") != "1"
        or os.getenv("NPA_INTEGRATION_E2E") != "1",
        reason="requires an explicitly selected existing private HTTPS installation",
    ),
]


def _snapshot(request):
    deployment = resource(request, "deployment", "npa-team")
    claim = resource(request, "persistentvolumeclaim", request.state_claim)
    services = kubernetes(
        request, "get", "services", "--namespace", request.namespace, "-o", "json"
    )["items"]
    return {
        "deployment": (deployment["metadata"]["uid"], deployment["spec"]),
        "claim": (claim["metadata"]["uid"], claim["spec"]),
        "services": {
            item["metadata"]["uid"]: (
                item["spec"],
                item.get("status", {}),
                item["metadata"].get("annotations", {}),
            )
            for item in services
        },
    }


def test_existing_setup_and_retry_preserve_live_server_without_personal_client(
    monkeypatch,
):
    request = read_setup_request(Path(os.environ["NPA_TEAM_SETUP_INPUT_PATH"]))
    assert request.existing_service_uid, "select the exact existing public Service UID"
    receipt = Path(os.environ["NPA_TEAM_SETUP_RECEIPT_PATH"])

    def client_forbidden(*args, **kwargs):
        pytest.fail("infrastructure setup constructed a personal Workbench client")

    monkeypatch.setattr(TeamClient, "__init__", client_forbidden)
    before = _snapshot(request)
    for _ in range(2):
        result = setup_control_plane(request, receipt)
        assert result["status"] == "ready"
        assert result["external_https_verified"] is True
        assert result["personal_client_required"] is False
    assert _snapshot(request) == before
    saved = json.loads(receipt.read_text())
    assert saved["service_uid"] == request.existing_service_uid
    assert saved["deployment_preserved"] is True
    assert saved["boundary_statuses"] == {
        "/health": 200,
        "/v1/me": 401,
        "/api/health": 404,
        "/api/status": 404,
        "/status": 404,
    }
