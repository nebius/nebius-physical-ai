"""Qualify fresh HTTPS deployment and retained-IP recovery on an isolated CPU namespace."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from npa.workbench.team.client import TeamClient
from npa.workbench.team.endpoint_manifests import OWNER_LABEL
from npa.workbench.team.setup import setup_control_plane
from npa.workbench.team.setup_models import read_setup_request
from npa.workbench.team.setup_transport import kubernetes, provider, resource

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(0)]


def _delete(request, kind, name):
    response = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            str(request.kubeconfig),
            "--context",
            request.context,
            "delete",
            kind,
            name,
            "--namespace",
            request.namespace,
            "--wait=true",
        ],
        capture_output=True,
        text=True,
    )
    assert response.returncode == 0, "owned test resource deletion failed"


def _cleanup(request, path):
    if not path.exists():
        return
    receipt = json.loads(path.read_text())
    for kind, name in (
        ("service", request.service_name),
        ("deployment", "npa-team"),
        ("configmap", "npa-team-https"),
        ("networkpolicy", "npa-team"),
    ):
        current = resource(request, kind, name)
        if current:
            assert (
                current["metadata"].get("labels", {}).get(OWNER_LABEL)
                == receipt["installation_id"]
            )
            _delete(request, kind, name)
    if receipt.get("allocation_id"):
        allocation = provider(
            "vpc", "allocation", "get", "--id", receipt["allocation_id"]
        )
        assert allocation["metadata"]["parent_id"] == request.project_id
        assert (
            allocation["status"]["details"]["allocated_cidr"]
            == receipt["address"] + "/32"
        )
        assert not allocation["metadata"].get("labels", {}).get("nebius.com/managed-by")
        provider("vpc", "allocation", "delete", "--id", receipt["allocation_id"])


@pytest.fixture
def fresh_installation():
    if (
        os.getenv("NPA_TEAM_SETUP_FRESH_LIVE") != "1"
        or os.getenv("NPA_INTEGRATION_E2E") != "1"
    ):
        pytest.skip("requires an explicitly selected disposable CPU installation")
    request = read_setup_request(Path(os.environ["NPA_TEAM_SETUP_FRESH_INPUT_PATH"]))
    path = Path(os.environ["NPA_TEAM_SETUP_FRESH_RECEIPT_PATH"])
    assert not path.exists(), "use a new private test receipt"
    assert not request.existing_service_uid and not request.existing_deployment_uid
    namespace = kubernetes(request, "get", "namespace", request.namespace, "-o", "json")
    assert (
        namespace["metadata"]
        .get("labels", {})
        .get("npa.nebius.ai/team-setup-live-test")
        == "true"
    )
    for kind, name in (("deployment", "npa-team"), ("service", request.service_name)):
        assert resource(request, kind, name) is None, (
            "fresh test cannot target an existing server"
        )
    try:
        yield request, path
    finally:
        _cleanup(request, path)


def _verified_setup(request, path):
    result = setup_control_plane(request, path)
    assert result["external_https_verified"] is True
    assert result["personal_client_required"] is False
    assert result["status"] in {"ready", "endpoint-awaiting-dns"}
    return json.loads(path.read_text())


def test_fresh_cpu_https_and_same_ip_after_service_recreation(
    fresh_installation,
    monkeypatch,
):
    request, path = fresh_installation

    def personal_client_forbidden(*args, **kwargs):
        pytest.fail("server setup constructed a personal Workbench client")

    monkeypatch.setattr(TeamClient, "__init__", personal_client_forbidden)
    first = _verified_setup(request, path)
    deployment = resource(request, "deployment", "npa-team")
    claim = resource(request, "persistentvolumeclaim", request.state_claim)
    service = resource(request, "service", request.service_name)
    retry = _verified_setup(request, path)
    assert retry["service_uid"] == first["service_uid"]
    assert retry["allocation_id"] == first["allocation_id"]
    assert service["metadata"]["labels"][OWNER_LABEL] == first["installation_id"]
    _delete(request, "service", request.service_name)
    recovered = _verified_setup(request, path)
    assert recovered["service_uid"] != first["service_uid"]
    assert recovered["allocation_id"] == first["allocation_id"]
    assert recovered["address"] == first["address"]
    assert (
        resource(request, "deployment", "npa-team")["metadata"]["uid"]
        == deployment["metadata"]["uid"]
    )
    assert (
        resource(request, "persistentvolumeclaim", request.state_claim)["metadata"][
            "uid"
        ]
        == claim["metadata"]["uid"]
    )
