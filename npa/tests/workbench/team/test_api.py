"""Verify the same authenticated ownership boundary across all run API operations."""

import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from npa.workbench.team.api import create_app
from npa.workbench.team.service import TeamService


@pytest.fixture
def application(config, tokens):
    policy = [config]
    release = threading.Event()

    def engine(*args, **kwargs):
        release.wait()
        return SimpleNamespace(status="succeeded")

    service = TeamService(
        lambda: policy[0], engine=engine, enrollment_check=lambda binding: None
    )
    app = create_app(None, service=service, verifier=tokens.verifier)
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client, service=service, policy=policy, release=release
        )
        release.set()
        workers = list(service._workers.values())
        for worker in workers:
            worker.join()


def _submit(application, tokens, workflow):
    response = application.client.post(
        "/v1/runs",
        headers={"Authorization": "Bearer " + tokens.sign()},
        json={
            "workspace": "robotics",
            "cluster": "east",
            "idempotency_key": "first",
            "workflow": workflow,
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["id"]


def test_no_proxy_header_or_body_identity_bypass(application, tokens, workflow):
    response = application.client.get(
        "/v1/runs?workspace=robotics", headers={"X-Auth-Request-Email": "alice"}
    )
    assert response.status_code == 401
    response = application.client.post(
        "/v1/runs",
        headers={"Authorization": "Bearer " + tokens.sign()},
        json={
            "workspace": "robotics",
            "cluster": "east",
            "idempotency_key": "key",
            "workflow": workflow,
            "subject": "administrator",
        },
    )
    assert response.status_code == 422
    assert "administrator" not in response.text


@pytest.mark.parametrize(
    "method,suffix",
    [
        ("GET", ""),
        ("GET", "/logs"),
        ("GET", "/artifacts"),
        ("GET", "/artifacts/output.txt"),
        ("POST", "/cancel"),
        ("POST", "/resume"),
    ],
)
def test_cross_user_run_operations_are_hidden(
    application, tokens, workflow, method, suffix
):
    run_id = _submit(application, tokens, workflow)
    response = application.client.request(
        method,
        f"/v1/runs/{run_id}{suffix}",
        headers={"Authorization": "Bearer " + tokens.sign("bob")},
    )
    assert response.status_code == 404
    assert response.json() == {"error": "run not found"}


def test_current_offboarding_applies_to_existing_tokens(application, tokens, workflow):
    run_id = _submit(application, tokens, workflow)
    application.policy[0] = application.policy[0].model_copy(
        update={"disabled_subjects": ("alice",)}
    )
    response = application.client.get(
        f"/v1/runs/{run_id}", headers={"Authorization": "Bearer " + tokens.sign()}
    )
    assert response.status_code == 403
    assert application.service.ledger.get(run_id)["status"] == "running"


def test_retries_return_same_run_and_actor_list_is_private(
    application, tokens, workflow
):
    assert _submit(application, tokens, workflow) == _submit(
        application, tokens, workflow
    )
    response = application.client.get(
        "/v1/runs?workspace=robotics",
        headers={"Authorization": "Bearer " + tokens.sign("bob")},
    )
    assert response.json() == {"runs": []}
