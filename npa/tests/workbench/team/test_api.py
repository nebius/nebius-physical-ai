"""Verify the same authenticated ownership boundary across all run API operations."""

import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from npa.workbench.team.api import create_app
from npa.workbench.team.service import TeamService
from npa.workbench.team.models import Grant
from npa.workbench.team.errors import AuthorizationError


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


def _submit(application, tokens, workflow, key="first"):
    response = application.client.post(
        "/v1/runs",
        headers={"Authorization": "Bearer " + tokens.sign()},
        json={
            "workspace": "robotics",
            "cluster": "east",
            "idempotency_key": key,
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
    "method,path",
    [
        ("GET", "/"),
        ("GET", "/portal.js"),
        ("GET", "/portal.css"),
        ("GET", "/auth/login"),
        ("GET", "/auth/callback"),
        ("GET", "/auth/methods"),
        ("POST", "/auth/key"),
        ("POST", "/auth/logout"),
    ],
)
def test_removed_portal_routes_cannot_authenticate(application, method, path):
    response = application.client.request(
        method,
        path,
        headers={"Cookie": "__Host-workbench-session=forged"},
    )
    assert response.status_code == 404
    assert "set-cookie" not in response.headers


def test_api_keeps_private_response_headers_without_cookie_auth(application, tokens):
    response = application.client.get(
        "/v1/me",
        headers={
            "Authorization": "Bearer " + tokens.sign(),
            "Cookie": "__Host-workbench-session=forged",
        },
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["content-security-policy"] == (
        "default-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'none'"
    )
    denied = application.client.get(
        "/v1/me", headers={"Cookie": "__Host-workbench-session=forged"}
    )
    assert denied.status_code == 401
    assert denied.headers["cache-control"] == "no-store"
    assert "set-cookie" not in denied.headers


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


def test_downgraded_reader_can_read_owned_outputs_but_cannot_mutate(
    application, tokens, workflow
):
    run_id = _submit(application, tokens, workflow)
    config = application.policy[0]
    workspace = config.workspaces["robotics"].model_copy(
        update={"grants": (Grant(kind="group", value="researchers", role="reader"),)}
    )
    application.policy[0] = config.model_copy(
        update={"workspaces": {"robotics": workspace}}
    )
    application.service.backend_factory = lambda *args: SimpleNamespace(
        logs=lambda: "owned log"
    )
    application.service.storage_factory = lambda grant: SimpleNamespace(
        list=lambda prefix: [{"key": "owned artifact"}]
    )
    headers = {"Authorization": "Bearer " + tokens.sign()}
    for suffix in ("", "/logs", "/artifacts"):
        response = application.client.get(f"/v1/runs/{run_id}{suffix}", headers=headers)
        assert response.status_code == 200, response.text
    for action in ("cancel", "resume"):
        response = application.client.post(
            f"/v1/runs/{run_id}/{action}", headers=headers
        )
        assert response.status_code == 403
    with pytest.raises(AuthorizationError, match="not authorized"):
        application.service._check_running(
            tokens.verifier.verify(headers["Authorization"]),
            application.service.ledger.get(run_id),
        )


def _set_runtime_source(application, source):
    config = application.policy[0]
    values = config.model_dump()
    values["workspaces"]["robotics"]["allocations"][0]["source_s3_uri"] = source
    application.policy[0] = type(config).model_validate(values)


@pytest.mark.parametrize(
    "original,updated",
    [
        ("", "s3://team-test-alice/personal/source/new/npa"),
        ("s3://team-test-alice/personal/source/old/npa", ""),
        (
            "s3://team-test-alice/personal/source/old/npa",
            "s3://team-test-alice/personal/source/new/npa",
        ),
    ],
)
def test_source_updates_retain_owned_outputs_retries_and_run_versions(
    application, tokens, workflow, original, updated
):
    _set_runtime_source(application, original)
    run_id = _submit(application, tokens, workflow)
    _set_runtime_source(application, updated)
    application.service.backend_factory = lambda config, binding, *args: (
        SimpleNamespace(logs=lambda: binding.allocation.source_s3_uri)
    )
    application.service.storage_factory = lambda grant: SimpleNamespace(
        list=lambda prefix: [{"key": prefix + "/output.txt"}]
    )
    headers = {"Authorization": "Bearer " + tokens.sign()}
    logs = application.client.get(f"/v1/runs/{run_id}/logs", headers=headers)
    assert logs.status_code == 200 and logs.json() == {"logs": original}
    artifacts = application.client.get(f"/v1/runs/{run_id}/artifacts", headers=headers)
    assert artifacts.status_code == 200
    assert artifacts.json()["artifacts"] == [
        {"key": f"personal/runs/{run_id}/output.txt"}
    ]
    assert _submit(application, tokens, workflow) == run_id
    actor = tokens.verifier.verify(headers["Authorization"])
    application.service._check_running(actor, application.service.ledger.get(run_id))
    new_id = _submit(application, tokens, workflow, key="new-version")
    assert new_id != run_id
    for selected, expected in [(run_id, original), (new_id, updated)]:
        recorded = json.loads(application.service.ledger.get(selected)["binding"])
        assert recorded.get("source_s3_uri", "") == expected


def test_resuming_after_source_update_keeps_original_package(
    application, tokens, workflow
):
    original = "s3://team-test-alice/personal/source/old/npa"
    _set_runtime_source(application, original)

    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated interrupted worker")

    application.service.engine = interrupted
    run_id = _submit(application, tokens, workflow)
    for worker in list(application.service._workers.values()):
        worker.join()
    assert application.service.ledger.get(run_id)["status"] == "recovery_required"
    _set_runtime_source(application, "s3://team-test-alice/personal/source/new/npa")
    observed = []

    def resumed(config, binding, record, backend, check, *, resume):
        observed.append((binding.allocation.source_s3_uri, resume))
        return SimpleNamespace(status="succeeded")

    application.service.engine = resumed
    response = application.client.post(
        f"/v1/runs/{run_id}/resume",
        headers={"Authorization": "Bearer " + tokens.sign()},
    )
    assert response.status_code == 202, response.text
    for worker in list(application.service._workers.values()):
        worker.join()
    assert observed == [(original, True)]


@pytest.mark.parametrize(
    "boundary", ["bucket", "prefix", "principal", "inputs", "cluster"]
)
def test_source_rotation_still_rejects_changed_execution_boundaries(
    application, tokens, workflow, boundary
):
    _set_runtime_source(application, "s3://team-test-alice/personal/source/old/npa")
    run_id = _submit(application, tokens, workflow)
    config = application.policy[0]
    values = config.model_dump()
    allocation = values["workspaces"]["robotics"]["allocations"][0]
    if boundary == "inputs":
        allocation["shared_inputs"] = ["s3://shared-test/reference/"]
    elif boundary == "cluster":
        values["clusters"]["east"]["context"] = "reallocated"
    else:
        allocation["storage"][boundary] = "reallocated"
    storage = allocation["storage"]
    allocation["source_s3_uri"] = (
        f"s3://{storage['bucket']}/{storage['prefix']}/source/new/npa"
    )
    application.policy[0] = type(config).model_validate(values)
    headers = {"Authorization": "Bearer " + tokens.sign()}
    for suffix in ("/logs", "/artifacts"):
        response = application.client.get(f"/v1/runs/{run_id}{suffix}", headers=headers)
        assert response.status_code == 409, response.text


def test_source_rotation_keeps_offboarding_and_foreign_owner_denial(
    application, tokens, workflow
):
    run_id = _submit(application, tokens, workflow)
    _set_runtime_source(application, "s3://team-test-alice/personal/source/new/npa")
    config = application.policy[0]
    application.policy[0] = config.model_copy(update={"disabled_subjects": ("alice",)})
    for subject, expected in [("alice", 403), ("bob", 404)]:
        for suffix in ("/logs", "/artifacts"):
            response = application.client.get(
                f"/v1/runs/{run_id}{suffix}",
                headers={"Authorization": "Bearer " + tokens.sign(subject)},
            )
            assert response.status_code == expected


def test_recorded_source_cannot_use_a_revoked_shared_input(
    application, tokens, workflow
):
    config = application.policy[0]
    values = config.model_dump()
    allocation = values["workspaces"]["robotics"]["allocations"][0]
    allocation["shared_inputs"] = ["s3://shared-test/source/"]
    allocation["source_s3_uri"] = "s3://shared-test/source/old/npa"
    application.policy[0] = type(config).model_validate(values)
    run_id = _submit(application, tokens, workflow)
    allocation["shared_inputs"] = []
    allocation["source_s3_uri"] = "s3://team-test-alice/personal/source/new/npa"
    application.policy[0] = type(config).model_validate(values)
    response = application.client.get(
        f"/v1/runs/{run_id}/artifacts",
        headers={"Authorization": "Bearer " + tokens.sign()},
    )
    assert response.status_code == 409
    assert response.json() == {
        "error": "recorded runtime source no longer uses an allocated input scope"
    }
