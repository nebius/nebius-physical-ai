"""Check authenticated progress from private runtime records, including unsafe files."""

import json

import pytest
from fastapi.testclient import TestClient

from npa.workbench.team.api import create_app
from npa.workbench.team.errors import RunNotFoundError
from npa.workbench.team.models import Actor, SubmitRequest
from npa.workbench.team.service import TeamService, binding_snapshot
from npa.workbench.team.session_storage import write_private_json


@pytest.fixture
def run(config, actor, binding, workflow):
    service = TeamService(lambda: config)
    record, _ = service.ledger.create(
        actor,
        SubmitRequest(
            workspace="robotics",
            cluster="east",
            idempotency_key="progress",
            workflow=workflow,
        ),
        binding_snapshot(binding),
    )
    service.ledger.transition(record["id"], ("accepted",), "running")
    root = config.state_dir / "runs" / record["id"] / "records" / "npa-workflow"
    return service, record["id"], root


def _record(root, run_id, **overrides):
    document = {
        "run_id": run_id,
        "waves": [
            {"states": ["prepare-input"], "status": "succeeded"},
            {
                "states": ["evaluate", "compare"],
                "status": "running",
                "job_id": "private-job",
            },
        ],
        "run_prefix_uri": "s3://synthetic-private-bucket/internal/",
        "error": "private worker details",
        **overrides,
    }
    write_private_json(root, "runtime.json", document)


def test_status_reports_last_persisted_parallel_stages_without_private_details(
    run, actor
):
    service, run_id, root = run
    _record(root, run_id)
    response = service.get(actor, run_id)
    assert response["status"] == "running"
    assert response["progress"] == {
        "states": ["evaluate", "compare"],
        "wave_status": "running",
        "completed_waves": 1,
        "recorded_waves": 2,
    }
    assert "private" not in json.dumps(response)


def test_other_owner_cannot_read_recorded_progress(run, config):
    service, run_id, root = run
    _record(root, run_id)
    foreign = Actor(
        issuer=config.identity.issuer, subject="bob", groups={"researchers"}
    )
    with pytest.raises(RunNotFoundError):
        service.get(foreign, run_id)


def test_authenticated_api_returns_progress_only_to_its_owner(run, tokens):
    service, run_id, root = run
    _record(root, run_id)
    with TestClient(
        create_app(None, service=service, verifier=tokens.verifier)
    ) as client:
        response = client.get(
            f"/v1/runs/{run_id}",
            headers={"Authorization": "Bearer " + tokens.sign()},
        )
        assert response.status_code == 200
        assert response.json()["progress"]["states"] == ["evaluate", "compare"]
        assert response.json()["status"] == service.ledger.get(run_id)["status"]
        assert client.get(f"/v1/runs/{run_id}").status_code == 401
        assert (
            client.get(
                f"/v1/runs/{run_id}",
                headers={"Authorization": "Bearer " + tokens.sign("bob")},
            ).status_code
            == 404
        )


@pytest.mark.parametrize(
    "defect",
    ["missing", "wrong-run", "malformed", "symlink", "public-file", "invalid-waves"],
)
def test_unavailable_progress_preserves_authoritative_status(run, actor, defect):
    service, run_id, root = run
    if defect != "missing":
        _record(root, "other-run" if defect == "wrong-run" else run_id)
        path = root / "runtime.json"
        if defect == "malformed":
            path.write_text("not-json")
        elif defect == "symlink":
            path.rename(root / "other.json")
            path.symlink_to(root / "other.json")
        elif defect == "public-file":
            path.chmod(0o644)
        elif defect == "invalid-waves":
            _record(root, run_id, waves=[None])
    response = service.get(actor, run_id)
    assert response["status"] == "running"
    assert "progress" not in response
