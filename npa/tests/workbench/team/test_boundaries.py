"""Exercise forged identities, quota multiplication, and execution boundary escapes."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
import yaml
from pydantic import ValidationError

from npa.workbench.team.authorization import authorize, bind_execution
from npa.workbench.team.deployment import server_config
from npa.workbench.team.errors import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    TeamError,
)
from npa.workbench.team.ledger import TeamLedger
from npa.workbench.team.manifests import execution_manifests
from npa.workbench.team.models import SubmitRequest, TeamConfig
from npa.workbench.team.storage import authorize_uri, object_key, storage_credentials
from npa.workbench.team.workflow_policy import (
    enforce_rendered_tasks,
    load_bound_spec,
    prepare_document,
    worker_context,
)


def test_real_signatures_and_verified_groups(tokens, actor):
    assert tokens.verifier.verify("Bearer " + tokens.sign()) == actor
    forged = jwt.encode(
        {"sub": "alice", "groups": ["researchers"]},
        "forged-key-for-unsupported-algorithm",
        algorithm="HS256",
    )
    with pytest.raises(AuthenticationError):
        tokens.verifier.verify("Bearer " + forged)


@pytest.mark.parametrize(
    "claims",
    [
        {"iss": "https://impostor.example.test"},
        {"aud": "different-product"},
        {"exp": 1},
        {"iat": 9999999999},
        {"groups": "researchers"},
        {"groups": ["researchers", 4]},
        {"sub": ""},
    ],
)
def test_reject_invalid_claims(tokens, claims):
    with pytest.raises(AuthenticationError):
        tokens.verifier.verify("Bearer " + tokens.sign(**claims))


def test_membership_and_allocations_are_separate(config, actor):
    newcomer = actor.model_copy(update={"subject": "newcomer"})
    assert authorize(config, newcomer, "robotics", "runner")
    with pytest.raises(AuthorizationError):
        bind_execution(config, newcomer, "robotics", "east")
    stranger = actor.model_copy(update={"groups": frozenset()})
    with pytest.raises(AuthorizationError):
        bind_execution(config, stranger, "robotics", "east")


def test_no_implicit_cross_project_membership(config, actor):
    document = config.model_dump(mode="json")
    document["workspaces"]["other"] = {
        "grants": [],
        "gpu_limit": 0,
        "gpu_limits": {},
        "allocations": [],
    }
    with pytest.raises(AuthorizationError):
        authorize(TeamConfig.model_validate(document), actor, "other", "reader")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ws: ws.update(gpu_limit=3),
        lambda ws: ws["gpu_limits"].update(east=1),
        lambda ws: ws["allocations"][0].update(gpu_limit=1),
        lambda ws: ws["allocations"][0]["clusters"].update(east=3),
        lambda ws: ws["allocations"][0]["clusters"].update(east=True),
        lambda ws: ws["allocations"][1]["storage"].update(principal="principal-alice"),
        lambda ws: ws["allocations"][1]["storage"].update(bucket="team-test-alice"),
    ],
)
def test_aggregate_quota_and_storage_boundaries(config, mutation):
    document = config.model_dump(mode="json")
    mutation(document["workspaces"]["robotics"])
    with pytest.raises(ValidationError):
        TeamConfig.model_validate(document)


def test_workflow_binding_and_stage_override(binding, workflow, tmp_path):
    spec, document = load_bound_spec(workflow, binding, "run-test", tmp_path)
    assert document["resources"]["cpu"]["region"] == worker_context(binding)
    assert document["config"]["bucket"] == "team-test-alice"
    assert "region" not in workflow["resources"]["cpu"]
    workflow["metadata"]["namespace"] = "default"
    with pytest.raises(AuthorizationError):
        prepare_document(workflow, binding, "run-test")


@pytest.mark.parametrize(
    "resources",
    [
        {"region": "other-cluster"},
        {"cloud": "aws"},
        {"kubernetes": {"pod_config": {"spec": {"serviceAccountName": "admin"}}}},
    ],
)
def test_placement_escape_rejected(binding, workflow, resources):
    workflow["resources"]["cpu"].update(resources)
    with pytest.raises(AuthorizationError):
        prepare_document(workflow, binding, "run-test")


def test_rendered_task_cannot_upload_server_files(binding, tmp_path):
    path = tmp_path / "wave.yaml"
    path.write_text(
        yaml.safe_dump(
            {"resources": {}, "file_mounts": {str(tmp_path / "key"): "/etc/private"}}
        )
    )
    with pytest.raises(AuthorizationError):
        enforce_rendered_tasks(path, binding, {})


@pytest.mark.parametrize(
    "extra",
    [
        {"name": "task'$(id)"},
        {"name": "task`id`"},
        {"name": "../../server"},
        {"api_server_access": True},
        {"pool": "shared-pool"},
        {"event_callback": "touch /tmp/control-plane"},
        {"file_mounts_blob_id": "unexpected-upload"},
        {"envs": {"VALUE;id": "x"}},
    ],
)
def test_scheduler_metadata_cannot_expand_server_execution(binding, tmp_path, extra):
    path = tmp_path / "wave.yaml"
    path.write_text(yaml.safe_dump({"name": "verify", "resources": {}, **extra}))
    with pytest.raises(AuthorizationError):
        enforce_rendered_tasks(path, binding, {})


def test_worker_identity_has_no_rbac_writes(binding):
    documents = execution_manifests(binding)
    quotas = [item for item in documents if item["kind"] == "ResourceQuota"]
    assert sorted(
        item["spec"]["hard"]["requests.nvidia.com/gpu"] for item in quotas
    ) == ["1"]
    for role in (item for item in documents if item["kind"] in {"Role", "ClusterRole"}):
        assert all(
            "rbac.authorization.k8s.io" not in rule["apiGroups"]
            for rule in role["rules"]
        )
    worker = next(
        item
        for item in documents
        if item["kind"] == "ServiceAccount"
        and item["metadata"]["namespace"] == binding.namespace
    )
    assert worker["automountServiceAccountToken"] is False


@pytest.mark.parametrize(
    "path",
    ["../secret", "/absolute", "x/../secret", "x//file", "x%2fsecret", "x\\secret"],
)
def test_artifact_path_escape(path):
    with pytest.raises(TeamError):
        object_key("personal/run", path)


def test_explicit_storage_principal_and_scope(binding, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "administrator")
    assert (
        storage_credentials(binding.allocation.storage)["aws_access_key_id"]
        == "test-alice"
    )
    with pytest.raises(AuthorizationError):
        authorize_uri("s3://team-test-bob/personal/file", binding.allocation.storage)
    with pytest.raises(AuthorizationError):
        authorize_uri(
            "s3://team-test-alice/personally/file", binding.allocation.storage
        )


def test_concurrent_idempotency_and_missing_launch_ack(config, actor, workflow):
    ledger = TeamLedger(config.state_dir)
    request = SubmitRequest(
        workspace="robotics", cluster="east", idempotency_key="same", workflow=workflow
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: ledger.create(actor, request, {}), range(16)))
    assert len({record["id"] for record, _ in results}) == 1
    assert sum(created for _, created in results) == 1
    record = results[0][0]
    assert ledger.transition(record["id"], ("accepted",), "running")
    ledger.begin_wave(record["id"], "wave-one")
    with pytest.raises(ConflictError):
        ledger.begin_wave(record["id"], "wave-one")
    ledger.recover()
    assert ledger.get(record["id"])["status"] == "recovery_required"
    changed = request.model_copy(update={"cluster": "west"})
    with pytest.raises(ConflictError):
        ledger.create(actor, changed, {})


def test_worker_configuration_never_contains_operator_credentials(config):
    rendered = json.dumps(server_config(config))
    assert "LOCAL_CREDENTIALS" not in rendered
    assert "NO_UPLOAD" not in rendered
    assert "test-alice" not in rendered
    assert "npa-team-controller" not in rendered and "npa-team-worker" in rendered


def test_concurrent_run_transitions_have_one_winner(config, actor, workflow):
    ledger = TeamLedger(config.state_dir)
    request = SubmitRequest(
        workspace="robotics", cluster="east", idempotency_key="race", workflow=workflow
    )
    record, _ = ledger.create(actor, request, {})
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: ledger.transition(record["id"], ("accepted",), "running"),
                range(16),
            )
        )
    assert sum(results) == 1
    assert ledger.get(record["id"])["status"] == "running"
    assert not ledger.transition(record["id"], (), "cancelled")
    assert not ledger.transition("missing", ("running",), "cancelled")


def test_ledger_upgrade_adds_safe_failure_columns_idempotently(tmp_path):
    root = tmp_path / "state"
    root.mkdir()
    path = root / "team.sqlite3"
    with sqlite3.connect(path) as database:
        database.execute(
            "CREATE TABLE runs (id TEXT PRIMARY KEY, status TEXT NOT NULL)"
        )

    TeamLedger(root)
    TeamLedger(root)

    with sqlite3.connect(path) as database:
        columns = {row[1] for row in database.execute("PRAGMA table_info(runs)")}
    assert {"failure_code", "failure_message"} <= columns
