"""Live lifecycle evidence must reject stale, foreign, and mismatched bindings."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def live():
    path = Path(__file__).parents[1] / "e2e/test_mk8s_provider_rpc_live.py"
    spec = importlib.util.spec_from_file_location("rpc_live_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lifecycle_documents(source):
    return {
        "provision_start": {
            "source_revision": source,
            "started_at": "attempt-one",
            "argv": [
                "npa",
                "cluster",
                "up",
                "--context",
                "owned-context",
                "--project",
                "owned",
            ],
        },
        "provision_result": {"started_at": "attempt-one", "exit": 0},
        "deployment_sidecar": {
            "project_id": "project-test",
            "cluster_name": "owned-cluster",
            "status": "deployed",
        },
        "terraform_state": {
            "resources": [
                {"type": kind, "instances": [{"attributes": {"id": identity}}]}
                for kind, identity in [
                    ("nebius_mk8s_v1_cluster", "cluster-test"),
                    ("nebius_mk8s_v1_node_group", "group-test"),
                ]
            ]
        },
    }


@pytest.fixture
def binding(tmp_path):
    source = "a" * 40
    config = {
        "source_revision": source,
        "context": "owned-context",
        "project_alias": "owned",
        "project_id": "project-test",
        "cluster_id": "cluster-test",
        "cluster_name": "owned-cluster",
        "node_group_ids": ["group-test"],
    }
    documents = _lifecycle_documents(source)
    for name, document in documents.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(document))
        config[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return config


def _change(config, field, update):
    path = Path(config[field]["path"])
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))
    config[field]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_lifecycle_binding_accepts_consistent_inputs(live, binding):
    assert live._bindings(binding, "a" * 40)["status"] == "deployed"


@pytest.mark.parametrize("wrong_source", ["b" * 40, ""])
def test_source_identity_mismatch_fails(live, binding, wrong_source):
    with pytest.raises(AssertionError):
        live._bindings(binding, wrong_source)


@pytest.mark.parametrize(
    "field,value", [("exit", 130), ("started_at", "different-attempt")]
)
def test_failed_or_other_producer_result_fails(live, binding, field, value):
    _change(binding, "provision_result", lambda doc: doc.update({field: value}))
    with pytest.raises(AssertionError):
        live._bindings(binding, "a" * 40)


@pytest.mark.parametrize(
    "field,value",
    [
        ("project_id", "foreign-project"),
        ("cluster_name", "foreign-cluster"),
        ("status", "provisioning"),
    ],
)
def test_foreign_or_incomplete_sidecar_fails(live, binding, field, value):
    _change(binding, "deployment_sidecar", lambda doc: doc.update({field: value}))
    with pytest.raises(AssertionError):
        live._bindings(binding, "a" * 40)


def test_changed_input_bytes_fail(live, binding):
    Path(binding["deployment_sidecar"]["path"]).write_text("{}")
    with pytest.raises(AssertionError, match="binding changed"):
        live._bindings(binding, "a" * 40)


@pytest.mark.parametrize(
    "parent,state", [("foreign", "RUNNING"), ("project-test", "DELETING")]
)
def test_live_identity_rejects_foreign_parent_and_stale_state(live, parent, state):
    response = {
        "metadata": {
            "id": "cluster-test",
            "parent_id": parent,
            "name": "owned-cluster",
        },
        "status": {"state": state},
    }
    with pytest.raises(AssertionError):
        live._check_live_identity(
            response, "cluster-test", "project-test", "owned-cluster"
        )


@pytest.mark.parametrize(
    "code,stdout,stderr",
    [
        (0, "{}", ""),
        (15, "", "code = PermissionDenied"),
        (13, "", "code = Unavailable"),
        (13, "{}", "code = NotFound"),
    ],
)
def test_absence_refuses_live_denied_and_unknown(live, code, stdout, stderr):
    with pytest.raises(AssertionError):
        live._check_absence(
            SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)
        )


def test_absence_requires_typed_not_found(live):
    live._check_absence(
        SimpleNamespace(
            returncode=13,
            stdout="",
            stderr="rpc error: code = NotFound desc = resource missing",
        )
    )
