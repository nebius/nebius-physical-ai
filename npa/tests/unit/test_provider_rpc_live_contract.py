"""Live lifecycle evidence must reject stale, foreign, and mismatched bindings."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import copy

import pytest
import yaml


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


def _authority_fixture(tmp_path):
    credentials = tmp_path / "credentials.json"
    credentials.write_text('{"test-only": "no credential material"}')
    profile_file = tmp_path / "config.yaml"
    profile_file.write_text(
        yaml.safe_dump(
            {
                "profiles": {
                    "owned-profile": {
                        "endpoint": "api.example.invalid",
                        "auth-type": "service account",
                        "parent-id": "project-test",
                        "tenant-id": "tenant-test",
                        "service-account-credentials-file-path": str(credentials),
                    }
                }
            }
        )
    )
    authority = {
        "profile": "owned-profile",
        "project_id": "project-test",
        "tenant_id": "tenant-test",
        "endpoint": "api.example.invalid",
    }
    for name, path in (
        ("config_file", profile_file),
        ("credentials_file", credentials),
    ):
        authority[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return authority


@pytest.fixture
def binding(tmp_path):
    source = "a" * 40
    authority = _authority_fixture(tmp_path)
    config = {
        "source_revision": source,
        "context": "owned-context",
        "project_alias": "owned",
        "project_id": "project-test",
        "tenant_id": "tenant-test",
        "profile": "owned-profile",
        "cluster_id": "cluster-test",
        "cluster_name": "owned-cluster",
        "node_group_ids": ["group-test"],
        "authority": authority,
    }
    documents = _lifecycle_documents(source)
    documents["provision_start"]["authority"] = copy.deepcopy(authority)
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


@pytest.mark.parametrize(
    "key,value",
    [
        ("profile", "another-profile"),
        ("endpoint", "other.example.invalid"),
        ("project_id", "foreign-project"),
        ("tenant_id", "foreign-tenant"),
    ],
)
def test_authority_cannot_be_replaced_after_provision(live, binding, key, value):
    binding["authority"][key] = value
    with pytest.raises(AssertionError, match="producing authority changed"):
        live._bindings(binding, "a" * 40)


@pytest.mark.parametrize("field", ["config_file", "credentials_file"])
def test_changed_authority_file_is_rejected(live, binding, field):
    Path(binding["authority"][field]["path"]).write_text("changed")
    with pytest.raises(AssertionError, match="input binding changed"):
        live._bindings(binding, "a" * 40)


def test_legacy_producer_without_authority_is_rejected(live, binding):
    _change(binding, "provision_start", lambda doc: doc.pop("authority"))
    with pytest.raises(KeyError):
        live._bindings(binding, "a" * 40)


def test_profile_selector_cannot_change_independently(live, binding):
    binding["profile"] = "foreign-profile"
    with pytest.raises(AssertionError):
        live._bindings(binding, "a" * 40)


def test_provider_subprocess_binds_authority_and_scrubs_selectors(
    live,
    binding,
    monkeypatch,
    tmp_path,
):
    for key in (
        "NEBIUS_ENDPOINT",
        "NEBIUS_CONFIG",
        "NEBIUS_PROFILE",
        "NEBIUS_IAM_TOKEN",
        "NPA_NEBIUS_IAM_TOKEN_FILE",
        "NEBIUS_IMPERSONATE_SERVICE_ACCOUNT_ID",
        "NPA_REUSE_IAM_TOKEN",
    ):
        monkeypatch.setenv(key, "hostile-ambient-selector")
    observed = []

    def run(argv, **kwargs):
        observed.append((argv, kwargs["env"]))
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(live.subprocess, "run", run)
    live._provider(binding, tmp_path, "bound", ["iam", "project", "get"])
    argv, env = observed[0]
    assert argv[1:5] == [
        "--config",
        binding["authority"]["config_file"]["path"],
        "--profile",
        "owned-profile",
    ]
    assert not any(value == "hostile-ambient-selector" for value in env.values())
    assert env["NEBIUS_CONFIG_DIR"] == str(tmp_path)
    assert env["NEBIUS_PROFILE"] == "owned-profile"


@pytest.mark.parametrize(
    "code,project,tenant,state",
    [
        (1, "project-test", "tenant-test", "ACTIVE"),
        (0, "foreign-project", "tenant-test", "ACTIVE"),
        (0, "project-test", "foreign-tenant", "ACTIVE"),
        (0, "project-test", "tenant-test", "DELETING"),
    ],
)
def test_project_authority_required_before_absence(
    live,
    binding,
    monkeypatch,
    tmp_path,
    code,
    project,
    tenant,
    state,
):
    response = {
        "metadata": {"id": project, "parent_id": tenant},
        "status": {"state": state},
    }
    monkeypatch.setattr(
        live,
        "_provider",
        lambda *args: SimpleNamespace(returncode=code, stdout=json.dumps(response)),
    )
    with pytest.raises(AssertionError):
        live._verify_project_authority(binding, tmp_path)


def test_cleanup_checks_authority_before_any_absence(
    live, binding, monkeypatch, tmp_path
):
    monkeypatch.setattr(live, "_configured", lambda phase: (binding, {}, tmp_path))
    calls = []

    def provider(config, evidence, label, args):
        calls.append(label)
        return SimpleNamespace(
            returncode=1, stdout="", stderr="code = PermissionDenied"
        )

    monkeypatch.setattr(live, "_provider", provider)
    with pytest.raises(AssertionError, match="project authority"):
        live.test_mk8s_provider_rpc_live_cleanup()
    assert calls == ["project-authority"]
