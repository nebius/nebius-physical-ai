"""Explicit external backend ownership must survive apply and remain read-only."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from npa.clients import config, nebius, terraform_storage
from npa.clients.config import EnvironmentConfig, TerraformStateConfig


@pytest.fixture
def binding(monkeypatch):
    saved = TerraformStateConfig(
        bucket="example-bucket",
        endpoint="https://storage.example.invalid",
        owner_project_id="project-storage",
        bucket_id="bucket-resource",
    )
    selected = EnvironmentConfig("project-compute", "tenant-test", "region-test")
    monkeypatch.setattr(terraform_storage, "resolve_terraform_state", lambda _: saved)
    monkeypatch.setattr(terraform_storage, "resolve_environment", lambda _: selected)
    calls = []
    responses = {
        "project-compute": {
            "metadata": {"id": "project-compute", "parent_id": "tenant-test"}
        },
        "project-storage": {
            "metadata": {"id": "project-storage", "parent_id": "tenant-test"}
        },
        "bucket-resource": {
            "metadata": {
                "id": "bucket-resource",
                "parent_id": "project-storage",
                "name": "example-bucket",
            }
        },
    }

    def run(args):
        calls.append(args)
        assert args[2] == "get", (
            "No provider mutation is authorized by storage selection"
        )
        return responses[args[-1]]

    monkeypatch.setattr(nebius, "_run_json", run)
    return saved, selected, calls, responses


def verify():
    return terraform_storage.verify_external_backend(
        project_alias="compute",
        project_id="project-compute",
        bucket_name="example-bucket",
        endpoint="https://storage.example.invalid",
    )


def test_external_binding_requires_exact_same_tenant_provider_proof(binding):
    assert verify() == {
        "owner_project_id": "project-storage",
        "bucket_id": "bucket-resource",
    }
    assert [call[-1] for call in binding[2]] == [
        "project-compute",
        "project-storage",
        "bucket-resource",
    ]


def test_binding_verification_does_not_migrate_credential_configuration(
    binding, monkeypatch
):
    from npa.lifecycle_intent import OperationIntent, current_intent

    def observe(_alias):
        assert current_intent() == OperationIntent.OBSERVE
        return binding[0]

    monkeypatch.setattr(terraform_storage, "resolve_terraform_state", observe)
    assert verify()["owner_project_id"] == "project-storage"


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_project_id", ""),
        ("bucket_id", ""),
        ("bucket", "different-bucket"),
        ("endpoint", "https://different.example.invalid"),
    ],
)
def test_incomplete_or_retargeted_binding_refuses_before_provider_io(
    binding, field, value
):
    setattr(binding[0], field, value)
    with pytest.raises(nebius.NebiusError, match="binding"):
        verify()
    assert binding[2] == []


@pytest.mark.parametrize(
    "resource,field,value",
    [
        ("project-compute", "id", "wrong-project"),
        ("project-compute", "parent_id", "wrong-tenant"),
        ("project-storage", "parent_id", "wrong-tenant"),
        ("bucket-resource", "id", "wrong-bucket"),
        ("bucket-resource", "parent_id", "wrong-owner"),
        ("bucket-resource", "name", "different-bucket"),
    ],
)
def test_provider_identity_substitution_refuses(binding, resource, field, value):
    binding[3][resource]["metadata"][field] = value
    with pytest.raises(nebius.NebiusError):
        verify()


@pytest.mark.parametrize(
    "identity", ["project-compute", "project-storage", "bucket-resource"]
)
def test_malformed_provider_metadata_refuses(binding, identity):
    binding[3][identity]["metadata"] = []
    with pytest.raises(nebius.NebiusError):
        verify()


def test_ordinary_backend_does_not_discover_or_adopt_external_storage(binding):
    binding[0].owner_project_id = binding[0].bucket_id = ""
    assert verify() is None
    assert binding[2] == []


def test_verified_external_binding_still_probes_exact_terraform_state(
    binding, monkeypatch
):
    from npa.cli import agent_terraform
    from npa.clients import storage_validation

    monkeypatch.setattr(
        nebius, "bucket_exists", lambda *_: pytest.fail("wrong owner lookup")
    )
    monkeypatch.setattr(agent_terraform, "current_operation", lambda: None)
    calls = []

    def probe(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(ok=False, summary="write refused")

    monkeypatch.setattr(storage_validation, "probe_terraform_backend", probe)
    with pytest.raises(nebius.NebiusError, match="write refused"):
        agent_terraform._ensure_terraform_state_bucket(
            project_alias="compute",
            project_id="project-compute",
            bucket_name="example-bucket",
            endpoint="https://storage.example.invalid",
            access_key="synthetic",
            secret_key="synthetic",
            region="region-test",
            agent_name="agent",
        )
    assert len(calls) == 1
    assert calls[0]["state_key"] == storage_validation.terraform_state_key(
        "compute", "agent"
    )


def test_backend_persistence_preserves_actual_external_owner(binding, monkeypatch):
    from npa.cli import agent, agent_terraform
    from npa.clients import project_credential_store

    monkeypatch.setattr(
        agent_terraform, "resolve_terraform_state", lambda _: binding[0]
    )
    monkeypatch.setattr(agent_terraform, "current_operation", lambda: None)
    captures = {}
    monkeypatch.setattr(
        project_credential_store,
        "write_project_credentials",
        lambda project, value, **_: captures.update(private=value),
    )
    monkeypatch.setattr(
        agent, "write_config", lambda value: captures.update(public=value)
    )
    agent_terraform._persist_agent_project_config(
        project="compute",
        project_id="project-compute",
        tenant_id="tenant-test",
        region="region-test",
        merged_vars={
            "s3_bucket": "example-bucket",
            "s3_endpoint": "https://storage.example.invalid",
            "nebius_secret_key": "synthetic",
        },
    )
    public = captures["public"]["projects"]["compute"]["terraform_state"]
    private = captures["private"]["terraform_state"]
    assert (
        public["owner_project_id"] == private["owner_project_id"] == "project-storage"
    )
    assert public["bucket_id"] == private["bucket_id"] == "bucket-resource"
    assert "secret_key" not in public


def test_config_resolves_explicit_owner_binding(monkeypatch):
    monkeypatch.setattr(
        config,
        "_load_yaml",
        lambda: {
            "projects": {
                "compute": {
                    "terraform_state": {
                        "bucket": "example-bucket",
                        "owner_project_id": "project-storage",
                        "bucket_id": "bucket-resource",
                    }
                }
            }
        },
    )
    result = config.resolve_terraform_state("compute")
    assert result.owner_project_id == "project-storage"
    assert result.bucket_id == "bucket-resource"


def test_external_bucket_is_recorded_with_actual_owner_before_agent_iam(binding):
    from npa.cli.agent_terraform import _record_configured_backend

    captured = []
    operation = SimpleNamespace(
        record_resource=lambda **kwargs: captured.append(kwargs)
    )
    _record_configured_backend(
        operation,
        project_alias="compute",
        project_id="project-compute",
        bucket="example-bucket",
        endpoint="https://storage.example.invalid",
    )
    assert len(captured) == 1
    assert captured[0]["project_id"] == "project-storage"
    assert captured[0]["ownership"] == "pre_existing"
    assert captured[0]["provider_id"] == "bucket-resource"
    binding[3]["bucket-resource"]["metadata"]["parent_id"] = "wrong-owner"
    with pytest.raises(nebius.NebiusError):
        _record_configured_backend(
            operation,
            project_alias="compute",
            project_id="project-compute",
            bucket="example-bucket",
            endpoint="https://storage.example.invalid",
        )
    assert len(captured) == 1
