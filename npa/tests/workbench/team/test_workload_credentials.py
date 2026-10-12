"""Verify explicit per-user provider credentials reach tasks without server secrets."""

import json

import pytest
import yaml

from npa.workbench.team.errors import BackendError
from npa.workbench.team.storage import workload_secrets
from npa.workbench.team.workflow_policy import enforce_rendered_tasks


def test_paidf_tokens_are_loaded_from_allocation_and_injected_into_task(
    binding, tmp_path, monkeypatch
):
    secrets = {"HF_TOKEN": "allocated-hf", "NEBIUS_TOKEN_FACTORY_KEY": "allocated-vlm"}
    path = tmp_path / "provider-credentials.json"
    path.write_text(json.dumps(secrets))
    path.chmod(0o600)
    allocation = binding.allocation.model_copy(update={"workload_secrets_file": path})
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "server-vlm")
    loaded = workload_secrets(allocation)
    assert loaded == secrets
    wave = tmp_path / "wave.yaml"
    wave.write_text(yaml.safe_dump({"resources": {}, "envs": {"HF_TOKEN": "client"}}))
    rendered = yaml.safe_load(enforce_rendered_tasks(wave, binding, loaded))
    assert rendered["envs"] == secrets
    assert "server-vlm" not in json.dumps(rendered)


def test_unallocated_provider_tokens_are_not_inherited(binding, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "server-hf")
    monkeypatch.setenv("NEBIUS_TOKEN_FACTORY_KEY", "server-vlm")
    assert workload_secrets(binding.allocation) == {}


@pytest.mark.parametrize(
    "name", ["NEBIUS_IAM_TOKEN", "AWS_SECRET_ACCESS_KEY", "KUBECONFIG"]
)
def test_provider_file_cannot_grant_operator_credentials(binding, tmp_path, name):
    path = tmp_path / "provider-credentials.json"
    path.write_text(json.dumps({"NEBIUS_TOKEN_FACTORY_KEY": "vlm", name: "operator"}))
    path.chmod(0o600)
    allocation = binding.allocation.model_copy(update={"workload_secrets_file": path})
    with pytest.raises(BackendError, match="allocated model-provider credentials"):
        workload_secrets(allocation)


def test_provider_credentials_require_private_file(binding, tmp_path):
    path = tmp_path / "provider-credentials.json"
    path.write_text(json.dumps({"NEBIUS_TOKEN_FACTORY_KEY": "vlm"}))
    path.chmod(0o644)
    allocation = binding.allocation.model_copy(update={"workload_secrets_file": path})
    with pytest.raises(BackendError, match="allocated model-provider credentials"):
        workload_secrets(allocation)
