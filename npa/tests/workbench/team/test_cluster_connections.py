"""Exercise portable EKS credentials, cross-cluster placement, and connection checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.workbench.team import app
from npa.workbench.team import cluster_connections as connections
from npa.workbench.team.deployment import allocations, server_config
from npa.workbench.team.errors import BackendError
from npa.workbench.team.models import TeamConfig
from npa.workbench.team.workflow_policy import prepare_document, worker_context


def kubeconfig(name):
    return {
        "apiVersion": "v1",
        "kind": "Config",
        "current-context": name,
        "clusters": [
            {"name": name, "cluster": {"server": f"https://{name}.example.test"}}
        ],
        "users": [
            {
                "name": name,
                "user": {
                    "exec": {
                        "apiVersion": "client.authentication.k8s.io/v1beta1",
                        "command": "/opt/homebrew/bin/aws",
                        "args": [
                            "--region",
                            "us-east-1",
                            "eks",
                            "get-token",
                            "--cluster-name",
                            name,
                        ],
                        "env": [
                            {"name": "AWS_PROFILE", "value": "scheduler"},
                            {
                                "name": "AWS_ACCESS_KEY_ID",
                                "value": "operator-only-test-key",
                            },
                            {"name": "AWS_CONFIG_FILE", "value": "/laptop/aws-config"},
                        ],
                    }
                },
            }
        ],
        "contexts": [{"name": name, "context": {"cluster": name, "user": name}}],
    }


def test_export_relocates_eks_for_gateway_and_scheduler(config, tmp_path, monkeypatch):
    called = []

    def read(binding, *args):
        called.append(binding.cluster)
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(kubeconfig(binding.cluster))
        )

    monkeypatch.setattr(connections, "kubectl", read)
    directory = tmp_path / "exported"
    result = connections.export_connections(config, directory)
    exported = TeamConfig.model_validate(
        yaml.safe_load((directory / "team.yaml").read_text())
    )
    assert called == ["east", "west"]
    assert set(result["clusters"]) == {"east", "west"}
    assert directory.stat().st_mode & 0o777 == 0o700
    scheduler = yaml.safe_load((directory / "sky-kubeconfig.yaml").read_text())
    assert len(scheduler["contexts"]) == 4
    for name in called:
        assert exported.clusters[name].context == name
        assert exported.clusters[name].kubeconfig == Path(
            f"/etc/npa-team/cluster-{name}.yaml"
        )
    for path in directory.iterdir():
        assert path.stat().st_mode & 0o777 == 0o600
        assert "operator-only-test-key" not in path.read_text()
        assert "/laptop/" not in path.read_text()
    for user in scheduler["users"]:
        plugin = user["user"]["exec"]
        assert plugin["command"] == "/opt/sky/bin/aws"
        assert plugin["interactiveMode"] == "Never"
        environment = {entry["name"]: entry["value"] for entry in plugin["env"]}
        assert environment["AWS_PROFILE"] == "scheduler"
        assert environment["AWS_CONFIG_FILE"] == "/etc/npa-team/aws-config"
        assert (
            environment["AWS_SHARED_CREDENTIALS_FILE"]
            == "/etc/npa-team/aws-credentials"
        )
        assert environment["AWS_EC2_METADATA_DISABLED"] == "true"
    assert exported.account_namespace == config.account_namespace
    assert exported.workspaces == config.workspaces
    with pytest.raises(FileExistsError):
        connections.export_connections(config, directory)


@pytest.mark.parametrize(
    "authentication",
    [
        {"tokenFile": "/var/run/projected/token"},
        {"client-certificate-data": "test-certificate", "client-key-data": "test-key"},
    ],
)
def test_non_eks_credentials_are_preserved(binding, monkeypatch, authentication):
    document = kubeconfig("selected")
    document["users"][0]["user"] = authentication
    monkeypatch.setattr(
        connections,
        "kubectl",
        lambda *args: SimpleNamespace(returncode=0, stdout=json.dumps(document)),
    )
    assert connections.read_connection(binding, Path("/etc/private")) == document


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(returncode=1, stdout="private-provider-error"),
        SimpleNamespace(returncode=0, stdout="not json"),
        SimpleNamespace(
            returncode=0, stdout='{"users":[],"clusters":[],"contexts":[]}'
        ),
    ],
)
def test_failed_export_leaves_no_secret_directory(
    config, tmp_path, monkeypatch, response
):
    monkeypatch.setattr(connections, "kubectl", lambda *args: response)
    with pytest.raises(BackendError) as failure:
        connections.export_connections(config, tmp_path / "output")
    assert "private-provider-error" not in str(failure.value)
    assert not (tmp_path / "output").exists()


def test_same_user_can_target_separate_existing_clusters_without_provider_ids(
    config, workflow
):
    bindings = list(allocations(config))[:2]
    assert bindings[0].allocation.subject == bindings[1].allocation.subject
    contexts = []
    for binding in bindings:
        prepared = prepare_document(workflow, binding, "run-selected")
        profile = prepared["resources"]["cpu"]
        assert profile["cloud"] == "kubernetes"
        assert profile["region"] == worker_context(binding)
        contexts.append(profile["region"])
    assert contexts[0] != contexts[1]
    assert server_config(config)["allowed_clouds"] == ["kubernetes"]


def test_cluster_check_fails_closed_and_never_uses_cloud_provisioning(
    config, monkeypatch
):
    checked = []
    monkeypatch.setattr(
        connections,
        "verify_enrollment",
        lambda binding: checked.append(binding.cluster),
    )
    assert connections.check_clusters(config) == {
        "status": "verified",
        "allocations": {"east": 2, "west": 2},
    }
    assert checked == ["east", "west", "east", "west"]

    def denied(binding):
        raise BackendError("cluster enrollment differs from current policy")

    monkeypatch.setattr(connections, "verify_enrollment", denied)
    with pytest.raises(BackendError):
        connections.check_clusters(config)


def test_connection_commands_use_shared_implementation(config, tmp_path, monkeypatch):
    path = tmp_path / "operator.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    monkeypatch.setattr(connections, "check_clusters", lambda _: {"status": "verified"})
    checked = CliRunner().invoke(app, ["check-clusters", "--config", str(path)])
    assert checked.exit_code == 0, checked.output
    assert json.loads(checked.stdout) == {"status": "verified"}
    monkeypatch.setattr(
        connections,
        "export_connections",
        lambda *args, **kwargs: {"files": ["team.yaml"]},
    )
    exported = CliRunner().invoke(
        app,
        [
            "export-connections",
            "--config",
            str(path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert exported.exit_code == 0, exported.output
    assert json.loads(exported.stdout) == {"files": ["team.yaml"]}
