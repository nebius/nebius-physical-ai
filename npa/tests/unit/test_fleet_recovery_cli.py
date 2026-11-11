"""Exercise the guarded Fleet recovery command without live infrastructure."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.fleet import recovery
from npa.lifecycle_intent import OperationIntent, current_intent
from npa.sdk import fleet as fleet_sdk

runner = CliRunner()


@pytest.fixture
def spec_path(tmp_path):
    path = tmp_path / "fleet.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "npa.fleet/v0.0.1",
                "name": "recovery-cli-test",
                "tenant_id": "tenant-test",
                "region": "test-region",
                "profile": "operator-test",
                "projects": [
                    {
                        "name": "team",
                        "project_id": "project-test",
                        "clusters": [
                            {
                                "name": "recover",
                                "cpu_nodes": {
                                    "count": 1,
                                    "platform": "cpu-d3",
                                    "preset": "48vcpu-192gb",
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _result():
    return {
        "backend": "mk8s",
        "project_key": "team",
        "project_id": "project-test",
        "cluster_name": "recover",
        "cluster_id": "cluster-test",
        "region": "test-region",
        "kube_context": "fleet-recovery-cli-test-team-recover",
        "kubeconfig": "/private/kubeconfig",
        "status": "recovered-pending-validation",
        "mk8s_recipe_release": {"release_id": "mk8s-test"},
    }


def test_json_is_one_document_and_propagates_recovery_scope(spec_path, monkeypatch):
    observed = []

    def recover(spec, **kwargs):
        observed.append((current_intent(), spec.name, kwargs))
        return _result()

    monkeypatch.setattr(recovery, "recover_interrupted_fleet_apply", recover)
    result = runner.invoke(
        app,
        [
            "fleet",
            "recover",
            "--spec",
            str(spec_path),
            "--only-projects",
            "team, other",
            "--only-clusters",
            "recover, another",
            "--project-prefix",
            "custom-",
            "--profile",
            "operator-test",
            "--yes",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == _result()
    assert observed == [
        (
            OperationIntent.MUTATE,
            "recovery-cli-test",
            {
                "only_projects": ["team", "other"],
                "only_clusters": ["recover", "another"],
                "project_prefix": "custom-",
                "profile": "operator-test",
            },
        )
    ]


def test_failure_is_sanitized_in_json_mode(spec_path, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise recovery.FleetRecoveryError("private-provider-receipt")

    monkeypatch.setattr(recovery, "recover_interrupted_fleet_apply", unavailable)
    result = runner.invoke(
        app,
        ["fleet", "recover", "--spec", str(spec_path), "--yes", "--output", "json"],
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "failure": "recovery_unavailable",
        "status": "failed",
    }
    assert "private-provider-receipt" not in result.output


def test_recovery_help_is_installed():
    result = runner.invoke(app, ["fleet", "recover", "--help"])
    assert result.exit_code == 0
    assert "--only-projects" in result.stdout
    assert "--yes" in result.stdout


def test_sdk_uses_the_shared_recovery_implementation(monkeypatch, tmp_path):
    observed = []

    def recover(spec, **kwargs):
        observed.append((spec, kwargs))
        return _result()

    monkeypatch.setattr(recovery, "recover_interrupted_fleet_apply", recover)
    assert (
        fleet_sdk.recover_interrupted_apply(
            "fleet-spec", only_projects=["team"], work_root=tmp_path
        )
        == _result()
    )
    assert observed == [
        (
            "fleet-spec",
            {
                "only_projects": ["team"],
                "only_clusters": None,
                "project_prefix": None,
                "profile": None,
                "work_root": tmp_path,
            },
        )
    ]
    assert "recover_interrupted_apply" in fleet_sdk.__all__
