"""Verify operator setup's JSON exit contract and independence from personal login."""

import json

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.workbench.team import app
from npa.workbench.team import setup
from npa.workbench.team.client import TeamClient


@pytest.mark.parametrize("phase,code", [("ready", 0), ("endpoint-awaiting-dns", 2)])
def test_operator_command_needs_no_personal_login(tmp_path, monkeypatch, phase, code):
    config = tmp_path / "setup.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "project_id": "project-test",
                "cluster_id": "cluster-test",
                "kubeconfig": str(tmp_path / "kubeconfig"),
                "context": "selected",
                "namespace": "management",
                "image": "example/team@sha256:" + "a" * 64,
                "configuration_secret": "configuration",
                "state_claim": "state",
                "tls_secret": "tls",
                "endpoint": "https://team.example.test",
                "node_selector": {"pool": "cpu"},
            }
        )
    )

    def personal_client_forbidden(*args, **kwargs):
        pytest.fail("operator setup attempted personal login")

    monkeypatch.setattr(TeamClient, "__init__", personal_client_forbidden)
    expected = {"status": phase, "personal_client_required": False}
    monkeypatch.setattr(setup, "setup_control_plane", lambda *args: expected)
    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--input-path",
            str(config),
            "--output-path",
            str(tmp_path / "installation.json"),
        ],
    )
    assert result.exit_code == code, result.output
    assert json.loads(result.stdout) == expected
