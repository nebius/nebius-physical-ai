"""Deploy, interrupt credential staging, resume the exact VM, and tear it down."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from npa.cli import agent
from npa.cli.main import app
from npa.clients.ssh import SSHError


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.agent_live,
    pytest.mark.skipif(
        not os.environ.get("NPA_AGENT_RECOVERY_LIVE_CONFIG"),
        reason="Requires private configuration for an isolated agent deploy/resume/destroy cycle.",
    ),
]


def _invoke(args: list[str], evidence: Path, label: str):
    result = CliRunner().invoke(app, args)
    (evidence / f"{label}.log").write_text(result.output)
    return result


@pytest.fixture
def deployment():
    config_path = Path(os.environ["NPA_AGENT_RECOVERY_LIVE_CONFIG"])
    assert config_path.stat().st_mode & 0o077 == 0, "Live config must be owner-only"
    config = json.loads(config_path.read_text())
    args = config["deploy_args"]
    assert args[:2] == ["agent", "deploy"]
    project, name = (args[args.index(flag) + 1] for flag in ("--project", "--name"))
    assert not agent._agent_record(project, name), "Choose an unused agent name"
    evidence = Path(config["evidence_dir"])
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert evidence.stat().st_mode & 0o077 == 0, "Evidence must be owner-only"
    try:
        yield args, project, name, evidence
    finally:
        cleanup = _invoke(
            ["agent", "destroy", "--project", project, "--name", name, "--yes"],
            evidence,
            "destroy",
        )
        assert cleanup.exit_code == 0, (
            "Exact-agent cleanup failed; inspect private destroy.log"
        )


def test_interrupted_credentials_resume_without_reinstalling(deployment, monkeypatch):
    args, project, name, evidence = deployment
    stage_source = Mock(wraps=agent._stage_agent_npa_source)
    monkeypatch.setattr(agent, "_stage_agent_npa_source", stage_source)
    write_credentials = agent._write_agent_nebius_env
    interrupted = Mock(side_effect=SSHError("injected credential transport loss"))
    monkeypatch.setattr(agent, "_write_agent_nebius_env", interrupted)
    first = _invoke(args, evidence, "interrupted")
    assert first.exit_code != 0
    interrupted.assert_called_once()
    instance_id = agent._agent_record(project, name)["instance_id"]
    monkeypatch.setattr(agent, "_write_agent_nebius_env", write_credentials)
    resumed = _invoke(args, evidence, "resumed")
    assert resumed.exit_code == 0, "Resume failed; inspect private resumed.log"
    assert agent._agent_record(project, name)["instance_id"] == instance_id
    assert stage_source.call_count == 1, "Resume reran source/service installation"
    assert "Reusing completed agent service installation" in resumed.output
    status = _invoke(
        ["agent", "status", "--project", project, "--name", name, "--json"],
        evidence,
        "status",
    )
    assert status.exit_code == 0
    payload = json.loads(status.stdout)
    assert payload["health"] is True
    assert payload["basic_auth_enforced"] is True
    (evidence / "result.json").write_text(
        json.dumps(
            {
                "same_instance": True,
                "service_installations": stage_source.call_count,
                "credential_resume": True,
                "authenticated_health": True,
            },
            indent=2,
        )
        + "\n"
    )
