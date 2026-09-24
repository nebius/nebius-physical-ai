"""Config-to-bootstrap coverage for the operator-controlled LeIsaac UI flag."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from npa.cli import agent, agent_records
from npa.cli.agent_contracts import rendered_agent_ui_html
from npa.clients import config


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (None, False),
        ({}, False),
        ({"ui": None}, False),
        ({"ui": True}, False),
        ({"ui": "leisaac_enabled"}, False),
        ({"ui": []}, False),
        ({"ui": {}}, False),
        ({"ui": {"leisaac_enabled": True}}, True),
        ({"ui": {"leisaac_enabled": False}}, False),
        ({"ui": {"leisaac_enabled": None}}, False),
        ({"ui": {"leisaac_enabled": "true"}}, False),
        ({"ui": {"leisaac_enabled": "false"}}, False),
        ({"ui": {"leisaac_enabled": 1}}, False),
        ({"ui": {"leisaac_enabled": 1.0}}, False),
        ({"ui": {"leisaac_enabled": [True]}}, False),
        ({"ui": {"leisaac_enabled": {"enabled": True}}}, False),
    ],
)
def test_leisaac_ui_flag_requires_boolean_true(record, expected) -> None:
    assert agent_records.leisaac_ui_enabled(record) is expected


def _rendered_flag(html: str) -> bool:
    match = re.search(r"const LEISAAC_UI_ENABLED = (true|false);", html)
    assert match, "the deployed UI must contain a resolved boolean constant"
    return json.loads(match.group(1))


@pytest.mark.parametrize("value", [None, "true", 1, [True], {"enabled": True}])
def test_renderer_does_not_coerce_truthy_nonboolean_settings(value) -> None:
    assert _rendered_flag(rendered_agent_ui_html(leisaac_enabled=value)) is False


def test_agent_record_updates_preserve_operator_ui_config(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.yaml")
    agent_records.store_agent_record(
        "selected", "agent", {"ui": {"leisaac_enabled": True}}
    )
    # Deploy persists partial and final records; bootstrap persists health state.
    for state in ("remote_bootstrap_pending", "healthy"):
        agent_records.store_agent_record("selected", "agent", {"setup_state": state})
        saved = agent_records.agent_record("selected", "agent")
        assert saved["setup_state"] == state
        assert agent_records.leisaac_ui_enabled(saved) is True
    agent_records.store_agent_record(
        "selected", "agent", {"ui": {"leisaac_enabled": False}}
    )
    assert (
        agent_records.leisaac_ui_enabled(
            agent_records.agent_record("selected", "agent")
        )
        is False
    )


@pytest.mark.parametrize(
    "ui",
    [
        None,
        {},
        {"leisaac_enabled": False},
        {"leisaac_enabled": True},
        {"leisaac_enabled": "true"},
    ],
)
def test_bootstrap_reads_only_the_selected_agents_yaml_ui_flag(
    monkeypatch, tmp_path, ui
) -> None:
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_project": "other",
                "projects": {
                    "selected": {
                        "agents": {
                            "agent": {} if ui is None else {"ui": ui},
                            "other-agent": {"ui": {"leisaac_enabled": True}},
                        }
                    },
                    "other": {"agents": {"agent": {"ui": {"leisaac_enabled": True}}}},
                },
            }
        ),
        encoding="utf-8",
    )
    ssh = Mock()
    ssh.run.return_value = None
    monkeypatch.setattr(agent, "SSHClient", lambda config: ssh)
    monkeypatch.setattr(
        agent, "resolve_ssh_config", lambda **kwargs: SimpleNamespace(ssh={})
    )
    for name in (
        "_stage_agent_npa_source",
        "_write_agent_s3_env",
        "_write_agent_artifact_sources_env",
        "_write_agent_operator_profile",
        "_write_agent_nebius_env",
        "verify_remote_deployment",
        "_record_remote_setup_ready",
    ):
        monkeypatch.setattr(agent, name, Mock())
    monkeypatch.setattr(agent.agent_llm_config, "write_agent_llm_env", Mock())

    agent._bootstrap_agent_stack(
        host="agent.example",
        ssh_user="ubuntu",
        ssh_key_path="/synthetic/key",
        project_alias="selected",
        agent_name="agent",
        project_id="project-test",
        tenant_id="tenant-test",
        region="eu-north1",
        auth_user="npa",
        auth_password="test-password",
        agent_port=8088,
        backend_port=8787,
        rerun_port=9090,
    )

    setup_scripts = [
        call.args[0]
        for call in ssh.upload_private_text.call_args_list
        if "npa-agent-bootstrap-" in call.args[1]
    ]
    assert len(setup_scripts) == 1
    assert _rendered_flag(setup_scripts[0]) is (
        isinstance(ui, dict) and ui.get("leisaac_enabled") is True
    )
