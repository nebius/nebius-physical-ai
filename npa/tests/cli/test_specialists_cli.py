"""Finite specialist commands emit one JSON result, including command failures."""

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from npa.cli.workbench import specialists


@pytest.mark.parametrize("output", [[], ["--output-format", "json"]])
@pytest.mark.parametrize(
    "arguments",
    [
        ["submit", "inspect"],
        ["status"],
        ["pause", "--task-id", "task"],
        ["cancel", "task"],
        ["reconcile", "task", "--retry"],
    ],
)
@pytest.mark.parametrize("fail", [False, True])
def test_management_json_contract(tmp_path, monkeypatch, arguments, output, fail):
    config = tmp_path / "team.json"
    config.write_text("{}")

    def operation(*args, **kwargs):
        print("third-party diagnostic")
        if fail:
            print('{"status": "success"}')
            raise ValueError("synthetic operation failure")
        return {"status": "queued", "id": "task"}

    team = SimpleNamespace(
        **{
            name: operation
            for name in ("submit", "status", "pause", "cancel", "reconcile")
        }
    )
    monkeypatch.setattr(specialists, "load_config", lambda path: None)
    monkeypatch.setattr(specialists, "SpecialistTeam", lambda config: team)
    result = CliRunner().invoke(
        specialists.app, ["--config", str(config), *arguments, *output]
    )
    document = json.loads(result.stdout)
    assert "third-party diagnostic" not in result.stdout
    if fail:
        assert result.exit_code != 0
        assert document.get("status") != "success"
    else:
        assert result.exit_code == 0, result.output
        assert document == {"status": "queued", "id": "task"}
