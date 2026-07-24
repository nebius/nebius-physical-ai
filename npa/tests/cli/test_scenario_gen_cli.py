from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.main import app

runner = CliRunner()


def test_scenario_gen_help_lists_commands() -> None:
    result = runner.invoke(app, ["workbench", "scenario-gen", "--help"])
    assert result.exit_code == 0
    assert "generate" in result.output
    assert "rank" in result.output


def test_scenario_gen_generate_help_contains_output() -> None:
    result = runner.invoke(app, ["workbench", "scenario-gen", "generate", "--help"])
    assert result.exit_code == 0
    assert "output" in result.output


def test_scenario_gen_generate_writes_local_manifest(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "scenario-gen",
            "generate",
            "--policy-uri",
            "s3://bucket/policy.ckpt",
            "--input-path",
            "s3://bucket/base.json",
            "--output-path",
            str(tmp_path / "adv"),
            "--num-scenarios",
            "5",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["scenario_count"] == 5
    assert Path(payload["manifest_uri"]).exists()


def test_scenario_gen_generate_requires_policy_uri(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "scenario-gen",
            "generate",
            "--input-path",
            "s3://bucket/base.json",
            "--output-path",
            str(tmp_path / "adv"),
        ],
    )
    assert result.exit_code != 0


def test_scenario_gen_rank_missing_input_fails(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "scenario-gen",
            "rank",
            "--input-path",
            str(tmp_path / "missing.json"),
            "--output-path",
            str(tmp_path / "ranked"),
        ],
    )
    assert result.exit_code != 0


def test_scenario_gen_service_mode_parity(monkeypatch: Any, tmp_path: Path) -> None:
    import npa.cli.workbench.scenario_gen as cli_module
    from npa.workbench.scenario_gen.service import create_app

    client = TestClient(create_app(auth_mode="none"))

    def fake_request(method: str, endpoint: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = client.request(method, path, json=kwargs.get("payload"), params=kwargs.get("params"))
        assert response.status_code == 200, response.text
        return response.json()

    monkeypatch.setattr(cli_module, "request_json", fake_request)

    result = runner.invoke(
        app,
        [
            "workbench",
            "scenario-gen",
            "generate",
            "--service",
            "--endpoint",
            "http://sg.example",
            "--policy-uri",
            "s3://bucket/policy.ckpt",
            "--input-path",
            "s3://bucket/base.json",
            "--output-path",
            str(tmp_path / "adv"),
            "--num-scenarios",
            "2",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["scenario_count"] == 2
