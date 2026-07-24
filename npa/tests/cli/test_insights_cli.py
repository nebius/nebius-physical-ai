from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.main import app

runner = CliRunner()


def _record(tmp_path: Path, store: str, run_id: str, metric: str, value: float) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "insights",
            "record",
            "--output-path",
            store,
            "--run-id",
            run_id,
            "--metric",
            metric,
            "--value",
            str(value),
            "--tool",
            "rl",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output


def test_insights_help_lists_commands() -> None:
    result = runner.invoke(app, ["workbench", "insights", "--help"])
    assert result.exit_code == 0
    for command in ("record", "ingest-run", "query", "lineage", "compare", "dashboard"):
        assert command in result.output


def test_insights_query_help_contains_input_path() -> None:
    result = runner.invoke(app, ["workbench", "insights", "query", "--help"])
    assert result.exit_code == 0
    assert "input-path" in result.output


def test_insights_record_and_query(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    _record(tmp_path, store, "r1", "accuracy", 0.9)
    result = runner.invoke(
        app,
        ["workbench", "insights", "query", "--input-path", store, "--metric-name", "accuracy", "--output", "json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["count"] == 1


def test_insights_record_requires_output_path() -> None:
    result = runner.invoke(app, ["workbench", "insights", "record", "--metric", "x", "--value", "1"])
    assert result.exit_code != 0


def test_insights_ingest_run_cli(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "npa.dataset.manifest.v1",
                "dataset_id": "d",
                "version": "v1",
                "record_count": 2,
                "modalities": ["camera"],
                "lineage": {"workflow_run": "r", "input_uris": []},
                "quality_stats": {"record_count": 2, "mean_completeness": 1.0, "corrupt_count": 0, "modalities": ["camera"]},
                "records": [],
            }
        )
    )
    result = runner.invoke(
        app,
        ["workbench", "insights", "ingest-run", "--input-path", str(run), "--output-path", str(tmp_path / "store"), "--output", "json"],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["recorded_count"] == 5


def test_insights_compare_and_dashboard_cli(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    _record(tmp_path, store, "r1", "accuracy", 0.8)
    _record(tmp_path, store, "r2", "accuracy", 0.9)

    compare = runner.invoke(
        app,
        ["workbench", "insights", "compare", "--input-path", store, "--base-run", "r1", "--candidate-run", "r2", "--output", "json"],
    )
    assert compare.exit_code == 0, compare.output
    assert json.loads(compare.output)["improved"] == ["accuracy"]

    dashboard = runner.invoke(
        app,
        ["workbench", "insights", "dashboard", "--input-path", store, "--output-path", str(tmp_path / "dash"), "--output", "json"],
    )
    assert dashboard.exit_code == 0, dashboard.output
    payload = json.loads(dashboard.output)
    assert payload["total_records"] == 2
    assert Path(payload["html_uri"]).exists()


def test_insights_compare_zero_match_fails(tmp_path: Path) -> None:
    store = str(tmp_path / "store")
    _record(tmp_path, store, "r1", "accuracy", 0.8)
    result = runner.invoke(
        app,
        ["workbench", "insights", "compare", "--input-path", store, "--base-run", "r1", "--candidate-run", "nope"],
    )
    assert result.exit_code != 0


def test_insights_service_mode_parity(monkeypatch: Any, tmp_path: Path) -> None:
    import npa.cli.workbench.insights as cli_module
    from npa.workbench.insights.service import create_app

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
            "insights",
            "record",
            "--service",
            "--endpoint",
            "http://insights.example",
            "--output-path",
            str(tmp_path / "store"),
            "--run-id",
            "r1",
            "--metric",
            "accuracy",
            "--value",
            "0.9",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["recorded_count"] == 1
