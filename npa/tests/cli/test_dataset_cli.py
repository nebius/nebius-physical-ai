from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.main import app

runner = CliRunner()


def _raw(tmp_path: Path) -> str:
    path = tmp_path / "raw.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {"record_id": "r1", "modality": "camera", "uri": "s3://b/r1", "event": "cut_in", "quality": {"corruption": 0.0}},
                    {"record_id": "r2", "modality": "camera", "uri": "s3://b/r2", "event": "jaywalk", "quality": {"corruption": 0.0}},
                ]
            }
        )
    )
    return str(path)


def test_dataset_help_lists_commands() -> None:
    result = runner.invoke(app, ["workbench", "dataset", "--help"])
    assert result.exit_code == 0
    for command in ("ingest", "validate", "curate", "query"):
        assert command in result.output


def test_dataset_ingest_help_contains_input_path() -> None:
    result = runner.invoke(app, ["workbench", "dataset", "ingest", "--help"])
    assert result.exit_code == 0
    assert "input-path" in result.output


def test_dataset_ingest_writes_versioned_manifest(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "dataset",
            "ingest",
            "--input-path",
            _raw(tmp_path),
            "--output-path",
            str(tmp_path / "ds"),
            "--dataset-id",
            "fleet",
            "--version",
            "v1",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["record_count"] == 2
    assert Path(payload["manifest_uri"]).exists()


def test_dataset_ingest_requires_dataset_id(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["workbench", "dataset", "ingest", "--input-path", _raw(tmp_path), "--output-path", str(tmp_path / "ds")],
    )
    assert result.exit_code != 0


def test_dataset_curate_zero_match_fails(tmp_path: Path) -> None:
    ingest = runner.invoke(
        app,
        ["workbench", "dataset", "ingest", "--input-path", _raw(tmp_path), "--output-path", str(tmp_path / "ds"), "--dataset-id", "fleet", "--output", "json"],
    )
    manifest_uri = json.loads(ingest.output)["manifest_uri"]
    result = runner.invoke(
        app,
        ["workbench", "dataset", "curate", "--input-path", manifest_uri, "--output-path", str(tmp_path / "cur"), "--event", "nope"],
    )
    assert result.exit_code != 0


def test_dataset_service_mode_parity(monkeypatch: Any, tmp_path: Path) -> None:
    import npa.cli.workbench.dataset as cli_module
    from npa.workbench.dataset.service import create_app

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
            "dataset",
            "ingest",
            "--service",
            "--endpoint",
            "http://ds.example",
            "--input-path",
            _raw(tmp_path),
            "--output-path",
            str(tmp_path / "ds"),
            "--dataset-id",
            "fleet",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["record_count"] == 2
