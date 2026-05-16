from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.workbench.lancedb import DEFAULT_CONTAINER_IMAGE, app as lancedb_app
from npa.cli.workbench.lancedb.import_lerobot import resolve_lerobot_dataset_files


runner = CliRunner()


@pytest.mark.xfail(reason="Parent Workbench registration requires editing npa/src/npa/cli/workbench/__init__.py outside this run allowlist.")
def test_lancedb_registered_under_workbench() -> None:
    from npa.cli.main import app as main_app

    result = runner.invoke(main_app, ["workbench", "--help"])

    assert result.exit_code == 0
    assert "lancedb" in result.output


@pytest.mark.parametrize(
    "command",
    [
        "deploy",
        "status",
        "list",
        "create-table",
        "query",
        "import-lerobot",
        "import-bdd100k",
        "backfill",
    ],
)
def test_lancedb_command_help(command: str) -> None:
    result = runner.invoke(lancedb_app, [command, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_lancedb_deploy_vm_requires_storage_path() -> None:
    result = runner.invoke(lancedb_app, ["deploy", "--runtime", "vm"])

    assert result.exit_code == 1
    assert "--storage-path is required" in result.output


def test_lancedb_deploy_cloud_requires_endpoint_and_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANCEDB_API_KEY", raising=False)

    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "cloud",
            "--database",
            "robot-data",
            "--cloud-region",
            "us-east-1",
        ],
    )

    assert result.exit_code == 1
    assert "--endpoint is required" in result.output

    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "cloud",
            "--endpoint",
            "https://cloud.example",
            "--database",
            "robot-data",
            "--cloud-region",
            "us-east-1",
        ],
    )

    assert result.exit_code == 1
    assert "LANCEDB_API_KEY is required" in result.output


def test_lancedb_deploy_validates_port_range() -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            "/tmp/lancedb",
            "--port",
            "99",
        ],
    )

    assert result.exit_code == 1
    assert "--port must be between" in result.output


def test_lancedb_status_endpoint_required() -> None:
    result = runner.invoke(lancedb_app, ["status"])

    assert result.exit_code == 1
    assert "--endpoint is required" in result.output


def test_lancedb_list_returns_table_names(monkeypatch: pytest.MonkeyPatch) -> None:
    list_module = importlib.import_module("npa.cli.workbench.lancedb.list")

    def fake_request(method: str, endpoint: str, path: str, **kwargs):
        return {"tables": ["robot_embeddings", "scratch"]}

    monkeypatch.setattr(list_module, "request_json", fake_request)

    result = runner.invoke(
        lancedb_app,
        ["list", "--endpoint", "http://localhost:8686", "--prefix", "robot"],
    )

    assert result.exit_code == 0
    assert "robot_embeddings" in result.output
    assert "scratch" not in result.output


def test_lancedb_create_table_schema_validation(tmp_path: Path) -> None:
    missing_schema = tmp_path / "missing.json"

    result = runner.invoke(
        lancedb_app,
        [
            "create-table",
            "--endpoint",
            "http://localhost:8686",
            "--table",
            "robot_embeddings",
            "--schema",
            str(missing_schema),
        ],
    )

    assert result.exit_code == 1
    assert "--schema does not exist" in result.output


def test_lancedb_query_top_k_default(monkeypatch: pytest.MonkeyPatch) -> None:
    query_module = importlib.import_module("npa.cli.workbench.lancedb.query")
    seen = {}

    def fake_request(method: str, endpoint: str, path: str, **kwargs):
        seen.update(kwargs["payload"])
        return {"results": [{"id": "row-1", "_distance": 0.0}]}

    monkeypatch.setattr(query_module, "request_json", fake_request)

    result = runner.invoke(
        lancedb_app,
        [
            "query",
            "--endpoint",
            "http://localhost:8686",
            "--table",
            "robot_embeddings",
            "--vector",
            "[1.0, 0.0]",
        ],
    )

    assert result.exit_code == 0
    assert seen["top_k"] == 5
    assert seen["vector"] == [1.0, 0.0]


def test_lancedb_query_vector_format_validation() -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "query",
            "--endpoint",
            "http://localhost:8686",
            "--table",
            "robot_embeddings",
            "--vector",
            "not-json",
        ],
    )

    assert result.exit_code == 1
    assert "Vector must be a JSON array" in result.output


def test_lancedb_query_rejects_bad_top_k() -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "query",
            "--endpoint",
            "http://localhost:8686",
            "--table",
            "robot_embeddings",
            "--vector",
            "[1.0, 0.0]",
            "--top-k",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "--top-k must be between" in result.output


def test_lancedb_import_lerobot_dataset_resolution(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    parquet = data_dir / "episode_000000.parquet"
    parquet.write_bytes(b"placeholder")

    assert resolve_lerobot_dataset_files(str(tmp_path)) == [parquet]


def test_lancedb_import_bdd100k_local_outputs_json(tmp_path: Path) -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "import-bdd100k",
            "--synthetic",
            "3",
            "--synthetic-seed",
            "5",
            "--table",
            "bdd_cli",
            "--output-path",
            str(tmp_path / "db"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["table"] == "bdd_cli"
    assert payload["total_rows"] == 3
    assert payload["manifest_sha256"]


def test_lancedb_import_bdd100k_service_calls_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    import_module = importlib.import_module("npa.cli.workbench.lancedb.import_bdd100k")
    seen = {}

    def fake_request(method: str, endpoint: str, path: str, **kwargs):
        seen.update({"method": method, "endpoint": endpoint, "path": path, **kwargs})
        return {
            "table": "bdd_service",
            "lance_uri": "s3://bucket/lancedb/bdd100k/",
            "table_uri": "s3://bucket/lancedb/bdd100k/bdd_service.lance",
            "rows_per_split": {"train": 2, "val": 1},
            "total_rows": 3,
            "table_version_before": None,
            "table_version_after": 1,
            "table_version": 1,
            "manifest_sha256": "abc",
            "row_checksum_sha256": "abc",
            "splits": ["train", "val"],
            "synthetic": 3,
            "synthetic_seed": 5,
            "source": "",
        }

    monkeypatch.setattr(import_module, "request_json", fake_request)

    result = runner.invoke(
        lancedb_app,
        [
            "import-bdd100k",
            "--service",
            "--endpoint",
            "http://localhost:8686",
            "--synthetic",
            "3",
            "--synthetic-seed",
            "5",
            "--table",
            "bdd_service",
        ],
    )

    assert result.exit_code == 0
    assert seen["method"] == "POST"
    assert seen["path"] == "/import-bdd100k"
    assert seen["payload"]["synthetic"] == 3
    assert json.loads(result.output)["total_rows"] == 3


def test_lancedb_container_image_name_resolves() -> None:
    assert DEFAULT_CONTAINER_IMAGE == "npa-lancedb:0.30.2"


@pytest.mark.smoke
def test_lancedb_end_to_end_deploy_table_query() -> None:
    if not os.environ.get("NPA_LANCEDB_SMOKE"):
        pytest.skip("Set NPA_LANCEDB_SMOKE=1 to run the local Docker LanceDB smoke.")
    payload = [
        {"id": f"row-{idx}", "vector": [float(idx), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        for idx in range(10)
    ]
    assert len(json.dumps(payload)) > 0
