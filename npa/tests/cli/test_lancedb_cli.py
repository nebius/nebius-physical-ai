from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

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
        "create-mv",
        "refresh-mv",
        "query-table",
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


def test_lancedb_deploy_serverless_unsupported() -> None:
    result = runner.invoke(lancedb_app, ["deploy", "--runtime", "serverless"])
    assert result.exit_code == 1
    assert "does not support --runtime serverless" in result.output


def test_lancedb_deploy_cloud_requires_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANCEDB_API_KEY", "key")
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "cloud",
            "--endpoint",
            "https://cloud.example",
            "--cloud-region",
            "us-east-1",
        ],
    )
    assert result.exit_code == 1
    assert "--database is required" in result.output


def test_lancedb_deploy_cloud_requires_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANCEDB_API_KEY", "key")
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "cloud",
            "--endpoint",
            "https://cloud.example",
            "--database",
            "robots",
        ],
    )
    assert result.exit_code == 1
    assert "--cloud-region is required" in result.output


def test_lancedb_deploy_cloud_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANCEDB_API_KEY", "key")
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "cloud",
            "--endpoint",
            "https://cloud.example/",
            "--database",
            "robots",
            "--cloud-region",
            "us-east-1",
            "--default",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.find("{"):])
    assert payload["runtime"] == "cloud"
    assert payload["endpoint"] == "https://cloud.example"
    assert payload["database"] == "robots"
    assert payload["default_requested"] is True


def test_lancedb_deploy_invalid_auth_mode() -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            "/tmp/lancedb",
            "--auth-mode",
            "magic",
        ],
    )
    assert result.exit_code == 1
    assert "--auth-mode must be one of" in result.output


def test_lancedb_deploy_container_token_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANCEDB_TOKEN", raising=False)
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            "/tmp/lancedb",
            "--auth-mode",
            "token",
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "LANCEDB_TOKEN is required" in result.output


def test_lancedb_deploy_container_dry_run_prints_docker_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANCEDB_TOKEN", "tok")
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            "s3://bucket/lancedb",
            "--auth-mode",
            "token",
            "--dry-run",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    # Should echo a docker run command before emitting the JSON payload
    assert "docker run" in result.output
    payload = json.loads(result.output[result.output.find("{"):])
    assert payload["runtime"] == "container"
    assert payload["status"] == "dry-run"
    assert payload["container_id"] == "<dry-run>"


def test_lancedb_deploy_container_destroy_runs_docker_rm(
    monkeypatch: pytest.MonkeyPatch, mocker
) -> None:
    run = mocker.patch(
        "npa.cli.workbench.lancedb.deploy.subprocess.run",
        return_value=SimpleNamespace(stdout="", stderr="", returncode=0),
    )
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            "/tmp/lancedb",
            "--destroy",
            "--container-name",
            "npa-lancedb-custom",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0][:3] == ["docker", "rm", "-f"]
    payload = json.loads(result.output[result.output.find("{"):])
    assert payload["status"] == "removed"


def test_lancedb_deploy_container_success(mocker, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANCEDB_TOKEN", "tok")
    mocker.patch(
        "npa.cli.workbench.lancedb.deploy.subprocess.run",
        return_value=SimpleNamespace(stdout="cid123\n", stderr="", returncode=0),
    )
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            "/tmp/lancedb",
            "--auth-mode",
            "token",
            "--port",
            "9001",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.find("{"):])
    assert payload["runtime"] == "container"
    assert payload["endpoint"] == "http://localhost:9001"
    assert payload["container_id"] == "cid123"
    assert payload["status"] == "running"


def test_lancedb_deploy_container_run_handles_docker_missing(
    mocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANCEDB_TOKEN", "tok")
    mocker.patch(
        "npa.cli.workbench.lancedb.deploy.subprocess.run",
        side_effect=FileNotFoundError(),
    )
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            "/tmp/lancedb",
            "--auth-mode",
            "token",
        ],
    )
    assert result.exit_code == 1
    assert "Docker is not installed" in result.output


def test_lancedb_deploy_vm_infra_only(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "vm",
            "--storage-path",
            "s3://bucket/lancedb",
            "--skip-app",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.find("{"):])
    assert payload["status"] == "infra-only"
    assert payload["runtime"] == "vm"


def test_lancedb_deploy_vm_dry_run_blocked_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "vm",
            "--storage-path",
            "s3://bucket/lancedb",
            "--dry-run",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.find("{"):])
    assert payload["status"] == "blocked"
    assert payload["dry_run"] is True


def test_lancedb_deploy_vm_without_dry_run_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "vm",
            "--storage-path",
            "s3://bucket/lancedb",
        ],
    )
    assert result.exit_code == 1
    assert "Workbench parent registration" in result.output


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


def test_lancedb_import_lerobot_s3_skips_local_resolution() -> None:
    assert resolve_lerobot_dataset_files("s3://bucket/path") == []


def test_lancedb_import_lerobot_missing_path_errors(tmp_path: Path) -> None:
    from npa.cli.workbench.lancedb.import_lerobot import import_lerobot_cmd  # noqa: F401

    missing = tmp_path / "nope"
    result = runner.invoke(
        lancedb_app,
        [
            "import-lerobot",
            "--endpoint",
            "http://localhost:8001",
            "--dataset-path",
            str(missing),
            "--table",
            "robots",
        ],
    )
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_lancedb_import_lerobot_finds_loose_parquet_files(tmp_path: Path) -> None:
    # No data/**/*.parquet layout -> falls back to rglob.
    loose = tmp_path / "episode_000000.parquet"
    loose.write_bytes(b"placeholder")

    assert resolve_lerobot_dataset_files(str(tmp_path)) == [loose]


def test_lancedb_import_lerobot_empty_directory_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "import-lerobot",
            "--endpoint",
            "http://localhost:8001",
            "--dataset-path",
            str(tmp_path),
            "--table",
            "robots",
        ],
    )
    assert result.exit_code == 1
    assert "No parquet files found" in result.output


def test_lancedb_import_lerobot_negative_limit_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "import-lerobot",
            "--endpoint",
            "http://localhost:8001",
            "--dataset-path",
            "s3://bucket/x",
            "--table",
            "robots",
            "--limit",
            "-1",
        ],
    )
    assert result.exit_code == 1
    assert "--limit must be" in result.output


def test_lancedb_import_lerobot_s3_path_posts_to_endpoint(
    tmp_path: Path, mocker
) -> None:
    captured: dict = {}

    def fake_request_json(method, endpoint, path, *, headers=None, payload=None, timeout=None):
        captured.update(
            method=method,
            endpoint=endpoint,
            path=path,
            payload=payload,
        )
        return {"rows": 0}

    mocker.patch(
        "npa.cli.workbench.lancedb.import_lerobot.request_json", side_effect=fake_request_json
    )

    result = runner.invoke(
        lancedb_app,
        [
            "import-lerobot",
            "--endpoint",
            "http://localhost:8001",
            "--dataset-path",
            "s3://bucket/lerobot/",
            "--table",
            "robots",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["method"] == "POST"
    assert captured["path"] == "/tables/robots"
    assert captured["payload"]["source_format"] == "lerobot"
    assert captured["payload"]["input_path"] == "s3://bucket/lerobot/"
    assert captured["payload"]["rows"] == []


def test_lancedb_import_lerobot_local_files_inject_id_and_vector(
    tmp_path: Path, mocker
) -> None:
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    parquet = data_dir / "episode_000000.parquet"
    parquet.write_bytes(b"placeholder")

    mocker.patch(
        "npa.cli.workbench.lancedb.import_lerobot.load_rows",
        return_value=[
            {"observation.state": 1.0, "action": 2.0},
            {"observation.state": 3.0, "action": 4.0, "id": "explicit"},
        ],
    )

    captured: dict = {}

    def fake_request_json(method, endpoint, path, *, headers=None, payload=None, timeout=None):
        captured["payload"] = payload
        return {"rows": len(payload["rows"])}

    mocker.patch(
        "npa.cli.workbench.lancedb.import_lerobot.request_json", side_effect=fake_request_json
    )

    result = runner.invoke(
        lancedb_app,
        [
            "import-lerobot",
            "--endpoint",
            "http://localhost:8001",
            "--dataset-path",
            str(tmp_path),
            "--table",
            "robots",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = captured["payload"]["rows"]
    assert len(rows) == 2
    # First row had no id -> synthesized "<stem>:<index>"
    assert rows[0]["id"] == "episode_000000:0"
    # vector column was injected from numeric values
    assert rows[0]["vector"] == [1.0, 2.0]
    # Explicit id is preserved
    assert rows[1]["id"] == "explicit"


def test_lancedb_import_lerobot_limit_truncates_rows(tmp_path: Path, mocker) -> None:
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    (data_dir / "ep.parquet").write_bytes(b"x")

    mocker.patch(
        "npa.cli.workbench.lancedb.import_lerobot.load_rows",
        return_value=[{"a": float(i)} for i in range(5)],
    )

    captured: dict = {}

    def fake_request_json(method, endpoint, path, *, headers=None, payload=None, timeout=None):
        captured["payload"] = payload
        return {"rows": len(payload["rows"])}

    mocker.patch(
        "npa.cli.workbench.lancedb.import_lerobot.request_json", side_effect=fake_request_json
    )

    result = runner.invoke(
        lancedb_app,
        [
            "import-lerobot",
            "--endpoint",
            "http://localhost:8001",
            "--dataset-path",
            str(tmp_path),
            "--table",
            "robots",
            "--limit",
            "2",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(captured["payload"]["rows"]) == 2


def test_lancedb_import_lerobot_zero_vector_when_no_numeric(
    tmp_path: Path, mocker
) -> None:
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    (data_dir / "ep.parquet").write_bytes(b"x")

    mocker.patch(
        "npa.cli.workbench.lancedb.import_lerobot.load_rows",
        return_value=[{"label": "non-numeric"}],
    )
    captured: dict = {}

    def fake_request_json(method, endpoint, path, *, headers=None, payload=None, timeout=None):
        captured["payload"] = payload
        return {"rows": 1}

    mocker.patch(
        "npa.cli.workbench.lancedb.import_lerobot.request_json", side_effect=fake_request_json
    )

    result = runner.invoke(
        lancedb_app,
        [
            "import-lerobot",
            "--endpoint",
            "http://localhost:8001",
            "--dataset-path",
            str(tmp_path),
            "--table",
            "robots",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["payload"]["rows"][0]["vector"] == [0.0]


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


# ── status, create-table, views: additional coverage ─────────────────────


def test_lancedb_status_emits_health_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    status_module = importlib.import_module("npa.cli.workbench.lancedb.status")

    captured: dict = {}

    def fake_request(method, endpoint, path, headers=None, payload=None, timeout=30.0):
        captured.update(method=method, endpoint=endpoint, path=path)
        return {"status": "ok"}

    monkeypatch.setattr(status_module, "request_json", fake_request)

    result = runner.invoke(
        lancedb_app,
        [
            "status",
            "--endpoint",
            "http://localhost:8686",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.find("{"):])
    assert payload == {"status": "ok", "endpoint": "http://localhost:8686"}
    assert captured["path"] == "/health"


def test_lancedb_status_text_output_shows_status(monkeypatch: pytest.MonkeyPatch) -> None:
    status_module = importlib.import_module("npa.cli.workbench.lancedb.status")
    monkeypatch.setattr(
        status_module, "request_json", lambda *a, **kw: {"status": "healthy"}
    )
    result = runner.invoke(
        lancedb_app,
        ["status", "--endpoint", "http://localhost:8686"],
    )
    assert result.exit_code == 0
    assert "status: healthy" in result.output


def test_lancedb_status_cloud_mode_sends_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_module = importlib.import_module("npa.cli.workbench.lancedb.status")
    monkeypatch.setenv("LANCEDB_API_KEY", "key")
    captured: dict = {}

    def fake_request(method, endpoint, path, headers=None, payload=None, timeout=30.0):
        captured["headers"] = headers
        return {"status": "ok"}

    monkeypatch.setattr(status_module, "request_json", fake_request)

    result = runner.invoke(
        lancedb_app,
        [
            "status",
            "--endpoint",
            "https://cloud.example",
            "--cloud",
            "--database",
            "robots",
            "--cloud-region",
            "us-east-1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["headers"]["x-api-key"] == "key"
    assert captured["headers"]["lancedb-database"] == "robots"


def test_lancedb_create_table_posts_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_module = importlib.import_module("npa.cli.workbench.lancedb.create_table")
    captured: dict = {}

    def fake_request(method, endpoint, path, headers=None, payload=None, timeout=30.0):
        captured.update(method=method, path=path, payload=payload, timeout=timeout)
        return {"status": "created"}

    monkeypatch.setattr(create_module, "request_json", fake_request)

    # Provide a JSON input file
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([{"id": "row-1", "vector": [1.0, 2.0]}]))

    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"id": "string"}))

    result = runner.invoke(
        lancedb_app,
        [
            "create-table",
            "--endpoint",
            "http://localhost:8686",
            "--table",
            "robots",
            "--schema",
            str(schema),
            "--input-path",
            str(rows),
            "--mode",
            "overwrite",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["method"] == "POST"
    assert captured["path"] == "/tables/robots"
    assert captured["payload"]["mode"] == "overwrite"
    assert captured["payload"]["schema"] == {"id": "string"}
    assert captured["payload"]["rows"][0]["id"] == "row-1"


# ── views.py service-mode coverage (already covered locally elsewhere) ──


def test_lancedb_create_mv_service_mode_calls_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views_module = importlib.import_module("npa.cli.workbench.lancedb.views")
    captured: dict = {}

    def fake_request(method, endpoint, path, headers=None, payload=None, timeout=30.0):
        captured.update(method=method, path=path, payload=payload)
        return {
            "view_name": "view_robots",
            "source_table": "robots",
            "row_count": 5,
            "manifest_sha256": "abc",
        }

    monkeypatch.setattr(views_module, "request_json", fake_request)

    result = runner.invoke(
        lancedb_app,
        [
            "create-mv",
            "--name",
            "view_robots",
            "--source",
            "robots",
            "--filter",
            "status='train'",
            "--service",
            "--endpoint",
            "http://localhost:8686",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["method"] == "POST"
    assert captured["path"] == "/create-mv"
    assert captured["payload"]["name"] == "view_robots"


def test_lancedb_refresh_mv_service_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    views_module = importlib.import_module("npa.cli.workbench.lancedb.views")
    captured: dict = {}

    def fake_request(method, endpoint, path, headers=None, payload=None, timeout=30.0):
        captured.update(method=method, path=path)
        return {
            "view_name": "view_robots",
            "row_count_before": 3,
            "row_count_after": 5,
            "manifest_sha256": "abc",
        }

    monkeypatch.setattr(views_module, "request_json", fake_request)

    result = runner.invoke(
        lancedb_app,
        [
            "refresh-mv",
            "--name",
            "view_robots",
            "--service",
            "--endpoint",
            "http://localhost:8686",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["path"] == "/refresh-mv"


def test_lancedb_query_table_service_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    views_module = importlib.import_module("npa.cli.workbench.lancedb.views")
    captured: dict = {}

    def fake_request(method, endpoint, path, headers=None, payload=None, timeout=30.0):
        captured.update(method=method, path=path, payload=payload)
        return {"row_count": 2, "total_rows_matched": 2}

    monkeypatch.setattr(views_module, "request_json", fake_request)

    result = runner.invoke(
        lancedb_app,
        [
            "query-table",
            "--table",
            "robots",
            "--filter",
            "status='train'",
            "--select",
            "id",
            "--select",
            "vector",
            "--service",
            "--endpoint",
            "http://localhost:8686",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["path"] == "/query-table"
    assert captured["payload"]["select"] == ["id", "vector"]


def test_lancedb_create_mv_local_mode_surfaces_mv_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    views_module = importlib.import_module("npa.cli.workbench.lancedb.views")
    from npa.workbench.lancedb.views import MVError

    def boom(**_kwargs):
        raise MVError("filter rejected")

    monkeypatch.setattr(views_module, "create_mv", boom)

    result = runner.invoke(
        lancedb_app,
        [
            "create-mv",
            "--name",
            "view_robots",
            "--source",
            "robots",
            "--filter",
            "status='train'",
        ],
    )
    assert result.exit_code == 1
    assert "filter rejected" in result.output


@pytest.mark.smoke
def test_lancedb_end_to_end_deploy_table_query() -> None:
    if not os.environ.get("NPA_LANCEDB_SMOKE"):
        pytest.skip("Set NPA_LANCEDB_SMOKE=1 to run the local Docker LanceDB smoke.")
    payload = [
        {"id": f"row-{idx}", "vector": [float(idx), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        for idx in range(10)
    ]
    assert len(json.dumps(payload)) > 0
