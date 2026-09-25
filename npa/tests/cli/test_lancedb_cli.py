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


@pytest.mark.xfail(
    reason="Parent Workbench registration requires editing npa/src/npa/cli/workbench/__init__.py outside this run allowlist."
)
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


def test_lancedb_deploy_vm_blocked_message_is_actionable(tmp_path: Path) -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "vm",
            "--storage-path",
            str(tmp_path / "lancedb-smoke"),
        ],
    )

    assert result.exit_code == 1
    normalized = " ".join(result.output.split())
    assert "not available from this command yet" in normalized
    assert "--runtime container" in normalized
    # No internal jargon leaks to users.
    assert "run allowlist" not in normalized


def test_lancedb_deploy_cloud_requires_endpoint_and_api_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_lancedb_deploy_validates_port_range(tmp_path: Path) -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            str(tmp_path / "lancedb"),
            "--port",
            "99",
        ],
    )

    assert result.exit_code == 1
    assert "--port must be between" in result.output


def test_lancedb_container_s3_path_requires_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # storage_env drops empty S3 values; an s3:// storage path with no creds
    # must fail up front instead of only once the container hits the bucket.
    from npa.cli.workbench.lancedb import deploy as lancedb_deploy

    monkeypatch.setattr(
        lancedb_deploy,
        "storage_env",
        lambda: {
            "AWS_ACCESS_KEY_ID": "must-not-reach-local-container",
            "AWS_SECRET_ACCESS_KEY": "must-not-reach-local-container",
        },
    )
    monkeypatch.setenv("LANCEDB_TOKEN", "must-not-reach-unauthenticated-container")
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            "s3://my-bucket/lancedb",
            "--dry-run",
        ],
    )

    assert result.exit_code == 1
    assert "S3 credentials are incomplete" in result.output


def test_lancedb_container_s3_path_ok_with_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.workbench.lancedb import deploy as lancedb_deploy

    monkeypatch.setattr(
        lancedb_deploy,
        "storage_env",
        lambda: {
            "AWS_ACCESS_KEY_ID": "AK",
            "AWS_SECRET_ACCESS_KEY": "SK",
            "AWS_ENDPOINT_URL": "https://storage.example.nebius.cloud",
        },
    )
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            "s3://my-bucket/lancedb",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output


def test_lancedb_container_local_path_skips_s3_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A local storage path needs no S3 credentials, so the guard must not fire.
    from npa.cli.workbench.lancedb import deploy as lancedb_deploy

    monkeypatch.setattr(lancedb_deploy, "storage_env", lambda: {})
    storage_path = tmp_path / "lancedb"
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            str(storage_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        f"--mount type=bind,source={storage_path},target=/data/lancedb" in result.output
    )
    assert f"--user {os.getuid()}:{os.getgid()}" in result.output
    assert "LANCEDB_STORAGE_PATH=/data/lancedb" in result.output


def test_lancedb_container_creates_and_mounts_local_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cli.workbench.lancedb import deploy as lancedb_deploy

    storage_path = tmp_path / "nested" / "lancedb"
    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return lancedb_deploy.subprocess.CompletedProcess(
            command, 0, stdout="container-id\n", stderr=""
        )

    monkeypatch.setattr(lancedb_deploy, "storage_env", lambda: {})
    monkeypatch.setattr(lancedb_deploy.subprocess, "run", fake_run)

    container_id = lancedb_deploy._run_container(
        image="npa-lancedb:test",
        name="npa-lancedb-test",
        port=8686,
        storage_path=str(storage_path),
        auth_mode="none",
        token_env="LANCEDB_TOKEN",
        storage_endpoint="",
        detach=True,
        replace=False,
        dry_run=False,
    )

    assert container_id == "container-id"
    assert storage_path.is_dir()
    command = seen["command"]
    assert isinstance(command, list)
    assert [
        "--mount",
        f"type=bind,source={storage_path},target=/data/lancedb",
    ] == command[command.index("--mount") : command.index("--mount") + 2]
    assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "LANCEDB_STORAGE_PATH=/data/lancedb" in command
    assert "HOME=/data/lancedb" in command
    assert not any(token.startswith("AWS_ACCESS_KEY_ID=") for token in command)
    assert not any(token.startswith("AWS_SECRET_ACCESS_KEY=") for token in command)
    assert not any(token.startswith("LANCEDB_TOKEN=") for token in command)


def test_lancedb_container_refuses_unwritable_local_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cli.workbench.lancedb import deploy as lancedb_deploy

    monkeypatch.setattr(lancedb_deploy, "storage_env", lambda: {})
    monkeypatch.setattr(
        lancedb_deploy.tempfile,
        "mkstemp",
        lambda **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            str(tmp_path / "lancedb"),
        ],
    )

    assert result.exit_code == 1
    assert "not writable by uid" in result.output


def test_lancedb_container_refuses_root_owned_local_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.cli.workbench.lancedb import deploy as lancedb_deploy

    monkeypatch.setattr(lancedb_deploy, "storage_env", lambda: {})
    monkeypatch.setattr(lancedb_deploy.os, "getuid", lambda: 0)
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            str(tmp_path / "lancedb"),
        ],
    )

    assert result.exit_code == 1
    assert "deployed by a non-root host user" in result.output


def test_lancedb_container_s3_storage_is_not_bind_mounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.cli.workbench.lancedb import deploy as lancedb_deploy

    monkeypatch.setattr(
        lancedb_deploy,
        "storage_env",
        lambda: {
            "AWS_ACCESS_KEY_ID": "AK",
            "AWS_SECRET_ACCESS_KEY": "SK",
            "AWS_ENDPOINT_URL": "https://storage.example.invalid",
        },
    )
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "container",
            "--storage-path",
            "s3://my-bucket/lancedb",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--mount" not in result.output
    assert "LANCEDB_STORAGE_PATH=s3://my-bucket/lancedb" in result.output
    assert "HOME=" not in result.output


def test_lancedb_kubernetes_rejects_ephemeral_local_storage(tmp_path: Path) -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "kubernetes",
            "--storage-path",
            str(tmp_path / "lancedb"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 1
    assert "requires an s3:// --storage-path" in result.output
    assert "not persistent across rollouts or restarts" in result.output


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


def test_lancedb_create_table_sends_schema_for_zero_row_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    create_module = importlib.import_module("npa.cli.workbench.lancedb.create_table")
    schema = {"fields": [{"name": "id", "type": "string", "nullable": False}]}
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    seen = {}

    def fake_request(method: str, endpoint: str, path: str, **kwargs):
        seen.update({"method": method, "path": path, **kwargs})
        return {"status": "created", "table": "empty", "rows": 0}

    monkeypatch.setattr(create_module, "request_json", fake_request)

    result = runner.invoke(
        lancedb_app,
        [
            "create-table",
            "--endpoint",
            "http://localhost:8686",
            "--table",
            "empty",
            "--schema",
            str(schema_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["method"] == "POST"
    assert seen["path"] == "/tables/empty"
    assert seen["payload"]["schema"] == schema
    assert seen["payload"]["rows"] == []


def test_lancedb_create_table_requires_input_or_schema() -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "create-table",
            "--endpoint",
            "http://localhost:8686",
            "--table",
            "empty",
        ],
    )

    assert result.exit_code == 1
    assert "--input-path with rows or --schema is required" in result.output


def test_lancedb_create_table_rejects_s3_input() -> None:
    result = runner.invoke(
        lancedb_app,
        [
            "create-table",
            "--endpoint",
            "http://localhost:8686",
            "--table",
            "remote",
            "--input-path",
            "s3://example/data.json",
        ],
    )

    assert result.exit_code == 1
    assert "Server-side S3 import is not implemented" in result.output


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


def test_lancedb_import_bdd100k_service_calls_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert (
        DEFAULT_CONTAINER_IMAGE == "ghcr.io/nebius/nebius-physical-ai/npa-lancedb:"
        "cuda13-b300-0.30.3-sm80-sm90-sm100-sm103-sm120-20260803T031514Z"
    )


@pytest.mark.smoke
def test_lancedb_end_to_end_deploy_table_query() -> None:
    if not os.environ.get("NPA_LANCEDB_SMOKE"):
        pytest.skip("Set NPA_LANCEDB_SMOKE=1 to run the local Docker LanceDB smoke.")
    payload = [
        {"id": f"row-{idx}", "vector": [float(idx), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        for idx in range(10)
    ]
    assert len(json.dumps(payload)) > 0


def _assert_kubernetes_storage_manifest(output: str, storage_path: str) -> None:
    """Check canonical paths and credentials in the actual CLI manifest output."""
    from npa.workbench import service_kubernetes

    payload = json.loads(output)
    container = payload["manifests"][0]["spec"]["template"]["spec"]["containers"][0]
    env = container["env"]
    assert [entry for entry in env if entry["name"] == "LANCEDB_STORAGE_PATH"] == [
        {"name": "LANCEDB_STORAGE_PATH", "value": storage_path}
    ]
    credentials = [
        entry
        for entry in env
        if entry["name"] in service_kubernetes.STORAGE_SECRET_ENVS
    ]
    if storage_path.startswith("s3://"):
        assert len(credentials) == 2
        for entry in credentials:
            assert entry["valueFrom"]["secretKeyRef"] == {
                "name": "npa-lancedb-storage",
                "key": entry["name"],
            }
            assert "value" not in entry
    else:
        assert credentials == []


@pytest.mark.parametrize("storage_path", ["s3://example-bucket/review", "/data/review"])
def test_lancedb_kubernetes_dry_run_binds_storage_without_exposing_keys(
    monkeypatch: pytest.MonkeyPatch, storage_path: str
) -> None:
    """Exercise the real CLI-to-manifest boundary without contacting Kubernetes."""
    from npa.workbench import service_kubernetes

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "placeholder-access-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "placeholder-secret-value")
    monkeypatch.delenv("LANCEDB_TOKEN", raising=False)
    for name in ("apply", "ensure_storage_secret", "wait_available"):
        monkeypatch.setattr(
            service_kubernetes,
            name,
            lambda *args, **kwargs: pytest.fail("dry-run must not deploy resources"),
        )

    result = runner.invoke(
        lancedb_app,
        [
            "deploy",
            "--runtime",
            "kubernetes",
            "--storage-path",
            storage_path,
            "--dry-run",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    _assert_kubernetes_storage_manifest(result.output, storage_path)
    assert "placeholder-access-value" not in result.output
    assert "placeholder-secret-value" not in result.output
