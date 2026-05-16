from __future__ import annotations

import json
import hashlib
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import pytest

from npa.clients.project_credentials import s3_client_for_project, storage_env_for_project
from npa.workbench.lancedb.bdd100k_import import manifest_checksum


pytestmark = pytest.mark.e2e_serverless

REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE = os.environ.get("NPA_E2E_LANCEDB_IMAGE", "npa-lancedb:0.30.2")
PROJECT_ALIAS = os.environ.get("NPA_E2E_LANCEDB_PROJECT_ALIAS", "eu-north1")
PROJECT_ID = os.environ.get("NPA_E2E_LANCEDB_PROJECT_ID", "YOUR_PROJECT_ID")
BUCKET = os.environ.get("NPA_E2E_LANCEDB_BUCKET", "YOUR_S3_BUCKET")
POLL_INTERVAL = 5.0
MAX_WAIT = float(os.environ.get("NPA_E2E_LANCEDB_MAX_WAIT", "300"))


@pytest.fixture(autouse=True)
def _require_lancedb_e2e(request: pytest.FixtureRequest) -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("NPA_INTEGRATION_E2E not set")
    if _run(["docker", "--version"], timeout=30).returncode != 0:
        pytest.skip("Docker is required for the LanceDB container-mode e2e fallback")
    request.getfixturevalue("s3_write_access_required")


@pytest.fixture
def test_id() -> str:
    return f"w7lancedb-e2e-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def lancedb_service(test_id: str) -> Iterator[dict[str, str]]:
    port = _free_port()
    container = f"{test_id}-container"
    endpoint = f"http://localhost:{port}"
    storage_prefix = f"{test_id}/db/"
    storage_path = f"s3://{BUCKET}/{storage_prefix}"
    env = _lancedb_env()

    try:
        _ensure_image(env)
        deploy = _run_npa(
            [
                "workbench",
                "lancedb",
                "deploy",
                "--runtime",
                "container",
                "--storage-path",
                storage_path,
                "--port",
                str(port),
                "--auth-mode",
                "none",
                "--replace",
                "--container-name",
                container,
                "--image",
                IMAGE,
                "--output",
                "json",
            ],
            env=env,
            timeout=180,
        )
        assert deploy.returncode == 0, _format_result(deploy)
        payload = json.loads(deploy.stdout)
        endpoint = payload["endpoint"]
        assert urlparse(endpoint).port == port
        assert payload["storage_path"] == storage_path
        _wait_for_ready(endpoint, env)
        yield {
            "container": container,
            "endpoint": endpoint,
            "storage_prefix": storage_prefix,
            "storage_path": storage_path,
            "env": env,
        }
    finally:
        destroy = _run_npa(
            [
                "workbench",
                "lancedb",
                "deploy",
                "--runtime",
                "container",
                "--storage-path",
                storage_path,
                "--port",
                str(port),
                "--container-name",
                container,
                "--destroy",
                "--output",
                "json",
            ],
            env=env,
            timeout=120,
        )
        if destroy.returncode != 0:
            print(f"!!! ORPHANED LANCEDB CONTAINER {container}: {_format_result(destroy)}", flush=True)

        remaining = _run(
            ["docker", "ps", "-a", "--filter", f"name=^{container}$", "--format", "{{.Names}}"],
            timeout=30,
        )
        if remaining.stdout.strip():
            print(f"!!! ORPHANED LANCEDB CONTAINER {container}", flush=True)


def test_lancedb_container_lifecycle_with_nebius_s3(
    tmp_path: Path,
    test_id: str,
    lancedb_service: dict[str, str],
) -> None:
    """Container-mode fallback for the VM e2e: create, query, persist to Nebius S3."""
    endpoint = lancedb_service["endpoint"]
    env = lancedb_service["env"]
    table = f"robot_vectors_{uuid.uuid4().hex[:8]}"
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(_rows()), encoding="utf-8")

    created = _run_npa(
        [
            "workbench",
            "lancedb",
            "create-table",
            "--endpoint",
            endpoint,
            "--table",
            table,
            "--input-path",
            str(rows_path),
            "--mode",
            "overwrite",
            "--output",
            "json",
        ],
        env=env,
        timeout=180,
    )
    assert created.returncode == 0, _format_result(created)
    created_payload = json.loads(created.stdout)
    assert created_payload["table"] == table
    assert created_payload["rows"] == 100

    listed_payload = _wait_for_table_listed(endpoint, table, env)
    assert table in listed_payload["tables"]

    vector = json.dumps([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    query = _run_npa(
        [
            "workbench",
            "lancedb",
            "query",
            "--endpoint",
            endpoint,
            "--table",
            table,
            "--vector",
            vector,
            "--top-k",
            "5",
            "--output",
            "json",
        ],
        env=env,
    )
    assert query.returncode == 0, _format_result(query)
    query_payload = json.loads(query.stdout)
    assert query_payload["count"] >= 1
    assert len(query_payload["results"]) <= 5
    assert any(_distance_value(row) is not None for row in query_payload["results"])

    filtered = _run_npa(
        [
            "workbench",
            "lancedb",
            "query",
            "--endpoint",
            endpoint,
            "--table",
            table,
            "--vector",
            vector,
            "--filter",
            "label = 'robot'",
            "--top-k",
            "10",
            "--output",
            "json",
        ],
        env=env,
    )
    assert filtered.returncode == 0, _format_result(filtered)
    filtered_payload = json.loads(filtered.stdout)
    assert 1 <= filtered_payload["count"] <= 10
    assert all(row["label"] == "robot" for row in filtered_payload["results"])

    objects = _wait_for_s3_objects(lancedb_service["storage_prefix"], table)
    assert objects, f"no LanceDB objects found for {test_id}"


def test_lancedb_bdd100k_import_endpoint_and_sdk(
    test_id: str,
    lancedb_service: dict[str, str],
) -> None:
    """Import synthetic BDD100K rows through the service and verify Lance state."""
    endpoint = lancedb_service["endpoint"]
    env = lancedb_service["env"]
    table = f"bdd100k_{uuid.uuid4().hex[:8]}"
    storage_path = lancedb_service["storage_path"]

    imported = _run_npa(
        [
            "workbench",
            "lancedb",
            "import-bdd100k",
            "--service",
            "--endpoint",
            endpoint,
            "--synthetic",
            "50",
            "--synthetic-seed",
            "42",
            "--table",
            table,
            "--lance-uri",
            storage_path,
            "--output",
            "json",
        ],
        env=env,
        timeout=240,
    )
    assert imported.returncode == 0, _format_result(imported)
    payload = json.loads(imported.stdout)
    assert payload["table"] == table
    assert payload["total_rows"] == 50
    assert payload["rows_per_split"] == {"train": 39, "val": 11}

    _install_s3_env()
    import lancedb
    from npa.workbench.lancedb import import_bdd100k as sdk_import_bdd100k
    from npa.workbench.lancedb.bdd100k_import import bdd100k_schema, manifest_checksum

    db = lancedb.connect(storage_path)
    lance_table = db.open_table(table)
    assert lance_table.count_rows() == 50
    assert [(field.name, str(field.type)) for field in lance_table.schema] == [
        (field.name, str(field.type)) for field in bdd100k_schema()
    ]
    rows = lance_table.to_arrow().to_pylist()
    entries = [
        (row["image_id"], row["split"], hashlib.sha256(row["image_bytes"]).hexdigest())
        for row in rows
    ]
    assert manifest_checksum(entries) == payload["manifest_sha256"]

    sdk_table = f"{table}_sdk"
    sdk_result = sdk_import_bdd100k(
        service=True,
        endpoint=endpoint,
        synthetic=5,
        synthetic_seed=42,
        table=sdk_table,
        lance_uri=storage_path,
    )
    assert sdk_result.table == sdk_table
    assert sdk_result.total_rows == 5


def test_lancedb_bdd100k_import_endpoint_and_sdk_service_mode(
    test_id: str,
    lancedb_service: dict[str, str],
) -> None:
    endpoint = lancedb_service["endpoint"]
    storage_path = lancedb_service["storage_path"]
    endpoint_table = f"bdd_endpoint_{uuid.uuid4().hex[:8]}"
    sdk_table = f"bdd_sdk_{uuid.uuid4().hex[:8]}"

    import httpx

    response = httpx.post(
        f"{endpoint}/import-bdd100k",
        json={
            "table": endpoint_table,
            "lance_uri": storage_path,
            "synthetic": 50,
            "synthetic_seed": 42,
        },
        timeout=180.0,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total_rows"] == 50
    assert payload["table"] == endpoint_table

    from npa.workbench.lancedb import import_bdd100k as sdk_import_bdd100k

    sdk_result = sdk_import_bdd100k(
        service=True,
        endpoint=endpoint,
        table=sdk_table,
        lance_uri=storage_path,
        synthetic=50,
        synthetic_seed=42,
    )
    assert sdk_result.total_rows == 50

    endpoint_state = _lancedb_table_state(storage_path, endpoint_table)
    sdk_state = _lancedb_table_state(storage_path, sdk_table)
    assert endpoint_state["count"] == 50
    assert sdk_state["count"] == 50
    assert endpoint_state["schema"]["image_bytes"] == "large_binary"
    assert endpoint_state["manifest_sha256"] == payload["manifest_sha256"]
    assert sdk_state["manifest_sha256"] == sdk_result.manifest_sha256


def test_lancedb_bdd100k_backfill_endpoint_cli_sdk(
    test_id: str,
    lancedb_service: dict[str, str],
) -> None:
    endpoint = lancedb_service["endpoint"]
    env = lancedb_service["env"]
    storage_path = lancedb_service["storage_path"]
    table = f"bdd_backfill_{uuid.uuid4().hex[:8]}"

    import httpx

    imported = httpx.post(
        f"{endpoint}/import-bdd100k",
        json={
            "table": table,
            "lance_uri": storage_path,
            "synthetic": 50,
            "synthetic_seed": 42,
        },
        timeout=180.0,
    )
    assert imported.status_code == 200, imported.text
    assert imported.json()["total_rows"] == 50

    for udf in ["has_person", "has_rider", "person_bbox_area_pct", "dhash", "is_duplicate"]:
        response = httpx.post(
            f"{endpoint}/backfill",
            json={"table": table, "lance_uri": storage_path, "udf": udf},
            timeout=180.0,
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["rows_updated"] == 50
        assert payload["rows_skipped"] == 0
        assert payload["column_added"] is True
        assert payload["table_version_after"] > payload["table_version_before"]

    _install_s3_env()
    import lancedb

    lance_table = lancedb.connect(storage_path).open_table(table)
    schema = {field.name: str(field.type) for field in lance_table.schema}
    assert len(schema) == 17
    assert schema["has_person"] == "bool"
    assert schema["has_rider"] == "bool"
    assert schema["person_bbox_area_pct"] == "float"
    assert schema["dhash"] == "int64"
    assert schema["is_duplicate"] == "bool"
    rows = lance_table.to_arrow().to_pylist()
    for column in ["has_person", "has_rider", "person_bbox_area_pct", "dhash", "is_duplicate"]:
        assert sum(row[column] is not None for row in rows) == 50

    idempotent = httpx.post(
        f"{endpoint}/backfill",
        json={"table": table, "lance_uri": storage_path, "udf": "has_person"},
        timeout=180.0,
    )
    assert idempotent.status_code == 200, idempotent.text
    assert idempotent.json()["rows_updated"] == 0
    assert idempotent.json()["rows_skipped"] == 50

    forced = httpx.post(
        f"{endpoint}/backfill",
        json={"table": table, "lance_uri": storage_path, "udf": "has_person", "force": True},
        timeout=180.0,
    )
    assert forced.status_code == 200, forced.text
    assert forced.json()["rows_updated"] == 50
    assert forced.json()["rows_skipped"] == 0

    manifests: list[str] = []
    for mode in ["api", "cli", "sdk"]:
        parity_table = f"bdd_parity_{mode}_{uuid.uuid4().hex[:8]}"
        response = httpx.post(
            f"{endpoint}/import-bdd100k",
            json={
                "table": parity_table,
                "lance_uri": storage_path,
                "synthetic": 20,
                "synthetic_seed": 123,
            },
            timeout=180.0,
        )
        assert response.status_code == 200, response.text
        if mode == "api":
            backfilled = httpx.post(
                f"{endpoint}/backfill",
                json={"table": parity_table, "lance_uri": storage_path, "udf": "has_person"},
                timeout=180.0,
            )
            assert backfilled.status_code == 200, backfilled.text
            manifests.append(backfilled.json()["manifest_sha256"])
        elif mode == "cli":
            backfilled = _run_npa(
                [
                    "workbench",
                    "lancedb",
                    "backfill",
                    "--service",
                    "--endpoint",
                    endpoint,
                    "--table",
                    parity_table,
                    "--lance-uri",
                    storage_path,
                    "--udf",
                    "has_person",
                    "--output",
                    "json",
                ],
                env=env,
                timeout=180,
            )
            assert backfilled.returncode == 0, _format_result(backfilled)
            manifests.append(json.loads(backfilled.stdout)["manifest_sha256"])
        else:
            from npa.workbench.lancedb import backfill as sdk_backfill

            result = sdk_backfill(
                service=True,
                endpoint=endpoint,
                table=parity_table,
                lance_uri=storage_path,
                udf="has_person",
            )
            manifests.append(result.manifest_sha256)

    assert len(set(manifests)) == 1


def _run_npa(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return _run([_npa_executable(), *args], env=env, timeout=timeout)


def _run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _npa_executable() -> str:
    script = Path(sys.executable).with_name("npa")
    if script.exists():
        return str(script)
    return "npa"


def _lancedb_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(storage_env_for_project(PROJECT_ALIAS))
    env["AWS_REGION"] = env.get("AWS_REGION") or "auto"
    env["AWS_DEFAULT_REGION"] = env.get("AWS_DEFAULT_REGION") or env["AWS_REGION"]
    if env.get("AWS_ENDPOINT_URL"):
        env["AWS_ENDPOINT_URL_S3"] = env["AWS_ENDPOINT_URL"]
    env["NPA_E2E_SERVERLESS_PROJECT"] = os.environ.get("NPA_E2E_SERVERLESS_PROJECT", PROJECT_ID)
    return env


def _ensure_image(env: dict[str, str]) -> None:
    inspect = _run(["docker", "image", "inspect", IMAGE], env=env, timeout=30)
    if inspect.returncode == 0 and os.environ.get("NPA_E2E_LANCEDB_REBUILD_IMAGE") != "1":
        return
    build = _run(
        ["docker", "build", "-f", "npa/docker/lancedb/Dockerfile", "-t", IMAGE, "npa/"],
        env=env,
        timeout=600,
    )
    assert build.returncode == 0, _format_result(build)


def _wait_for_ready(endpoint: str, env: dict[str, str]) -> None:
    deadline = time.monotonic() + MAX_WAIT
    last: subprocess.CompletedProcess[str] | None = None
    while time.monotonic() < deadline:
        last = _run_npa(
            [
                "workbench",
                "lancedb",
                "status",
                "--endpoint",
                endpoint,
                "--output",
                "json",
            ],
            env=env,
            timeout=30,
        )
        if last.returncode == 0:
            payload = json.loads(last.stdout)
            if payload.get("status") == "ok":
                return
        time.sleep(POLL_INTERVAL)
    if last is None:
        pytest.fail(f"LanceDB endpoint did not become ready: {endpoint}")
    pytest.fail(_format_result(last))


def _wait_for_table_listed(endpoint: str, table: str, env: dict[str, str]) -> dict[str, object]:
    deadline = time.monotonic() + 60
    last: subprocess.CompletedProcess[str] | None = None
    while time.monotonic() < deadline:
        last = _run_npa(
            [
                "workbench",
                "lancedb",
                "list",
                "--endpoint",
                endpoint,
                "--prefix",
                table,
                "--output",
                "json",
            ],
            env=env,
        )
        if last.returncode == 0:
            payload = json.loads(last.stdout)
            if table in payload.get("tables", []):
                return payload
        time.sleep(3)
    if last is None:
        pytest.fail(f"LanceDB table was not listed: {table}")
    pytest.fail(_format_result(last))


def _wait_for_s3_objects(prefix: str, table: str) -> list[str]:
    client = s3_client_for_project(PROJECT_ALIAS)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        keys: list[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        matching = [key for key in keys if table in key and ".lance" in key]
        if matching:
            return matching
        time.sleep(3)
    return []


def _lancedb_table_state(storage_path: str, table_name: str) -> dict[str, object]:
    import hashlib

    _install_s3_env()
    import lancedb

    table = lancedb.connect(storage_path).open_table(table_name)
    rows = table.to_arrow().to_pylist()
    entries = [
        (row["image_id"], row["split"], hashlib.sha256(row["image_bytes"]).hexdigest())
        for row in rows
    ]
    return {
        "count": table.count_rows(),
        "schema": {field.name: str(field.type) for field in table.schema},
        "manifest_sha256": manifest_checksum(entries),
    }


def _install_s3_env() -> None:
    env = storage_env_for_project(PROJECT_ALIAS)
    os.environ.update(env)
    os.environ["AWS_REGION"] = os.environ.get("AWS_REGION") or "auto"
    os.environ["AWS_DEFAULT_REGION"] = os.environ.get("AWS_DEFAULT_REGION") or os.environ["AWS_REGION"]
    if os.environ.get("AWS_ENDPOINT_URL"):
        os.environ["AWS_ENDPOINT_URL_S3"] = os.environ["AWS_ENDPOINT_URL"]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(100):
        base = float(idx % 10) / 10.0
        rows.append(
            {
                "id": f"row-{idx:03d}",
                "label": "robot" if idx % 2 == 0 else "arm",
                "episode": idx // 10,
                "vector": [base + float(dim) / 10.0 for dim in range(8)],
            }
        )
    return rows


def _distance_value(row: dict[str, object]) -> float | None:
    for key in ("_distance", "distance"):
        value = row.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _format_result(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"command failed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
