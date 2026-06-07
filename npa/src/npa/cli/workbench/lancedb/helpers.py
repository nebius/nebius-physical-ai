"""Shared helpers for the LanceDB Workbench CLI."""

from __future__ import annotations

import json
import math
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import typer

from npa.clients.config import default_project_name, default_workbench_name, list_projects
from npa.clients.credentials import load_credentials
from npa.deploy.images import DEFAULT_CONTAINER_REGISTRY, container_image_for_tool

LANCEDB_VERSION = "0.30.2"
DEFAULT_PORT = 8686
DEFAULT_TOKEN_ENV = "LANCEDB_TOKEN"
DEFAULT_API_KEY_ENV = "LANCEDB_API_KEY"
DEFAULT_CONTAINER_IMAGE = container_image_for_tool("lancedb", registry=DEFAULT_CONTAINER_REGISTRY)
DEFAULT_CONTAINER_NAME = "npa-lancedb"
TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class LanceDBRuntime(str, Enum):
    vm = "vm"
    container = "container"
    byovm = "byovm"
    cloud = "cloud"
    serverless = "serverless"


def fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


def emit(payload: dict[str, Any], *, output: OutputFormat, text: str | None = None) -> None:
    if output == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(text if text is not None else _text_lines(payload))


def _text_lines(payload: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in payload.items())


def validate_port(port: int) -> int:
    if port < 1024 or port > 65535:
        fail("--port must be between 1024 and 65535")
    return port


def validate_limit(limit: int) -> int:
    if limit < 1 or limit > 10000:
        fail("--limit must be between 1 and 10000")
    return limit


def validate_top_k(top_k: int) -> int:
    if top_k < 1 or top_k > 1000:
        fail("--top-k must be between 1 and 1000")
    return top_k


def validate_storage_path(storage_path: str, *, required: bool = True) -> str:
    value = storage_path.strip()
    if not value:
        if required:
            fail("--storage-path is required for vm, container, and byovm runtimes")
        return ""
    if value.startswith("s3://"):
        parsed = urlparse(value)
        if not parsed.netloc:
            fail("--storage-path S3 URI must include a bucket")
        return value
    if Path(value).is_absolute():
        return value
    fail("--storage-path must be an s3:// URI or an absolute local path")
    return value


def validate_endpoint(endpoint: str) -> str:
    value = endpoint.strip()
    if not value:
        fail("--endpoint is required")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        fail("--endpoint must be an http:// or https:// URL")
    return value.rstrip("/")


def resolve_endpoint(endpoint: str = "") -> str:
    if endpoint:
        return validate_endpoint(endpoint)

    projects = list_projects()
    project = default_project_name()
    workbench = default_workbench_name()
    candidate = (
        projects.get(project, {})
        .get("workbenches", {})
        .get(workbench, {})
    )
    if candidate.get("workbench_type") == "lancedb" and candidate.get("endpoint"):
        return validate_endpoint(str(candidate["endpoint"]))

    for project_cfg in projects.values():
        for workbench_cfg in project_cfg.get("workbenches", {}).values():
            if workbench_cfg.get("workbench_type") == "lancedb" and workbench_cfg.get("endpoint"):
                return validate_endpoint(str(workbench_cfg["endpoint"]))

    fail("--endpoint is required; no saved LanceDB endpoint was found")
    return ""


def validate_table_name(table: str) -> str:
    value = table.strip()
    if not value:
        fail("--table is required")
    if not TABLE_NAME_RE.fullmatch(value):
        fail("--table must start with a letter or underscore and contain only letters, digits, dot, dash, or underscore")
    return value


def parse_vector(vector: str = "", vector_file: Path | None = None) -> list[float]:
    if vector and vector_file:
        fail("--vector and --vector-file are mutually exclusive")
    if vector_file is not None:
        if not vector_file.is_file():
            fail(f"--vector-file does not exist: {vector_file}")
        raw = vector_file.read_text()
    elif vector:
        raw = vector
    else:
        fail("--vector or --vector-file is required")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"Vector must be a JSON array: {exc.msg}")
    if not isinstance(data, list) or not data:
        fail("Vector must be a non-empty JSON array")

    values: list[float] = []
    for item in data:
        if not isinstance(item, (int, float)):
            fail("Vector values must be numbers")
        number = float(item)
        if not math.isfinite(number):
            fail("Vector values must be finite numbers")
        values.append(number)
    return values


def load_schema(schema: Path | None) -> dict[str, Any] | None:
    if schema is None:
        return None
    if not schema.is_file():
        fail(f"--schema does not exist: {schema}")
    try:
        data = json.loads(schema.read_text())
    except json.JSONDecodeError as exc:
        fail(f"--schema must be valid JSON: {exc.msg}")
    if not isinstance(data, dict):
        fail("--schema must be a JSON object")
    return data


def load_rows(input_path: str) -> list[dict[str, Any]]:
    if not input_path:
        return []
    if input_path.startswith("s3://"):
        return []
    path = Path(input_path)
    if not path.exists():
        fail(f"--input-path does not exist: {input_path}")
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for child in sorted(path.rglob("*.parquet")):
            rows.extend(load_rows(str(child)))
        if not rows:
            fail(f"--input-path directory contains no parquet files: {input_path}")
        return rows
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            fail("Reading parquet input requires pyarrow")
        return [_jsonable(row) for row in pq.read_table(path).to_pylist()]
    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return [_require_object(row) for row in data]
        if isinstance(data, dict):
            rows = data.get("rows")
            if isinstance(rows, list):
                return [_require_object(row) for row in rows]
        fail("--input-path JSON must be a list of objects or an object with a rows list")
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(_require_object(json.loads(line)))
        return rows
    fail("--input-path must be a local parquet, json, jsonl, directory, or s3:// URI")
    return []


def _require_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail("Input rows must be JSON objects")
    return _jsonable(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "as_py"):
        return _jsonable(value.as_py())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return value


def auth_headers(
    *,
    token_env: str = DEFAULT_TOKEN_ENV,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    database: str = "",
    cloud_region: str = "",
    cloud: bool = False,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if cloud:
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            fail(f"{api_key_env} is required for LanceDB Cloud")
        headers["x-api-key"] = api_key
        if database:
            headers["lancedb-database"] = database
        if cloud_region:
            headers["lancedb-region"] = cloud_region
        return headers

    token = os.environ.get(token_env, "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def request_json(
    method: str,
    endpoint: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    try:
        response = httpx.request(method, url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        fail(f"LanceDB request failed ({exc.response.status_code}): {detail}")
    except httpx.HTTPError as exc:
        fail(f"Cannot reach LanceDB endpoint {endpoint}: {exc}")
    try:
        data = response.json()
    except ValueError:
        fail(f"LanceDB endpoint returned non-JSON response from {url}")
    if not isinstance(data, dict):
        fail(f"LanceDB endpoint returned unexpected response from {url}")
    return data


def storage_env() -> dict[str, str]:
    creds = load_credentials()
    env = {
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID") or creds.s3_access_key_id,
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY") or creds.s3_secret_access_key,
        "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL") or creds.s3_endpoint,
        "AWS_REGION": os.environ.get("AWS_REGION", "auto"),
    }
    return {key: value for key, value in env.items() if value}


def container_image(image: str = "") -> str:
    return image.strip() or container_image_for_tool("lancedb")
