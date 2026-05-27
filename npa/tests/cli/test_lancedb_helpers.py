"""Unit tests for `npa.cli.workbench.lancedb.helpers`."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import typer

from npa.cli.workbench.lancedb import helpers


# ── Validators ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("port", [1023, 0, -1, 70000])
def test_validate_port_rejects_out_of_range(port: int) -> None:
    with pytest.raises(typer.Exit):
        helpers.validate_port(port)


def test_validate_port_accepts_valid() -> None:
    assert helpers.validate_port(8080) == 8080


@pytest.mark.parametrize("limit", [0, -1, 10001])
def test_validate_limit_rejects_out_of_range(limit: int) -> None:
    with pytest.raises(typer.Exit):
        helpers.validate_limit(limit)


@pytest.mark.parametrize("top_k", [0, -3, 1001])
def test_validate_top_k_rejects_out_of_range(top_k: int) -> None:
    with pytest.raises(typer.Exit):
        helpers.validate_top_k(top_k)


def test_validate_storage_path_accepts_s3_uri() -> None:
    assert helpers.validate_storage_path("s3://bucket/path") == "s3://bucket/path"


def test_validate_storage_path_rejects_s3_without_bucket() -> None:
    with pytest.raises(typer.Exit):
        helpers.validate_storage_path("s3:///no-bucket")


def test_validate_storage_path_accepts_absolute_local_path(tmp_path: Path) -> None:
    assert helpers.validate_storage_path(str(tmp_path)) == str(tmp_path)


def test_validate_storage_path_rejects_relative_path() -> None:
    with pytest.raises(typer.Exit):
        helpers.validate_storage_path("relative/path")


def test_validate_storage_path_optional_empty_returns_empty() -> None:
    assert helpers.validate_storage_path("", required=False) == ""


def test_validate_storage_path_required_empty_fails() -> None:
    with pytest.raises(typer.Exit):
        helpers.validate_storage_path("")


@pytest.mark.parametrize(
    "endpoint", ["", "not-a-url", "ftp://example.com", "http://"]
)
def test_validate_endpoint_rejects_invalid(endpoint: str) -> None:
    with pytest.raises(typer.Exit):
        helpers.validate_endpoint(endpoint)


def test_validate_endpoint_strips_trailing_slash() -> None:
    assert (
        helpers.validate_endpoint("https://example.com/api/")
        == "https://example.com/api"
    )


# ── Endpoint resolution ───────────────────────────────────────────────────


def test_resolve_endpoint_uses_provided() -> None:
    assert helpers.resolve_endpoint("http://localhost:8080") == "http://localhost:8080"


def test_resolve_endpoint_picks_default_workbench(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers, "default_project_name", lambda: "proj")
    monkeypatch.setattr(helpers, "default_workbench_name", lambda: "wb")
    monkeypatch.setattr(
        helpers,
        "list_projects",
        lambda: {
            "proj": {
                "workbenches": {
                    "wb": {
                        "workbench_type": "lancedb",
                        "endpoint": "http://default.example",
                    }
                }
            }
        },
    )
    assert helpers.resolve_endpoint() == "http://default.example"


def test_resolve_endpoint_falls_back_to_any_lancedb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers, "default_project_name", lambda: "proj")
    monkeypatch.setattr(helpers, "default_workbench_name", lambda: "wb")
    monkeypatch.setattr(
        helpers,
        "list_projects",
        lambda: {
            "proj": {
                "workbenches": {
                    "other": {
                        "workbench_type": "lancedb",
                        "endpoint": "http://fallback.example",
                    }
                }
            }
        },
    )
    assert helpers.resolve_endpoint() == "http://fallback.example"


def test_resolve_endpoint_fails_when_no_lancedb_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helpers, "default_project_name", lambda: "proj")
    monkeypatch.setattr(helpers, "default_workbench_name", lambda: "wb")
    monkeypatch.setattr(helpers, "list_projects", lambda: {})

    with pytest.raises(typer.Exit):
        helpers.resolve_endpoint()


# ── Table-name validation ─────────────────────────────────────────────────


@pytest.mark.parametrize("table", ["", "  ", "1starts-with-digit", "has space", "bad/slash"])
def test_validate_table_name_rejects_invalid(table: str) -> None:
    with pytest.raises(typer.Exit):
        helpers.validate_table_name(table)


def test_validate_table_name_accepts_valid() -> None:
    assert helpers.validate_table_name("My_Table.v1-2") == "My_Table.v1-2"


# ── Vector parsing ────────────────────────────────────────────────────────


def test_parse_vector_from_string() -> None:
    assert helpers.parse_vector("[1, 2.5, 3]") == [1.0, 2.5, 3.0]


def test_parse_vector_from_file(tmp_path: Path) -> None:
    f = tmp_path / "v.json"
    f.write_text("[0.1, 0.2]")
    assert helpers.parse_vector("", vector_file=f) == [0.1, 0.2]


def test_parse_vector_mutually_exclusive(tmp_path: Path) -> None:
    f = tmp_path / "v.json"
    f.write_text("[1]")
    with pytest.raises(typer.Exit):
        helpers.parse_vector("[1]", vector_file=f)


def test_parse_vector_requires_input() -> None:
    with pytest.raises(typer.Exit):
        helpers.parse_vector("")


def test_parse_vector_file_must_exist(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        helpers.parse_vector("", vector_file=tmp_path / "missing.json")


def test_parse_vector_rejects_invalid_json() -> None:
    with pytest.raises(typer.Exit):
        helpers.parse_vector("[1, 2,")


def test_parse_vector_rejects_non_array() -> None:
    with pytest.raises(typer.Exit):
        helpers.parse_vector('{"a": 1}')


def test_parse_vector_rejects_empty_array() -> None:
    with pytest.raises(typer.Exit):
        helpers.parse_vector("[]")


def test_parse_vector_rejects_non_numeric_items() -> None:
    with pytest.raises(typer.Exit):
        helpers.parse_vector('[1, "two"]')


def test_parse_vector_rejects_non_finite_numbers() -> None:
    with pytest.raises(typer.Exit):
        helpers.parse_vector("[1, 1e500]")


# ── Schema loading ────────────────────────────────────────────────────────


def test_load_schema_none_returns_none() -> None:
    assert helpers.load_schema(None) is None


def test_load_schema_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        helpers.load_schema(tmp_path / "missing.json")


def test_load_schema_invalid_json_fails(tmp_path: Path) -> None:
    f = tmp_path / "s.json"
    f.write_text("{bad")
    with pytest.raises(typer.Exit):
        helpers.load_schema(f)


def test_load_schema_non_object_fails(tmp_path: Path) -> None:
    f = tmp_path / "s.json"
    f.write_text("[]")
    with pytest.raises(typer.Exit):
        helpers.load_schema(f)


def test_load_schema_valid(tmp_path: Path) -> None:
    f = tmp_path / "s.json"
    f.write_text('{"name": "string"}')
    assert helpers.load_schema(f) == {"name": "string"}


# ── load_rows ─────────────────────────────────────────────────────────────


def test_load_rows_empty_input_returns_empty_list() -> None:
    assert helpers.load_rows("") == []


def test_load_rows_s3_returns_empty_list() -> None:
    assert helpers.load_rows("s3://bucket/path") == []


def test_load_rows_missing_path_fails() -> None:
    with pytest.raises(typer.Exit):
        helpers.load_rows("/nope/does-not-exist-12345.json")


def test_load_rows_json_array(tmp_path: Path) -> None:
    f = tmp_path / "rows.json"
    f.write_text(json.dumps([{"a": 1}, {"a": 2}]))
    assert helpers.load_rows(str(f)) == [{"a": 1}, {"a": 2}]


def test_load_rows_json_with_rows_key(tmp_path: Path) -> None:
    f = tmp_path / "rows.json"
    f.write_text(json.dumps({"rows": [{"x": 1}]}))
    assert helpers.load_rows(str(f)) == [{"x": 1}]


def test_load_rows_json_invalid_shape_fails(tmp_path: Path) -> None:
    f = tmp_path / "rows.json"
    f.write_text(json.dumps({"not": "rows"}))
    with pytest.raises(typer.Exit):
        helpers.load_rows(str(f))


def test_load_rows_jsonl(tmp_path: Path) -> None:
    f = tmp_path / "rows.jsonl"
    f.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert helpers.load_rows(str(f)) == [{"a": 1}, {"a": 2}]


def test_load_rows_unsupported_suffix_fails(tmp_path: Path) -> None:
    f = tmp_path / "rows.txt"
    f.write_text("hello")
    with pytest.raises(typer.Exit):
        helpers.load_rows(str(f))


def test_load_rows_empty_directory_fails(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit):
        helpers.load_rows(str(tmp_path))


def test_load_rows_directory_recurses(tmp_path: Path) -> None:
    inner = tmp_path / "inner"
    inner.mkdir()
    (inner / "rows.json").write_text(json.dumps([{"a": 1}]))
    # Directory recursion path: rglob picks any *.parquet first, but if none,
    # falls through to "no parquet files". The implementation only rglobs
    # parquet for directories, so a JSON-only dir should fail.
    with pytest.raises(typer.Exit):
        helpers.load_rows(str(tmp_path))


def test_require_object_rejects_non_dict() -> None:
    with pytest.raises(typer.Exit):
        helpers._require_object(["not", "a", "dict"])


def test_jsonable_handles_lists_and_as_py() -> None:
    class FakeArrow:
        def as_py(self):
            return {"k": 1}

    class FakeNumpy:
        def tolist(self):
            return [1, 2]

    result = helpers._jsonable(
        {"arrow": FakeArrow(), "numpy": FakeNumpy(), "list": [1, 2]}
    )
    assert result == {"arrow": {"k": 1}, "numpy": [1, 2], "list": [1, 2]}


# ── auth_headers ──────────────────────────────────────────────────────────


def test_auth_headers_local_with_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANCEDB_TOKEN", "tok")
    assert helpers.auth_headers() == {"Authorization": "Bearer tok"}


def test_auth_headers_local_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANCEDB_TOKEN", raising=False)
    assert helpers.auth_headers() == {}


def test_auth_headers_cloud_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANCEDB_API_KEY", raising=False)
    with pytest.raises(typer.Exit):
        helpers.auth_headers(cloud=True)


def test_auth_headers_cloud_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANCEDB_API_KEY", "key")
    headers = helpers.auth_headers(
        cloud=True, database="robots", cloud_region="us-east-1"
    )
    assert headers == {
        "x-api-key": "key",
        "lancedb-database": "robots",
        "lancedb-region": "us-east-1",
    }


# ── request_json ──────────────────────────────────────────────────────────


def _install_fake_httpx(monkeypatch, *, status_code=200, body=None, raise_exc=None):
    def fake_request(method, url, headers=None, json=None, timeout=None):
        if raise_exc is not None:
            raise raise_exc
        return httpx.Response(
            status_code,
            json=body if body is not None else {},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx, "request", fake_request)


def test_request_json_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_httpx(monkeypatch, body={"ok": True})
    assert helpers.request_json("GET", "http://svc/", "/x") == {"ok": True}


def test_request_json_http_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_httpx(monkeypatch, status_code=500, body={"detail": "boom"})
    with pytest.raises(typer.Exit):
        helpers.request_json("GET", "http://svc", "/x")


def test_request_json_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_httpx(monkeypatch, raise_exc=httpx.ConnectError("nope"))
    with pytest.raises(typer.Exit):
        helpers.request_json("GET", "http://svc", "/x")


def test_request_json_non_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method, url, headers=None, json=None, timeout=None):
        return httpx.Response(200, content=b"<html>", request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "request", fake_request)
    with pytest.raises(typer.Exit):
        helpers.request_json("GET", "http://svc", "/x")


def test_request_json_non_object_response(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_httpx(monkeypatch, body=["list", "not", "obj"])
    with pytest.raises(typer.Exit):
        helpers.request_json("GET", "http://svc", "/x")


# ── storage_env / container_image / emit ─────────────────────────────────


def test_storage_env_uses_env_overrides_then_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.setattr(
        helpers,
        "load_credentials",
        lambda: SimpleNamespace(
            s3_access_key_id="creds-key",
            s3_secret_access_key="creds-secret",
            s3_endpoint="https://creds.example",
        ),
    )
    env = helpers.storage_env()
    assert env["AWS_ACCESS_KEY_ID"] == "env-key"
    assert env["AWS_SECRET_ACCESS_KEY"] == "creds-secret"
    assert env["AWS_ENDPOINT_URL"] == "https://creds.example"
    assert env["AWS_REGION"] == "auto"


def test_storage_env_drops_empty_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        helpers,
        "load_credentials",
        lambda: SimpleNamespace(
            s3_access_key_id="", s3_secret_access_key="", s3_endpoint=""
        ),
    )
    env = helpers.storage_env()
    # All empty -> only AWS_REGION remains (default "auto").
    assert env == {"AWS_REGION": "auto"}


def test_container_image_default_and_override() -> None:
    assert helpers.container_image() == helpers.DEFAULT_CONTAINER_IMAGE
    assert helpers.container_image("custom:tag") == "custom:tag"


def test_emit_json_format(capsys: pytest.CaptureFixture[str]) -> None:
    helpers.emit({"b": 1, "a": 2}, output=helpers.OutputFormat.json)
    captured = capsys.readouterr().out
    # Sorted keys
    assert '"a": 2' in captured and '"b": 1' in captured
    assert captured.index('"a"') < captured.index('"b"')


def test_emit_text_default_lines(capsys: pytest.CaptureFixture[str]) -> None:
    helpers.emit({"x": 1, "y": "z"}, output=helpers.OutputFormat.text)
    out = capsys.readouterr().out
    assert "x: 1" in out
    assert "y: z" in out


def test_emit_text_uses_provided_text(capsys: pytest.CaptureFixture[str]) -> None:
    helpers.emit({"k": 1}, output=helpers.OutputFormat.text, text="custom")
    assert capsys.readouterr().out.strip() == "custom"
