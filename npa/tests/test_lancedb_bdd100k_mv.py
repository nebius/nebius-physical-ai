from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from npa.cli.workbench.lancedb import app as lancedb_app
from npa.workbench.lancedb.server import create_app
from npa.workbench.lancedb.views import (
    MVConflictError,
    MV_REGISTRY_TABLE,
    MVResult,
    create_bdd100k_failure_mode_views,
    create_mv,
    query_table,
    refresh_mv,
)


runner = CliRunner()


def test_create_mv_creates_lance_table_with_expected_rows_and_schema(tmp_path: Path) -> None:
    lance_uri = _write_source(tmp_path)

    result = create_mv(
        name="person_train",
        source_table="bdd",
        filter_sql="has_person = true AND split = 'train'",
        lance_uri=lance_uri,
    )

    assert result.view_name == "person_train"
    assert result.row_count == 2
    assert result.manifest_sha256

    import lancedb

    view = lancedb.connect(lance_uri).open_table("person_train")
    assert view.count_rows() == 2
    assert {field.name for field in view.schema} == set(_source_schema().names)


def test_create_mv_is_idempotent_for_same_source_and_filter(tmp_path: Path) -> None:
    lance_uri = _write_source(tmp_path)

    first = create_mv(name="person_train", source_table="bdd", filter_sql="has_person = true", lance_uri=lance_uri)
    second = create_mv(name="person_train", source_table="bdd", filter_sql="has_person = true", lance_uri=lance_uri)

    assert second.row_count == first.row_count
    assert second.view_table_version == first.view_table_version
    assert second.manifest_sha256 == first.manifest_sha256


def test_create_mv_force_recomputes_existing_view(tmp_path: Path) -> None:
    lance_uri = _write_source(tmp_path)
    first = create_mv(name="person_train", source_table="bdd", filter_sql="has_person = true", lance_uri=lance_uri)
    _append_source_row(lance_uri, _row("train-004", split="train", has_person=True, timeofday="daytime"))

    forced = create_mv(
        name="person_train",
        source_table="bdd",
        filter_sql="has_person = true",
        lance_uri=lance_uri,
        force=True,
    )

    assert first.row_count == 3
    assert forced.row_count == 4
    assert forced.manifest_sha256 != first.manifest_sha256


def test_create_mv_name_collision_with_different_definition_errors(tmp_path: Path) -> None:
    lance_uri = _write_source(tmp_path)
    create_mv(name="person_train", source_table="bdd", filter_sql="has_person = true", lance_uri=lance_uri)

    with pytest.raises(MVConflictError, match="different source_table or filter_sql"):
        create_mv(name="person_train", source_table="bdd", filter_sql="has_rider = true", lance_uri=lance_uri)


def test_refresh_mv_recomputes_and_updates_registry(tmp_path: Path) -> None:
    lance_uri = _write_source(tmp_path)
    create_mv(name="person_train", source_table="bdd", filter_sql="has_person = true", lance_uri=lance_uri)
    before = _registry_row(lance_uri, "person_train")
    _append_source_row(lance_uri, _row("train-004", split="train", has_person=True, timeofday="night"))

    refreshed = refresh_mv(name="person_train", lance_uri=lance_uri)
    after = _registry_row(lance_uri, "person_train")

    assert refreshed.row_count_before == 3
    assert refreshed.row_count_after == 4
    assert after["row_count_at_last_refresh"] == 4
    assert after["last_refreshed"] >= before["last_refreshed"]


def test_query_table_filters_selects_limits_and_excludes_image_bytes_by_default(tmp_path: Path) -> None:
    lance_uri = _write_source(tmp_path)

    default = query_table(table="bdd", lance_uri=lance_uri, filter_sql="split = 'train'", limit=2)
    selected = query_table(
        table="bdd",
        lance_uri=lance_uri,
        filter_sql="split = 'train'",
        select=["image_id", "image_bytes"],
        limit=1,
    )
    unfiltered = query_table(table="bdd", lance_uri=lance_uri, select=["image_id"], limit=10)

    assert default.total_rows_matched == 3
    assert default.row_count == 2
    assert all("image_bytes" not in row for row in default.rows)
    assert selected.row_count == 1
    assert set(selected.rows[0]) == {"image_id", "image_bytes"}
    assert isinstance(selected.rows[0]["image_bytes"], str)
    assert unfiltered.total_rows_matched == 4


def test_create_bdd100k_failure_mode_views_uses_documented_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.workbench.lancedb.views as views_module

    calls: list[dict[str, Any]] = []

    def fake_create_mv(**kwargs: Any) -> MVResult:
        calls.append(kwargs)
        return MVResult(
            view_name=kwargs["name"],
            source_table=kwargs["source_table"],
            filter_sql=kwargs["filter_sql"],
            row_count=0,
            view_table_version=1,
            manifest_sha256=hashlib.sha256(kwargs["filter_sql"].encode("utf-8")).hexdigest(),
        )

    monkeypatch.setattr(views_module, "create_mv", fake_create_mv)

    results = create_bdd100k_failure_mode_views(
        lance_uri="s3://bucket/db/",
        source_table="bdd_source",
        distant_person_threshold=0.025,
    )

    assert [result.view_name for result in results] == [
        "bdd100k_rider_train",
        "bdd100k_nighttime_person_train",
        "bdd100k_distant_person_train",
    ]
    assert [call["filter_sql"] for call in calls] == [
        "has_rider = true AND split = 'train'",
        "timeofday = 'night' AND has_person = true AND split = 'train'",
        "has_person = true AND person_bbox_area_pct < 0.025 AND split = 'train'",
    ]


def test_api_cli_sdk_service_mode_parity_for_mv_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.cli.workbench.lancedb.views as cli_views
    import npa.workbench.lancedb as sdk_module

    lance_uri = _write_source(tmp_path)
    app = create_app(storage_path=str(tmp_path / "service-root"), auth_mode="none")
    client = TestClient(app)

    def client_request(method: str, endpoint: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = client.request(method, path, json=kwargs.get("payload"))
        assert response.status_code == 200, response.text
        return response.json()

    def sdk_post_json(**kwargs: Any) -> dict[str, Any]:
        response = client.post(kwargs.get("path", "/import-bdd100k"), json=kwargs["payload"])
        assert response.status_code == 200, response.text
        return response.json()

    monkeypatch.setattr(cli_views, "request_json", client_request)
    monkeypatch.setattr(sdk_module, "_post_json", sdk_post_json)

    payload = {
        "name": "person_train",
        "source_table": "bdd",
        "filter_sql": "has_person = true AND split = 'train'",
        "lance_uri": lance_uri,
    }
    api_create = client.post("/create-mv", json=payload)
    assert api_create.status_code == 200
    cli_create = runner.invoke(
        lancedb_app,
        [
            "create-mv",
            "--service",
            "--endpoint",
            "http://lancedb.example",
            "--name",
            payload["name"],
            "--source",
            payload["source_table"],
            "--filter",
            payload["filter_sql"],
            "--lance-uri",
            lance_uri,
        ],
    )
    assert cli_create.exit_code == 0, cli_create.output
    sdk_create = sdk_module.create_mv(service=True, endpoint="http://lancedb.example", **payload)

    create_manifests = {
        api_create.json()["manifest_sha256"],
        json.loads(cli_create.output)["manifest_sha256"],
        sdk_create.manifest_sha256,
    }
    assert len(create_manifests) == 1

    api_refresh = client.post("/refresh-mv", json={"name": "person_train", "lance_uri": lance_uri})
    assert api_refresh.status_code == 200
    cli_refresh = runner.invoke(
        lancedb_app,
        [
            "refresh-mv",
            "--service",
            "--endpoint",
            "http://lancedb.example",
            "--name",
            "person_train",
            "--lance-uri",
            lance_uri,
        ],
    )
    assert cli_refresh.exit_code == 0, cli_refresh.output
    sdk_refresh = sdk_module.refresh_mv(
        service=True,
        endpoint="http://lancedb.example",
        name="person_train",
        lance_uri=lance_uri,
    )
    refresh_manifests = {
        api_refresh.json()["manifest_sha256"],
        json.loads(cli_refresh.output)["manifest_sha256"],
        sdk_refresh.manifest_sha256,
    }
    assert len(refresh_manifests) == 1

    query_payload = {
        "table": "person_train",
        "lance_uri": lance_uri,
        "filter_sql": None,
        "select": ["image_id", "split"],
        "limit": 10,
    }
    api_query = client.post("/query-table", json=query_payload)
    assert api_query.status_code == 200
    cli_query = runner.invoke(
        lancedb_app,
        [
            "query-table",
            "--service",
            "--endpoint",
            "http://lancedb.example",
            "--table",
            "person_train",
            "--lance-uri",
            lance_uri,
            "--select",
            "image_id",
            "--select",
            "split",
            "--limit",
            "10",
        ],
    )
    assert cli_query.exit_code == 0, cli_query.output
    sdk_query = sdk_module.query_table(
        service=True,
        endpoint="http://lancedb.example",
        table="person_train",
        lance_uri=lance_uri,
        select=["image_id", "split"],
        limit=10,
    )

    row_hashes = {
        _row_hash(api_query.json()["rows"]),
        _row_hash(json.loads(cli_query.output)["rows"]),
        _row_hash(sdk_query.rows),
    }
    assert len(row_hashes) == 1


def _write_source(tmp_path: Path) -> str:
    import lancedb

    lance_uri = str(tmp_path / "db")
    rows = [
        _row("train-001", split="train", has_person=True, has_rider=False, timeofday="night", area=0.005),
        _row("train-002", split="train", has_person=True, has_rider=True, timeofday="daytime", area=0.02),
        _row("val-001", split="val", has_person=True, has_rider=False, timeofday="night", area=0.001),
        _row("train-003", split="train", has_person=False, has_rider=False, timeofday="night", area=0.0),
    ]
    lancedb.connect(lance_uri).create_table("bdd", data=pa.Table.from_pylist(rows, schema=_source_schema()), mode="overwrite")
    return lance_uri


def _append_source_row(lance_uri: str, row: dict[str, Any]) -> None:
    import lancedb

    table = lancedb.connect(lance_uri).open_table("bdd")
    table.add(pa.Table.from_pylist([row], schema=_source_schema()))


def _row(
    image_id: str,
    *,
    split: str,
    has_person: bool,
    has_rider: bool = False,
    timeofday: str = "daytime",
    area: float = 0.0,
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "image_bytes": image_id.encode("utf-8"),
        "split": split,
        "timeofday": timeofday,
        "has_person": has_person,
        "has_rider": has_rider,
        "person_bbox_area_pct": area,
    }


def _source_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("image_id", pa.string()),
            pa.field("image_bytes", pa.large_binary()),
            pa.field("split", pa.string()),
            pa.field("timeofday", pa.string()),
            pa.field("has_person", pa.bool_()),
            pa.field("has_rider", pa.bool_()),
            pa.field("person_bbox_area_pct", pa.float32()),
        ]
    )


def _registry_row(lance_uri: str, name: str) -> dict[str, Any]:
    import lancedb

    rows = lancedb.connect(lance_uri).open_table(MV_REGISTRY_TABLE).to_arrow().to_pylist()
    return next(row for row in rows if row["name"] == name)


def _row_hash(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: str(row.get("image_id", "")))
    payload = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
