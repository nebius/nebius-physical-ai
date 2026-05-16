from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from npa.cli.workbench.lancedb import app as lancedb_app
from npa.workbench.lancedb.backfill import (
    BackfillTableNotFoundError,
    MissingDependencyError,
    UnknownUDFError,
    backfill_column,
)
from npa.workbench.lancedb.bdd100k_import import import_bdd100k
from npa.workbench.lancedb.bdd100k_udfs import (
    BDD100K_UDFS,
    udf_dhash,
    udf_has_person,
    udf_has_rider,
    udf_is_duplicate,
    udf_person_bbox_area_pct,
)
from npa.workbench.lancedb.server import create_app


runner = CliRunner()


@pytest.mark.parametrize(
    ("udf_name", "expected"),
    [
        ("has_person", [True, False, False, False]),
        ("has_rider", [False, True, False, False]),
        ("person_bbox_area_pct", pytest.approx([0.01, 0.0, 0.0, 0.0])),
    ],
)
def test_metadata_udfs_produce_expected_outputs(udf_name: str, expected: object) -> None:
    batch = _known_batch()

    values = BDD100K_UDFS[udf_name].function(batch).to_pylist()

    assert values == expected


def test_has_person_and_has_rider_handle_empty_categories() -> None:
    batch = _known_batch()

    assert udf_has_person(batch).to_pylist()[2:] == [False, False]
    assert udf_has_rider(batch).to_pylist()[2:] == [False, False]


def test_person_bbox_area_pct_sums_multiple_person_boxes() -> None:
    batch = pa.table(
        {
            "image_id": ["multi"],
            "width": pa.array([100], type=pa.int32()),
            "height": pa.array([100], type=pa.int32()),
            "ann_categories": pa.array([["person", "person", "car"]], type=pa.list_(pa.string())),
            "ann_bboxes": pa.array(
                [[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 40.0], [0.0, 0.0, 100.0, 100.0]]],
                type=pa.list_(pa.list_(pa.float32())),
            ),
        }
    ).to_batches()[0]

    assert udf_person_bbox_area_pct(batch).to_pylist() == pytest.approx([0.03])


def test_dhash_is_deterministic_for_same_image() -> None:
    batch = pa.table({"image_bytes": pa.array([_jpeg_bytes(), _jpeg_bytes()], type=pa.large_binary())}).to_batches()[0]

    first = udf_dhash(batch).to_pylist()
    second = udf_dhash(batch).to_pylist()

    assert first == second
    assert first[0] == first[1]


def test_is_duplicate_respects_threshold() -> None:
    batch = pa.table({"image_id": ["a", "b", "c"], "dhash": [0, 3, 1024]}).to_batches()[0]

    assert udf_is_duplicate(batch, hamming_threshold=1).to_pylist() == [False, False, True]
    assert udf_is_duplicate(batch, hamming_threshold=2).to_pylist() == [False, True, True]


def test_backfill_is_idempotent_and_force_recomputes(tmp_path: Path) -> None:
    import_bdd100k(synthetic=6, synthetic_seed=3, lance_uri=str(tmp_path / "db"), table="bdd")

    first = backfill_column(lance_uri=str(tmp_path / "db"), table="bdd", udf="has_person", batch_size=2)
    second = backfill_column(lance_uri=str(tmp_path / "db"), table="bdd", udf="has_person", batch_size=2)
    forced = backfill_column(lance_uri=str(tmp_path / "db"), table="bdd", udf="has_person", force=True, batch_size=2)

    assert first.rows_updated == 6
    assert first.rows_skipped == 0
    assert first.column_added is True
    assert second.rows_updated == 0
    assert second.rows_skipped == 6
    assert second.manifest_sha256 == first.manifest_sha256
    assert forced.rows_updated == 6
    assert forced.rows_skipped == 0
    assert forced.manifest_sha256 == first.manifest_sha256


def test_backfill_missing_dependency_and_unknown_udf_raise_typed_errors(tmp_path: Path) -> None:
    import_bdd100k(synthetic=2, synthetic_seed=4, lance_uri=str(tmp_path / "db"), table="bdd")

    with pytest.raises(MissingDependencyError, match="dhash"):
        backfill_column(lance_uri=str(tmp_path / "db"), table="bdd", udf="is_duplicate")

    with pytest.raises(UnknownUDFError, match="unknown UDF"):
        backfill_column(lance_uri=str(tmp_path / "db"), table="bdd", udf="not_a_udf")


def test_backfill_missing_table_raises_typed_error(tmp_path: Path) -> None:
    with pytest.raises(BackfillTableNotFoundError, match="not found"):
        backfill_column(lance_uri=str(tmp_path / "db"), table="missing", udf="has_person")


def test_sdk_local_matches_direct_module_call(tmp_path: Path) -> None:
    from npa.workbench.lancedb import backfill as sdk_backfill

    import_bdd100k(synthetic=5, synthetic_seed=9, lance_uri=str(tmp_path / "direct-db"), table="bdd")
    import_bdd100k(synthetic=5, synthetic_seed=9, lance_uri=str(tmp_path / "sdk-db"), table="bdd")

    direct = backfill_column(lance_uri=str(tmp_path / "direct-db"), table="bdd", udf="has_person")
    sdk = sdk_backfill(lance_uri=str(tmp_path / "sdk-db"), table="bdd", udf="has_person")

    assert sdk.rows_updated == direct.rows_updated
    assert sdk.rows_skipped == direct.rows_skipped
    assert sdk.manifest_sha256 == direct.manifest_sha256


def test_sdk_service_mode_matches_http_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.workbench.lancedb as sdk_module

    app = create_app(storage_path=str(tmp_path / "service-root"), auth_mode="none")
    client = TestClient(app)
    import_response = client.post(
        "/import-bdd100k",
        json={"synthetic": 5, "synthetic_seed": 15, "lance_uri": str(tmp_path / "db"), "table": "bdd"},
    )
    assert import_response.status_code == 200
    response = client.post("/backfill", json={"lance_uri": str(tmp_path / "db"), "table": "bdd", "udf": "has_person"})
    assert response.status_code == 200
    endpoint_payload = response.json()

    monkeypatch.setattr(sdk_module, "_post_json", lambda **kwargs: endpoint_payload)
    sdk_result = sdk_module.backfill(
        service=True,
        endpoint="http://lancedb.example",
        lance_uri=str(tmp_path / "mocked-db"),
        table="bdd",
        udf="has_person",
    )

    assert sdk_result.to_dict() == endpoint_payload


def test_cli_local_outputs_json(tmp_path: Path) -> None:
    import_bdd100k(synthetic=4, synthetic_seed=20, lance_uri=str(tmp_path / "db"), table="bdd_cli")

    result = runner.invoke(
        lancedb_app,
        [
            "backfill",
            "--udf",
            "has_person",
            "--table",
            "bdd_cli",
            "--lance-uri",
            str(tmp_path / "db"),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["table"] == "bdd_cli"
    assert payload["udf"] == "has_person"
    assert payload["rows_updated"] == 4
    assert payload["manifest_sha256"]


def test_cli_service_calls_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    import npa.cli.workbench.lancedb.backfill as backfill_module

    seen = {}

    def fake_request(method: str, endpoint: str, path: str, **kwargs):
        seen.update({"method": method, "endpoint": endpoint, "path": path, **kwargs})
        return {
            "table": "bdd_service",
            "lance_uri": "s3://bucket/lancedb/bdd100k/",
            "rows_updated": 3,
            "rows_skipped": 0,
            "table_version_before": 1,
            "table_version_after": 3,
            "udf": "has_person",
            "output_column": "has_person",
            "column_added": True,
            "duration_ms": 10,
            "manifest_sha256": "abc",
        }

    monkeypatch.setattr(backfill_module, "request_json", fake_request)

    result = runner.invoke(
        lancedb_app,
        [
            "backfill",
            "--service",
            "--endpoint",
            "http://localhost:8686",
            "--udf",
            "has_person",
            "--table",
            "bdd_service",
        ],
    )

    assert result.exit_code == 0
    assert seen["method"] == "POST"
    assert seen["path"] == "/backfill"
    assert seen["payload"]["udf"] == "has_person"
    assert json.loads(result.output)["rows_updated"] == 3


def test_api_cli_sdk_manifest_parity(tmp_path: Path) -> None:
    from npa.workbench.lancedb import backfill as sdk_backfill

    app = create_app(storage_path=str(tmp_path / "service-root"), auth_mode="none")
    client = TestClient(app)
    manifests: list[str] = []
    for mode in ("api", "cli", "sdk"):
        table = f"bdd_{mode}"
        lance_uri = str(tmp_path / mode)
        imported = client.post(
            "/import-bdd100k",
            json={"synthetic": 7, "synthetic_seed": 44, "lance_uri": lance_uri, "table": table},
        )
        assert imported.status_code == 200
        if mode == "api":
            response = client.post("/backfill", json={"lance_uri": lance_uri, "table": table, "udf": "has_person"})
            assert response.status_code == 200
            manifests.append(response.json()["manifest_sha256"])
        elif mode == "cli":
            result = runner.invoke(
                lancedb_app,
                ["backfill", "--udf", "has_person", "--table", table, "--lance-uri", lance_uri],
            )
            assert result.exit_code == 0
            manifests.append(json.loads(result.output)["manifest_sha256"])
        else:
            result = sdk_backfill(lance_uri=lance_uri, table=table, udf="has_person")
            manifests.append(result.manifest_sha256)

    assert len(set(manifests)) == 1


def test_endpoint_maps_backfill_errors_to_status_codes(tmp_path: Path) -> None:
    app = create_app(storage_path=str(tmp_path / "service-root"), auth_mode="none")
    client = TestClient(app)
    imported = client.post(
        "/import-bdd100k",
        json={"synthetic": 3, "synthetic_seed": 55, "lance_uri": str(tmp_path / "db"), "table": "bdd"},
    )
    assert imported.status_code == 200

    assert client.post("/backfill", json={"lance_uri": str(tmp_path / "db"), "table": "bdd", "udf": "not_a_udf"}).status_code == 422
    assert client.post("/backfill", json={"lance_uri": str(tmp_path / "db"), "table": "missing", "udf": "has_person"}).status_code == 404
    assert client.post("/backfill", json={"lance_uri": str(tmp_path / "db"), "table": "bdd", "udf": "is_duplicate"}).status_code == 409


def _known_batch() -> pa.RecordBatch:
    return pa.table(
        {
            "image_id": pa.array(["row-1", "row-2", "row-3", "row-4"], type=pa.string()),
            "image_bytes": pa.array([_jpeg_bytes(), _jpeg_bytes(), _jpeg_bytes(), _jpeg_bytes()], type=pa.large_binary()),
            "width": pa.array([100, 100, 100, 100], type=pa.int32()),
            "height": pa.array([100, 100, 100, 100], type=pa.int32()),
            "ann_categories": pa.array(
                [["person", "car"], ["rider"], [], None],
                type=pa.list_(pa.string()),
            ),
            "ann_bboxes": pa.array(
                [
                    [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 20.0, 20.0]],
                    [[0.0, 0.0, 5.0, 5.0]],
                    [],
                    None,
                ],
                type=pa.list_(pa.list_(pa.float32())),
            ),
        }
    ).to_batches()[0]


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (32, 16), (120, 40, 200))
    draw = ImageDraw.Draw(image)
    draw.line([0, 0, 31, 15], fill=(255, 255, 255), width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
