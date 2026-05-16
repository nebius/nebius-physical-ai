from __future__ import annotations

import io
import importlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from npa.cli.workbench.lancedb import app as lancedb_app
from npa.workbench.lancedb.backfill import (
    BackfillTableNotFoundError,
    GPUOOMAtMinimumBatchError,
    MissingDependencyError,
    UnknownUDFError,
    backfill_column,
)
from npa.workbench.lancedb.bdd100k_import import import_bdd100k
from npa.workbench.lancedb.bdd100k_udfs import (
    BDD100K_UDFS,
    CLIP_EMBEDDING_DIM,
    CLIP_OUTPUT_TYPE,
    udf_dhash,
    udf_has_person,
    udf_has_rider,
    udf_is_duplicate,
    udf_person_bbox_area_pct,
)
from npa.workbench.lancedb.server import create_app


runner = CliRunner()
backfill_module = importlib.import_module("npa.workbench.lancedb.backfill")


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


def test_clip_udf_registered_with_gpu_flag() -> None:
    spec = BDD100K_UDFS["clip_embedding"]

    assert spec.gpu is True
    assert spec.input_columns == ("image_bytes",)
    assert spec.output_column == "clip_embedding"
    assert spec.output_type == CLIP_OUTPUT_TYPE
    assert all(BDD100K_UDFS[name].gpu is False for name in ("has_person", "has_rider", "person_bbox_area_pct", "dhash", "is_duplicate"))


def test_clip_udf_output_schema_512_float32() -> None:
    from npa.workbench.lancedb import bdd100k_udfs as udfs

    values = [[0.1] * CLIP_EMBEDDING_DIM, [0.2] * CLIP_EMBEDDING_DIM]

    output = udfs._clip_vectors_to_array(values)

    assert output.type == CLIP_OUTPUT_TYPE
    assert output.to_pylist()[0] == pytest.approx([0.1] * CLIP_EMBEDDING_DIM)


def test_backfill_dispatches_gpu_udf_through_gpu_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_run_gpu_udf(table_obj, spec, **kwargs):
        seen.update({"spec": spec.name, **kwargs})
        return 2, 0

    monkeypatch.setattr(backfill_module, "_run_gpu_udf", fake_run_gpu_udf)
    import_bdd100k(synthetic=2, synthetic_seed=31, lance_uri=str(tmp_path / "db"), table="bdd")

    result = backfill_column(
        lance_uri=str(tmp_path / "db"),
        table="bdd",
        udf="clip_embedding",
        batch_size=7,
        device="cuda:0",
        precision="float32",
    )

    assert seen["spec"] == "clip_embedding"
    assert seen["batch_size"] == 7
    assert seen["device"] == "cuda:0"
    assert seen["precision"] == "float32"
    assert result.rows_updated == 2
    assert result.gpu_used is True


def test_backfill_preserves_cpu_udf_behavior() -> None:
    expected = {
        "has_person": (("ann_categories",), "has_person", pa.bool_(), ()),
        "has_rider": (("ann_categories",), "has_rider", pa.bool_(), ()),
        "person_bbox_area_pct": (("ann_categories", "ann_bboxes", "width", "height"), "person_bbox_area_pct", pa.float32(), ()),
        "dhash": (("image_bytes",), "dhash", pa.int64(), ()),
        "is_duplicate": (("image_id", "dhash"), "is_duplicate", pa.bool_(), ("dhash",)),
    }

    for name, (input_columns, output_column, output_type, dependencies) in expected.items():
        spec = BDD100K_UDFS[name]
        assert spec.input_columns == input_columns
        assert spec.output_column == output_column
        assert spec.output_type == output_type
        assert spec.dependencies == dependencies
        assert spec.gpu is False


def test_gpu_udf_batch_oom_fallback(tmp_path: Path) -> None:
    import lancedb

    calls: list[int] = []

    def fake_clip(batch: pa.RecordBatch, **kwargs) -> pa.Array:
        calls.append(batch.num_rows)
        if batch.num_rows > 2:
            raise RuntimeError("CUDA out of memory")
        return _fake_clip_array(batch)

    import_bdd100k(synthetic=4, synthetic_seed=32, lance_uri=str(tmp_path / "db"), table="bdd")
    table = lancedb.connect(str(tmp_path / "db")).open_table("bdd")
    spec = replace(BDD100K_UDFS["clip_embedding"], function=fake_clip)
    table.add_columns([pa.field(spec.output_column, spec.output_type)])
    batch = next(iter(table.search().select(["image_id", "image_bytes", "clip_embedding"]).to_batches(batch_size=4)))

    rows_updated, rows_skipped = backfill_module._backfill_gpu_batch(
        table,
        spec,
        batch,
        batch_size=4,
        force=False,
        device="cuda:0",
        precision="float32",
    )

    assert calls == [4, 2, 2]
    assert rows_updated == 4
    assert rows_skipped == 0


def test_gpu_udf_oom_at_minimum_batch_raises_halt(tmp_path: Path) -> None:
    import lancedb

    def fake_clip(batch: pa.RecordBatch, **kwargs) -> pa.Array:
        raise RuntimeError("CUDA out of memory")

    import_bdd100k(synthetic=1, synthetic_seed=33, lance_uri=str(tmp_path / "db"), table="bdd")
    table = lancedb.connect(str(tmp_path / "db")).open_table("bdd")
    spec = replace(BDD100K_UDFS["clip_embedding"], function=fake_clip)
    table.add_columns([pa.field(spec.output_column, spec.output_type)])
    batch = next(iter(table.search().select(["image_id", "image_bytes", "clip_embedding"]).to_batches(batch_size=1)))

    with pytest.raises(GPUOOMAtMinimumBatchError, match="HALT_GPU_OOM_AT_MINIMUM_BATCH"):
        backfill_module._backfill_gpu_batch(
            table,
            spec,
            batch,
            batch_size=1,
            force=False,
            device="cuda:0",
            precision="float32",
        )


def test_clip_backfill_is_idempotent_and_force_recomputes(tmp_path: Path) -> None:
    original = BDD100K_UDFS["clip_embedding"]
    BDD100K_UDFS["clip_embedding"] = replace(original, function=_fake_clip_array)
    try:
        import_bdd100k(synthetic=3, synthetic_seed=34, lance_uri=str(tmp_path / "db"), table="bdd")

        first = backfill_column(lance_uri=str(tmp_path / "db"), table="bdd", udf="clip_embedding", batch_size=2, precision="float32")
        second = backfill_column(lance_uri=str(tmp_path / "db"), table="bdd", udf="clip_embedding", batch_size=2, precision="float32")
        forced = backfill_column(
            lance_uri=str(tmp_path / "db"),
            table="bdd",
            udf="clip_embedding",
            batch_size=2,
            force_recompute=True,
            precision="float32",
        )
    finally:
        BDD100K_UDFS["clip_embedding"] = original

    assert first.rows_updated == 3
    assert first.rows_skipped == 0
    assert first.gpu_used is True
    assert second.rows_updated == 0
    assert second.rows_skipped == 3
    assert second.manifest_sha256 == first.manifest_sha256
    assert forced.rows_updated == 3
    assert forced.rows_skipped == 0
    assert forced.manifest_sha256 == first.manifest_sha256


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


def test_clip_api_cli_sdk_manifest_parity_with_mocked_udf(tmp_path: Path) -> None:
    from npa.workbench.lancedb import backfill as sdk_backfill

    original = BDD100K_UDFS["clip_embedding"]
    BDD100K_UDFS["clip_embedding"] = replace(original, function=_fake_clip_array)
    try:
        app = create_app(storage_path=str(tmp_path / "service-root"), auth_mode="none")
        client = TestClient(app)
        manifests: list[str] = []
        gpu_flags: list[bool] = []
        for mode in ("api", "cli", "sdk"):
            table = f"bdd_clip_{mode}"
            lance_uri = str(tmp_path / f"clip-{mode}")
            imported = client.post(
                "/import-bdd100k",
                json={"synthetic": 4, "synthetic_seed": 45, "lance_uri": lance_uri, "table": table},
            )
            assert imported.status_code == 200
            if mode == "api":
                response = client.post(
                    "/backfill",
                    json={"lance_uri": lance_uri, "table": table, "udf": "clip_embedding", "precision": "float32"},
                )
                assert response.status_code == 200
                payload = response.json()
                manifests.append(payload["manifest_sha256"])
                gpu_flags.append(payload["gpu_used"])
            elif mode == "cli":
                result = runner.invoke(
                    lancedb_app,
                    ["backfill", "--udf", "clip_embedding", "--table", table, "--lance-uri", lance_uri],
                )
                assert result.exit_code == 0
                payload = json.loads(result.output)
                manifests.append(payload["manifest_sha256"])
                gpu_flags.append(payload["gpu_used"])
            else:
                result = sdk_backfill(lance_uri=lance_uri, table=table, udf="clip_embedding", precision="float32")
                manifests.append(result.manifest_sha256)
                gpu_flags.append(result.gpu_used)
    finally:
        BDD100K_UDFS["clip_embedding"] = original

    assert len(set(manifests)) == 1
    assert gpu_flags == [True, True, True]


@pytest.mark.e2e
def test_clip_embedding_deployed_service_e2e() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("set NPA_INTEGRATION_E2E=1 to run deployed CLIP embedding validation")
    endpoint = os.environ.get("NPA_LANCEDB_ENDPOINT", "")
    if not endpoint:
        pytest.skip("NPA_LANCEDB_ENDPOINT is required for deployed CLIP embedding validation")

    from npa.workbench.lancedb import backfill as sdk_backfill
    from npa.workbench.lancedb import import_bdd100k as sdk_import_bdd100k
    from npa.workbench.lancedb import query_table as sdk_query_table

    run_id = os.environ.get("NPA_TEST_RUN_ID", "manual")
    lance_uri = os.environ.get(
        "NPA_TEST_LANCEDB_URI",
        f"s3://YOUR_S3_BUCKET/lancedb/_validation/clip-embedding-{run_id}/",
    )
    table = f"bdd_clip_{run_id.replace('-', '_')}"

    sdk_import_bdd100k(synthetic=100, synthetic_seed=51, lance_uri=lance_uri, table=table, service=True, endpoint=endpoint)
    cpu_result = sdk_backfill(lance_uri=lance_uri, table=table, udf="has_person", service=True, endpoint=endpoint)
    clip_result = sdk_backfill(lance_uri=lance_uri, table=table, udf="clip_embedding", batch_size=32, service=True, endpoint=endpoint)
    query = sdk_query_table(
        lance_uri=lance_uri,
        table=table,
        select=["image_id", "clip_embedding"],
        limit=10,
        service=True,
        endpoint=endpoint,
    )

    embeddings = [row["clip_embedding"] for row in query.rows]
    assert cpu_result.rows_updated == 100
    assert clip_result.rows_updated == 100
    assert clip_result.gpu_used is True
    assert len(embeddings) == 10
    assert all(len(value) == CLIP_EMBEDDING_DIM for value in embeddings)
    assert len({tuple(value) for value in embeddings}) > 1


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


def _fake_clip_array(batch: pa.RecordBatch, **kwargs) -> pa.Array:
    image_ids = batch.column("image_id").to_pylist()
    rows: list[list[float]] = []
    for image_id in image_ids:
        seed = sum(ord(char) for char in str(image_id)) % 997
        rows.append([float((seed + index) % 251) / 251.0 for index in range(CLIP_EMBEDDING_DIM)])
    return pa.array(rows, type=CLIP_OUTPUT_TYPE)
