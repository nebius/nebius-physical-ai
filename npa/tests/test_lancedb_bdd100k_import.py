from __future__ import annotations

import hashlib
import io
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import boto3
import pyarrow as pa
import pytest
from botocore.stub import Stubber
from fastapi.testclient import TestClient
from PIL import Image

from npa.workbench.lancedb.bdd100k_import import (
    BDD100KSourceError,
    BDD100KValidationError,
    bdd100k_schema,
    import_bdd100k,
    manifest_checksum,
    schema_summary,
)
from npa.workbench.lancedb.server import create_app


def test_bdd100k_synthetic_mode_produces_declared_row_count(tmp_path: Path) -> None:
    result = import_bdd100k(
        synthetic=13,
        synthetic_seed=42,
        lance_uri=str(tmp_path / "db"),
        table="bdd_synthetic",
    )

    assert result.total_rows == 13
    assert result.rows_per_split == {"train": 10, "val": 3}
    assert result.table_version == 1


def test_bdd100k_schema_fields_have_expected_types() -> None:
    schema = bdd100k_schema()

    assert schema_summary() == {
        "image_id": "string",
        "image_bytes": "large_binary",
        "width": "int32",
        "height": "int32",
        "weather": "string",
        "scene": "string",
        "timeofday": "string",
        "timestamp": "timestamp[ms]",
        "ann_categories": "list<item: string>",
        "ann_bboxes": "list<item: list<item: float>>",
        "ann_occluded": "list<item: bool>",
        "split": "string",
    }
    assert schema.field("image_bytes").type == pa.large_binary()


def test_bdd100k_manifest_checksum_is_deterministic_for_seed(tmp_path: Path) -> None:
    first = import_bdd100k(
        synthetic=8,
        synthetic_seed=7,
        lance_uri=str(tmp_path / "db-a"),
        table="bdd_seed",
    )
    second = import_bdd100k(
        synthetic=8,
        synthetic_seed=7,
        lance_uri=str(tmp_path / "db-b"),
        table="bdd_seed",
    )

    assert first.manifest_sha256 == second.manifest_sha256


def test_bdd100k_manifest_checksum_differs_for_different_seeds(tmp_path: Path) -> None:
    first = import_bdd100k(
        synthetic=8,
        synthetic_seed=7,
        lance_uri=str(tmp_path / "db-a"),
        table="bdd_seed",
    )
    second = import_bdd100k(
        synthetic=8,
        synthetic_seed=8,
        lance_uri=str(tmp_path / "db-b"),
        table="bdd_seed",
    )

    assert first.manifest_sha256 != second.manifest_sha256


def test_bdd100k_empty_source_path_raises_clear_error() -> None:
    with pytest.raises(BDD100KSourceError, match="source is required"):
        import_bdd100k(source="", synthetic=None, write=False)


def test_bdd100k_synthetic_zero_falls_back_to_source() -> None:
    # ``--synthetic 0`` means "no synthetic rows -> read the real source" so a
    # pipeline can pass a single ``--synthetic {{config.synthetic_rows}}`` arg and
    # toggle synthetic-smoke vs real ingest purely by config. With no source that
    # must surface the same "source is required" error as synthetic=None.
    with pytest.raises(BDD100KSourceError, match="source is required"):
        import_bdd100k(source="", synthetic=0, write=False)


def test_bdd100k_synthetic_negative_raises_clear_error() -> None:
    with pytest.raises(BDD100KValidationError, match="non-negative"):
        import_bdd100k(synthetic=-1, table="bdd_neg", write=False)


def test_bdd100k_invalid_split_raises_clear_error() -> None:
    with pytest.raises(BDD100KValidationError, match="invalid split"):
        import_bdd100k(synthetic=1, splits=["dev"], write=False)


def test_bdd100k_accepts_both_label_filename_conventions(tmp_path: Path) -> None:
    _write_fixture_split(tmp_path, "train", "det_train.json", "train-000.jpg")
    _write_fixture_split(
        tmp_path, "val", "bdd100k_labels_images_val.json", "val-000.jpg"
    )

    result = import_bdd100k(
        source=str(tmp_path),
        splits=["train", "val"],
        limit=1,
        lance_uri=str(tmp_path / "db"),
        table="bdd_real_subset",
    )

    assert result.total_rows == 2
    assert result.rows_per_split == {"train": 1, "val": 1}


@pytest.mark.parametrize("occluded", [True, False])
def test_bdd100k_accepts_boolean_occlusion_values(
    tmp_path: Path, occluded: bool
) -> None:
    _write_occlusion_fixture(
        tmp_path,
        [[{"attributes": {"occluded": occluded}}]],
    )

    rows = _imported_fixture_rows(tmp_path, table="bdd_boolean_occlusion")

    assert rows[0]["ann_occluded"] == [occluded]


def test_bdd100k_missing_occlusion_defaults_to_false(tmp_path: Path) -> None:
    _write_occlusion_fixture(tmp_path, [[{}]])

    rows = _imported_fixture_rows(tmp_path, table="bdd_missing_occlusion")

    assert rows[0]["ann_occluded"] == [False]


@pytest.mark.parametrize("occluded", [True, False])
def test_bdd100k_accepts_top_level_occlusion_fallback(
    tmp_path: Path, occluded: bool
) -> None:
    _write_occlusion_fixture(tmp_path, [[{"occluded": occluded}]])

    rows = _imported_fixture_rows(tmp_path, table="bdd_top_level_occlusion")

    assert rows[0]["ann_occluded"] == [occluded]


@pytest.mark.parametrize(
    ("nested", "top_level"),
    [(False, True), (True, False)],
)
def test_bdd100k_nested_occlusion_precedes_top_level_fallback(
    tmp_path: Path, nested: bool, top_level: bool
) -> None:
    annotation = {
        "attributes": {"occluded": nested},
        "occluded": top_level,
    }
    _write_occlusion_fixture(tmp_path, [[annotation]])

    rows = _imported_fixture_rows(tmp_path, table="bdd_nested_occlusion")

    assert rows[0]["ann_occluded"] == [nested]


@pytest.mark.parametrize("malformed", ["false", "true", 0, 1, None, 0.5, [], {}])
def test_bdd100k_rejects_malformed_occlusion_before_writing_rows(
    tmp_path: Path, malformed: object
) -> None:
    _write_occlusion_fixture(
        tmp_path,
        [
            [{"attributes": {"occluded": False}}],
            [{"attributes": {"occluded": malformed}}],
        ],
    )
    database_path = tmp_path / "db"

    with pytest.raises(
        BDD100KValidationError,
        match=r"annotation 0 for image 'train-001.jpg'.*expected a boolean",
    ):
        import_bdd100k(
            source=str(tmp_path),
            splits=["train"],
            lance_uri=str(database_path),
            table="bdd_malformed_occlusion",
            batch_size=1,
        )

    import lancedb

    assert (
        "bdd_malformed_occlusion"
        not in lancedb.connect(str(database_path)).table_names()
    )


@pytest.mark.parametrize("source_kind", ["local", "s3"])
@pytest.mark.parametrize("existing", [False, True])
def test_occlusion_in_later_split_cannot_partially_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_kind: str, existing: bool
) -> None:
    import lancedb

    source = tmp_path / "input"
    _write_invalid_last_split(source)
    database = tmp_path / "db"
    before = None
    if existing:
        before = import_bdd100k(synthetic=1, lance_uri=str(database)).table_version
    context = nullcontext()
    location = str(source)
    if source_kind == "s3":
        context = _stub_s3_source(
            source, monkeypatch, ["det_train.json", "det_val.json"]
        )
        location = "s3://test-bucket/bundle"

    with context, pytest.raises(BDD100KValidationError, match="val-000.jpg"):
        import_bdd100k(source=location, lance_uri=str(database), batch_size=1)

    connection = lancedb.connect(str(database))
    if existing:
        table = connection.open_table("bdd100k")
        assert table.version == before
        assert table.count_rows() == 1
    else:
        assert "bdd100k" not in connection.table_names()


def test_local_import_uses_the_validated_label_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_occlusion_fixture(tmp_path, [[{"occluded": False}], [{"occluded": True}]])
    labels = tmp_path / "det_train.json"
    original_read = Path.read_bytes
    reads = []

    def read_then_replace(path):
        raw = original_read(path)
        if path == labels:
            reads.append(path)
            changed = json.loads(raw)
            changed[1]["labels"][0]["occluded"] = "false"
            labels.write_text(json.dumps(changed), encoding="utf-8")
        return raw

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    rows = _imported_fixture_rows(tmp_path, table="bdd_snapshot", batch_size=1)

    assert [row["ann_occluded"] for row in rows] == [[False], [True]]
    assert reads == [labels]


def test_s3_import_writes_the_validated_labels_without_refetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lancedb

    source = tmp_path / "input"
    _write_occlusion_fixture(source, [[{"occluded": False}], [{"occluded": True}]])
    objects = [
        "det_train.json",
        "images/100k/train/train-000.jpg",
        "images/100k/train/train-001.jpg",
    ]
    with _stub_s3_source(source, monkeypatch, objects):
        result = import_bdd100k(
            source="s3://test-bucket/bundle",
            splits=["train"],
            lance_uri=str(tmp_path / "db"),
            batch_size=1,
        )

    rows = (
        lancedb.connect(str(tmp_path / "db"))
        .open_table("bdd100k")
        .to_arrow()
        .to_pylist()
    )
    assert result.total_rows == 2
    assert [row["ann_occluded"] for row in rows] == [[False], [True]]


def test_import_limit_excludes_unselected_malformed_occlusion(tmp_path: Path) -> None:
    _write_occlusion_fixture(tmp_path, [[{"occluded": False}], [{"occluded": "false"}]])

    result = import_bdd100k(
        source=str(tmp_path),
        splits=["train"],
        limit=1,
        lance_uri=str(tmp_path / "db"),
        batch_size=1,
    )

    assert result.total_rows == 1


def _write_invalid_last_split(source: Path) -> None:
    _write_fixture_split(source, "train", "det_train.json", "train-000.jpg")
    _write_fixture_split(source, "val", "det_val.json", "val-000.jpg")
    labels = source / "det_val.json"
    entries = json.loads(labels.read_text())
    entries[0]["labels"][0]["attributes"]["occluded"] = "false"
    labels.write_text(json.dumps(entries), encoding="utf-8")


@contextmanager
def _stub_s3_source(
    source: Path, monkeypatch: pytest.MonkeyPatch, downloads: list[str]
):
    from npa.workbench.lancedb import bdd100k_import

    client = boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url="https://s3.example.invalid",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    monkeypatch.setattr(bdd100k_import, "_s3_client", lambda: client)
    objects = [
        {"Key": "bundle/" + path.relative_to(source).as_posix()}
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]
    with Stubber(client) as stubber:
        stubber.add_response(
            "list_objects_v2",
            {"Contents": objects, "IsTruncated": False},
            {"Bucket": "test-bucket", "Prefix": "bundle/"},
        )
        for name in downloads:
            stubber.add_response(
                "get_object",
                {"Body": io.BytesIO((source / name).read_bytes())},
                {"Bucket": "test-bucket", "Key": "bundle/" + name},
            )
        yield
        stubber.assert_no_pending_responses()


def test_bdd100k_sdk_function_returns_typed_result(tmp_path: Path) -> None:
    from npa.workbench.lancedb import BDD100KImportResult, import_bdd100k as sdk_import

    result = sdk_import(
        synthetic=4,
        synthetic_seed=11,
        lance_uri=str(tmp_path / "sdk-db"),
        table="bdd_sdk",
    )

    assert isinstance(result, BDD100KImportResult)
    assert result.total_rows == 4
    assert result.table == "bdd_sdk"


def test_bdd100k_sdk_local_matches_direct_module_call(tmp_path: Path) -> None:
    from npa.workbench.lancedb import import_bdd100k as sdk_import

    direct = import_bdd100k(
        synthetic=6,
        synthetic_seed=99,
        lance_uri=str(tmp_path / "direct-db"),
        table="bdd_direct",
    )
    sdk = sdk_import(
        synthetic=6,
        synthetic_seed=99,
        lance_uri=str(tmp_path / "sdk-db"),
        table="bdd_direct",
    )

    assert sdk.total_rows == direct.total_rows
    assert sdk.rows_per_split == direct.rows_per_split
    assert sdk.manifest_sha256 == direct.manifest_sha256


def test_bdd100k_sdk_accepts_explicit_local_mode(tmp_path: Path) -> None:
    from npa.workbench.lancedb import import_bdd100k as sdk_import

    result = sdk_import(
        mode="local",
        synthetic=3,
        synthetic_seed=12,
        lance_uri=str(tmp_path / "mode-db"),
        table="bdd_mode",
    )

    assert result.total_rows == 3


def test_bdd100k_sdk_service_mode_matches_http_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import npa.workbench.lancedb as sdk_module

    app = create_app(storage_path=str(tmp_path / "service-root"), auth_mode="none")
    client = TestClient(app)
    payload = {
        "synthetic": 5,
        "synthetic_seed": 23,
        "lance_uri": str(tmp_path / "endpoint-db"),
        "table": "bdd_endpoint",
    }
    response = client.post("/import-bdd100k", json=payload)
    assert response.status_code == 200
    endpoint_payload = response.json()

    monkeypatch.setattr(sdk_module, "_post_json", lambda **kwargs: endpoint_payload)
    sdk_result = sdk_module.import_bdd100k(
        service=True,
        endpoint="http://lancedb.example",
        synthetic=5,
        synthetic_seed=23,
        lance_uri=str(tmp_path / "mocked-service-db"),
        table="bdd_endpoint",
    )

    assert sdk_result.to_dict() == endpoint_payload


def test_lancedb_token_auth_rejects_missing_and_invalid_tokens(tmp_path: Path) -> None:
    app = create_app(
        storage_path=str(tmp_path / "auth-root"), auth_mode="token", token="s3cr3t"
    )
    client = TestClient(app)

    assert client.get("/health").status_code == 401
    assert (
        client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert client.get("/health", headers={"Authorization": "s3cr3t"}).status_code == 401

    ok = client.get("/health", headers={"Authorization": "Bearer s3cr3t"})
    assert ok.status_code == 200, ok.text
    assert ok.json()["status"] == "ok"


def test_lancedb_token_auth_mode_without_token_is_misconfiguration(
    tmp_path: Path,
) -> None:
    app = create_app(
        storage_path=str(tmp_path / "auth-root"), auth_mode="token", token=""
    )
    client = TestClient(app)
    response = client.get("/health", headers={"Authorization": "Bearer anything"})
    assert response.status_code == 500
    assert "not configured" in response.json()["detail"]


def test_bdd100k_endpoint_rejects_invalid_split(tmp_path: Path) -> None:
    app = create_app(storage_path=str(tmp_path / "service-root"), auth_mode="none")
    client = TestClient(app)

    response = client.post(
        "/import-bdd100k",
        json={
            "synthetic": 1,
            "lance_uri": str(tmp_path / "endpoint-db"),
            "splits": ["dev"],
        },
    )

    assert response.status_code == 422
    assert "invalid split" in response.text


def test_manifest_checksum_matches_table_rows(tmp_path: Path) -> None:
    result = import_bdd100k(
        synthetic=5,
        synthetic_seed=31,
        lance_uri=str(tmp_path / "db"),
        table="bdd_manifest",
    )

    import lancedb

    table = lancedb.connect(str(tmp_path / "db")).open_table("bdd_manifest")
    rows = table.to_arrow().to_pylist()
    entries = [
        (row["image_id"], row["split"], hashlib.sha256(row["image_bytes"]).hexdigest())
        for row in rows
    ]
    assert manifest_checksum(entries) == result.manifest_sha256


def _write_fixture_split(
    root: Path, split: str, label_name: str, image_name: str
) -> None:
    image_dir = root / "images" / "100k" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    _jpeg_bytes(image_dir / image_name)
    labels = [
        {
            "name": image_name,
            "attributes": {
                "weather": "clear",
                "scene": "city street",
                "timeofday": "daytime",
            },
            "timestamp": 1234,
            "labels": [
                {
                    "category": "car",
                    "attributes": {"occluded": False},
                    "box2d": {"x1": 1.0, "y1": 2.0, "x2": 30.0, "y2": 40.0},
                }
            ],
        }
    ]
    (root / label_name).write_text(json.dumps(labels), encoding="utf-8")


def _write_occlusion_fixture(
    root: Path,
    annotations_by_image: list[list[dict[str, object]]],
) -> None:
    image_dir = root / "images" / "100k" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for image_index, image_annotations in enumerate(annotations_by_image):
        image_name = f"train-{image_index:03d}.jpg"
        _jpeg_bytes(image_dir / image_name)
        entries.append(
            {
                "name": image_name,
                "labels": [
                    {
                        "category": "car",
                        "box2d": {"x1": 1.0, "y1": 2.0, "x2": 30.0, "y2": 40.0},
                        **annotation,
                    }
                    for annotation in image_annotations
                ],
            }
        )
    (root / "det_train.json").write_text(json.dumps(entries), encoding="utf-8")


def _imported_fixture_rows(
    root: Path, *, table: str, batch_size: int = 200
) -> list[dict[str, object]]:
    database_path = root / "db"
    import_bdd100k(
        source=str(root),
        splits=["train"],
        lance_uri=str(database_path),
        table=table,
        batch_size=batch_size,
    )

    import lancedb

    return lancedb.connect(str(database_path)).open_table(table).to_arrow().to_pylist()


def _jpeg_bytes(path: Path) -> bytes:
    image = Image.new("RGB", (64, 32), (120, 40, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    raw = buffer.getvalue()
    path.write_bytes(raw)
    return raw
