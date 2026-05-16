from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pytest
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


def test_bdd100k_invalid_split_raises_clear_error() -> None:
    with pytest.raises(BDD100KValidationError, match="invalid split"):
        import_bdd100k(synthetic=1, splits=["dev"], write=False)


def test_bdd100k_accepts_both_label_filename_conventions(tmp_path: Path) -> None:
    _write_fixture_split(tmp_path, "train", "det_train.json", "train-000.jpg")
    _write_fixture_split(tmp_path, "val", "bdd100k_labels_images_val.json", "val-000.jpg")

    result = import_bdd100k(
        source=str(tmp_path),
        splits=["train", "val"],
        limit=1,
        lance_uri=str(tmp_path / "db"),
        table="bdd_real_subset",
    )

    assert result.total_rows == 2
    assert result.rows_per_split == {"train": 1, "val": 1}


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


def test_bdd100k_sdk_service_mode_matches_http_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def _write_fixture_split(root: Path, split: str, label_name: str, image_name: str) -> None:
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


def _jpeg_bytes(path: Path) -> bytes:
    image = Image.new("RGB", (64, 32), (120, 40, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    raw = buffer.getvalue()
    path.write_bytes(raw)
    return raw
