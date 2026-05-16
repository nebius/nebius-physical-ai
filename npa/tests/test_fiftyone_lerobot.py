from __future__ import annotations

import base64
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from npa.fiftyone_lerobot import (
    build_lerobot_import_plan,
    materialize_lerobot_source,
)


_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {key: [row[key] for row in rows] for key in rows[0]}
    pq.write_table(pa.table(columns), path)


def _write_image_lerobot_dataset(
    root: Path,
    *,
    include_optional_columns: bool = True,
) -> Path:
    images = root / "images" / "observation.image"
    images.mkdir(parents=True)
    for idx in range(2):
        (images / f"frame-{idx}.png").write_bytes(_TINY_PNG)

    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({
            "codebase_version": "v3.0",
            "features": {
                "observation.image": {
                    "dtype": "image",
                    "shape": [1, 1, 3],
                }
            },
        })
    )

    rows: list[dict] = [
        {"observation.image": "frame-0.png"},
        {"observation.image": "frame-1.png"},
    ]
    if include_optional_columns:
        rows = [
            {
                **rows[0],
                "episode_index": 7,
                "frame_index": 0,
                "timestamp": 0.0,
                "task_success": True,
                "task_index": 0,
            },
            {
                **rows[1],
                "episode_index": 7,
                "frame_index": 1,
                "timestamp": 0.05,
                "task_success": True,
                "task_index": 0,
            },
        ]
    _write_parquet(root / "data" / "chunk-000" / "file-000.parquet", rows)
    _write_parquet(
        root / "meta" / "tasks.parquet",
        [{"task_index": 0, "task": "push cube"}],
    )
    _write_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        [{"episode_index": 7, "length": 2, "tasks": ["push cube"]}],
    )
    return root


def test_build_lerobot_import_plan_maps_parquet_metadata(tmp_path: Path) -> None:
    root = _write_image_lerobot_dataset(tmp_path / "dataset")

    plan = build_lerobot_import_plan(root, tmp_path / "frames")

    assert len(plan.samples) == 2
    assert plan.media_keys == ["observation.image"]
    assert plan.metadata_fields == [
        "episode_index",
        "frame_index",
        "timestamp",
        "task_success",
    ]
    first = plan.samples[0]
    assert first.filepath.endswith("frame-0.png")
    assert first.fields["episode_index"] == 7
    assert first.fields["frame_index"] == 0
    assert first.fields["timestamp"] == 0.0
    assert first.fields["task_success"] is True
    assert first.fields["task_index"] == 0
    assert first.fields["task"] == "push cube"


def test_build_lerobot_import_plan_skips_missing_columns_with_warnings(tmp_path: Path) -> None:
    root = _write_image_lerobot_dataset(
        tmp_path / "dataset",
        include_optional_columns=False,
    )

    plan = build_lerobot_import_plan(root, tmp_path / "frames")

    assert len(plan.samples) == 2
    assert "episode_index" not in plan.samples[0].fields
    assert "frame_index" not in plan.samples[0].fields
    assert "timestamp" not in plan.samples[0].fields
    assert "task_success" not in plan.samples[0].fields
    assert any("no episode_index column" in warning for warning in plan.warnings)
    assert any("no frame_index column" in warning for warning in plan.warnings)
    assert any("no timestamp column" in warning for warning in plan.warnings)
    assert any("no success/failure column" in warning for warning in plan.warnings)


def test_materialize_lerobot_source_detects_local_s3_and_hf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    local = tmp_path / "local"
    local.mkdir()

    assert materialize_lerobot_source(str(local), "ds", tmp_path) == (local, "local")

    s3_target = tmp_path / "s3"
    hf_target = tmp_path / "hf"
    monkeypatch.setattr(
        "npa.fiftyone_lerobot._download_s3_source",
        lambda source, name, datasets_dir: s3_target,
    )
    monkeypatch.setattr(
        "npa.fiftyone_lerobot._download_huggingface_source",
        lambda source, name, datasets_dir: hf_target,
    )

    assert materialize_lerobot_source("s3://bucket/dataset", "ds", tmp_path) == (
        s3_target,
        "s3",
    )
    assert materialize_lerobot_source("lerobot/pusht", "ds", tmp_path) == (
        hf_target,
        "huggingface",
    )
