from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pytest
import pyarrow.parquet as pq

from npa.workflows.encord_groot_loop import MaterializeRequest, materialize


class FakeStorage:
    def __init__(self, source: Path, generated: Path) -> None:
        self.source, self.generated = source, generated
        self.uploaded: Path | None = None

    def download_directory(self, uri: str, destination: str) -> str:
        shutil.copytree(self.source if uri == "s3://test/source/" else self.generated, destination)
        return destination

    def upload_directory(self, local: str, uri: str) -> str:
        self.uploaded = self.generated.parent / "uploaded"
        shutil.copytree(local, self.uploaded)
        return uri

    def upload_file(self, local: str, uri: str) -> str:
        return uri


def _dataset(root: Path) -> None:
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.front" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": pa.array([0, 0], type=pa.int64()), "index": pa.array([0, 1], type=pa.int64()), "frame_index": pa.array([0, 1], type=pa.int64()), "task_index": pa.array([0, 0], type=pa.int64()), "timestamp": pa.array([0.0, 0.05], type=pa.float32()), "observation.state": pa.array([[0.5], [0.6]]), "action": pa.array([[1.0], [2.0]])}), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"episode_index": pa.array([0], type=pa.int64()), "data/chunk_index": pa.array([0], type=pa.int64()), "data/file_index": pa.array([0], type=pa.int64()), "dataset_from_index": pa.array([0], type=pa.int64()), "dataset_to_index": pa.array([2], type=pa.int64()), "videos/observation.images.front/chunk_index": pa.array([0], type=pa.int64()), "videos/observation.images.front/file_index": pa.array([0], type=pa.int64()), "videos/observation.images.front/from_timestamp": pa.array([0.0]), "videos/observation.images.front/to_timestamp": pa.array([0.1])}), root / "meta/episodes/chunk-000/file-000.parquet")
    (root / "meta/info.json").write_text(json.dumps({"fps": 20, "total_episodes": 1, "total_frames": 2, "features": {"observation.images.front": {"dtype": "video", "shape": [64, 64, 3]}, "observation.state": {"dtype": "float32", "shape": [1]}, "action": {"dtype": "float32", "shape": [1]}, "episode_index": {"dtype": "int64", "shape": [1]}, "frame_index": {"dtype": "int64", "shape": [1]}, "task_index": {"dtype": "int64", "shape": [1]}, "timestamp": {"dtype": "float32", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]}}}))
    (root / "videos/observation.images.front/chunk-000/file-000.mp4").write_bytes(b"original")


def test_materialize_preserves_original_and_adds_one_synthetic_episode(tmp_path: Path) -> None:
    source, generated = tmp_path / "source", tmp_path / "generated"
    _dataset(source)
    (generated / "variant-1").mkdir(parents=True)
    (generated / "variant-1" / "vision.mp4").write_bytes(b"synthetic")
    storage = FakeStorage(source, generated)
    request = MaterializeRequest.from_argv(
        [
            "s3://test/source/",
            "s3://test/generated/",
            "s3://test/output/",
            "observation.images.front",
            "0",
            "s3://test/output/materialization.json",
        ]
    )
    summary = materialize(request, storage_client=storage)
    assert summary["original_episodes"] == 1 and summary["synthetic_episodes"] == 1
    assert storage.uploaded is not None
    data = pa.concat_tables(
        [
            pq.read_table(storage.uploaded / "data/chunk-000/episode_000000.parquet"),
            pq.read_table(storage.uploaded / "data/chunk-000/episode_000001.parquet"),
        ]
    )
    assert data["episode_index"].to_pylist() == [0, 0, 1, 1]
    assert (
        storage.uploaded / "videos/chunk-000/observation.images.front/episode_000001.mp4"
    ).read_bytes() == b"synthetic"
    assert (storage.uploaded / "meta/modality.json").is_file()
    modality_config = (storage.uploaded / "meta/npa_groot_modality_config.py").read_text()
    assert "ActionRepresentation.RELATIVE" not in modality_config
    assert "ActionRepresentation.ABSOLUTE" in modality_config


def test_materialize_request_rejects_a_non_integer_episode() -> None:
    from npa.workflows.encord_groot_loop import EncordGrootError

    with pytest.raises(EncordGrootError, match="must be an integer"):
        MaterializeRequest.from_argv(["s", "a", "o", "cam", "zero", "m"])
    with pytest.raises(SystemExit):
        MaterializeRequest.from_argv(["too", "few"])
