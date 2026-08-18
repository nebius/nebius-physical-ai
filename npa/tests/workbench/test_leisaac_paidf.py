from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from npa.workbench.leisaac import dataset as leisaac_dataset
from npa.workbench.leisaac import paidf as leisaac_paidf
from npa.workbench.leisaac.dataset import DatasetError, _encode_frames, sha256_file
from npa.workbench.leisaac.paidf import (
    assert_temporal_alignment,
    export_episode_to_paidf,
    materialize_paidf_dataset,
)


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, dict[str, str]]] = {}

    def add(self, bucket: str, key: str, body: bytes, *, sha256: str = "") -> None:
        metadata = {"sha256": sha256} if sha256 else {}
        self.objects[(bucket, key)] = (body, metadata)

    def put_object(
        self, *, Bucket, Key, Body, Metadata=None, IfNoneMatch=None, **_kwargs
    ):
        if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
            raise RuntimeError("precondition failed")
        body = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.objects[(Bucket, Key)] = (body, dict(Metadata or {}))

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)][0])}

    def head_object(self, *, Bucket, Key):
        return {"Metadata": self.objects[(Bucket, Key)][1]}

    def copy_object(self, *, Bucket, Key, CopySource, IfNoneMatch=None, **_kwargs):
        if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
            raise RuntimeError("precondition failed")
        self.objects[(Bucket, Key)] = self.objects[
            (CopySource["Bucket"], CopySource["Key"])
        ]

    def download_file(self, bucket, key, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(self.objects[(bucket, key)][0])

    def list_objects_v2(self, *, Bucket, Prefix, **_kwargs):
        return {
            "Contents": [
                {"Key": key, "Size": len(value[0])}
                for (bucket, key), value in sorted(self.objects.items())
                if bucket == Bucket and key.startswith(Prefix)
            ],
            "IsTruncated": False,
        }


def _video(tmp_path: Path, name: str, colors: list[tuple[int, int, int]]) -> Path:
    frames = tmp_path / f"{name}-frames"
    frames.mkdir()
    for index, color in enumerate(colors):
        Image.new("RGB", (1280, 720), color).save(frames / f"frame-{index:06d}.jpg")
    path = tmp_path / f"{name}.mp4"
    _encode_frames(frames, path)
    return path


def _source(fake: FakeS3, video: Path) -> tuple[str, dict]:
    dataset_uri = "s3://bucket/source/versions/v000001-test"
    video_bytes = video.read_bytes()
    video_sha = sha256_file(video)
    frame_objects = [
        {
            "key": f"source/episodes/000000/frames/frame-{index:06d}.jpg",
            "sha256": format(index + 1, "064x"),
            "bytes": 1024,
        }
        for index in range(3)
    ]
    commit = {
        "schema": "npa.leisaac.episode-commit.v1",
        "episode_index": 0,
        "episode_uuid": "episode-uuid",
        "metadata": {
            "task": "LeIsaac-SO101-PickOrange-v0",
            "environment_id": "kitchen-a",
            "environment_index": 0,
            "seed": 42,
            "frame_count": 3,
        },
        "objects": {
            "video": {
                "key": "source/episodes/000000/episode.mp4",
                "sha256": video_sha,
                "bytes": len(video_bytes),
            },
            "records": {
                "key": "source/episodes/000000/records.jsonl",
                "sha256": "b" * 64,
            },
            "frames": frame_objects,
        },
    }
    commit_uri = "s3://bucket/source/commits/episode-000000.json"
    manifest = {
        "schema": "npa.leisaac.dataset.v1",
        "lerobot_version": "0.5.1",
        "dataset_uri": dataset_uri,
        "version": "v000001-test",
        "episode_commits": [commit_uri],
        "files": [
            {
                "key": "source/versions/v000001-test/meta/info.json",
                "sha256": "c" * 64,
                "bytes": 2,
            },
            {
                "key": "source/versions/v000001-test/data/chunk-000/file-000.parquet",
                "sha256": "d" * 64,
                "bytes": 7,
            },
            {
                "key": "source/versions/v000001-test/videos/observation.images.front/chunk-000/file-000.mp4",
                "sha256": video_sha,
                "bytes": len(video_bytes),
            },
        ],
    }
    manifest_bytes = json.dumps(manifest).encode()
    manifest_sha = __import__("hashlib").sha256(manifest_bytes).hexdigest()
    fake.add(
        "bucket",
        "source/versions/v000001-test/npa-dataset.json",
        manifest_bytes,
        sha256=manifest_sha,
    )
    commit_bytes = json.dumps(commit).encode()
    fake.add(
        "bucket",
        "source/commits/episode-000000.json",
        commit_bytes,
        sha256=__import__("hashlib").sha256(commit_bytes).hexdigest(),
    )
    fake.add(
        "bucket", "source/episodes/000000/episode.mp4", video_bytes, sha256=video_sha
    )
    for frame in frame_objects:
        fake.add(
            "bucket",
            frame["key"],
            b"jpeg",
            sha256=frame["sha256"],
        )
    fake.add(
        "bucket", "source/versions/v000001-test/meta/info.json", b"{}", sha256="c" * 64
    )
    fake.add(
        "bucket",
        "source/versions/v000001-test/data/chunk-000/file-000.parquet",
        b"parquet",
        sha256="d" * 64,
    )
    fake.add(
        "bucket",
        "source/versions/v000001-test/videos/observation.images.front/chunk-000/file-000.mp4",
        video_bytes,
        sha256=video_sha,
    )
    return dataset_uri, commit


def test_export_paidf_is_input_conditioned_and_preserves_source_lineage(
    tmp_path: Path,
) -> None:
    fake = FakeS3()
    source_video = _video(
        tmp_path, "source", [(30, 30, 30), (100, 80, 60), (180, 150, 120)]
    )
    dataset_uri, commit = _source(fake, source_video)
    result = export_episode_to_paidf(
        dataset_uri=dataset_uri,
        episode_index=0,
        paidf_run_id="paidf-demo-1",
        paidf_output_path="s3://bucket/physical-ai-data-factory/paidf-demo-1",
        client=fake,
    )
    assert result["tool_ref"] == "workbench.cosmos2.transfer_execute"
    assert result["condition_on_input"] is True
    assert "NPA_COSMOS_CONDITION_ON_INPUT=1" in result["command"]
    lineage = json.loads(
        fake.objects[
            (
                "bucket",
                "physical-ai-data-factory/paidf-demo-1/input/leisaac-lineage.json",
            )
        ][0]
    )
    assert lineage["source"]["video_sha256"] == commit["objects"]["video"]["sha256"]
    assert lineage["source"]["task"] == "LeIsaac-SO101-PickOrange-v0"
    assert lineage["paidf"]["required_engine"] == "cosmos_transfer2.5_gpu"
    assert result["annotation_frame_count"] == 3
    assert [item["frame_index"] for item in lineage["paidf"]["annotation_frames"]] == [
        0,
        1,
        2,
    ]
    assert all(
        item["sha256"] == commit["objects"]["frames"][item["frame_index"]]["sha256"]
        for item in lineage["paidf"]["annotation_frames"]
    )
    nested = export_episode_to_paidf(
        dataset_uri=dataset_uri,
        episode_index=0,
        paidf_run_id="paidf-demo-2",
        paidf_output_path="s3://bucket/checkpoints/physical-ai-data-factory/paidf-demo-2",
        client=fake,
    )
    assert nested["paidf_run_uri"].endswith(
        "/checkpoints/physical-ai-data-factory/paidf-demo-2"
    )
    with pytest.raises(RuntimeError, match="precondition"):
        export_episode_to_paidf(
            dataset_uri=dataset_uri,
            episode_index=0,
            paidf_run_id="paidf-demo-1",
            paidf_output_path="s3://bucket/physical-ai-data-factory/paidf-demo-1",
            client=fake,
        )
    with pytest.raises(DatasetError, match="clean run prefix"):
        export_episode_to_paidf(
            dataset_uri=dataset_uri,
            episode_index=0,
            paidf_run_id="paidf-demo-1",
            paidf_output_path="s3://bucket/random/place",
            client=fake,
        )


def test_export_paidf_accepts_provider_normalized_metadata_casing(
    tmp_path: Path,
) -> None:
    fake = FakeS3()
    source_video = _video(
        tmp_path, "source", [(30, 30, 30), (100, 80, 60), (180, 150, 120)]
    )
    dataset_uri, _commit = _source(fake, source_video)
    for key, (body, metadata) in list(fake.objects.items()):
        fake.objects[key] = (
            body,
            {
                (name.capitalize() if name.lower() == "sha256" else name): value
                for name, value in metadata.items()
            },
        )

    result = export_episode_to_paidf(
        dataset_uri=dataset_uri,
        episode_index=0,
        paidf_run_id="paidf-metadata-casing",
        paidf_output_path=(
            "s3://bucket/physical-ai-data-factory/paidf-metadata-casing"
        ),
        client=fake,
    )

    assert result["status"] == "exported"
    assert result["condition_on_input"] is True


def test_temporal_alignment_rejects_frame_count_mismatch_and_nonblank_passes(
    tmp_path: Path,
) -> None:
    source = _video(
        tmp_path, "source", [(20, 20, 20), (100, 100, 100), (200, 200, 200)]
    )
    aligned = _video(
        tmp_path, "aligned", [(20, 40, 60), (80, 120, 160), (160, 200, 220)]
    )
    report = assert_temporal_alignment(source, aligned)
    assert report["frame_count"] == 3
    short = _video(tmp_path, "short", [(20, 20, 20), (200, 200, 200)])
    with pytest.raises(DatasetError, match="frame-count mismatch"):
        assert_temporal_alignment(source, short)


def test_packaged_ffmpeg_fallback_encodes_probes_and_decodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(leisaac_dataset.shutil, "which", lambda _name: None)
    video = _video(
        tmp_path,
        "packaged-runtime",
        [(20, 30, 40), (90, 110, 130), (180, 200, 220)],
    )

    timestamps = leisaac_paidf._video_timestamps(video)
    assert timestamps == pytest.approx([0.0, 0.0625, 0.125])
    leisaac_paidf._assert_nonblank(video)


def test_materialize_replaces_only_visual_modality_after_real_engine_proof(
    tmp_path: Path,
) -> None:
    fake = FakeS3()
    source = _video(
        tmp_path, "source", [(20, 20, 20), (100, 100, 100), (200, 200, 200)]
    )
    augmented = _video(
        tmp_path, "augmented", [(20, 50, 80), (90, 130, 170), (170, 210, 230)]
    )
    dataset_uri, _commit = _source(fake, source)
    export_episode_to_paidf(
        dataset_uri=dataset_uri,
        episode_index=0,
        paidf_run_id="paidf-demo-2",
        paidf_output_path="s3://bucket/physical-ai-data-factory/paidf-demo-2",
        client=fake,
    )
    augmented_key = (
        "physical-ai-data-factory/paidf-demo-2/cosmos_augmented/clip/variant-000.mp4"
    )
    fake.add(
        "bucket", augmented_key, augmented.read_bytes(), sha256=sha256_file(augmented)
    )
    transfer = {
        "schema": "npa.cosmos2.transfer.v1",
        "mode": "cosmos_transfer2.5_gpu",
        "status": "executed",
        "input_conditioned": True,
        "conditioned_input": "leisaac-episode-000000.mp4",
        "control": "edge",
        "clips": [
            {"variants": [{"augmented_video_uri": f"s3://bucket/{augmented_key}"}]}
        ],
    }
    fake.add(
        "bucket",
        "physical-ai-data-factory/paidf-demo-2/cosmos_augmented/manifest.json",
        json.dumps(transfer).encode(),
    )
    result = materialize_paidf_dataset(
        dataset_uri=dataset_uri,
        episode_index=0,
        paidf_run_uri="s3://bucket/physical-ai-data-factory/paidf-demo-2",
        output_path="s3://bucket/derived/leisaac",
        client=fake,
    )
    assert result["augmentation_engine"] == "cosmos_transfer2.5_gpu"
    assert result["input_conditioned"] is True
    prefix = result["dataset_uri"].split("s3://bucket/", 1)[1]
    assert (
        fake.objects[("bucket", f"{prefix}/data/chunk-000/file-000.parquet")][0]
        == b"parquet"
    )
    assert (
        fake.objects[
            (
                "bucket",
                f"{prefix}/videos/observation.images.front/chunk-000/file-000.mp4",
            )
        ][0]
        == augmented.read_bytes()
    )
    lineage = json.loads(fake.objects[("bucket", f"{prefix}/meta/npa-lineage.json")][0])
    assert lineage["parent_dataset_uri"] == dataset_uri
    assert lineage["nonvisual_labels"] == "byte-identical parent Parquet records"
