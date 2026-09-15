"""Verify selective LeRobot worker input using real episode video decoding."""

from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

from botocore.exceptions import ClientError
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.workflows import paidf_cosmos3 as c3
from npa.workflows.paidf_cosmos3_media import probe_video


def _storage(objects: dict[str, bytes]):
    s3 = Mock(spec=["head_object", "get_object", "list_objects_v2", "download_file"])

    def head(*, Bucket, Key):
        if Key not in objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "HeadObject")
        return {"ContentLength": len(objects[Key])}

    s3.head_object.side_effect = head
    s3.get_object.side_effect = lambda **kw: {"Body": BytesIO(objects[kw["Key"]])}
    s3.list_objects_v2.side_effect = lambda **kw: {
        "Contents": [{"Key": key} for key in objects if key.startswith(kw["Prefix"])]
    }
    s3.download_file.side_effect = lambda bucket, key, path: Path(path).write_bytes(objects[key])
    return SimpleNamespace(
        s3=s3,
        download_path=Mock(side_effect=AssertionError("Do not download the dataset tree")),
    )


def _shared_video(path: Path) -> bytes:
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=64x64:r=24:d=1",
        "-f", "lavfi", "-i", "color=blue:s=64x64:r=24:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(path),
    ], check=True)
    return path.read_bytes()


def _dataset(tmp_path: Path, version: str, camera: str) -> tuple[dict[str, bytes], str]:
    prefix = "dataset/"
    objects = {prefix + "meta/info.json": json.dumps({
        "codebase_version": version, "total_episodes": 2, "chunks_size": 1000,
        "features": {camera: {"dtype": "video"}},
    }).encode()}
    if version.startswith("v3"):
        key = prefix + f"videos/{camera}/chunk-000/file-000.mp4"
        metadata = tmp_path / "episodes.parquet"
        pq.write_table(pa.Table.from_pylist([{
            "episode_index": 1, f"videos/{camera}/chunk_index": 0,
            f"videos/{camera}/file_index": 0, f"videos/{camera}/from_timestamp": 1.0,
            f"videos/{camera}/to_timestamp": 2.0,
        }]), metadata)
        objects[prefix + "meta/episodes/chunk-002/file-000.parquet"] = metadata.read_bytes()
    else:
        key = prefix + f"videos/chunk-000/{camera}/episode_000001.mp4"
    objects[key] = _shared_video(tmp_path / "shared.mp4")
    objects[prefix + "videos/unselected.mp4"] = b"must never be downloaded"
    return objects, key


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required")
@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_s3_worker_selects_one_episode_without_fetching_dataset(
    tmp_path: Path, version: str,
) -> None:
    camera = "observation.images.front"
    objects, selected_key = _dataset(tmp_path, version, camera)
    storage = _storage(objects)
    output = tmp_path / "input"
    result = c3.prepare_input(
        "lerobot", "", "s3://example-bucket/dataset/", 1, camera,
        str(output), str(output / "provenance.json"), "unit-run",
        storage=storage, conditioning_fps=24,
    )
    assert result["source_kind"] == "lerobot_dataset"
    assert result["episode"] == 1 and result["camera"] == camera
    storage.download_path.assert_not_called()
    downloads = [call.args[1] for call in storage.s3.download_file.call_args_list]
    assert [key for key in downloads if key.endswith(".mp4")] == [selected_key]
    expected_frames = 24 if version.startswith("v3") else 48
    assert probe_video(output / "source.mp4")["decoded_frames"] == expected_frames
    if version.startswith("v3"):
        pixel = subprocess.check_output([
            "ffmpeg", "-v", "error", "-i", str(output / "source.mp4"),
            "-vf", "scale=1:1", "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ])
        assert pixel[2] > pixel[0] + 100, "Episode 1 must contain the blue second interval"


@pytest.mark.parametrize("start,end", [(None, None), (0.0, None), (0.0, float("nan"))])
def test_local_v3_refuses_missing_or_nonfinite_episode_timestamps(
    tmp_path: Path, start: float | None, end: float | None,
) -> None:
    camera = "observation.images.front"
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "features": {camera: {"dtype": "video"}},
    }))
    metadata = tmp_path / "meta/episodes/chunk-000"
    metadata.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{
        "episode_index": 1, f"videos/{camera}/file_index": 0,
        f"videos/{camera}/from_timestamp": start, f"videos/{camera}/to_timestamp": end,
    }]), metadata / "file-000.parquet")
    video = tmp_path / "videos" / camera / "chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    with pytest.raises(c3.PaidfCosmos3Error, match="timestamps"):
        c3._select_lerobot_video(tmp_path, 1, camera)
