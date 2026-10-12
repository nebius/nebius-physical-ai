"""Verify input diagnostics with real H.264 media and read-only dataset selection."""

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from npa.workflows import data_factory_input_diagnostics as diagnostics


@pytest.fixture
def video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for real media decoding")
    path = tmp_path / "robot.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=64x64:r=24:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def test_real_input_decodes_without_storage(video, monkeypatch):
    monkeypatch.setattr(
        diagnostics, "_storage", Mock(side_effect=AssertionError("no S3"))
    )
    result = diagnostics.check_paidf_input(input_video=video)
    assert result["status"] == "passed"
    assert result["media"]["decoded_frames"] == 24
    assert result["media"]["full_decode_passed"] is True
    assert str(video) not in json.dumps(result)
    assert "sha256" not in json.dumps(result)


def test_corrupt_input_has_failure_and_hint(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video")
    result = diagnostics.check_paidf_input(input_video=path)
    assert result["status"] == "failed"
    assert result["hint"]


def test_lerobot_requires_explicit_camera_and_episode():
    result = diagnostics.check_paidf_input(lerobot_uri="s3://test-bucket/dataset/")
    assert result["status"] == "failed"
    assert "--lerobot-camera" in result["error"]
    assert "--lerobot-episode" in result["error"]


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_lerobot_selection_reuses_submit_reader_without_uploads(
    tmp_path, video, version
):
    from io import BytesIO

    camera = "observation.images.front"
    metadata = {
        "codebase_version": version,
        "total_episodes": 1,
        "features": {camera: {"dtype": "video"}},
    }
    objects = {"dataset/meta/info.json": json.dumps(metadata).encode()}
    if version == "v2.1":
        objects[f"dataset/videos/chunk-000/{camera}/episode_000000.mp4"] = (
            video.read_bytes()
        )
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet = tmp_path / "episodes.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "episode_index": 0,
                        f"videos/{camera}/chunk_index": 0,
                        f"videos/{camera}/file_index": 0,
                        f"videos/{camera}/from_timestamp": 0.0,
                        f"videos/{camera}/to_timestamp": 0.5,
                    }
                ]
            ),
            parquet,
        )
        objects["dataset/meta/episodes/chunk-000/file-000.parquet"] = (
            parquet.read_bytes()
        )
        objects[f"dataset/videos/{camera}/chunk-000/file-000.mp4"] = video.read_bytes()
    s3 = Mock(spec=["get_object", "head_object", "list_objects_v2", "download_file"])
    s3.get_object.side_effect = lambda **kw: {"Body": BytesIO(objects[kw["Key"]])}
    s3.head_object.side_effect = lambda **kw: {"ContentLength": len(objects[kw["Key"]])}
    s3.list_objects_v2.side_effect = lambda **kw: {
        "Contents": [{"Key": key} for key in objects if key.startswith(kw["Prefix"])]
    }
    s3.download_file.side_effect = lambda bucket, key, path: Path(path).write_bytes(
        objects[key]
    )
    result = diagnostics.check_paidf_input(
        lerobot_uri="s3://test-bucket/dataset/",
        camera=camera,
        episode=0,
        storage_client=SimpleNamespace(s3=s3),
    )
    assert result["status"] == "passed", result
    assert result["media"]["decoded_frames"] == (24 if version == "v2.1" else 12)
    assert "dataset/" not in json.dumps(result)


def test_storage_failure_does_not_print_provider_secret(monkeypatch):
    from botocore.exceptions import ClientError

    client = SimpleNamespace(s3=Mock())
    client.s3.download_file.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "private-credential"}},
        "GetObject",
    )
    result = diagnostics.check_paidf_input(
        input_uri="s3://test-bucket/private.mp4", storage_client=client
    )
    assert result["status"] == "failed"
    assert "private-credential" not in json.dumps(result)


def test_named_project_missing_credentials_never_uses_ambient_boto(monkeypatch):
    from npa.clients import project_credentials

    monkeypatch.setattr(
        project_credentials, "list_projects", lambda: ["synthetic-project"]
    )
    monkeypatch.setattr(
        project_credentials,
        "resolve_project_storage",
        lambda *a, **k: SimpleNamespace(
            endpoint_url="",
            aws_access_key_id="",
            aws_secret_access_key="",
            checkpoint_bucket="",
        ),
    )
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(
        project_credentials,
        "load_credentials",
        Mock(side_effect=AssertionError("no host credential fallback")),
    )
    ambient = Mock(side_effect=AssertionError("no boto default credential chain"))
    monkeypatch.setattr(project_credentials.boto3, "client", ambient)
    result = diagnostics.check_paidf_input(
        project="synthetic-project", input_uri="s3://test-bucket/robot.mp4"
    )
    assert result["status"] == "failed"
    assert "synthetic-project" not in json.dumps(result)
    ambient.assert_not_called()


@pytest.mark.parametrize(
    "code,expected",
    [("AccessDenied", "denied"), ("NoSuchKey", "missing"), ("ExpiredToken", "expired")],
)
def test_storage_failures_have_specific_private_safe_diagnosis(code, expected):
    from botocore.exceptions import ClientError

    client = SimpleNamespace(s3=Mock())
    client.s3.download_file.side_effect = ClientError(
        {"Error": {"Code": code, "Message": "private-provider-detail"}}, "GetObject"
    )
    result = diagnostics.check_paidf_input(
        input_uri="s3://test-bucket/robot.mp4", storage_client=client
    )
    assert result["status"] == "failed"
    assert expected in result["error"]
    assert "private-provider-detail" not in json.dumps(result)
    assert "test-bucket" not in json.dumps(result)
