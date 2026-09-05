"""Contract tests; live GPU acceptance is the separately retained 1+8 workload."""

from __future__ import annotations

import json
import shutil
import subprocess

import httpx
import pytest

from npa.workbench.cosmos import nano_video as video


@pytest.fixture
def fake_device_sampler(monkeypatch):
    monkeypatch.setattr(video.DeviceMemorySampler, "start", lambda self: None)
    monkeypatch.setattr(video.DeviceMemorySampler, "stop", lambda self: {"peak_used_mib": 15000, "error": None, "samples": []})


def test_rollout_plan_obeys_pixel_and_latent_contract():
    assert all(5 <= count <= 300 and (count - 1) % 4 == 0 for count in video.CHUNK_FRAMES)
    assert sum(video.CHUNK_FRAMES) - 2 * video.PREFIX_FRAMES - 1 == 30 * video.FPS
    first = json.loads(video.request_fields("robot", 3, 297, continuation=False)["extra_params"])
    continuation = json.loads(video.request_fields("robot", 4, 297, continuation=True)["extra_params"])
    assert "condition_video_keep" not in first
    assert continuation["condition_frame_indexes_vision"] == [0, 1]
    assert continuation["condition_video_keep"] == "last"
    assert continuation["max_sequence_length"] == 4096
    assert continuation["guardrails"] is False


@pytest.mark.parametrize("frames", [0, 1, 296, 300, 301, 401])
def test_invalid_frame_request_refused(frames):
    with pytest.raises(ValueError):
        video.request_fields("robot", 4, frames, continuation=True)


@pytest.mark.parametrize("value", [None, "bad", "nan", "inf", "0", "-1"])
def test_missing_or_invalid_memory_measurement_cannot_pass(value):
    headers = httpx.Headers({} if value is None else {"X-Peak-Memory-MB": value})
    with pytest.raises(video.NanoVideoError):
        video._positive_header(headers, "X-Peak-Memory-MB")


def test_failed_gpu_request_is_not_retried_and_evidence_survives(tmp_path, monkeypatch, fake_device_sampler):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] is None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            calls.append(kwargs)
            return httpx.Response(503)

    monkeypatch.setattr(video.httpx, "Client", Client)
    target = tmp_path / "failed"
    with pytest.raises(video.NanoVideoError, match="HTTP 503"):
        video.run_rollout(endpoint="http://localhost", output_dir=target,
                          prompt="robot", seed=7, replica_id="test-replica")
    report = json.loads((target / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["chunks"][0]["http_status"] == 503
    assert report["chunks"][0]["status"] == "failed"
    assert len(calls) == 1
    assert "input_reference" not in calls[0]["files"]
    with pytest.raises(FileExistsError):
        video.run_rollout(endpoint="http://localhost", output_dir=target,
                          prompt="robot", seed=7, replica_id="test-replica")
    assert len(calls) == 1


def test_continuation_uploads_actual_previous_bytes(tmp_path, monkeypatch, fake_device_sampler):
    seen = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, *, files, headers):
            previous = files.get("input_reference")
            seen.append(previous[1].read() if previous else None)
            return httpx.Response(200, content=f"chunk-{len(seen)}".encode(), headers={
                "Content-Type": "video/mp4", "X-Inference-Time-S": "12.5",
                "X-Peak-Memory-MB": "12345", "X-Stage-Durations": '{"denoise":12}',
            })

    monkeypatch.setattr(video.httpx, "Client", Client)
    monkeypatch.setattr(video, "validate_video", lambda path, frames: {"valid": True, "decoded_frames": frames})
    monkeypatch.setattr(video, "stitch_chunks", lambda chunks, target: target.write_bytes(b"stitched-test-fixture"))
    monkeypatch.setattr(video, "seam_evidence", lambda *args: [])
    report = video.run_rollout(endpoint="http://localhost", output_dir=tmp_path / "rollout",
                              prompt="robot", seed=7, replica_id="test-replica")
    assert seen == [None, b"chunk-1", b"chunk-2"]
    assert report["status"] == "succeeded"
    assert [chunk["validation"]["decoded_frames"] for chunk in report["chunks"]] == [297, 297, 137]
    assert report["peak_memory_mb"] == 12345
    assert all("/" not in item["path"] for item in report["artifacts"])


def test_untrusted_artifact_cannot_escape_output_directory(tmp_path):
    with pytest.raises(video.NanoVideoError, match="unsafe artifact"):
        video._download_result(None, "http://localhost", "sample", tmp_path,
                               {"artifacts": [{"path": "../private"}]})


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires FFmpeg")
def test_actual_stitch_retains_720_frames_and_direct_boundaries(tmp_path):
    chunks = []
    for index, (frames, color) in enumerate(zip(video.CHUNK_FRAMES, ["red", "white", "blue"], strict=True)):
        path = tmp_path / f"chunk-{index}.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            f"color={color}:s=832x480:r=24", "-frames:v", str(frames),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path),
        ], check=True)
        chunks.append(path)
    target = tmp_path / "stitched.mp4"
    video.stitch_chunks(chunks, target)
    validation = video.validate_video(target, 720)
    assert validation["duration_seconds"] == 30
    seams = video.seam_evidence(target, tmp_path)
    assert [item["first_new_frame"] for item in seams] == [297, 589]
    assert all(item["boundary_mean_absolute_gray_difference"] > 10 for item in seams)
    assert all(item["neighbor_median_mean_absolute_gray_difference"] < 1 for item in seams)
    assert all((tmp_path / item["contact_sheet"]).stat().st_size > 0 for item in seams)
    with pytest.raises(video.NanoVideoError, match="does not match"):
        video.validate_video(target, 721)
