"""Exercise the complete batch fanout gate with deterministic service evidence."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from npa.workbench.cosmos import nano_video as video


def _validation(frames: int) -> dict:
    return {
        "valid": True,
        "decoded_frames": frames,
        "fps": 24.0,
        "width": 832,
        "height": 480,
        "duration_seconds": frames / 24,
        "full_decode_passed": True,
    }


def _service(monkeypatch, *, replicas=None, serial=False, failed_index=None,
             corrupt_download=False, unsafe_artifact=False, serial_chunks=False, mutate_report=None):
    """Install a service that returns fully specified, independently timed runs."""
    payloads = {
        **{f"chunk-{index}.mp4": f"synthetic-chunk-{index}".encode() for index in range(1, 4)},
        "video-30s.mp4": b"synthetic-complete-video",
    }
    artifacts = [
        {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in payloads.items()
    ]
    if unsafe_artifact:
        artifacts[0]["path"] = "../escaped.mp4"
    observed = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] is None
            assert kwargs["trust_env"] is False
            assert kwargs["headers"]["Authorization"] == "Bearer synthetic-test-token"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, url, *, json):
            index = int(json["request_id"].rsplit("-", 1)[1])
            observed.append(json["request_id"])
            request = httpx.Request("POST", url)
            if index == failed_index:
                return httpx.Response(503, request=request)
            started = datetime(2026, 1, 1, tzinfo=timezone.utc)
            if serial:
                started += timedelta(seconds=index * 10)
            report = {
                "schema_version": video.SCHEMA,
                "status": "succeeded",
                "request_id": json["request_id"],
                "started_at": started.isoformat(),
                "finished_at": (started + timedelta(seconds=100 if serial_chunks else 9)).isoformat(),
                "total_wall_seconds": 100 if serial_chunks else 9,
                "device_peak_used_mib": 40000,
                "model": "nvidia/Cosmos3-Nano",
                "model_revision": video.MODEL_REVISION,
                "pipeline": "Cosmos3OmniDiffusersPipeline",
                "guardrails": False,
                "dtype": "bfloat16",
                "tensor_parallel_size": 1,
                "replica_id": replicas[index] if replicas else f"synthetic-replica-{index}",
                "prompt_sha256": hashlib.sha256(json["prompt"].encode()).hexdigest(),
                "seed": json["seed"],
                "chunks": [
                    {
                        "index": ordinal,
                        "status": "succeeded",
                        "requested_frames": frames,
                        "seed": json["seed"] + ordinal - 1,
                        "http_status": 200,
                        "started_at": (started + timedelta(seconds=(index * 9 if serial_chunks else 0) + (ordinal - 1) * 3)).isoformat(),
                        "finished_at": (started + timedelta(seconds=(index * 9 if serial_chunks else 0) + (ordinal - 1) * 3 + 2)).isoformat(),
                        "wall_seconds": 2,
                        "inference_seconds": 1.5,
                        "peak_memory_mb": 12345,
                        "stage_durations": {"denoise": 1.5},
                        "artifact": copy.deepcopy(artifacts[ordinal - 1]),
                        "validation": _validation(frames),
                    }
                    for ordinal, frames in enumerate(video.CHUNK_FRAMES, 1)
                ],
                "artifacts": copy.deepcopy(artifacts),
                "validation": _validation(720),
                "peak_memory_mb": 12345,
                "seams": [],
            }
            if mutate_report:
                mutate_report(report)
            return httpx.Response(200, json=report, request=request)

        def get(self, url):
            name = url.rsplit("/", 1)[-1]
            content = payloads[name]
            if corrupt_download:
                content = b"different-video-bytes"
            return httpx.Response(200, content=content, request=httpx.Request("GET", url))

    monkeypatch.setattr(video.httpx, "Client", Client)
    monkeypatch.setattr(video, "validate_video", lambda path, frames: _validation(frames))
    return observed


def _run(tmp_path: Path):
    return video.run_batch(
        endpoint="http://localhost", output_dir=tmp_path / "batch", concurrency=8,
        token="synthetic-test-token", prompt="synthetic robot scene",
    )


def test_eight_complete_overlapping_distinct_replicas_pass(tmp_path, monkeypatch):
    observed = _service(monkeypatch)
    decoded = []

    def decode(path, frames):
        decoded.append((path.name, frames))
        return _validation(frames)

    monkeypatch.setattr(video, "validate_video", decode)
    batch = _run(tmp_path)
    assert batch["status"] == "succeeded"
    assert batch["fanout_verified"] is True
    assert batch["completed"] == 8
    assert batch["distinct_replicas"] == 8
    assert batch["peak_overlapping_rollouts"] == 8
    assert batch["peak_overlapping_chunk_requests"] == 8
    assert len(decoded) == 32
    assert all(decoded.count(pair) == 8 for pair in [
        ("chunk-1.mp4", 297), ("chunk-2.mp4", 297), ("chunk-3.mp4", 137), ("video-30s.mp4", 720),
    ])
    assert len(observed) == len(set(observed)) == 8
    for item in batch["requests"]:
        report = item["report"]
        assert len(report["chunks"]) == 3
        assert report["validation"]["duration_seconds"] == 30
        path = tmp_path / "batch" / item["request_id"] / "video-30s.mp4"
        assert path.read_bytes() == b"synthetic-complete-video"
    saved = json.loads((tmp_path / "batch" / "batch.json").read_text())
    assert saved["fanout_verified"] is True
    assert "synthetic-test-token" not in json.dumps(saved)


def test_overlapping_requests_on_one_replica_do_not_prove_fanout(tmp_path, monkeypatch):
    _service(monkeypatch, replicas=["synthetic-single-replica"] * 8)
    batch = _run(tmp_path)
    assert batch["completed"] == 8
    assert batch["peak_overlapping_rollouts"] == 8
    assert batch["distinct_replicas"] == 1
    assert batch["status"] == "failed"
    assert batch["fanout_verified"] is False


def test_eight_serial_completions_do_not_prove_concurrency(tmp_path, monkeypatch):
    _service(monkeypatch, serial=True)
    batch = _run(tmp_path)
    assert batch["completed"] == batch["distinct_replicas"] == 8
    assert batch["peak_overlapping_rollouts"] == 1
    assert batch["status"] == "failed"
    assert batch["fanout_verified"] is False


def test_one_failed_complete_request_fails_batch_without_retry(tmp_path, monkeypatch):
    observed = _service(monkeypatch, failed_index=3)
    batch = _run(tmp_path)
    assert len(observed) == len(set(observed)) == 8
    assert batch["completed"] == 7
    assert batch["status"] == "failed"
    assert batch["fanout_verified"] is False
    failed = [item for item in batch["requests"] if item["status"] == "failed"]
    assert len(failed) == 1
    evidence = tmp_path / "batch" / failed[0]["request_id"] / "client-failure.json"
    assert json.loads(evidence.read_text())["status"] == "failed"


@pytest.mark.parametrize("violation", ["hash", "path"])
def test_download_integrity_violation_cannot_pass_batch(tmp_path, monkeypatch, violation):
    _service(monkeypatch, corrupt_download=violation == "hash", unsafe_artifact=violation == "path")
    batch = _run(tmp_path)
    assert batch["status"] == "failed"
    assert batch["fanout_verified"] is False
    assert batch["completed"] == 0
    assert all(item["error_type"] == "NanoVideoError" for item in batch["requests"])
    assert not list(tmp_path.rglob("escaped.mp4"))
    assert len(list((tmp_path / "batch").glob("*/client-failure.json"))) == 8


@pytest.mark.parametrize("output_path", [
    "/tmp/local-output", "file:///tmp/local-output", "https://example.com/output",
    "s3://", "s3://synthetic-bucket", "s3://synthetic-bucket/?signature=sample",
    "s3://synthetic-bucket/output?signature=sample", "s3://synthetic-bucket/output#fragment",
])
def test_invalid_public_output_fails_before_generation(tmp_path, monkeypatch, output_path):
    called = []
    monkeypatch.setattr(video, "run_batch", lambda **kwargs: called.append(kwargs))
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_RECOVERY_DIR", str(tmp_path / "recovery"))
    with pytest.raises(ValueError):
        video.submit_batch(output_path=output_path, concurrency=8, endpoint="http://localhost",
                           storage_client=object())
    assert called == []


@pytest.mark.parametrize("input_path", [
    "/tmp/local-input.json", "file:///tmp/local-input.json", "https://example.com/input.json",
    "s3://", "s3://synthetic-bucket", "s3://synthetic-bucket/input.json?signature=sample",
    "s3://synthetic-bucket/input.json#fragment",
])
def test_invalid_public_input_fails_before_download_or_generation(tmp_path, monkeypatch, input_path):
    called = []
    monkeypatch.setattr(video, "run_batch", lambda **kwargs: called.append(kwargs))
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_RECOVERY_DIR", str(tmp_path / "recovery"))

    class Storage:
        def download_file(self, *args):
            pytest.fail("Invalid public input must fail before download")

    with pytest.raises(ValueError):
        video.submit_batch(output_path="s3://synthetic-bucket/output", input_path=input_path,
                           concurrency=8, endpoint="http://localhost", storage_client=Storage())
    assert called == []


def test_publication_hash_failure_keeps_complete_local_run_without_retry(tmp_path, monkeypatch):
    import io

    calls = []
    retained = tmp_path / "recovery"
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_RECOVERY_DIR", str(retained))
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_TOKEN", "synthetic-test-token")

    def complete_run(**kwargs):
        calls.append(kwargs)
        output = kwargs["output_dir"]
        output.mkdir()
        (output / "video-30s.mp4").write_bytes(b"completed-synthetic-video")
        report = {"status": "succeeded", "concurrency": 8, "completed": 8,
                  "distinct_replicas": 8, "peak_overlapping_rollouts": 8,
                  "total_wall_seconds": 9}
        (output / "batch.json").write_text(json.dumps(report))
        return report

    class S3:
        def __init__(self):
            self.objects = {}

        def list_objects_v2(self, **kwargs):
            return {"Contents": []}

        def head_bucket(self, **kwargs):
            return {}

        def put_object(self, **kwargs):
            self.objects[kwargs["Key"]] = kwargs["Body"]

        def get_object(self, **kwargs):
            value = self.objects[kwargs["Key"]]
            if kwargs["Key"].endswith("video-30s.mp4"):
                value = b"corrupt-object-after-completed-generation"
            return {"Body": io.BytesIO(value)}

        def delete_object(self, **kwargs):
            self.objects.pop(kwargs["Key"], None)

    class Storage:
        s3 = S3()

    monkeypatch.setattr(video, "run_batch", complete_run)
    with pytest.raises(video.NanoVideoError, match="hash mismatch"):
        video.submit_batch(output_path="s3://synthetic-bucket/output", concurrency=8,
                           endpoint="http://localhost", storage_client=Storage())
    assert len(calls) == 1
    copies = list(retained.glob("*/batch/video-30s.mp4"))
    assert len(copies) == 1
    assert copies[0].read_bytes() == b"completed-synthetic-video"
    report = json.loads(copies[0].with_name("batch.json").read_text())
    assert report["completed"] == 8
    assert report["status"] == "succeeded"


@pytest.mark.parametrize("gpu_rows", [
    b"NVIDIA H200, 20000, 140000\n",
    b"NVIDIA B200, 20000, 180000\nNVIDIA B200, 20000, 180000\n",
])
def test_device_memory_sampler_refuses_wrong_or_multiple_gpus_before_generation(monkeypatch, gpu_rows):
    monkeypatch.setattr(video, "_command", lambda _argv: gpu_rows)
    sampler = video.DeviceMemorySampler()
    with pytest.raises(video.NanoVideoError):
        sampler.start()
    assert sampler.samples == []
    assert sampler._thread is None


def test_existing_s3_output_prefix_fails_before_gpu_or_write(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(video, "run_batch", lambda **kwargs: called.append(kwargs))
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_RECOVERY_DIR", str(tmp_path / "recovery"))

    class S3:
        def list_objects_v2(self, **kwargs):
            return {"KeyCount": 1, "Contents": [{"Key": "output/reservation.json"}]}

        def put_object(self, **kwargs):
            pytest.fail("Existing output must fail before any write")

    class Storage:
        s3 = S3()

    with pytest.raises(video.NanoVideoError, match="already contains"):
        video.submit_batch(output_path="s3://synthetic-bucket/output", concurrency=8,
                           endpoint="http://localhost", storage_client=Storage())
    assert called == []


def test_vram_sample_targets_ray_assigned_gpu_on_multigpu_worker(monkeypatch):
    observed = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    def command(argv):
        observed.append(argv)
        return b"NVIDIA B200, 40000, 180000\n"

    monkeypatch.setattr(video, "_command", command)
    sampler = video.DeviceMemorySampler()
    sampler._sample()
    assert "--id=3" in observed[0]
    assert sampler.samples[0]["used_mib"] == 40000
    assert sampler.samples[0]["total_mib"] == 180000


def test_overlapping_rollouts_with_serial_chunk_requests_fail_fanout(tmp_path, monkeypatch):
    _service(monkeypatch, serial_chunks=True)
    batch = _run(tmp_path)
    assert batch["completed"] == batch["distinct_replicas"] == 8
    assert batch["peak_overlapping_rollouts"] == 8
    assert batch["peak_overlapping_chunk_requests"] == 1
    assert batch["status"] == "failed"
    assert batch["fanout_verified"] is False


@pytest.mark.parametrize("missing", [
    "schema_version", "model", "model_revision", "pipeline", "dtype", "tensor_parallel_size",
    "guardrails", "replica_id", "device_peak_used_mib", "total_wall_seconds", "validation",
])
def test_success_status_without_required_report_evidence_is_rejected(tmp_path, monkeypatch, missing):
    _service(monkeypatch, mutate_report=lambda report: report.pop(missing))
    batch = _run(tmp_path)
    assert batch["completed"] == 0
    assert batch["fanout_verified"] is False


@pytest.mark.parametrize("field,value", [
    ("status", "running"), ("requested_frames", 297), ("inference_seconds", None),
    ("inference_seconds", 0), ("wall_seconds", -1), ("peak_memory_mb", "unknown"),
    ("http_status", 503), ("started_at", "invalid-time"),
])
def test_incomplete_third_chunk_cannot_claim_complete_generation(tmp_path, monkeypatch, field, value):
    _service(monkeypatch, mutate_report=lambda report: report["chunks"][2].update({field: value}))
    batch = _run(tmp_path)
    assert batch["completed"] == 0
    assert batch["status"] == "failed"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_nonfinite_or_boolean_measurement_is_refused(value):
    with pytest.raises(video.NanoVideoError):
        video._measurement(value, "inference_seconds")


def test_missing_chunk_mp4_manifest_fails_before_download(tmp_path, monkeypatch):
    _service(monkeypatch, mutate_report=lambda report: report["artifacts"].pop(1))
    batch = _run(tmp_path)
    assert batch["completed"] == 0
    assert not list((tmp_path / "batch").glob("*/*.mp4"))
