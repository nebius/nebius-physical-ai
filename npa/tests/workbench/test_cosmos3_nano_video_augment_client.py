"""CPU transport/recovery tests; synthetic bytes are not model-quality evidence."""

from email import message_from_bytes
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from botocore.exceptions import ClientError
import httpx
import pytest

from npa.workbench.cosmos import nano_video_augment_client as client


class StorageFailure(Exception):
    def __init__(self, code):
        self.response = {"Error": {"Code": code}}


class SyntheticStorage:
    def __init__(self):
        self.s3 = self
        self.objects = {("example-bucket", "source.mp4"): b"synthetic-source-video"}
        self.fail_key = None

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
        found = [key for bucket, key in self.objects if bucket == Bucket and key.startswith(Prefix)]
        return {"KeyCount": min(len(found), MaxKeys)}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch):
        assert IfNoneMatch == "*"
        if Key == self.fail_key:
            raise StorageFailure("SyntheticUnavailable")
        if (Bucket, Key) in self.objects:
            raise StorageFailure("PreconditionFailed")
        self.objects[Bucket, Key] = bytes(Body)

    def get_object(self, *, Bucket, Key):
        return {"Body": BytesIO(self.objects[Bucket, Key])}

    def download_file(self, uri, path):
        bucket, key = client._s3(uri)
        Path(path).write_bytes(self.objects[bucket, key])


def _hash(data):
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("actual", [b"expected", b"conflict", b"short"])
def test_key_already_exists_requires_exact_storage_readback(actual):
    calls = []

    def put_object(**kwargs):
        assert kwargs["IfNoneMatch"] == "*"
        raise ClientError({"Error": {"Code": "KeyAlreadyExists",
                          "Message": "Object already exists in the bucket, but If-None-Match header was sent"}},
                          "PutObject")

    def get_object(**kwargs):
        calls.append(kwargs)
        return {"Body": BytesIO(actual)}

    storage = SimpleNamespace(s3=SimpleNamespace(put_object=put_object, get_object=get_object))
    if actual == b"expected":
        assert client._put_verified(storage, "example-bucket", "artifact.mp4", b"expected") == {
            "bytes": 8, "sha256": _hash(b"expected")}
    else:
        with pytest.raises(client.AugmentationClientError, match="differs"):
            client._put_verified(storage, "example-bucket", "artifact.mp4", b"expected")
    assert calls == [{"Bucket": "example-bucket", "Key": "artifact.mp4"}]


@pytest.mark.parametrize("code", ["AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch",
                                  "InternalError", "keyalreadyexists", "KeyAlreadyExistsElsewhere"])
def test_non_collision_storage_failures_are_not_converted_to_readback(code):
    failure = ClientError({"Error": {"Code": code}}, "PutObject")

    def put_object(**kwargs):
        raise failure

    def get_object(**kwargs):
        pytest.fail("unexpected storage error was treated as a conditional-write collision")

    storage = SimpleNamespace(s3=SimpleNamespace(put_object=put_object, get_object=get_object))
    with pytest.raises(ClientError) as observed:
        client._put_verified(storage, "example-bucket", "artifact.mp4", b"expected")
    assert observed.value is failure


@pytest.fixture
def boundary(monkeypatch, tmp_path):
    """Stub only model/decoder boundaries; exercise real HTTP parsing and S3 logic."""
    state = SimpleNamespace(storage=SyntheticStorage(), posts=0, gets=0, report=None,
                            lost_post=False, result_status=200, corrupt=False, mutate_report=None,
                            redirect_download=False)
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_TOKEN", "synthetic-test-token")
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_RECOVERY_DIR", str(tmp_path))
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_ENDPOINT", "http://testserver")
    metadata = {"valid": True, "full_decode_passed": True, "decoded_frames": 9,
                "fps": 24.0, "width": 832, "height": 480, "duration_seconds": 9 / 24,
                "container_duration_seconds": 9 / 24, "timestamps_verified": True}
    monkeypatch.setattr(client, "_probe", lambda *args, **kwargs: metadata)
    state.payloads = {"input.mp4": b"synthetic-source-video",
                      "augmented.mp4": b"synthetic-augmented-video",
                      "comparison.mp4": b"synthetic-paired-video",
                      "chunk-000.mp4": b"synthetic-chunk-video",
                      "control-000.mkv": b"synthetic-structural-control",
                      "gpu-memory.json": b'{"synthetic_fixture":true}'}

    def handler(request):
        assert request.headers["Authorization"] == "Bearer synthetic-test-token"
        if request.method == "POST":
            state.posts += 1
            assert request.url.path == "/run"
            message = message_from_bytes(
                ("Content-Type: " + request.headers["content-type"] + "\r\n\r\n").encode() + request.read())
            parts = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                     for part in message.get_payload()}
            assert set(parts) == {"request", "input_reference"}
            assert parts["input_reference"] == state.payloads["input.mp4"]
            body = json.loads(parts["request"])
            state.payloads["request.json"] = parts["request"]
            state.payloads["request-000.json"] = b'{"synthetic_fixture":true}'
            artifacts = [{"path": name, "bytes": len(data), "sha256": _hash(data)}
                         for name, data in state.payloads.items()]
            state.report = {"schema_version": client.core.SCHEMA, "status": "succeeded",
                            "request_id": body["request_id"], "request": body,
                            "model_revision": client.core.MODEL_REVISION, "replica_id": "synthetic-replica",
                            "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:00:11+00:00",
                            "request_sha256": client.core.request_sha256(body),
                            "source": {**artifacts[0], "video": metadata},
                            "output": {**artifacts[1], "video": metadata},
                            "comparison": {**artifacts[2], "video": {**metadata, "width": 1664}},
                            "chunks": [{"index": 0, "source_start": 0, "source_frames": 9,
                                        "model_chunk_frames": 9, "drop_prefix_frames": 0,
                                        "status": "succeeded", "seed": body["seed"], "http_status": 200,
                                        "output_path": "chunk-000.mp4", "control_path": "control-000.mkv",
                                        "request_path": "request-000.json", "reference_path": None,
                                        "started_at": "2026-01-01T00:00:01+00:00",
                                        "finished_at": "2026-01-01T00:00:10+00:00",
                                        "control_provenance": {"source_start": 0, "source_frames": 9,
                                            "original_source_sha256": body["source_sha256"],
                                            "engine": "vllm-omni.cosmos3.transfer.make_edge_control",
                                            "preset": body["edge_threshold"],
                                            "canny_thresholds": {"low": [50, 100], "medium": [100, 200],
                                                                 "high": [200, 300]}[body["edge_threshold"]],
                                            "source_rgb_sha256": _hash(b"synthetic-decoded-source"),
                                            "control_rgb_sha256": _hash(b"synthetic-decoded-control"),
                                            "upstream_module_sha256": _hash(b"synthetic-upstream-module"),
                                            "lossless_upstream_readback_equal": True,
                                            "source": "original input.mp4 only", "control_video": metadata},
                                        "validation": metadata, "wall_seconds": 9.0,
                                        "server_handler_seconds": 8.8, "engine_peak_memory_mb": 5000,
                                        "device_peak_used_mib": 10000, "stage_durations": {},
                                        "effective": {"positive_prompt": body["prompt"],
                                          "negative_prompt": body["negative_prompt"],
                                          "system_prompt": body["system_prompt"],
                                          "sampling": {"num_inference_steps": body["num_inference_steps"],
                                            "max_sequence_length": body["max_sequence_length"],
                                            "resolution": "480", "fps": 24, "use_system_prompt": True,
                                            "use_duration_template": False, "use_resolution_template": False},
                                          "transfer_config": {
                                            "num_video_frames_per_chunk": 9, "max_frames": 9,
                                            "num_conditional_frames": 5, "num_first_chunk_conditional_frames": 0,
                                            "control_guidance": body["control_guidance"],
                                            "guidance_scale": body["guidance_scale"], "flow_shift": body["flow_shift"],
                                            "share_vision_temporal_positions": True,
                                            "control_guidance_interval": None, "fps": 24.0, "num_frames": 9,
                                            "show_input": False, "show_control_condition": False,
                                            "hints": {"edge": {"key": "edge", "control": None,
                                                                "control_path": str(tmp_path / "control-000.mkv"),
                                                                "preset_blur_strength": "medium",
                                                                "preset_edge_threshold": body["edge_threshold"]}},
                                        }}}],
                            "artifacts": artifacts, "total_wall_seconds": 10.5,
                            "device_peak_used_mib": 10000}
            if state.mutate_report:
                state.mutate_report(state.report)
            if state.lost_post:
                raise httpx.ReadError("synthetic lost POST response", request=request)
            return httpx.Response(200, json=state.report)
        state.gets += 1
        if request.url.path == "/result":
            if state.result_status != 200:
                return httpx.Response(state.result_status, json={"status": "running"})
            return httpx.Response(200, json=state.report)
        data = state.payloads[request.url.path.rsplit("/", 1)[-1]]
        if state.redirect_download:
            return httpx.Response(307, headers={"Location": "https://example.com/leak"})
        return httpx.Response(200, content=b"corrupt" if state.corrupt else data)

    monkeypatch.setattr(client, "_http", lambda token: httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=False,
        headers={"Authorization": f"Bearer {token}"}))
    state.submit = lambda **kwargs: client.submit_augmentation(
        input_path="s3://example-bucket/source.mp4", output_path="s3://example-bucket/result",
        prompt="Synthetic fixture: dim warm damp warehouse", storage_client=state.storage, **kwargs)
    state.recover = lambda: client.recover_augmentation(
        output_path="s3://example-bucket/result", storage_client=state.storage)
    state.root = lambda: client._private_root("s3://example-bucket/result", fresh=False)
    return state


def test_complete_source_upload_download_and_s3_readback(boundary):
    result = boundary.submit()
    assert result["status"] == "succeeded"
    assert result["technical_validation_passed"] is True
    assert result["publication_verified"] is True
    assert result["quality_review_status"] == "pending"
    assert boundary.posts == 1
    assert boundary.storage.objects["example-bucket", "result/input.mp4"] == b"synthetic-source-video"
    assert boundary.storage.objects["example-bucket", "result/control-000.mkv"] == b"synthetic-structural-control"
    assert result["source_sha256"] != result["output_sha256"]


def test_lost_post_response_recovers_same_request_without_repost(boundary):
    boundary.lost_post = True
    assert boundary.submit()["status"] == "succeeded"
    assert boundary.posts == 1
    assert (boundary.root() / "transport-failure.json").is_file()


@pytest.mark.parametrize("status,expected", [(404, "unknown"), (202, "running")])
def test_ambiguous_missing_or_running_never_regenerates(boundary, status, expected):
    boundary.lost_post, boundary.result_status = True, status
    result = boundary.submit()
    assert result["status"] == "pending" and result["generation_status"] == expected
    for _ in range(2):
        assert boundary.recover()["status"] == "pending"
    assert boundary.posts == 1
    boundary.result_status = 200
    assert boundary.recover()["status"] == "succeeded"
    assert boundary.posts == 1


def test_partial_publication_retries_equal_objects_without_service(boundary, monkeypatch):
    boundary.storage.fail_key = "result/comparison.mp4"
    with pytest.raises(StorageFailure):
        boundary.submit()
    assert (boundary.root() / "publication-pending.json").is_file()
    boundary.storage.fail_key = None
    monkeypatch.setattr(client, "_http", lambda token: pytest.fail("publication retry contacted inference service"))
    monkeypatch.delenv("NPA_COSMOS3_VIDEO_TOKEN")
    monkeypatch.delenv("NPA_COSMOS3_VIDEO_ENDPOINT")
    first = boundary.recover()
    before = dict(boundary.storage.objects)
    second = boundary.recover()
    assert first == second and second["status"] == "succeeded"
    assert boundary.storage.objects == before
    assert boundary.posts == 1


def test_conflicting_immutable_object_is_never_overwritten(boundary):
    boundary.submit()
    boundary.storage.objects["example-bucket", "result/augmented.mp4"] = b"conflicting-output"
    with pytest.raises(client.AugmentationClientError, match="differs"):
        boundary.recover()
    assert boundary.storage.objects["example-bucket", "result/augmented.mp4"] == b"conflicting-output"
    assert boundary.posts == 1


def test_corrupt_download_recovery_reuses_completed_generation(boundary):
    boundary.corrupt = True
    with pytest.raises(client.AugmentationClientError, match="hash or length"):
        boundary.submit()
    boundary.corrupt = False
    assert boundary.recover()["status"] == "succeeded"
    assert boundary.posts == 1


def test_incomplete_source_coverage_is_rejected_before_download(boundary):
    boundary.mutate_report = lambda report: report["chunks"][0].update(source_start=1)
    with pytest.raises(client.video.NanoVideoError, match="coverage"):
        boundary.submit()
    assert boundary.posts == 1 and boundary.gets == 0


def test_failed_generation_recovery_never_submits_again(boundary):
    boundary.lost_post, boundary.result_status = True, 202
    assert boundary.submit()["status"] == "pending"
    boundary.result_status = 200
    boundary.report.update(status="failed", error_type="SyntheticGenerationFailure")
    result = boundary.recover()
    assert result["status"] == "failed" and result["generation_status"] == "failed"
    assert result["error_type"] == "SyntheticGenerationFailure"
    assert boundary.posts == 1


def test_existing_prefix_rejected_before_new_generation(boundary):
    boundary.submit()
    with pytest.raises(client.AugmentationClientError, match="recover"):
        boundary.submit()
    assert boundary.posts == 1


@pytest.mark.parametrize("path", ["/tmp/input.mp4", "https://example.com/input.mp4", "s3://example-bucket/",
                                  "s3://example-bucket/input.mp4?token=x", "s3://example-bucket/input.mp4#x"])
def test_invalid_handoff_before_storage_or_generation(monkeypatch, path):
    monkeypatch.setattr(client, "_connection", lambda *args: pytest.fail("invalid path reached connection"))
    with pytest.raises(ValueError):
        client.submit_augmentation(input_path=path, output_path="s3://example-bucket/result", prompt="test")


@pytest.mark.parametrize("name", ["../escape.mp4", "nested/file.mp4", "%2e%2e.mp4", "/absolute.mp4",
                                  ".hidden", "drive:file.mp4", "a\\b.mp4"])
def test_flat_artifact_contract_rejects_unsafe_names(tmp_path, name):
    with pytest.raises(client.AugmentationClientError):
        client._safe_file(tmp_path, name)


def test_artifact_symlink_is_rejected(tmp_path):
    (tmp_path / "input.mp4").symlink_to(tmp_path / "elsewhere.mp4")
    with pytest.raises(client.AugmentationClientError):
        client._safe_file(tmp_path, "input.mp4")


def test_changed_reservation_cannot_recover_another_request(boundary):
    boundary.submit()
    key = ("example-bucket", "result/reservation.json")
    reservation = json.loads(boundary.storage.objects[key])
    reservation["request"]["prompt"] = "different request"
    boundary.storage.objects[key] = json.dumps(reservation).encode()
    with pytest.raises(client.AugmentationClientError, match="digest"):
        boundary.recover()
    assert boundary.posts == 1


@pytest.mark.parametrize("values", [{"seed": True}, {"guidance_scale": float("nan")},
                                   {"control_guidance": float("inf")}, {"chunk_frames": 120},
                                   {"num_inference_steps": 0}, {"max_sequence_length": True},
                                   {"num_inference_steps": 201}, {"guidance_scale": 20.1}])
def test_invalid_sampling_rejected_before_storage_or_generation(boundary, monkeypatch, values):
    monkeypatch.setattr(boundary.storage, "list_objects_v2",
                        lambda **kwargs: pytest.fail("invalid sampling reached storage"))
    with pytest.raises(client.core.AugmentationInputError):
        boundary.submit(**values)
    assert boundary.posts == 0


def test_stale_local_validation_proof_cannot_be_published(boundary):
    boundary.submit()
    proof = boundary.root() / "client-validation.json"
    value = json.loads(proof.read_text())
    value["report_sha256"] = "0" * 64
    proof.write_text(json.dumps(value))
    with pytest.raises(client.AugmentationClientError, match="decode proof"):
        boundary.recover()
    assert boundary.posts == 1


def test_artifact_redirect_is_not_followed_with_bearer_token(boundary):
    boundary.redirect_download = True
    with pytest.raises(httpx.HTTPStatusError):
        boundary.submit()
    assert boundary.posts == 1 and boundary.gets == 1


@pytest.mark.parametrize("field,value", [
    ("positive_prompt", "unrelated fresh text-to-video"),
    ("negative_prompt", "silently changed negative prompt"),
    ("system_prompt", "silently changed system prompt"),
    ("sampling", {"num_inference_steps": 1}),
])
def test_effective_prompt_or_sampling_mismatch_rejected_before_download(boundary, field, value):
    boundary.mutate_report = lambda report: report["chunks"][0]["effective"].update({field: value})
    with pytest.raises(client.video.NanoVideoError, match="coverage"):
        boundary.submit()
    assert boundary.posts == 1 and boundary.gets == 0


@pytest.mark.parametrize("field,value", [
    ("original_source_sha256", "0" * 64),
    ("engine", "synthetic-unrelated-preprocessor"),
    ("preset", "low"),
    ("canny_thresholds", [50, 100]),
    ("source_rgb_sha256", "missing"),
    ("control_rgb_sha256", "missing"),
    ("upstream_module_sha256", "missing"),
])
def test_wrong_control_provenance_rejected_before_download(boundary, field, value):
    boundary.mutate_report = lambda report: report["chunks"][0]["control_provenance"].update({field: value})
    with pytest.raises(client.video.NanoVideoError, match="coverage"):
        boundary.submit()
    assert boundary.posts == 1 and boundary.gets == 0


@pytest.mark.parametrize("value", ["unknown", -1.0, True])
def test_malformed_stage_duration_rejected_before_download(boundary, value):
    boundary.mutate_report = lambda report: report["chunks"][0].update(stage_durations={"denoise": value})
    with pytest.raises(client.video.NanoVideoError, match="coverage"):
        boundary.submit()
    assert boundary.posts == 1 and boundary.gets == 0


@pytest.mark.parametrize("field,value", [
    ("control_guidance_interval", [0.0, 1.0]),
    ("fps", 12), ("num_frames", 8),
    ("show_input", True), ("show_control_condition", True),
    ("unsupported_extra", 1),
])
def test_transfer_override_rejected_before_download(boundary, field, value):
    boundary.mutate_report = lambda report: report["chunks"][0]["effective"]["transfer_config"].update({field: value})
    with pytest.raises(client.video.NanoVideoError, match="coverage"):
        boundary.submit()
    assert boundary.posts == 1 and boundary.gets == 0


@pytest.mark.parametrize("field,value", [
    ("key", "depth"), ("control", [1]), ("control_path", "control-000.mkv"),
    ("preset_blur_strength", "high"), ("unsupported_extra", 1),
])
def test_hint_override_rejected_before_download(boundary, field, value):
    boundary.mutate_report = lambda report: report["chunks"][0]["effective"]["transfer_config"]["hints"]["edge"].update({field: value})
    with pytest.raises(client.video.NanoVideoError, match="coverage"):
        boundary.submit()
    assert boundary.posts == 1 and boundary.gets == 0


def test_full_decode_real_cpu_fixture_and_wrong_frame_count(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg tools unavailable for CPU fixture decode")
    path = tmp_path / "fixture.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=832x480:rate=24",
                    "-frames:v", "9", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    assert client._probe(path, expected_frames=9)["full_decode_passed"] is True
    with pytest.raises(client.AugmentationClientError, match="dimensions, frames"):
        client._probe(path, expected_frames=10)
