"""Exercise Marble provider failures, coordinate geometry, and evidence integrity."""

import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from npa.workbench.marble import acquisition, api, runtime
from npa.workbench.marble.cameras import camera_sweep, transform_splats
from npa.workbench.marble.demo import write_demo


@pytest.mark.parametrize("wrapped", [False, True])
def test_completed_operation_resolves_live_and_documented_world_shapes(
    monkeypatch, wrapped
):
    world = {"world_id": "existing-world", "assets": {"splats": {"spz_urls": {}}}}
    calls = []

    def request(method, route):
        calls.append((method, route))
        if route.startswith("operations/"):
            return {"done": True, "response": {"world_id": "existing-world"}}
        return {"world": world} if wrapped else world

    monkeypatch.setattr(api, "api_request", request)
    assert api.await_world("existing-operation")["id"] == "existing-world"
    assert calls == [
        ("GET", "operations/existing-operation"),
        ("GET", "worlds/existing-world"),
    ]


def test_generated_mesh_and_splats_share_provider_metric_transform(monkeypatch):
    monkeypatch.setattr(acquisition, "_generation_operation", lambda _: "accepted")
    monkeypatch.setattr(
        acquisition,
        "await_world",
        lambda _: {
            "id": "test-world",
            "assets": {
                "splats": {
                    "semantics_metadata": {
                        "metric_scale_factor": 2.5,
                        "ground_plane_offset": 1.5,
                    },
                    "spz_urls": {"500k": "https://example.org/world.spz"},
                },
                "mesh": {"collider_mesh_url": "https://example.org/collider.glb"},
            },
        },
    )
    world = acquisition._generated(_generation_request())
    assert world["mesh_transform"] == world["splat_transform"]
    assert world["mesh_transform"] == {
        "scale": [2.5, -2.5, -2.5],
        "translation": [0, 1.5, 0],
    }


def test_missing_key_fails_before_network(monkeypatch):
    monkeypatch.delenv("WLT_API_KEY", raising=False)
    monkeypatch.setattr(api, "load_credentials", lambda: SimpleNamespace(tokens={}))
    monkeypatch.setattr(
        httpx, "request", lambda *a, **k: pytest.fail("unexpected provider call")
    )
    with pytest.raises(api.MarbleError, match="WLT_API_KEY"):
        api.api_request("POST", "worlds:generate", {})


def test_provider_error_does_not_leak_or_retry(monkeypatch):
    monkeypatch.setenv("WLT_API_KEY", "private-test-key")
    calls = []

    def deny(method, url, **kwargs):
        calls.append((method, url))
        return httpx.Response(
            402, text="secret-response", request=httpx.Request(method, url)
        )

    monkeypatch.setattr(httpx, "request", deny)
    with pytest.raises(api.MarbleError, match="HTTP 402") as error:
        api.api_request("POST", "worlds:generate", {})
    assert "secret" not in str(error.value)
    assert len(calls) == 1


def test_camera_poses_are_rigid_and_intrinsics_match():
    poses, intrinsic = camera_sweep(12, 960, 540)
    rotation = poses[:, :3, :3]
    assert np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-6)
    assert np.allclose(np.linalg.det(rotation), 1)
    assert intrinsic[0, 2] == 480 and intrinsic[1, 2] == 270
    assert not np.array_equal(poses[0], poses[-1])


def test_splat_metric_transform_preserves_log_scale_semantics():
    cloud = SimpleNamespace(
        positions=np.array([1.0, 2.0, 3.0]),
        scales=np.zeros(3),
        rotations=np.array([0.0, 0.0, 0.0, 1.0]),
        alphas=np.zeros(1),
        colors=np.zeros(3),
    )
    means, scales, rotation, _, _ = transform_splats(
        cloud, {"scale": [2, -2, -2], "translation": [0, 0.5, 0]}
    )
    assert np.allclose(means, [[2, -3.5, -6]])
    assert np.allclose(np.exp(scales), 2)
    assert np.allclose(rotation, [[0, 0, 0, 1]])


def test_artifact_hash_mismatch_refuses_consumer(monkeypatch, tmp_path):
    manifest = {"files": {"world.spz": {"sha256": "wrong"}}}
    monkeypatch.setattr(
        runtime,
        "read_bytes_uri",
        lambda uri: (
            json.dumps(manifest).encode() if uri.endswith("world.json") else b"tampered"
        ),
    )
    with pytest.raises(api.MarbleError, match="SHA-256"):
        runtime._materialize("s3://example-bucket/world", "world.json", tmp_path)


def test_bundle_paths_cannot_escape(monkeypatch, tmp_path):
    manifest = {"files": {"../outside": {"sha256": "wrong"}}}
    monkeypatch.setattr(
        runtime, "read_bytes_uri", lambda uri: json.dumps(manifest).encode()
    )
    with pytest.raises(api.MarbleError, match="escapes"):
        runtime._materialize("s3://example-bucket/world", "world.json", tmp_path)


@pytest.mark.parametrize("timing", [[], [0], [-1], [float("nan")]])
def test_report_rejects_missing_gpu_work(timing):
    with pytest.raises(api.MarbleError, match="positive measured CUDA"):
        runtime.validate_result(
            {
                "run_id": "run",
                "gpu": {"name": "test GPU"},
                "frames": 1,
                "metrics": {"cuda_frame_ms": timing},
            },
            "run",
        )


def test_html_embedded_metadata_cannot_close_script(tmp_path):
    path = write_demo(tmp_path, {"caption": "</script><script>alert(1)</script>"})
    assert "</script><script>alert" not in path.read_text()
    assert "\\u003c/script>" in path.read_text()


def test_generation_resume_never_reposts(monkeypatch):
    import hashlib

    request = SimpleNamespace(
        output_path="s3://example-bucket/world",
        model="marble-1.1",
        prompt="workshop",
        run_id="run",
    )
    fingerprint = hashlib.sha256(b"marble-1.1\nworkshop").hexdigest()
    monkeypatch.setattr(acquisition, "require_api_key", lambda: "test-key")
    monkeypatch.setattr(
        acquisition,
        "read_bytes_uri",
        lambda _: json.dumps(
            {"request_sha256": fingerprint, "operation_id": "accepted", "run_id": "run"}
        ).encode(),
    )
    monkeypatch.setattr(
        acquisition, "api_request", lambda *a: pytest.fail("duplicate generation")
    )
    assert acquisition._generation_operation(request) == "accepted"


def test_uncertain_generation_is_not_retried(monkeypatch):
    import hashlib

    request = SimpleNamespace(
        output_path="s3://example-bucket/world",
        model="marble-1.1",
        prompt="workshop",
        run_id="run",
    )
    fingerprint = hashlib.sha256(b"marble-1.1\nworkshop").hexdigest()
    monkeypatch.setattr(acquisition, "require_api_key", lambda: "test-key")
    monkeypatch.setattr(
        acquisition,
        "read_bytes_uri",
        lambda _: json.dumps(
            {"request_sha256": fingerprint, "state": "request-intent", "run_id": "run"}
        ).encode(),
    )
    monkeypatch.setattr(
        acquisition, "api_request", lambda *a: pytest.fail("duplicate generation")
    )
    with pytest.raises(api.MarbleError, match="acceptance is uncertain"):
        acquisition._generation_operation(request)


def test_missing_key_does_not_persist_generation_intent(monkeypatch):
    monkeypatch.delenv("WLT_API_KEY", raising=False)
    monkeypatch.setattr(api, "load_credentials", lambda: SimpleNamespace(tokens={}))
    monkeypatch.setattr(
        acquisition, "write_bytes_uri", lambda *a: pytest.fail("unexpected intent")
    )
    with pytest.raises(api.MarbleError, match="WLT_API_KEY"):
        acquisition._generation_operation(SimpleNamespace())


def test_world_api_auth_uses_provider_header_without_logging_secret(
    monkeypatch, capsys
):
    monkeypatch.setenv("WLT_API_KEY", "  unit-world-api-key  ")

    def request(method, url, **kwargs):
        assert method == "POST"
        assert url == "https://api.worldlabs.ai/marble/v1/worlds:generate"
        assert kwargs["headers"] == {"WLT-Api-Key": "unit-world-api-key"}
        return httpx.Response(
            200,
            json={"operation_id": "unit-operation"},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx, "request", request)
    assert api.api_request("POST", "worlds:generate", {}) == {
        "operation_id": "unit-operation"
    }
    assert "unit-world-api-key" not in str(capsys.readouterr())


def _generation_request():
    return SimpleNamespace(
        output_path="s3://example-bucket/world",
        model="marble-1.1",
        prompt="workshop",
        run_id="run",
    )


def test_concurrent_generation_claim_does_not_call_provider(monkeypatch):
    from botocore.exceptions import ClientError
    from npa.clients.storage import StoragePreconditionFailed

    def missing(_):
        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    def conflict(payload, uri, **kwargs):
        assert kwargs["if_none_match"] is True
        assert kwargs["if_match"] == ""
        raise StoragePreconditionFailed("competing writer")

    monkeypatch.setattr(acquisition, "require_api_key", lambda: "unit-key")
    monkeypatch.setattr(acquisition, "read_bytes_uri", missing)
    monkeypatch.setattr(
        acquisition,
        "LazyStorageClient",
        lambda: SimpleNamespace(put_bytes_conditional=conflict),
    )
    monkeypatch.setattr(
        acquisition, "api_request", lambda *a: pytest.fail("duplicate paid request")
    )
    with pytest.raises(api.MarbleError, match="changed concurrently"):
        acquisition._generation_operation(_generation_request())


def test_generation_records_operation_with_claim_version(monkeypatch):
    from botocore.exceptions import ClientError

    writes = []

    def missing(_):
        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    def persist(payload, uri, **kwargs):
        writes.append((json.loads(payload), kwargs))
        return "unit-etag"

    monkeypatch.setattr(acquisition, "require_api_key", lambda: "unit-key")
    monkeypatch.setattr(acquisition, "read_bytes_uri", missing)
    monkeypatch.setattr(
        acquisition,
        "LazyStorageClient",
        lambda: SimpleNamespace(put_bytes_conditional=persist),
    )
    monkeypatch.setattr(
        acquisition, "api_request", lambda *a: {"operation_id": "accepted"}
    )
    assert acquisition._generation_operation(_generation_request()) == "accepted"
    assert writes[0][0]["state"] == "request-intent"
    assert writes[0][1]["if_none_match"] is True
    assert writes[1][0]["operation_id"] == "accepted"
    assert writes[1][1]["if_match"] == "unit-etag"
    assert writes[1][1]["if_none_match"] is False


def test_generation_cannot_claim_another_runs_operation(monkeypatch):
    import hashlib

    journal = {
        "request_sha256": hashlib.sha256(b"marble-1.1\nworkshop").hexdigest(),
        "run_id": "different-run",
        "operation_id": "accepted",
    }
    monkeypatch.setattr(acquisition, "require_api_key", lambda: "unit-key")
    monkeypatch.setattr(
        acquisition, "read_bytes_uri", lambda _: json.dumps(journal).encode()
    )
    monkeypatch.setattr(
        acquisition, "api_request", lambda *a: pytest.fail("unexpected provider call")
    )
    with pytest.raises(api.MarbleError, match="different request or run"):
        acquisition._generation_operation(_generation_request())
