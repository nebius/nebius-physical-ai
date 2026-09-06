"""CPU verification of authentication, immutable requests, and checkpoint integrity."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from npa.workbench.cosmos import (
    nano_video,
    nano_video_server as server,
    nano_video_stage as stage,
)


def request(body=None, token="test-only-token"):
    headers = [(b"authorization", f"Bearer {token}".encode())] if token else []

    async def receive():
        return {
            "type": "http.request",
            "body": json.dumps(body).encode(),
            "more_body": False,
        }

    return Request({"type": "http", "headers": headers}, receive=receive)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    model = tmp_path / "model"
    (model / "vae").mkdir(parents=True)
    (model / "READY.json").write_text(
        json.dumps(
            {
                "revision": server.MODEL_REVISION,
                "precision": "BF16",
                "tensor_count": 1,
                "files": [{"path": "model.safetensors"}],
            }
        )
    )
    (model / "model_index.json").write_text(
        json.dumps({"_class_name": server.PIPELINE})
    )
    (model / "vae" / "config.json").write_text(
        json.dumps({"scale_factor_temporal": 4, "scale_factor_spatial": 16})
    )
    monkeypatch.setenv("NPA_COSMOS3_MODEL_PATH", str(model))
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("NPA_COSMOS3_VIDEO_TOKEN", "test-only-token")
    result = server.NanoVideoRuntime()
    result.endpoint = "http://127.0.0.1:8000"
    return result


def test_authentication_precedes_generation_and_file_access(runtime, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("unauthenticated request reached generation")

    monkeypatch.setattr(nano_video, "run_rollout", forbidden)
    for token in ("", "incorrect"):
        with pytest.raises(HTTPException) as caught:
            asyncio.run(
                runtime.run(
                    request(
                        {"prompt": "scene", "seed": 1, "request_id": "run-1"}, token
                    )
                )
            )
        assert caught.value.status_code == 401
        with pytest.raises(HTTPException) as caught:
            runtime.artifact(request(token=token), "run-1", "video.mp4")
        assert caught.value.status_code == 401
    assert list(runtime.output_root.iterdir()) == []


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "scene", "seed": 1, "request_id": "../outside"},
        {"prompt": "scene", "seed": True, "request_id": "run-1"},
        {"prompt": "scene", "seed": -1, "request_id": "run-1"},
        {"prompt": " ", "seed": 1, "request_id": "run-1"},
        {"prompt": "scene", "seed": 1, "request_id": "run-1", "model_path": "/other"},
    ],
)
def test_invalid_request_never_creates_output(runtime, body):
    with pytest.raises(HTTPException) as caught:
        asyncio.run(runtime.run(request(body)))
    assert caught.value.status_code == 400
    assert list(runtime.output_root.iterdir()) == []


def test_duplicate_request_is_conflict_without_repeating_generation(
    runtime, monkeypatch
):
    calls = []

    def generate(**kwargs):
        kwargs["output_dir"].mkdir(exist_ok=False)
        calls.append(kwargs)
        return {"status": "succeeded", "artifacts": []}

    monkeypatch.setattr(nano_video, "run_rollout", generate)
    payload = {"prompt": "scene", "seed": 1, "request_id": "run-1"}
    response = asyncio.run(runtime.run(request(payload)))
    assert response["request_id"] == "run-1"
    with pytest.raises(HTTPException) as caught:
        asyncio.run(runtime.run(request(payload)))
    assert caught.value.status_code == 409
    assert len(calls) == 1


def test_failure_response_does_not_disclose_exception_details(runtime, monkeypatch):
    def generate(**kwargs):
        kwargs["output_dir"].mkdir()
        (kwargs["output_dir"] / "report.json").write_text(
            json.dumps({"status": "failed"})
        )
        raise RuntimeError("private operational details")

    monkeypatch.setattr(nano_video, "run_rollout", generate)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            runtime.run(request({"prompt": "scene", "seed": 1, "request_id": "run-1"}))
        )
    assert caught.value.status_code == 500
    assert "private operational details" not in caught.value.detail
    assert runtime.status(request(), "run-1")["status"] == "failed"


def test_artifacts_require_manifest_membership_and_cannot_follow_symlinks(
    runtime, tmp_path
):
    directory = runtime.output_root / "run-1"
    directory.mkdir()
    (directory / "video.mp4").write_bytes(b"test video bytes")
    (directory / "unlisted.json").write_text("{}")
    (directory / "report.json").write_text(
        json.dumps({"status": "succeeded", "artifacts": [{"path": "video.mp4"}]})
    )
    assert (
        runtime.artifact(request(), "run-1", "video.mp4").path
        == directory / "video.mp4"
    )
    for name in ("unlisted.json", "report.json", "../outside.mp4"):
        with pytest.raises(HTTPException):
            runtime.artifact(request(), "run-1", name)
    (directory / "video.mp4").unlink()
    (tmp_path / "outside.mp4").write_bytes(b"private")
    (directory / "video.mp4").symlink_to(tmp_path / "outside.mp4")
    with pytest.raises(HTTPException):
        runtime.artifact(request(), "run-1", "video.mp4")


def test_pipeline_cli_preserves_the_requested_diffusion_contract():
    config = Path("/tmp/synthetic-stage.json")
    args = server.server_argv(Path("/models/Cosmos3-Nano"), 18080, config)
    assert args[:3] == ["vllm", "serve", "/models/Cosmos3-Nano"]
    assert "--omni" in args and "--no-guardrails" in args
    for option, expected in (
        ("--init-timeout", "1800"),
        ("--tensor-parallel-size", "1"),
        ("--dtype", "bfloat16"),
    ):
        assert args[args.index(option) + 1] == expected
    assert args[args.index("--model-class-name") + 1] == server.PIPELINE
    assert args[args.index("--stage-configs-path") + 1] == str(config)
    assert "--stage-overrides" not in args
    assert "--enable-diffusion-pipeline-profiler" in args


def test_explicit_stage_keeps_audio_off_without_custom_pipeline_override():
    stages = server.diffusion_stage_config()["stage_args"]
    assert len(stages) == 1
    stage = stages[0]
    assert stage["stage_id"] == 0 and stage["stage_type"] == "diffusion"
    assert stage["runtime"] == {"process": True, "devices": "0"}
    engine = stage["engine_args"]
    assert engine["model_config"] == {"sound_gen": False, "guardrails": False}
    assert engine["dtype"] == "bfloat16"
    assert engine["parallel_config"] == {"tensor_parallel_size": 1}
    assert engine["enable_diffusion_pipeline_profiler"] is True
    assert "custom_pipeline_args" not in engine


def test_checkpoint_mismatch_fails_before_launch(runtime):
    marker = runtime.model_path / "READY.json"
    payload = json.loads(marker.read_text())
    payload["revision"] = "incorrect"
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="revision"):
        server.validate_weights(runtime.model_path)


def tensor_file(path, dtype="BF16"):
    header = json.dumps(
        {"tensor": {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]}}
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")


def test_staging_verifies_all_shards_precision_and_file_hashes(tmp_path):
    transformer = tmp_path / "transformer"
    tensor_file(transformer / "shard.safetensors")
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"tensor": "shard.safetensors"}})
    )
    tensor_file(tmp_path / "vae" / "model.safetensors")
    manifest = stage.checkpoint_manifest(tmp_path)
    assert manifest["tensor_count"] == 2
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    tensor_file(tmp_path / "vae" / "model.safetensors", "F32")
    with pytest.raises(ValueError, match="precision"):
        stage.checkpoint_manifest(tmp_path)
    (transformer / "shard.safetensors").unlink()
    tensor_file(tmp_path / "vae" / "model.safetensors")
    with pytest.raises(ValueError, match="shard"):
        stage.checkpoint_manifest(tmp_path)


def test_router_considers_every_replica_in_one_rank(monkeypatch):
    # Test the candidate policy independently of Ray's installed runtime.
    class BaseRouter:
        def __init__(self, **kwargs):
            pass

    fake = ModuleType("ray.serve.request_router")
    fake.FIFOMixin = type("FIFOMixin", (), {})
    fake.RequestRouter = BaseRouter
    monkeypatch.setitem(sys.modules, "ray.serve.request_router", fake)
    path = Path(server.__file__).with_name("nano_video_router.py")
    spec = importlib.util.spec_from_file_location("test_nano_video_router", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidates = list(range(16))
    chosen = asyncio.run(module.LeastOutstandingRouter().choose_replicas(candidates))
    assert chosen == [candidates]
    assert candidates == list(range(16))
