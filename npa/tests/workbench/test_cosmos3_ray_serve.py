"""Client integrity checks against the pinned native SampleOutputs contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from npa.workbench.cosmos.ray_serve import (
    Cosmos3RayServeError,
    RayBatchRequest,
    load_batch_request,
    submit_batch,
)

MEDIA = b"synthetic-image-bytes"


def _batch(path: Path, **overrides: object) -> None:
    path.write_text(
        json.dumps(
            {
                "model": "Cosmos3-Nano",
                "request_id": "request-safe",
                "samples": [
                    {
                        "name": "one",
                        "model_mode": "text2image",
                        "prompt": "red cube",
                        "seed": 17,
                    },
                    {
                        "name": "two",
                        "model_mode": "text2image",
                        "prompt": "blue cube",
                        "seed": 23,
                    },
                ],
                **overrides,
            }
        ),
        encoding="utf-8",
    )


def _response(request_id: str = "request-safe") -> dict:
    # Native OmniSampleArgs is resolved (including num_frames) and the output
    # writer records each file relative to the service output root.
    return {
        "schema_version": "npa.cosmos3.ray-serve.batch.v1",
        "request_id": request_id,
        "model": "Cosmos3-Nano",
        "batch_size": 2,
        "outputs": [
            {
                "args": {
                    "name": name,
                    "model": "",
                    "model_mode": "text2image",
                    "num_frames": 1,
                    "seed": seed,
                },
                "status": "success",
                "outputs": [
                    {"content": {}, "files": [f"{request_id}/{name}/vision.jpg"]}
                ],
            }
            for name, seed in [("one", 17), ("two", 23)]
        ],
        "artifacts": [
            _artifact(f"{request_id}/{name}/vision.jpg", name)
            for name in ("one", "two")
        ],
        "guardrails": True,
        "max_batch_size": 4,
        "framework_revision": "5e67049cd94acb667786f1e6dd0dab821cb90c97",
        "server_source_revision": "a" * 40,
    }


def _artifact(path: str, sample: str) -> dict:
    return {
        "sample": sample,
        "path": path,
        "bytes": len(MEDIA),
        "sha256": hashlib.sha256(MEDIA).hexdigest(),
    }


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Mock, Mock]:
    payload = _response()
    post = Mock(
        side_effect=lambda method, url, **kwargs: httpx.Response(
            200, request=httpx.Request(method, url), json=payload
        )
    )
    get = Mock(
        side_effect=lambda url, **kwargs: httpx.Response(
            200, request=httpx.Request("GET", url), content=MEDIA
        )
    )
    monkeypatch.setattr(httpx, "request", post)
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setenv("NPA_COSMOS3_RAY_TOKEN", "secret-for-test")
    return payload, post, get


def _submit(source: Path, destination: str, **kwargs: object) -> dict:
    return submit_batch(
        input_path=str(source),
        output_path=destination,
        endpoint="http://service.invalid:8000",
        **kwargs,
    )


def test_batch_requires_unique_named_samples() -> None:
    with pytest.raises(ValueError, match="duplicate sample"):
        RayBatchRequest(samples=[{"name": "same"}, {"name": "same"}])
    with pytest.raises(ValueError, match="name is required"):
        RayBatchRequest(samples=[{"prompt": "missing"}])
    with pytest.raises(ValueError, match="must use only"):
        RayBatchRequest(samples=[{"name": "../../escape"}])


@pytest.mark.parametrize("mode", ["", "none", "disabled", "false"])
def test_server_rejects_disabled_management_auth_before_runtime_import(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from npa.workbench.cosmos.ray_server import main

    monkeypatch.setenv("RAY_AUTH_MODE", mode)
    with pytest.raises(RuntimeError, match="requires RAY_AUTH_MODE=token"):
        main()


def test_server_enables_management_auth_without_reusing_application_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    from npa.workbench.cosmos.ray_server import _require_ray_authentication

    monkeypatch.delenv("RAY_AUTH_MODE", raising=False)
    monkeypatch.delenv("RAY_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("RAY_AUTH_TOKEN_PATH", "operator-token-file")
    monkeypatch.setenv("NPA_COSMOS3_RAY_TOKEN", "application-test-credential")
    token_file = _require_ray_authentication()
    try:
        assert os.environ["RAY_AUTH_MODE"] == "token"
        assert "RAY_AUTH_TOKEN" not in os.environ
        assert os.environ["RAY_AUTH_TOKEN_PATH"] == str(token_file)
        assert token_file.stat().st_mode & 0o777 == 0o600
        assert len(token_file.read_bytes()) >= 32
        assert token_file.read_text() != "application-test-credential"
    finally:
        token_file.unlink()


def test_server_removes_management_credential_when_runtime_fails(monkeypatch, tmp_path):
    from npa.workbench.cosmos import ray_server

    credential = tmp_path / "credential"
    credential.write_text("private test credential")
    monkeypatch.setattr(ray_server, "_require_ray_authentication", lambda: credential)

    def fail():
        raise RuntimeError("runtime failed")

    monkeypatch.setattr(ray_server, "_run_server", fail)
    with pytest.raises(RuntimeError, match="runtime failed"):
        ray_server.main()
    assert not credential.exists()


def test_batch_ingress_keeps_json_body_after_ray_signature_rewrite(monkeypatch, tmp_path):
    import inspect
    import sys
    from types import ModuleType, SimpleNamespace

    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    from npa.workbench.cosmos import ray_server

    class CapturedIngress(Exception):
        pass

    captured = {}

    def ingress(api):
        def capture(cls):
            captured["router"] = cls(None)
            raise CapturedIngress

        return capture

    ray = ModuleType("ray")
    serve = ModuleType("ray.serve")
    serve.Deployment = object
    serve.deployment = lambda **kwargs: lambda cls: cls
    serve.ingress = ingress
    ray.serve = serve
    upstream_args = ModuleType("cosmos_framework.inference.args")
    upstream_args.OmniSampleOverrides = Mock()
    upstream_args.OmniSetupOverrides = SimpleNamespace(
        model_validate=lambda value: SimpleNamespace(build_setup=lambda **kwargs: None)
    )
    upstream_serve = ModuleType("cosmos_framework.inference.ray.serve")
    upstream_serve.OmniModelDeployment = Mock()
    for name, module in {
        "ray": ray,
        "ray.serve": serve,
        "cosmos_framework.inference.args": upstream_args,
        "cosmos_framework.inference.ray.serve": upstream_serve,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("NPA_COSMOS3_RAY_TOKEN", "test-application-token")
    monkeypatch.setenv("HF_TOKEN", "test-model-token")
    monkeypatch.setenv("NPA_COSMOS3_RAY_OUTPUT_DIR", str(tmp_path))
    with pytest.raises(CapturedIngress):
        ray_server._run_server()

    router = captured["router"]
    endpoint = type(router).batches
    signature = inspect.signature(endpoint)
    parameters = list(signature.parameters.values())
    # Ray's class-based ingress injects self and makes the remaining arguments
    # keyword-only, then FastAPI analyzes the rewritten endpoint again.
    endpoint.__signature__ = signature.replace(parameters=[
        parameters[0].replace(default=Depends(lambda: router)),
        *(parameter.replace(kind=inspect.Parameter.KEYWORD_ONLY)
          for parameter in parameters[1:]),
    ])
    api = FastAPI()
    api.post("/v1/batches")(endpoint)
    schema = api.openapi()["paths"]["/v1/batches"]["post"]
    assert "application/json" in schema["requestBody"]["content"]
    assert all(parameter["name"] != "body" for parameter in schema.get("parameters", []))
    payload = {"model": "not-loaded", "samples": [{"name": "one", "prompt": "cube"}]}
    with TestClient(api) as client:
        assert client.post("/v1/batches", json=payload).status_code == 401
        response = client.post(
            "/v1/batches", json=payload,
            headers={"Authorization": "Bearer test-application-token"},
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "model 'not-loaded' is not loaded"
        missing_body = client.post(
            "/v1/batches", params={"body": json.dumps(payload)},
            headers={"Authorization": "Bearer test-application-token"},
        )
        assert missing_body.status_code == 422
        assert missing_body.json()["detail"][0]["loc"] == ["body"]


def test_load_batch_accepts_list_shorthand(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text('[{"name":"one","prompt":"cube"}]', encoding="utf-8")
    assert load_batch_request(str(path)).samples[0]["name"] == "one"


def test_dry_run_is_import_light_and_keeps_guardrail_posture_server_owned(
    tmp_path: Path,
) -> None:
    source = tmp_path / "batch.json"
    _batch(source)
    result = _submit(source, str(tmp_path / "out"), dry_run=True)
    assert result["status"] == "planned"
    assert result["batch_size"] == 2
    assert result["backend"] == "cosmos-framework-native-ray-serve"
    assert result["weights_baked"] is False


def test_submit_downloads_hash_checks_and_persists_structured_outputs(
    tmp_path: Path, transport
) -> None:
    source = tmp_path / "batch.json"
    destination = tmp_path / "published"
    _batch(source)
    payload, post, get = transport
    result = _submit(source, str(destination))
    assert result["status"] == "completed"
    assert result["guardrails"] is True
    assert post.call_args.kwargs["json"]["request_id"] == "request-safe"
    assert get.call_count == 2
    for name in ("one", "two"):
        assert (
            destination / f"artifacts/request-safe/{name}/vision.jpg"
        ).read_bytes() == MEDIA
    assert (
        json.loads((destination / "request.json").read_text())["request_id"]
        == "request-safe"
    )
    assert json.loads((destination / "response.json").read_text()) == payload
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["structured_outputs"] == payload["outputs"]


def test_generated_request_identity_is_sent_and_persisted(
    tmp_path: Path, transport
) -> None:
    source = tmp_path / "batch.json"
    _batch(source, request_id="")
    _, post, _ = transport
    post.side_effect = lambda method, url, **kwargs: httpx.Response(
        200,
        request=httpx.Request(method, url),
        json=_response(kwargs["json"]["request_id"]),
    )
    destination = tmp_path / "out"
    first = _submit(source, str(destination))
    assert len(first["request_id"]) == 32
    assert (
        json.loads((destination / "request.json").read_text())["request_id"]
        == first["request_id"]
    )
    second = _submit(source, str(tmp_path / "second"))
    assert second["request_id"] != first["request_id"]


def _set(payload: dict, path: tuple, value: object) -> None:
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), "unknown"),
        (("schema_version",), None),
        (("model",), "Cosmos3-Super"),
        (("framework_revision",), "unknown"),
        (("request_id",), "foreign"),
        (("batch_size",), 1),
        (("batch_size",), True),
        (("outputs",), []),
        (("outputs", 1, "args", "name"), "one"),
        (("outputs", 1, "args", "name"), "foreign"),
        (("outputs", 1, "args", "name"), None),
        (("outputs", 1, "args"), {}),
        (("outputs", 1, "args"), None),
        (("outputs", 1, "status"), "error"),
        (("outputs", 1, "status"), "skip"),
        (("outputs", 1, "status"), "unknown"),
        (("outputs", 1, "status"), None),
        (("outputs", 1, "args", "seed"), 999),
        (("outputs", 1, "args", "model_mode"), "text2video"),
        (("outputs", 1, "args", "model_mode"), []),
        (("outputs", 1, "args", "num_frames"), 0),
        (("outputs", 1, "args", "num_frames"), True),
        (("outputs", 1, "outputs"), []),
        (("outputs", 1, "outputs"), None),
        (("outputs", 1, "outputs", 0), "invalid"),
        (("outputs", 1, "outputs", 0, "files"), []),
        (("outputs", 1, "outputs", 0, "files"), "request-safe/two/vision.jpg"),
        (("outputs", 1, "outputs", 0, "content"), None),
        (("artifacts",), []),
        (("artifacts", 1, "sample"), "foreign"),
        (("artifacts", 1, "sample"), "one"),
        (("artifacts", 1, "path"), "request-safe/two/unlisted.jpg"),
        (("artifacts", 1, "sha256"), "invalid"),
        (("artifacts", 1, "bytes"), 0),
        (("artifacts", 1, "bytes"), True),
    ],
)
@pytest.mark.parametrize("s3", [False, True])
def test_invalid_response_never_downloads_or_publishes(
    tmp_path: Path, transport, path, value, s3
) -> None:
    source = tmp_path / "batch.json"
    _batch(source)
    payload, _, get = transport
    _set(payload, path, value)
    storage = Mock()
    destination = tmp_path / "out"
    with pytest.raises(Cosmos3RayServeError):
        _submit(
            source,
            "s3://test-bucket/result/" if s3 else str(destination),
            storage_client=storage,
        )
    get.assert_not_called()
    storage.upload_directory.assert_not_called()
    assert not destination.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_schema",
        "missing_sample_artifact",
        "duplicate_artifact",
        "duplicate_file",
        "foreign_file",
        "debug_only",
    ],
)
def test_incomplete_and_duplicate_manifests_are_rejected(
    tmp_path: Path, transport, mutation: str
) -> None:
    source = tmp_path / "batch.json"
    _batch(source)
    payload, _, get = transport
    if mutation == "missing_schema":
        del payload["schema_version"]
    elif mutation == "missing_sample_artifact":
        payload["artifacts"].pop()
    elif mutation == "duplicate_artifact":
        payload["artifacts"].append(copy.deepcopy(payload["artifacts"][0]))
    elif mutation == "duplicate_file":
        payload["outputs"][0]["outputs"][0]["files"] *= 2
    elif mutation == "foreign_file":
        payload["artifacts"].append(
            _artifact("request-safe/foreign/vision.jpg", "foreign")
        )
    else:
        payload["outputs"][1]["outputs"][0]["files"] = [
            "request-safe/two/output.safetensors"
        ]
        payload["artifacts"][1]["path"] = "request-safe/two/output.safetensors"
    with pytest.raises(Cosmos3RayServeError):
        _submit(source, str(tmp_path / "out"))
    get.assert_not_called()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/request-safe/two/vision.jpg",
        "request-safe/two/../one/vision.jpg",
        "request-safe/two/./vision.jpg",
        "request-safe/two//vision.jpg",
        "request-safe/two/vision.jpg/",
        "request-safe/two/vision.jpg?download=1",
        "request-safe/two/vision.jpg#fragment",
        "request-safe/two/%2e%2e/vision.jpg",
        "request-safe/two/%252e%252e/vision.jpg",
        "request-safe/two/vision%2f.jpg",
        "request-safe/two/vision\\image.jpg",
        "request-safe/two/vision\x00.jpg",
        "request-safe/two/vision\n.jpg",
        "foreign/two/vision.jpg",
        "request-safe/one/vision.jpg",
        "https://foreign/vision.jpg",
    ],
)
def test_paths_are_safe_even_when_both_manifests_agree(
    tmp_path: Path, transport, path: str
) -> None:
    source = tmp_path / "batch.json"
    _batch(source)
    payload, _, get = transport
    payload["outputs"][1]["outputs"][0]["files"] = [path]
    payload["artifacts"][1]["path"] = path
    with pytest.raises(Cosmos3RayServeError):
        _submit(source, str(tmp_path / "out"))
    get.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "frames", "primary", "extra"),
    [
        ("text2image", 1, "vision.jpg", "output.safetensors"),
        ("text2video", 9, "vision.mp4", "output.pickle"),
        ("reasoner", 1, "reasoner_text.txt", None),
        ("video2video", 9, "vision.mp4", "control_depth.mp4"),
        ("forward_dynamics", 9, "vision.mp4", None),
    ],
)
def test_output_contract_preserves_text_video_control_and_debug_files(
    tmp_path: Path,
    transport,
    mode,
    frames,
    primary,
    extra,
) -> None:
    source = tmp_path / "batch.json"
    _batch(
        source,
        samples=[
            {
                "name": "one",
                "model_mode": mode,
                "num_frames": 6 if frames > 1 and mode != "reasoner" else 1,
            }
        ],
    )
    payload, _, get = transport
    payload["batch_size"] = 1
    payload["outputs"] = payload["outputs"][:1]
    args = payload["outputs"][0]["args"]
    args.update(model_mode=mode, num_frames=frames)
    files = [f"request-safe/one/{primary}"]
    if extra:
        files.append(f"request-safe/one/{extra}")
    content = (
        {"reasoner_text": "A robotic workcell."}
        if mode == "reasoner"
        else {"action": [[0.1, 0.2]]}
    )
    payload["outputs"][0]["outputs"] = [{"content": content, "files": files}]
    payload["artifacts"] = [_artifact(p, "one") for p in files]
    result = _submit(source, str(tmp_path / "out"))
    assert result["status"] == "completed"
    assert get.call_count == len(files)


def test_resolved_defaults_and_reordered_results_match_by_sample_identity(
    tmp_path: Path, transport
) -> None:
    source = tmp_path / "batch.json"
    _batch(
        source,
        samples=[{"name": "one", "num_frames": 1}, {"name": "two", "num_frames": 1}],
    )
    payload, _, _ = transport
    payload["outputs"].reverse()
    payload["artifacts"].reverse()
    assert _submit(source, str(tmp_path / "out"))["status"] == "completed"


@pytest.mark.parametrize("corrupt", ["bytes", "sha256"])
def test_submit_rejects_artifact_integrity_mismatch(
    tmp_path: Path, transport, corrupt: str
) -> None:
    source = tmp_path / "batch.json"
    _batch(source)
    payload, _, get = transport
    payload["artifacts"][1][corrupt] = 2 if corrupt == "bytes" else "0" * 64
    storage = Mock()
    with pytest.raises(Cosmos3RayServeError, match="integrity mismatch"):
        _submit(source, "s3://test-bucket/result/", storage_client=storage)
    assert get.call_count == 2  # Reach the second download, after one valid file.
    storage.upload_directory.assert_not_called()


def test_valid_s3_publication_contains_all_verified_files(
    tmp_path: Path, transport
) -> None:
    source = tmp_path / "batch.json"
    _batch(source)
    storage = Mock()

    def upload(root, destination):
        files = [p for p in Path(root).rglob("*") if p.is_file()]
        assert len(files) == 5
        assert (
            json.loads((Path(root) / "provenance.json").read_text())["status"]
            == "completed"
        )
        return destination

    storage.upload_directory.side_effect = upload
    assert (
        _submit(source, "s3://test-bucket/result/", storage_client=storage)["status"]
        == "completed"
    )
    storage.upload_directory.assert_called_once()


@pytest.mark.parametrize(
    "sample",
    [
        {"name": " padded "},
        {"name": 123},
        {"name": "one", "num_outputs": 2},
        {"name": "one", "num_outputs": "2"},
    ],
)
def test_unsupported_request_fails_before_inference(
    tmp_path: Path, transport, sample
) -> None:
    source = tmp_path / "batch.json"
    _batch(source, samples=[sample])
    _, post, get = transport
    with pytest.raises(Cosmos3RayServeError):
        _submit(source, str(tmp_path / "out"))
    post.assert_not_called()
    get.assert_not_called()


@pytest.mark.parametrize(
    ("sample", "returned_mode", "frames"),
    [
        ({"name": "one"}, "text2video", 9),
        (
            {
                "name": "one",
                "vision_path": "https://example.org/input.JPG",
                "num_frames": 1,
            },
            "image2image",
            1,
        ),
        (
            {"name": "one", "vision_path": "https://example.org/input.png"},
            "image2video",
            9,
        ),
        (
            {"name": "one", "vision_path": "s3://test-bucket/input.mp4"},
            "video2video",
            9,
        ),
        (
            {"name": "one", "vision_path": "s3://test-bucket/input%2Epng"},
            "image2video",
            9,
        ),
        (
            {"name": "one", "vision_path": "s3://test-bucket/input.%6dp4"},
            "video2video",
            9,
        ),
        (
            {"name": "one", "vision_path": "s3://test-bucket/input.%70ng",
             "num_frames": 1},
            "image2image",
            1,
        ),
    ],
)
def test_implicit_native_mode_is_bound_before_sample_defaults(
    tmp_path: Path, transport, sample, returned_mode, frames
):
    source = tmp_path / "batch.json"
    _batch(source, samples=[sample])
    payload, post, get = transport
    payload["batch_size"] = 1
    payload["outputs"] = payload["outputs"][:1]
    args = payload["outputs"][0]["args"]
    args.update(model_mode=returned_mode, num_frames=frames)
    primary = "vision.jpg" if frames == 1 else "vision.mp4"
    path = f"request-safe/one/{primary}"
    payload["outputs"][0]["outputs"][0]["files"] = [path]
    payload["artifacts"] = [_artifact(path, "one")]
    assert _submit(source, str(tmp_path / "valid"))["status"] == "completed"
    assert post.call_args.kwargs["json"]["samples"] == [sample]
    get.reset_mock()
    args["model_mode"] = "reasoner"
    payload["outputs"][0]["outputs"][0]["files"] = [
        "request-safe/one/reasoner_text.txt"
    ]
    payload["artifacts"] = [_artifact("request-safe/one/reasoner_text.txt", "one")]
    with pytest.raises(Cosmos3RayServeError, match="different model_mode"):
        _submit(source, str(tmp_path / "invalid"))
    get.assert_not_called()


@pytest.mark.parametrize("key", ["input%2Epng", "input.%70ng", "input.JPG", "input.mp4"])
def test_implicit_mode_matches_authorized_server_staging(tmp_path: Path, key: str):
    from npa.workbench.cosmos.ray_inputs import stage_sample_inputs
    from npa.workbench.cosmos.ray_serve import _requested_mode
    from npa.workbench.storage_scope import StorageScope

    sample = {"name": "one", "vision_path": "s3://test-bucket/inputs/" + key}
    storage = Mock()
    storage.download_file.side_effect = lambda uri, path: Path(path).write_bytes(MEDIA)
    staged = stage_sample_inputs(
        sample, tmp_path / "staged",
        scope=StorageScope.from_config(s3_roots=["s3://test-bucket/inputs/"]),
        storage_client=storage,
    )
    assert _requested_mode(sample) == _requested_mode(staged)
    storage.download_file.assert_called_once()


@pytest.mark.parametrize("key", [
    "input%252Epng", "input.png?query=1", "input.png#fragment",
    "%2E%2E/input.png", "input%5C.png", "input.gif",
])
def test_unbindable_implicit_s3_mode_fails_before_inference(tmp_path: Path, transport, key):
    source = tmp_path / "batch.json"
    _batch(source, samples=[{"name": "one", "vision_path": "s3://test-bucket/" + key}])
    _, post, get = transport
    with pytest.raises(Cosmos3RayServeError):
        _submit(source, str(tmp_path / "out"))
    post.assert_not_called()
    get.assert_not_called()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("requested", "returned"), [(9, 1), (1, 9)])
def test_response_cannot_swap_requested_image_and_video(
    tmp_path: Path, transport, requested, returned
):
    source = tmp_path / "batch.json"
    _batch(
        source,
        samples=[{"name": "one", "model_mode": "text2video", "num_frames": requested}],
    )
    payload, _, get = transport
    payload["batch_size"] = 1
    payload["outputs"] = payload["outputs"][:1]
    payload["outputs"][0]["args"].update(model_mode="text2video", num_frames=returned)
    extension = "jpg" if returned == 1 else "mp4"
    path = f"request-safe/one/vision.{extension}"
    payload["outputs"][0]["outputs"][0]["files"] = [path]
    payload["artifacts"] = [_artifact(path, "one")]
    with pytest.raises(Cosmos3RayServeError, match="different frame category"):
        _submit(source, str(tmp_path / "out"))
    get.assert_not_called()


@pytest.mark.parametrize("value", ["1", "1.0", " 1 ", 1.0, True])
def test_native_numeric_coercion_is_applied_before_identity_binding(
    tmp_path: Path, transport, value
):
    source = tmp_path / "batch.json"
    _batch(
        source,
        samples=[
            {"name": "one", "seed": "17", "num_frames": value, "num_outputs": value}
        ],
    )
    payload, post, _ = transport
    payload["batch_size"] = 1
    payload["outputs"] = payload["outputs"][:1]
    payload["artifacts"] = payload["artifacts"][:1]
    result = _submit(source, str(tmp_path / "out"))
    assert result["status"] == "completed"
    sent = post.call_args.kwargs["json"]["samples"][0]
    assert sent == {"name": "one", "seed": 17, "num_frames": 1, "num_outputs": 1}
    assert type(sent["num_frames"]) is int


@pytest.mark.parametrize("destination", ["local", "s3"])
@pytest.mark.parametrize("override", [{}, {"num_frames": None}])
@pytest.mark.parametrize(
    ("sample", "frames"),
    [
        ({"model_mode": "text2image"}, 1),
        ({"model_mode": "image2image"}, 1),
        ({"model_mode": "text2video"}, 189),
        ({"model_mode": "image2video"}, 189),
        ({"model_mode": "video2video"}, 189),
        ({"model_mode": "audio_image2video"}, 189),
        ({"model_mode": "forward_dynamics"}, 189),
        ({"model_mode": "inverse_dynamics"}, 189),
        ({"model_mode": "wam"}, 189),
        ({"model_mode": "text2image", "wsm": {}, "edge": None}, 101),
        ({"model_mode": "text2image", "wsm": {}, "edge": {}}, 1),
        ({"model_mode": "text2image", "wsm": None}, 1),
    ],
)
def test_native_default_frame_category_is_bound_before_publication(
    tmp_path: Path, transport, destination, override, sample, frames
):
    source = tmp_path / "batch.json"
    _batch(source, samples=[{"name": "one", **sample, **override}])
    payload, _, get = transport
    payload["batch_size"] = 1
    payload["outputs"] = payload["outputs"][:1]
    args = payload["outputs"][0]["args"]
    args.update(**sample, num_frames=frames)

    def set_files(frame_count):
        extension = "jpg" if frame_count == 1 else "mp4"
        path = f"request-safe/one/vision.{extension}"
        files = [path] + [
            f"request-safe/one/control_{hint}.{extension}"
            for hint in ("edge", "blur", "depth", "seg", "wsm")
            if sample.get(hint) is not None
        ]
        payload["outputs"][0]["outputs"][0]["files"] = files
        payload["artifacts"] = [_artifact(path, "one") for path in files]

    set_files(frames)
    assert _submit(source, str(tmp_path / "valid"))["status"] == "completed"
    get.reset_mock()
    args["num_frames"] = 9 if frames == 1 else 1
    set_files(args["num_frames"])
    storage = Mock()
    output = (
        str(tmp_path / "invalid")
        if destination == "local"
        else "s3://test-bucket/invalid/"
    )
    with pytest.raises(Cosmos3RayServeError, match="different frame category"):
        _submit(source, output, storage_client=storage)
    get.assert_not_called()
    storage.upload_directory.assert_not_called()
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("override", [{}, {"num_frames": None}, {"num_frames": 1}])
@pytest.mark.parametrize("mode", ["text2image", "reasoner"])
def test_custom_defaults_are_not_trusted_for_output_binding(
    tmp_path: Path, transport, override, mode
):
    source = tmp_path / "batch.json"
    _batch(
        source,
        samples=[
            {
                "name": "one",
                "model_mode": mode,
                "defaults_file": "custom.json",
                **override,
            }
        ],
    )
    _, post, get = transport
    with pytest.raises(Cosmos3RayServeError, match="defaults_file cannot be bound"):
        _submit(source, str(tmp_path / "out"))
    post.assert_not_called()
    get.assert_not_called()


@pytest.mark.parametrize("frames", [1, 9])
def test_explicit_frames_override_single_wsm(tmp_path: Path, transport, frames):
    source = tmp_path / "batch.json"
    _batch(
        source,
        samples=[
            {
                "name": "one",
                "model_mode": "text2image",
                "num_frames": frames,
                "wsm": {},
            }
        ],
    )
    payload, _, _ = transport
    payload["batch_size"] = 1
    payload["outputs"] = payload["outputs"][:1]
    payload["outputs"][0]["args"]["num_frames"] = frames
    payload["outputs"][0]["args"]["wsm"] = {}
    extension = "jpg" if frames == 1 else "mp4"
    path = f"request-safe/one/vision.{extension}"
    files = [path, f"request-safe/one/control_wsm.{extension}"]
    payload["outputs"][0]["outputs"][0]["files"] = files
    payload["artifacts"] = [_artifact(path, "one") for path in files]
    assert _submit(source, str(tmp_path / "out"))["status"] == "completed"


@pytest.mark.parametrize("field", ["seed", "num_frames", "num_outputs"])
def test_invalid_numeric_binding_fails_before_inference(
    tmp_path: Path, transport, field
):
    source = tmp_path / "batch.json"
    _batch(source, samples=[{"name": "one", field: "invalid"}])
    _, post, get = transport
    with pytest.raises(Cosmos3RayServeError, match="invalid numeric"):
        _submit(source, str(tmp_path / "out"))
    post.assert_not_called()
    get.assert_not_called()


@pytest.mark.parametrize("hint", ["edge", "blur", "depth", "seg", "wsm"])
@pytest.mark.parametrize("frames", [1, 9])
@pytest.mark.parametrize("destination", ["local", "s3"])
def test_requested_control_cannot_disappear_from_both_manifests(
    tmp_path: Path, transport, hint, frames, destination
):
    source = tmp_path / "batch.json"
    sample = {
        "name": "one",
        "model_mode": "video2video",
        "num_frames": frames,
        "vision_path": "https://example.org/input.mp4",
        hint: {},
        "show_control_condition": False,
    }
    _batch(source, samples=[sample])
    payload, _, get = transport
    payload["batch_size"] = 1
    payload["outputs"] = payload["outputs"][:1]
    payload["outputs"][0]["args"].update(sample)
    extension = "jpg" if frames == 1 else "mp4"
    files = [
        f"request-safe/one/{stem}.{extension}" for stem in ["vision", f"control_{hint}"]
    ]
    payload["outputs"][0]["outputs"][0]["files"] = files
    payload["artifacts"] = [_artifact(path, "one") for path in files]
    assert _submit(source, str(tmp_path / "valid"))["status"] == "completed"
    get.reset_mock()
    files.pop()
    payload["artifacts"].pop()
    storage = Mock()
    output = (
        str(tmp_path / "invalid")
        if destination == "local"
        else "s3://test-bucket/invalid/"
    )
    with pytest.raises(Cosmos3RayServeError, match="missing a requested control"):
        _submit(source, output, storage_client=storage)
    get.assert_not_called()
    storage.upload_directory.assert_not_called()
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    "returned_hints", [{}, {"depth": {}}, {"edge": {}, "depth": {}}]
)
def test_response_cannot_change_the_requested_transfer_hint_set(
    tmp_path: Path, transport, returned_hints
):
    source = tmp_path / "batch.json"
    _batch(source, samples=[{"name": "one", "model_mode": "text2image", "edge": {}}])
    payload, _, get = transport
    payload["batch_size"] = 1
    payload["outputs"] = payload["outputs"][:1]
    payload["outputs"][0]["args"].update(returned_hints)
    payload["artifacts"] = payload["artifacts"][:1]
    with pytest.raises(Cosmos3RayServeError, match="different transfer hints"):
        _submit(source, str(tmp_path / "out"))
    get.assert_not_called()


def test_transfer_dispatch_precedes_reasoner_file_contract(tmp_path: Path, transport):
    source = tmp_path / "batch.json"
    _batch(source, samples=[{"name": "one", "model_mode": "reasoner", "edge": {}}])
    payload, _, _ = transport
    payload["batch_size"] = 1
    payload["outputs"] = payload["outputs"][:1]
    payload["outputs"][0]["args"].update(model_mode="reasoner", edge={})
    files = ["request-safe/one/vision.jpg", "request-safe/one/control_edge.jpg"]
    payload["outputs"][0]["outputs"][0]["files"] = files
    payload["artifacts"] = [_artifact(path, "one") for path in files]
    assert _submit(source, str(tmp_path / "out"))["status"] == "completed"


@pytest.mark.parametrize("encoded", ["%3F", "%23", "%09", "%0D", "%0A"])
@pytest.mark.parametrize("mode", [None, "image2image"])
@pytest.mark.parametrize("field", ["vision_path", "prompt_path", "edge"])
def test_reinterpreted_conditioning_key_fails_before_post(tmp_path, transport, encoded, mode, field):
    source = tmp_path / "batch.json"
    value = f"s3://test-bucket/media/input{encoded}variant.png"
    sample = {"name": "one", field: {"control_path": value} if field == "edge" else value}
    if mode is not None:
        sample["model_mode"] = mode
    _batch(source, samples=[sample])
    _, post, get = transport
    with pytest.raises(Cosmos3RayServeError, match="conditioning S3 URI"):
        _submit(source, str(tmp_path / "out"))
    post.assert_not_called()
    get.assert_not_called()
    assert not (tmp_path / "out").exists()
