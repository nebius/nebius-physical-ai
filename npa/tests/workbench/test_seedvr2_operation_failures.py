"""Exercise real storage operations through the service, CLI, and SDK boundaries."""

from functools import partial
import io
import json

from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    IncompleteReadError,
)
from botocore.response import StreamingBody
from fastapi.testclient import TestClient
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.storage import StorageClient
from npa.sdk.workbench import seedvr2 as sdk
from npa.workbench.seedvr2 import artifacts, runtime
from npa.workbench.seedvr2.service import create_app


class _Provider:
    def __init__(self, failure, boundary):
        self.failure = failure
        self.boundary = boundary
        self.bodies = []
        self.writes = []
        self.failed = False

    def _fail(self):
        self.failed = True
        if self.failure == "endpoint":
            raise EndpointConnectionError(endpoint_url="https://private-canary.invalid")
        if self.failure == "denied":
            raise ClientError(
                {
                    "Error": {"Code": "AccessDenied", "Message": "private-canary"},
                    "ResponseMetadata": {"HTTPStatusCode": 403},
                },
                "GetObject",
            )
        return self._body(b"private-canary", truncated=True)

    def _body(self, data, *, truncated=False):
        raw = io.BytesIO(data)
        self.bodies.append(raw)
        size = len(data) + int(truncated)
        return {"Body": StreamingBody(raw, size), "ContentLength": size, "ETag": "etag"}

    def get_object(self, *, Bucket, Key):
        if self.boundary == "input" or (self.boundary == "readback" and self.writes):
            return self._fail()
        if Key == "input.mp4":
            return self._body(b"fixture-media")
        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    def put_object(self, **kwargs):
        assert kwargs["IfNoneMatch"] == "*"
        if self.boundary == "write":
            return self._fail()
        self.writes.append(kwargs)
        return {"ETag": "etag"}


def _wire_storage(monkeypatch, tmp_path, verb, failure, boundary="input"):
    provider = _Provider(failure, boundary)
    storage = object.__new__(StorageClient)
    storage._s3 = provider
    module = runtime if verb == "restore" else artifacts
    # Bind only the factory; the production operation and its wrappers still run.
    monkeypatch.setattr(
        module, verb, partial(getattr(module, verb), storage_factory=lambda: storage)
    )
    monkeypatch.setenv("NPA_SEEDVR2_WORK_DIR", str(tmp_path / "runs"))
    payload = {
        "input_path": "s3://example-bucket/input."
        + ("mp4" if verb in {"probe", "restore"} else "json"),
        "output_path": "s3://example-bucket/output."
        + ("json" if verb in {"probe", "verify"} else "prefix"),
        "run_id": "operation-failure",
    }
    if verb == "restore":
        payload["probe_path"] = "s3://example-bucket/probe.json"
    return provider, payload


def _assert_failed(provider, failure):
    assert provider.failed
    assert all(body.closed for body in provider.bodies)
    return 400 if failure == "denied" else 503


@pytest.mark.parametrize("verb", ["probe", "restore", "verify", "review"])
@pytest.mark.parametrize("failure", ["endpoint", "truncated", "denied"])
def test_actual_input_download_service_failure(monkeypatch, tmp_path, verb, failure):
    provider, payload = _wire_storage(monkeypatch, tmp_path, verb, failure)
    headers = {"Authorization": "Bearer fixture-token"}
    service = create_app(
        token="fixture-token", allowed_s3_roots=["s3://example-bucket/"]
    )
    with TestClient(service) as client:
        response = client.post("/" + verb, headers=headers, json=payload)
        assert response.status_code == _assert_failed(provider, failure)
        assert "private-canary" not in response.text
        assert client.get("/status", headers=headers).json() == {"busy": False}
    assert not provider.writes
    if verb == "restore":
        retained = list((tmp_path / "runs").rglob("failure.json"))
        assert len(retained) == 1
        expected = (
            "SeedVR2Error" if failure == "denied" else "SeedVR2StorageUnavailable"
        )
        assert json.loads(retained[0].read_text())["error"] == expected


@pytest.mark.parametrize("verb", ["probe", "restore", "verify", "review"])
@pytest.mark.parametrize("failure", ["endpoint", "truncated", "denied"])
def test_actual_input_download_cli_failure(monkeypatch, tmp_path, verb, failure):
    provider, payload = _wire_storage(monkeypatch, tmp_path, verb, failure)
    argv = ["workbench", "seedvr2", verb]
    for key, value in payload.items():
        argv.extend(["--" + key.replace("_", "-"), value])
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 1
    assert "SeedVR2 failed: object storage" in result.stderr
    assert "private-canary" not in result.output
    assert json.loads(result.stdout)["result"] == "error"
    assert ("temporarily unavailable" in result.stderr) == (failure != "denied")
    _assert_failed(provider, failure)
    assert not provider.writes


@pytest.mark.parametrize("verb", ["probe", "restore", "verify", "review"])
@pytest.mark.parametrize("failure", ["endpoint", "truncated", "denied"])
def test_actual_input_download_sdk_preserves_class_and_cause(
    monkeypatch, tmp_path, verb, failure
):
    provider, payload = _wire_storage(monkeypatch, tmp_path, verb, failure)
    with pytest.raises(runtime.SeedVR2Error) as caught:
        getattr(sdk, verb)(**payload)
    assert isinstance(caught.value, runtime.SeedVR2StorageUnavailable) == (
        failure != "denied"
    )
    expected_cause = {
        "endpoint": EndpointConnectionError,
        "truncated": IncompleteReadError,
        "denied": ClientError,
    }[failure]
    assert isinstance(caught.value.__cause__, expected_cause)
    assert "private-canary" not in str(caught.value)
    _assert_failed(provider, failure)


@pytest.mark.parametrize(
    "boundary,failure",
    [
        ("write", "endpoint"),
        ("write", "denied"),
        ("readback", "endpoint"),
        ("readback", "truncated"),
        ("readback", "denied"),
    ],
)
def test_actual_probe_publication_service_failure(
    monkeypatch, tmp_path, boundary, failure
):
    provider, payload = _wire_storage(monkeypatch, tmp_path, "probe", failure, boundary)
    # Only media decoding is synthetic. Input/absence/conditional PUT/readback use
    # the real StorageClient and real botocore streaming bodies.
    monkeypatch.setattr(artifacts, "_probe_video", lambda path: {"frames": 2})
    monkeypatch.setattr(artifacts, "_decoded_frame_hashes", lambda path: ["one", "two"])
    service = create_app(
        token="fixture-token", allowed_s3_roots=["s3://example-bucket/"]
    )
    headers = {"Authorization": "Bearer fixture-token"}
    with TestClient(service) as client:
        response = client.post("/probe", headers=headers, json=payload)
        assert response.status_code == _assert_failed(provider, failure)
        assert "private-canary" not in response.text
        assert client.get("/status", headers=headers).json() == {"busy": False}
    assert len(provider.writes) == int(boundary == "readback")
