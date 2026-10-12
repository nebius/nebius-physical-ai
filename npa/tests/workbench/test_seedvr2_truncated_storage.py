"""Check real SDK short-body failures through SeedVR2 HTTP storage boundaries."""

import io

from botocore.exceptions import ClientError, IncompleteReadError
from botocore.response import StreamingBody
from fastapi.testclient import TestClient
import pytest

from npa.clients.storage import StorageClient
from npa.workbench.seedvr2 import artifacts, runtime
from npa.workbench.seedvr2.service import create_app


class _StreamingObjectProvider:
    def __init__(self, response_kind):
        self.response_kind = response_kind
        self.raw_stream = io.BytesIO(b"private-body-canary")
        self.reads = 0

    def get_object(self, **kwargs):
        self.reads += 1
        if self.response_kind == "absent":
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        size = len(self.raw_stream.getvalue())
        if self.response_kind == "truncated":
            size += 1
        return {
            "Body": StreamingBody(self.raw_stream, size),
            "ETag": "" if self.response_kind == "missing-etag" else "fixture-etag",
        }

    def put_object(self, **kwargs):
        pytest.fail("absence controls must not write objects")


def _absence_operation(provider, failures):
    storage = object.__new__(StorageClient)
    storage._s3 = provider

    def check(request):
        try:
            runtime._ensure_artifacts_absent(storage, [request.output_path])
        except runtime.SeedVR2Error as error:
            failures.append(error)
            raise
        return {"status": "absence-control-only"}

    return check


@pytest.mark.parametrize("verb", ["probe", "restore", "verify", "review"])
@pytest.mark.parametrize(
    "response_kind, expected_status",
    [("truncated", 503), ("complete", 400), ("missing-etag", 400), ("absent", 200)],
)
def test_streaming_body_classification_closes_body_and_releases_http_lock(
    monkeypatch, verb, response_kind, expected_status
):
    provider = _StreamingObjectProvider(response_kind)
    failures = []
    module = runtime if verb == "restore" else artifacts
    monkeypatch.setattr(module, verb, _absence_operation(provider, failures))
    app = create_app(token="fixture-token", allowed_s3_roots=["s3://example-bucket/"])
    headers = {"Authorization": "Bearer fixture-token"}
    payload = {
        "input_path": "s3://example-bucket/input.mp4",
        "output_path": "s3://example-bucket/output.json",
        "run_id": "streaming-body-control",
    }
    with TestClient(app) as client:
        response = client.post(f"/{verb}", headers=headers, json=payload)
        assert client.get("/status", headers=headers).json() == {"busy": False}
        assert "private-body-canary" not in response.text
        assert provider.reads == 1
        if response_kind != "absent":
            assert provider.raw_stream.closed
        monkeypatch.setattr(module, verb, lambda request: {"status": "next-control"})
        assert client.post(f"/{verb}", headers=headers, json=payload).status_code == 200
    assert response.status_code == expected_status
    if response_kind == "truncated":
        assert isinstance(failures[0], runtime.SeedVR2StorageUnavailable)
        assert isinstance(failures[0].__cause__, IncompleteReadError)
