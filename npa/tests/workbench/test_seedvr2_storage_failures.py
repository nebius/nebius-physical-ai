"""Distinguish transient storage outages from permanent SeedVR2 request failures."""

from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    EndpointConnectionError,
    NoCredentialsError,
    ParamValidationError,
    ReadTimeoutError,
)
from fastapi.testclient import TestClient
import pytest

from npa.clients.storage import StorageError
from npa.workbench.seedvr2 import artifacts, runtime
from npa.workbench.seedvr2.service import create_app


class _FailedStorage:
    def __init__(self, error):
        self.error = error
        self.reads = 0

    def read_bytes_with_etag(self, uri):
        self.reads += 1
        raise self.error


def _client_error(code, status=400):
    return ClientError(
        {
            "Error": {"Code": code, "Message": "private-provider-detail"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "GetObject",
    )


@pytest.mark.parametrize(
    "code",
    [
        "RequestTimeout",
        "RequestTimeoutException",
        "PriorRequestNotComplete",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "ThrottledException",
        "RequestThrottled",
        "RequestThrottledException",
        "TooManyRequestsException",
        "ServiceUnavailable",
        "InternalError",
        "InternalFailure",
    ],
)
def test_retryable_provider_codes_keep_cause_and_stop_publication(code):
    original = _client_error(code)
    storage = _FailedStorage(original)
    with pytest.raises(runtime.SeedVR2StorageUnavailable) as error:
        runtime._ensure_artifacts_absent(storage, ["first", "second"])
    assert error.value.__cause__ is original
    assert "private-provider-detail" not in str(error.value)
    assert storage.reads == 1


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_retryable_provider_status_without_known_code(status):
    with pytest.raises(runtime.SeedVR2StorageUnavailable):
        runtime._ensure_artifacts_absent(
            _FailedStorage(_client_error("UnknownProviderError", status)), ["output"]
        )


@pytest.mark.parametrize(
    "original",
    [
        EndpointConnectionError(endpoint_url="https://example.invalid"),
        ConnectionClosedError(endpoint_url="https://example.invalid"),
        ReadTimeoutError(endpoint_url="https://example.invalid"),
    ],
)
def test_retryable_transport_failure_keeps_private_cause(original):
    with pytest.raises(runtime.SeedVR2StorageUnavailable) as error:
        runtime._ensure_artifacts_absent(_FailedStorage(original), ["output"])
    assert error.value.__cause__ is original
    assert "example.invalid" not in str(error.value)


@pytest.mark.parametrize(
    "original",
    [
        _client_error("AccessDenied", 403),
        _client_error("InvalidArgument", 400),
        _client_error("NoSuchBucket", 404),
        NoCredentialsError(),
        ParamValidationError(report="private-provider-detail"),
        StorageError("private-provider-detail"),
    ],
)
def test_permanent_storage_failure_is_not_retryable(original):
    with pytest.raises(runtime.SeedVR2Error) as error:
        runtime._ensure_artifacts_absent(_FailedStorage(original), ["output"])
    assert not isinstance(error.value, runtime.SeedVR2StorageUnavailable)
    assert error.value.__cause__ is original


@pytest.mark.parametrize("verb", ["probe", "restore", "verify", "review"])
@pytest.mark.parametrize("code, status", [("AccessDenied", 400), ("SlowDown", 503)])
def test_http_storage_classification_releases_lock_and_withholds_provider_text(
    monkeypatch, verb, code, status
):
    def failed(request):
        runtime._ensure_artifacts_absent(
            _FailedStorage(_client_error(code)), [request.output_path]
        )
        pytest.fail("failed absence check must not reach publication")

    module = runtime if verb == "restore" else artifacts
    monkeypatch.setattr(module, verb, failed)
    client = TestClient(
        create_app(token="test-token", allowed_s3_roots=["s3://example-bucket/"])
    )
    headers = {"Authorization": "Bearer test-token"}
    response = client.post(
        f"/{verb}",
        headers=headers,
        json={
            "input_path": "s3://example-bucket/input.mp4",
            "output_path": "s3://example-bucket/output/",
            "run_id": "storage-failure",
        },
    )
    assert response.status_code == status
    assert "private-provider-detail" not in response.text
    assert client.get("/status", headers=headers).json() == {"busy": False}
