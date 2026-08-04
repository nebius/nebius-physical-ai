from __future__ import annotations

import json

import pytest

from npa.clients.storage_validation import probe_storage_write


class ProviderError(Exception):
    def __init__(self, code: str, status: int, message: str = "") -> None:
        super().__init__(message)
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class FakeS3:
    def __init__(
        self,
        *,
        put_error: Exception | None = None,
        delete_error: Exception | None = None,
    ):
        self.put_error = put_error
        self.delete_error = delete_error
        self.puts: list[dict] = []
        self.deletes: list[dict] = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        if self.put_error:
            raise self.put_error

    def delete_object(self, **kwargs):
        self.deletes.append(kwargs)
        if self.delete_error:
            raise self.delete_error


class EndpointConnectionError(Exception):
    pass


def test_probe_requires_each_storage_prerequisite_without_using_a_client() -> None:
    result = probe_storage_write(
        bucket="", endpoint_url="", access_key_id="", secret_access_key=""
    )

    assert result.ok is False
    assert result.code == "missing_configuration"
    assert (
        "bucket, endpoint, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY" in result.summary
    )


def test_probe_writes_and_cleans_one_isolated_object() -> None:
    client = FakeS3()

    result = probe_storage_write(
        bucket="s3://bucket/checkpoints/",
        endpoint_url="https://storage.example",
        access_key_id="access",
        secret_access_key="secret",
        prefix="checks",
        client=client,
    )

    assert result.ok and result.cleanup_succeeded
    assert len(client.puts) == len(client.deletes) == 1
    assert client.puts[0]["Bucket"] == "bucket"
    assert client.puts[0]["Key"] == client.deletes[0]["Key"] == result.probe_key
    assert result.probe_key.startswith("checks/.npa-probes/write-")


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ProviderError("AccessDenied", 403), "forbidden"),
        (ProviderError("NoSuchBucket", 404), "bucket_unreachable"),
    ],
)
def test_probe_classifies_write_failures_without_leaking_provider_text(
    error, code
) -> None:
    canary = "NPA_CANARY_STORAGE_SECRET_DO_NOT_DISCLOSE"
    error.args = (canary,)
    client = FakeS3(put_error=error)

    result = probe_storage_write(
        bucket="bucket",
        endpoint_url="https://storage.example",
        access_key_id=canary,
        secret_access_key=canary,
        client=client,
    )

    assert result.ok is False
    assert result.code == code
    assert canary not in json.dumps(result.__dict__)
    assert client.deletes == []


def test_probe_reports_an_unreachable_endpoint_without_echoing_the_exception() -> None:
    canary = "NPA_CANARY_ENDPOINT_SECRET_DO_NOT_DISCLOSE"
    result = probe_storage_write(
        bucket="bucket",
        endpoint_url="https://unreachable.example",
        access_key_id=canary,
        secret_access_key=canary,
        client=FakeS3(put_error=EndpointConnectionError(canary)),
    )

    assert result.code == "endpoint_unreachable"
    assert canary not in result.summary


def test_probe_cleanup_failure_is_not_a_false_green() -> None:
    client = FakeS3(delete_error=ProviderError("AccessDenied", 403))

    result = probe_storage_write(
        bucket="bucket",
        endpoint_url="https://storage.example",
        access_key_id="access",
        secret_access_key="secret",
        client=client,
    )

    assert result.ok is False
    assert result.code == "cleanup_failed"
    assert result.cleanup_attempted and not result.cleanup_succeeded
    assert f"s3://bucket/{result.probe_key}" in result.summary
