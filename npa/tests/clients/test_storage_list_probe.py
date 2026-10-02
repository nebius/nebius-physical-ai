"""Check the S3 wire contract for constant-size credential probes."""

import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from npa.clients.storage import (
    StorageClient,
    StorageError,
    StoragePreconditionFailed,
)


@pytest.fixture
def storage():
    return StorageClient(
        endpoint_url="https://storage.invalid",
        aws_access_key_id="synthetic-access",
        aws_secret_access_key="synthetic-secret",
    )


@pytest.mark.parametrize("prefix", ["", "run", "run/"])
@pytest.mark.parametrize("truncated", [False, True])
def test_probe_checks_empty_or_populated_prefix_without_paginating(
    storage, prefix, truncated
):
    expected_prefix = prefix.rstrip("/") + "/" if prefix else ""
    response = {"IsTruncated": truncated, "KeyCount": int(truncated)}
    if truncated:
        response["NextContinuationToken"] = "must-not-be-followed"
    with Stubber(storage.s3) as stubber:
        stubber.add_response(
            "list_objects_v2",
            response,
            {
                "Bucket": "test-bucket",
                "Prefix": expected_prefix,
                "Delimiter": "/",
                "MaxKeys": 1,
            },
        )
        assert storage.probe_list_access(f"s3://test-bucket/{prefix}") is None
        stubber.assert_no_pending_responses()


@pytest.mark.parametrize(
    "code,status",
    [
        ("AccessDenied", 403),
        ("SignatureDoesNotMatch", 403),
        ("NoSuchBucket", 404),
        ("ServiceUnavailable", 503),
    ],
)
def test_probe_preserves_service_denial(storage, code, status):
    with Stubber(storage.s3) as stubber:
        stubber.add_client_error(
            "list_objects_v2", service_error_code=code, http_status_code=status
        )
        with pytest.raises(ClientError, match=code):
            storage.probe_list_access("s3://test-bucket")


@pytest.mark.parametrize(
    "uri", ["test-bucket/prefix", "https://storage.invalid/bucket"]
)
def test_probe_rejects_non_s3_destinations_before_request(storage, uri):
    with Stubber(storage.s3), pytest.raises(StorageError, match="Expected s3://"):
        storage.probe_list_access(uri)


def test_conditional_delete_forwards_the_observed_etag(storage):
    with Stubber(storage.s3) as stubber:
        stubber.add_response(
            "delete_object",
            {},
            {
                "Bucket": "test-bucket",
                "Key": "reports/sim2real.mcap",
                "IfMatch": '"observed-etag"',
            },
        )
        storage.delete_file_conditional(
            "s3://test-bucket/reports/sim2real.mcap",
            if_match='"observed-etag"',
        )
        stubber.assert_no_pending_responses()


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("PreconditionFailed", 412),
        ("ConditionalRequestConflict", 409),
    ],
)
def test_conditional_delete_types_superseded_writers(storage, code, status):
    with Stubber(storage.s3) as stubber:
        stubber.add_client_error(
            "delete_object",
            service_error_code=code,
            http_status_code=status,
            expected_params={
                "Bucket": "test-bucket",
                "Key": "reports/sim2real.mcap",
                "IfMatch": '"observed-etag"',
            },
        )
        with pytest.raises(StoragePreconditionFailed, match="superseded"):
            storage.delete_file_conditional(
                "s3://test-bucket/reports/sim2real.mcap",
                if_match='"observed-etag"',
            )


def test_conditional_delete_requires_an_exact_object_and_nonempty_etag(storage):
    with Stubber(storage.s3):
        with pytest.raises(StorageError, match="exact S3 object URI"):
            storage.delete_file_conditional(
                "s3://test-bucket/reports/",
                if_match='"observed-etag"',
            )
        with pytest.raises(ValueError, match="requires an ETag"):
            storage.delete_file_conditional(
                "s3://test-bucket/reports/sim2real.mcap",
                if_match="",
            )
