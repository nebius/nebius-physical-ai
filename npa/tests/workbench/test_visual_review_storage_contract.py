"""Exercise rich journal dependencies through the unchanged production S3 client."""

from io import BytesIO

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber
import pytest

from npa.clients.storage import StorageClient
from npa.workbench.vlm_eval import visual_review


@pytest.fixture
def storage():
    client = StorageClient.__new__(StorageClient)
    client._s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="synthetic",
        aws_secret_access_key="synthetic",
    )
    with Stubber(client._s3) as stub:
        yield client, stub
        stub.assert_no_pending_responses()


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
def test_real_client_missing_object_is_none(storage, code):
    client, stub = storage
    stub.add_client_error(
        "get_object",
        service_error_code=code,
        http_status_code=404,
        expected_params={"Bucket": "fixture-bucket", "Key": "review/report.json"},
    )
    assert (
        visual_review._read_s3_object(client, "s3://fixture-bucket/review/report.json")
        is None
    )


def test_real_client_read_returns_exact_bytes_and_closes_stream(storage):
    client, stub = storage
    stream = BytesIO(b'{"reserved":true}')
    stub.add_response(
        "get_object",
        {"Body": StreamingBody(stream, len(stream.getvalue())), "ETag": '"v1"'},
        {"Bucket": "fixture-bucket", "Key": "review/report.json"},
    )
    assert visual_review._read_s3_object(
        client, "s3://fixture-bucket/review/report.json"
    ) == (
        b'{"reserved":true}',
        '"v1"',
    )
    assert stream.closed


@pytest.mark.parametrize("conflict", [False, True])
def test_real_client_journal_create_uses_atomic_if_none_match(storage, conflict):
    client, stub = storage
    expected = {
        "Bucket": "fixture-bucket",
        "Key": "review/reservation.json",
        "Body": b"{}",
        "ContentType": "application/json",
        "IfNoneMatch": "*",
    }
    if conflict:
        stub.add_client_error(
            "put_object",
            service_error_code="PreconditionFailed",
            http_status_code=412,
            expected_params=expected,
        )
        with pytest.raises(
            visual_review.VlmVisualReviewError, match="already reserved"
        ):
            visual_review._conditional_object_create(
                b"{}",
                "s3://fixture-bucket/review/reservation.json",
                client,
                conflict_message="already reserved",
            )
    else:
        stub.add_response("put_object", {"ETag": '"v1"'}, expected)
        visual_review._conditional_object_create(
            b"{}",
            "s3://fixture-bucket/review/reservation.json",
            client,
            conflict_message="already reserved",
        )


def test_real_client_access_denial_is_not_object_absence(storage):
    client, stub = storage
    stub.add_client_error(
        "get_object",
        service_error_code="AccessDenied",
        http_status_code=403,
        expected_params={"Bucket": "fixture-bucket", "Key": "review/report.json"},
    )
    with pytest.raises(
        visual_review.VlmVisualReviewError, match="could not be inspected"
    ):
        visual_review._read_s3_object(client, "s3://fixture-bucket/review/report.json")
