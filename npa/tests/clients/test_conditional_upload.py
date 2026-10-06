"""Exercise large conditional publication and cleanup after concurrent writes."""

import io
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from npa.clients import conditional_upload
from npa.clients.storage import StorageClient, StoragePreconditionFailed


@pytest.fixture
def provider(monkeypatch):
    """Use small parts to exercise multipart behavior without large fixtures."""
    monkeypatch.setattr(conditional_upload, "_MULTIPART_THRESHOLD", 4)
    monkeypatch.setattr(conditional_upload, "_PART_SIZE", 3)
    client = Mock()
    client.create_multipart_upload.return_value = {"UploadId": "upload"}
    client.upload_part.side_effect = lambda **kwargs: {
        "ETag": f'"part-{kwargs["PartNumber"]}"'
    }
    client.complete_multipart_upload.return_value = {"ETag": '"completed"'}
    return client


@pytest.mark.parametrize("condition", [{"IfNoneMatch": "*"}, {"IfMatch": '"old"'}])
def test_large_stream_keeps_condition_at_atomic_completion(provider, condition):
    """Preserve stream contents, metadata, part order and the completion guard."""
    body = io.BytesIO(b"skip-payload")
    body.seek(5)
    response = conditional_upload.put_object_conditional(
        provider,
        {
            "Bucket": "bucket",
            "Key": "recording.rrd",
            "Body": body,
            "ContentType": "application/vnd.rerun.rrd",
            "Metadata": {"npa-sha256": "digest"},
            **condition,
        },
    )
    assert response["ETag"] == '"completed"'
    parts = provider.upload_part.call_args_list
    assert b"".join(call.kwargs["Body"] for call in parts) == b"payload"
    assert [call.kwargs["PartNumber"] for call in parts] == [1, 2, 3]
    provider.create_multipart_upload.assert_called_once_with(
        Bucket="bucket",
        Key="recording.rrd",
        ContentType="application/vnd.rerun.rrd",
        Metadata={"npa-sha256": "digest"},
    )
    assert provider.complete_multipart_upload.call_args.kwargs == {
        "Bucket": "bucket",
        "Key": "recording.rrd",
        "UploadId": "upload",
        "MultipartUpload": {
            "Parts": [
                {"PartNumber": number, "ETag": f'"part-{number}"'}
                for number in (1, 2, 3)
            ]
        },
        **condition,
    }
    provider.put_object.assert_not_called()
    provider.abort_multipart_upload.assert_not_called()


@pytest.mark.parametrize("status", [409, 412])
def test_late_writer_conflict_aborts_parts_and_preserves_typed_failure(
    provider, status
):
    """Reject a concurrent publisher at completion and discard unfinished parts."""
    failure = ClientError(
        {
            "Error": {"Code": "ConditionalRequestConflict"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "CompleteMultipartUpload",
    )
    provider.complete_multipart_upload.side_effect = failure
    client = StorageClient.__new__(StorageClient)
    client._s3 = provider
    with pytest.raises(StoragePreconditionFailed) as caught:
        client.put_bytes_conditional(
            b"payload", "s3://bucket/recording.rrd", if_none_match=True
        )
    assert caught.value.__cause__ is failure
    provider.abort_multipart_upload.assert_called_once_with(
        Bucket="bucket",
        Key="recording.rrd",
        UploadId="upload",
    )


def test_part_failure_aborts_without_masking_original_failure(provider):
    """Keep the upload failure visible even when cleanup also fails."""
    failure = ClientError({"Error": {"Code": "InternalError"}}, "UploadPart")
    provider.upload_part.side_effect = failure
    provider.abort_multipart_upload.side_effect = ClientError(
        {"Error": {"Code": "NoSuchUpload"}},
        "AbortMultipartUpload",
    )
    with pytest.raises(ClientError) as caught:
        conditional_upload.put_object_conditional(
            provider,
            {
                "Bucket": "bucket",
                "Key": "recording.rrd",
                "Body": b"payload",
                "IfNoneMatch": "*",
            },
        )
    assert caught.value is failure
    provider.complete_multipart_upload.assert_not_called()
    provider.abort_multipart_upload.assert_called_once()


def test_small_conditional_payload_uses_original_request(provider):
    """Keep small journal writes on the existing conditional PutObject path."""
    request = {
        "Bucket": "bucket",
        "Key": "report.json",
        "Body": b"{}",
        "IfNoneMatch": "*",
    }
    conditional_upload.put_object_conditional(provider, request)
    provider.put_object.assert_called_once_with(**request)
    provider.create_multipart_upload.assert_not_called()


@pytest.mark.parametrize("conditions", [{}, {"IfMatch": '"old"', "IfNoneMatch": "*"}])
def test_unguarded_or_ambiguous_upload_is_rejected_before_provider_call(
    provider, conditions
):
    """Reject missing or ambiguous guards before starting an upload."""
    with pytest.raises(ValueError, match="exactly one"):
        conditional_upload.put_object_conditional(
            provider,
            {
                "Bucket": "bucket",
                "Key": "recording.rrd",
                "Body": b"payload",
                **conditions,
            },
        )
    assert not provider.mock_calls
