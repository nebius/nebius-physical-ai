"""The actual boto3 parser preserves error statuses and successful checksums."""

import base64
import io
import zlib

import boto3
from botocore.awsrequest import AWSResponse
from botocore.config import Config
from botocore.exceptions import ClientError, FlexibleChecksumError
import pytest

from npa.clients.storage import register_s3_error_body_compat
from npa.clients.storage import StorageClient
from npa.agent_backend import canonical_mcap
from npa.agent_backend.publication_reader import PublicationConflict


class _RawResponse(io.BytesIO):
    def stream(self, amt=None, decode_content=False):
        while block := self.read(amt or 1024):
            yield block


def _client(status, body, checksum=True):
    client = boto3.client(
        "s3",
        region_name="test",
        endpoint_url="https://storage.example.test",
        aws_access_key_id="test-access",
        aws_secret_access_key="test-secret",
        config=Config(retries={"total_max_attempts": 1}),
    )
    register_s3_error_body_compat(client)

    def respond(**_kwargs):
        headers = {"content-length": str(len(body))}
        if checksum:
            headers["x-amz-checksum-crc32"] = base64.b64encode(
                zlib.crc32(b"expected-success").to_bytes(4, "big")
            ).decode()
        return AWSResponse(
            "https://storage.example.test/object", status, headers, _RawResponse(body)
        )

    client.meta.events.register("before-send.s3.GetObject", respond)
    return client


@pytest.mark.parametrize("status", [403, 404, 409, 412, 503])
def test_actual_error_parser_preserves_http_status_with_object_checksum(status):
    client = _client(status, b"<Error><Code>PreconditionFailed</Code></Error>")
    with pytest.raises(ClientError) as rejected:
        client.get_object(Bucket="demo-bucket", Key="object", IfMatch='"old"')
    assert rejected.value.response["ResponseMetadata"]["HTTPStatusCode"] == status
    assert rejected.value.response["Error"]["Code"] == str(status)


def test_successful_payload_checksum_stays_enabled():
    client = _client(200, b"expected-success")
    body = client.get_object(Bucket="demo-bucket", Key="object")["Body"]
    try:
        assert body.read() == b"expected-success"
    finally:
        body.close()


def test_successful_corrupt_payload_checksum_is_not_suppressed():
    client = _client(200, b"corrupt-success")
    body = client.get_object(Bucket="demo-bucket", Key="object")["Body"]
    try:
        with pytest.raises(FlexibleChecksumError):
            body.read()
    finally:
        body.close()


def test_normal_error_xml_is_preserved_without_the_sdk_wrap():
    client = _client(
        403, b"<Error><Code>ActualDeniedCode</Code></Error>", checksum=False
    )
    with pytest.raises(ClientError) as rejected:
        client.get_object(Bucket="demo-bucket", Key="object")
    assert rejected.value.response["Error"]["Code"] == "ActualDeniedCode"


def test_wrapped_404_is_absent_at_actual_storage_reader():
    client = StorageClient.__new__(StorageClient)
    client._s3 = _client(404, b"<Error><Code>NoSuchKey</Code></Error>")
    assert client.read_bytes_with_etag("s3://demo-bucket/recording") is None


def test_wrapped_412_is_conflict_at_actual_canonical_consumer(monkeypatch, tmp_path):
    client = _client(412, b"<Error><Code>PreconditionFailed</Code></Error>")
    monkeypatch.setattr(
        client, "head_object", lambda **_kwargs: {"ContentLength": 3, "ETag": '"old"'}
    )
    destination = tmp_path / "recording.mcap"
    destination.write_bytes(b"previous-verified")
    with pytest.raises(PublicationConflict):
        canonical_mcap._download_s3_snapshot(
            client, bucket="demo-bucket", key="recording", destination=destination
        )
    assert destination.read_bytes() == b"previous-verified"
