"""Exercise real Botocore streaming-body behavior and complete input hashing."""

import hashlib
import io
from unittest.mock import Mock

from botocore.response import StreamingBody
import pytest

from npa.solutions.media_input import download_input


@pytest.mark.parametrize("matches", [True, False])
def test_download_hashes_complete_stream_and_closes_body(
    monkeypatch, tmp_path, matches
):
    payload = b"first block" * 110000 + b"distinct final bytes"
    stream = io.BytesIO(payload)
    client = Mock()
    client.get_object.return_value = {"Body": StreamingBody(stream, len(payload))}
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: client)
    target = tmp_path / "input.mp4"
    checksum = hashlib.sha256(payload if matches else payload[:-1]).hexdigest()
    if matches:
        assert (
            download_input("s3://example-bucket/clip.mp4", checksum, target) == target
        )
        assert target.read_bytes() == payload
    else:
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            download_input("s3://example-bucket/clip.mp4", checksum, target)
        assert not target.exists()
    assert stream.closed
    client.get_object.assert_called_once_with(Bucket="example-bucket", Key="clip.mp4")
