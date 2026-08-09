from __future__ import annotations

from pathlib import Path

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from npa.workflows.sim2real import isaac_job_io


class _FlakyS3:
    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = failures
        self.calls: list[tuple[str, str, str]] = []

    def upload_file(self, source: str, bucket: str, key: str) -> None:
        self.calls.append((source, bucket, key))
        if self.failures:
            raise self.failures.pop(0)


def _client_error(status: int, code: str) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": "structured test error"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "PutObject",
    )


def test_upload_recovers_from_typed_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    s3 = _FlakyS3([EndpointConnectionError(endpoint_url="https://storage.invalid")])
    sleeps: list[float] = []
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)
    monkeypatch.setattr(isaac_job_io.time, "sleep", sleeps.append)

    isaac_job_io.upload(source, "s3://bucket/prefix/artifact.bin")

    assert len(s3.calls) == 2
    assert sleeps == [2.0]
    output = capsys.readouterr().out
    assert "classification=EndpointConnectionError" in output
    assert "state=retrying" in output


@pytest.mark.parametrize(
    ("status", "code"),
    [(503, "ServiceUnavailable"), (429, "SlowDown"), (400, "RequestTimeout")],
)
def test_upload_recovers_from_structured_service_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: int,
    code: str,
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    s3 = _FlakyS3([_client_error(status, code)])
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)
    monkeypatch.setattr(isaac_job_io.time, "sleep", lambda _delay: None)

    isaac_job_io.upload(source, "s3://bucket/prefix/artifact.bin")

    assert len(s3.calls) == 2


def test_upload_fails_closed_for_nonretryable_client_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    error = _client_error(403, "AccessDenied")
    s3 = _FlakyS3([error])
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)
    monkeypatch.setattr(
        isaac_job_io.time,
        "sleep",
        lambda _delay: pytest.fail("nonretryable errors must not sleep"),
    )

    with pytest.raises(ClientError) as raised:
        isaac_job_io.upload(source, "s3://bucket/prefix/artifact.bin")

    assert raised.value is error
    assert len(s3.calls) == 1


def test_upload_tree_retries_current_file_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "camera-0001.png").write_bytes(b"one")
    (tmp_path / "camera-0002.png").write_bytes(b"two")
    s3 = _FlakyS3([_client_error(503, "ServiceUnavailable")])
    monkeypatch.setattr(isaac_job_io, "_s3", lambda: s3)
    monkeypatch.setattr(isaac_job_io.time, "sleep", lambda _delay: None)

    isaac_job_io.upload_tree(tmp_path, "s3://bucket/run/renders")

    assert [call[2] for call in s3.calls] == [
        "run/renders/camera-0001.png",
        "run/renders/camera-0001.png",
        "run/renders/camera-0002.png",
    ]
    output = capsys.readouterr().out
    assert "operation=upload-tree state=progress files=1 bytes=3" in output
    assert "UPLOADED_TREE uri=s3://bucket/run/renders files=2" in output


def test_retry_delay_configuration_must_be_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NPA_S3_IO_RETRY_BASE_SECONDS", "0")

    with pytest.raises(ValueError, match="must be positive"):
        isaac_job_io._retry_delay(1)
