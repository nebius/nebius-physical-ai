"""Verify exact video destinations, full byte verification and storage failures."""

import hashlib
import io
from pathlib import Path

from botocore.exceptions import ClientError
import pytest
from yaml import YAMLError

from npa import video_output
from npa.clients.config import ConfigError
from npa.clients.credentials import CredentialStoreError
from npa.clients.project_credential_store import ProjectCredentialStoreError
from npa.errors import ScopedCredentialError


class _Storage:
    def __init__(self):
        self.calls = []
        self.payload = b""
        self.corrupt = False
        self.failure = None
        self.change_source = False

    def upload_file(self, source, bucket, key, **options):
        self.calls.append((bucket, key, options))
        if self.failure:
            raise self.failure
        self.payload = Path(source).read_bytes()
        if self.change_source:
            Path(source).write_bytes(b"a later render")

    def get_object(self, **options):
        self.calls.append(options)
        payload = b"x" * len(self.payload) if self.corrupt else self.payload
        self.body = io.BytesIO(payload)
        return {"Body": self.body, "VersionId": "test-version"}


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    source = tmp_path / "film.mp4"
    source.write_bytes(b"retained video payload" * 100000)
    storage = _Storage()
    projects = []

    def client(project):
        projects.append(project)
        return storage

    monkeypatch.setattr(video_output, "s3_client_for_project", client)
    return source, storage, projects


@pytest.mark.parametrize("project", [None, "test-storage"])
def test_delivers_exact_key_and_verifies_readback(delivery, project):
    source, storage, projects = delivery
    before = source.read_bytes()
    target = "s3://test-bucket/a folder/literal%20name.mp4"
    result = video_output.write_video_output(source, target, project=project)
    assert projects == [project]
    assert storage.calls[0] == (
        "test-bucket",
        "a folder/literal%20name.mp4",
        {
            "ExtraArgs": {
                "ContentType": "video/mp4",
                "Metadata": {"sha256": hashlib.sha256(before).hexdigest()},
            }
        },
    )
    assert storage.calls[1] == {
        "Bucket": "test-bucket",
        "Key": "a folder/literal%20name.mp4",
    }
    assert result == {
        "output_path": target,
        "sha256": hashlib.sha256(before).hexdigest(),
        "bytes": len(before),
        "verified": True,
        "version_id": "test-version",
    }
    assert source.read_bytes() == before
    assert storage.body.closed


@pytest.mark.parametrize(
    "target",
    [
        "",
        "film.mp4",
        "https://example.invalid/video.mp4",
        "s3://test-bucket/",
        "s3://test-bucket/video",
        "s3://test-bucket/video.mp4?signature=private",
        "s3://test-bucket/video.mp4#fragment",
        "s3://user:secret@test-bucket/video.mp4",
        "s3://test-bucket:443/video.mp4",
        "s3://test bucket/video.mp4",
    ],
)
def test_invalid_destination_never_resolves_credentials(delivery, target):
    source, storage, projects = delivery
    with pytest.raises(ValueError):
        video_output.write_video_output(source, target)
    assert not projects and not storage.calls


def test_corruption_fails_even_when_byte_count_matches(delivery):
    source, storage, _ = delivery
    before = source.read_bytes()
    storage.corrupt = True
    with pytest.raises(ValueError, match="uploaded bytes differ"):
        video_output.write_video_output(source, "s3://test-bucket/video.mp4")
    assert source.read_bytes() == before
    assert storage.body.closed


def test_rejects_local_video_changed_during_delivery(delivery):
    source, storage, _ = delivery
    storage.change_source = True
    with pytest.raises(ValueError, match="Local video changed"):
        video_output.write_video_output(source, "s3://test-bucket/video.mp4")


def test_upload_failure_retains_local_video_and_redacts_provider_error(delivery):
    source, storage, _ = delivery
    before = source.read_bytes()
    storage.failure = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "private-provider-detail"}},
        "PutObject",
    )
    with pytest.raises(ValueError, match="local video retained") as failure:
        video_output.write_video_output(source, "s3://test-bucket/video.mp4")
    assert "private-provider-detail" not in str(failure.value)
    assert source.read_bytes() == before
    assert len(storage.calls) == 1


def test_selected_project_failure_never_falls_back(delivery, monkeypatch):
    source, storage, projects = delivery

    def deny(project):
        projects.append(project)
        raise ScopedCredentialError("test-bucket", "private-provider-detail")

    monkeypatch.setattr(video_output, "s3_client_for_project", deny)
    with pytest.raises(ValueError, match="storage configuration") as failure:
        video_output.write_video_output(
            source, "s3://test-bucket/video.mp4", project="missing"
        )
    assert "private-provider-detail" not in str(failure.value)
    assert projects == ["missing"] and not storage.calls


@pytest.mark.parametrize(
    "error",
    [
        ConfigError("private-configuration-content"),
        CredentialStoreError(
            "malformed", Path("config.yaml"), "private-configuration-content"
        ),
        ProjectCredentialStoreError("private-configuration-content"),
        YAMLError("private-configuration-content"),
    ],
)
def test_invalid_configuration_reports_a_sanitized_failure(
    delivery, monkeypatch, error
):
    source, storage, _ = delivery

    def invalid(project):
        raise error

    monkeypatch.setattr(video_output, "s3_client_for_project", invalid)
    with pytest.raises(ValueError, match="storage configuration") as failure:
        video_output.write_video_output(source, "s3://test-bucket/video.mp4")
    assert "private-configuration-content" not in str(failure.value)
    assert not storage.calls


def test_download_failure_is_not_reported_as_success(delivery, monkeypatch):
    source, storage, _ = delivery

    def deny(**options):
        raise ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "private"}}, "GetObject"
        )

    monkeypatch.setattr(storage, "get_object", deny)
    with pytest.raises(ValueError, match="S3 video delivery failed"):
        video_output.write_video_output(source, "s3://test-bucket/video.mp4")
    assert source.is_file()
