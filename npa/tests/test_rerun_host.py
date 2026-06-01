from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path

from botocore.exceptions import ClientError, NoCredentialsError
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.rerun import RERUN_VERSION, RerunHostError, host_recording
from npa.clients.config import StorageConfig
from npa.errors import ScopedCredentialError


runner = CliRunner()


class RerunHostFakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict] = {}
        self.put_calls: list[tuple[str, str, dict[str, str]]] = []
        self.head_calls: list[tuple[str, str]] = []
        self.get_calls: list[tuple[str, str]] = []
        self.presign_calls: list[dict] = []
        self.head_error: Exception | None = None
        self.get_error: Exception | None = None
        self.put_error: Exception | None = None

    def add(
        self, bucket: str, key: str, body: bytes, metadata: dict[str, str] | None = None
    ) -> None:
        self.objects[(bucket, key)] = {"Body": body, "Metadata": metadata or {}}

    def head_object(self, *, Bucket: str, Key: str):
        self.head_calls.append((Bucket, Key))
        if self.head_error is not None:
            raise self.head_error
        item = self.objects.get((Bucket, Key))
        if item is None:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
            )
        return {"ContentLength": len(item["Body"]), "Metadata": dict(item["Metadata"])}

    def get_object(self, *, Bucket: str, Key: str):
        self.get_calls.append((Bucket, Key))
        if self.get_error is not None:
            raise self.get_error
        item = self.objects[(Bucket, Key)]
        return {"Body": BytesIO(item["Body"])}

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str]
    ) -> None:
        if self.put_error is not None:
            raise self.put_error
        self.put_calls.append((Bucket, Key, dict(Metadata)))
        self.add(Bucket, Key, Body, Metadata)

    def generate_presigned_url(self, operation: str, *, Params: dict, ExpiresIn: int):
        self.presign_calls.append(
            {"operation": operation, "Params": Params, "ExpiresIn": ExpiresIn}
        )
        bucket = Params["Bucket"]
        key = Params["Key"]
        return f"https://storage.example/{bucket}/{key}?expires={ExpiresIn}"


class NoCallS3:
    def __getattr__(self, name: str):
        raise AssertionError(f"unexpected S3 call: {name}")


def _storage() -> StorageConfig:
    return StorageConfig(
        checkpoint_bucket="s3://default-bucket/checkpoints/",
        endpoint_url="https://storage.example",
        aws_access_key_id="scoped",
        aws_secret_access_key="secret",
    )


def _rrd(tmp_path: Path, body: bytes = b"recording") -> tuple[Path, str]:
    path = tmp_path / "recording.rrd"
    path.write_bytes(body)
    return path, hashlib.sha256(body).hexdigest()


def test_local_file_uploads_with_sha_metadata_and_generates_versioned_url(
    tmp_path: Path, mocker
) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    path, sha = _rrd(tmp_path)
    s3 = RerunHostFakeS3()

    result = host_recording(
        str(path),
        target_bucket="target",
        s3_client=s3,
        host_s3_client=RerunHostFakeS3(),
        now=datetime(2026, 5, 11, 22, 30, tzinfo=timezone.utc),
    )

    key = f"rerun-shared/{sha}.rrd"
    assert result.rrd_s3_uri == f"s3://target/{key}"
    assert result.sha256 == sha
    assert result.rerun_version == RERUN_VERSION
    assert result.ttl_expires_at == "2026-05-11T23:30:00Z"
    assert result.share_url.startswith(
        f"https://app.rerun.io/version/{RERUN_VERSION}/?url="
    )
    assert s3.objects[("target", key)]["Metadata"] == {"sha256": sha}
    assert s3.put_calls == [("target", key, {"sha256": sha})]
    assert s3.presign_calls[-1]["ExpiresIn"] == 3600


def test_s3_file_with_matching_metadata_skips_upload(mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    body = b"remote"
    sha = hashlib.sha256(body).hexdigest()
    s3 = RerunHostFakeS3()
    s3.add("bucket", "path/input.rrd", body, {"sha256": sha})

    result = host_recording(
        "s3://bucket/path/input.rrd",
        s3_client=s3,
        host_s3_client=RerunHostFakeS3(),
    )

    assert result.rrd_s3_uri == "s3://bucket/path/input.rrd"
    assert result.sha256 == sha
    assert s3.put_calls == []
    assert s3.get_calls == [("bucket", "path/input.rrd")]


def test_s3_file_with_stale_metadata_reuploads(mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    body = b"remote"
    sha = hashlib.sha256(body).hexdigest()
    s3 = RerunHostFakeS3()
    s3.add("bucket", "path/input.rrd", body, {"sha256": "stale"})

    host_recording("s3://bucket/path/input.rrd", s3_client=s3, host_s3_client=RerunHostFakeS3())

    assert s3.put_calls == [("bucket", "path/input.rrd", {"sha256": sha})]
    assert s3.objects[("bucket", "path/input.rrd")]["Metadata"]["sha256"] == sha


def test_s3_file_with_missing_metadata_reuploads(mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    body = b"remote"
    sha = hashlib.sha256(body).hexdigest()
    s3 = RerunHostFakeS3()
    s3.add("bucket", "path/input.rrd", body)

    host_recording("s3://bucket/path/input.rrd", s3_client=s3, host_s3_client=RerunHostFakeS3())

    assert s3.put_calls == [("bucket", "path/input.rrd", {"sha256": sha})]


def test_missing_file_is_clear_error(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())

    with pytest.raises(RerunHostError, match="does not exist"):
        host_recording(
            str(tmp_path / "missing.rrd"),
            s3_client=RerunHostFakeS3(),
            host_s3_client=RerunHostFakeS3(),
        )


def test_missing_scoped_creds_without_flag_raises_scoped_credential_error(
    tmp_path: Path, mocker
) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    path, _ = _rrd(tmp_path)
    s3 = RerunHostFakeS3()
    s3.head_error = NoCredentialsError()

    with pytest.raises(ScopedCredentialError, match="target"):
        host_recording(
            str(path),
            target_bucket="target",
            s3_client=s3,
            host_s3_client=RerunHostFakeS3(),
        )


def test_missing_scoped_creds_with_flag_warns_and_uses_host_fallback(
    tmp_path: Path, mocker, caplog
) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    path, sha = _rrd(tmp_path)
    scoped = RerunHostFakeS3()
    scoped.head_error = NoCredentialsError()
    scoped.put_error = NoCredentialsError()
    host = RerunHostFakeS3()

    with caplog.at_level("WARNING"):
        result = host_recording(
            str(path),
            target_bucket="target",
            allow_host_creds=True,
            s3_client=scoped,
            host_s3_client=host,
        )

    assert result.sha256 == sha
    assert "target" in caplog.text
    assert "falling back to host credentials" in caplog.text
    assert host.put_calls == [("target", f"rerun-shared/{sha}.rrd", {"sha256": sha})]


def test_ttl_over_limit_fails_before_s3_calls() -> None:
    with pytest.raises(RerunHostError, match="cannot exceed"):
        host_recording(
            "s3://bucket/path/input.rrd",
            ttl_hours=200,
            s3_client=NoCallS3(),
            host_s3_client=NoCallS3(),
        )


def test_json_output_contains_full_schema(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    scoped = RerunHostFakeS3()
    host = RerunHostFakeS3()
    mocker.patch("npa.cli.rerun.s3_client_for_project", return_value=scoped)
    mocker.patch("npa.cli.rerun._host_s3_client", return_value=host)
    path, sha = _rrd(tmp_path)

    result = runner.invoke(
        app,
        [
            "rerun",
            "host",
            str(path),
            "--target-bucket",
            "target",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert set(data) == {
        "share_url",
        "rrd_s3_uri",
        "presigned_url",
        "ttl_expires_at",
        "sha256",
        "rerun_version",
    }
    assert data["sha256"] == sha
    assert data["rrd_s3_uri"] == f"s3://target/rerun-shared/{sha}.rrd"
    assert data["rerun_version"] == RERUN_VERSION
    assert "app.rerun.io" in data["share_url"]
    assert "version" in data["share_url"]


def test_host_cli_routes_target_project_credentials(
    tmp_path: Path,
    mocker,
) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    scoped = RerunHostFakeS3()
    mocker.patch("npa.cli.rerun._host_s3_client", return_value=RerunHostFakeS3())
    s3_client_for_project = mocker.patch(
        "npa.cli.rerun.s3_client_for_project", return_value=scoped
    )
    path, sha = _rrd(tmp_path)

    result = runner.invoke(
        app,
        [
            "rerun",
            "host",
            str(path),
            "--target-bucket",
            "target",
            "--target-project",
            "project-target",
        ],
    )

    assert result.exit_code == 0
    s3_client_for_project.assert_called_once_with(
        "project-target", allow_host_creds=False
    )
    assert scoped.put_calls == [("target", f"rerun-shared/{sha}.rrd", {"sha256": sha})]


def test_host_cli_routes_source_project_for_s3_input(mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    body = b"remote-recording"
    sha = hashlib.sha256(body).hexdigest()
    scoped = RerunHostFakeS3()
    scoped.add("source", "recordings/input.rrd", body, {"sha256": sha})
    mocker.patch("npa.cli.rerun._host_s3_client", return_value=RerunHostFakeS3())
    s3_client_for_project = mocker.patch(
        "npa.cli.rerun.s3_client_for_project", return_value=scoped
    )

    result = runner.invoke(
        app,
        [
            "rerun",
            "host",
            "s3://source/recordings/input.rrd",
            "--source-project",
            "project-source",
        ],
    )

    assert result.exit_code == 0
    s3_client_for_project.assert_called_once_with(
        "project-source", allow_host_creds=False
    )
    assert scoped.get_calls == [("source", "recordings/input.rrd")]


def test_expiration_parameter_matches_ttl_hours(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    path, _ = _rrd(tmp_path)
    s3 = RerunHostFakeS3()

    host_recording(
        str(path),
        target_bucket="target",
        ttl_hours=6,
        s3_client=s3,
        host_s3_client=RerunHostFakeS3(),
    )

    assert s3.presign_calls[-1]["ExpiresIn"] == 6 * 3600
