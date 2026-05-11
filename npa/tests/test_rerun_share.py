from __future__ import annotations

from datetime import UTC, datetime, timedelta
from io import BytesIO
import hashlib
import json
from pathlib import Path

from botocore.exceptions import ClientError
import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.rerun import (
    MAX_TTL_HOURS,
    RerunHostError,
    list_share_items,
    revoke_share,
    share_recording,
)
from npa.clients.config import StorageConfig


runner = CliRunner()


class RerunShareFakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict] = {}
        self.put_calls: list[tuple[str, str, dict[str, str]]] = []
        self.delete_calls: list[tuple[str, str]] = []
        self.presign_calls: list[dict] = []

    def add(
        self,
        bucket: str,
        key: str,
        body: bytes = b"recording",
        metadata: dict[str, str] | None = None,
        last_modified: datetime | None = None,
    ) -> None:
        self.objects[(bucket, key)] = {
            "Body": body,
            "Metadata": metadata or {},
            "LastModified": last_modified or datetime(2026, 5, 11, tzinfo=UTC),
        }

    def head_object(self, *, Bucket: str, Key: str):
        item = self.objects.get((Bucket, Key))
        if item is None:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
            )
        return {"ContentLength": len(item["Body"]), "Metadata": dict(item["Metadata"])}

    def get_object(self, *, Bucket: str, Key: str):
        item = self.objects[(Bucket, Key)]
        return {"Body": BytesIO(item["Body"])}

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str]
    ) -> None:
        self.put_calls.append((Bucket, Key, dict(Metadata)))
        self.add(Bucket, Key, Body, Metadata)

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str, ContinuationToken: str | None = None
    ):
        contents = [
            {
                "Key": key,
                "Size": len(item["Body"]),
                "LastModified": item["LastModified"],
            }
            for (bucket, key), item in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"IsTruncated": False, "KeyCount": len(contents), "Contents": contents}

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        self.delete_calls.append((Bucket, Key))
        self.objects.pop((Bucket, Key), None)

    def generate_presigned_url(self, operation: str, *, Params: dict, ExpiresIn: int):
        self.presign_calls.append(
            {"operation": operation, "Params": Params, "ExpiresIn": ExpiresIn}
        )
        return f"https://storage.example/{Params['Bucket']}/{Params['Key']}?ttl={ExpiresIn}"


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


def test_share_default_ttl_generates_seven_day_url(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    path, sha = _rrd(tmp_path)
    s3 = RerunShareFakeS3()

    result = share_recording(
        str(path),
        target_bucket="target",
        s3_client=s3,
        host_s3_client=RerunShareFakeS3(),
        now=datetime(2026, 5, 11, 22, 30, tzinfo=UTC),
    )

    assert result.rrd_s3_uri == f"s3://target/rerun-shares/default/{sha}.rrd"
    assert result.ttl_expires_at == "2026-05-18T22:30:00Z"
    assert s3.presign_calls[-1]["ExpiresIn"] == MAX_TTL_HOURS * 3600


def test_share_label_is_stored_in_metadata(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    path, sha = _rrd(tmp_path)
    s3 = RerunShareFakeS3()

    share_recording(
        str(path),
        target_bucket="target",
        label="weekly-failure-review",
        s3_client=s3,
        host_s3_client=RerunShareFakeS3(),
    )

    metadata = s3.objects[("target", f"rerun-shares/default/{sha}.rrd")]["Metadata"]
    assert metadata["sha256"] == sha
    assert metadata["rerun-label"] == "weekly-failure-review"


def test_share_workspace_controls_object_path(tmp_path: Path, mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    path, sha = _rrd(tmp_path)
    s3 = RerunShareFakeS3()

    result = share_recording(
        str(path),
        target_bucket="s3://target/prefix",
        workspace="team-perception",
        s3_client=s3,
        host_s3_client=RerunShareFakeS3(),
    )

    key = f"prefix/rerun-shares/team-perception/{sha}.rrd"
    assert result.rrd_s3_uri == f"s3://target/{key}"
    assert ("target", key) in s3.objects


def test_list_shares_returns_metadata_for_active_shares(mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    s3 = RerunShareFakeS3()
    now = datetime(2026, 5, 11, 22, 30, tzinfo=UTC)
    s3.add(
        "target",
        "rerun-shares/team-a/aaa.rrd",
        metadata={
            "sha256": "aaa",
            "rerun-label": "review-a",
            "rerun-workspace": "team-a",
        },
        last_modified=now - timedelta(hours=2),
    )
    s3.add(
        "target",
        "rerun-shares/team-b/bbb.rrd",
        metadata={"sha256": "bbb", "rerun-workspace": "team-b"},
        last_modified=now - timedelta(minutes=5),
    )

    items = list_share_items(target_bucket="target", s3_client=s3, now=now)

    assert [(item.label, item.workspace, item.age, item.sha256) for item in items] == [
        ("review-a", "team-a", "2h", "aaa"),
        ("", "team-b", "5m", "bbb"),
    ]


def test_revoke_by_sha_deletes_matching_object(mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    s3 = RerunShareFakeS3()
    s3.add(
        "target",
        "rerun-shares/default/aaa.rrd",
        metadata={"sha256": "aaa", "rerun-workspace": "default"},
    )

    deleted = revoke_share("aaa", target_bucket="target", s3_client=s3)

    assert deleted == 1
    assert s3.delete_calls == [("target", "rerun-shares/default/aaa.rrd")]
    assert ("target", "rerun-shares/default/aaa.rrd") not in s3.objects


def test_revoke_by_label_deletes_matching_object(mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    s3 = RerunShareFakeS3()
    s3.add(
        "target",
        "rerun-shares/default/aaa.rrd",
        metadata={
            "sha256": "aaa",
            "rerun-label": "weekly",
            "rerun-workspace": "default",
        },
    )
    s3.add(
        "target",
        "rerun-shares/default/bbb.rrd",
        metadata={
            "sha256": "bbb",
            "rerun-label": "other",
            "rerun-workspace": "default",
        },
    )

    deleted = revoke_share("weekly", target_bucket="target", s3_client=s3)

    assert deleted == 1
    assert ("target", "rerun-shares/default/aaa.rrd") not in s3.objects
    assert ("target", "rerun-shares/default/bbb.rrd") in s3.objects


def test_revoke_when_already_gone_is_success(mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    s3 = RerunShareFakeS3()

    assert revoke_share("missing", target_bucket="target", s3_client=s3) == 0
    assert s3.delete_calls == []


def test_share_ttl_over_limit_fails_before_s3_calls() -> None:
    with pytest.raises(RerunHostError, match="cannot exceed"):
        share_recording(
            "s3://bucket/path/input.rrd",
            ttl_hours=169,
            s3_client=NoCallS3(),
            host_s3_client=NoCallS3(),
        )


def test_list_shares_json_cli_outputs_programmatic_schema(mocker) -> None:
    mocker.patch("npa.cli.rerun.resolve_project_storage", return_value=_storage())
    s3 = RerunShareFakeS3()
    mocker.patch("npa.cli.rerun._s3_client", return_value=s3)
    mocker.patch("npa.cli.rerun._host_s3_client", return_value=RerunShareFakeS3())
    s3.add(
        "target",
        "rerun-shares/default/aaa.rrd",
        metadata={
            "sha256": "aaa",
            "rerun-label": "review",
            "rerun-workspace": "default",
        },
    )

    result = runner.invoke(
        app,
        ["rerun", "list-shares", "--target-bucket", "target", "--output", "json"],
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data[0]["label"] == "review"
    assert data[0]["workspace"] == "default"
    assert data[0]["sha256"] == "aaa"
    assert "expires_at" not in data[0]
