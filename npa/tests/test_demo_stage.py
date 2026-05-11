from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path

from botocore.exceptions import ClientError
import pytest
from typer.testing import CliRunner

from npa.cli.demo import load_manifest, stage_artifacts, verify_artifacts
from npa.cli.main import app
from npa.errors import ScopedCredentialError


runner = CliRunner()


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict] = {}
        self.put_calls: list[tuple[str, str]] = []
        self.copy_calls: list[tuple[str, str, str, str]] = []
        self.fail_get: Exception | None = None

    def add(
        self, bucket: str, key: str, body: bytes, metadata: dict[str, str] | None = None
    ) -> None:
        self.objects[(bucket, key)] = {"Body": body, "Metadata": metadata or {}}

    def head_object(self, *, Bucket: str, Key: str):
        item = self.objects.get((Bucket, Key))
        if item is None:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
            )
        return {"ContentLength": len(item["Body"]), "Metadata": dict(item["Metadata"])}

    def get_object(self, *, Bucket: str, Key: str):
        if self.fail_get is not None:
            raise self.fail_get
        item = self.objects[(Bucket, Key)]
        return {"Body": BytesIO(item["Body"])}

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str]
    ):
        self.put_calls.append((Bucket, Key))
        self.add(Bucket, Key, Body, Metadata)

    def copy_object(
        self,
        *,
        Bucket: str,
        Key: str,
        CopySource: dict[str, str],
        MetadataDirective: str,
    ):
        self.copy_calls.append((CopySource["Bucket"], CopySource["Key"], Bucket, Key))
        source = self.objects[(CopySource["Bucket"], CopySource["Key"])]
        self.add(Bucket, Key, source["Body"], dict(source["Metadata"]))

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str, ContinuationToken: str | None = None
    ):
        contents = [
            {"Key": key, "Size": len(item["Body"])}
            for (bucket, key), item in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"IsTruncated": False, "KeyCount": len(contents), "Contents": contents}


def _access_denied() -> ClientError:
    return ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "denied"}},
        "GetObject",
    )


def _manifest(path: Path, *, sha: str, size: int = 5) -> Path:
    path.write_text(
        f"""\
version: 1
artifacts:
  - name: file-one
    source_uri: s3://source/path/file.bin
    target_path: staged/file.bin
    sha256: {sha}
    size_bytes: {size}
"""
    )
    return path


def _prefix_manifest(path: Path) -> Path:
    path.write_text(
        """\
version: 1
artifacts:
  - name: prefix-one
    source_uri: s3://source/dataset/
    target_path: staged/dataset/
    is_prefix: true
    expected_count: 2
    total_size_bytes: 7
"""
    )
    return path


def test_default_demo_manifest_parses() -> None:
    manifest = load_manifest(Path("npa/manifests/demo-8gpu-h200.yaml"))

    assert manifest.version == 1
    assert len(manifest.artifacts) >= 10
    assert any(
        artifact.name == "groot-lerobot-dataset" for artifact in manifest.artifacts
    )


def test_stage_is_idempotent_with_sha_metadata(tmp_path: Path) -> None:
    body = b"hello"
    sha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    manifest = _manifest(tmp_path / "manifest.yaml", sha=sha)
    s3 = FakeS3()
    s3.add("source", "path/file.bin", body)

    first = stage_artifacts(
        target_bucket="target", manifest_path=manifest, s3_client=s3
    )
    second = stage_artifacts(
        target_bucket="target", manifest_path=manifest, s3_client=s3
    )

    assert first == [{"name": "file-one", "action": "upload"}]
    assert second == [{"name": "file-one", "action": "skip"}]
    assert s3.put_calls == [("target", "staged/file.bin")]
    assert s3.objects[("target", "staged/file.bin")]["Metadata"]["sha256"] == sha


def test_stage_hash_mismatch_redownloads_and_uploads(tmp_path: Path) -> None:
    body = b"hello"
    sha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    manifest = _manifest(tmp_path / "manifest.yaml", sha=sha)
    s3 = FakeS3()
    s3.add("source", "path/file.bin", body)
    s3.add("target", "staged/file.bin", b"old", {"sha256": "stale"})

    result = stage_artifacts(
        target_bucket="target", manifest_path=manifest, s3_client=s3
    )

    assert result == [{"name": "file-one", "action": "upload"}]
    assert s3.objects[("target", "staged/file.bin")]["Body"] == body


def test_stage_missing_metadata_redownloads_legacy_object(tmp_path: Path) -> None:
    body = b"hello"
    sha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    manifest = _manifest(tmp_path / "manifest.yaml", sha=sha)
    s3 = FakeS3()
    s3.add("source", "path/file.bin", body)
    s3.add("target", "staged/file.bin", body)

    result = stage_artifacts(
        target_bucket="target", manifest_path=manifest, s3_client=s3
    )

    assert result == [{"name": "file-one", "action": "upload"}]
    assert s3.objects[("target", "staged/file.bin")]["Metadata"]["sha256"] == sha


def test_stage_auth_error_raises_scoped_credential_error(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.yaml", sha="abc", size=0)
    s3 = FakeS3()
    s3.fail_get = _access_denied()

    with pytest.raises(ScopedCredentialError, match="source"):
        stage_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)


def test_verify_returns_no_issues_on_clean_state(tmp_path: Path) -> None:
    body = b"hello"
    sha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    manifest = _manifest(tmp_path / "manifest.yaml", sha=sha)
    s3 = FakeS3()
    s3.add("target", "staged/file.bin", body, {"sha256": sha})

    assert (
        verify_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)
        == []
    )


def test_verify_cli_exits_nonzero_on_missing_artifact(tmp_path: Path, mocker) -> None:
    sha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    manifest = _manifest(tmp_path / "manifest.yaml", sha=sha)
    mocker.patch("npa.cli.demo._s3_client", return_value=FakeS3())

    result = runner.invoke(
        app,
        ["demo", "verify", "--target-bucket", "target", "--manifest", str(manifest)],
    )

    assert result.exit_code == 1
    assert "missing target object" in result.output


def test_verify_returns_issue_on_hash_mismatch(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.yaml", sha="expected", size=5)
    s3 = FakeS3()
    s3.add("target", "staged/file.bin", b"hello", {"sha256": "actual"})

    issues = verify_artifacts(
        target_bucket="target", manifest_path=manifest, s3_client=s3
    )

    assert any("sha256 metadata mismatch" in issue for issue in issues)


def test_prefix_artifacts_verified_by_listing(tmp_path: Path) -> None:
    manifest = _prefix_manifest(tmp_path / "manifest.yaml")
    s3 = FakeS3()
    s3.add("target", "staged/dataset/a.bin", b"abc")
    s3.add("target", "staged/dataset/b.bin", b"defg")

    assert (
        verify_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)
        == []
    )


@pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION") != "1"
    or not os.environ.get("NPA_DEMO_STAGE_TEST_BUCKET"),
    reason="requires NPA_INTEGRATION=1 and NPA_DEMO_STAGE_TEST_BUCKET",
)
def test_demo_stage_integration_round_trip() -> None:
    bucket = os.environ["NPA_DEMO_STAGE_TEST_BUCKET"]

    stage_artifacts(target_bucket=bucket)
    assert verify_artifacts(target_bucket=bucket) == []
    stage_artifacts(target_bucket=bucket)
    assert verify_artifacts(target_bucket=bucket) == []
