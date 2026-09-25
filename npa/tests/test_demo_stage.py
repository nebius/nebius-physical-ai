from __future__ import annotations

import hashlib
from io import BytesIO
import logging
import os
from pathlib import Path

from botocore.exceptions import ClientError
import pytest
from typer.testing import CliRunner

from npa.cli.demo import (
    DemoManifestError,
    load_manifest,
    stage_artifacts,
    verify_artifacts,
)
from npa.cli.main import app
from npa.errors import ScopedCredentialError


runner = CliRunner()
HELLO_SHA256 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


class TrackingBody(BytesIO):
    def __init__(self, body: bytes, read_sizes: list[int]) -> None:
        super().__init__(body)
        self._read_sizes = read_sizes

    def read(self, size: int = -1) -> bytes:
        self._read_sizes.append(size)
        return super().read(size)


class DemoStageFakeS3:
    def __init__(self, objects: dict[tuple[str, str], dict] | None = None) -> None:
        self.objects = objects if objects is not None else {}
        self.put_calls: list[tuple[str, str]] = []
        self.copy_calls: list[tuple[str, str, str, str]] = []
        self.get_calls: list[tuple[str, str]] = []
        self.read_sizes: list[int] = []
        self.fail_get: Exception | None = None
        self.fail_get_bucket: str | None = None
        self.fail_put: Exception | None = None
        self.corrupt_put = False
        self.list_calls: list[tuple[str, str, str | None]] = []
        self.list_pages: dict[tuple[str, str, str | None], dict] = {}

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
        if self.fail_get is not None and (
            self.fail_get_bucket is None or self.fail_get_bucket == Bucket
        ):
            raise self.fail_get
        self.get_calls.append((Bucket, Key))
        item = self.objects.get((Bucket, Key))
        if item is None:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        return {"Body": TrackingBody(item["Body"], self.read_sizes)}

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str]
    ):
        if self.fail_put is not None:
            raise self.fail_put
        self.put_calls.append((Bucket, Key))
        stored_body = b"x" * len(Body) if self.corrupt_put else Body
        self.add(Bucket, Key, stored_body, Metadata)

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
        self.list_calls.append((Bucket, Prefix, ContinuationToken))
        configured = self.list_pages.get((Bucket, Prefix, ContinuationToken))
        if configured is not None:
            return configured
        contents = [
            {"Key": key, "Size": len(item["Body"])}
            for (bucket, key), item in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"IsTruncated": False, "KeyCount": len(contents), "Contents": contents}

    def set_list_page(
        self,
        bucket: str,
        prefix: str,
        token: str | None,
        page: dict,
    ) -> None:
        self.list_pages[(bucket, prefix, token)] = page

    def set_two_page_listing(
        self,
        bucket: str,
        prefix: str,
        first_contents: list[dict],
        second_contents: list[dict],
        *,
        token: str,
    ) -> None:
        self.set_list_page(
            bucket,
            prefix,
            None,
            {
                "IsTruncated": True,
                "NextContinuationToken": token,
                "Contents": first_contents,
            },
        )
        self.set_list_page(
            bucket,
            prefix,
            token,
            {"IsTruncated": False, "Contents": second_contents},
        )


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


def _overlap_manifest(path: Path) -> Path:
    path.write_text(
        f"""\
version: 1
artifacts:
  - name: file-one
    source_uri: s3://source/files/file.bin
    target_path: staged/dataset/file.bin
    sha256: {HELLO_SHA256}
    size_bytes: 5
  - name: prefix-one
    source_uri: s3://source/dataset/
    target_path: staged/dataset/
    is_prefix: true
    expected_count: 2
    total_size_bytes: 9
"""
    )
    return path


def _same_bucket_manifest(path: Path) -> Path:
    path.write_text(
        f"""\
version: 1
artifacts:
  - name: file-one
    source_uri: s3://shared/source/file.bin
    target_path: staged/file.bin
    sha256: {HELLO_SHA256}
    size_bytes: 5
"""
    )
    return path


def test_default_demo_manifest_parses() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "manifests"
        / "workbench"
        / "demo-8gpu-h200.yaml"
    )
    manifest = load_manifest(manifest_path)

    assert manifest.version == 1
    assert len(manifest.artifacts) >= 10
    assert any(
        artifact.name == "groot-lerobot-dataset" for artifact in manifest.artifacts
    )


def test_stage_is_idempotent_with_sha_metadata(tmp_path: Path) -> None:
    body = b"hello"
    sha = HELLO_SHA256
    manifest = _manifest(tmp_path / "manifest.yaml", sha=sha)
    s3 = DemoStageFakeS3()
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
    sha = HELLO_SHA256
    manifest = _manifest(tmp_path / "manifest.yaml", sha=sha)
    s3 = DemoStageFakeS3()
    s3.add("source", "path/file.bin", body)
    s3.add("target", "staged/file.bin", b"old", {"sha256": "stale"})

    result = stage_artifacts(
        target_bucket="target", manifest_path=manifest, s3_client=s3
    )

    assert result == [{"name": "file-one", "action": "upload"}]
    assert s3.objects[("target", "staged/file.bin")]["Body"] == body


def test_stage_skips_matching_bytes_without_metadata(tmp_path: Path) -> None:
    body = b"hello"
    sha = HELLO_SHA256
    manifest = _manifest(tmp_path / "manifest.yaml", sha=sha)
    s3 = DemoStageFakeS3()
    s3.add("source", "path/file.bin", body)
    s3.add("target", "staged/file.bin", body)

    result = stage_artifacts(
        target_bucket="target", manifest_path=manifest, s3_client=s3
    )

    assert result == [{"name": "file-one", "action": "skip"}]
    assert s3.put_calls == []


def test_stage_repairs_forged_matching_metadata_with_wrong_bytes(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.yaml", sha=HELLO_SHA256)
    s3 = DemoStageFakeS3()
    s3.add("source", "path/file.bin", b"hello")
    s3.add("target", "staged/file.bin", b"jello", {"sha256": HELLO_SHA256})

    result = stage_artifacts(
        target_bucket="target", manifest_path=manifest, s3_client=s3
    )

    assert result == [{"name": "file-one", "action": "upload"}]
    assert s3.objects[("target", "staged/file.bin")]["Body"] == b"hello"


def test_stage_fails_closed_when_uploaded_bytes_do_not_match(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.yaml", sha=HELLO_SHA256)
    s3 = DemoStageFakeS3()
    s3.add("source", "path/file.bin", b"hello")
    s3.corrupt_put = True

    with pytest.raises(DemoManifestError, match="upload verification failed"):
        stage_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)


def test_stage_does_not_clobber_file_metadata_with_prefix_upload(
    tmp_path: Path,
) -> None:
    manifest = _overlap_manifest(tmp_path / "manifest.yaml")
    s3 = DemoStageFakeS3()
    s3.add("source", "files/file.bin", b"hello")
    s3.add("source", "dataset/file.bin", b"hello")
    s3.add("source", "dataset/other.bin", b"data")

    result = stage_artifacts(
        target_bucket="target", manifest_path=manifest, s3_client=s3
    )

    assert result == [
        {"name": "file-one", "action": "upload"},
        {"name": "prefix-one", "action": "copy"},
    ]
    assert ("source", "dataset/file.bin", "target", "staged/dataset/file.bin") not in (
        s3.copy_calls
    )
    assert s3.copy_calls == [
        ("source", "dataset/other.bin", "target", "staged/dataset/other.bin")
    ]
    assert s3.objects[("target", "staged/dataset/file.bin")]["Metadata"] == {
        "sha256": HELLO_SHA256
    }


def test_stage_verify_works_in_isolation_after_fresh_stage(tmp_path: Path) -> None:
    manifest = _overlap_manifest(tmp_path / "manifest.yaml")
    s3 = DemoStageFakeS3()
    s3.add("source", "files/file.bin", b"hello")
    s3.add("source", "dataset/file.bin", b"hello")
    s3.add("source", "dataset/other.bin", b"data")

    stage_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)

    assert (
        verify_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)
        == []
    )


def test_stage_auth_error_raises_scoped_credential_error(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.yaml", sha="abc", size=0)
    s3 = DemoStageFakeS3()
    s3.fail_get = _access_denied()
    s3.fail_get_bucket = "source"

    with pytest.raises(ScopedCredentialError, match="source"):
        stage_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)


def test_demo_stage_scoped_creds_fail_without_flag(tmp_path: Path) -> None:
    body = b"hello"
    sha = HELLO_SHA256
    manifest = _manifest(tmp_path / "manifest.yaml", sha=sha)
    objects: dict[tuple[str, str], dict] = {}
    scoped_s3 = DemoStageFakeS3(objects)
    host_s3 = DemoStageFakeS3(objects)
    scoped_s3.add("source", "path/file.bin", body)
    scoped_s3.fail_put = _access_denied()

    with pytest.raises(ScopedCredentialError, match="target"):
        stage_artifacts(
            target_bucket="target",
            manifest_path=manifest,
            s3_client=scoped_s3,
            host_s3_client=host_s3,
            allow_host_creds=False,
        )


def test_demo_stage_scoped_creds_fail_with_flag_warns_and_falls_back(
    tmp_path: Path,
    caplog,
) -> None:
    body = b"hello"
    sha = HELLO_SHA256
    manifest = _manifest(tmp_path / "manifest.yaml", sha=sha)
    objects: dict[tuple[str, str], dict] = {}
    scoped_s3 = DemoStageFakeS3(objects)
    host_s3 = DemoStageFakeS3(objects)
    scoped_s3.add("source", "path/file.bin", body)
    scoped_s3.fail_put = _access_denied()

    with caplog.at_level(logging.WARNING, logger="npa.cli.demo"):
        result = stage_artifacts(
            target_bucket="target",
            manifest_path=manifest,
            s3_client=scoped_s3,
            host_s3_client=host_s3,
            allow_host_creds=True,
        )

    assert result == [{"name": "file-one", "action": "upload"}]
    assert scoped_s3.put_calls == []
    assert host_s3.put_calls == [("target", "staged/file.bin")]
    assert "falling back to host credentials" in caplog.text
    assert "target" in caplog.text


def test_stage_target_readback_never_uses_source_project_for_shared_bucket(
    tmp_path: Path,
    mocker,
) -> None:
    manifest = _same_bucket_manifest(tmp_path / "manifest.yaml")
    source_s3 = DemoStageFakeS3()
    target_s3 = DemoStageFakeS3()
    source_s3.add("shared", "source/file.bin", b"hello")
    source_s3.add("shared", "staged/file.bin", b"hello")
    target_s3.add("shared", "staged/file.bin", b"jello", {"sha256": HELLO_SHA256})
    mocker.patch(
        "npa.cli.demo.s3_client_for_project",
        side_effect=[source_s3, target_s3],
    )
    mocker.patch(
        "npa.cli.demo._host_client_for_project",
        side_effect=[DemoStageFakeS3(), DemoStageFakeS3()],
    )

    result = stage_artifacts(
        target_bucket="shared",
        manifest_path=manifest,
        source_project="source-project",
        target_project="target-project",
    )

    assert result == [{"name": "file-one", "action": "upload"}]
    assert source_s3.get_calls == [("shared", "source/file.bin")]
    assert target_s3.get_calls == [
        ("shared", "staged/file.bin"),
        ("shared", "staged/file.bin"),
    ]
    assert target_s3.objects[("shared", "staged/file.bin")]["Body"] == b"hello"


def test_demo_stage_help_includes_allow_host_creds() -> None:
    result = runner.invoke(app, ["demo", "stage", "--help"])

    assert result.exit_code == 0
    assert "allow-host-creds" in result.output or "allow_host_creds" in result.output


def test_demo_verify_help_includes_project_credential_flags() -> None:
    result = runner.invoke(app, ["demo", "verify", "--help"])

    assert result.exit_code == 0
    assert "target-project" in result.output or "target_project" in result.output
    assert "allow-host-creds" in result.output or "allow_host_creds" in result.output


def test_verify_returns_no_issues_on_clean_state(tmp_path: Path) -> None:
    body = b"hello"
    sha = HELLO_SHA256
    manifest = _manifest(tmp_path / "manifest.yaml", sha=sha)
    s3 = DemoStageFakeS3()
    s3.add("target", "staged/file.bin", body, {"sha256": sha})

    assert (
        verify_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)
        == []
    )


def test_verify_ignores_sha256_metadata_when_target_bytes_match(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.yaml", sha=HELLO_SHA256)
    s3 = DemoStageFakeS3()
    s3.add("target", "staged/file.bin", b"hello", {"sha256": "stale"})

    assert (
        verify_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)
        == []
    )


def test_stage_skips_existing_artifact_after_hashing_target_bytes(
    tmp_path: Path,
) -> None:
    body = b"hello"
    manifest = _manifest(tmp_path / "manifest.yaml", sha=HELLO_SHA256)
    s3 = DemoStageFakeS3()
    s3.add("source", "path/file.bin", body)
    s3.add("target", "staged/file.bin", body, {"Sha256": HELLO_SHA256})

    result = stage_artifacts(
        target_bucket="target", manifest_path=manifest, s3_client=s3
    )

    assert result == [{"name": "file-one", "action": "skip"}]
    assert s3.put_calls == []
    assert ("target", "staged/file.bin") in s3.get_calls


def test_verify_uses_project_scoped_credentials(tmp_path: Path, mocker) -> None:
    manifest = _manifest(tmp_path / "manifest.yaml", sha=HELLO_SHA256)
    s3 = DemoStageFakeS3()
    mock_resolve = mocker.patch("npa.cli.demo.s3_client_for_project", return_value=s3)

    result = runner.invoke(
        app,
        [
            "demo",
            "verify",
            "--target-bucket",
            "target",
            "--target-project",
            "project-a",
            "--manifest",
            str(manifest),
        ],
    )

    assert result.exit_code == 1
    assert "missing target object" in result.output
    mock_resolve.assert_called_once_with("project-a", allow_host_creds=False)


def test_verify_rejects_default_project_when_unset_and_default_misconfigured(
    tmp_path: Path,
    mocker,
) -> None:
    manifest = _manifest(tmp_path / "manifest.yaml", sha=HELLO_SHA256)
    mocker.patch(
        "npa.cli.demo.s3_client_for_project",
        side_effect=ScopedCredentialError(
            "default",
            "resolve storage credentials for project 'default'",
            remediation="Configure object-storage credentials for this project.",
            failed_project="default",
        ),
    )

    result = runner.invoke(
        app,
        ["demo", "verify", "--target-bucket", "target", "--manifest", str(manifest)],
    )

    assert result.exit_code == 1
    assert "Configure object-storage credentials" in result.output


def test_verify_respects_allow_host_creds_flag(tmp_path: Path, mocker) -> None:
    manifest = _manifest(tmp_path / "manifest.yaml", sha=HELLO_SHA256)
    s3 = DemoStageFakeS3()
    mock_resolve = mocker.patch("npa.cli.demo.s3_client_for_project", return_value=s3)

    result = runner.invoke(
        app,
        [
            "demo",
            "verify",
            "--target-bucket",
            "target",
            "--target-project",
            "project-a",
            "--allow-host-creds",
            "--manifest",
            str(manifest),
        ],
    )

    assert result.exit_code == 1
    assert "missing target object" in result.output
    mock_resolve.assert_called_once_with("project-a", allow_host_creds=True)


def test_verify_cli_exits_nonzero_on_missing_artifact(tmp_path: Path, mocker) -> None:
    sha = HELLO_SHA256
    manifest = _manifest(tmp_path / "manifest.yaml", sha=sha)
    mocker.patch("npa.cli.demo.s3_client_for_project", return_value=DemoStageFakeS3())

    result = runner.invoke(
        app,
        ["demo", "verify", "--target-bucket", "target", "--manifest", str(manifest)],
    )

    assert result.exit_code == 1
    assert "missing target object" in result.output


def test_verify_detects_corrupt_bytes_with_matching_metadata_and_size(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path / "manifest.yaml", sha=HELLO_SHA256, size=5)
    s3 = DemoStageFakeS3()
    s3.add("target", "staged/file.bin", b"jello", {"sha256": HELLO_SHA256})

    issues = verify_artifacts(
        target_bucket="target", manifest_path=manifest, s3_client=s3
    )

    assert len(issues) == 1
    assert "sha256 mismatch" in issues[0]


def test_verify_hashes_target_in_bounded_chunks(tmp_path: Path) -> None:
    body = b"a" * (1024 * 1024 + 1)
    manifest = _manifest(
        tmp_path / "manifest.yaml",
        sha=hashlib.sha256(body).hexdigest(),
        size=len(body),
    )
    s3 = DemoStageFakeS3()
    s3.add("target", "staged/file.bin", body)

    assert (
        verify_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)
        == []
    )
    assert s3.read_sizes == [1024 * 1024, 1024 * 1024, 1024 * 1024]


def test_prefix_artifacts_verified_by_listing(tmp_path: Path) -> None:
    manifest = _prefix_manifest(tmp_path / "manifest.yaml")
    s3 = DemoStageFakeS3()
    s3.add("target", "staged/dataset/a.bin", b"abc")
    s3.add("target", "staged/dataset/b.bin", b"defg")

    assert (
        verify_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)
        == []
    )


def test_stage_prefix_rejects_truncated_source_page_without_token(
    tmp_path: Path,
) -> None:
    manifest = _prefix_manifest(tmp_path / "manifest.yaml")
    s3 = DemoStageFakeS3()
    s3.set_list_page(
        "source",
        "dataset/",
        None,
        {
            "IsTruncated": True,
            "Contents": [{"Key": "dataset/a.bin", "Size": 3}],
        },
    )

    with pytest.raises(DemoManifestError, match="missing a continuation token"):
        stage_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)

    assert s3.copy_calls == []
    assert s3.list_calls == [("source", "dataset/", None)]


@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize(
    "tokens", [("page-2", "page-2"), ("page-2", "page-3", "page-2")]
)
def test_stage_prefix_rejects_continuation_token_cycles(
    tmp_path: Path,
    side: str,
    tokens: tuple[str, ...],
) -> None:
    manifest = _prefix_manifest(tmp_path / "manifest.yaml")
    s3 = DemoStageFakeS3()
    s3.add("source", "dataset/a.bin", b"abc")
    s3.add("source", "dataset/b.bin", b"defg")
    prefix = "dataset/" if side == "source" else "staged/dataset/"
    previous = None
    for token in tokens:
        s3.set_list_page(
            side,
            prefix,
            previous,
            {
                "IsTruncated": True,
                "NextContinuationToken": token,
                "Contents": [{"Key": prefix + "a.bin", "Size": 3}],
            },
        )
        previous = token

    with pytest.raises(DemoManifestError, match="repeated continuation token"):
        stage_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)

    assert s3.copy_calls == []
    assert [call[2] for call in s3.list_calls if call[0] == side] == [
        None,
        *tokens[:-1],
    ]


def test_verify_prefix_rejects_malformed_target_pagination(tmp_path: Path) -> None:
    manifest = _prefix_manifest(tmp_path / "manifest.yaml")
    s3 = DemoStageFakeS3()
    s3.set_list_page(
        "target",
        "staged/dataset/",
        None,
        {
            "IsTruncated": True,
            "Contents": [{"Key": "staged/dataset/a.bin", "Size": 3}],
        },
    )

    with pytest.raises(DemoManifestError, match="missing a continuation token"):
        verify_artifacts(target_bucket="target", manifest_path=manifest, s3_client=s3)


def test_prefix_staging_and_verification_accept_valid_multi_page_listings(
    tmp_path: Path,
) -> None:
    manifest = _prefix_manifest(tmp_path / "manifest.yaml")
    s3 = DemoStageFakeS3()
    s3.add("source", "dataset/a.bin", b"abc")
    s3.add("source", "dataset/b.bin", b"defg")
    s3.set_two_page_listing(
        "source",
        "dataset/",
        [{"Key": "dataset/a.bin", "Size": 3}],
        [{"Key": "dataset/b.bin", "Size": 4}],
        token="source-page-2",
    )

    result = stage_artifacts(
        target_bucket="target", manifest_path=manifest, s3_client=s3
    )

    assert result == [{"name": "prefix-one", "action": "copy"}]
    assert s3.copy_calls == [
        ("source", "dataset/a.bin", "target", "staged/dataset/a.bin"),
        ("source", "dataset/b.bin", "target", "staged/dataset/b.bin"),
    ]

    s3.set_two_page_listing(
        "target",
        "staged/dataset/",
        [{"Key": "staged/dataset/a.bin", "Size": 3}],
        [{"Key": "staged/dataset/b.bin", "Size": 4}],
        token="target-page-2",
    )

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
