from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path

from botocore.exceptions import ClientError
import pytest

from npa.cli.demo import stage_artifacts
from npa.errors import ScopedCredentialError


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict] = {}
        self.get_calls: list[tuple[str, str]] = []
        self.put_calls: list[tuple[str, str]] = []
        self.fail_get: Exception | None = None
        self.fail_put: Exception | None = None

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
        self.get_calls.append((Bucket, Key))
        item = self.objects[(Bucket, Key)]
        return {"Body": BytesIO(item["Body"]), "Metadata": dict(item["Metadata"])}

    def put_object(
        self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict[str, str]
    ) -> None:
        if self.fail_put is not None:
            raise self.fail_put
        self.put_calls.append((Bucket, Key))
        self.add(Bucket, Key, Body, Metadata)

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
        "PutObject",
    )


def _manifest(path: Path) -> Path:
    path.write_text(
        """\
version: 1
artifacts:
  - name: file-one
    source_uri: s3://source/path/file.bin
    target_path: staged/file.bin
    sha256: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    size_bytes: 5
"""
    )
    return path


def _fake_s3_factory(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeS3]:
    clients = {
        "src-key": FakeS3(),
        "tgt-key": FakeS3(),
        "host:None": FakeS3(),
    }
    clients["src-key"].add("source", "path/file.bin", b"hello")
    clients["host:None"].objects = clients["tgt-key"].objects

    def fake_client(
        service_name: str,
        *,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        config=None,
    ):
        assert service_name == "s3"
        return clients.setdefault(aws_access_key_id or f"host:{endpoint_url}", FakeS3())

    monkeypatch.setattr("boto3.client", fake_client)
    return clients


def test_mock_cross_project_creds_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_cross_project_creds,
) -> None:
    clients = _fake_s3_factory(monkeypatch)

    result = stage_artifacts(
        target_bucket="target",
        manifest_path=_manifest(tmp_path / "manifest.yaml"),
        source_project="project-source",
        target_project="project-target",
    )

    assert result == [{"name": "file-one", "action": "upload"}]
    assert clients["src-key"].get_calls == [("source", "path/file.bin")]
    assert clients["tgt-key"].put_calls == [("target", "staged/file.bin")]


def test_mock_cross_project_creds_target_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_cross_project_creds,
) -> None:
    clients = _fake_s3_factory(monkeypatch)
    clients["tgt-key"].fail_put = _access_denied()

    with pytest.raises(ScopedCredentialError, match="project-target"):
        stage_artifacts(
            target_bucket="target",
            manifest_path=_manifest(tmp_path / "manifest.yaml"),
            source_project="project-source",
            target_project="project-target",
        )


def test_mock_cross_project_creds_source_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_cross_project_creds,
) -> None:
    clients = _fake_s3_factory(monkeypatch)
    clients["src-key"].fail_get = _access_denied()

    with pytest.raises(ScopedCredentialError, match="project-source"):
        stage_artifacts(
            target_bucket="target",
            manifest_path=_manifest(tmp_path / "manifest.yaml"),
            source_project="project-source",
            target_project="project-target",
        )


def test_mock_cross_project_creds_host_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_cross_project_creds,
    caplog,
) -> None:
    clients = _fake_s3_factory(monkeypatch)
    clients["tgt-key"].fail_put = _access_denied()

    with caplog.at_level("WARNING"):
        result = stage_artifacts(
            target_bucket="target",
            manifest_path=_manifest(tmp_path / "manifest.yaml"),
            source_project="project-source",
            target_project="project-target",
            allow_host_creds=True,
        )

    assert result == [{"name": "file-one", "action": "upload"}]
    assert "falling back to host credentials" in caplog.text


@pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_CROSS_PROJECT") != "1",
    reason="set NPA_INTEGRATION_CROSS_PROJECT=1 and provide real test projects",
)
def test_live_cross_project_demo_stage_placeholder() -> None:
    pytest.skip(
        "Live cross-project wiring requires two Nebius test projects, distinct "
        "scoped principals, and one bucket in each project."
    )
