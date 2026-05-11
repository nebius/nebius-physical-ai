from __future__ import annotations

from io import BytesIO
from pathlib import Path

from botocore.exceptions import ClientError
import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.demo import stage_artifacts
from npa.cli.main import app
from npa.clients import config
from npa.clients.project_credentials import resolve_credentials
from npa.errors import ScopedCredentialError


runner = CliRunner()


class FakeS3:
    def __init__(self, name: str) -> None:
        self.name = name
        self.objects: dict[tuple[str, str], dict] = {}
        self.get_calls: list[tuple[str, str]] = []
        self.put_calls: list[tuple[str, str]] = []
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


@pytest.fixture()
def cross_project_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cfg = tmp_path / ".npa" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        yaml.safe_dump(
            {
                "default_project": "project-source",
                "projects": {
                    "project-source": {
                        "storage": {
                            "endpoint_url": "https://source-storage.example",
                            "aws_access_key_id": "src-key",
                            "aws_secret_access_key": "src-secret",
                            "bucket": "s3://source/default/",
                        }
                    },
                    "project-target": {
                        "storage": {
                            "endpoint_url": "https://target-storage.example",
                            "aws_access_key_id": "tgt-key",
                            "aws_secret_access_key": "tgt-secret",
                            "bucket": "s3://target/default/",
                        }
                    },
                },
            },
            sort_keys=False,
        )
    )
    monkeypatch.setattr(config, "CONFIG_PATH", cfg)
    return cfg


def _manifest(path: Path) -> Path:
    body = b"hello"
    path.write_text(
        f"""\
version: 1
artifacts:
  - name: file-one
    source_uri: s3://source/path/file.bin
    target_path: staged/file.bin
    sha256: {"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"}
    size_bytes: {len(body)}
"""
    )
    return path


def _access_denied() -> ClientError:
    return ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "denied"}},
        "PutObject",
    )


def _fake_s3_factory(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeS3]:
    clients = {
        "src-key": FakeS3("src-key"),
        "tgt-key": FakeS3("tgt-key"),
        "host:https://source-storage.example": FakeS3("host-source"),
        "host:https://target-storage.example": FakeS3("host-target"),
    }
    clients["host:https://source-storage.example"].objects = clients["src-key"].objects
    clients["host:https://target-storage.example"].objects = clients["tgt-key"].objects
    clients["src-key"].add("source", "path/file.bin", b"hello")

    def fake_client(
        service_name: str,
        *,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        config=None,
    ):
        assert service_name == "s3"
        key = aws_access_key_id or f"host:{endpoint_url}"
        return clients.setdefault(key, FakeS3(key))

    monkeypatch.setattr("boto3.client", fake_client)
    return clients


def test_resolve_credentials_default_project(cross_project_config: Path) -> None:
    credentials = resolve_credentials(project=None)

    assert credentials.project is None
    assert credentials.aws_access_key_id == "src-key"
    assert credentials.aws_secret_access_key == "src-secret"


def test_resolve_credentials_other_project(cross_project_config: Path) -> None:
    credentials = resolve_credentials(project="project-target")

    assert credentials.project == "project-target"
    assert credentials.endpoint_url == "https://target-storage.example"
    assert credentials.aws_access_key_id == "tgt-key"


def test_resolve_credentials_nonexistent_project_raises(
    cross_project_config: Path,
) -> None:
    with pytest.raises(ScopedCredentialError, match="missing-project"):
        resolve_credentials(project="missing-project")


def test_demo_stage_cross_project_uses_source_and_target_credentials(
    tmp_path: Path,
    cross_project_config: Path,
    monkeypatch: pytest.MonkeyPatch,
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


def test_demo_stage_cross_project_failure_names_target_project(
    tmp_path: Path,
    cross_project_config: Path,
    monkeypatch: pytest.MonkeyPatch,
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


def test_demo_stage_cross_project_host_fallback_uses_target_host_credentials(
    tmp_path: Path,
    cross_project_config: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    assert clients["host:https://target-storage.example"].put_calls == [
        ("target", "staged/file.bin")
    ]
    assert "project-target" in caplog.text
    assert "falling back to host credentials" in caplog.text


def test_cross_project_flags_are_visible_in_help() -> None:
    commands = [
        ["demo", "stage"],
        ["workbench", "cosmos", "infer"],
        ["workbench", "groot", "infer"],
        ["workbench", "isaac-lab", "export-lerobot"],
    ]
    for command in commands:
        result = runner.invoke(app, [*command, "--help"])
        assert result.exit_code == 0
        assert "target-project" in result.output
        if command[-1] != "export-lerobot":
            assert "source-project" in result.output
