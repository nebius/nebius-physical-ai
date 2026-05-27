from __future__ import annotations

import json
from io import BytesIO

from typer.testing import CliRunner

from npa.cli.main import app


runner = CliRunner()


class FakeS3:
    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.objects: dict[tuple[str, str], dict] = {}

    def add(self, bucket: str, key: str, body: bytes, metadata: dict | None = None) -> None:
        self.objects[(bucket, key)] = {"Body": body, "Metadata": metadata or {}}

    def get_object(self, *, Bucket: str, Key: str):
        item = self.objects[(Bucket, Key)]
        return {"Body": BytesIO(item["Body"]), "Metadata": dict(item["Metadata"])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict) -> None:
        self.add(Bucket, Key, Body, Metadata)

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str,
        ContinuationToken: str | None = None,
    ):
        del ContinuationToken
        contents = [
            {"Key": key, "Size": len(item["Body"])}
            for (bucket, key), item in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"IsTruncated": False, "KeyCount": len(contents), "Contents": contents}


def test_workbench_data_command_help() -> None:
    result = runner.invoke(app, ["workbench", "data", "--help"])

    assert result.exit_code == 0
    assert "S3 data import bridge" in result.output


def test_workbench_data_sync_copies_between_s3_prefixes(monkeypatch) -> None:
    source = FakeS3("source")
    target = FakeS3("target")
    source.add("raw", "sereact/a.json", b"{}")
    source.add("raw", "sereact/nested/b.json", b'{"b": true}')

    def fake_client(project, allow_host_creds=False):
        assert allow_host_creds is False
        return source if project == "src" else target

    monkeypatch.setattr("npa.cli.workbench.data.s3_client_for_project", fake_client)

    result = runner.invoke(
        app,
        [
            "workbench",
            "data",
            "sync",
            "--source-project",
            "src",
            "--target-project",
            "dst",
            "--input-path",
            "s3://raw/sereact/",
            "--output-path",
            "s3://stage/imported/",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "synced"
    assert payload["object_count"] == 2
    assert sorted(target.objects) == [
        ("stage", "imported/a.json"),
        ("stage", "imported/nested/b.json"),
    ]


def test_workbench_data_sync_respects_dry_run(monkeypatch) -> None:
    source = FakeS3("source")
    target = FakeS3("target")
    source.add("raw", "sereact/a.json", b"{}")

    def fake_client(project, allow_host_creds=False):
        return source if project == "src" else target

    monkeypatch.setattr("npa.cli.workbench.data.s3_client_for_project", fake_client)
    monkeypatch.setenv("NPA_DRY_RUN", "1")

    result = runner.invoke(
        app,
        [
            "workbench",
            "data",
            "sync",
            "--source-project",
            "src",
            "--target-project",
            "dst",
            "--input-path",
            "s3://raw/sereact/",
            "--output-path",
            "s3://stage/imported/",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "dry_run"
    assert payload["dry_run"] is True
    assert target.objects == {}


def test_workbench_data_status_and_list(monkeypatch) -> None:
    source = FakeS3("source")
    source.add("raw", "sereact/a.json", b"1234")
    source.add("raw", "sereact/b.json", b"12")
    monkeypatch.setattr(
        "npa.cli.workbench.data.s3_client_for_project",
        lambda project, allow_host_creds=False: source,
    )

    status = runner.invoke(
        app,
        [
            "workbench",
            "data",
            "status",
            "--input-path",
            "s3://raw/sereact/",
            "--output",
            "json",
        ],
    )
    listing = runner.invoke(
        app,
        [
            "workbench",
            "data",
            "list",
            "--input-path",
            "s3://raw/sereact/",
            "--limit",
            "1",
            "--output",
            "json",
        ],
    )

    assert status.exit_code == 0
    assert json.loads(status.output)["bytes_total"] == 6
    assert listing.exit_code == 0
    assert len(json.loads(listing.output)["objects"]) == 1
