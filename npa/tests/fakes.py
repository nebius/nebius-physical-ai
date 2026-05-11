from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Literal

from botocore.exceptions import ClientError
import pytest


class FakeS3:
    def __init__(self, name: str = "fake") -> None:
        self.name = name
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


def _fake_s3_factory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    host_mode: Literal["configured", "default"] = "configured",
) -> dict[str, FakeS3]:
    clients = {
        "src-key": FakeS3("src-key"),
        "tgt-key": FakeS3("tgt-key"),
    }
    if host_mode == "configured":
        clients["host:https://source-storage.example"] = FakeS3("host-source")
        clients["host:https://target-storage.example"] = FakeS3("host-target")
        clients["host:https://source-storage.example"].objects = clients[
            "src-key"
        ].objects
        clients["host:https://target-storage.example"].objects = clients[
            "tgt-key"
        ].objects
    else:
        clients["host:None"] = FakeS3("host-default")
        clients["host:None"].objects = clients["tgt-key"].objects

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
