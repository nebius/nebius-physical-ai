"""Exchange hash-verified XR1 artifacts through an explicitly selected S3 run prefix."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


def _relative(name: str) -> Path:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or "\x00" in name or path.as_posix() != name or name == "."):
        raise ValueError("Artifact names must remain inside their selected directory")
    return Path(*path.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _Store:
    def __init__(self, uri: str):
        import boto3

        location = urlsplit(uri)
        if (location.scheme != "s3" or not location.netloc or not location.path.strip("/")
                or location.query or location.fragment or ".." in PurePosixPath(location.path).parts):
            raise ValueError("XR1 artifacts require an explicit S3 run prefix")
        self.bucket, self.prefix = location.netloc, location.path.strip("/")
        self.client = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{_relative(name).as_posix()}"

    def read_json(self, name: str) -> dict:
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(name))
        with response["Body"] as stream:
            return json.loads(stream.read())

    def download(self, name: str, target: Path, expected: str) -> None:
        key = self._key(name)
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(target))
        if _sha256(target) != expected:
            raise ValueError(f"Downloaded artifact checksum differs: {name}")

    def publish(self, root: Path) -> dict:
        manifest = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            name = path.relative_to(root).as_posix()
            digest = _sha256(path)
            self.client.upload_file(str(path), self.bucket, self._key(name),
                                    ExtraArgs={"Metadata": {"sha256": digest}})
            self.verify(name, digest)
            manifest[name] = {"sha256": digest, "bytes": path.stat().st_size}
        self.client.put_object(Bucket=self.bucket, Key=self._key("manifest.json"),
                               Body=json.dumps(manifest, sort_keys=True).encode())
        return manifest

    def verify(self, name: str, expected: str) -> None:
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(name))
        digest = hashlib.sha256()
        with response["Body"] as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Published artifact failed full S3 readback: {name}")
