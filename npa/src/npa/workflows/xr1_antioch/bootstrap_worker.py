"""Read scoped, signed S3 inputs into a fresh Antioch session directory and verify bytes."""

from __future__ import annotations

import hashlib
from contextlib import closing
from http.client import HTTPException, HTTPSConnection
import json
from pathlib import Path, PurePosixPath
import stat
import sys
from urllib.parse import urlsplit
import zipfile


def _relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not name:
        raise ValueError("Input name must remain inside its fresh destination")
    return path


def _receive_file(response, path: Path) -> tuple[str, int]:
    if response.status != 200:
        raise RuntimeError("Signed artifact download did not return success")
    digest, size = hashlib.sha256(), 0
    with path.open("xb") as output:
        for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
            output.write(block)
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _download(root: Path, name: str, entry: dict) -> dict:
    path = root / _relative(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = urlsplit(entry["url"])
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.fragment
    ):
        raise ValueError(
            "Signed downloads require HTTPS without user information or fragments"
        )
    try:
        with closing(
            HTTPSConnection(url.hostname, url.port, timeout=120)
        ) as connection:
            connection.request("GET", url.path + ("?" + url.query if url.query else ""))
            with connection.getresponse() as response:
                digest, size = _receive_file(response, path)
    except (HTTPException, OSError) as error:
        raise RuntimeError(
            f"Signed artifact download failed: {type(error).__name__}"
        ) from None
    if digest != entry["sha256"] or size != entry["bytes"]:
        raise ValueError(f"S3 input differs from its approved identity: {name}")
    return {"sha256": digest, "bytes": size}


def _extract(root: Path, name: str) -> None:
    with zipfile.ZipFile(root / _relative(name)) as archive:
        for entry in archive.infolist():
            _relative(entry.filename)
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Source bundles cannot contain symbolic links")
        archive.extractall(root / "source")


def _main() -> None:
    request = json.load(sys.stdin)
    root = Path(request["root"])
    root.mkdir(parents=True, exist_ok=False)
    verified = {
        name: _download(root, name, entry) for name, entry in request["files"].items()
    }
    if request.get("source_bundle"):
        _extract(root, request["source_bundle"])
    (root / "input-receipt.json").write_text(json.dumps(verified, indent=2))
    print(
        json.dumps(
            {"files": verified, "source": "nebius_s3", "download_verified": True}
        ),
        flush=True,
    )


if __name__ == "__main__":
    _main()
