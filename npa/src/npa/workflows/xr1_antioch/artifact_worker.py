"""Transfer explicit artifact files using presigned requests supplied over standard input."""

from __future__ import annotations

import hashlib
from contextlib import closing
from http.client import HTTPException, HTTPSConnection
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Artifact roots cannot contain symlinks")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = {
                "sha256": _digest(path), "bytes": path.stat().st_size,
            }
    if not files:
        raise ValueError("Artifact directory contains no files")
    return files


def _upload(root: Path, transfers: dict) -> dict:
    completed = []
    for relative, transfer in transfers.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Upload path must be a file inside the artifact directory")
        if _digest(path) != transfer["sha256"]:
            raise ValueError("Artifact changed after its transfer manifest was sealed")
        url = urlsplit(transfer["url"])
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.fragment:
            raise ValueError("Signed uploads require HTTPS without user information or fragments")
        with closing(HTTPSConnection(url.hostname, url.port, timeout=120)) as connection, path.open("rb") as stream:
            connection.request("PUT", url.path + ("?" + url.query if url.query else ""), body=stream, headers={
                "Content-Length": str(path.stat().st_size),
                "x-amz-meta-sha256": transfer["sha256"],
            })
            if connection.getresponse().status != 200:
                raise RuntimeError("Artifact upload did not return success")
        completed.append(relative)
    return {"uploaded": completed}


def _main() -> None:
    request = json.load(sys.stdin)
    root = Path(request["root"]).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("An artifact root must be a directory")
    try:
        if request["operation"] == "manifest":
            result = _manifest(root)
        elif request["operation"] == "upload":
            result = _upload(root, request["transfers"])
        else:
            raise ValueError("Unsupported artifact operation")
    except (HTTPException, OSError) as error:
        # Signed requests contain credentials, which must not enter error logs.
        raise RuntimeError(f"Artifact request failed ({type(error).__name__})") from None
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    _main()
