"""Send only failed RoboTwin image evidence to an operator-signed private PUT."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
from urllib.parse import urlsplit

import httpx


UPLOAD_ENV = "ROBOTWIN_PRIVATE_FAILURE_UPLOAD_URL"
MEMBERS = ("image.tar", "scan/report.json", "scan/records.jsonl")


def _add_evidence(archive: tarfile.TarFile, phase: Path, name: str) -> None:
    """Copy a fixed regular file through its opened descriptor."""
    descriptor = os.open(phase / name, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Invalid private evidence file")
        member = tarfile.TarInfo(name)
        member.size = info.st_size
        member.mode = 0o600
        archive.addfile(member, source)


def upload_failure(phase: Path, url: str) -> None:
    """Upload a three-file tar through one HTTPS PUT without following redirects.

    Args:
        phase: Private failed byte-gate directory.
        url: Operator-provided URL granting PUT for one private object.
    Returns:
        None after a successful response.
    Raises:
        ValueError: The destination or an evidence file is invalid.
        OSError, httpx.HTTPError: Evidence preparation or transport failed.
    """
    destination = urlsplit(url)
    if (
        destination.scheme != "https"
        or not destination.hostname
        or destination.username is not None
        or destination.password is not None
        or destination.fragment
    ):
        raise ValueError("Invalid private upload destination")
    with tempfile.TemporaryFile(dir=phase) as bundle:
        with tarfile.open(fileobj=bundle, mode="w") as archive:
            for name in MEMBERS:
                _add_evidence(archive, phase, name)
        size = bundle.tell()
        bundle.seek(0)
        with httpx.Client(
            verify=True, follow_redirects=False, trust_env=False
        ) as client:
            with client.stream(
                "PUT",
                url,
                content=iter(lambda: bundle.read(1024 * 1024), b""),
                headers={
                    "Content-Type": "application/x-tar",
                    "Content-Length": str(size),
                },
            ) as response:
                response.raise_for_status()


def main() -> int:
    """Run the optional upload without emitting private exceptions or arguments.

    Args:
        None; the phase is positional and the signed URL is environment-only.
    Returns:
        Zero for upload success, otherwise one; never changes the scan verdict.
    Raises:
        None for ordinary evidence or transport failures.
    """
    try:
        upload_failure(Path(sys.argv[1]), os.environ[UPLOAD_ENV])
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
