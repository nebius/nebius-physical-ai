"""Anonymous immutable acquisition of the public PPISP COLMAP source."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable

from npa.errors import NpaError
from npa.workbench.nurec.source_staging import (
    DATASET_MEMBER,
    DATASET_REPOSITORY,
    DATASET_REVISION,
)


ACQUISITION_FORMAT = "npa_ncore_public_source_acquisition_v1"
SOURCE_SHA256 = "cf7ab7f100da66b2bf05b178ebcfa3a950e1bf2b1d7ff64a6c7a1e1f682afa8d"
SOURCE_URL = (
    f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
    f"{DATASET_REVISION}/{DATASET_MEMBER}"
)


class NcoreSourceAcquisitionError(NpaError):
    """The exact public qualification source could not be acquired."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trusted_hf_redirect(host: str) -> bool:
    return bool(
        re.fullmatch(r"cdn-lfs(?:-[a-z0-9-]+)?\.hf\.co", host)
        or host == "cas-bridge.xethub.hf.co"
        or host == "cdn.hf.co"
        or host.endswith(".cdn.hf.co")
    )


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def acquire_public_source(
    output_path: Path,
    receipt_path: Path,
    *,
    downloader: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Download one exact anonymous source archive and bind its public provenance."""
    from npa._public_https import download_public_https

    if output_path.exists() or output_path.is_symlink():
        raise NcoreSourceAcquisitionError("source output path must be new")
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        output_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    fetch = downloader or download_public_https
    try:
        with os.fdopen(descriptor, "wb") as stream:
            fetch(
                SOURCE_URL,
                stream,
                allowed_hosts=frozenset({"huggingface.co"}),
                redirect_host_policy=_trusted_hf_redirect,
            )
            stream.flush()
            os.fsync(stream.fileno())
        observed = _sha(output_path)
        if observed != SOURCE_SHA256:
            raise NcoreSourceAcquisitionError("source archive SHA-256 differs")
        receipt = {
            "format": ACQUISITION_FORMAT,
            "status": "pass",
            "source": "anonymous_public_https",
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "member": DATASET_MEMBER,
            "license": "CC-BY-4.0",
            "archive_sha256": observed,
            "bytes": output_path.stat().st_size,
            "credentials_forwarded": False,
        }
        _write_private(receipt_path, receipt)
        return receipt
    except Exception:
        output_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        raise
