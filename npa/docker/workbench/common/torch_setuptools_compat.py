"""Repair the stale Setuptools runtime bound in the exact Torch 2.11 wheel.

PyTorch upstream commit c776884 removed the unnecessary ``setuptools<82``
runtime constraint after confirming that current Setuptools is supported.  The
build-time constraint remains separate.  Torch 2.11's released cu128 wheel
predates that correction, so this module applies the same runtime metadata
change to exact reviewed wheel bytes and updates RECORD.  It does not change
Torch code, versions, licenses, or vulnerability data.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
from importlib import metadata
from pathlib import Path


TORCH_VERSION = "2.11.0+cu128"
METADATA_RELATIVE = "torch-2.11.0+cu128.dist-info/METADATA"
RECORD_RELATIVE = "torch-2.11.0+cu128.dist-info/RECORD"
ORIGINAL_SHA256 = "de881a744ff66b27fffc338c1624660cf7c87d3019574ad71e3dda886b2a172b"
PATCHED_SHA256 = "63d6d6eaa7465a24d0178ca256c85831a4e9e90708301b889a263e1238b961ee"
ORIGINAL_REQUIREMENT = b"Requires-Dist: setuptools<82\n"
PATCHED_REQUIREMENT = b"Requires-Dist: setuptools>=77.0.3\n"


def patched_metadata(source: bytes) -> bytes:
    """Return exact upstream-aligned metadata, rejecting all unknown bytes."""
    digest = hashlib.sha256(source).hexdigest()
    if digest == PATCHED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise RuntimeError("Torch metadata differs from reviewed 2.11.0+cu128 bytes")
    if source.count(ORIGINAL_REQUIREMENT) != 1:
        raise RuntimeError("Torch metadata must contain one reviewed Setuptools bound")
    result = source.replace(ORIGINAL_REQUIREMENT, PATCHED_REQUIREMENT, 1)
    if hashlib.sha256(result).hexdigest() != PATCHED_SHA256:
        raise RuntimeError("Torch Setuptools compatibility patch checksum mismatch")
    return result


def _regular_unshared_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise RuntimeError(f"Torch {label} must be one regular unshared file")


def main() -> None:
    """Patch exact installed metadata and its sole RECORD row fail closed."""
    distribution = metadata.distribution("torch")
    if distribution.version != TORCH_VERSION:
        raise RuntimeError(f"Setuptools compatibility requires Torch {TORCH_VERSION}")
    root = Path(distribution.locate_file(""))
    target = root / METADATA_RELATIVE
    record = root / RECORD_RELATIVE
    _regular_unshared_file(target, "METADATA")
    _regular_unshared_file(record, "RECORD")
    content = patched_metadata(target.read_bytes())
    rows = list(csv.reader(io.StringIO(record.read_text())))
    if sum(row and row[0] == METADATA_RELATIVE for row in rows) != 1:
        raise RuntimeError("Torch RECORD must contain exactly one METADATA entry")
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
    )
    rows = [
        [METADATA_RELATIVE, "sha256=" + digest, str(len(content))]
        if row and row[0] == METADATA_RELATIVE
        else row
        for row in rows
    ]
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    target.write_bytes(content)
    record.write_text(output.getvalue())
    print(
        f"patched {METADATA_RELATIVE} to sha256={PATCHED_SHA256} "
        "using PyTorch upstream c776884"
    )


if __name__ == "__main__":
    main()
