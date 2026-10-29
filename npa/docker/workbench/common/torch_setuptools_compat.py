"""Keep the exact Torch 2.11 CUDA 12.8 wheel compatible with the secure closure.

Torch 2.11.0+cu128 declares ``setuptools<82`` although its extension interface
only requires the long-established lower-bound API.  Isaac Lab 3's reviewed
security closure pins Setuptools 84.2.1.  This adapter permits that exact,
tested combination without lowering the security floor or hiding a failed
``pip check``.  It is intentionally limited to the reviewed wheel metadata
and maintains the wheel's RECORD integrity.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
from importlib import metadata
from pathlib import Path


ORIGINAL_SHA256 = "de881a744ff66b27fffc338c1624660cf7c87d3019574ad71e3dda886b2a172b"
PATCHED_SHA256 = "63d6d6eaa7465a24d0178ca256c85831a4e9e90708301b889a263e1238b961ee"
ORIGINAL_REQUIREMENT = b"Requires-Dist: setuptools<82"
PATCHED_REQUIREMENT = b"Requires-Dist: setuptools>=77.0.3"
TORCH_VERSION = "2.11.0+cu128"
SETUPTOOLS_VERSION = "84.2.1"


def patched_metadata(source: bytes) -> bytes:
    """Change precisely the reviewed bound, otherwise fail closed."""
    digest = hashlib.sha256(source).hexdigest()
    if digest == PATCHED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise RuntimeError("Torch metadata differs from reviewed 2.11.0+cu128 bytes")
    if source.count(ORIGINAL_REQUIREMENT) != 1:
        raise RuntimeError(
            "Torch metadata has an unexpected Setuptools requirement count"
        )
    result = source.replace(ORIGINAL_REQUIREMENT, PATCHED_REQUIREMENT, 1)
    if hashlib.sha256(result).hexdigest() != PATCHED_SHA256:
        raise RuntimeError("Torch metadata compatibility checksum mismatch")
    return result


def main() -> None:
    """Repair the exact installed metadata and the corresponding RECORD row."""
    distribution = metadata.distribution("torch")
    if distribution.version != TORCH_VERSION:
        raise RuntimeError(f"compatibility repair requires Torch {TORCH_VERSION}")
    if metadata.version("setuptools") != SETUPTOOLS_VERSION:
        raise RuntimeError(
            f"compatibility repair requires secure Setuptools {SETUPTOOLS_VERSION}"
        )
    root = Path(distribution.locate_file(""))
    relative = f"torch-{TORCH_VERSION}.dist-info/METADATA"
    target = root / relative
    content = patched_metadata(target.read_bytes())
    record = root / f"torch-{TORCH_VERSION}.dist-info/RECORD"
    rows = list(csv.reader(io.StringIO(record.read_text())))
    if sum(row[0] == relative for row in rows) != 1:
        raise RuntimeError("Torch RECORD must contain exactly one METADATA entry")
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
    )
    rows = [
        [relative, "sha256=" + digest, str(len(content))] if row[0] == relative else row
        for row in rows
    ]
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    target.write_bytes(content)
    record.write_text(output.getvalue())


if __name__ == "__main__":
    main()
