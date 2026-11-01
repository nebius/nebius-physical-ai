"""Bind MoviePy 2.2.1 metadata to the exact qualified Pillow 13 release.

MoviePy's released metadata caps Pillow below 12 even though its unchanged
2.2.1 source passes the complete upstream suite against Pillow 13.0.0. Keep
those released code bytes, require exactly the security-fixed Pillow 13.0.0
closure, and update only the installed metadata and RECORD. Versions and
licenses stay unchanged; this is not a vulnerability-version override.
"""

import base64
import csv
import hashlib
import io
from importlib import metadata
from pathlib import Path

from packaging.version import Version


ORIGINAL_SHA256 = "ba08814030a33c196589cb232d87518b2be874b342e8e5f6dfefb464e52e5569"
PATCHED_SHA256 = "40bc8558f5dba3d3c3d31831b8f32253e1b9ad74527ced0242130c201a2e2595"


def patched_metadata(source: bytes) -> bytes:
    """Reject unknown metadata before changing the reviewed dependency bound."""
    digest = hashlib.sha256(source).hexdigest()
    if digest == PATCHED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise RuntimeError("MoviePy metadata differs from reviewed 2.2.1 bytes")
    result = source.replace(
        b"Requires-Dist: pillow<12.0,>=9.2.0",
        b"Requires-Dist: pillow==13.0.0",
    )
    if hashlib.sha256(result).hexdigest() != PATCHED_SHA256:
        raise RuntimeError("MoviePy compatibility patch checksum mismatch")
    return result


def main() -> None:
    """Record the compatibility patch only when patched Pillow is installed."""
    distribution = metadata.distribution("moviepy")
    if distribution.version != "2.2.1":
        raise RuntimeError("Pillow compatibility patch requires MoviePy 2.2.1")
    if Version(metadata.version("pillow")) != Version("13.0.0"):
        raise RuntimeError("MoviePy compatibility patch requires Pillow 13.0.0")
    root = Path(distribution.locate_file(""))
    relative = "moviepy-2.2.1.dist-info/METADATA"
    target = root / relative
    content = patched_metadata(target.read_bytes())
    record = root / "moviepy-2.2.1.dist-info/RECORD"
    rows = list(csv.reader(io.StringIO(record.read_text())))
    if sum(row[0] == relative for row in rows) != 1:
        raise RuntimeError("MoviePy RECORD must contain exactly one METADATA entry")
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
