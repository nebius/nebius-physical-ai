"""Apply MoviePy's upstream Pillow compatibility bound to exact 2.2.1 metadata.

Upstream 211e4b15f6ce4f34a6a9efbfff40590e43a68f77 removes the Pillow<12
bound. Keep the released 2.2.1 code, require the security-fixed Pillow>=13
bytes, and bound compatibility below Pillow 14. Versions and licenses stay
unchanged; this is not a vulnerability-version override.
"""

import base64
import csv
import hashlib
import io
from importlib import metadata
from pathlib import Path

from packaging.version import Version


ORIGINAL_SHA256 = "ba08814030a33c196589cb232d87518b2be874b342e8e5f6dfefb464e52e5569"
PATCHED_SHA256 = "f641dfe9064cb9118ec7ddd72916c32b9b2a061d485e47c41e11455f857767ad"


def patched_metadata(source: bytes) -> bytes:
    """Reject unknown metadata before changing the reviewed dependency bound."""
    digest = hashlib.sha256(source).hexdigest()
    if digest == PATCHED_SHA256:
        return source
    if digest != ORIGINAL_SHA256:
        raise RuntimeError("MoviePy metadata differs from reviewed 2.2.1 bytes")
    result = source.replace(
        b"Requires-Dist: pillow<12.0,>=9.2.0",
        b"Requires-Dist: pillow<14.0,>=13.0.0",
    )
    if hashlib.sha256(result).hexdigest() != PATCHED_SHA256:
        raise RuntimeError("MoviePy compatibility patch checksum mismatch")
    return result


def main() -> None:
    """Record the compatibility patch only when patched Pillow is installed."""
    distribution = metadata.distribution("moviepy")
    if distribution.version != "2.2.1":
        raise RuntimeError("Pillow compatibility patch requires MoviePy 2.2.1")
    if not Version("13.0.0") <= Version(metadata.version("pillow")) < Version("14"):
        raise RuntimeError(
            "Pillow compatibility patch requires security-fixed Pillow 13.x"
        )
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
