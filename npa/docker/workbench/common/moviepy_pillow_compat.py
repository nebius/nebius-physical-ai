"""Bind MoviePy 2.2.1 metadata to the exact qualified Pillow 13 release.

MoviePy's released metadata caps Pillow below 12. NPA's reviewed compatibility
policy permits only unchanged MoviePy 2.2.1 release bytes with the exact
security-fixed Pillow 13.0.0 closure. Update only the installed metadata and
RECORD. Versions and licenses stay unchanged; this is not a
vulnerability-version override or a claim that a build runs MoviePy's suite.
"""

import base64
import csv
import hashlib
import io
from importlib import metadata
from pathlib import Path

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


def patched_record(source: bytes, relative: str, content: bytes) -> bytes:
    """Update only the target RECORD row, preserving every other byte."""
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("MoviePy RECORD is not valid UTF-8") from exc
    lines = text.splitlines(keepends=True)
    matches: list[tuple[int, list[str], str]] = []
    for index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        rows = list(csv.reader([body]))
        if len(rows) != 1:
            raise RuntimeError("MoviePy RECORD contains an invalid CSV row")
        if rows[0] and rows[0][0] == relative:
            matches.append((index, rows[0], ending))
    if len(matches) != 1:
        raise RuntimeError("MoviePy RECORD must contain exactly one METADATA entry")
    index, row, ending = matches[0]
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
    )
    row[1:] = ["sha256=" + digest, str(len(content))]
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator=ending or "\n").writerow(row)
    lines[index] = output.getvalue()
    return "".join(lines).encode("utf-8")


def main() -> None:
    """Record the compatibility patch only when patched Pillow is installed."""
    distribution = metadata.distribution("moviepy")
    if distribution.version != "2.2.1":
        raise RuntimeError("Pillow compatibility patch requires MoviePy 2.2.1")
    if metadata.version("pillow") != "13.0.0":
        raise RuntimeError("MoviePy compatibility patch requires Pillow 13.0.0")
    root = Path(distribution.locate_file(""))
    relative = "moviepy-2.2.1.dist-info/METADATA"
    target = root / relative
    content = patched_metadata(target.read_bytes())
    record = root / "moviepy-2.2.1.dist-info/RECORD"
    record_content = patched_record(record.read_bytes(), relative, content)
    target.write_bytes(content)
    record.write_bytes(record_content)


if __name__ == "__main__":
    main()
