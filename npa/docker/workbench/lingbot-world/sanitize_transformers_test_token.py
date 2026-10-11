"""Remove the pinned Transformers test credential before copying a runtime layer."""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import io
import sys
from pathlib import Path

SOURCE_SHA256 = "399dd0399c430d53da151c4edfeaf381617289b99a88902c0a354696a6c57668"
SOURCE_PATH = "transformers/testing_utils.py"
MODIFICATION_NOTICE = (
    "# Modified by Nebius Physical AI: remove the bundled staging test credential.\n"
)


def _sanitize_source(original: bytes) -> bytes:
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError(
            "Transformers test-helper source differs from the reviewed pin"
        )
    assignments = [
        node
        for node in ast.parse(original).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "TOKEN"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError("Expected one top-level test TOKEN assignment")
    assignment = assignments[0]
    value = assignment.value
    if (
        not isinstance(value, ast.Constant)
        or not isinstance(value.value, str)
        or not value.value
    ):
        raise ValueError("Expected a nonempty literal test credential")
    lines = original.decode("utf-8").splitlines(keepends=True)
    replacement = MODIFICATION_NOTICE + 'TOKEN = ""\n'
    lines[assignment.lineno - 1 : assignment.end_lineno] = [replacement]
    sanitized = "".join(lines).encode("utf-8")
    if value.value.encode("utf-8") in sanitized:
        raise ValueError("Test credential remains elsewhere in the pinned source")
    ast.parse(sanitized)
    return sanitized


def _updated_record(record: Path, sanitized: bytes) -> str:
    rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
    source_rows = [row for row in rows if row and row[0] == SOURCE_PATH]
    if len(source_rows) != 1 or len(source_rows[0]) != 3:
        raise ValueError("Expected one installed test-helper RECORD row")
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(sanitized).digest())
        .rstrip(b"=")
        .decode()
    )
    source_rows[0][1:] = ["sha256=" + digest, str(len(sanitized))]
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for row in rows:
        if (
            row
            and row[0].startswith("transformers/")
            and row[0].endswith((".pyc", ".pyo"))
        ):
            continue
        writer.writerow(row)
    return output.getvalue()


def sanitize(site_packages: Path) -> None:
    """Remove the test credential and stale bytecode from the build-stage copy.

    Args:
        site_packages: The copied Wan Python 3.10 site-packages directory.
    Returns:
        None.
    Raises:
        ValueError: The pinned source or installed package RECORD differs.
        OSError: A build-stage file cannot be read, written, or removed.
    """
    source = site_packages / SOURCE_PATH
    record = site_packages / "transformers-5.5.0.dist-info/RECORD"
    sanitized = _sanitize_source(source.read_bytes())
    updated_record = _updated_record(record, sanitized)
    source.write_bytes(sanitized)
    for pattern in ("*.pyc", "*.pyo"):
        for bytecode in (site_packages / "transformers").rglob(pattern):
            bytecode.unlink()
    record.write_text(updated_record, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: sanitize_transformers_test_token.py SITE_PACKAGES")
    sanitize(Path(sys.argv[1]))
