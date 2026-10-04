"""Remove one hash-bound inert sample-download example, preserving executable ASTs."""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
from importlib import metadata
from io import StringIO
import json
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "50e6234fa2170820eaf8d0f8f42b51905822afc3680a4f09113fa11d435f7fb4"
)
EXAMPLE_MARKER = "The following code was used to obtain the final image."


def _without_sample_example(raw: bytes) -> bytes:
    tree = ast.parse(raw)
    grass = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "grass"
    )
    examples = [
        node
        for node in grass.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and EXAMPLE_MARKER in node.value.value
        and "?token=" in node.value.value
    ]
    if len(examples) != 1 or examples[0] is grass.body[0]:
        raise RuntimeError("expected one inert grass sample-download example")
    example = examples[0]
    lines = raw.splitlines(keepends=True)
    repaired = b"".join(lines[: example.lineno - 1] + lines[example.end_lineno :])
    grass.body.remove(example)
    if ast.dump(ast.parse(repaired), include_attributes=False) != ast.dump(
        tree, include_attributes=False
    ):
        raise RuntimeError(
            "sample-example removal changed executable or documented API"
        )
    return repaired


def _record_digest(value: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(value).digest()
    ).decode().rstrip("=")


def _updated_record(raw: str, original: bytes, repaired: bytes) -> str:
    records = list(csv.reader(StringIO(raw)))
    if any(len(row) != 3 for row in records):
        raise RuntimeError("malformed installed RECORD")
    entries = [row for row in records if row[0] == "skimage/data/_fetchers.py"]
    if len(entries) != 1 or entries[0][1:] != [
        _record_digest(original),
        str(len(original)),
    ]:
        raise RuntimeError(
            "installed sample-fetcher RECORD does not match original bytes"
        )
    entries[0][1:] = [_record_digest(repaired), str(len(repaired))]
    output = StringIO()
    csv.writer(output).writerows(records)
    return output.getvalue()


def _sanitize_module(path: Path, record: Path) -> dict[str, str]:
    if path.is_symlink() or record.is_symlink():
        raise RuntimeError("sample-source and RECORD must be owned regular files")
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("unreviewed sample-fetcher source")
    repaired = _without_sample_example(original)
    updated_record = _updated_record(record.read_text(), original, repaired)
    path.write_bytes(repaired)
    record.write_text(updated_record)
    return {
        "original_sha256": EXPECTED_SOURCE_SHA256,
        "repaired_sha256": hashlib.sha256(repaired).hexdigest(),
        "scope": "One inert sample-download example removed; executable AST and primary API documentation unchanged. RECORD updated; original image failures retained externally.",
    }


if __name__ == "__main__":
    distribution = metadata.distribution("scikit-image")
    source = Path(distribution.locate_file("skimage/data/_fetchers.py"))
    record_path = Path(distribution._path) / "RECORD"
    receipt = _sanitize_module(source, record_path)
    print(json.dumps(receipt, sort_keys=True))
