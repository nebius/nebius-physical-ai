"""Apply the upstream nested-Any fix to the TensorFlow-compatible protobuf pin.

Backport of protocolbuffers/protobuf commit
5ebddcb1bcbe51d1fe323baa145e85f4f23128cf. The installed protobuf BSD-3-Clause
license and copyright notices remain intact. Dependency constraints are unchanged;
the installed RECORD and checked-hash bytecode are refreshed for the patched file.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
from importlib import metadata
from pathlib import Path
import py_compile

ORIGINAL_SHA256 = "01795eef8361486af4a29f0df8eace5e82f42d0fc286c2e4c6249bc31405a339"
PATCHED_SHA256 = "07213c2aa14ca29ce3a8bc31c0aff9f38855ce794f03dc387b56600911ff1c38"
ORIGINAL = (
    b"      methodcaller(_WKTJSONMETHODS[full_name][1], value['value'], sub_message,\n"
    b"                   '{0}.value'.format(path))(\n"
    b"                       self)"
)
PATCHED = (
    b"      # Backported by Nebius from protobuf 5ebddcb1: account for nested Any.\n"
    b"      self.ConvertMessage(value['value'], sub_message,\n"
    b"                          '{0}.value'.format(path))"
)


def apply_backport(path: Path) -> None:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest == PATCHED_SHA256:
        return
    if digest != ORIGINAL_SHA256 or source.count(ORIGINAL) != 1:
        raise RuntimeError(
            "protobuf JSON parser differs from the reviewed upstream bytes"
        )
    patched = source.replace(ORIGINAL, PATCHED)
    if hashlib.sha256(patched).hexdigest() != PATCHED_SHA256:
        raise RuntimeError("protobuf JSON parser backport hash does not match")
    path.write_bytes(patched)


def main() -> None:
    distribution = metadata.distribution("protobuf")
    if distribution.version != "4.25.8":
        raise RuntimeError("the JSON parser backport requires exact protobuf 4.25.8")
    root = Path(distribution.locate_file(""))
    source = root / "google/protobuf/json_format.py"
    apply_backport(source)
    compiled = Path(
        py_compile.compile(
            str(source),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
        )
    )
    record = Path(
        distribution.locate_file(
            next(
                path
                for path in distribution.files or ()
                if str(path).endswith(".dist-info/RECORD")
            )
        )
    )
    updates = {}
    for path in (source, compiled):
        content = path.read_bytes()
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .rstrip(b"=")
            .decode()
        )
        relative = path.relative_to(root).as_posix()
        updates[relative] = [relative, "sha256=" + digest, str(len(content))]
    rows = list(csv.reader(io.StringIO(record.read_text())))
    if not any(row[0] == source.relative_to(root).as_posix() for row in rows):
        raise RuntimeError("protobuf RECORD does not contain the reviewed parser")
    rows = [updates.pop(row[0], row) for row in rows]
    rows.extend(updates.values())
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    record.write_text(output.getvalue())


if __name__ == "__main__":
    main()
