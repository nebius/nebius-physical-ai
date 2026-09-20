"""Verify exact installed Lightwheel package metadata against reviewed evidence."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys


def _metadata_bytes(distribution: metadata.Distribution) -> tuple[str, bytes]:
    members = [
        member
        for member in distribution.files or ()
        if str(member).endswith(".dist-info/METADATA")
    ]
    if len(members) != 1:
        raise RuntimeError("installed Lightwheel distribution has no unique METADATA")
    return str(members[0]), Path(distribution.locate_file(members[0])).read_bytes()


def verify(evidence_path: Path) -> None:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))["lightwheel_sdk"]
    distribution = metadata.distribution(evidence["distribution_name"])
    if distribution.version != evidence["version"]:
        raise RuntimeError(
            "installed Lightwheel version differs from reviewed evidence"
        )
    metadata_member, metadata_bytes = _metadata_bytes(distribution)
    if metadata_member != evidence["metadata_member"]:
        raise RuntimeError(
            "installed Lightwheel METADATA path differs from reviewed evidence"
        )
    if hashlib.sha256(metadata_bytes).hexdigest() != evidence["metadata_sha256"]:
        raise RuntimeError(
            "installed Lightwheel METADATA differs from the reviewed wheel"
        )
    if b"Licensed under the Apache License, Version 2.0" not in metadata_bytes:
        raise RuntimeError("installed Lightwheel METADATA lost its Apache-2.0 notice")
    if distribution.metadata.get("License"):
        raise RuntimeError(
            "Lightwheel license-field shape changed; review the new wheel"
        )
    license_members = [
        str(member)
        for member in distribution.files or ()
        if "license" in Path(str(member)).name.lower()
    ]
    if evidence["standalone_license_member"] is not None or license_members:
        raise RuntimeError(
            "Lightwheel standalone-license shape changed; review the new wheel"
        )


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
