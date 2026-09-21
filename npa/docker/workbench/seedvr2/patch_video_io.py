#!/usr/bin/env python3
"""Apply the reviewed public-video compatibility patch to pinned SeedVR2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_ENTRYPOINT_SHA256 = (
    "089de47cd576bfd51b63b77b8f430146ae85bdd98bc2076011f869e54e2922ee"
)
ORIGINAL_IMPORT = b"from torchvision.io.video import read_video\n"
PATCHED_IMPORT = b"from npa_seedvr2_video_io import read_video\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def patch_video_io(entrypoint: Path, adapter: Path) -> dict[str, str]:
    """Patch one exact upstream entrypoint and install the audited adapter."""

    source = entrypoint.read_bytes()
    source_sha256 = _sha256(source)
    if source_sha256 != EXPECTED_ENTRYPOINT_SHA256:
        raise RuntimeError(
            "SeedVR2 inference entrypoint identity differs from the reviewed source"
        )
    if source.count(ORIGINAL_IMPORT) != 1 or PATCHED_IMPORT in source:
        raise RuntimeError("SeedVR2 video import does not match the reviewed patch")

    adapter_bytes = adapter.read_bytes()
    compile(adapter_bytes, str(adapter), "exec")
    destination = entrypoint.parents[1] / "npa_seedvr2_video_io.py"
    if destination.exists():
        raise RuntimeError("SeedVR2 video adapter destination already exists")

    patched = source.replace(ORIGINAL_IMPORT, PATCHED_IMPORT)
    entrypoint.write_bytes(patched)
    destination.write_bytes(adapter_bytes)
    destination.chmod(0o444)
    return {
        "source_sha256": source_sha256,
        "patched_source_sha256": _sha256(patched),
        "adapter_sha256": _sha256(adapter_bytes),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(json.dumps(patch_video_io(args.entrypoint, args.adapter), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
