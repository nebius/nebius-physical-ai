#!/usr/bin/env python3
"""Fail closed if the final SeedVR2 service layer contains model-like payloads."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys

SCHEMA = "npa.seedvr2.weight-payload-scan.v1"
SCAN_ROOTS = (Path("/opt/npa-src"), Path("/opt/npa-venv"))
RERUN_PATH_FILE = Path("/opt/npa-venv/lib/python3.12/site-packages/rerun_sdk.pth")
RERUN_PATH_FILE_BYTES = b"rerun_sdk\n"
MODEL_SUFFIXES = (".pth", ".pt", ".safetensors", ".ckpt")


def _is_model_like(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(MODEL_SUFFIXES) or (
        name.startswith("pytorch_model") and name.endswith(".bin")
    )


def _walk(root: Path):
    if not root.is_dir():
        raise ValueError(f"required scan root is not a directory: {root}")

    def raise_walk_error(error: OSError) -> None:
        raise error

    for directory, directories, files in os.walk(
        root, followlinks=False, onerror=raise_walk_error
    ):
        for name in sorted([*directories, *files]):
            yield Path(directory) / name


def _verify_allowed_path_file(path: Path, expected_bytes: bytes) -> dict[str, object]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"allowed Python path file is not regular: {path}")
    payload = path.read_bytes()
    if payload != expected_bytes:
        raise ValueError(f"allowed Python path file bytes differ: {path}")
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify(
    roots: tuple[Path, ...],
    *,
    allowed_path_file: Path,
    expected_path_file_bytes: bytes,
) -> dict[str, object]:
    """Verify model-like names and one exact Python import-path file.

    Args:
        roots: Final image directories whose complete trees must be inspected.
        allowed_path_file: Sole legitimate `.pth` path in the inspected trees.
        expected_path_file_bytes: Exact authorized bytes for that path.
    Returns:
        A deterministic report describing the completed scan.
    Raises:
        OSError: A root cannot be completely walked or a candidate cannot be read.
        ValueError: A root, allowed path file, or model-like payload is invalid.
    """

    allowed_record = None
    rejected: list[str] = []
    for root in roots:
        for path in _walk(root):
            if not _is_model_like(path):
                continue
            if path == allowed_path_file:
                allowed_record = _verify_allowed_path_file(
                    path, expected_path_file_bytes
                )
            else:
                rejected.append(str(path))
    if allowed_record is None:
        raise ValueError(f"allowed Python path file is missing: {allowed_path_file}")
    if rejected:
        raise ValueError("model-like payloads are forbidden: " + ", ".join(rejected))
    return {
        "schema": SCHEMA,
        "roots": [str(root) for root in roots],
        "allowed_python_path_file": allowed_record,
        "rejected_payloads": [],
    }


def main() -> int:
    """Run the fixed final-image scan and emit its deterministic JSON report."""

    try:
        report = verify(
            SCAN_ROOTS,
            allowed_path_file=RERUN_PATH_FILE,
            expected_path_file_bytes=RERUN_PATH_FILE_BYTES,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
