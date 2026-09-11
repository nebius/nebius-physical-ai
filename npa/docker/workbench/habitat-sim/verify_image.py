#!/usr/bin/env python3
"""Verify a Habitat-Sim OCI archive without starting the image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

NPA_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(NPA_ROOT / "scripts"))

from image_byte_scan import core as W  # noqa: E402
from image_byte_scan import habitat_sim_verification as H  # noqa: E402
from image_byte_scan import prepare as P  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the complete Habitat OCI verification command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oci-archive", type=Path, required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-dpkg-inventory-sha256", required=True)
    parser.add_argument("--expected-native-closure-sha256", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    contract = json.loads(Path(__file__).with_name("runtime-payload.json").read_text())
    report: dict[str, object]
    try:
        archive, fd, info = W.open_private_fd(args.oci_archive)
        try:
            archive_hash = P.binding(archive)["sha256"]
            report = H.verify(
                fd,
                info.st_size,
                args.expected_image_id,
                contract,
                archive_hash,
                args.expected_source_revision,
                args.expected_dpkg_inventory_sha256,
                args.expected_native_closure_sha256,
            )
        finally:
            os.close(fd)
    except W.INPUT_ERRORS:
        report = {
            "schema_version": H.SCHEMA,
            "valid": False,
            "findings": [{"code": "unreadable_or_incomplete_image_evidence"}],
        }
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "Habitat-Sim complete image verification "
        + ("passed" if report["valid"] else "failed")
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
