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


def _validate_source_contract(
    contract: dict[str, object], manifest: dict[str, object]
) -> None:
    rows = manifest["source"]["required_projection_files"]
    expected = {f"/usr/src/habitat-sim/{row['path']}": row["sha256"] for row in rows}
    prefix = "/usr/src/habitat-sim/data/pbr/"
    observed = {
        path: digest
        for path, digest in contract["required_final_file_sha256"].items()
        if path.startswith(prefix)
    }
    if observed != expected:
        raise ValueError("runtime PBR byte contract differs from source manifest")
    required_paths = set(contract["required_final_paths"])
    if not expected.keys() <= required_paths:
        raise ValueError("runtime PBR path contract is incomplete")
    notice = "/usr/share/doc/npa-habitat-sim/THIRD_PARTY_NOTICES.md"
    if (
        notice not in required_paths
        or notice not in contract["required_final_file_sha256"]
    ):
        raise ValueError("runtime PBR notice contract is incomplete")


def _load_contract() -> dict[str, object]:
    package = Path(__file__).resolve().parent
    contract = json.loads((package / "runtime-payload.json").read_text())
    manifest = json.loads((package / "source-manifest.json").read_text())
    _validate_source_contract(contract, manifest)
    return contract


def main(argv: list[str] | None = None) -> int:
    """Run the complete Habitat OCI verification command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oci-archive", type=Path, required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-dpkg-inventory-sha256", required=True)
    parser.add_argument("--expected-python-venv-inventory-sha256", required=True)
    parser.add_argument("--expected-native-closure-sha256", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    report: dict[str, object]
    try:
        contract = _load_contract()
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
                args.expected_python_venv_inventory_sha256,
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
