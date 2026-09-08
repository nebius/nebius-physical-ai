"""Apply the reviewed CUDA 13 layer verifier to RoboCasa's pinned wheel closure."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tarfile


CUROBO = Path(__file__).resolve().parent.parent / "curobo"


def _runtime_contract() -> dict:
    contract = json.loads((CUROBO / "runtime-payload.json").read_text())
    contract["nvshmem_notice"]["path"] = "usr/share/doc/npa-robocasa/NVSHMEM-LICENSE.txt"
    return contract


def verify_image(archive: Path, *, expected_image_id: str) -> dict:
    """Verify all inherited layers against the shared, exact CUDA wheel inventory.

    Args:
        archive: A saved archive containing exactly one inspected image.
        expected_image_id: Independently observed Docker image identity.

    Returns:
        Complete-layer verification results with RoboCasa's notice location.

    Raises:
        ValueError: The archive, image identity, or payload inventory is invalid.
        OSError: The saved archive or reviewed source inventory cannot be read.
    """
    spec = importlib.util.spec_from_file_location("curobo_payload_verifier", CUROBO / "verify_image.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    report = verifier.verify_image(
        archive, expected_image_id=expected_image_id, contract=_runtime_contract()
    )
    return {**report, "tool": "robocasa"}


def main() -> int:
    """Run the saved-image verification CLI.

    Args:
        None.

    Returns:
        Zero only when complete image-layer verification passed.

    Raises:
        OSError: The report cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-save", type=Path, required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify_image(args.docker_save, expected_image_id=args.expected_image_id)
    except (OSError, EOFError, ValueError, KeyError, TypeError, tarfile.TarError):
        report = {"tool": "robocasa", "valid": False,
                  "findings": [{"code": "unreadable_or_incomplete_image_evidence"}]}
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print("RoboCasa CUDA payload verification " + ("passed" if report["valid"] else "failed"))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
