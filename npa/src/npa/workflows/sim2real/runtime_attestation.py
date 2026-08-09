"""Fail-closed source attestation executed inside every production image."""

from __future__ import annotations

import json
import os
import re


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def attest_runtime_source() -> dict[str, str]:
    """Prove that the running image was built from the controller's source SHA."""

    expected = os.environ.get("NPA_SIM2REAL_SOURCE_SHA", "").strip().lower()
    actual = os.environ.get("NPA_IMAGE_SOURCE_SHA", "").strip().lower()
    if not _SHA_RE.fullmatch(expected):
        raise RuntimeError("NPA_SIM2REAL_SOURCE_SHA must be an exact 40-character SHA")
    if not _SHA_RE.fullmatch(actual):
        raise RuntimeError(
            "image is missing its exact NPA_IMAGE_SOURCE_SHA attestation"
        )
    if actual != expected:
        raise RuntimeError(
            f"runtime source mismatch: expected={expected} image={actual}"
        )
    runtime_image = os.environ.get("NPA_SIM2REAL_RUNTIME_IMAGE", "").strip()
    if runtime_image and "@sha256:" not in runtime_image:
        raise RuntimeError("NPA_SIM2REAL_RUNTIME_IMAGE is not immutable")
    return {"source_sha": actual, "runtime_image": runtime_image}


def main() -> None:
    print(json.dumps({"runtime_attestation": attest_runtime_source()}, sort_keys=True))


if __name__ == "__main__":
    main()
