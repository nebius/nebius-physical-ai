"""Fail-closed source attestation executed inside every production image."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _attest_isaac_cache() -> dict[str, str]:
    """Attest the operator-fetched closure consumed by an Isaac GPU Job."""

    if os.environ.get("NPA_ISAAC_CACHE_READONLY", "").strip() != "1":
        return {}
    if os.environ.get("NPA_ISAAC_BOOTSTRAP_OFFLINE", "").strip() != "1":
        raise RuntimeError("read-only Isaac cache consumption must also be offline")

    root = Path(os.environ.get("NPA_ISAAC_CACHE_DIR", "/opt/isaac-cache")).resolve()
    current = root / "current"
    if not current.is_symlink():
        raise RuntimeError(f"Isaac cache current pointer is not a symlink: {current}")
    try:
        target = current.resolve(strict=True)
        target.relative_to(root / "v")
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(
            "Isaac cache current pointer escapes or is incomplete"
        ) from exc
    if target.parent != root / "v" or not _SHA256_RE.fullmatch(target.name):
        raise RuntimeError("Isaac cache target is not a full content-addressed tree")
    if not (target / ".complete").is_file():
        raise RuntimeError("Isaac cache tree has no atomic completion marker")
    if not (target / "venv" / "bin" / "python").is_file():
        raise RuntimeError("Isaac cache tree has no runtime interpreter")

    manifest_path = target / "MANIFEST.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Isaac cache manifest is missing or malformed") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("Isaac cache manifest must be a JSON object")
    if manifest.get("format") != "npa_isaac_runtime_cache_v1":
        raise RuntimeError("Isaac cache manifest format is unsupported")
    if manifest.get("cache_stamp") != target.name:
        raise RuntimeError("Isaac cache manifest does not attest its target stamp")

    bootstrap_path = Path(
        os.environ.get("NPA_ISAAC_BOOTSTRAP_PATH", "/opt/npa/bin/isaac-bootstrap")
    )
    try:
        bootstrap_sha = hashlib.sha256(bootstrap_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError("Isaac runtime bootstrap is missing") from exc
    if manifest.get("bootstrap_sha256") != bootstrap_sha:
        raise RuntimeError("Isaac cache was warmed by a different bootstrap")

    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    return {
        "isaac_cache_pvc": os.environ.get("NPA_SIM2REAL_ISAAC_CACHE_PVC", "").strip(),
        "isaac_cache_stamp": target.name,
        "isaac_cache_manifest_sha256": manifest_sha,
        "isaac_bootstrap_sha256": bootstrap_sha,
        "isaac_cache_mode": "offline-readonly",
    }


def attest_runtime_source() -> dict[str, Any]:
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
    return {
        "source_sha": actual,
        "runtime_image": runtime_image,
        **_attest_isaac_cache(),
    }


def main() -> None:
    print(json.dumps({"runtime_attestation": attest_runtime_source()}, sort_keys=True))


if __name__ == "__main__":
    main()
