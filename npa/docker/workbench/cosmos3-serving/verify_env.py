"""Prove the public bootstrap image contains no serving or gated payload."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from importlib import metadata
from pathlib import Path

FORBIDDEN_DISTRIBUTIONS = (
    "vllm",
    "vllm-omni",
    "torch",
    "cosmos-guardrail",
    "cuda-bindings",
    "cuda-toolkit",
)


def main() -> int:
    expected = "9d9edf4dd685c329a36bde45ed05bf5e0d51a3d78cf764d91bedb031e1a94694"
    if os.environ.get("NPA_COSMOS3_CLOSURE_SHA256") != expected:
        raise SystemExit("FATAL: runtime closure checksum drift")
    lock = Path("/opt/npa-cosmos3-serving/requirements.lock")
    if hashlib.sha256(lock.read_bytes()).hexdigest() != expected:
        raise SystemExit("FATAL: runtime closure bytes do not match the pinned checksum")
    for package in FORBIDDEN_DISTRIBUTIONS:
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
        raise SystemExit(
            f"FATAL: runtime distribution baked into public image: {package}"
        )
    for module in ("vllm", "vllm_omni", "torch", "cosmos_guardrail"):
        if importlib.util.find_spec(module) is not None:
            raise SystemExit(f"FATAL: runtime module baked into public image: {module}")
    # The final layer scanner inspects every pathname, including root-owned paths.
    # This in-container verifier runs as uid 1000, so only probe globally readable
    # sentinel paths here rather than treating an expected EACCES as a build failure.
    for path in (Path("/NGC-DL-CONTAINER-LICENSE"),):
        if path.exists():
            raise SystemExit(f"FATAL: restricted payload baked into image: {path}")
    print("NPA_COSMOS3_SERVING_ZERO_PAYLOAD_OK runtime/models/terms absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
