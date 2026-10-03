"""Install the pinned XR1 policy runtime beside Isaac using a verified SM120 wheel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .runtime import _cuda_proof, _install
from .storage import _relative, _sha256


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-path", type=Path, required=True)
    parser.add_argument("--wheel-root", type=Path, required=True)
    args = parser.parse_args()
    import torch

    if sys.version_info[:2] != (3, 11) or torch.__version__ != "2.8.0+cu128":
        raise ValueError(
            "The policy runtime requires Python 3.11 and PyTorch 2.8.0+cu128"
        )
    receipt = json.loads((args.wheel_root / "input-receipt.json").read_text())
    wheels = {
        name: identity for name, identity in receipt.items() if name.endswith(".whl")
    }
    if len(wheels) != 1:
        raise ValueError("Expected one verified FlashAttention SM120 wheel")
    for name, identity in wheels.items():
        if _sha256(args.wheel_root / _relative(name)) != identity["sha256"]:
            raise ValueError("Runtime wheel differs from its S3 input receipt")
    args.work_path.mkdir(parents=True, exist_ok=False)
    _install(args.work_path, args.wheel_root / "wheels")
    proof = {
        "scope": "policy inference runtime",
        **_cuda_proof(require_optimizer=False),
    }
    (args.work_path / "runtime-proof.json").write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    _main()
