"""Cosmos2-Transfer golden eval (GPU runtime probe).

The transfer runtime installs PyTorch in ``/opt/cosmos/cosmos-transfer2.5/.venv``
(Python 3.10 + cu128). The golden-eval wrapper image bakes that venv; the smoke
script discovers the venv python before running the CUDA matmul probe.
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("[FAIL] cuda available: torch.cuda.is_available() is False")
        return 1
    device = torch.cuda.get_device_name(0)
    tensor = torch.ones((1024, 1024), device="cuda")
    result = float(tensor.sum().item())
    print(f"[PASS] cuda available: {device}")
    print(f"[PASS] cuda matmul: sum={result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
