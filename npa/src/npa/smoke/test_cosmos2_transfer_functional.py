"""Cosmos2-Transfer external image golden eval (GPU runtime probe).

The transfer image is built outside this repo; this smoke proves the published
image exposes a working CUDA PyTorch stack. Replace with a transfer inference
probe when the external image ships a stable smoke entrypoint.
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
