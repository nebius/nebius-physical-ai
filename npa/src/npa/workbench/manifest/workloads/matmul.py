"""GPU workload: real CUDA matmul producing a validated artifact.

Runs inside the container. Not a smoke test: a large deterministic matmul
(seed-controlled) whose checksum must match across identical runs.
"""

import argparse
import json
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "none"

    torch.manual_seed(args.seed)
    if cuda_ok:
        torch.cuda.manual_seed_all(args.seed)
        a = torch.randn(args.size, args.size, device="cuda")
        b = torch.randn(args.size, args.size, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        c = a @ b
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        checksum = float(c.double().sum().item())
    else:
        elapsed, checksum = 0.0, 0.0

    result = {
        "device_ok": cuda_ok,
        "device": device_name,
        "size": args.size,
        "seed": args.seed,
        "elapsed_s": elapsed,
        "checksum": checksum,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
