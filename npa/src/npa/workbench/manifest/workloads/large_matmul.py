"""Large-artifact GPU workload: tiled CUDA matmul with S3-staged input.

Proves the generalization dimensions the small MVP left open:
- S3 input staging (config arrives via s3_uri, not baked in)
- a large binary artifact (full FP32 result matrix, ~1 GiB at 16384)
  transported via the descriptor's artifact_store, not container logs
- a non-trivial flag set (--tiles changes the computation shape)
- a small JSON summary carried inline for success checks
"""

import argparse
import json
import os
import time

import numpy as np
import torch


def tiled_matmul(a: torch.Tensor, b: torch.Tensor, tiles: int) -> torch.Tensor:
    n = a.shape[0]
    t = n // tiles
    out = torch.empty_like(a)
    for i in range(tiles):
        for j in range(tiles):
            acc = torch.zeros((t, t), device=a.device, dtype=a.dtype)
            for k in range(tiles):
                acc += (
                    a[i * t : (i + 1) * t, k * t : (k + 1) * t]
                    @ b[k * t : (k + 1) * t, j * t : (j + 1) * t]
                )
            out[i * t : (i + 1) * t, j * t : (j + 1) * t] = acc
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiles", type=int, default=4)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-npy", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    scale = float(config.get("scale", 1.0))

    cuda_ok = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "none"
    assert cuda_ok, "no CUDA device"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    a = torch.randn(args.size, args.size, device="cuda") * scale
    b = torch.randn(args.size, args.size, device="cuda") * scale
    torch.cuda.synchronize()
    t0 = time.time()
    c = tiled_matmul(a, b, args.tiles)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    arr = c.cpu().numpy()
    with open(args.out_npy, "wb") as f:
        np.save(f, arr)
    npy_bytes = os.path.getsize(args.out_npy)
    checksum = float(arr.sum(dtype=np.float64))

    summary = {
        "device_ok": True,
        "device": device_name,
        "size": args.size,
        "seed": args.seed,
        "tiles": args.tiles,
        "config_label": config.get("label"),
        "elapsed_s": elapsed,
        "npy_bytes": npy_bytes,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "checksum": checksum,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
