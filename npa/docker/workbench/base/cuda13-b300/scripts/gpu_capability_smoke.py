#!/usr/bin/env python
"""Real capability smoke for the CUDA 13 Blackwell base image.

Runs *inside* a workbench container on a real GPU. An import check is not
enough: ``import flash_attn`` succeeds on every Blackwell part, but the
flash-attn-4 CuTe forward kernel only executes on TMA-capable architectures.
That gap went unnoticed for months because the golden eval only imported.

Checks, in order:

1. The device is the architecture we meant to validate, and the wheel carries
   native SASS for it (so kernels are not silently PTX-JIT-ing).
2. A bf16 tensor-core matmul produces finite results. This is the control: if
   it fails, the GPU or the wheel is wrong, not the kernel under test.
3. torch SDPA runs, as a second control on the attention shape itself.
4. The flash-attn-4 CuTe forward kernel executes and matches SDPA.

Step 4 is expected to fail on ``sm_120`` (RTX PRO 6000): flash-attn-4's CuTe
kernel partitions its epilogue with ``cpasync.tma_partition``, and the TMA copy
atom is unavailable on workstation Blackwell. Pass ``--allow-no-tma`` to record
that as a known gap rather than a failure; never pass it for a datacenter part
(``sm_90``/``sm_100``/``sm_103``), where TMA exists and a failure is real.

Usage:
  python gpu_capability_smoke.py --expect-capability 10.0
  python gpu_capability_smoke.py --expect-capability 12.0 --allow-no-tma
"""

from __future__ import annotations

import argparse
import sys

# Architectures with the Tensor Memory Accelerator that flash-attn-4 CuTe needs.
TMA_CAPABLE = {(9, 0), (10, 0), (10, 3)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--expect-capability",
        default="",
        metavar="CC",
        help="compute capability the device must report, e.g. 10.0",
    )
    parser.add_argument(
        "--allow-no-tma",
        action="store_true",
        help="treat a missing-TMA flash-attn failure as a known gap (sm_120 only)",
    )
    args = parser.parse_args(argv)

    import torch

    arch_flags = (torch._C._cuda_getArchFlags() or "").split()
    capability = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    device_arch = f"sm_{capability[0]}{capability[1]}"

    print(f"torch {torch.__version__} (cuda {torch.version.cuda})")
    print(f"wheel arch flags: {arch_flags}")
    print(f"device: {name}")
    print(f"capability: {capability} -> {device_arch}")

    failures: list[str] = []

    if args.expect_capability:
        expected = tuple(int(part) for part in args.expect_capability.split("."))
        if capability != expected:
            failures.append(f"expected capability {expected}, landed on {capability}")

    if device_arch not in arch_flags:
        failures.append(
            f"no native SASS for {device_arch} in {arch_flags}; kernels would PTX-JIT"
        )
    else:
        print(f"native SASS present for {device_arch}")

    torch.manual_seed(0)

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    product = a @ a
    torch.cuda.synchronize()
    if not bool(torch.isfinite(product).all()):
        failures.append("control bf16 matmul produced non-finite values")
    else:
        print(f"control bf16 matmul ok {tuple(product.shape)}")

    shape = (2, 256, 8, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
    ).transpose(1, 2)
    torch.cuda.synchronize()
    print(f"control torch SDPA ok {tuple(reference.shape)}")

    import flash_attn
    from flash_attn import flash_attn_func

    print(f"flash-attn-4 {flash_attn.__version__}")
    try:
        out = flash_attn_func(q, k, v)
        torch.cuda.synchronize()
        error = (out.float() - reference).abs().max().item()
        print(f"flash_attn_func ok {tuple(out.shape)}, max abs error vs SDPA {error:.5f}")
        if not (error < 0.05):
            failures.append(f"flash-attn output diverges from SDPA (max abs error {error})")
    except Exception as exc:  # noqa: BLE001 - the failure mode is the result
        known_tma_gap = args.allow_no_tma and capability not in TMA_CAPABLE
        detail = f"{type(exc).__name__}: {exc}"
        if known_tma_gap:
            print(
                f"flash_attn_func unavailable on {device_arch} (known gap): {detail}. "
                "flash-attn-4's CuTe kernel needs TMA, which datacenter parts have "
                "and workstation Blackwell does not."
            )
        else:
            failures.append(f"flash_attn_func failed on {device_arch}: {detail}")

    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures:
        print("GPU_CAPABILITY_SMOKE_FAILED")
        return 1
    print("GPU_CAPABILITY_SMOKE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
