"""Check real B200 CUDA forward/backward execution without claiming WAM training."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F


def probe():
    from cosmos_framework.utils.generator.fused_adam import FusedAdam

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    if "B200" not in properties.name:
        raise RuntimeError("this qualification requires a real B200")
    torch.manual_seed(42)
    tensors = [
        torch.randn(1, 4, 32, 64, device="cuda", dtype=torch.bfloat16).requires_grad_()
        for _ in range(3)
    ]
    output = F.scaled_dot_product_attention(*tensors)
    reference = F.scaled_dot_product_attention(
        *(tensor.detach().float().cpu() for tensor in tensors)
    )
    torch.testing.assert_close(output.float().cpu(), reference, atol=0.03, rtol=0.03)
    output.float().square().mean().backward()
    if not all(
        tensor.grad is not None
        and torch.isfinite(tensor.grad).all().item()
        and tensor.grad.abs().sum().item() > 0
        for tensor in tensors
    ):
        raise RuntimeError("CUDA attention backward produced invalid gradients")
    parameter = torch.nn.Parameter(torch.zeros(64, device="cuda", dtype=torch.float32))
    optimizer = FusedAdam(
        [parameter], lr=5e-5, eps=1e-8, master_weights=False, capturable=True
    )
    parameter.sum().backward()
    optimizer.step()
    if not torch.isfinite(parameter).all().item() or not (parameter < 0).all().item():
        raise RuntimeError("native FusedAdam did not apply a finite CUDA update")
    torch.cuda.synchronize()
    drivers = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    return {
        "schema": "npa.cosmos3.wam-runtime-probe.v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "gpu_name": properties.name,
        "visible_gpu_count": torch.cuda.device_count(),
        "tested_device": device,
        "gpu_memory_bytes": properties.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "driver_versions": sorted(set(drivers)),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "nccl_version": torch.cuda.nccl.version(),
        "bf16_attention_forward_matches_fp32_reference": True,
        "bf16_attention_backward_finite_nonzero": True,
        "native_fused_adam_fp32_parameter_update": True,
        "forward_max_absolute_error": (output.float().cpu() - reference)
        .abs()
        .max()
        .item(),
        "scope": "one-device CUDA kernel qualification; no WAM training or multi-node claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe()
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
