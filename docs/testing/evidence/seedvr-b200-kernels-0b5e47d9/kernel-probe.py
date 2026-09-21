"""Prospective B200 kernel controls; not authorized for execution by this file."""

import csv
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
from pathlib import Path

import torch
from apex.normalization import FusedLayerNorm, FusedRMSNorm
from flash_attn import flash_attn_varlen_func


def hardware():
    fields = "name,uuid,memory.total,driver_version,mig.mode.current"
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    assert len(rows) == torch.cuda.device_count() == 1
    name, uuid, memory, driver, mig = [value.strip() for value in rows[0]]
    assert "B200" in name and mig == "Disabled"
    assert torch.cuda.get_device_capability(0) == (10, 0)
    assert "B200" in torch.cuda.get_device_name(0)
    return {
        "name": name,
        "uuid": uuid,
        "memory_mib": int(memory),
        "driver": driver,
        "mig": mig,
        "compute": [10, 0],
    }


def close_record(name, actual, expected):
    torch.cuda.synchronize()
    assert actual.is_cuda and torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.02)
    difference = (actual.float() - expected.float()).abs()
    return {
        "name": name,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "finite": True,
        "reference_tolerance": {"rtol": 0.02, "atol": 0.02},
        "pass": True,
    }


def torch_controls():
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    matmul = close_record("torch_matmul_bf16", a @ b, a.float() @ b.float())
    x = torch.randn(1, 8, 16, 16, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(8, 8, 3, 3, device="cuda", dtype=torch.bfloat16)
    assert torch.backends.cudnn.is_available() and torch.backends.cudnn.enabled
    actual = torch.ops.aten.cudnn_convolution.default(
        x,
        w,
        [1, 1],
        [1, 1],
        [1, 1],
        1,
        False,
        False,
        False,
    )
    expected = torch.nn.functional.conv2d(x.float(), w.float(), padding=1)
    return [matmul, close_record("cudnn_conv2d_bf16", actual, expected)]


def flash_control():
    # Directly exercise the mandatory varlen kernel used by SeedVR's attention.
    q, k, v = [
        torch.randn(80, 4, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    cumulative = torch.tensor([0, 32, 80], device="cuda", dtype=torch.int32)
    actual = flash_attn_varlen_func(
        q,
        k,
        v,
        cumulative,
        cumulative,
        48,
        48,
        dropout_p=0.0,
        causal=False,
    )
    references = []
    for start, end in ((0, 32), (32, 80)):
        qs, ks, vs = [value[start:end].float().transpose(0, 1) for value in (q, k, v)]
        scores = (qs @ ks.transpose(-1, -2)) / (64**0.5)
        references.append((scores.softmax(-1) @ vs).transpose(0, 1))
    return close_record("flash_attn_varlen_bf16", actual, torch.cat(references))


def apex_controls():
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    layer = FusedLayerNorm(64, eps=1e-5).cuda().to(torch.bfloat16)
    rms = FusedRMSNorm(64, eps=1e-5).cuda().to(torch.bfloat16)
    expected_ln = torch.nn.functional.layer_norm(
        x.float(),
        (64,),
        layer.weight.float(),
        layer.bias.float(),
        layer.eps,
    )
    expected_rms = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + rms.eps
    )
    expected_rms = expected_rms * rms.weight.float()
    return [
        close_record("apex_fused_layer_norm_bf16", layer(x), expected_ln),
        close_record("apex_fused_rms_norm_bf16", rms(x), expected_rms),
    ]


def main():
    assert (
        os.environ["EXPECTED_PROBE_SHA256"]
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    assert (
        Path("/opt/npa-source-revision").read_text().strip()
        == "0b5e47d97414a2fe46d6184efe9b4423b2b52198"
    )
    assert torch.__version__ == "2.13.0+cu130" and torch.version.cuda == "13.0"
    assert importlib.metadata.version("flash-attn") == "2.8.3.post1"
    identity = hardware()
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    results = torch_controls() + [flash_control()] + apex_controls()
    torch.cuda.synchronize()
    report = {
        "schema": "seedvr.b200.kernel-probe.v1",
        "hardware": identity,
        "baked_source_revision": Path("/opt/npa-source-revision").read_text().strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "flash_attn": importlib.metadata.version("flash-attn"),
        "apex": importlib.metadata.version("apex"),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": 666,
        "controls": results,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "image_binding": "Verify separately against actual Pod imageID and accepted immutable OCI source labels; this script does not infer image identity from caller-supplied environment values.",
        "claim": "Native kernel compatibility only; no model, dataset, restoration quality, throughput, or other GPU comparison.",
    }
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
