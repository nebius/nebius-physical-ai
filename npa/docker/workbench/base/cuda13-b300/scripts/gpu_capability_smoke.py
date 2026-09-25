#!/usr/bin/env python
"""Validate native CUDA coverage and real FA4 outputs and gradients on a GPU.

Run inside the CUDA 13 base image. Dense, grouped-query and variable-length
cases cover FP16/BF16, head dimensions 64/128 and causal/noncausal attention.
Every case compares outputs and dQ/dK/dV against independent FP64 attention.
Unsupported features and other model shapes require separate qualification.

Usage:
  python gpu_capability_smoke.py --expect-capability 12.0
  python gpu_capability_smoke.py --expect-capability 10.0 --json-output report.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from importlib import metadata
from pathlib import Path


def _parse_sass_arch(flag: str) -> tuple[int, int] | None:
    if not flag.startswith("sm_"):
        return None
    digits = flag.removeprefix("sm_")
    if len(digits) < 2 or not digits.isdigit():
        return None
    return int(digits[:-1]), int(digits[-1])


def covering_sass_arch(
    capability: tuple[int, int], arch_flags: list[str]
) -> tuple[int, int] | None:
    """Find compatible native SASS without crossing a CUDA major.

    Args:
        capability: Device compute capability, as major and minor.
        arch_flags: Architecture flags reported by the installed Torch wheel.
    Returns:
        The highest compatible SASS capability, or None when only PTX is usable.
    Raises:
        None.
    """
    available = [arch for flag in arch_flags if (arch := _parse_sass_arch(flag))]
    compatible = [
        arch
        for arch in available
        if arch[0] == capability[0] and arch[1] <= capability[1]
    ]
    return max(compatible, default=None)


def _check_device(torch, expected: str) -> dict:
    capability = torch.cuda.get_device_capability()
    flags = torch.cuda.get_arch_list()
    if expected and capability != tuple(int(part) for part in expected.split(".")):
        raise ValueError(f"Expected capability {expected}, landed on {capability}")
    if covering_sass_arch(capability, flags) is None:
        raise ValueError(f"No compatible native SASS for {capability} in {flags}")
    return {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(capability),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "wheel_arch_flags": flags,
        "packages": {
            name: metadata.version(name)
            for name in ("flash-attn-4", "nvidia-cutlass-dsl", "quack-kernels")
        },
    }


def _check_controls(torch) -> None:
    matrix = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    product = matrix @ matrix
    query = torch.randn(2, 4, 129, 64, device="cuda", dtype=torch.bfloat16)
    output = torch.nn.functional.scaled_dot_product_attention(query, query, query)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(product).all() and torch.isfinite(output).all()):
        raise AssertionError("Control matmul or Torch SDPA produced non-finite values")


def _reference_attention(torch, query, key, value, causal: bool):
    # FA causal cross-attention aligns at the bottom right; Torch SDPA's
    # is_causal=True uses a different alignment when Q/K lengths differ.
    repeats = query.shape[2] // key.shape[2]
    key = key.repeat_interleave(repeats, dim=2)
    value = value.repeat_interleave(repeats, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / query.shape[-1] ** 0.5
    if causal:
        rows = torch.arange(query.shape[1], device=query.device)[:, None]
        columns = torch.arange(key.shape[1], device=query.device)[None, :]
        mask = columns <= rows + key.shape[1] - query.shape[1]
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), value)


def _reference_varlen(torch, tensors, lengths, causal: bool):
    query, key, value = tensors
    queries = query.split(lengths[0])
    keys = key.split(lengths[1])
    values = value.split(lengths[1])
    outputs = [
        _reference_attention(torch, q[None], k[None], v[None], causal)[0]
        for q, k, v in zip(queries, keys, values, strict=True)
    ]
    return torch.cat(outputs)


def _compare_tensor(torch, actual, reference, dtype: str) -> dict:
    actual = actual.double()
    if not bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()):
        raise AssertionError("Attention output or gradient contains non-finite values")
    difference = actual - reference
    relative = (difference.norm() / reference.norm().clamp_min(1e-12)).item()
    tolerance = 0.02 if dtype == "bfloat16" else 0.003
    torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    relative_limit = 0.01 if dtype == "bfloat16" else 0.002
    if relative > relative_limit:
        raise AssertionError(f"Relative L2 error {relative} exceeds {relative_limit}")
    return {
        "max_abs_error": difference.abs().max().item(),
        "relative_l2_error": relative,
    }


def _make_inputs(torch, kind: str, dtype: str, head_dim: int):
    query_lengths, key_lengths = (17, 65, 129, 33), (31, 97, 129, 65)
    query_shape, key_shape = (2, 129, 4, head_dim), (2, 193, 4, head_dim)
    if kind == "gqa":
        key_shape = (2, 193, 2, head_dim)
    if kind == "varlen":
        query_shape = (sum(query_lengths), 4, head_dim)
        key_shape = (sum(key_lengths), 4, head_dim)
    tensors = tuple(
        torch.randn(
            shape, device="cuda", dtype=getattr(torch, dtype), requires_grad=True
        )
        for shape in (query_shape, key_shape, key_shape)
    )
    return tensors, (query_lengths, key_lengths)


def _cumulative_lengths(torch, lengths):
    return torch.tensor(
        [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32
    )


def _attention_functions():
    # Use the FA4 namespace explicitly: a root flash_attn import can resolve FA2.
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

    return flash_attn_func, flash_attn_varlen_func


def _run_attention(torch, functions, tensors, lengths, kind: str, causal: bool):
    dense, varlen = functions
    # Disable only packing and split-KV optimizations, never change the backend.
    options = {"causal": causal, "pack_gqa": False, "num_splits": 1}
    if kind != "varlen":
        return dense(*tensors, **options)
    return varlen(
        *tensors,
        cu_seqlens_q=_cumulative_lengths(torch, lengths[0]),
        cu_seqlens_k=_cumulative_lengths(torch, lengths[1]),
        max_seqlen_q=max(lengths[0]),
        max_seqlen_k=max(lengths[1]),
        **options,
    )


def _check_attention_case(torch, functions, kind, dtype, head_dim, causal) -> dict:
    tensors, lengths = _make_inputs(torch, kind, dtype, head_dim)
    reference_inputs = tuple(
        tensor.detach().double().requires_grad_() for tensor in tensors
    )
    output = _run_attention(torch, functions, tensors, lengths, kind, causal)
    if isinstance(output, tuple):
        output = output[0]
    if kind == "varlen":
        reference = _reference_varlen(torch, reference_inputs, lengths, causal)
    else:
        reference = _reference_attention(torch, *reference_inputs, causal)
    upstream_gradient = torch.randn_like(output)
    gradients = torch.autograd.grad(output, tensors, upstream_gradient)
    reference_gradients = torch.autograd.grad(
        reference, reference_inputs, upstream_gradient.double()
    )
    torch.cuda.synchronize()
    metrics = {"output": _compare_tensor(torch, output, reference, dtype)}
    for name, actual, expected in zip(
        ("dQ", "dK", "dV"), gradients, reference_gradients, strict=True
    ):
        metrics[name] = _compare_tensor(torch, actual, expected, dtype)
    return {
        "kind": kind,
        "dtype": dtype,
        "head_dim": head_dim,
        "causal": causal,
        "status": "passed",
        "metrics": metrics,
    }


def _run_checks(expected: str, report: dict) -> None:
    import torch

    torch.manual_seed(0)
    report["environment"] = _check_device(torch, expected)
    _check_controls(torch)
    report["controls"] = "passed"
    functions = _attention_functions()
    cases = itertools.product(
        ("dense", "gqa", "varlen"), ("float16", "bfloat16"), (64, 128), (False, True)
    )
    for kind, dtype, head_dim, causal in cases:
        label = f"{kind}/{dtype}/d{head_dim}/causal={causal}"
        print(f"Checking {label}", flush=True)
        report["active_case"] = label
        report["cases"].append(
            _check_attention_case(torch, functions, kind, dtype, head_dim, causal)
        )
    report.pop("active_case", None)


def main(argv: list[str] | None = None) -> int:
    """Run strict GPU checks and optionally write their measured results.

    Args:
        argv: Command arguments, or None to read the process arguments.
    Returns:
        Zero only when every device, control, output and gradient check passes.
    Raises:
        OSError: The requested JSON output cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-capability", default="", metavar="CC")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    report = {
        "schema_version": 1,
        "backend": "flash_attn.cute",
        "status": "failed",
        "cases": [],
    }
    try:
        _run_checks(args.expect_capability, report)
        report["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - a smoke must report every runtime failure
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    passed = report["status"] == "passed"
    print(f"FA4 cases passed: {len(report['cases'])}/24")
    print("GPU_CAPABILITY_SMOKE_OK" if passed else "GPU_CAPABILITY_SMOKE_FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
