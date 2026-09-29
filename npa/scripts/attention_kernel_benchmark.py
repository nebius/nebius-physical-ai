"""Measure standalone FA2/FA4 kernels and optional FA4 inference tile candidates."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time

from attention_benchmark_backend import attention_backend, benchmark_environment


def _cases():
    cases = []
    for sequence, heads in ((1024, 20), (1536, 20), (4096, 10), (6144, 10)):
        for keys in (77, sequence):
            cases.append(
                {
                    "name": f"sdxl-q{sequence}-k{keys}",
                    "batch": 2,
                    "queries": sequence,
                    "keys": keys,
                    "heads": heads,
                    "kv_heads": heads,
                    "head_dim": 64,
                    "dtype": "float16",
                    "causal": False,
                    "backward": False,
                }
            )
    for sequence in (1024, 4096):
        for kv_heads in (4, 16):
            cases.append(
                {
                    "name": f"transformer-s{sequence}-kv{kv_heads}",
                    "batch": 2,
                    "queries": sequence,
                    "keys": sequence,
                    "heads": 16,
                    "kv_heads": kv_heads,
                    "head_dim": 128,
                    "dtype": "bfloat16",
                    "causal": True,
                    "backward": True,
                }
            )
    return cases


def _inputs(torch, case, backward):
    torch.manual_seed(17)
    common = {
        "device": "cuda",
        "dtype": getattr(torch, case["dtype"]),
        "requires_grad": backward,
    }
    query = torch.randn(
        case["batch"], case["queries"], case["heads"], case["head_dim"], **common
    )
    key = torch.randn(
        case["batch"], case["keys"], case["kv_heads"], case["head_dim"], **common
    )
    return query, key, torch.randn_like(key, requires_grad=backward)


def _reference(torch, tensors, causal):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    query, key, value = (tensor.float().transpose(1, 2) for tensor in tensors)
    repeats = query.shape[1] // key.shape[1]
    with sdpa_kernel(SDPBackend.MATH):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key.repeat_interleave(repeats, dim=1),
            value.repeat_interleave(repeats, dim=1),
            is_causal=causal,
        ).transpose(1, 2)


def _error(torch, actual, expected, dtype):
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("Non-finite output or gradient")
    difference = actual.float() - expected.float()
    relative = (difference.norm() / expected.float().norm().clamp_min(1e-12)).item()
    tolerance = 0.02 if dtype == "bfloat16" else 0.003
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=tolerance, atol=tolerance
    )
    if relative > (0.01 if dtype == "bfloat16" else 0.002):
        raise AssertionError(f"Relative L2 error too large: {relative}")
    return {"relative_l2": relative, "max_abs": difference.abs().max().item()}


def _correctness(torch, function, tensors, case, backward):
    reference_inputs = tuple(
        t.detach().float().requires_grad_(backward) for t in tensors
    )
    context = nullcontext() if backward else torch.inference_mode()
    with context:
        actual = function(*tensors, causal=case["causal"])
        reference = _reference(torch, reference_inputs, case["causal"])
        errors = {"output": _error(torch, actual, reference, case["dtype"])}
        if backward:
            gradient = torch.randn_like(actual)
            gradients = torch.autograd.grad(actual, tensors, gradient)
            references = torch.autograd.grad(
                reference, reference_inputs, gradient.float()
            )
            for name, value, expected in zip(
                ("dQ", "dK", "dV"), gradients, references, strict=True
            ):
                errors[name] = _error(torch, value, expected, case["dtype"])
    return errors


def _timings(torch, operation, warmup, iterations, repeats):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        started = time.perf_counter()
        start.record()
        for _ in range(iterations):
            operation()
        stop.record()
        stop.synchronize()
        samples.append(
            {
                "cuda_ms": start.elapsed_time(stop) / iterations,
                "wall_ms": (time.perf_counter() - started) * 1000 / iterations,
            }
        )
    return samples


def _profile(torch, operation):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as trace:
        operation()
        torch.cuda.synchronize()
    kernels = sorted(
        {
            event.name
            for event in trace.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        }
    )
    if not kernels:
        raise RuntimeError("No CUDA kernels captured")
    return kernels


def _measure(torch, function, case, args, backward):
    tensors = _inputs(torch, case, backward)
    errors = _correctness(torch, function, tensors, case, backward)
    gradient = torch.randn_like(tensors[0])

    def operation():
        output = function(*tensors, causal=case["causal"])
        if backward:
            torch.autograd.grad(output, tensors, gradient)

    with nullcontext() if backward else torch.inference_mode():
        samples = _timings(torch, operation, args.warmup, args.iterations, args.repeats)
        kernels = _profile(torch, operation)
    return {
        "case": case,
        "mode": "forward+backward" if backward else "forward",
        "correctness": errors,
        "samples": samples,
        "kernels": kernels,
        "median_cuda_ms": statistics.median(s["cuda_ms"] for s in samples),
        "median_wall_ms": statistics.median(s["wall_ms"] for s in samples),
    }


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fa2", "fa4"), required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    tuning = parser.add_mutually_exclusive_group()
    tuning.add_argument("--tile", choices=("64x64", "64x128", "128x64", "128x128"))
    tuning.add_argument("--tuning", choices=("rtx6000-inference",))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1:
        parser.error("warmup, iterations and repeats must be positive")
    if (args.tile or args.tuning) and args.backend != "fa4":
        parser.error("Inference tuning requires --backend fa4")
    return args


def main():
    """Write correctness, kernel identity and steady-state timing measurements.

    Args:
        None; configuration comes from command-line arguments.
    Returns:
        None.
    Raises:
        RuntimeError: The GPU, backend or profiler qualification fails.
        AssertionError: An output or gradient fails the independent reference.
    """
    import torch

    args = _arguments()
    tile = tuple(map(int, args.tile.split("x"))) if args.tile else None
    function = attention_backend(args.backend, tile, args.tuning)
    report = {
        "schema_version": 1,
        "backend": args.backend,
        "tile": tile,
        "tuning": args.tuning,
        "environment": benchmark_environment(args.backend),
        "results": [],
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "status": "incomplete",
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    for case in _cases():
        modes = (
            (False, True)
            if case["backward"] and not (tile or args.tuning)
            else (False,)
        )
        for backward in modes:
            result = _measure(torch, function, case, args, backward)
            report["results"].append(result)
            args.output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(case["name"], result["mode"], result["median_cuda_ms"], flush=True)
    report["status"] = "passed"
    args.output_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
