"""Qualify the RTX inference launcher against FP64, streams, replay, and cached code."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import re
import time

import torch

from attention_benchmark_backend import benchmark_environment
from attention_kernel_benchmark import _cases, _error


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--expect-helper-sha256", required=True)
    parser.add_argument("--require-cache-hit", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.expect_helper_sha256):
        parser.error("--expect-helper-sha256 requires the complete source SHA256")
    return args


def _reference(tensors, causal):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    query, key, value = (tensor.double().transpose(1, 2) for tensor in tensors)
    groups = query.shape[1] // key.shape[1]
    # Every causal case is square; unequal Q/K lengths occur only without masking.
    with sdpa_kernel(SDPBackend.MATH):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key.repeat_interleave(groups, dim=1),
            value.repeat_interleave(groups, dim=1),
            is_causal=causal,
        ).transpose(1, 2)


def _inputs(case, seed, scale):
    torch.manual_seed(seed)
    options = {"device": "cuda", "dtype": getattr(torch, case["dtype"])}
    query = (
        torch.randn(
            case["batch"], case["queries"], case["heads"], case["head_dim"], **options
        )
        * scale
    )
    key = (
        torch.randn(
            case["batch"], case["keys"], case["kv_heads"], case["head_dim"], **options
        )
        * scale
    )
    return query, key, torch.randn_like(key)


def _record(report, output_path, kind, case, actual, expected, **details):
    errors = _error(torch, actual, expected, case["dtype"])
    report["checks"].append(
        {"kind": kind, "case": case["name"], "errors": errors, **details}
    )
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(kind, case["name"], errors["relative_l2"], flush=True)


def _check_inputs(attention, case, record):
    previous = None
    # These seeds differ from the exploratory sweep, including zero and large logits.
    for seed, scale in ((101, 1), (202, 4), (303, 0)):
        tensors = _inputs(case, seed, scale)
        expected = _reference(tensors, case["causal"])
        started = time.perf_counter()
        actual = attention(*tensors, causal=case["causal"])
        torch.cuda.synchronize()
        if previous is not None and previous.data_ptr() == actual.data_ptr():
            raise AssertionError("Separate calls reused the same output allocation")
        previous = actual
        record(
            "fp64-new-input",
            case,
            actual,
            expected,
            seed=seed,
            scale=scale,
            call_seconds=time.perf_counter() - started,
        )


def _check_streams(attention, case, record):
    for index in range(2):
        tensors = _inputs(case, 404 + index, 1)
        expected = _reference(tensors, case["causal"])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = attention(*tensors, causal=case["causal"])
        stream.synchronize()
        record("cuda-stream", case, actual, expected, stream=index)


def _check_native_paths(attention, record):
    for case in (_cases()[1], _cases()[8]):
        padded = []
        for tensor in _inputs(case, 505, 1):
            storage = torch.empty(
                *tensor.shape[:-1],
                tensor.shape[-1] + 8,
                device="cuda",
                dtype=tensor.dtype,
            )
            view = storage[..., : tensor.shape[-1]]
            view.copy_(tensor)
            padded.append(view)
        actual = attention(*padded, causal=case["causal"])
        record("strided-native-path", case, actual, _reference(padded, case["causal"]))
    case = {**_cases()[1], "queries": 2048, "keys": 2048, "name": "unlisted-s2048"}
    tensors = _inputs(case, 606, 1)
    record(
        "unlisted-native-path",
        case,
        attention(*tensors, causal=False),
        _reference(tensors, False),
    )


def _check_graph(attention, record):
    case = _cases()[1]
    tensors = _inputs(case, 707, 1)
    attention(*tensors, causal=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = attention(*tensors, causal=False)
    for original, replacement in zip(tensors, _inputs(case, 808, 1), strict=True):
        original.copy_(replacement)
    graph.replay()
    torch.cuda.synchronize()
    record("graph-replay-new-values", case, actual, _reference(tensors, False))


def _prepare(args):
    import flash_attn.rtx as rtx

    digest = hashlib.sha256(Path(rtx.__file__).read_bytes()).hexdigest()
    if digest != args.expect_helper_sha256:
        raise RuntimeError(
            "Installed inference helper differs from the expected source"
        )
    if args.require_cache_hit:

        def reject_compile(*arguments, **options):
            raise AssertionError("Persistent-cache check attempted JIT compilation")

        rtx._compile_kernel = reject_compile
    report = {
        "helper_sha256": digest,
        "validation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "environment": benchmark_environment("fa4"),
        "checks": [],
        "status": "incomplete",
        "persistent_cache_reload": args.require_cache_hit,
    }
    return rtx.make_inference_attention(), report


def main():
    """Write real-GPU correctness evidence for the exact installed inference helper.

    Args:
        None; read output, expected source hash and cache mode from CLI arguments.
    Returns:
        None.
    Raises:
        AssertionError: Numerical, allocation, stream, replay or cache checks fail.
        RuntimeError: The device, source or backend does not match the request.
        OSError: Evidence cannot be written.
    """
    args = _arguments()
    attention, report = _prepare(args)
    record = partial(_record, report, args.output_path)
    with torch.inference_mode():
        for case in _cases():
            _check_inputs(attention, case, record)
            _check_streams(attention, case, record)
        if not args.require_cache_hit:
            _check_native_paths(attention, record)
        _check_graph(attention, record)
    report["status"] = "passed"
    args.output_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
