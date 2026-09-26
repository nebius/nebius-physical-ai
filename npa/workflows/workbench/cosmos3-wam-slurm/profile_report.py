"""Summarize real CUDA traces without treating overlapping kernel time as wall time."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re


def _union(intervals):
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def _category(name):
    for category, pattern in (
        ("collectives", r"nccl"),
        ("attention", r"fmha|flash|attention|attn"),
        ("convolution", r"conv|cudnn"),
        ("matrix_multiply", r"gemm|matmul|cutlass|mma"),
    ):
        if re.search(pattern, name, re.IGNORECASE):
            return category
    return "other"


def _trace(path):
    with gzip.open(path, "rt") as stream:
        trace = json.load(stream)
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    steps = [
        event for event in events if event.get("name", "").startswith("ProfilerStep#")
    ]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    if len(steps) < 2 or not kernels:
        raise ValueError(
            "profile must contain at least two real steps and CUDA kernels"
        )
    start = min(event["ts"] for event in steps)
    stop = max(event["ts"] + event["dur"] for event in steps)
    if stop <= start:
        raise ValueError("invalid profiler window")
    return steps, kernels, start, stop


def _kernel_metrics(kernels, start, stop):
    categories, intervals = defaultdict(float), defaultdict(list)
    counts = defaultdict(int)
    for event in kernels:
        begin = max(start, event["ts"])
        end = min(stop, event["ts"] + event["dur"])
        if end <= begin:
            continue
        device = str(event.get("args", {}).get("device", "unknown"))
        intervals[device].append((begin, end))
        category = _category(event["name"])
        categories[category] += (end - begin) / 1e6
        counts[category] += 1
    if not categories:
        raise ValueError("no CUDA kernels overlap the selected profiler steps")
    return {
        "kernel_duration_seconds_by_category": dict(categories),
        "kernel_counts_by_category": dict(counts),
        "observed_kernel_busy_seconds_by_device": {
            device: _union(values) / 1e6 for device, values in intervals.items()
        },
    }


def _summarize(path):
    steps, kernels, start, stop = _trace(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "trace_sha256": digest,
        "trace_bytes": path.stat().st_size,
        "profiler_steps": [event["name"] for event in steps],
        "profile_window_seconds": (stop - start) / 1e6,
        **_kernel_metrics(kernels, start, stop),
    }


def _main(args):
    settings = json.loads((args.run_dir / "run.json").read_text())
    if not settings["profile"]:
        raise ValueError("requires a separate profiled run")
    job = args.run_dir / "output/cosmos3_wam/libero_10" / settings["name"]
    profiles = {}
    for node in range(settings["nodes"]):
        rank = 8 * node
        receipt = json.loads((args.run_dir / f"node-{node}.finished.json").read_text())
        if receipt["returncode"] != 0:
            raise ValueError("profiled training did not finish successfully")
        path = job / "torch_trace/iteration_100" / f"rank{rank}_trace.json.gz"
        profiles[str(rank)] = _summarize(path)
    result = {
        "schema": "npa.cosmos3.wam-profile.v1",
        "ranks": profiles,
        "interpretation": (
            "Kernel categories use names and may overlap in time. Summed kernel "
            "durations are not wall-time fractions or exposed communication stalls. "
            "Busy time is the union of observed kernel intervals per device; "
            "it excludes memory-copy activities. VAE timing comes from native logs."
        ),
    }
    (args.run_dir / "profile-summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    _main(parser.parse_args())
