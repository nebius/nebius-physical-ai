"""Export a CUDA activity timeline from completed, hash-linked native profiles."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path


def _load_reporter(recipe):
    path = recipe / "profile_report.py"
    spec = importlib.util.spec_from_file_location("wam_profile_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


def _merge(intervals):
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(stop, merged[-1][1])
        else:
            merged.append([start, stop])
    return merged


def _intervals(kernels, start, stop, reporter):
    grouped = defaultdict(list)
    for event in kernels:
        first, last = max(start, event["ts"]), min(stop, event["ts"] + event["dur"])
        if last <= first:
            continue
        device = event.get("args", {}).get("device")
        if type(device) is not int or device < 0:
            raise ValueError("CUDA kernel does not identify a physical device index")
        interval = ((first - start) / 1e6, (last - start) / 1e6)
        grouped[device, "all_kernels"].append(interval)
        grouped[device, reporter._kernel_category(event)].append(interval)
    if not grouped:
        raise ValueError("no observed CUDA kernels in profiler window")
    return {key: _merge(values) for key, values in grouped.items()}


def _rows(grouped, rank, duration, width):
    for (device, category), intervals in sorted(grouped.items()):
        for index in range(math.ceil(duration / width)):
            start, stop = index * width, min(duration, (index + 1) * width)
            busy = sum(
                max(0, min(stop, last) - max(start, first)) for first, last in intervals
            )
            if not 0 <= busy <= stop - start + 1e-9:
                raise ValueError("invalid interval union")
            yield {
                "rank": rank,
                "device": device,
                "category": category,
                "start_seconds": start,
                "stop_seconds": stop,
                "kernel_busy_seconds": busy,
            }


def _export_rank(path, rank, receipt, reporter, output, width):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != receipt["trace_sha256"]:
        raise ValueError("native trace differs from its completed profile report")
    steps, kernels, start, stop = reporter._trace(path)
    duration = (stop - start) / 1e6
    if duration != receipt["profile_window_seconds"]:
        raise ValueError("profiler window differs from completed report")
    grouped = _intervals(kernels, start, stop, reporter)
    rows = list(_rows(grouped, rank, duration, width))
    csv_path = output / f"rank-{rank}-timeline.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "rank": rank,
        "trace_sha256": digest,
        "trace_bytes": path.stat().st_size,
        "profile_window_seconds": duration,
        "steps": [
            {
                "name": step["name"],
                "start_seconds": (step["ts"] - start) / 1e6,
                "stop_seconds": (step["ts"] + step["dur"] - start) / 1e6,
            }
            for step in steps
        ],
        "timeline": {
            "file": csv_path.name,
            "rows": len(rows),
            "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        },
    }


def _read_summary(run_dir, summary_path):
    settings = json.loads((run_dir / "run.json").read_text())
    if settings["profile"] is not True:
        raise ValueError("requires a separately profiled training run")
    raw = (summary_path or run_dir / "profile-summary.json").read_bytes()
    summary = json.loads(raw)
    expected = {str(node * 8) for node in range(settings["nodes"])}
    if set(summary["ranks"]) != expected:
        raise ValueError("profile report does not cover one rank per node")
    return settings, raw, summary


def _main(args):
    if not math.isfinite(args.bin_seconds) or args.bin_seconds <= 0:
        raise ValueError("bin width must be finite and positive")
    reporter, source_hash = _load_reporter(args.recipe_dir)
    settings, raw, summary = _read_summary(args.run_dir, args.summary_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    job = args.run_dir / "output/cosmos3_wam/libero_10" / settings["name"]
    ranks = [
        _export_rank(
            job / "torch_trace/iteration_100" / f"rank{rank}_trace.json.gz",
            int(rank),
            receipt,
            reporter,
            args.output_dir,
            args.bin_seconds,
        )
        for rank, receipt in sorted(
            summary["ranks"].items(), key=lambda item: int(item[0])
        )
    ]
    result = {
        "schema": "npa.cosmos3.wam-profile-timeline.v1",
        "gpus": settings["gpus"],
        "bin_seconds": args.bin_seconds,
        "ranks": ranks,
        "profile_summary_sha256": hashlib.sha256(raw).hexdigest(),
        "profile_report_source_sha256": source_hash,
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Per-category interval unions in each bin; categories overlap and must not be stacked. All-kernel activity excludes memcpy, CPU work and unobserved activity. Categories prefer linked CPU operators; NCCL and unlinked kernels use names. Profiles are excluded from throughput comparisons.",
    }
    (args.output_dir / "profile-summary.json").write_bytes(raw)
    (args.output_dir / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-seconds", type=float, default=0.25)
    _main(parser.parse_args())
