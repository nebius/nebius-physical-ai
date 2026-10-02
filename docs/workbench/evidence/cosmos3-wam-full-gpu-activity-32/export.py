"""Export every worker's numeric telemetry for a completed 32-B200 schedule."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def _load_reducer(source):
    spec = importlib.util.spec_from_file_location("wam_numeric_telemetry", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _window(run):
    raw = (run / "run.json").read_bytes()
    settings = json.loads(raw)
    if (settings["nodes"], settings["gpus"]) != (4, 32):
        raise ValueError("requires the completed four-node, thirty-two-GPU run")
    digest = hashlib.sha256(raw).hexdigest()
    intervals, receipts = [], {}
    for rank in range(4):
        raw = (run / f"node-{rank}.finished.json").read_bytes()
        receipt = json.loads(raw)
        if receipt["returncode"] != 0 or receipt["run_sha256"] != digest:
            raise ValueError("completion receipt does not verify this run")
        stop, duration = receipt["ended_unix"], receipt["train_process_seconds"]
        if not all(math.isfinite(v) for v in (stop, duration)) or duration <= 0:
            raise ValueError("invalid native process timing")
        intervals.append((stop - duration, stop))
        receipts[str(rank)] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "started_unix": stop - duration,
            "ended_unix": stop,
            "duration_seconds": duration,
        }
    start, stop = min(v[0] for v in intervals), max(v[1] for v in intervals)
    return {
        "started_unix": start,
        "ended_unix": stop,
        "duration_seconds": stop - start,
        "gpus": 8,
        "allocation_gpus": 32,
        "run_settings_sha256": digest,
        "node_completion_receipts": receipts,
        "scope": "Union of all four native training intervals on recorded UTC clocks.",
    }


def main(args):
    reducer = _load_reducer(args.numeric_reducer)
    window = _window(args.run_dir)
    records, digest = reducer._extract(args.telemetry, window)
    devices = {
        str(i): reducer._device_summary(
            [row for row in records if row["gpu_index"] == i],
            window["duration_seconds"],
        )
        for i in range(8)
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "npa.cosmos3.wam-full-node-telemetry.v1",
        "node_rank": args.node_rank,
        "window": window,
        "devices": devices,
        "selected_original_rows_sha256": digest,
        "samples": reducer._write_samples(args.output_dir, records),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numeric_reducer_sha256": hashlib.sha256(
            args.numeric_reducer.read_bytes()
        ).hexdigest(),
        "interpretation": "Sampled whole-device readings across the four native processes. "
        "Memory maxima are sampled device usage, not allocator or subsecond peaks. "
        "Utilization is a device reading, not model FLOP utilization. Means are "
        "arithmetic sample means, not time-weighted integrals.",
    }
    (args.output_dir / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, choices=range(4), required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--numeric-reducer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
