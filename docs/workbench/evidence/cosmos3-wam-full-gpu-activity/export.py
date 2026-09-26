"""Extract numeric GPU telemetry within a completed native training process."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

FIELDS = (
    "elapsed_seconds",
    "gpu_index",
    "memory_used_mib",
    "utilization_percent",
    "power_watts",
)


def _window(run):
    raw = (run / "node-0.finished.json").read_bytes()
    receipt = json.loads(raw)
    settings_raw = (run / "run.json").read_bytes()
    settings = json.loads(settings_raw)
    if receipt["returncode"] != 0 or settings["nodes"] != 1:
        raise ValueError("requires a completed single-node run")
    if hashlib.sha256(settings_raw).hexdigest() != receipt["run_sha256"]:
        raise ValueError("run settings differ from the completion receipt")
    duration = receipt["train_process_seconds"]
    stop = receipt["ended_unix"]
    return {
        "started_unix": stop - duration,
        "ended_unix": stop,
        "duration_seconds": duration,
        "gpus": settings["gpus"],
        "completion_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "run_settings_sha256": receipt["run_sha256"],
    }


def _sample(line, window):
    values = next(csv.reader([line.decode()]))
    if len(values) != 9:
        raise ValueError("unexpected GPU recorder row")
    values = [value.strip() for value in values]
    stamp = datetime.strptime(values[0], "%Y/%m/%d %H:%M:%S.%f").replace(
        tzinfo=timezone.utc
    )
    observed = stamp.timestamp()
    if not window["started_unix"] <= observed <= window["ended_unix"]:
        return observed, None
    if values[2] != "NVIDIA B200":
        raise ValueError("unexpected GPU model")
    row = dict(
        zip(
            FIELDS,
            [
                observed - window["started_unix"],
                int(values[1]),
                float(values[3]),
                float(values[4]),
                float(values[5]),
            ],
            strict=True,
        )
    )
    if not all(math.isfinite(value) for value in row.values()):
        raise ValueError("non-finite device telemetry")
    if min(row.values()) < 0 or not 0 <= row["utilization_percent"] <= 100:
        raise ValueError("invalid device telemetry")
    return observed, row


def _extract(source, window):
    records, digest = [], hashlib.sha256()
    with source.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            observed, row = _sample(line, window)
            if row is not None:
                records.append(row)
                digest.update(line)
            if observed > window["ended_unix"] + 2:
                break
    if {row["gpu_index"] for row in records} != set(range(window["gpus"])):
        raise ValueError("missing or unexpected GPU indices")
    return records, digest.hexdigest()


def _device_summary(records, duration):
    times = [row["elapsed_seconds"] for row in records]
    intervals = [later - earlier for earlier, later in pairwise(times)]
    if len(records) < 2 or min(intervals) <= 0:
        raise ValueError("GPU timestamps must increase strictly")
    if times[0] > 5 or duration - times[-1] > 5:
        raise ValueError("telemetry does not cover the native process boundaries")
    metrics = {}
    for field in FIELDS[2:]:
        values = [row[field] for row in records]
        metrics[field] = {
            "minimum": min(values),
            "maximum": max(values),
            "sample_mean": statistics.mean(values),
            "sample_median": statistics.median(values),
        }
    return {
        "samples": len(records),
        "first_elapsed_seconds": times[0],
        "last_elapsed_seconds": times[-1],
        "sampling_interval_median_seconds": statistics.median(intervals),
        "sampling_interval_max_seconds": max(intervals),
        "metrics": metrics,
    }


def _write_samples(output, records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    raw = stream.getvalue().encode()
    compressed = gzip.compress(raw, mtime=0)
    path = output / "gpu-samples.csv.gz"
    path.write_bytes(compressed)
    return {
        "file": path.name,
        "rows": len(records),
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(compressed),
    }


def _main(args):
    window = _window(args.run_dir)
    records, source_digest = _extract(args.telemetry, window)
    devices = {
        str(index): _device_summary(
            [row for row in records if row["gpu_index"] == index],
            window["duration_seconds"],
        )
        for index in range(window["gpus"])
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "npa.cosmos3.wam-full-gpu-telemetry.v1",
        "window": window,
        "devices": devices,
        "selected_original_rows_sha256": source_digest,
        "samples": _write_samples(args.output_dir, records),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": (
            "Sampled whole-device measurements across the native process, including "
            "startup and checkpoint saves. Memory maxima are sampled device-memory "
            "maxima, not allocator or subsecond peaks. Utilization is the recorder's "
            "GPU-utilization sample, not model FLOP utilization. Sample means are "
            "arithmetic means, not time-weighted integrals."
        ),
    }
    (args.output_dir / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    _main(parser.parse_args())
