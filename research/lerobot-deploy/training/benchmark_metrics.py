#!/usr/bin/env python3
"""Collect and summarize host/GPU metrics for LeRobot benchmark runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Union


SAMPLE_FIELDS = [
    "timestamp",
    "cpu_util_pct",
    "cpu_iowait_pct",
    "host_mem_util_pct",
    "load1",
    "load5",
    "gpu_count",
    "gpu_names",
    "gpu_avg_util_pct",
    "gpu_max_util_pct",
    "gpu_avg_mem_util_pct",
    "gpu_max_mem_util_pct",
    "gpu_total_mem_mb",
    "gpu_max_mem_used_mb",
    "gpu_util_values_pct",
    "gpu_mem_used_values_mb",
]


GpuQueryRow = dict[str, Union[str, float]]
SampleRow = dict[str, Union[float, list[float], list[str]]]


def mean(values: Iterable[float]) -> float:
    seq = list(values)
    return sum(seq) / len(seq) if seq else 0.0


def percentile(values: Iterable[float], pct: float) -> float:
    seq = sorted(values)
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    rank = max(0.0, min(100.0, pct)) / 100.0 * (len(seq) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(seq[lower])
    lower_value = float(seq[lower])
    upper_value = float(seq[upper])
    fraction = rank - lower
    return lower_value + (upper_value - lower_value) * fraction


def stddev(values: Iterable[float]) -> float:
    seq = list(values)
    if len(seq) < 2:
        return 0.0
    avg = mean(seq)
    variance = sum((value - avg) ** 2 for value in seq) / len(seq)
    return math.sqrt(variance)


def unique_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def parse_float(value: str | None) -> float:
    if value is None:
        return 0.0
    cleaned = value.strip()
    if cleaned in {"", "N/A", "[Not Supported]"}:
        return 0.0
    return float(cleaned)


def read_cpu_times() -> tuple[int, int, int]:
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        cpu = handle.readline().split()
    values = [int(part) for part in cpu[1:]]
    idle = values[3]
    iowait = values[4] if len(values) > 4 else 0
    total = sum(values)
    return total, idle, iowait


def read_mem_util_pct() -> float:
    mem_total = 0
    mem_available = 0
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1])
    if mem_total == 0:
        return 0.0
    used = mem_total - mem_available
    return (used / mem_total) * 100.0


def read_load_average() -> tuple[float, float]:
    load1, load5, _ = os_getloadavg()
    return load1, load5


def os_getloadavg() -> tuple[float, float, float]:
    try:
        return tuple(os.getloadavg())  # type: ignore[name-defined]
    except (AttributeError, OSError, NameError):
        return (0.0, 0.0, 0.0)


def query_gpus() -> list[GpuQueryRow]:
    if shutil.which("nvidia-smi") is None:
        return []
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    # Respect CUDA_VISIBLE_DEVICES so the sampler only reports GPUs that
    # the training process can actually see.  nvidia-smi ignores the env var
    # itself, but its --id= flag achieves the same filtering.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        command.insert(1, f"--id={cuda_visible}")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return []

    rows: list[GpuQueryRow] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": parse_float(parts[0]),
                "name": parts[1],
                "utilization_gpu": parse_float(parts[2]),
                "utilization_mem": parse_float(parts[3]),
                "memory_used": parse_float(parts[4]),
                "memory_total": parse_float(parts[5]),
            }
        )
    return rows


def emit_sample(output: Path, interval: float) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    previous_total, previous_idle, previous_iowait = read_cpu_times()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()

        while not stop:
            started = time.time()
            total, idle, iowait = read_cpu_times()
            total_delta = max(total - previous_total, 1)
            idle_delta = idle - previous_idle
            iowait_delta = iowait - previous_iowait
            previous_total, previous_idle, previous_iowait = total, idle, iowait

            cpu_util_pct = ((total_delta - idle_delta) / total_delta) * 100.0
            cpu_iowait_pct = (iowait_delta / total_delta) * 100.0
            host_mem_util_pct = read_mem_util_pct()
            load1, load5 = read_load_average()

            gpus = query_gpus()
            gpu_utils = [gpu["utilization_gpu"] for gpu in gpus]
            gpu_mem_utils = [gpu["utilization_mem"] for gpu in gpus]
            gpu_mem_used = [gpu["memory_used"] for gpu in gpus]
            gpu_mem_total = [gpu["memory_total"] for gpu in gpus]

            writer.writerow(
                {
                    "timestamp": f"{started:.3f}",
                    "cpu_util_pct": f"{cpu_util_pct:.2f}",
                    "cpu_iowait_pct": f"{cpu_iowait_pct:.2f}",
                    "host_mem_util_pct": f"{host_mem_util_pct:.2f}",
                    "load1": f"{load1:.2f}",
                    "load5": f"{load5:.2f}",
                    "gpu_count": len(gpus),
                    "gpu_names": "|".join(str(gpu.get("name", "")) for gpu in gpus),
                    "gpu_avg_util_pct": f"{mean(gpu_utils):.2f}",
                    "gpu_max_util_pct": f"{max(gpu_utils) if gpu_utils else 0.0:.2f}",
                    "gpu_avg_mem_util_pct": f"{mean(gpu_mem_utils):.2f}",
                    "gpu_max_mem_util_pct": f"{max(gpu_mem_utils) if gpu_mem_utils else 0.0:.2f}",
                    "gpu_total_mem_mb": f"{sum(gpu_mem_total):.2f}",
                    "gpu_max_mem_used_mb": f"{max(gpu_mem_used) if gpu_mem_used else 0.0:.2f}",
                    "gpu_util_values_pct": "|".join(f"{value:.2f}" for value in gpu_utils),
                    "gpu_mem_used_values_mb": "|".join(f"{value:.2f}" for value in gpu_mem_used),
                }
            )
            handle.flush()

            elapsed = time.time() - started
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    return 0


def parse_samples(path: Path) -> list[SampleRow]:
    if not path.exists():
        return []
    rows: list[SampleRow] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gpu_util_values = [
                parse_float(value)
                for value in row.get("gpu_util_values_pct", "").split("|")
                if value.strip()
            ]
            gpu_names = [
                value.strip()
                for value in row.get("gpu_names", "").split("|")
                if value.strip()
            ]
            gpu_mem_used_values = [
                parse_float(value)
                for value in row.get("gpu_mem_used_values_mb", "").split("|")
                if value.strip()
            ]
            rows.append(
                {
                    "timestamp": parse_float(row.get("timestamp")),
                    "cpu_util_pct": parse_float(row.get("cpu_util_pct")),
                    "cpu_iowait_pct": parse_float(row.get("cpu_iowait_pct")),
                    "host_mem_util_pct": parse_float(row.get("host_mem_util_pct")),
                    "load1": parse_float(row.get("load1")),
                    "load5": parse_float(row.get("load5")),
                    "gpu_count": parse_float(row.get("gpu_count")),
                    "gpu_names": gpu_names,
                    "gpu_avg_util_pct": parse_float(row.get("gpu_avg_util_pct")),
                    "gpu_max_util_pct": parse_float(row.get("gpu_max_util_pct")),
                    "gpu_avg_mem_util_pct": parse_float(row.get("gpu_avg_mem_util_pct")),
                    "gpu_max_mem_util_pct": parse_float(row.get("gpu_max_mem_util_pct")),
                    "gpu_total_mem_mb": parse_float(row.get("gpu_total_mem_mb")),
                    "gpu_max_mem_used_mb": parse_float(row.get("gpu_max_mem_used_mb")),
                    "gpu_util_values": gpu_util_values,
                    "gpu_mem_used_values": gpu_mem_used_values,
                }
            )
    return rows


def summarize_gpu_model_names(rows: list[SampleRow]) -> list[str]:
    names: list[str] = []
    for row in rows:
        raw_names = row.get("gpu_names", [])
        if isinstance(raw_names, list):
            names.extend(str(value) for value in raw_names)
    return unique_preserve(names)


def format_gpu_model_label(models: Iterable[str], gpu_count: int) -> str:
    unique_models = unique_preserve(models)
    if not unique_models:
        return f"{gpu_count} GPU(s)" if gpu_count > 0 else "unknown GPU"
    if len(unique_models) == 1:
        return unique_models[0]
    return ", ".join(unique_models)


def load_torch_memory_summary(path: Path | None) -> dict[str, object]:
    summary: dict[str, object] = {
        "available": False,
        "process_count": 0,
        "peak_allocated_mb": 0.0,
        "peak_reserved_mb": 0.0,
        "device_names": [],
    }
    if path is None or not path.exists():
        return summary

    payloads: list[dict[str, object]] = []
    for item in sorted(path.glob("*.json")):
        try:
            payload = json.loads(item.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)

    if not payloads:
        return summary

    max_allocated_bytes = 0
    max_reserved_bytes = 0
    device_names: list[str] = []
    for payload in payloads:
        max_allocated_bytes = max(
            max_allocated_bytes,
            int(payload.get("max_memory_allocated_bytes", 0) or 0),
        )
        max_reserved_bytes = max(
            max_reserved_bytes,
            int(payload.get("max_memory_reserved_bytes", 0) or 0),
        )
        raw_names = payload.get("device_names", [])
        if isinstance(raw_names, list):
            device_names.extend(str(name) for name in raw_names)

    summary.update(
        {
            "available": True,
            "process_count": len(payloads),
            "peak_allocated_mb": round(max_allocated_bytes / (1024 * 1024), 2),
            "peak_reserved_mb": round(max_reserved_bytes / (1024 * 1024), 2),
            "device_names": unique_preserve(device_names),
        }
    )
    return summary


def trim_warmup_samples(
    rows: list[SampleRow],
    trim_seconds: float,
    min_samples_remaining: int = 3,
) -> tuple[list[SampleRow], int, float]:
    if trim_seconds <= 0.0 or len(rows) <= min_samples_remaining:
        return rows, 0, 0.0

    first_timestamp = float(rows[0].get("timestamp", 0.0) or 0.0)
    trimmed_rows = [
        row
        for row in rows
        if float(row.get("timestamp", 0.0) or 0.0) - first_timestamp >= trim_seconds
    ]
    if len(trimmed_rows) < min_samples_remaining:
        return rows, 0, 0.0

    return trimmed_rows, len(rows) - len(trimmed_rows), trim_seconds


def detect_bottleneck(summary: dict[str, object]) -> tuple[str, str]:
    failure_kind = str(summary.get("failure_kind", "none"))
    gpu = summary.get("gpu", {})
    cpu = summary.get("cpu", {})
    if failure_kind == "oom":
        return "memory_ceiling", "Run failed with an out-of-memory signature before steady-state throughput."

    gpu_count = int(gpu.get("count", 0)) if isinstance(gpu, dict) else 0
    if gpu_count == 0:
        return "gpu_metrics_unavailable", "nvidia-smi was unavailable, so GPU bottleneck classification is inconclusive."

    avg_gpu_util = float(gpu.get("avg_util_pct", 0.0)) if isinstance(gpu, dict) else 0.0
    avg_mem_util = float(gpu.get("avg_mem_util_pct", 0.0)) if isinstance(gpu, dict) else 0.0
    p50_gpu_util = float(gpu.get("p50_util_pct", avg_gpu_util)) if isinstance(gpu, dict) else avg_gpu_util
    p90_gpu_util = float(gpu.get("p90_util_pct", avg_gpu_util)) if isinstance(gpu, dict) else avg_gpu_util
    stddev_gpu_util = float(gpu.get("stddev_util_pct", 0.0)) if isinstance(gpu, dict) else 0.0
    avg_cpu_util = float(cpu.get("avg_util_pct", 0.0)) if isinstance(cpu, dict) else 0.0
    p90_cpu_util = float(cpu.get("p90_util_pct", avg_cpu_util)) if isinstance(cpu, dict) else avg_cpu_util
    avg_iowait = float(cpu.get("avg_iowait_pct", 0.0)) if isinstance(cpu, dict) else 0.0
    peak_torch_allocated_mb = (
        float(gpu.get("peak_torch_memory_allocated_mb", 0.0)) if isinstance(gpu, dict) else 0.0
    )
    peak_torch_reserved_mb = (
        float(gpu.get("peak_torch_memory_reserved_mb", 0.0)) if isinstance(gpu, dict) else 0.0
    )
    total_memory_mb = float(gpu.get("total_memory_mb", 0.0)) if isinstance(gpu, dict) else 0.0
    # total_memory_mb is the sum across all visible GPUs; use per-GPU memory
    # for bottleneck detection since torch peak memory is per-process.
    per_gpu_memory_mb = (total_memory_mb / gpu_count) if gpu_count > 0 else total_memory_mb

    torch_allocated_pct = (peak_torch_allocated_mb / per_gpu_memory_mb * 100.0) if per_gpu_memory_mb > 0 else 0.0
    torch_reserved_pct = (peak_torch_reserved_mb / per_gpu_memory_mb * 100.0) if per_gpu_memory_mb > 0 else 0.0
    gpu_spiky_or_bursty = (
        avg_gpu_util < 70.0
        and p90_gpu_util >= 80.0
        and stddev_gpu_util >= 20.0
        and avg_gpu_util <= p90_gpu_util - 20.0
    )
    consistently_low_gpu_util = avg_gpu_util < 60.0 and p90_gpu_util < 65.0

    if torch_allocated_pct >= 85.0 and avg_gpu_util < 75.0:
        return (
            "memory_pressure_bound",
            f"Peak torch-allocated memory reached {torch_allocated_pct:.1f}% of per-GPU memory "
            f"({peak_torch_allocated_mb:.0f}/{per_gpu_memory_mb:.0f} MiB) without matching compute utilization, "
            "suggesting the model is memory-constrained rather than compute-bound.",
        )

    if avg_mem_util >= 90.0 and avg_gpu_util < 70.0:
        detail = (
            f"Peak torch-reserved memory reached {torch_reserved_pct:.1f}% of GPU memory. "
            if torch_reserved_pct > 0.0
            else ""
        )
        return (
            "memory_pressure_bound",
            f"GPU memory stayed near capacity ({avg_mem_util:.1f}% average memory utilization) without matching compute utilization. "
            f"{detail}This suggests memory pressure or allocator overhead rather than pure compute saturation.",
        )
    if gpu_spiky_or_bursty and p90_cpu_util >= 85.0:
        return (
            "cpu_or_decode_bound",
            f"GPU utilization was bursty rather than uniformly low (avg {avg_gpu_util:.1f}%, p90 {p90_gpu_util:.1f}%, stddev {stddev_gpu_util:.1f}%) "
            f"while host CPU p90 reached {p90_cpu_util:.1f}%, which points to decode or input-pipeline stalls between compute bursts.",
        )
    if gpu_spiky_or_bursty and avg_iowait >= 15.0:
        return (
            "storage_or_input_io_bound",
            f"GPU utilization was bursty (avg {avg_gpu_util:.1f}%, p90 {p90_gpu_util:.1f}%, stddev {stddev_gpu_util:.1f}%) "
            f"while host iowait averaged {avg_iowait:.1f}%, suggesting storage or input stalls between compute bursts.",
        )
    if avg_cpu_util >= 85.0 and avg_gpu_util < 60.0:
        return "cpu_or_decode_bound", "Host CPU stayed saturated while GPU utilization remained low, suggesting an input/decode bottleneck."
    if avg_iowait >= 20.0 and avg_gpu_util < 60.0:
        return "storage_or_input_io_bound", "Host iowait stayed elevated while GPU utilization remained low, suggesting storage or input stalls."
    if avg_gpu_util >= 85.0 or (p50_gpu_util >= 80.0 and p90_gpu_util >= 90.0 and stddev_gpu_util < 15.0):
        return "gpu_compute_bound", "GPU utilization stayed high enough that the model is likely compute-bound."
    if consistently_low_gpu_util:
        return (
            "underutilized_or_hardware_mismatch",
            f"GPU utilization stayed consistently low (avg {avg_gpu_util:.1f}%, p90 {p90_gpu_util:.1f}%), indicating underutilized hardware or another upstream bottleneck.",
        )
    if gpu_spiky_or_bursty:
        return (
            "mixed_or_unclassified",
            f"GPU utilization was bursty (avg {avg_gpu_util:.1f}%, p90 {p90_gpu_util:.1f}%, stddev {stddev_gpu_util:.1f}%), "
            "which points to transient pipeline stalls, but the CPU and I/O signals were not strong enough to classify the bottleneck more narrowly.",
        )
    return "mixed_or_unclassified", "No single resource dominated strongly enough for a higher-confidence classification."


def summarize_run(args: argparse.Namespace) -> int:
    raw_rows = parse_samples(Path(args.samples))
    requested_warmup_trim_seconds = max(args.warmup_trim_seconds, 0.0)
    effective_warmup_trim_seconds = requested_warmup_trim_seconds
    if args.run_seconds > 0.0 and requested_warmup_trim_seconds > 0.0:
        effective_warmup_trim_seconds = min(requested_warmup_trim_seconds, args.run_seconds * 0.2)
    rows, trimmed_prefix_count, applied_warmup_trim_seconds = trim_warmup_samples(
        raw_rows,
        effective_warmup_trim_seconds,
    )
    cpu_utils = [float(row["cpu_util_pct"]) for row in rows]
    cpu_iowaits = [float(row["cpu_iowait_pct"]) for row in rows]
    host_mem_utils = [float(row["host_mem_util_pct"]) for row in rows]
    load1_values = [float(row["load1"]) for row in rows]
    load5_values = [float(row["load5"]) for row in rows]
    gpu_avg_utils = [float(row["gpu_avg_util_pct"]) for row in rows]
    gpu_max_utils = [float(row["gpu_max_util_pct"]) for row in rows]
    gpu_avg_mem_utils = [float(row["gpu_avg_mem_util_pct"]) for row in rows]
    gpu_max_mem_utils = [float(row["gpu_max_mem_util_pct"]) for row in rows]
    gpu_peak_used = [float(row["gpu_max_mem_used_mb"]) for row in rows]
    gpu_totals = [float(row["gpu_total_mem_mb"]) for row in rows]
    gpu_counts = [int(float(row["gpu_count"])) for row in rows]
    gpu_models = summarize_gpu_model_names(rows)
    torch_memory = load_torch_memory_summary(Path(args.torch_memory_dir) if args.torch_memory_dir else None)

    # In DDP, effective batch = batch_size × gpu_count.  Track both
    # steps/sec and samples/sec so scaling comparisons are apples-to-apples.
    effective_batch_size = args.batch_size * args.gpu_count
    steps_per_second = round(args.work_count / args.run_seconds, 6) if args.run_seconds > 0 else 0.0
    samples_per_second = round(steps_per_second * effective_batch_size, 6)

    summary: dict[str, object] = {
        "phase": args.phase,
        "policy": args.policy,
        "gpu_count_requested": args.gpu_count,
        "batch_size": args.batch_size,
        "effective_batch_size": effective_batch_size,
        "status": args.status,
        "failure_kind": args.failure_kind,
        "run_seconds": round(args.run_seconds, 3),
        "work_units": args.work_units,
        "work_count": args.work_count,
        "throughput_per_second": steps_per_second,
        "samples_per_second": samples_per_second,
        "samples": {
            "count": len(raw_rows),
            "steady_state_count": len(rows),
            "trimmed_prefix_count": trimmed_prefix_count,
            "path": args.samples,
            "warmup_trim_seconds_requested": round(requested_warmup_trim_seconds, 3),
            "warmup_trim_seconds_applied": round(applied_warmup_trim_seconds, 3),
        },
        "gpu_identity": {
            "models": gpu_models,
            "label": format_gpu_model_label(gpu_models, max(gpu_counts) if gpu_counts else 0),
        },
        "cpu": {
            "avg_util_pct": round(mean(cpu_utils), 2),
            "max_util_pct": round(max(cpu_utils) if cpu_utils else 0.0, 2),
            "avg_iowait_pct": round(mean(cpu_iowaits), 2),
            "max_iowait_pct": round(max(cpu_iowaits) if cpu_iowaits else 0.0, 2),
            "p90_util_pct": round(percentile(cpu_utils, 90.0), 2),
        },
        "host_memory": {
            "avg_util_pct": round(mean(host_mem_utils), 2),
            "max_util_pct": round(max(host_mem_utils) if host_mem_utils else 0.0, 2),
        },
        "load_average": {
            "avg_load1": round(mean(load1_values), 2),
            "avg_load5": round(mean(load5_values), 2),
        },
        "gpu": {
            "count": max(gpu_counts) if gpu_counts else 0,
            "models": gpu_models,
            "model_name": format_gpu_model_label(gpu_models, max(gpu_counts) if gpu_counts else 0),
            "avg_util_pct": round(mean(gpu_avg_utils), 2),
            "max_util_pct": round(max(gpu_max_utils) if gpu_max_utils else 0.0, 2),
            "p50_util_pct": round(percentile(gpu_avg_utils, 50.0), 2),
            "p90_util_pct": round(percentile(gpu_avg_utils, 90.0), 2),
            "stddev_util_pct": round(stddev(gpu_avg_utils), 2),
            "avg_mem_util_pct": round(mean(gpu_avg_mem_utils), 2),
            "max_mem_util_pct": round(max(gpu_max_mem_utils) if gpu_max_mem_utils else 0.0, 2),
            "peak_memory_used_mb": round(max(gpu_peak_used) if gpu_peak_used else 0.0, 2),
            "peak_nvidia_smi_memory_used_mb": round(max(gpu_peak_used) if gpu_peak_used else 0.0, 2),
            "peak_torch_memory_allocated_mb": torch_memory.get("peak_allocated_mb", 0.0),
            "peak_torch_memory_reserved_mb": torch_memory.get("peak_reserved_mb", 0.0),
            "torch_memory_process_count": torch_memory.get("process_count", 0),
            "total_memory_mb": round(max(gpu_totals) if gpu_totals else 0.0, 2),
        },
    }

    classification, reason = detect_bottleneck(summary)
    summary["bottleneck"] = {
        "classification": classification,
        "reason": reason,
        "gpu_underutilized": classification == "underutilized_or_hardware_mismatch",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def load_json(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def maybe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_eval_behavior(path: str | None) -> dict[str, object]:
    if not path:
        return {}

    if not Path(path).is_file():
        return {
            "path": path,
            "status": "missing_file",
        }

    payload = load_json(path)
    overall = payload.get("overall", {})
    if not isinstance(overall, dict):
        return {
            "path": path,
            "status": "missing_overall",
        }

    per_task = payload.get("per_task", [])
    task_count = len(per_task) if isinstance(per_task, list) else 0
    behavior: dict[str, object] = {
        "path": path,
        "status": "available",
        "pc_success": overall.get("pc_success"),
        "avg_sum_reward": overall.get("avg_sum_reward"),
        "avg_max_reward": overall.get("avg_max_reward"),
        "n_episodes": overall.get("n_episodes"),
        "eval_s": overall.get("eval_s"),
        "eval_ep_s": overall.get("eval_ep_s"),
        "task_count": task_count,
    }
    n_episodes = maybe_float(behavior.get("n_episodes"))
    if task_count > 0 and n_episodes is not None:
        behavior["episodes_per_task"] = round(n_episodes / task_count, 3)
    return behavior


def row_gpu_label(row: dict[str, object]) -> str:
    gpu = row.get("gpu", {})
    if isinstance(gpu, dict):
        model_name = str(gpu.get("model_name", "")).strip()
        if model_name:
            return model_name
        models = gpu.get("models", [])
        if isinstance(models, list):
            return format_gpu_model_label((str(model) for model in models), int(row.get("gpu_count_requested", 0) or 0))
    return format_gpu_model_label([], int(row.get("gpu_count_requested", 0) or 0))


def row_bottleneck(row: dict[str, object]) -> str:
    bottleneck = row.get("bottleneck", {})
    if isinstance(bottleneck, dict):
        return str(bottleneck.get("classification", "unknown"))
    return "unknown"


def _row_samples_per_second(row: dict[str, object]) -> float:
    """Extract samples_per_second, falling back to deriving it for older summaries."""
    sps = row.get("samples_per_second")
    if sps is not None and float(sps) > 0:
        return float(sps)
    # Older summaries only have throughput_per_second (steps/sec).
    tps = float(row.get("throughput_per_second", 0.0) or 0.0)
    bs = int(row.get("batch_size", 0) or 0)
    gpus = int(row.get("gpu_count_requested", 1) or 1)
    return tps * bs * gpus


def build_scaling_transitions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    transitions: list[dict[str, object]] = []
    for previous, current in zip(rows, rows[1:]):
        previous_gpus = int(previous.get("gpu_count_requested", 0) or 0)
        current_gpus = int(current.get("gpu_count_requested", 0) or 0)
        previous_seconds = float(previous.get("run_seconds", 0.0) or 0.0)
        current_seconds = float(current.get("run_seconds", 0.0) or 0.0)
        previous_sps = _row_samples_per_second(previous)
        current_sps = _row_samples_per_second(current)
        gpu_ratio = (current_gpus / previous_gpus) if previous_gpus > 0 else 0.0
        step_speedup = (previous_seconds / current_seconds) if previous_seconds > 0 and current_seconds > 0 else 0.0
        sample_throughput_ratio = (current_sps / previous_sps) if previous_sps > 0 else 0.0
        efficiency = (sample_throughput_ratio / gpu_ratio) if gpu_ratio > 0 else 0.0
        sample_throughput_gain_pct = (
            ((current_sps - previous_sps) / previous_sps) * 100.0
            if previous_sps > 0
            else 0.0
        )
        previous_bottleneck = row_bottleneck(previous)
        current_bottleneck = row_bottleneck(current)
        transitions.append(
            {
                "from_gpu_count": previous_gpus,
                "to_gpu_count": current_gpus,
                "step_speedup_vs_previous": round(step_speedup, 4),
                "sample_throughput_ratio_vs_previous": round(sample_throughput_ratio, 4),
                "gpu_ratio_vs_previous": round(gpu_ratio, 4),
                "scaling_efficiency_vs_previous": round(efficiency, 4),
                "sample_throughput_gain_pct": round(sample_throughput_gain_pct, 2),
                "from_bottleneck": previous_bottleneck,
                "to_bottleneck": current_bottleneck,
                "bottleneck_changed": previous_bottleneck != current_bottleneck,
            }
        )
    return transitions


def combine_profile(args: argparse.Namespace) -> int:
    train = load_json(args.train_summary)
    eval_summary = load_json(args.eval_summary)
    eval_behavior = load_eval_behavior(args.eval_info)
    train_seconds = float(train.get("run_seconds", 0.0))
    eval_seconds = float(eval_summary.get("run_seconds", 0.0))
    total_seconds = train_seconds + eval_seconds
    eval_over_train = (eval_seconds / train_seconds) if train_seconds > 0 else 0.0
    eval_share_pct = (eval_seconds / total_seconds * 100.0) if total_seconds > 0 else 0.0

    headline = []
    train_bottleneck = train.get("bottleneck", {})
    gpu_label = row_gpu_label(train)
    if isinstance(train_bottleneck, dict) and train_bottleneck.get("gpu_underutilized"):
        headline.append(
            f"Training only averaged {train['gpu']['avg_util_pct']}% GPU utilization on {gpu_label}, which points to hardware mismatch or an upstream bottleneck."
        )
    if eval_over_train >= 1.0:
        headline.append(
            f"Evaluation took {eval_over_train:.2f}x as long as training for this workflow configuration."
        )
    elif eval_share_pct >= 40.0:
        headline.append(
            f"Evaluation consumed {eval_share_pct:.1f}% of end-to-end runtime, so eval cost is operationally significant."
        )
    if eval_behavior:
        if eval_behavior.get("status") != "available":
            headline.append(
                f"Behavioral eval metrics were unavailable; expected eval_info at {eval_behavior.get('path')}."
            )
        else:
            pc_success = maybe_float(eval_behavior.get("pc_success"))
            avg_sum_reward = maybe_float(eval_behavior.get("avg_sum_reward"))
            n_episodes = int(maybe_float(eval_behavior.get("n_episodes")) or 0)
            if pc_success == 0.0 and avg_sum_reward == 0.0 and n_episodes > 0:
                headline.append(
                    f"Evaluation achieved zero task success across {n_episodes} episodes."
                )
            elif pc_success is not None and n_episodes > 0:
                headline.append(
                    f"Evaluation behavior was pc_success={pc_success} across {n_episodes} episodes."
                )

    combined = {
        "policy": train.get("policy"),
        "gpu_count_requested": train.get("gpu_count_requested"),
        "batch_size": train.get("batch_size"),
        "train": train,
        "eval": eval_summary,
        "eval_behavior": eval_behavior,
        "workflow_split": {
            "train_seconds": round(train_seconds, 3),
            "eval_seconds": round(eval_seconds, 3),
            "eval_over_train_ratio": round(eval_over_train, 4),
            "eval_share_pct": round(eval_share_pct, 2),
        },
        "headline_findings": headline,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def summarize_scaling(args: argparse.Namespace) -> int:
    rows = [load_json(path) for path in args.run_summaries]
    successful = [row for row in rows if row.get("status") == "success"]
    if successful:
        baseline = min(successful, key=lambda row: int(row.get("gpu_count_requested", 0)))
        baseline_seconds = float(baseline.get("run_seconds", 0.0))
        baseline_gpus = int(baseline.get("gpu_count_requested", 1))
    else:
        baseline = None
        baseline_seconds = 0.0
        baseline_gpus = 1

    baseline_sps = _row_samples_per_second(baseline) if baseline else 0.0

    table = []
    for row in sorted(rows, key=lambda item: int(item.get("gpu_count_requested", 0))):
        gpu_count = int(row.get("gpu_count_requested", 0))
        seconds = float(row.get("run_seconds", 0.0))
        sps = _row_samples_per_second(row)
        if row.get("status") == "success" and baseline_seconds > 0 and seconds > 0:
            step_speedup = baseline_seconds / seconds
            gpu_ratio = gpu_count / baseline_gpus if baseline_gpus > 0 else 0.0
            # In DDP, effective batch = batch_size × gpu_count.  Scaling
            # efficiency compares sample throughput, not step throughput.
            sample_throughput_ratio = (sps / baseline_sps) if baseline_sps > 0 else 0.0
            efficiency = (sample_throughput_ratio / gpu_ratio) if gpu_ratio > 0 else 0.0
        else:
            step_speedup = 0.0
            sample_throughput_ratio = 0.0
            efficiency = 0.0
        table.append(
            {
                "policy": row.get("policy"),
                "gpu_model": row_gpu_label(row),
                "gpu_count": gpu_count,
                "status": row.get("status"),
                "train_seconds": round(seconds, 3),
                "throughput_per_second": row.get("throughput_per_second"),
                "samples_per_second": sps,
                "step_speedup_vs_baseline": round(step_speedup, 4),
                "sample_throughput_ratio_vs_baseline": round(sample_throughput_ratio, 4),
                "scaling_efficiency_vs_baseline": round(efficiency, 4),
                "avg_gpu_util_pct": row.get("gpu", {}).get("avg_util_pct", 0.0),  # type: ignore[union-attr]
                "bottleneck": row.get("bottleneck", {}).get("classification", "unknown"),  # type: ignore[union-attr]
            }
        )

    successful_sorted = sorted(successful, key=lambda item: int(item.get("gpu_count_requested", 0) or 0))
    transitions = build_scaling_transitions(successful_sorted)
    headline_findings: list[str] = []
    successful_table = [row for row in table if row["status"] == "success"]
    if baseline and successful_table:
        last = successful_table[-1]
        headline_findings.append(
            f"On {row_gpu_label(baseline)}, scaling from {baseline_gpus} to {last['gpu_count']} GPU(s) reached "
            f"{last['sample_throughput_ratio_vs_baseline']:.2f}x sample throughput with {last['scaling_efficiency_vs_baseline']:.2f} efficiency."
        )
    if len(successful_sorted) >= 3:
        highest = successful_sorted[-1]
        highest_gpus = int(highest.get("gpu_count_requested", 0) or 0)
        highest_sps = _row_samples_per_second(highest)
        flattening_candidates: list[dict[str, object]] = []
        for earlier in successful_sorted[1:-1]:
            earlier_gpus = int(earlier.get("gpu_count_requested", 0) or 0)
            earlier_sps = _row_samples_per_second(earlier)
            gpu_ratio = (highest_gpus / earlier_gpus) if earlier_gpus > 0 else 0.0
            sample_ratio = (highest_sps / earlier_sps) if earlier_sps > 0 else 0.0
            efficiency = (sample_ratio / gpu_ratio) if gpu_ratio > 0 else 0.0
            if gpu_ratio >= 2.0:
                flattening_candidates.append(
                    {
                        "from_gpu_count": earlier_gpus,
                        "to_gpu_count": highest_gpus,
                        "sample_throughput_ratio": sample_ratio,
                        "gpu_ratio": gpu_ratio,
                        "efficiency": efficiency,
                    }
                )
        if flattening_candidates:
            weakest = min(flattening_candidates, key=lambda item: float(item["efficiency"]))
            if float(weakest["efficiency"]) < 0.5 or float(weakest["sample_throughput_ratio"]) < 1.25:
                headline_findings.append(
                    f"Scaling flattened from {weakest['from_gpu_count']} to {weakest['to_gpu_count']} GPU(s): "
                    f"only {float(weakest['sample_throughput_ratio']):.2f}x sample throughput for {float(weakest['gpu_ratio']):.2f}x more GPUs "
                    f"({float(weakest['efficiency']):.2f} efficiency)."
                )
    bottleneck_shift = next(
        (
            transition
            for transition in reversed(transitions)
            if transition["bottleneck_changed"]
        ),
        None,
    )
    if bottleneck_shift:
        headline_findings.append(
            f"Bottleneck shifted from {bottleneck_shift['from_bottleneck']} at {bottleneck_shift['from_gpu_count']} GPU(s) "
            f"to {bottleneck_shift['to_bottleneck']} at {bottleneck_shift['to_gpu_count']} GPU(s)."
        )

    payload = {
        "policy": rows[0].get("policy") if rows else None,
        "baseline_gpu_count": baseline_gpus if baseline else None,
        "rows": table,
        "transitions": transitions,
        "headline": headline_findings[0] if headline_findings else "",
        "headline_findings": headline_findings,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "policy",
                "gpu_model",
                "gpu_count",
                "status",
                "train_seconds",
                "throughput_per_second",
                "samples_per_second",
                "step_speedup_vs_baseline",
                "sample_throughput_ratio_vs_baseline",
                "scaling_efficiency_vs_baseline",
                "avg_gpu_util_pct",
                "bottleneck",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table)
    return 0


def summarize_memory(args: argparse.Namespace) -> int:
    rows = [load_json(path) for path in args.run_summaries]
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("policy")),
            row_gpu_label(row),
            int(row.get("gpu_count_requested", 0) or 0),
        )
        grouped.setdefault(key, []).append(row)

    group_payloads: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    for (policy, gpu_model, gpu_count), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        ordered = sorted(group_rows, key=lambda row: int(row.get("batch_size", 0) or 0))
        max_success = None
        min_oom = None
        for row in ordered:
            batch_size = int(row.get("batch_size", 0) or 0)
            if row.get("status") == "success":
                max_success = batch_size
            elif row.get("failure_kind") == "oom" and min_oom is None:
                min_oom = batch_size

        headline = f"No memory ceiling was found in the tested batch sizes on {gpu_model}."
        if max_success is not None and min_oom is not None:
            headline = (
                f"On {gpu_model} ({gpu_count} GPU(s)), highest successful batch size was {max_success}; "
                f"first OOM occurred at batch size {min_oom}."
            )
        elif max_success is None and ordered:
            headline = (
                f"On {gpu_model} ({gpu_count} GPU(s)), all tested batch sizes failed; "
                f"first tested batch size was {ordered[0].get('batch_size')}."
            )

        table = []
        for row in ordered:
            gpu = row.get("gpu", {})
            table_row = {
                "policy": row.get("policy"),
                "gpu_model": gpu_model,
                "gpu_count": row.get("gpu_count_requested"),
                "batch_size": row.get("batch_size"),
                "status": row.get("status"),
                "failure_kind": row.get("failure_kind"),
                "train_seconds": row.get("run_seconds"),
                "peak_nvidia_smi_memory_mb": gpu.get("peak_nvidia_smi_memory_used_mb", gpu.get("peak_memory_used_mb", 0.0)) if isinstance(gpu, dict) else 0.0,
                "peak_torch_memory_allocated_mb": gpu.get("peak_torch_memory_allocated_mb", 0.0) if isinstance(gpu, dict) else 0.0,
                "peak_torch_memory_reserved_mb": gpu.get("peak_torch_memory_reserved_mb", 0.0) if isinstance(gpu, dict) else 0.0,
                "avg_gpu_util_pct": gpu.get("avg_util_pct", 0.0) if isinstance(gpu, dict) else 0.0,
            }
            table.append(table_row)
            csv_rows.append(table_row)

        group_payloads.append(
            {
                "policy": policy,
                "gpu_model": gpu_model,
                "gpu_count": gpu_count,
                "max_successful_batch_size": max_success,
                "min_oom_batch_size": min_oom,
                "rows": table,
                "headline": headline,
            }
        )

    comparison_findings: list[str] = []
    if len(group_payloads) > 1:
        comparable = [group for group in group_payloads if group.get("max_successful_batch_size") is not None]
        if len(comparable) >= 2:
            best = max(comparable, key=lambda group: int(group.get("max_successful_batch_size") or -1))
            worst = min(comparable, key=lambda group: int(group.get("max_successful_batch_size") or -1))
            if best is not worst and best.get("max_successful_batch_size") != worst.get("max_successful_batch_size"):
                comparison_findings.append(
                    f"{best['gpu_model']} sustained batch size {best['max_successful_batch_size']}, while "
                    f"{worst['gpu_model']} topped out at batch size {worst['max_successful_batch_size']}."
                )
            if worst.get("min_oom_batch_size") is not None and best.get("max_successful_batch_size") is not None:
                if int(best["max_successful_batch_size"]) >= int(worst["min_oom_batch_size"]):
                    comparison_findings.append(
                        f"{worst['gpu_model']} hit OOM at batch size {worst['min_oom_batch_size']}, "
                        f"but {best['gpu_model']} handled that batch size."
                    )

    if len(group_payloads) == 1:
        only = group_payloads[0]
        payload = dict(only)
        payload["headline_findings"] = [str(only["headline"])]
    else:
        payload = {
            "policy": rows[0].get("policy") if rows else None,
            "groups": group_payloads,
            "comparison_findings": comparison_findings,
            "headline": comparison_findings[0] if comparison_findings else "",
            "headline_findings": comparison_findings,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "policy",
                "gpu_model",
                "gpu_count",
                "batch_size",
                "status",
                "failure_kind",
                "train_seconds",
                "peak_nvidia_smi_memory_mb",
                "peak_torch_memory_allocated_mb",
                "peak_torch_memory_reserved_mb",
                "avg_gpu_util_pct",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample")
    sample.add_argument("--output", required=True)
    sample.add_argument("--interval", type=float, default=1.0)
    sample.set_defaults(func=lambda args: emit_sample(Path(args.output), args.interval))

    summarize = subparsers.add_parser("summarize-run")
    summarize.add_argument("--samples", required=True)
    summarize.add_argument("--phase", required=True)
    summarize.add_argument("--policy", required=True)
    summarize.add_argument("--gpu-count", type=int, required=True)
    summarize.add_argument("--batch-size", type=int, required=True)
    summarize.add_argument("--status", required=True)
    summarize.add_argument("--failure-kind", required=True)
    summarize.add_argument("--run-seconds", type=float, required=True)
    summarize.add_argument("--work-units", required=True)
    summarize.add_argument("--work-count", type=int, required=True)
    summarize.add_argument("--warmup-trim-seconds", type=float, default=20.0)
    summarize.add_argument("--torch-memory-dir")
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(func=summarize_run)

    combine = subparsers.add_parser("combine-profile")
    combine.add_argument("--train-summary", required=True)
    combine.add_argument("--eval-summary", required=True)
    combine.add_argument("--eval-info")
    combine.add_argument("--output", required=True)
    combine.set_defaults(func=combine_profile)

    scaling = subparsers.add_parser("summarize-scaling")
    scaling.add_argument("--run-summaries", nargs="+", required=True)
    scaling.add_argument("--output", required=True)
    scaling.add_argument("--csv-output")
    scaling.set_defaults(func=summarize_scaling)

    memory = subparsers.add_parser("summarize-memory")
    memory.add_argument("--run-summaries", nargs="+", required=True)
    memory.add_argument("--output", required=True)
    memory.add_argument("--csv-output")
    memory.set_defaults(func=summarize_memory)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
