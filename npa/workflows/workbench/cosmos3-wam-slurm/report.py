"""Reduce completed Slurm WAM runs into measured, comparison-checked scaling evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

_ITERATION = re.compile(
    r"(\d+) : iter_speed (\S+) seconds per iteration \| Loss: (\S+)"
)


def _timings(log, warmup, steps):
    if "skipping optimizer step" in log:
        raise ValueError("native training skipped an unstable optimizer step")
    values = {}
    for match in _ITERATION.finditer(log):
        iteration, seconds, loss = int(match[1]), float(match[2]), float(match[3])
        if iteration in values:
            raise ValueError(
                "duplicate iteration timing; do not concatenate resumed runs"
            )
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(loss):
            raise ValueError("non-finite loss or invalid step timing")
        if iteration > steps:
            raise ValueError("timing exceeds the requested optimizer steps")
        values[iteration] = seconds
    selected = [seconds for step, seconds in sorted(values.items()) if step > warmup]
    if len(selected) < 2:
        raise ValueError("need at least two measured optimizer steps after warmup")
    # The pinned IterSpeed callback warms up for 50 steps and initializes at 51.
    expected = set(range(max(warmup + 1, 52), steps + 1))
    if not expected.issubset(values):
        raise ValueError("missing optimizer-step timings")
    return selected


def _checkpoint(run, settings):
    job = run / "output/cosmos3_wam/libero_10" / settings["name"]
    checkpoint = job / "checkpoints" / f"iter_{settings['steps']:09d}"
    if not (job / "config.yaml").is_file():
        raise ValueError("missing native resolved config")
    for component in ("model", "optim", "scheduler", "trainer"):
        directory = checkpoint / component
        if not (directory / ".metadata").is_file() or not list(
            directory.glob("*.distcp")
        ):
            raise ValueError(f"incomplete final checkpoint: {component}")
    manifest = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file():
            manifest[str(path.relative_to(checkpoint))] = {
                "bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return digest, manifest


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nodes(run, settings):
    preflight = json.loads((run / "distributed-preflight.json").read_text())
    if (
        preflight["status"],
        preflight["backend"],
        preflight["world_size"],
        preflight["hosts"],
    ) != ("passed", "nccl", settings["gpus"], settings["nodes"]):
        raise ValueError("missing matching distributed NCCL proof")
    nodes = [
        json.loads((run / f"node-{rank}.finished.json").read_text())
        for rank in range(settings["nodes"])
    ]
    for rank, node in enumerate(nodes):
        if (
            node["run_sha256"]
            != hashlib.sha256((run / "run.json").read_bytes()).hexdigest()
        ):
            raise ValueError("run settings changed after execution")
        names = node["hardware"]["gpu_names"]
        if node["rank"] != rank or node["returncode"] != 0:
            raise ValueError("every allocated node must finish successfully")
        if len(names) != 8 or any("B200" not in name for name in names):
            raise ValueError("expected eight B200 GPUs on every node")
        if (
            not math.isfinite(node["train_process_seconds"])
            or node["train_process_seconds"] <= 0
        ):
            raise ValueError("invalid training process duration")
    if any(node["hardware"] != nodes[0]["hardware"] for node in nodes):
        raise ValueError("worker GPU/runtime versions differ")
    return nodes


def _summarize(run, warmup):
    settings = json.loads((run / "run.json").read_text())
    nodes = _nodes(run, settings)
    values = _timings((run / "node-0.log").read_text(), warmup, settings["steps"])
    digest, checkpoint = _checkpoint(run, settings)
    elapsed = max(node["train_process_seconds"] for node in nodes)
    comparable = {
        key: settings[key]
        for key in (
            "sources",
            "steps",
            "samples_per_rank",
            "global_batch",
            "seed",
            "profile",
        )
    }
    report = {
        "schema": "npa.cosmos3.wam-measurement.v1",
        "status": "measured",
        "comparison_contract": comparable,
        "nodes": settings["nodes"],
        "gpus": settings["gpus"],
        "hardware": nodes[0]["hardware"],
        "warmup_steps_excluded": warmup,
        "measured_steps": len(values),
        "step_mean_seconds": statistics.mean(values),
        "step_p50_seconds": statistics.median(values),
        "step_p95_seconds": statistics.quantiles(values, n=100, method="inclusive")[94],
        "train_process_seconds": elapsed,
        "training_gpu_hours": sum(node["train_process_seconds"] * 8 for node in nodes)
        / 3600,
        "checkpoint_manifest_sha256": digest,
        "quality_measured": False,
        "timing_scope": "optimizer iterations include checkpoint stalls; process includes model load and final save",
    }
    (run / "checkpoint-hashes.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
    return report


def _compare(report, baseline):
    if (
        report["comparison_contract"]["profile"]
        or baseline["comparison_contract"]["profile"]
    ):
        raise ValueError("profile runs cannot be used for throughput comparisons")
    if report["comparison_contract"] != baseline["comparison_contract"]:
        raise ValueError("dataset/model/steps/batch/seed differ from baseline")
    if report["hardware"] != baseline["hardware"] or baseline["gpus"] != 8:
        raise ValueError("baseline must use the same runtime and eight B200 GPUs")
    speedup = baseline["step_mean_seconds"] / report["step_mean_seconds"]
    report.update(
        speedup_vs_8_gpus=speedup,
        scaling_efficiency=speedup / (report["gpus"] / baseline["gpus"]),
    )


def _main(args):
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    report = _summarize(args.run_dir, args.warmup)
    if args.baseline:
        baseline = _summarize(args.baseline, args.warmup)
        _compare(report, baseline)
    (args.run_dir / "measurement.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--warmup", type=int, default=50)
    _main(parser.parse_args())
