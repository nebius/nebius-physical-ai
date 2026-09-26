"""Join complete checkpoint evaluations to observed training checkpoint times."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from evaluate import _checkpoint_identity, _task_results, _wilson
from report import _nodes, _timings

_SAVED = re.compile(
    r"^\[(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\|[^\n]+\] "
    r"Saved checkpoint to [^\n]*/iter_(\d+)\s*$",
    re.MULTILINE,
)
_STEPS = [500, 1000, 1500, 2000]


def _timestamp(stamp, started, ended):
    first = datetime.fromtimestamp(started, timezone.utc).year
    last = datetime.fromtimestamp(ended, timezone.utc).year
    candidates = []
    for year in range(first, last + 1):
        value = datetime.strptime(f"{year}-{stamp}", "%Y-%m-%d %H:%M:%S")
        value = value.replace(tzinfo=timezone.utc).timestamp()
        if started - 1 <= value <= ended + 1:
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError("checkpoint timestamp does not identify one UTC training time")
    return candidates[0]


def _checkpoint_times(log, node):
    ended = node["ended_unix"]
    started = ended - node["train_process_seconds"]
    timings = {}
    for match in _SAVED.finditer(log):
        step = int(match[2])
        if step in timings:
            raise ValueError("duplicate checkpoint completion event")
        stamp = _timestamp(match[1], started, ended)
        timings[step] = [max(0, stamp - started), stamp + 1 - started]
    if sorted(timings) != _STEPS:
        raise ValueError(
            "need native completion events for all four scheduled checkpoints"
        )
    if any(timings[a][0] >= timings[b][0] for a, b in zip(_STEPS, _STEPS[1:])):
        raise ValueError("checkpoint completion times are out of order")
    return timings


def _evaluation_settings(directory, run, settings):
    observed = json.loads((directory / "settings.json").read_text())
    if Path(observed["run_dir"]).resolve() != run.resolve():
        raise ValueError("evaluation belongs to a different training run")
    if observed["seed"] != settings["seed"] or observed["trials"] != 50:
        raise ValueError("evaluation seed or trial count differs from the protocol")
    if observed["workers"] not in range(1, 9):
        raise ValueError("invalid number of evaluation GPU workers")
    if observed["record_rollouts"]:
        raise ValueError(
            "illustrative video trials cannot replace the quality benchmark"
        )
    return observed


def _quality_counts(quality, observed):
    tasks = _task_results([quality], 50)
    successes = sum(task["successes"] for task in tasks)
    if (
        quality["schema"] != "npa.cosmos3.wam-quality.v1"
        or quality["step"] != observed["step"]
        or quality["full_500_trial_evaluation"] is not True
        or quality["trials"] != 500
        or quality["successes"] != successes
        or quality["success_rate"] != successes / 500
        or quality["threshold"] != 0.9
        or quality["threshold_met"] != (successes >= 450)
    ):
        raise ValueError("quality summary does not match its individual trials")
    elapsed = quality["elapsed_seconds"]
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("invalid evaluation-process duration")
    return tasks, successes


def _point(directory, run, settings, timings):
    observed = _evaluation_settings(directory, run, settings)
    raw = (directory / "quality.json").read_bytes()
    quality = json.loads(raw)
    tasks, successes = _quality_counts(quality, observed)
    step = quality["step"]
    if step not in timings:
        raise ValueError("evaluation step is outside the scheduled checkpoints")
    job = run / "output/cosmos3_wam/libero_10" / settings["name"]
    digest, _ = _checkpoint_identity(job / "checkpoints" / f"iter_{step:09d}")
    if digest != quality["model_manifest_sha256"]:
        raise ValueError("evaluated model hash differs from the training checkpoint")
    return {
        "step": step,
        "successes": successes,
        "trials": 500,
        "success_rate": successes / 500,
        "wilson_95_interval": _wilson(successes, 500),
        "threshold_met": successes >= 450,
        "checkpoint_ready_train_seconds_interval": timings[step],
        "evaluation_process_seconds": quality["elapsed_seconds"],
        "evaluation_process_gpu_hours": quality["elapsed_seconds"]
        * observed["workers"]
        / 3600,
        "model_manifest_sha256": digest,
        "quality_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "task_successes": {str(task["task_id"]): task["successes"] for task in tasks},
    }


def _first_passing(points):
    for index, point in enumerate(points):
        if point["threshold_met"]:
            return {
                "status": "observed_at_scheduled_checkpoint",
                "step": point["step"],
                "train_seconds_interval": point[
                    "checkpoint_ready_train_seconds_interval"
                ],
                "preceding_evaluated_step": points[index - 1]["step"]
                if index
                else None,
            }
    return {"status": "not_reached_at_any_scheduled_checkpoint", "step": None}


def _main(args):
    settings = json.loads((args.run_dir / "run.json").read_text())
    if settings["steps"] != 2000 or settings["profile"]:
        raise ValueError(
            "quality curve requires the full unprofiled 2,000-step schedule"
        )
    nodes = _nodes(args.run_dir, settings)
    log = (args.run_dir / "node-0.log").read_text()
    _timings(log, 50, 2000)
    timings = _checkpoint_times(log, nodes[0])
    points = [
        _point(path, args.run_dir, settings, timings) for path in args.evaluations
    ]
    points.sort(key=lambda point: point["step"])
    if [point["step"] for point in points] != _STEPS:
        raise ValueError("need one complete evaluation for each scheduled checkpoint")
    result = {
        "schema": "npa.cosmos3.wam-quality-curve.v1",
        "gpus": settings["gpus"],
        "training_process_seconds": max(
            node["train_process_seconds"] for node in nodes
        ),
        "quality_threshold": 0.9,
        "points": points,
        "time_to_quality": _first_passing(points),
        "interpretation": (
            "Checkpoint times use UTC native log timestamps with one-second resolution. "
            "Training-process start is reconstructed from the completion receipt. "
            "The first passing scheduled checkpoint is observed; an earlier exact crossing "
            "is not inferred. The pooled Wilson interval does not measure seed-to-seed "
            "or task-to-task variability. Evaluation time is reported separately."
        ),
    }
    args.output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    _main(parser.parse_args())
