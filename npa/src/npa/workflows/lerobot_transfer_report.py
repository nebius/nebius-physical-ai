"""Compare paired transfer trials and derive a validation-only demonstration queue."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from npa.workflows.lerobot_transfer_data import file_sha256, write_json


def compare_trials(evaluation: dict) -> dict:
    """Select on validation and test a preregistered augmentation hypothesis.

    Args:
        evaluation: Exact-checkpoint evaluation with the complete paired trial grid.
    Returns:
        Selection, uncertainty, held-out results, and next-demonstration requests.
    Raises:
        ValueError: Trials are missing, duplicated, nonfinite, or unpaired.
    """
    recipe = evaluation["recipe"]
    trials = _validated_trials(evaluation)
    rates = _success_rates(trials, recipe)
    selected = max(("baseline", "robust"), key=lambda arm: min(rates["validation"][arm].values()))
    interval = _paired_interval(trials, recipe, recipe["conditions"])
    clean = _paired_interval(trials, recipe, ["clean"])
    improved = (selected == "robust" and interval[0] > 0 and clean[0] >= -0.05
                and min(rates["test"][selected].values()) >= recipe["minimum_success"])
    return {
        "schema": "npa.lerobot-transfer.report.v1", "selected_arm": selected,
        "selection_rule": "highest worst-condition validation success; baseline wins ties",
        "success_rates": rates, "test_paired_delta_95ci": interval,
        "test_clean_delta_95ci": clean, "improvement_demonstrated": improved,
        "uncertainty": "10000 paired bootstrap resamples clustered by reset seed across conditions",
        "physical_robot_tested": False, "ready_for_robot_deployment": False,
        "next_demonstrations": _collection_queue(trials, selected),
        "checkpoint_hashes": evaluation["checkpoint_hashes"],
        "recipe_sha256": evaluation["recipe_sha256"],
        "limitations": [
            "PushT simulation robustness is not evidence of physical robot transfer.",
            "One training seed per arm; intervals describe reset variability, not training variability.",
            "Reserved demonstration episodes are excluded from training and normalization, not scored here.",
            "Only validation failures drive data collection; keep test resets sealed in later iterations.",
        ],
    }


def _validated_trials(evaluation: dict) -> dict:
    recipe = evaluation["recipe"]
    if recipe["conditions"] != ["clean", "dim", "warm", "delay"]:
        raise ValueError("Evaluation conditions differ from the benchmark contract")
    seeds = {split: set(range(recipe[f"{split}_seed"], recipe[f"{split}_seed"]
                              + recipe[f"{split}_episodes"])) for split in ("validation", "test")}
    if min(map(len, seeds.values())) < 2 or seeds["validation"] & seeds["test"]:
        raise ValueError("Validation and test require disjoint sets of at least two reset seeds")
    expected = {(arm, split, condition, seed) for arm in ("baseline", "robust")
                for split in seeds for condition in recipe["conditions"] for seed in seeds[split]}
    trials = {}
    for trial in evaluation["trials"]:
        key = tuple(trial[k] for k in ("arm", "split", "condition", "seed"))
        if key in trials or key not in expected or type(trial["success"]) is not bool:
            raise ValueError("Unexpected, duplicate, or invalid evaluation trial")
        if not np.isfinite([trial["sum_reward"], trial["max_reward"]]).all():
            raise ValueError("Nonfinite evaluation reward")
        if not 0 <= trial["max_reward"] <= 1:
            raise ValueError("PushT maximum reward must be in [0, 1]")
        trials[key] = trial
    if trials.keys() != expected:
        raise ValueError("Evaluation grid is incomplete")
    return trials


def _success_rates(trials: dict, recipe: dict) -> dict:
    rates = {}
    for split in ("validation", "test"):
        rates[split] = {}
        for arm in ("baseline", "robust"):
            rates[split][arm] = {}
            for condition in recipe["conditions"]:
                successes = [trial["success"] for key, trial in trials.items()
                             if key[:3] == (arm, split, condition)]
                rates[split][arm][condition] = float(np.mean(successes))
    return rates


def _paired_interval(trials: dict, recipe: dict, conditions: list[str]) -> list[float]:
    differences = []
    for seed in range(recipe["test_seed"], recipe["test_seed"] + recipe["test_episodes"]):
        paired = [int(trials["robust", "test", condition, seed]["success"])
                  - int(trials["baseline", "test", condition, seed]["success"])
                  for condition in conditions]
        differences.append(np.mean(paired))
    generator = np.random.default_rng(recipe["seed"])
    samples = generator.choice(differences, size=(10_000, len(differences)), replace=True)
    return np.quantile(samples.mean(axis=1), [0.025, 0.975]).tolist()


def _collection_queue(trials: dict, selected: str) -> list[dict]:
    requests = []
    instructions = {
        "clean": "Collect a successful expert recovery from this initial state.",
        "dim": "Collect an expert demonstration under reduced illumination with unchanged geometry.",
        "warm": "Collect an expert demonstration under a warm camera color response.",
        "delay": "Collect synchronized expert observations and actions with measured control latency.",
    }
    failures = [trial for key, trial in trials.items()
                if key[:2] == (selected, "validation") and not trial["success"]]
    for trial in sorted(failures, key=lambda row: (row["max_reward"], row["condition"], row["seed"])):
        requests.append({
            "condition": trial["condition"], "reset_seed": trial["seed"],
            "observed_max_reward": trial["max_reward"], "source_split": "validation",
            "request": instructions[trial["condition"]], "action_source_required": "expert",
        })
    return requests


def report_results(evaluated: Path, output: Path, run_id: str) -> None:
    """Write the measured report, collection queue, chart, and factual Rerun recording.

    Args:
        evaluated: Verified native rollout artifact directory.
        output: New report directory.
        run_id: Workflow run identity for Rerun.
    Returns:
        None.
    Raises:
        ValueError: Evaluation evidence is incomplete or a video cannot be decoded.
        OSError: A mandatory artifact cannot be written.
    """
    evaluation = json.loads((evaluated / "evaluation.json").read_text())
    report = compare_trials(evaluation)
    report["evaluation_sha256"] = file_sha256(evaluated / "evaluation.json")
    report["videos"] = _inspect_videos(evaluated)
    write_json(output / "next-demonstrations.json", report["next_demonstrations"])
    _plot_rates(report, output / "success.png")
    _record_trials(evaluation, output / "transfer.rrd", run_id)
    _inspect_recording(output / "transfer.rrd", run_id)
    report["recording_sha256"] = file_sha256(output / "transfer.rrd")
    write_json(output / "report.json", report)


def _inspect_videos(evaluated: Path) -> dict:
    import av

    videos = {}
    for video in sorted(evaluated.rglob("*.mp4")):
        with av.open(str(video)) as container:
            frames = sum(1 for _ in container.decode(video=0))
        if frames <= 0:
            raise ValueError("Native rollout video has no decodable frames")
        videos[video.relative_to(evaluated).as_posix()] = {"frames": frames, "sha256": file_sha256(video)}
    if len(videos) != 16:
        raise ValueError("Expected one real rollout video for each arm, split, and condition")
    return videos


def _plot_rates(report: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(report["success_rates"]["test"]["baseline"])
    figure, axis = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(conditions))
    for arm, offset, color in (("baseline", -0.18, "#687b8c"), ("robust", 0.18, "#008f7a")):
        rates = [report["success_rates"]["test"][arm][condition] for condition in conditions]
        axis.bar(positions + offset, rates, 0.36, label=arm, color=color)
    axis.set(xticks=positions, xticklabels=conditions, ylim=(0, 1), ylabel="Native task success",
             title="Held-out PushT simulation · matched reset seeds")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _record_trials(evaluation: dict, output: Path, run_id: str) -> None:
    import rerun as rr

    recording = rr.RecordingStream("npa_lerobot_transfer", recording_id=run_id)
    recording.save(str(output))
    recording.log("provenance", rr.TextDocument(json.dumps({
        "recipe_sha256": evaluation["recipe_sha256"], "physical_robot_tested": False,
        "timeline": "simulator reset seed; not capture time",
    })), static=True)
    for trial in evaluation["trials"]:
        recording.set_time("reset_seed", sequence=trial["seed"])
        entity = f"{trial['split']}/{trial['condition']}/{trial['arm']}"
        recording.log(f"{entity}/success", rr.Scalars(int(trial["success"])))
        recording.log(f"{entity}/max_reward", rr.Scalars(trial["max_reward"]))
    recording.flush()
    recording.disconnect()


def _inspect_recording(path: Path, run_id: str) -> None:
    binary = str(Path(sys.executable).parent / "rerun")
    verified = subprocess.run([binary, "rrd", "verify", str(path)],
                              capture_output=True, text=True, check=True)
    decoded = subprocess.run([binary, "rrd", "print", "-vv", str(path)],
                             capture_output=True, text=True, check=True)
    for expected in ("npa_lerobot_transfer", run_id, "reset_seed", "test/clean/robust/success"):
        if expected not in decoded.stdout:
            raise ValueError(f"Required Rerun content is missing: {expected}")
    path.with_suffix(".inspection.txt").write_text(verified.stdout + decoded.stdout)
