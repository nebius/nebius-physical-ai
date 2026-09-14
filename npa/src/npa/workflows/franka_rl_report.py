"""Publish measured Franka robustness results and visual LeRobot rollout evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np

from npa.workflows.lerobot_transfer_data import file_sha256, write_json


def summarize(evaluation: dict) -> dict:
    """Check paired test coverage and compute success rates and qualification.

    Args:
        evaluation: Actual simulator trials with their sealed recipe.
    Returns:
        Rates, bootstrap interval, and explicit simulation-only qualification.
    Raises:
        ValueError: Trial coverage, checkpoint lineage, or values are invalid.
    """
    recipe, rows = evaluation["recipe"], evaluation["trials"]
    expected = {(arm, condition, index) for arm in ("initial", "trained")
                for condition in recipe["conditions"] for index in range(recipe["eval_episodes"])}
    indexed = {(row["arm"], row["condition"], row["env_index"]): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Franka test grid contains missing, extra, or duplicate episodes")
    selected = evaluation["selection"]["selected_checkpoint_sha256"]
    for row in rows:
        if row["split"] != "test" or row["reset_seed"] != recipe["test_seed"]:
            raise ValueError("Franka test resets do not match the sealed protocol")
        if type(row["success"]) is not bool or not np.isfinite(row["closest_distance_m"]):
            raise ValueError("Franka episode metrics are invalid")
        if row["arm"] == "trained" and row["checkpoint_sha256"] != selected:
            raise ValueError("Test did not use the selected Franka checkpoint")
    _validate_pairing(indexed, recipe)
    rates = {arm: {condition: float(np.mean([indexed[arm, condition, i]["success"]
             for i in range(recipe["eval_episodes"])])) for condition in recipe["conditions"]}
             for arm in ("initial", "trained")}
    return {"schema": "npa.franka-rl.report.v1", "success_rates": rates,
            "paired_delta_95ci": _interval(indexed, recipe),
            "simulation_qualified": all(rate >= recipe["minimum_success"] for rate in rates["trained"].values()),
            "physical_robot_tested": False, "ready_for_robot_deployment": False,
            "selected_checkpoint_sha256": selected, "selection": evaluation["selection"],
            "test_episodes": len(rows), "validation_episodes": len(evaluation["validation"]),
            "limitations": ["Privileged simulator-state policy; physical perception and control remain unvalidated.",
                            "One training seed; uncertainty covers reset variability only.",
                            "Task is lift and hold at a commanded goal, not released placement."]}


def _validate_pairing(indexed: dict, recipe: dict) -> None:
    for index in range(recipe["eval_episodes"]):
        states = {indexed[arm, condition, index]["initial_state_sha256"]
                  for arm in ("initial", "trained") for condition in recipe["conditions"]}
        if len(states) != 1:
            raise ValueError("Franka test arms or conditions began from different physical resets")
    for row in indexed.values():
        if row["success"] != (row["longest_stable_steps"] >= recipe["stable_steps"]):
            raise ValueError("Franka success disagrees with the sustained physical event")


def _interval(indexed: dict, recipe: dict) -> list[float]:
    differences = [np.mean([int(indexed["trained", c, i]["success"]) - int(indexed["initial", c, i]["success"])
                           for c in recipe["conditions"]]) for i in range(recipe["eval_episodes"])]
    generator = np.random.default_rng(recipe["seed"])
    draws = generator.choice(differences, size=(10_000, len(differences)), replace=True)
    return np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()


def _convert_trajectories(source: Path, output: Path) -> None:
    from npa.adapter.isaac_lab_lerobot import LeRobotFeatureSpec, convert
    from npa.workflows.franka_rl_recording import write_recording

    metadata = json.loads((source / "meta.json").read_text())
    spec = LeRobotFeatureSpec(metadata["state_names"], metadata["action_names"], "franka")
    convert(source, output / "lerobot", fps=round(metadata["fps"]), robot_type="franka",
            task="Lift the cube and hold at its commanded goal", spec=spec)
    shutil.copy2(source / "meta.json", output / "capture.json")
    counts = write_recording(output / "lerobot", output / "franka.rrd", metadata)
    write_json(output / "recording-validation.json", {"run_id": metadata["run_id"], "entity_counts": counts})


def _plot(report: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    conditions = list(report["success_rates"]["trained"])
    positions = np.arange(len(conditions))
    figure, axis = plt.subplots(figsize=(8, 4))
    for arm, offset, color in (("initial", -0.18, "#8398ad"), ("trained", 0.18, "#167f6a")):
        rates = [report["success_rates"][arm][condition] for condition in conditions]
        axis.bar(positions + offset, rates, 0.36, label=arm, color=color)
    axis.set(xticks=positions, xticklabels=conditions, ylim=(0, 1), ylabel="Lift-and-hold success rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "success.png", dpi=160)
    plt.close(figure)


def report_results(evaluated: Path, output: Path) -> None:
    """Write verified trial tables, qualification, LeRobotDataset, and real Rerun evidence.

    Args:
        evaluated: Verified simulation evaluation and trajectory stage.
        output: New report directory.
    Returns:
        None.
    Raises:
        ValueError: Evaluation or trajectory evidence is invalid.
        RuntimeError: Native conversion fails.
        OSError: Artifact files cannot be written.
    """
    evaluation = json.loads((evaluated / "evaluation.json").read_text())
    report = summarize(evaluation)
    output.mkdir(parents=True)
    _convert_trajectories(evaluated / "trajectories", output)
    _plot(report, output)
    with (output / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(evaluation["trials"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(evaluation["trials"])
    report["recording_sha256"] = file_sha256(output / "franka.rrd")
    report["evaluation_sha256"] = file_sha256(evaluated / "evaluation.json")
    report["recipe"] = evaluation["recipe"]
    write_json(output / "report.json", report)
