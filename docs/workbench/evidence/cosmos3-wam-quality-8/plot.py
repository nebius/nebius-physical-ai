"""Render checkpoint-linked policy outcomes from completed native evaluations."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read(root):
    raw = (root / "quality-curve.json").read_bytes()
    curve = json.loads(raw)
    assert curve["schema"] == "npa.cosmos3.wam-quality-curve.v1"
    assert [point["step"] for point in curve["points"]] == [500, 1000, 1500, 2000]
    for point in curve["points"]:
        source = (root / f"quality-step-{point['step']}.json").read_bytes()
        assert hashlib.sha256(source).hexdigest() == point["quality_receipt_sha256"]
        quality = json.loads(source)
        assert quality["full_500_trial_evaluation"] is True
        assert quality["model_manifest_sha256"] == point["model_manifest_sha256"]
        model = json.loads(
            (root / f"model-hashes-step-{point['step']}.json").read_text()
        )
        model_digest = hashlib.sha256(
            json.dumps(model, sort_keys=True).encode()
        ).hexdigest()
        assert model_digest == point["model_manifest_sha256"]
        assert quality["successes"] == point["successes"]
        assert quality["trials"] == point["trials"] == 500
        assert set(point["task_successes"]) == set(map(str, range(10)))
        assert sum(point["task_successes"].values()) == point["successes"]
        for task in quality["task_results"]:
            assert task["successes"] == point["task_successes"][str(task["task_id"])]
            assert len(task["episode_results"]) == task["episodes"] == 50
            assert (
                sum(episode["success"] for episode in task["episode_results"])
                == task["successes"]
            )
            assert all(
                episode["steps"] > 0 and not episode["error"]
                for episode in task["episode_results"]
            )
    return curve, hashlib.sha256(raw).hexdigest()


def _aggregate(axis, curve):
    points = curve["points"]
    hours = [
        point["checkpoint_ready_train_seconds_interval"][0] / 3600 for point in points
    ]
    rates = np.array([point["success_rate"] for point in points]) * 100
    intervals = np.array([point["wilson_95_interval"] for point in points]) * 100
    errors = np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates))
    axis.errorbar(hours, rates, yerr=errors, fmt="o-", capsize=5, lw=2, color="#146c94")
    axis.axhline(90, color="#a75313", ls="--", lw=1.5, label="Predeclared 90% target")
    for hour, rate, point in zip(hours, rates, points):
        axis.annotate(
            f"{point['step']:,} updates\n{point['successes']}/500",
            (hour, rate),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.set_xlim(min(hours) - 0.65, max(hours) + 0.65)
    axis.set_ylim(0, 113)
    axis.set_yticks(range(0, 101, 20))
    axis.set_xlabel("Observed training hours until checkpoint was saved")
    axis.set_ylabel("Task success (%)")
    axis.set_title("All 500 trials per checkpoint", loc="left")
    axis.legend(loc="lower right", fontsize=9)
    axis.grid(alpha=0.2)


def _tasks(figure, axis, curve):
    matrix = np.array(
        [
            [point["task_successes"][str(task)] for point in curve["points"]]
            for task in range(10)
        ]
    )
    heat = axis.imshow(
        matrix / 50 * 100, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto"
    )
    axis.set_xticks(range(4), [f"{point['step']:,}" for point in curve["points"]])
    axis.set_yticks(range(10), [f"Task {task}" for task in range(10)])
    axis.set_xlabel("Training checkpoint (optimizer updates)")
    axis.set_title("Per-task successes / 50 trials", loc="left")
    for task in range(10):
        for index in range(4):
            axis.text(
                index,
                task,
                f"{matrix[task, index]}/50",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if matrix[task, index] >= 32 else "#122538",
            )
    figure.colorbar(heat, ax=axis, label="Success (%)", pad=0.03, fraction=0.04)


def _captions(figure, curve):
    outcome = curve["time_to_quality"]
    if outcome["step"] is None:
        result = "No scheduled checkpoint reached the predeclared 90% target."
    else:
        hours = outcome["train_seconds_interval"][0] / 3600
        result = f"First scheduled checkpoint at or above 90%: update {outcome['step']:,}, after {hours:.2f} training hours."
    figure.text(
        0.07,
        0.955,
        f"Policy quality after training on {curve['gpus']} B200s",
        fontsize=22,
        weight="bold",
        color="#122538",
    )
    figure.text(0.07, 0.905, result, fontsize=12)
    figure.text(
        0.07,
        0.085,
        f"Same ten tasks, 50 initial states per task and evaluation seed {curve['evaluation_contract']['seed']}. "
        "All scheduled checkpoints are shown, including failures.",
        fontsize=10,
    )
    figure.text(
        0.07,
        0.052,
        "Error bars: pooled 95% Wilson intervals; they do not measure task or seed variability. "
        "Training timestamps have one-second resolution.",
        fontsize=10,
        color="#526274",
    )
    figure.text(
        0.07,
        0.022,
        "Training time is retrospective: evaluations ran separately. "
        "The first passing saved checkpoint does not identify an exact quality-crossing time.",
        fontsize=10,
        color="#526274",
    )


def _main():
    root = Path(__file__).resolve().parent
    curve, digest = _read(root)
    figure, axes = plt.subplots(
        1, 2, figsize=(15, 8), dpi=150, gridspec_kw={"width_ratios": [1.1, 1]}
    )
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(left=0.07, right=0.94, top=0.84, bottom=0.19, wspace=0.3)
    _aggregate(axes[0], curve)
    _tasks(figure, axes[1], curve)
    _captions(figure, curve)
    output = root / "policy-quality.png"
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)
    receipt = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "quality_curve_sha256": digest,
        "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (root / "render-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    _main()
