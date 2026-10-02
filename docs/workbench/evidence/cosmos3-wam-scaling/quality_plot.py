"""Compare checkpoint-linked quality by observed training time and optimizer update."""

import hashlib
import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#345eb7", "#087f75"]
MARKERS = ["o", "s"]


def _load(root):
    reader = runpy.run_path(str(root.parent / "cosmos3-wam-quality-8/plot.py"))["_read"]
    pairs = [reader(root.parent / f"cosmos3-wam-quality-{gpus}") for gpus in (8, 16)]
    curves = [pair[0] for pair in pairs]
    assert [curve["gpus"] for curve in curves] == [8, 16]
    assert curves[0]["evaluation_contract"] == curves[1]["evaluation_contract"]
    for curve in curves:
        passing = [point for point in curve["points"] if point["successes"] >= 450]
        expected = passing[0]["step"] if passing else None
        assert curve["time_to_quality"]["step"] == expected
        assert curve["quality_threshold"] == 0.9
    return curves, {str(curve["gpus"]): digest for curve, digest in pairs}


def _line(axis, curve, color, marker, by_time):
    points = curve["points"]
    x = (
        [point["checkpoint_ready_train_seconds_interval"][0] / 3600 for point in points]
        if by_time
        else [point["step"] for point in points]
    )
    y = np.array([point["success_rate"] for point in points]) * 100
    bounds = np.array([point["wilson_95_interval"] for point in points]) * 100
    error = np.vstack((y - bounds[:, 0], bounds[:, 1] - y))
    axis.errorbar(
        x,
        y,
        yerr=error,
        color=color,
        marker=marker,
        capsize=5,
        lw=2,
        label=f"{curve['gpus']} training GPUs",
    )
    if by_time:
        for point, position in zip(points, x, strict=True):
            axis.annotate(
                f"{point['step']:,}",
                (position, point["success_rate"] * 100),
                xytext=(0, 15 if curve["gpus"] == 8 else -25),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=color,
            )


def _axis(axis, by_time):
    axis.axhline(90, color="#a75313", ls="--", lw=1.5, label="Predeclared 90% target")
    axis.set_ylim(0, 111)
    axis.set_yticks(range(0, 101, 20))
    axis.set_ylabel("Success across 500 trials (%)")
    axis.set_xlabel(
        "Observed training hours until weights were saved"
        if by_time
        else "Saved checkpoint (optimizer update)"
    )
    axis.set_title(
        "When are useful weights available?"
        if by_time
        else "Quality at the same scheduled updates",
        loc="left",
        pad=14,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", fontsize=9)
    if not by_time:
        axis.set_xticks([500, 1000, 1500, 2000], ["500", "1,000", "1,500", "2,000"])
        axis.set_xlim(390, 2110)


def _outcome(curve):
    first = curve["time_to_quality"]
    if first["step"] is None:
        return f"{curve['gpus']} GPUs: no saved checkpoint reached 90%"
    hours = first["train_seconds_interval"][0] / 3600
    return f"{curve['gpus']} GPUs: first passing checkpoint {first['step']:,}, available after {hours:.2f} h"


def _captions(figure, curves):
    figure.text(
        0.08,
        0.95,
        "Time to useful WAM policy quality · eight versus sixteen B200s",
        fontsize=21,
        weight="bold",
        color="#122538",
    )
    figure.text(
        0.08,
        0.899,
        "LIBERO-10 · four saved checkpoints per topology · 500 trials per checkpoint · all scheduled results shown",
        fontsize=11,
    )
    for index, curve in enumerate(curves):
        figure.text(
            0.08,
            0.842 - index * 0.040,
            _outcome(curve),
            fontsize=12,
            weight="bold",
            color=COLORS[index],
        )
    _footer(figure)


def _footer(figure):
    figure.text(
        0.08,
        0.10,
        "Left annotations identify optimizer updates. Lines guide the eye; an exact threshold crossing between checkpoints is not measured.",
        fontsize=9,
    )
    figure.text(
        0.08,
        0.06,
        "Quality was evaluated afterward on eight B200 policy servers for both training topologies. Evaluation time is separate from training time.",
        fontsize=9,
    )
    figure.text(
        0.08,
        0.025,
        "Error bars: pooled 95% Wilson intervals. One training seed and the same ten task types; no estimate of seed variability or unseen-task generalization.",
        fontsize=9,
        color="#526274",
    )


def _main():
    root = Path(__file__).resolve().parent
    curves, hashes = _load(root)
    figure, axes = plt.subplots(1, 2, figsize=(16, 8.5), dpi=150)
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(left=0.08, right=0.97, top=0.71, bottom=0.22, wspace=0.23)
    for index, axis in enumerate(axes):
        by_time = index == 0
        for curve, color, marker in zip(curves, COLORS, MARKERS, strict=True):
            _line(axis, curve, color, marker, by_time)
        _axis(axis, by_time)
    _captions(figure, curves)
    output = root / "quality-comparison.png"
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)
    receipt = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "quality_curve_sha256": hashes,
        "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (root / "quality-render-manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    _main()
