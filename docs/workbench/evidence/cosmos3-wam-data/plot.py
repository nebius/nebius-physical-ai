"""Render the recorded LIBERO sample and its native action transformation."""

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _cameras(figure, sample, root):
    axis = figure.add_axes([0.06, 0.465, 0.44, 0.34])
    frames = [
        np.asarray(Image.open(root / name)) for name in ("front.png", "wrist.png")
    ]
    axis.imshow(np.concatenate(frames, axis=1))
    axis.set_axis_off()
    axis.set_title("Recorded frame 0 · front camera + wrist camera", loc="left", pad=10)
    caption = "Task: " + sample["task"]
    figure.text(0.06, 0.435, textwrap.fill(caption, 68), fontsize=10, color="#334155")


def _pipeline(figure):
    axis = figure.add_axes([0.56, 0.47, 0.38, 0.35])
    axis.set_axis_off()
    stages = [
        (
            0.92,
            "Stored actions · 16 × 7",
            "translation delta (3) + axis-angle (3) + gripper (1)",
        ),
        (
            0.57,
            "Native rotation conversion · 16 × 10",
            "translation delta (3) + rotation 6D (6) + gripper (1)",
        ),
        (
            0.22,
            "Training actions · 16 × 10",
            "quantile_rot normalization using global_raw statistics",
        ),
    ]
    for position, title, detail in stages:
        axis.text(0, position, title, weight="bold", fontsize=13, color="#0f766e")
        axis.text(0, position - 0.09, detail, fontsize=9.5, color="#475569")
    for position in (0.7, 0.35):
        axis.annotate(
            "",
            xy=(0.45, position - 0.08),
            xytext=(0.45, position + 0.01),
            arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.5},
        )


def _actions(figure, sample):
    axis = figure.add_axes([0.075, 0.13, 0.405, 0.235])
    actions = np.asarray(sample["raw_actions"])
    seconds = np.arange(16) / sample["fps"]
    for column, label, color in zip(
        range(3), ("x", "y", "z"), ("#0284c7", "#0d9488", "#d97706")
    ):
        axis.plot(
            seconds, actions[:, column], "o-", label=label, color=color, markersize=3
        )
    axis.set(
        title="Stored translation commands in this window",
        xlabel="Seconds from window start",
        ylabel="Native command units",
    )
    axis.legend(ncol=3, frameon=False)
    axis.grid(alpha=0.16)


def _normalized(figure, sample):
    axis = figure.add_axes([0.57, 0.13, 0.34, 0.235])
    actions = np.asarray(sample["normalized_actions"])
    limit = max(1.0, np.ceil(np.abs(actions).max() * 10) / 10)
    heatmap = axis.imshow(
        actions, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
    )
    labels = ["x", "y", "z", "r₁x", "r₁y", "r₁z", "r₂x", "r₂y", "r₂z", "g"]
    axis.set(
        title="Actual normalized training-action values",
        ylabel="Action offset",
        xticks=range(10),
        xticklabels=labels,
        yticks=[0, 5, 10, 15],
    )
    for column in (2.5, 8.5):
        axis.axvline(column, color="white", linewidth=1.4)
    figure.colorbar(heatmap, cax=figure.add_axes([0.925, 0.13, 0.012, 0.235]))


def _header(figure, sample):
    figure.text(
        0.06,
        0.935,
        "What goes into Cosmos3-Nano WAM training",
        fontsize=24,
        weight="bold",
        color="#122538",
    )
    figure.text(
        0.06,
        0.889,
        "Actual LIBERO-10 sample · episode 0 · 20 Hz · 16 actions + 17 paired video frames",
        fontsize=12,
        color="#475569",
    )
    figure.text(
        0.06,
        0.85,
        f"Dataset: {sample['total_episodes']:,} episodes / {sample['total_frames']:,} frames"
        f"    |    Native train split: {sample['train_episodes']:,} episodes / {sample['training_windows']:,} windows",
        fontsize=10.5,
        color="#0f766e",
    )


def _save(figure, root):
    for suffix in ("png", "svg"):
        figure.savefig(
            root / ("data-pipeline." + suffix), facecolor=figure.get_facecolor()
        )
    path = root / "data-pipeline.svg"
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


def _main():
    root = Path(__file__).resolve().parent
    sample = json.loads((root / "sample.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure = plt.figure(figsize=(15, 9), dpi=160, facecolor="#f5f7fb")
    _header(figure, sample)
    _cameras(figure, sample, root)
    _pipeline(figure)
    _actions(figure, sample)
    _normalized(figure, sample)
    figure.text(
        0.06,
        0.045,
        "Recorded demonstrations and native CPU transforms; these are training inputs, not model predictions.",
        fontsize=10,
        color="#475569",
    )
    _save(figure, root)
    plt.close(figure)


if __name__ == "__main__":
    _main()
