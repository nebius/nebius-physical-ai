"""Render three completed 32-B200 repetitions from verified numeric records."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _verified_series(root, group):
    series = []
    for index, repeat in enumerate(group["repetitions"], 1):
        folder = root / f"repeat-{index}"
        report = (folder / "measurement.json").read_bytes()
        timing = (folder / "iteration-series.csv").read_bytes()
        assert hashlib.sha256(report).hexdigest() == repeat["measurement_sha256"]
        assert hashlib.sha256(timing).hexdigest() == repeat["iteration_series_sha256"]
        rows = list(csv.DictReader(timing.decode().splitlines()))
        assert [int(row["step"]) for row in rows] == list(range(52, 201))
        series.append(np.array([float(row["iteration_seconds"]) for row in rows]))
    return series


def _timing_axes(series, group):
    figure, axes = plt.subplots(1, 2, figsize=(14, 7.5), dpi=150)
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(left=0.08, right=0.97, top=0.69, bottom=0.23, wspace=0.3)
    colors = ["#087f75", "#146c94", "#b96617"]
    means = [row["step_mean_seconds"] for row in group["repetitions"]]
    axes[0].bar(range(3), means, color=colors, width=0.55)
    for index, mean in enumerate(means):
        axes[0].text(
            index, mean, f"{mean:.4f} s", ha="center", va="bottom", fontsize=12
        )
    axes[0].set_xticks(range(3), ["Repeat 1", "Repeat 2", "Repeat 3"])
    axes[0].set_ylim(0, max(means) * 1.2)
    axes[0].set_ylabel("Mean measured optimizer-update seconds")
    axes[0].set_title("Three independently launched timing runs", loc="left")
    for index, values in enumerate(series):
        ordered = np.sort(values)
        axes[1].step(
            ordered,
            np.arange(1, len(ordered) + 1) / len(ordered) * 100,
            where="post",
            label=f"Repeat {index + 1}",
            color=colors[index],
        )
    axes[1].set_xlabel("Measured optimizer-update seconds")
    axes[1].set_ylabel("Cumulative share of timed updates (%)")
    axes[1].set_title("All 149 measured updates per run", loc="left")
    axes[1].legend(loc="lower right")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    return figure


def _title(figure, group):
    spread = group["replicate_step_means_seconds"]
    figure.text(
        0.08,
        0.945,
        "Repeated WAM throughput on four B200 nodes",
        fontsize=22,
        weight="bold",
        color="#122538",
    )
    figure.text(
        0.08,
        0.885,
        "32 GPUs · NPA Soperator · 200 updates per run · nominal global batch 2,048 · seed 42",
        fontsize=12,
    )
    figure.text(
        0.08,
        0.79,
        f"Mean across runs: {spread['mean']:.4f} s/update  |  Run-mean sample SD: {spread['sample_standard_deviation']:.4f} s",
        fontsize=14,
        weight="bold",
    )


def _footer(figure, group):
    figure.text(
        0.08,
        0.135,
        f"Pooled native token throughput: {group['pooled_tokens_per_second']:,.0f} tokens/s across {group['measured_steps']} timed updates.",
        fontsize=11,
    )
    figure.text(
        0.08,
        0.09,
        "Initial 50-update warmup excluded; native timing begins at update 52. Final checkpoint writing is outside these timed updates.",
        fontsize=10,
    )
    figure.text(
        0.08,
        0.05,
        "One training seed. Run-mean SD is not a confidence interval. Full-schedule time, allocation and policy quality are separate.",
        fontsize=10,
        color="#526274",
    )
    figure.text(
        0.08,
        0.015,
        "This Soperator cohort has different driver, scheduler and storage conditions from the historical native-VM 8/16-GPU comparison.",
        fontsize=9,
        color="#526274",
    )


def _write(root, figure, raw):
    output = root / "repeated-timing.png"
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)
    receipt = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "repeated_timing_sha256": hashlib.sha256(raw).hexdigest(),
        "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (root / "render-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


def _main():
    root = Path(__file__).resolve().parent
    raw = (root / "repeated-timing.json").read_bytes()
    summary = json.loads(raw)
    if summary["schema"] != "npa.cosmos3.wam-repeated-timing.v1":
        raise ValueError("requires the standalone repeated-timing report")
    if summary["comparison"] is not None or set(summary["groups"]) != {"32"}:
        raise ValueError("requires only the recorded 32-GPU topology")
    group = summary["groups"]["32"]
    series = _verified_series(root, group)
    figure = _timing_axes(series, group)
    _title(figure, group)
    _footer(figure, group)
    _write(root, figure, raw)


if __name__ == "__main__":
    _main()
