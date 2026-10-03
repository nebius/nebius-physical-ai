"""Render the attributed live GPU sample; no throughput is inferred."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _panel(axis, labels, values, title, limit, suffix):
    axis.barh(labels, values, color="#446bff", height=0.6)
    axis.invert_yaxis()
    axis.set_xlim(0, limit)
    axis.set_xlabel(title, labelpad=10)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", alpha=0.15)
    axis.set_axisbelow(True)
    for row, value in enumerate(values):
        axis.text(
            value - limit * 0.025,
            row,
            f"{value:.1f}{suffix}",
            va="center",
            ha="right",
            color="white",
            fontsize=10,
        )


def main():
    root = Path(__file__).resolve().parent
    record = json.loads((root / "live-gpu-8.json").read_text())
    devices = sorted(record["gpus"], key=lambda gpu: gpu["gpu_index"])
    assert len(devices) == record["training_processes_matched_on_this_host"] == 8
    labels = [f"GPU {gpu['gpu_index']}" for gpu in devices]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")
    _panel(
        axes[0],
        labels,
        [gpu["utilization_percent"] for gpu in devices],
        "GPU utilization (%)",
        105,
        "%",
    )
    _panel(
        axes[1],
        labels,
        [gpu["memory_used_mib"] / 1024 for gpu in devices],
        "GPU memory used (GiB)",
        65,
        "",
    )
    step = record["last_monitored_optimizer_update"]
    stamp = record["captured_utc"].split(".")[0].replace("T", " ")
    figure.suptitle(
        f"Fresh WAM run on 8 B200 GPUs\nLive sample after update {step} · {stamp} UTC",
        fontsize=16,
    )
    figure.supxlabel(
        "All 8 processes matched the training command, Slurm job, and configuration.\n"
        "One sample per device; this is not a throughput measurement.",
        fontsize=10,
    )
    figure.savefig(root / "live-gpu-8.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
