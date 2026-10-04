"""Render every sample in the fixed, attributed two-node GPU telemetry window."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "live-training-16"


def arrays(window):
    utilization = np.full((16, 120), np.nan)
    memory = np.full((16, 120), np.nan)
    for node in (0, 1):
        with (DATA / f"gpu-window-node-{node}.csv").open() as stream:
            rows = list(csv.reader(stream, skipinitialspace=True))
        assert len(rows) == window["samples_per_node"]
        for row in rows:
            stamp = datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f").replace(
                tzinfo=timezone.utc
            )
            seconds = stamp.timestamp()
            assert window["start_unix"] <= seconds < window["end_unix"]
            column = int(seconds - window["start_unix"])
            rank = node * 8 + int(row[1])
            assert row[2] == "NVIDIA B200" and 0 <= column < 120
            assert np.isnan(utilization[rank, column])
            utilization[rank, column] = float(row[4])
            memory[rank, column] = float(row[3]) / 1024
    assert np.isfinite(utilization).all() and np.isfinite(memory).all()
    assert ((0 <= utilization) & (utilization <= 100)).all()
    return utilization, memory


def heatmap(figure, axis, utilization, labels):
    heat = axis.imshow(
        utilization,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=100,
        extent=(0, 120, 15.5, -0.5),
        interpolation="nearest",
    )
    axis.set_yticks(range(16), labels)
    axis.set_xticks(range(0, 121, 20))
    axis.set_xlabel("Seconds into the fixed telemetry window")
    axis.set_title("GPU utilization (%)", loc="left", pad=14, weight="bold")
    axis.axhline(7.5, color="#173137", linewidth=1.4)
    figure.colorbar(heat, ax=axis, pad=0.02, shrink=0.82, label="NVML utilization (%)")


def memory_panel(axis, memory, labels):
    means = memory.mean(axis=1)
    colors = ["#446bff"] * 8 + ["#16967e"] * 8
    axis.barh(range(16), means, color=colors, height=0.7)
    axis.set_yticks(range(16), labels)
    axis.set_ylim(15.5, -0.5)
    axis.set_xlim(0, 65)
    axis.set_xlabel("Mean GPU memory used (GiB)")
    axis.set_title("Memory in the same window", loc="left", pad=14, weight="bold")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    for rank, value in enumerate(means):
        axis.text(
            value - 1,
            rank,
            f"{value:.1f}",
            ha="right",
            va="center",
            color="white",
            fontsize=9,
        )


def render(window, utilization, memory):
    labels = [f"Node {rank // 8} / GPU {rank % 8}" for rank in range(16)]
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        width_ratios=(2.3, 1),
        layout="constrained",
    )
    heatmap(figure, axes[0], utilization, labels)
    memory_panel(axes[1], memory, labels)
    figure.suptitle(
        "16 B200 GPUs running native WAM training\n"
        "All 1,920 one-second samples retained, including zero readings",
        fontsize=17,
    )
    means = [utilization[:8].mean(), utilization[8:].mean()]
    start = window["start_utc"].split(".")[0].replace("T", " ")
    figure.supxlabel(
        f"120 seconds from {start} UTC · Sample means: node 0 {means[0]:.1f}%, node 1 {means[1]:.1f}%\n"
        "Native trainer ranks verified before and after the window. This is not whole-run utilization or throughput.",
        fontsize=10,
    )
    figure.savefig(ROOT / "live-gpu-16-window.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    window = json.loads((DATA / "window.json").read_text())
    for name, checksum in window["source_files_sha256"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == checksum
    utilization, memory = arrays(window)
    render(window, utilization, memory)
    paths = [Path(__file__), DATA / "window.json", ROOT / "live-gpu-16-window.png"]
    result = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    (ROOT / "telemetry-render-manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
