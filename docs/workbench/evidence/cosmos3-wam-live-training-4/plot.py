"""Render the fixed four-GPU execution window from recorded NVIDIA telemetry."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _records(root):
    receipt = json.loads((root / "evidence.json").read_text())
    for name, expected in receipt["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    proof = json.loads((root / "live-training-startup.json").read_text())
    window = json.loads((root / "live-training-window.json").read_text())
    with (root / "live-training-window.csv").open() as stream:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    groups = {
        index: [row for row in rows if int(row["index"]) == index] for index in range(8)
    }
    assert len(rows) == window["device_samples"] == 480
    assert all(len(rows) == 60 for rows in groups.values())
    assert sorted(
        gpu["training_rank"]
        for gpu in proof["gpus"]
        if gpu["training_rank"] is not None
    ) == list(range(4))
    return proof, groups


def _panel(axis, values, labels, color, title, limit, suffix):
    axis.barh(range(8), values, color=color, height=0.6)
    for index, value in enumerate(values):
        axis.text(value + limit * 0.018, index, f"{value:.1f}{suffix}", va="center")
    axis.set_yticks(range(8), labels)
    axis.invert_yaxis()
    axis.set_xlim(0, limit * 1.17)
    axis.set_title(title, loc="left", pad=16, weight="bold")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_facecolor("#f5f7fb")


def _figure(proof, groups):
    devices = sorted(proof["gpus"], key=lambda gpu: gpu["gpu_index"])
    labels = [
        f"GPU {gpu['gpu_index']} · "
        + (
            f"rank {gpu['training_rank']}"
            if gpu["training_rank"] is not None
            else "idle"
        )
        for gpu in devices
    ]
    colors = [
        "#087f75" if gpu["training_rank"] is not None else "#bcc8d1" for gpu in devices
    ]
    utilization = [
        statistics.mean(
            float(row["utilization.gpu [%]"].split()[0]) for row in groups[index]
        )
        for index in range(8)
    ]
    memory = [
        max(float(row["memory.used [MiB]"].split()[0]) for row in groups[index]) / 1024
        for index in range(8)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.3), dpi=180)
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(top=0.71, bottom=0.19, left=0.13, right=0.97, wspace=0.4)
    _panel(
        axes[0],
        utilization,
        labels,
        colors,
        "Mean GPU utilization · 60 seconds",
        100,
        "%",
    )
    _panel(axes[1], memory, labels, colors, "Sampled memory maximum · GiB", 179.1, "")
    _captions(figure, proof)
    return figure


def _captions(figure, proof):
    figure.text(
        0.07,
        0.93,
        "Four B200s training · eight-GPU VM retained",
        fontsize=21,
        weight="bold",
        color="#122538",
    )
    figure.text(
        0.07,
        0.865,
        "Actual NVIDIA telemetry · 2026-09-29 18:04:11–18:05:11 UTC",
        fontsize=12,
    )
    update = proof["last_observed_optimizer_update"]
    figure.text(
        0.07,
        0.813,
        f"Four native processes matched Slurm and ranks 0–3 · "
        f"update {update} observed before this window",
        fontsize=11,
    )
    figure.text(
        0.07,
        0.095,
        "480 samples, including all four idle devices. "
        "Four-rank NCCL all-reduce passed on one host.",
        fontsize=10,
    )
    figure.text(
        0.07,
        0.047,
        "Early execution evidence. Full-run duration, "
        "repeated scaling and policy quality are pending.",
        fontsize=10,
        color="#526274",
    )


def _main():
    root = Path(__file__).resolve().parent
    proof, groups = _records(root)
    figure = _figure(proof, groups)
    output = root / "training-snapshot.png"
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)
    result = {
        "matplotlib": matplotlib.__version__,
        "evidence_sha256": hashlib.sha256(
            (root / "evidence.json").read_bytes()
        ).hexdigest(),
        "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (root / "render-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    _main()
