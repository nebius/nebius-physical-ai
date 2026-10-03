"""Render a recorded, process-attributed eight-B200 training snapshot."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _panel(axis, values, total, color, title, unit):
    indices = list(range(len(values)))
    axis.barh(indices, total, color="#e3e9ef", height=0.58)
    axis.barh(indices, values, color=color, height=0.58)
    for index, value in enumerate(values):
        label = f"{value:.1f} {unit}" if unit == "GiB" else f"{value:g}%"
        axis.text(value + total * 0.025, index, label, va="center", fontsize=11)
    axis.set_yticks(indices, [f"GPU {index}" for index in indices])
    axis.invert_yaxis()
    axis.set_xlim(0, total * 1.2)
    if unit == "%":
        axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_title(title, loc="left", fontsize=13, pad=16, weight="bold")
    axis.set_xlabel(unit if unit == "GiB" else "Percent")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_facecolor("#f5f7fb")


def _figure(evidence):
    devices = sorted(evidence["gpus"], key=lambda item: item["gpu_index"])
    assert [item["gpu_index"] for item in devices] == list(range(8))
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.3), dpi=150)
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(top=0.73, bottom=0.19, left=0.075, right=0.96, wspace=0.28)
    _panel(
        axes[0],
        [item["utilization_percent"] for item in devices],
        100,
        "#087f75",
        "Reported GPU utilization",
        "%",
    )
    capacity = devices[0]["memory_total_mib"] / 1024
    assert all(item["memory_total_mib"] / 1024 == capacity for item in devices)
    _panel(
        axes[1],
        [item["memory_used_mib"] / 1024 for item in devices],
        capacity,
        "#345eb7",
        f"Device memory · {capacity:.1f} GiB total",
        "GiB",
    )
    _captions(figure, evidence)
    return figure


def _captions(figure, evidence):
    figure.text(
        0.075,
        0.93,
        "Cosmos3-Nano WAM · eight B200s training",
        fontsize=21,
        weight="bold",
        color="#122538",
    )
    stamp = evidence["captured_utc"][:19].replace("T", " ")
    figure.text(0.075, 0.875, f"Actual nvidia-smi readings · {stamp} UTC", fontsize=12)
    step = evidence["last_monitored_optimizer_update"]
    figure.text(
        0.075,
        0.827,
        f"Last monitored update: {step} / 2,000 · "
        "one Slurm worker · eight-rank NCCL preflight passed",
        fontsize=11,
    )
    figure.text(
        0.075,
        0.095,
        "All eight GPU processes matched this training command and its Slurm cgroup.",
        fontsize=10,
        color="#122538",
    )
    figure.text(
        0.075,
        0.05,
        "One live sample during an incomplete run. "
        "Utilization is not throughput; this does not establish policy quality.",
        fontsize=10,
        color="#526274",
    )


def _main():
    root = Path(__file__).resolve().parent
    evidence = json.loads((root / "evidence.json").read_text())
    figure = _figure(evidence)
    output = root / "training-snapshot.png"
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)
    receipt = {
        "matplotlib": matplotlib.__version__,
        "evidence_sha256": hashlib.sha256(
            (root / "evidence.json").read_bytes()
        ).hexdigest(),
        "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (root / "render-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    _main()
