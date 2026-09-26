"""Render full-run sampled GPU activity from a hash-verified numeric export."""

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read(root):
    raw = (root / "evidence.json").read_bytes()
    evidence = json.loads(raw)
    receipt = evidence["samples"]
    compressed = (root / receipt["file"]).read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == receipt["sha256"]
    samples = gzip.decompress(compressed)
    assert hashlib.sha256(samples).hexdigest() == receipt["uncompressed_sha256"]
    rows = list(csv.DictReader(io.StringIO(samples.decode())))
    assert len(rows) == receipt["rows"]
    for index, device in evidence["devices"].items():
        selected = [row for row in rows if row["gpu_index"] == index]
        assert len(selected) == device["samples"]
        for field, values in device["metrics"].items():
            observed = [float(row[field]) for row in selected]
            assert max(observed) == values["maximum"]
            assert min(observed) == values["minimum"]
            assert np.isclose(np.mean(observed), values["sample_mean"], rtol=1e-12)
    return evidence, rows, hashlib.sha256(raw).hexdigest()


def _bins(evidence, rows):
    duration = evidence["window"]["duration_seconds"]
    edges = np.arange(0, duration, 60.0)
    edges = np.append(edges, duration)
    shape = (evidence["window"]["gpus"], len(edges) - 1)
    counts, utilization = np.zeros(shape), np.zeros(shape)
    memory = np.full(shape, np.nan)
    for row in rows:
        index = int(row["gpu_index"])
        column = min(int(float(row["elapsed_seconds"]) // 60), shape[1] - 1)
        counts[index, column] += 1
        utilization[index, column] += float(row["utilization_percent"])
        value = float(row["memory_used_mib"]) / 1024
        memory[index, column] = np.fmax(memory[index, column], value)
    utilization = np.divide(
        utilization, counts, out=np.full(shape, np.nan), where=counts > 0
    )
    return edges / 3600, utilization, memory


def _panels(figure, axes, edges, utilization, memory):
    heat = axes[0].pcolormesh(
        edges,
        np.arange(utilization.shape[0] + 1),
        utilization,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        rasterized=True,
    )
    axes[0].set_yticks(np.arange(utilization.shape[0]) + 0.5)
    axes[0].set_yticklabels([f"GPU {index}" for index in range(utilization.shape[0])])
    axes[0].invert_yaxis()
    axes[0].set_title(
        "GPU utilization · arithmetic mean of samples per minute", loc="left"
    )
    box = axes[0].get_position()
    color_axis = figure.add_axes([box.x1 + 0.012, box.y0, 0.012, box.height])
    figure.colorbar(heat, cax=color_axis, label="Utilization (%)")
    centers = (edges[:-1] + edges[1:]) / 2
    for index, values in enumerate(memory):
        axes[1].plot(centers, values, label=f"GPU {index}", lw=1.2)
    axes[1].set_title("Device memory · largest observed sample per minute", loc="left")
    axes[1].set_ylabel("GiB used")
    axes[1].set_xlabel("Hours from native training-process start")
    axes[1].legend(ncol=4, fontsize=9, loc="lower right")
    axes[1].grid(alpha=0.2)
    for axis in axes:
        axis.set_xlim(0, edges[-1])


def _captions(figure, evidence):
    window = evidence["window"]
    maxima = [
        device["metrics"]["memory_used_mib"]["maximum"] / 1024
        for device in evidence["devices"].values()
    ]
    figure.text(
        0.085,
        0.955,
        f"The complete {window['gpus']}-B200 WAM training run",
        fontsize=22,
        weight="bold",
        color="#122538",
    )
    figure.text(
        0.085,
        0.912,
        f"{window['duration_seconds'] / 3600:.2f} hours · native GPU recorder · "
        f"{evidence['samples']['rows']:,} device samples",
        fontsize=12,
    )
    figure.text(
        0.085,
        0.085,
        f"Across GPUs, sampled device-memory maxima range from {min(maxima):.2f} "
        f"to {max(maxima):.2f} GiB. Includes startup and checkpoint saves.",
        fontsize=11,
    )
    figure.text(
        0.085,
        0.048,
        "Whole-device samples, not model FLOP utilization or allocator peaks. "
        "One-minute bins; the last bin is shorter. Raw numeric samples are included.",
        fontsize=10,
        color="#526274",
    )


def _checkpoints(root, axes, evidence):
    raw = (root / "checkpoint-times.json").read_bytes()
    events = json.loads(raw)["events"]
    assert [event["step"] for event in events] == [500, 1000, 1500, 2000]
    for event in events:
        seconds = event["checkpoint_ready_train_seconds_interval"][0]
        assert 0 < seconds < evidence["window"]["duration_seconds"]
        hours = seconds / 3600
        for axis in axes:
            axis.axvline(hours, color="#9c5b0b", ls="--", lw=1)
        axes[0].text(
            hours - 0.025,
            0.3,
            f"Saved {event['step']:,}",
            ha="right",
            va="top",
            fontsize=9,
            color="#593105",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    return hashlib.sha256(raw).hexdigest()


def _main():
    root = Path(__file__).resolve().parent
    evidence, rows, digest = _read(root)
    edges, utilization, memory = _bins(evidence, rows)
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=150)
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(left=0.085, right=0.93, top=0.855, bottom=0.18, hspace=0.4)
    _panels(figure, axes, edges, utilization, memory)
    checkpoint_digest = _checkpoints(root, axes, evidence)
    _captions(figure, evidence)
    gap = max(
        device["sampling_interval_max_seconds"]
        for device in evidence["devices"].values()
    )
    figure.text(
        0.085,
        0.022,
        f"Recorder target: one sample per second; largest observed gap: {gap:.1f} seconds. Dashed lines mark checkpoint completion.",
        fontsize=10,
        color="#526274",
    )
    output = root / "full-gpu-activity.png"
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)
    receipt = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "evidence_sha256": digest,
        "checkpoint_times_sha256": checkpoint_digest,
        "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (root / "render-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    _main()
