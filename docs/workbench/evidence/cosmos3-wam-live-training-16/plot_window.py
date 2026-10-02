"""Render a fixed observation window from actual two-node GPU telemetry."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(root):
    window_raw = (root / "live-window.json").read_bytes()
    window = json.loads(window_raw)
    assert window["duration_seconds"] == 120 and window["gpus"] == 16
    records, receipts = {}, {}
    for rank in range(2):
        folder = root / f"window-node-{rank}"
        raw = (folder / "evidence.json").read_bytes()
        evidence = json.loads(raw)
        assert evidence["node_rank"] == rank
        assert (
            evidence["window"]["window_receipt_sha256"]
            == hashlib.sha256(window_raw).hexdigest()
        )
        source = evidence["samples"]
        compressed = (folder / source["file"]).read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == source["sha256"]
        decoded = gzip.decompress(compressed)
        assert hashlib.sha256(decoded).hexdigest() == source["uncompressed_sha256"]
        rows = list(csv.DictReader(decoded.decode().splitlines()))
        assert len(rows) == source["rows"]
        assert {int(row["gpu_index"]) for row in rows} == set(range(8))
        records[rank], receipts[rank] = rows, hashlib.sha256(raw).hexdigest()
    return window, records, receipts


def _heatmap(axis, records):
    sums, counts = np.zeros((16, 120)), np.zeros((16, 120))
    for rank, rows in records.items():
        for row in rows:
            index = rank * 8 + int(row["gpu_index"])
            column = int(float(row["elapsed_seconds"]))
            value = float(row["utilization_percent"])
            assert 0 <= column < 120 and 0 <= value <= 100
            sums[index, column] += value
            counts[index, column] += 1
    matrix = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=matrix, where=counts > 0)
    heat = axis.pcolormesh(
        np.arange(121),
        np.arange(17),
        matrix,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        rasterized=True,
    )
    labels = [f"Node {rank} / GPU {gpu}" for rank in range(2) for gpu in range(8)]
    axis.set_yticks(np.arange(16) + 0.5, labels, fontsize=9)
    axis.invert_yaxis()
    axis.axhline(8, color="#122538", lw=1.4)
    axis.set_xlim(0, 120)
    axis.set_title("Reported GPU utilization · one-second bins", loc="left", pad=12)
    return heat


def _memory(axis, records):
    colors = ["#087f75", "#345eb7"]
    for rank, rows in records.items():
        for gpu in range(8):
            selected = [row for row in rows if int(row["gpu_index"]) == gpu]
            x = [float(row["elapsed_seconds"]) for row in selected]
            y = [float(row["memory_used_mib"]) / 1024 for row in selected]
            axis.plot(
                x,
                y,
                color=colors[rank],
                alpha=0.65,
                lw=0.9,
                label=f"Node {rank}: eight GPUs" if gpu == 0 else None,
            )
    axis.set_xlim(0, 120)
    axis.set_xlabel("Seconds from the start of the fixed observation window")
    axis.set_ylabel("Device memory (GiB)")
    axis.set_title(
        "Sampled memory on each device · zoomed vertical scale", loc="left", pad=12
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", fontsize=9)
    axis.grid(axis="y", color="#dce3eb", lw=0.6)


def _captions(figure):
    figure.text(
        0.14,
        0.95,
        "Two minutes of real WAM training on sixteen B200s",
        fontsize=21,
        weight="bold",
        color="#122538",
    )
    figure.text(
        0.14,
        0.912,
        "Two reserved nodes · native Slurm training · fixed prospective window · actual nvidia-smi telemetry",
        fontsize=11,
    )
    figure.text(
        0.14,
        0.075,
        "Each heatmap cell averages available samples in one second; empty cells remain blank. Device samples have separate timestamps.",
        fontsize=9,
    )
    figure.text(
        0.14,
        0.045,
        "The separate process snapshots attribute all sixteen GPUs to this training command and Slurm job. No utilization-based window selection.",
        fontsize=9,
    )
    figure.text(
        0.14,
        0.017,
        "This is execution evidence, not throughput, model FLOP utilization or policy quality. Sampled memory does not establish allocator peaks.",
        fontsize=9,
        color="#526274",
    )


def _main():
    root = Path(__file__).resolve().parent
    _, records, receipts = _load(root)
    figure, axes = plt.subplots(
        2, 1, figsize=(15, 10.2), dpi=150, gridspec_kw={"height_ratios": [2.5, 1]}
    )
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(top=0.84, bottom=0.15, left=0.14, right=0.88, hspace=0.42)
    heat = _heatmap(axes[0], records)
    _memory(axes[1], records)
    color_axis = figure.add_axes([0.90, 0.48, 0.016, 0.30])
    figure.colorbar(heat, cax=color_axis, label="GPU utilization (%)")
    _captions(figure)
    output = root / "training-window.png"
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)
    receipt = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "node_evidence_sha256": receipts,
        "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (root / "window-render-manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    _main()
