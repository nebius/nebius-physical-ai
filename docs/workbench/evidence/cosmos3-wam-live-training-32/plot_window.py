"""Render the prospectively selected four-node window from verified GPU samples."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _load(root):
    raw = (root / "live-window.json").read_bytes()
    window = json.loads(raw)
    assert window["gpus"] == 32 and window["duration_seconds"] == 120
    records, hashes = {}, {}
    for node in range(4):
        folder = root / f"window-node-{node}"
        receipt = (folder / "evidence.json").read_bytes()
        evidence = json.loads(receipt)
        assert evidence["node_rank"] == node
        assert evidence["window"]["window_receipt_sha256"] == _digest(raw)
        source = evidence["samples"]
        compressed = (folder / source["file"]).read_bytes()
        assert _digest(compressed) == source["sha256"]
        decoded = gzip.decompress(compressed)
        assert _digest(decoded) == source["uncompressed_sha256"]
        rows = list(csv.DictReader(decoded.decode().splitlines()))
        assert len(rows) == source["rows"]
        assert {int(row["gpu_index"]) for row in rows} == set(range(8))
        records[node], hashes[node] = rows, _digest(receipt)
    return records, hashes


def _heatmap(axis, records):
    sums, counts = np.zeros((32, 120)), np.zeros((32, 120))
    for node, rows in records.items():
        for row in rows:
            index = node * 8 + int(row["gpu_index"])
            second = int(float(row["elapsed_seconds"]))
            value = float(row["utilization_percent"])
            assert 0 <= second < 120 and 0 <= value <= 100
            sums[index, second] += value
            counts[index, second] += 1
    values = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=values, where=counts > 0)
    heat = axis.pcolormesh(
        np.arange(121), np.arange(33), values, cmap="YlGnBu", vmin=0, vmax=100
    )
    labels = [f"{node} / {gpu}" for node in range(4) for gpu in range(8)]
    axis.set_yticks(np.arange(32) + 0.5, labels, fontsize=8)
    axis.set_ylabel("Slurm node rank / physical GPU index", fontsize=9)
    axis.invert_yaxis()
    for edge in (8, 16, 24):
        axis.axhline(edge, color="#122538", lw=1.2)
    axis.set_title(
        "Reported GPU utilization · one-second bins", loc="left", fontsize=11
    )
    axis.set_xlim(0, 120)
    return heat


def _memory(axis, records):
    colors = ("#087f75", "#345eb7", "#9853a1", "#b97018")
    for node, rows in records.items():
        for gpu in range(8):
            selected = [row for row in rows if int(row["gpu_index"]) == gpu]
            axis.plot(
                [float(row["elapsed_seconds"]) for row in selected],
                [float(row["memory_used_mib"]) / 1024 for row in selected],
                color=colors[node],
                alpha=0.6,
                lw=0.8,
                label=f"Node {node}" if gpu == 0 else None,
            )
    axis.set(
        xlim=(0, 120),
        xlabel="Seconds since the fixed window started",
        ylabel="Device memory (GiB)",
    )
    axis.set_title("Sampled memory · zoomed vertical scale", loc="left", fontsize=11)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=4, fontsize=8, loc="upper right")
    axis.grid(axis="y", color="#dce3eb", lw=0.6)


def _captions(figure):
    figure.text(
        0.08, 0.965, "Real WAM training on 32 B200 GPUs", fontsize=16, weight="bold"
    )
    figure.text(
        0.08,
        0.939,
        "Four reserved nodes · NPA Soperator · 32 attributed training processes",
        fontsize=9,
    )
    figure.text(
        0.08,
        0.918,
        "30 September 2026 · 02:41:40.949–02:43:40.949 UTC · 3,300 device samples",
        fontsize=8,
    )
    figure.text(
        0.08,
        0.081,
        "Each cell averages available samples; unsampled seconds remain blank.",
        fontsize=8,
    )
    figure.text(
        0.08,
        0.061,
        "The window was fixed prospectively after warmup, without selecting for utilization.",
        fontsize=8,
    )
    limits = (
        "Device memory is sampled, not an allocator peak. Utilization is not model FLOP utilization.",
        "This figure proves execution; full-run timing and checkpoint quality are separate measurements.",
    )
    for position, label in zip((0.041, 0.021), limits, strict=True):
        figure.text(0.08, position, label, fontsize=7.5)


def _main():
    root = Path(__file__).resolve().parent
    records, hashes = _load(root)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.6, 10.4),
        dpi=240,
        gridspec_kw={"height_ratios": [4.3, 1]},
    )
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(top=0.88, bottom=0.14, left=0.15, right=0.86, hspace=0.32)
    heat = _heatmap(axes[0], records)
    _memory(axes[1], records)
    color_axis = figure.add_axes([0.89, 0.46, 0.018, 0.36])
    figure.colorbar(heat, cax=color_axis, label="GPU utilization (%)")
    _captions(figure)
    output = root / "training-window.png"
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)
    receipt = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "node_evidence_sha256": hashes,
        "image_sha256": _digest(output.read_bytes()),
    }
    (root / "render-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    _main()
