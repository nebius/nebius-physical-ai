"""Render recorded GPU and disk telemetry around native WAM checkpoint saves."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read(root, receipt):
    path = root / receipt["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == receipt["rows"]
    values = {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}
    assert all(np.isfinite(array).all() for array in values.values())
    return values


def _gpu(axis, values, ready):
    assert set(values["gpu_index"]) == set(range(8))
    for index in range(8):
        selected = values["gpu_index"] == index
        axis.plot(
            values["unix_seconds"][selected] - ready,
            values["utilization_percent"][selected],
            color="#087f75",
            alpha=0.55,
            lw=1,
        )
    axis.set_ylim(-3, 105)
    axis.set_yticks([0, 50, 100])
    axis.set_ylabel("GPU utilization (%)")


def _disk(axes, values, ready):
    elapsed = np.diff(values["monotonic_seconds"])
    written = np.diff(values["sectors_written"])
    read = np.diff(values["sectors_read"])
    busy = np.diff(values["io_milliseconds"])
    assert np.all(elapsed > 0) and min(written.min(), read.min(), busy.min()) >= 0
    assert np.all(values["sector_bytes"] == 512)
    times = values["unix_seconds"][1:] - ready
    axes[0].plot(
        times, written * 512 / 1024**2 / elapsed, color="#345eb7", lw=1, label="Write"
    )
    axes[0].plot(
        times, read * 512 / 1024**2 / elapsed, color="#8a98a8", lw=1, label="Read"
    )
    axes[0].set_ylabel("Whole-device I/O (MiB/s)")
    axes[0].set_ylim(-25, 850)
    axes[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
    percent = busy / elapsed / 10
    axes[1].plot(times, percent, color="#9c5b0b", lw=1)
    axes[1].set_ylim(-3, max(105, float(percent.max()) * 1.03))
    axes[1].set_yticks([0, 50, 100])
    axes[1].set_ylabel("Disk busy time (%)")


def _format(axes, event):
    duration = event["iteration_seconds"]
    for axis in axes:
        axis.axvspan(-duration, 0, color="#dae2ec", alpha=0.45, zorder=0)
        axis.axvline(0, color="#122538", ls="--", lw=1)
        axis.set_xlim(-duration - 45, 45)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_facecolor("#f5f7fb")
        axis.tick_params(labelsize=9)
    axes[0].set_title(
        f"Update {event['step']:,} · {duration:.2f} s iteration",
        fontsize=13,
        loc="left",
        pad=14,
    )
    axes[-1].set_xlabel("Seconds relative to logged checkpoint completion")


def _captions(figure):
    figure.text(
        0.08,
        0.95,
        "Checkpoint writes pause GPU training",
        fontsize=21,
        weight="bold",
        color="#122538",
    )
    figure.text(
        0.08,
        0.905,
        "Actual eight-B200 Slurm run · individual GPU traces · storage host disk counters",
        fontsize=11,
    )
    figure.text(
        0.08,
        0.07,
        "Gray: complete checkpoint-bearing iteration. Dashed line: native checkpoint-completion log (1 s resolution).",
        fontsize=10,
    )
    figure.text(
        0.08,
        0.03,
        "GPU readings are sampled about once per second. Disk rates use 512-byte sectors and monotonic elapsed time.",
        fontsize=10,
        color="#526274",
    )


def _main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "evidence.json").read_text())
    assert [event["step"] for event in manifest["events"]] == [1000, 1500]
    figure, axes = plt.subplots(3, 2, figsize=(14, 8.5), dpi=150, sharex="col")
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(
        top=0.82, bottom=0.16, left=0.08, right=0.97, wspace=0.2, hspace=0.24
    )
    for column, event in enumerate(manifest["events"]):
        _gpu(axes[0, column], _read(root, event["gpu"]), event["checkpoint_saved_unix"])
        _disk(
            axes[1:, column], _read(root, event["disk"]), event["checkpoint_saved_unix"]
        )
        _format(axes[:, column], event)
    _captions(figure)
    output = root / "checkpoint-io.png"
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)
    receipt = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "evidence_sha256": hashlib.sha256(
            (root / "evidence.json").read_bytes()
        ).hexdigest(),
        "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (root / "render-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    _main()
