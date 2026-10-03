"""Render measured CUDA activity without stacking overlapping kernel categories."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CATEGORIES = [
    "all_kernels",
    "matrix_multiply",
    "attention",
    "convolution",
    "collectives",
    "other",
]
LABELS = [
    "All observed kernels",
    "Matrix multiply ops",
    "Attention ops",
    "Convolution ops",
    "NCCL collectives",
    "Other / fused kernels",
]


def _read(root, rank, summary):
    source = rank["timeline"]
    raw = (root / source["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == source["sha256"]
    rows = list(csv.DictReader(raw.decode().splitlines()))
    assert len(rows) == source["rows"]
    devices = {int(row["device"]) for row in rows}
    assert len(devices) == 1, "render each physical device separately"
    device = next(iter(devices))
    all_rows = [row for row in rows if row["category"] == "all_kernels"]
    edges = [float(row["start_seconds"]) for row in all_rows]
    edges.append(float(all_rows[-1]["stop_seconds"]))
    assert np.all(np.diff(edges) > 0)
    matrix = np.zeros((len(CATEGORIES), len(edges) - 1))
    for category in CATEGORIES:
        selected = [row for row in rows if row["category"] == category]
        if not selected:
            continue
        assert [float(row["start_seconds"]) for row in selected] == edges[:-1]
        matrix[CATEGORIES.index(category)] = np.array(
            [float(row["kernel_busy_seconds"]) for row in selected]
        ) / np.diff(edges)
    assert np.isfinite(matrix).all() and matrix.min() >= 0 and matrix.max() <= 1 + 1e-8
    observed = float(np.sum(matrix[0] * np.diff(edges)))
    expected = summary["observed_kernel_busy_seconds_by_device"][str(device)]
    assert np.isclose(observed, expected, rtol=1e-9, atol=1e-8)
    return edges, matrix


def _timeline(axis, rank, edges, matrix):
    heat = axis.pcolormesh(
        edges,
        np.arange(len(CATEGORIES) + 1),
        matrix * 100,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        rasterized=True,
    )
    axis.set_yticks(np.arange(len(CATEGORIES)) + 0.5, LABELS)
    axis.invert_yaxis()
    axis.set_xlim(0, edges[-1])
    axis.set_xlabel("Seconds from the start of this rank's recorded profiler window")
    axis.set_title(
        f"Rank {rank['rank']} · {edges[-1]:.2f} s recorded window", loc="left", pad=12
    )
    for step in rank["steps"][1:]:
        axis.axvline(step["start_seconds"], color="#9c5b0b", ls="--", lw=1.5)
    return heat


def _captions(figure, evidence):
    figure.text(
        0.16,
        0.95,
        f"Inside two WAM training steps on {evidence['gpus']} B200s",
        fontsize=21,
        weight="bold",
        color="#122538",
    )
    figure.text(
        0.16,
        0.91,
        "Actual PyTorch CUDA traces · separate profiling run · one observed rank per node",
        fontsize=11,
    )
    figure.text(
        0.16,
        0.085,
        f"Color: kernel interval coverage in each {evidence['bin_seconds']:.2f} s bin. Dashed line: recorded optimizer-step boundary.",
        fontsize=10,
    )
    figure.text(
        0.16,
        0.055,
        "Groups use linked CPU operators, with kernel-name fallback. Rows can overlap; memory copies and unobserved activity are excluded.",
        fontsize=10,
    )
    figure.text(
        0.16,
        0.025,
        "This is an instrumented execution sample, not GPU utilization or a throughput benchmark. Rank windows have independent origins.",
        fontsize=9,
        color="#526274",
    )


def _main():
    root = Path(__file__).resolve().parent
    raw = (root / "evidence.json").read_bytes()
    evidence = json.loads(raw)
    summary_raw = (root / "profile-summary.json").read_bytes()
    assert hashlib.sha256(summary_raw).hexdigest() == evidence["profile_summary_sha256"]
    summary = json.loads(summary_raw)
    count = len(evidence["ranks"])
    figure, axes = plt.subplots(
        count, 1, figsize=(14, 5.5 + 2.8 * count), dpi=150, squeeze=False
    )
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(top=0.83, bottom=0.18, left=0.16, right=0.88, hspace=0.8)
    for axis, rank in zip(axes[:, 0], evidence["ranks"]):
        edges, matrix = _read(root, rank, summary["ranks"][str(rank["rank"])])
        heat = _timeline(axis, rank, edges, matrix)
        figure.colorbar(
            heat,
            ax=axis,
            label="Observed kernel coverage (%)",
            pad=0.025,
            fraction=0.025,
        )
    _captions(figure, evidence)
    output = root / "cuda-timeline.png"
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)
    receipt = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "evidence_sha256": hashlib.sha256(raw).hexdigest(),
        "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (root / "render-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    _main()
