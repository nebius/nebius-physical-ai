"""Render the deployed native cluster from its committed readiness receipt."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
INK = "#172023"
MUTED = "#586669"
YELLOW = "#f2f95b"


def box(ax, xy, size, title, lines, color="#f1f4f1"):
    x, y = xy
    width, height = size
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            facecolor=color,
            edgecolor="#c9d2cf",
            linewidth=1,
        )
    )
    ax.text(x + 0.015, y + height - 0.045, title, fontsize=12, weight="bold", color=INK)
    ax.text(
        x + 0.015,
        y + height - 0.082,
        "\n".join(lines),
        fontsize=10,
        va="top",
        color=MUTED,
        linespacing=1.65,
    )


def arrow(ax, start, end, bidirectional=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="<->" if bidirectional else "->",
            mutation_scale=13,
            linewidth=1.6,
            color=INK,
        )
    )


def worker(ax, x, name, note):
    box(ax, (x, 0.32), (0.40, 0.23), name, [note], "#ffffff")
    for index in range(8):
        gx = x + 0.017 + index * 0.046
        ax.add_patch(
            FancyBboxPatch(
                (gx, 0.347),
                0.037,
                0.063,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                facecolor=YELLOW,
                edgecolor=INK,
                linewidth=0.6,
            )
        )
        ax.text(gx + 0.0185, 0.379, str(index), ha="center", va="center", fontsize=9)


def header(ax):
    ax.text(
        0.02,
        0.965,
        "Fresh WAM reproduction on Nebius",
        fontsize=24,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.02,
        0.922,
        "Deployed in us-central1 · 16 reserved NVIDIA B200 GPUs · native Slurm",
        fontsize=12,
        color=MUTED,
    )


def inputs(ax, data):
    box(
        ax,
        (0.02, 0.68),
        (0.275, 0.17),
        "Pinned inputs",
        [
            "Cosmos 3 Nano + LIBERO-10",
            f"{data['episodes']:,} episodes · {data['frames']:,} frames",
        ],
    )
    box(
        ax,
        (0.36, 0.68),
        (0.275, 0.17),
        "Shared preparation",
        ["Frozen framework and weights", "Data validation + CUDA probes"],
    )
    box(
        ax,
        (0.70, 0.68),
        (0.275, 0.17),
        "Slurm allocation",
        ["1 task / node → torchrun", "8 trainer ranks / node"],
    )
    arrow(ax, (0.303, 0.765), (0.348, 0.765))
    arrow(ax, (0.644, 0.765), (0.688, 0.765))
    arrow(ax, (0.837, 0.665), (0.837, 0.57))
    ax.plot([0.22, 0.837], [0.59, 0.59], color=INK, linewidth=1.6)
    arrow(ax, (0.22, 0.59), (0.22, 0.565))


def training(ax):
    ax.text(
        0.02,
        0.625,
        "One worker for 8-GPU runs; both workers for 16-GPU runs",
        fontsize=12,
        weight="bold",
        color=INK,
    )
    worker(
        ax, 0.02, "Worker 0 · 8 × B200", "Slurm controller / accounting + NFS server"
    )
    worker(ax, 0.575, "Worker 1 · 8 × B200", "Slurm worker + NFS client")
    arrow(ax, (0.435, 0.43), (0.56, 0.43), bidirectional=True)
    ax.text(0.498, 0.48, "NCCL", ha="center", fontsize=10, weight="bold", color=INK)
    ax.text(0.498, 0.465, "InfiniBand", ha="center", va="top", fontsize=9, color=MUTED)
    ax.text(
        0.02,
        0.27,
        "Shared 2 TiB storage: runtime, input data, checkpoints, and run records",
        fontsize=11,
        color=MUTED,
    )


def outputs(ax):
    box(
        ax,
        (0.02, 0.075),
        (0.44, 0.135),
        "Checkpoint evaluation",
        ["Separate GPU jobs · 500 trials / checkpoint"],
    )
    box(
        ax,
        (0.535, 0.075),
        (0.44, 0.135),
        "Private object archive",
        ["Upload → full GET + SHA-256 → local pruning"],
    )
    arrow(ax, (0.473, 0.14), (0.522, 0.14))
    ax.text(
        0.02,
        0.018,
        "Evaluation and archive I/O run outside timed training. Topology diagram; completion is recorded separately.",
        fontsize=9,
        color=MUTED,
    )


def render():
    readiness = json.loads((ROOT / "readiness.json").read_text())
    data = json.loads((ROOT / "inputs.json").read_text())["data"]
    deployment = readiness["deployment"]
    assert (deployment["nodes"], deployment["gpus_per_node"]) == (2, 8)
    assert deployment["controller_disk_gib"] == 2048
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "svg.hashsalt": "wam-repro-20261002"}
    )
    fig, ax = plt.subplots(figsize=(13, 8), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    header(ax)
    inputs(ax, data)
    training(ax)
    outputs(ax)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.98)
    for extension in ("png", "svg"):
        metadata = {"Date": None} if extension == "svg" else {"Software": "Matplotlib"}
        fig.savefig(ROOT / f"reproduction-topology.{extension}", metadata=metadata)
    svg = ROOT / "reproduction-topology.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    render()
    names = [
        "plot-topology.py",
        "readiness.json",
        "inputs.json",
        "reproduction-topology.png",
        "reproduction-topology.svg",
    ]
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names
    }
    (ROOT / "topology-render-manifest.json").write_text(
        json.dumps({"matplotlib": matplotlib.__version__, "sha256": hashes}, indent=2)
        + "\n"
    )
