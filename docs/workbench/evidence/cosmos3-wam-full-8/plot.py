"""Render the completed eight-B200 training timings from verified numeric evidence."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def _load(root):
    evidence = json.loads((root / "evidence.json").read_text())
    for name, digest in evidence["files_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    report = json.loads((root / "measurement.json").read_text())
    manifest = json.loads((root / "checkpoint-hashes.json").read_text())
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert digest == report["checkpoint_manifest_sha256"]
    with (root / "iteration-series.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(row["step"]) for row in rows]
    seconds = [float(row["iteration_seconds"]) for row in rows]
    assert steps == list(range(52, 2001))
    assert statistics.mean(seconds) == report["step_mean_seconds"]
    assert statistics.median(seconds) == report["step_p50_seconds"]
    return evidence, report, steps, seconds


def _header(figure, report):
    elapsed = round(report["train_process_seconds"])
    hours, remaining = divmod(elapsed, 3600)
    minutes, seconds = divmod(remaining, 60)
    figure.text(
        0.08,
        0.94,
        "A complete WAM training run on eight B200s",
        fontsize=21,
        weight="bold",
    )
    figure.text(
        0.08,
        0.895,
        "2,000 optimizer updates · nominal global batch 2,048 · one Slurm node · seed 42",
        fontsize=11,
    )
    values = [
        (0.08, "Native training process", f"{hours} h {minutes:02d} m {seconds:02d} s"),
        (0.43, "Training process GPU-hours", f"{report['training_gpu_hours']:.2f}"),
        (
            0.75,
            "Median / p95 update",
            f"{report['step_p50_seconds']:.2f} / {report['step_p95_seconds']:.2f} s",
        ),
    ]
    for left, label, value in values:
        figure.text(left, 0.81, label, fontsize=10, color="#526274")
        figure.text(left, 0.755, value, fontsize=19, weight="bold", color="#122538")


def _timings(axis, steps, seconds):
    axis.plot(steps, seconds, color="#087f75", lw=1, zorder=2)
    for step in (500, 1000, 1500, 2000):
        value = seconds[steps.index(step)]
        axis.scatter([step], [value], color="#b96617", s=32, zorder=3)
        axis.annotate(
            f"Update {step:,}\n{value:.2f} s",
            (step, value),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color="#784409",
        )
    axis.set_yscale("log")
    axis.set_ylim(9, 1000)
    axis.set_xlim(0, 2110)
    axis.set_yticks([10, 20, 50, 100, 200, 500])
    axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.set_xticks([0, 500, 1000, 1500, 2000])
    axis.set_xlabel("Optimizer update", fontsize=11)
    axis.set_ylabel("Iteration seconds · logarithmic scale", fontsize=11)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", which="major", color="#dce3eb", lw=0.7)
    axis.set_facecolor("#f5f7fb")


def _footer(figure, report):
    figure.text(
        0.08,
        0.13,
        f"Mean measured iteration: {report['step_mean_seconds']:.2f} s. Orange points include the checkpoint save and that update's computation.",
        fontsize=10,
    )
    figure.text(
        0.08,
        0.08,
        "Timings cover updates 52–2,000. Process time also includes startup, untimed initial updates and final cleanup.",
        fontsize=10,
    )
    figure.text(
        0.08,
        0.035,
        "Observed full-run conditions include warm caches and early visualization traffic on shared storage. Separate repetitions measure scaling.",
        fontsize=9,
        color="#526274",
    )


def _main():
    root = Path(__file__).resolve().parent
    evidence, report, steps, seconds = _load(root)
    assert evidence["scheduler"]["state"] == "COMPLETED"
    figure, axis = plt.subplots(figsize=(14, 7.5), dpi=150)
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(left=0.08, right=0.97, top=0.65, bottom=0.24)
    _header(figure, report)
    _timings(axis, steps, seconds)
    _footer(figure, report)
    output = root / "full-training.png"
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
