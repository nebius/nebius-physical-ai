"""Render three measured eight-B200 timing repetitions from verified reports."""

import hashlib
import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(root):
    evidence = json.loads((root / "evidence.json").read_text())
    for name, digest in evidence["files_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    recipe = root.parents[3] / "npa/workflows/workbench/cosmos3-wam-slurm"
    analyzer = runpy.run_path(str(recipe / "scaling_report.py"))
    directories = [root / f"repeat-{index}" for index in range(1, 4)]
    calculated = analyzer["_summarize"](directories)
    saved = json.loads((root / "repeated-scaling.json").read_text())
    assert saved == calculated
    assert saved["status"] == "baseline_measured"
    records = [analyzer["_read"](directory) for directory in directories]
    return evidence, saved["groups"]["8"], records


def _header(figure, group):
    spread = group["replicate_step_means_seconds"]
    figure.text(
        0.08,
        0.945,
        "How repeatable is eight-B200 WAM training?",
        fontsize=22,
        weight="bold",
        color="#122538",
    )
    figure.text(
        0.08,
        0.895,
        "Three separate 200-update runs · same base weights and seed · nominal global batch 2,048",
        fontsize=11,
    )
    figure.text(
        0.08,
        0.82,
        f"Mean step: {spread['mean']:.4f} s    |    Across-run SD: {spread['sample_standard_deviation']:.4f} s",
        fontsize=16,
        weight="bold",
        color="#087f75",
    )
    figure.text(
        0.08,
        0.775,
        f"{group['measured_steps']} recorded iterations · pooled throughput {group['pooled_tokens_per_second']:,.1f} tokens/s",
        fontsize=11,
        color="#526274",
    )


def _traces(axes, records):
    values = [value for record in records for value in record["seconds"]]
    margin = max((max(values) - min(values)) * 0.12, 0.02)
    for index, (axis, record) in enumerate(zip(axes, records, strict=True), 1):
        report = record["measurement"]
        axis.plot(range(52, 201), record["seconds"], lw=1, color="#087f75")
        axis.axhline(report["step_mean_seconds"], color="#b96617", ls="--", lw=1)
        axis.set_ylim(min(values) - margin, max(values) + margin)
        axis.set_xlim(50, 202)
        axis.set_ylabel(f"Run {index}\nseconds", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dce3eb", lw=0.6)
        axis.set_facecolor("#f5f7fb")
        axis.text(
            1.04,
            0.83,
            f"Mean  {report['step_mean_seconds']:.4f} s\n"
            f"p95     {report['step_p95_seconds']:.2f} s\n"
            f"{report['work']['tokens_per_second']:,.1f} tokens/s",
            transform=axis.transAxes,
            va="top",
            fontsize=10,
            linespacing=1.65,
        )
    axes[-1].set_xlabel("Optimizer update · 52–200 in each run", fontsize=11)


def _footer(figure):
    figure.text(
        0.08,
        0.095,
        "Shared vertical scale is zoomed. Dashed lines show each run's mean. Final checkpoint saves are outside these timed iterations.",
        fontsize=9,
    )
    figure.text(
        0.08,
        0.048,
        "Across-run SD describes three run means, not 447 independent trials. Same-seed repetitions do not measure policy-quality variability.",
        fontsize=9,
        color="#526274",
    )


def _save(root, figure):
    output = root / "repeated-timings.png"
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


def _main():
    root = Path(__file__).resolve().parent
    _, group, records = _load(root)
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 8), dpi=150)
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(left=0.08, right=0.77, top=0.71, bottom=0.20, hspace=0.24)
    _header(figure, group)
    _traces(axes, records)
    _footer(figure)
    _save(root, figure)


if __name__ == "__main__":
    _main()
