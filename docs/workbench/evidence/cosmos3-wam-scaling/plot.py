"""Render matched timing repetitions and completed full-schedule measurements."""

import hashlib
import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#345eb7", "#087f75"]


def _load(root):
    evidence = json.loads((root / "evidence.json").read_text())
    for name, digest in evidence["files_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    recipe = root.parents[3] / "npa/workflows/workbench/cosmos3-wam-slurm"
    analyzer = runpy.run_path(str(recipe / "scaling_report.py"))
    directories = [
        root.parent / f"cosmos3-wam-timing-{gpus}/repeat-{index}"
        for gpus in (8, 16)
        for index in range(1, 4)
    ]
    scaling = json.loads((root / "repeated-scaling.json").read_text())
    assert scaling == analyzer["_summarize"](directories)
    assert scaling["status"] == "scaling_measured"
    full = [
        json.loads(
            (root.parent / f"cosmos3-wam-full-{gpus}/measurement.json").read_text()
        )
        for gpus in (8, 16)
    ]
    for gpus, record in zip((8, 16), full, strict=True):
        assert record["status"] == "measured" and record["gpus"] == gpus
        assert record["comparison_contract"]["steps"] == 2000
        assert record["comparison_contract"]["profile"] is False
        assert record["hardware"] == scaling["hardware"]
        for key in ("sources", "samples_per_rank", "global_batch", "seed"):
            assert (
                record["comparison_contract"][key]
                == scaling["comparison_contract"][key]
            )
    return scaling, full


def _bars(axis, values, title, unit, labels, errors=None):
    axis.bar(
        [0, 1],
        values,
        color=COLORS,
        width=0.55,
        yerr=errors,
        capsize=6 if errors else 0,
        zorder=3,
    )
    axis.set_xticks([0, 1], ["8 B200s · one node", "16 B200s · two nodes"])
    axis.set_title(title, loc="left", fontsize=12, pad=14)
    axis.set_ylabel(unit)
    axis.set_ylim(0, max(values) * 1.3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dce3eb", lw=0.7, zorder=0)
    axis.set_facecolor("#f5f7fb")
    for index, (value, label) in enumerate(zip(values, labels, strict=True)):
        axis.text(
            index,
            value + max(values) * 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )


def _timing_panels(axes, scaling):
    groups = [scaling["groups"][str(gpus)] for gpus in (8, 16)]
    means = [group["replicate_step_means_seconds"]["mean"] for group in groups]
    spread = [
        group["replicate_step_means_seconds"]["sample_standard_deviation"]
        for group in groups
    ]
    _bars(
        axes[0],
        means,
        "Steady optimizer step · three runs per topology",
        "Seconds (lower is faster)",
        [f"{value:.4f} s" for value in means],
        spread,
    )
    rates = [group["pooled_tokens_per_second"] / 1000 for group in groups]
    _bars(
        axes[1],
        rates,
        "Pooled native token throughput · all timed iterations",
        "Thousands of tokens / second",
        [f"{value:.2f}k" for value in rates],
    )


def _full_panels(axes, full):
    hours = [record["train_process_seconds"] / 3600 for record in full]
    _bars(
        axes[0],
        hours,
        "Complete 2,000-update native training process",
        "Hours (includes startup and saves)",
        [f"{value:.2f} h" for value in hours],
    )
    compute = [record["training_gpu_hours"] for record in full]
    _bars(
        axes[1],
        compute,
        "Compute consumed by the full training processes",
        "GPU-hours",
        [f"{value:.2f}" for value in compute],
    )


def _captions(figure, scaling):
    comparison = scaling["comparison"]
    figure.text(
        0.08,
        0.955,
        "What changes when WAM training uses two B200 nodes?",
        fontsize=22,
        weight="bold",
        color="#122538",
    )
    figure.text(
        0.08,
        0.91,
        "Same model, data, seed and nominal global batch 2,048 · accumulation 4 on eight GPUs, 2 on sixteen",
        fontsize=11,
    )
    figure.text(
        0.08,
        0.86,
        f"Measured step speedup: {comparison['speedup_vs_8_gpus']:.3f}×    |    Scaling efficiency: {comparison['scaling_efficiency'] * 100:.1f}%",
        fontsize=16,
        weight="bold",
        color="#087f75",
    )
    _footer(figure)


def _footer(figure):
    figure.text(
        0.08,
        0.085,
        "Top: three separate 200-update runs per topology, updates 52–200. Error bars are the sample SD of three run means.",
        fontsize=10,
    )
    figure.text(
        0.08,
        0.05,
        "Bottom: actual complete schedules, including checkpoint writes. Preparation, evaluation and queue time are separate.",
        fontsize=10,
    )
    figure.text(
        0.08,
        0.017,
        "Full-run cache and storage conditions are documented in the evidence. Same-seed timing repeats do not measure policy-quality variability.",
        fontsize=9,
        color="#526274",
    )


def _main():
    root = Path(__file__).resolve().parent
    scaling, full = _load(root)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(
        left=0.08, right=0.96, top=0.77, bottom=0.17, hspace=0.6, wspace=0.30
    )
    _timing_panels(axes[0], scaling)
    _full_panels(axes[1], full)
    _captions(figure, scaling)
    output = root / "scaling-comparison.png"
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
