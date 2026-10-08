"""Compare the original and fresh three-repeat native B200 measurements."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "cosmos3-wam-scaling/repeated-scaling.json"
FRESH = ROOT / "repeated-scaling.json"
COLORS = ("#b7c3bf", "#edf55b")


def panel(ax, reports, field, statistic, divisor, title, units):
    for cohort, report in enumerate(reports):
        for index, gpus in enumerate(("8", "16")):
            group = report["groups"][gpus]
            samples = group["repetitions"]
            assert len(samples) == 3
            assert all(item["measured_steps"] == 149 for item in samples)
            summary = group[statistic]
            mean = summary["mean"] / divisor
            deviation = summary["sample_standard_deviation"] / divisor
            x = index + (cohort - 0.5) * 0.36
            ax.bar(x, mean, width=0.32, color=COLORS[cohort], edgecolor="#34423d")
            ax.errorbar(x, mean, yerr=deviation, color="#172023", capsize=4)
            values = [sample[field] / divisor for sample in samples]
            ax.scatter(x + np.linspace(-0.075, 0.075, 3), values, s=15, c="#172023")
            ax.text(x, mean + ax.get_ylim()[1] * 0.035, f"{mean:.3f}", ha="center")
    ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=16)
    ax.set_xticks([0, 1], ["8 B200 GPUs", "16 B200 GPUs"])
    ax.set_ylabel(units)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)


def render():
    reports = [json.loads(path.read_text()) for path in (ORIGINAL, FRESH)]
    assert reports[0]["comparison_contract"] == reports[1]["comparison_contract"]
    assert reports[0]["hardware"] == reports[1]["hardware"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.9), dpi=180)
    axes[0].set_ylim(0, 16)
    axes[1].set_ylim(0, 160)
    panel(
        axes[0],
        reports,
        "step_mean_seconds",
        "replicate_step_means_seconds",
        1,
        "Mean step time",
        "Seconds per optimizer update",
    )
    panel(
        axes[1],
        reports,
        "tokens_per_second",
        "replicate_token_throughputs",
        1000,
        "Processed token throughput",
        "Thousands of tokens per second",
    )
    fig.suptitle(
        "Cosmos 3 Nano WAM: B200 scaling",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=19,
        weight="bold",
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#34423d")
        for color in COLORS
    ]
    fig.legend(
        handles,
        ["Original campaign", "Fresh reproduction"],
        loc="upper left",
        bbox_to_anchor=(0.065, 0.94),
        ncol=2,
        frameon=False,
    )
    comparison = reports[1]["comparison"]
    fig.text(
        0.07,
        0.075,
        f"Fresh step speedup: {comparison['speedup_vs_8_gpus']:.4f}×    "
        f"Scaling efficiency: {comparison['scaling_efficiency']:.2%}",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.07,
        0.028,
        "Three runs per topology; dots are run means, error bars are sample SD. "
        "149 timed updates per run; one training seed.",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.07, right=0.975, bottom=0.22, top=0.79, wspace=0.28)
    fig.savefig(ROOT / "repeated-scaling.png", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    paths = [ORIGINAL, FRESH, Path(__file__), ROOT / "repeated-scaling.png"]
    manifest = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "sha256": {
            str(path.relative_to(ROOT.parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        },
    }
    (ROOT / "scaling-render-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    render()
