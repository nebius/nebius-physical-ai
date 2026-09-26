"""Plot the measured CUDA training journal from the sanitized numeric export."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _header(fig):
    """Label the measured device and frozen recipe."""
    fig.suptitle(
        "Real B200 training: 32 CUDA optimizer updates",
        x=0.08,
        ha="left",
        y=0.96,
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.87,
        "1,024 generated examples | 8 features | SGD momentum 0.8 | float64",
        fontsize=11,
        color="#334155",
    )


def _loss_panel(axis, metrics, journal):
    """Plot pre-update measurements and the final train/held-out losses."""
    losses = [row["loss"] for row in journal] + [metrics["final_train_loss"]]
    axis.semilogy(
        range(33),
        losses,
        "o-",
        color="#1565c0",
        markersize=3,
        label="Measured training loss",
    )
    axis.scatter(
        [32],
        [metrics["heldout_loss"]],
        marker="D",
        color="#d97706",
        zorder=4,
        label="Final held-out loss",
    )
    axis.set(
        title="Loss after each completed update",
        xlabel="Completed optimizer updates",
        ylabel="Mean squared error (log scale)",
    )


def _motion_panel(axis, journal):
    """Plot measured gradient norms and parameter changes."""
    steps = [row["optimizer_step"] for row in journal]
    axis.semilogy(
        steps,
        [row["gradient_norm"] for row in journal],
        color="#1565c0",
        label="Gradient L2 norm",
    )
    axis.semilogy(
        steps,
        [row["parameter_delta"] for row in journal],
        color="#16826b",
        label="Parameter change L2 norm",
    )
    axis.set(
        title="Measured gradients and parameter changes",
        xlabel="Optimizer update",
        ylabel="L2 norm (log scale)",
    )


def _footer(fig):
    """State the verified numerical scope and its limitation."""
    fig.text(
        0.08,
        0.09,
        "CPU verification: 32 gradient norms and parameter changes, plus checkpoint weights and momentum.",
        fontsize=10,
    )
    fig.text(
        0.08,
        0.045,
        "Final train loss: 0.00165440 | Held-out: 0.00155750 | Synthetic workflow proof; not robot-policy quality.",
        fontsize=10,
        color="#334155",
    )


def main():
    """Create the two-panel chart from actual exported measurements."""
    root = Path(__file__).parent
    data = json.loads((root / "baseline-numeric.json").read_text())
    metrics = data["metrics"]
    journal = metrics["journal"]
    assert [row["optimizer_step"] for row in journal] == list(range(1, 33))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.20, top=0.79, wspace=0.29)
    _header(fig)
    _loss_panel(axes[0], metrics, journal)
    _motion_panel(axes[1], journal)
    for axis in axes:
        axis.grid(True, which="major", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=9)
        axis.set_xlim(-0.5, 33)
    _footer(fig)
    fig.savefig(
        root / "loss-and-updates.png",
        dpi=160,
        facecolor="white",
        metadata={
            "Software": "Matplotlib",
            "Description": "Measured CUDA journal; numeric source in baseline-numeric.json",
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
