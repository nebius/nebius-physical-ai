"""Render a recorded, process-attributed sixteen-B200 training snapshot."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _panel(axis, values, total, color, title, unit):
    indices = list(range(len(values)))
    axis.barh(indices, total, color="#e3e9ef", height=0.58)
    axis.barh(indices, values, color=color, height=0.58)
    for index, value in enumerate(values):
        label = f"{value:.1f} {unit}" if unit == "GiB" else f"{value:g}%"
        axis.text(value + total * 0.025, index, label, va="center", fontsize=11)
    axis.set_yticks(indices, [f"GPU {index}" for index in indices])
    axis.invert_yaxis()
    axis.set_xlim(0, total * 1.2)
    if unit == "%":
        axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_title(title, loc="left", fontsize=13, pad=16, weight="bold")
    axis.set_xlabel(unit if unit == "GiB" else "Percent")
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.set_facecolor("#f5f7fb")


def _load(root):
    evidence = json.loads((root / "evidence.json").read_text())
    nodes = []
    for record in evidence["snapshots"]:
        raw = (root / record["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        nodes.append(json.loads(raw))
    assert {node["node_rank"] for node in nodes} == {0, 1}
    assert len({node["run_settings_sha256"] for node in nodes}) == 1
    devices = [gpu for node in nodes for gpu in node["gpus"]]
    assert len(devices) == 16 and {gpu["rank"] for gpu in devices} == set(range(16))
    assert all(
        gpu["process_matches_training_command_and_slurm_cgroup"] for gpu in devices
    )
    for node in nodes:
        assert node["training_gpus"] == 16
        assert node["distributed_preflight"]["status"] == "passed"
        assert node["distributed_preflight"]["world_size"] == 16
        assert node["distributed_preflight"]["hosts"] == 2
        assert node["training_processes_matched_on_this_host"] == 8
    return sorted(nodes, key=lambda node: node["node_rank"])


def _node(axes, node):
    devices = sorted(node["gpus"], key=lambda item: item["gpu_index"])
    assert [item["gpu_index"] for item in devices] == list(range(8))
    label = f"Node rank {node['node_rank']}"
    _panel(
        axes[0],
        [item["utilization_percent"] for item in devices],
        100,
        "#087f75",
        f"{label} · GPU utilization",
        "%",
    )
    capacity = devices[0]["memory_total_mib"] / 1024
    assert all(item["memory_total_mib"] / 1024 == capacity for item in devices)
    _panel(
        axes[1],
        [item["memory_used_mib"] / 1024 for item in devices],
        capacity,
        "#345eb7",
        f"{label} · device memory ({capacity:.1f} GiB total)",
        "GiB",
    )


def _captions(figure, nodes):
    figure.text(
        0.08,
        0.95,
        "Cosmos3-Nano WAM · sixteen B200s training",
        fontsize=22,
        weight="bold",
        color="#122538",
    )
    stamps = [node["captured_utc"][:19].replace("T", " ") for node in nodes]
    figure.text(
        0.08,
        0.907,
        f"Actual nvidia-smi readings · {min(stamps)} to {max(stamps)} UTC",
        fontsize=11,
    )
    steps = [node["last_monitored_optimizer_update"] for node in nodes]
    figure.text(
        0.08,
        0.87,
        f"Latest timed updates at capture: {min(steps)}–{max(steps)} / 2,000 · sixteen-rank NCCL preflight passed",
        fontsize=11,
    )
    figure.text(
        0.08,
        0.085,
        "All sixteen GPU processes matched the native training command, configuration, Slurm cgroup and rank assignment.",
        fontsize=10,
    )
    figure.text(
        0.08,
        0.047,
        "One live sample per device. Utilization is not throughput; this snapshot does not establish policy quality.",
        fontsize=10,
        color="#526274",
    )


def _main():
    root = Path(__file__).resolve().parent
    nodes = _load(root)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(
        top=0.79, bottom=0.16, left=0.08, right=0.95, wspace=0.27, hspace=0.57
    )
    for pair, node in zip(axes, nodes, strict=True):
        _node(pair, node)
    _captions(figure, nodes)
    output = root / "training-snapshot.png"
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
