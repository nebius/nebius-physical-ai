"""Render checkpoint-linked simulator contact sheets from recorded MP4 bytes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _frame(path, index, shape):
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select=eq(n\\,{index})",
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    data = subprocess.check_output(command)
    width, height = shape
    if len(data) != width * height * 3:
        raise ValueError("decoded frame does not match recorded dimensions")
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def _title(figure, sample, campaign):
    outcome = "Successful" if sample["success"] else "Unsuccessful"
    figure.text(
        0.125,
        0.94,
        f"{outcome} WAM rollout · checkpoint {campaign['checkpoint_step']}",
        fontsize=20,
        weight="bold",
        color="#122538",
    )
    figure.text(
        0.125,
        0.9,
        f"LIBERO task {sample['task_id']}: {sample['task_description']}",
        fontsize=10.5,
    )
    figure.text(
        0.125,
        0.865,
        f"{campaign['training_gpus']} B200 training GPUs → "
        "1 B200 policy server → native MuJoCo simulation",
        fontsize=10,
        color="#087f75",
    )


def _plot(root, sample, campaign):
    video = root / sample["video"]
    if hashlib.sha256(video.read_bytes()).hexdigest() != sample["video_sha256"]:
        raise ValueError("video bytes differ from the evidence manifest")
    figure, axes = plt.subplots(2, 3, figsize=(12, 8.8), dpi=150)
    figure.patch.set_facecolor("#f5f7fb")
    figure.subplots_adjust(top=0.81, bottom=0.1, wspace=0.04, hspace=0.25)
    for axis, index in zip(axes.flat, sample["contact_frame_indices"]):
        axis.imshow(_frame(video, index, sample["dimensions"]))
        axis.set_axis_off()
        axis.set_title(f"Simulation: {index / sample['fps']:g} seconds", fontsize=10)
    _title(figure, sample, campaign)
    outcome = "success" if sample["success"] else "unsuccessful"
    color = "#087f75" if sample["success"] else "#92400e"
    figure.text(
        0.125,
        0.045,
        f"Native task result: {outcome} after {sample['steps']} simulator steps. "
        "One illustrative trial; not a success-rate estimate.",
        fontsize=9.5,
        color=color,
    )
    target = root / sample["poster"]
    figure.savefig(target, facecolor=figure.get_facecolor())
    plt.close(figure)
    return {
        "file": target.name,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def _main(args):
    manifest = json.loads((args.root / "evidence.json").read_text())
    outputs = [_plot(args.root, sample, manifest) for sample in manifest["samples"]]
    receipt = {"matplotlib": matplotlib.__version__, "outputs": outputs}
    (args.root / "render-manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    _main(parser.parse_args())
