"""Export a real native LIBERO training window for the reproducible data figure."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def _load(root):
    from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import (
        LIBEROLeRobotDataset,
    )

    sources = json.loads(Path(__file__).with_name("sources.json").read_text())
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root / "framework", text=True
    ).strip()
    if revision != sources["framework"]:
        raise ValueError("framework revision differs from the recipe")
    dataset = LIBEROLeRobotDataset(
        root=str(root / "data/libero_10"),
        fps=20,
        chunk_length=16,
        image_size=256,
        mode="wam",
        camera_mode="concat_view",
        action_space="frame_wise_relative",
        rotation_space="6d",
        pose_coordinate_frame="native",
        action_normalization="quantile_rot",
        val_ratio=0.01,
        seed=0,
    )
    return sources, dataset


def _sample_record(sources, dataset, sample):
    start = int(dataset._ep_starts[0])
    raw = dataset._row_action[start : start + 16]
    converted = dataset._build_frame_wise_action(raw)
    return {
        "dataset_revision": sources["dataset"],
        "framework_revision": sources["framework"],
        "episode": int(dataset._ep_vals[0]),
        "start_frame": 0,
        "task_index": int(dataset._row_task[start]),
        "task": sample["ai_caption"],
        "fps": int(sample["conditioning_fps"]),
        "video_timestamps": dataset._row_timestamp[start : start + 17].tolist(),
        "raw_actions": raw.tolist(),
        "native_rot6d_actions": converted.tolist(),
        "normalized_actions": sample["action"].tolist(),
        "normalization": "quantile_rot using global_raw statistics; no forward clamp",
        "stats_sha256": hashlib.sha256(dataset._stats_file.read_bytes()).hexdigest(),
        "train_episodes": len(dataset._ep_vals),
        "total_episodes": dataset._info["total_episodes"],
        "total_frames": dataset._info["total_frames"],
        "training_windows": len(dataset),
        "episode_split_seed": 0,
        "scope": "Native decoded dataset window before the SFT tokenizer/resolution transform; not model predictions",
    }


def _main(args):
    from PIL import Image

    sources, dataset = _load(args.shared_root)
    sample = dataset[0]
    if list(sample["video"].shape) != [3, 17, 256, 512]:
        raise ValueError("unexpected native video window shape")
    if list(sample["action"].shape) != [16, 10]:
        raise ValueError("unexpected native action window shape")
    args.output_path.mkdir(parents=True, exist_ok=False)
    frame = sample["video"][:, 0].permute(1, 2, 0).numpy()
    for name, pixels in (("front.png", frame[:, :256]), ("wrist.png", frame[:, 256:])):
        Image.fromarray(pixels).save(args.output_path / name)
    record = _sample_record(sources, dataset, sample)
    record["images_sha256"] = {
        name: hashlib.sha256((args.output_path / name).read_bytes()).hexdigest()
        for name in ("front.png", "wrist.png")
    }
    (args.output_path / "sample.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"episode": record["episode"], "training_windows": len(dataset)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    os.umask(0o077)
    _main(parser.parse_args())
