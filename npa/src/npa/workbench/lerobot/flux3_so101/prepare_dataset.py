"""Download the pinned Pick Orange demonstrations and convert them locally to LeRobot v3."""

import json
import math
import shutil
from pathlib import Path

from .calibration import ID as CALIBRATION_ID, to_policy, transform_stats

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def check_info(info: dict, spec: dict, version: str) -> None:
    if info["codebase_version"] != version:
        raise ValueError(
            f"dataset format: expected {version}, got {info['codebase_version']}"
        )
    for field, expected in (
        ("fps", spec["fps"]),
        ("total_episodes", spec["episodes"]),
        ("total_frames", spec["frames"]),
    ):
        if info[field] != expected:
            raise ValueError(f"{field}: expected {expected}, got {info[field]}")
    features = info["features"]
    for key in ("action", "observation.state"):
        feature = features[key]
        if feature["shape"] != [6] or feature["names"] != JOINTS:
            raise ValueError(f"{key}: SO-101 joint order or shape mismatch")
    cameras = {
        name for name, feature in features.items() if feature.get("dtype") == "video"
    }
    if cameras != set(spec["cameras"]):
        raise ValueError(f"camera keys mismatch: {cameras}")


def check_episode_stats(path: Path, episodes: int) -> None:
    seen = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        index = row["episode_index"]
        if index in seen or not 0 <= index < episodes:
            raise ValueError(f"invalid episode statistics index: {index}")
        seen.add(index)
        for key in ("action", "observation.state"):
            for label in ("min", "max", "mean", "std"):
                values = row["stats"][key][label]
                if len(values) != 6 or any(
                    not math.isfinite(float(value)) for value in values
                ):
                    raise ValueError(f"invalid {key} {label} statistics")
                if label in {"min", "max"} and (
                    any(abs(value) > 180.001 for value in values[:5])
                    or not -0.001 <= values[5] <= 100.001
                ):
                    raise ValueError(f"{key} {label} outside SO-101 degrees/percent")
    if len(seen) != episodes:
        raise ValueError(f"expected {episodes} episode statistics, got {len(seen)}")


def calibrate_source(root: Path, episodes: int) -> None:
    """Convert source motor values and their metadata before v2.1 -> v3."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if len(paths) != episodes:
        raise ValueError(f"expected {episodes} source parquet files, got {len(paths)}")
    for path in paths:
        table = pq.read_table(path)
        for key in ("observation.state", "action"):
            index = table.schema.get_field_index(key)
            if index < 0:
                raise ValueError(f"missing {key} in {path}")
            values = [to_policy(value) for value in table.column(index).to_pylist()]
            table = table.set_column(
                index, key, pa.array(values, type=table.schema.field(index).type)
            )
        temporary = path.with_suffix(".calibrated.parquet")
        pq.write_table(table, temporary)
        temporary.replace(path)

    stats_path = root / "meta/episodes_stats.jsonl"
    rows = [json.loads(line) for line in stats_path.read_text().splitlines()]
    for row in rows:
        transform_stats(row["stats"])
    stats_path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def prepare(root: Path, spec: dict, report_dir: Path, instruction: str) -> dict:
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset
    from PIL import Image
    import torch

    if root.exists():
        raise FileExistsError(f"local dataset path already exists: {root}")
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="dataset",
        revision=spec["revision"],
        local_dir=root,
        allow_patterns=["data/**", "videos/**", "meta/**"],
    )
    source_info = json.loads((root / "meta/info.json").read_text())
    check_info(source_info, spec, "v2.1")
    check_episode_stats(root / "meta/episodes_stats.jsonl", spec["episodes"])
    calibrate_source(root, spec["episodes"])
    convert_dataset(spec["repo_id"], root=root, push_to_hub=False)
    converted_info = json.loads((root / "meta/info.json").read_text())
    check_info(converted_info, spec, "v3.0")

    dataset = LeRobotDataset(spec["repo_id"], root=root, video_backend="pyav")
    if dataset.num_episodes != spec["episodes"] or len(dataset) != spec["frames"]:
        raise ValueError("converted dataset episode/frame count mismatch")
    report_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        frame = dataset[index]
        for key in ("action", "observation.state"):
            values = frame[key]
            if tuple(values.shape) != (6,) or not torch.isfinite(values).all():
                raise ValueError(f"{key} invalid at row {index}")
            if (values[:5].abs() > 200).any() or (values[5] < 0) or (values[5] > 100):
                raise ValueError(
                    f"{key} outside calibrated SO-101 frame at row {index}"
                )
        if frame["task"] != instruction:
            raise ValueError(
                f"task instruction mismatch at row {index}: {frame['task']!r}"
            )
        for camera in spec["cameras"]:
            image = frame[camera]
            if (
                image.ndim != 3
                or image.shape[0] != 3
                or not torch.isfinite(image).all()
            ):
                raise ValueError(f"{camera} failed to decode at row {index}")
        examples.append({"index": index, "task": frame["task"]})
        if index == 0:
            images = []
            for camera in spec["cameras"]:
                pixels = (frame[camera].permute(1, 2, 0).numpy() * 255).clip(0, 255)
                images.append(Image.fromarray(pixels.astype("uint8")))
            preview = Image.new(
                "RGB",
                (
                    sum(image.width for image in images),
                    max(image.height for image in images),
                ),
            )
            x = 0
            for image in images:
                preview.paste(image, (x, 0))
                x += image.width
            preview.save(report_dir / "camera-preview.jpg")

    report = {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "source_format": "v2.1",
        "converted_format": "v3.0",
        "episodes": dataset.num_episodes,
        "frames": len(dataset),
        "fps": converted_info["fps"],
        "camera_map": {
            "observation.images.front": "observation.images.scene",
            "observation.images.wrist": "observation.images.wrist",
        },
        "calibration_id": CALIBRATION_ID,
        "sampled_rows": examples,
    }
    (report_dir / "dataset.json").write_text(json.dumps(report, indent=2) + "\n")
    # The converter keeps its input as a sibling. Training needs only the verified v3 copy.
    shutil.rmtree(root.parent / f"{root.name}_old")
    return report
