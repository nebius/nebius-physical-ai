"""Run RoboTwin's pinned collector and validate its native artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit


TASK = "beat_block_hammer"
CONFIG = "demo_clean"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare_packages() -> None:
    """Apply only the two compatibility edits in pinned scripts/_install.sh."""
    sapien = importlib.metadata.distribution("sapien")
    loader = Path(sapien.locate_file("sapien/wrapper/urdf_loader.py"))
    text = loader.read_text()
    old = 'with open(urdf_file, "r") as f:'
    if text.count(old) != 1:
        raise ValueError("SAPIEN pinned loader contract changed")
    loader.write_text(
        text.replace(old, 'with open(urdf_file, "r", encoding="utf-8") as f:')
    )
    mplib = importlib.metadata.distribution("mplib")
    planner = Path(mplib.locate_file("mplib/planner.py"))
    text = planner.read_text()
    old = "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:"
    if text.count(old) != 1:
        raise ValueError("MPLib pinned planner contract changed")
    planner.write_text(
        text.replace(
            old, "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:"
        )
    )


def validate_native(output: Path) -> dict:
    import cv2
    import h5py
    import numpy as np

    episodes = list(
        output.glob("native/demo_clean/beat_block_hammer/aloha_agilex/data/*.hdf5")
    )
    videos = list(
        output.glob("native/demo_clean/beat_block_hammer/aloha_agilex/video/*.mp4")
    )
    if len(episodes) != 1 or len(videos) != 1:
        raise ValueError("Expected exactly one native HDF5 episode and MP4")
    frames_dir = output / "frames"
    frames_dir.mkdir()
    with h5py.File(episodes[0], "r") as episode:
        if episode.attrs.get("source_format") != "RoboTwin":
            raise ValueError("Native RoboTwin provenance missing")
        counts = set()
        motion = False
        for name in (
            "left_arm_joint_states",
            "right_arm_joint_states",
            "left_ee_joint_states",
            "right_ee_joint_states",
        ):
            state = np.asarray(episode[f"state/{name}"])
            action = np.asarray(episode[f"action/{name}"])
            if state.shape != action.shape or state.ndim != 2 or len(state) < 2:
                raise ValueError("Malformed native state/action arrays")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError("Nonfinite native state/action arrays")
            if not np.allclose(action[:-1], state[1:]):
                raise ValueError("Native action/state temporal pairing is invalid")
            motion |= bool(np.any(np.abs(action - state) > 1e-6))
            counts.add(len(state))
        if len(counts) != 1 or not motion:
            raise ValueError("Episode has no aligned articulated motion")
        count = counts.pop()
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            encoded = episode[f"vision/{camera}/colors"]
            if len(encoded) != count:
                raise ValueError("Camera/state timeline mismatch")
            for index in {0, count // 2, count - 1}:
                frame = cv2.imdecode(
                    np.frombuffer(encoded[index], dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None or frame.ndim != 3 or float(frame.std()) <= 1:
                    raise ValueError("Native camera frame did not decode")
    video = cv2.VideoCapture(str(videos[0]))
    decoded = 0
    frame_hashes = set()
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or float(frame.std()) <= 1:
                raise ValueError("Video contains an empty frame")
            if decoded in {0, count // 2, count}:
                if not cv2.imwrite(str(frames_dir / f"frame-{decoded:06d}.png"), frame):
                    raise ValueError("Frame export failed")
            frame_hashes.add(hashlib.sha256(frame.tobytes()).hexdigest())
            decoded += 1
    finally:
        video.release()
    if decoded != count + 1 or len(frame_hashes) < 2:
        raise ValueError("Video/native timeline mismatch or static video")
    return {
        "state_action_pairs": count,
        "decoded_video_frames": decoded,
        "distinct_video_frames": len(frame_hashes),
        "hdf5": str(episodes[0].relative_to(output)),
        "video": str(videos[0].relative_to(output)),
    }


def collect(source: Path, output: Path) -> int:
    import torch
    import yaml

    if torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible GPU is required")
    gpu = torch.cuda.get_device_name(0)
    if (
        "RTX PRO 6000" not in gpu
        or "Blackwell" not in gpu
        or torch.cuda.get_device_capability(0) != (12, 0)
    ):
        raise ValueError("RoboTwin requires RTX PRO 6000 Blackwell")
    torch.multiprocessing.set_start_method("spawn", force=True)
    prepare_packages()
    sys.path.insert(0, str(source))
    # This is the pinned upstream asset-path renderer, without its interactive
    # fallback: the locked extraction must have supplied the expected directory.
    if not (source / "assets/embodiments/aloha-agilex").is_dir():
        raise ValueError("Locked embodiment is missing")
    importlib.import_module("scripts.update_embodiment_config_path").main()
    config_path = source / "env_cfg/task_config/demo_clean.yml"
    config = yaml.safe_load(config_path.read_text())
    config.update(episode_num=1, save_path=str(output / "native"))
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    collector = importlib.import_module("scripts.collect_data")
    original_factory = collector.class_decorator
    evidence = {"planning_successes": 0, "replay_successes": 0, "seeds_attempted": []}

    def factory(task_name):
        task = original_factory(task_name)
        original_setup = task.setup_demo
        original_check = task.check_success
        phase = {"planning": True}

        def setup(**kwargs):
            phase["planning"] = kwargs["need_plan"]
            if phase["planning"]:
                evidence["seeds_attempted"].append(kwargs["seed"])
            return original_setup(**kwargs)

        def check():
            success = original_check()
            if success:
                key = "planning_successes" if phase["planning"] else "replay_successes"
                evidence[key] += 1
            return success

        task.setup_demo = setup
        task.check_success = check
        return task

    collector.class_decorator = factory
    # Real upstream CuRobo planning, SAPIEN physics and Vulkan rendering; no
    # replacement motion plan or simulated success condition is introduced.
    collector.main(task_name=TASK, task_config=CONFIG)
    if not evidence["planning_successes"] or not evidence["replay_successes"]:
        raise ValueError("Upstream successful seed was not replayed successfully")
    native = validate_native(output)
    seed_path = output / "native/demo_clean/beat_block_hammer/aloha_agilex/seed.txt"
    seeds = [int(value) for value in seed_path.read_text().split()]
    if len(seeds) != 1 or seeds[0] not in evidence["seeds_attempted"]:
        raise ValueError("Successful seed evidence is inconsistent")
    # Pickled motion/cache files are native intermediate data, not deliverables.
    for path in (output / "native").rglob("*.pkl"):
        path.unlink()
    artifacts = [
        {
            "path": str(path.relative_to(output)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    summary = {
        "schema_version": "npa.robotwin.successful-seed-replay.v1",
        "solution": "robotwin",
        "status": "passed",
        "task": TASK,
        "task_config": CONFIG,
        "capability": "beat_block_hammer_successful_seed_replay_collection",
        "source_revision": "96c1feab536306b50c26af200044fcdf126e8904",
        "curobo_revision": "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
        "asset_revision": "785feb15aa4a4f532395ad2b1d2be5f28cb561ad",
        "bootstrap_image_digest": os.environ["NPA_ROBOTWIN_BOOTSTRAP_DIGEST"],
        "gpu": {"name": gpu, "count": 1, "compute_capability": "12.0"},
        "successful_seed": seeds[0],
        "upstream": evidence,
        "native": native,
        "artifacts": artifacts,
    }
    (output / "robotwin-smoke.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "solution": "robotwin",
                "status": "passed",
                "smoke_artifact": "robotwin-smoke.json",
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def upload(output: Path) -> int:
    import boto3

    prefix = urlsplit(os.environ["S3_OUTPUT_PREFIX"])
    if (
        prefix.scheme != "s3"
        or prefix.netloc != os.environ["NPA_S3_BUCKET"]
        or not prefix.path.endswith("/")
        or prefix.query
        or prefix.fragment
    ):
        raise ValueError("Authorized output destination is malformed")
    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("NEBIUS_S3_ENDPOINT"),
    )
    paths = [path for path in sorted(output.rglob("*")) if path.is_file()]
    # Summary is the last object: consumers never mistake a partial upload for
    # completed capability evidence.
    paths.sort(key=lambda path: path.name == "summary.json")
    for path in paths:
        if path.is_symlink():
            raise ValueError("Output symlinks are not permitted")
        key = prefix.path.lstrip("/") + path.relative_to(output).as_posix()
        client.upload_file(str(path), prefix.netloc, key)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "upload"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return (
        collect(args.source, args.output)
        if args.command == "collect"
        else upload(args.output)
    )


if __name__ == "__main__":
    raise SystemExit(main())
