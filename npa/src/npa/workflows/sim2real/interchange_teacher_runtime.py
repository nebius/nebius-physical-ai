"""Collect successful geometry-aware teacher episodes for any sealed arm profile."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from npa.workflows.lerobot_transfer_data import write_json
from npa.workflows.sim2real.interchange_contract import (
    Transform,
    policy_profile_from_dict,
    rigid_object_profile_from_dict,
    robot_sensor_profile_from_dict,
    surface_placement_task_from_dict,
)
from npa.workflows.sim2real.interchange_isaac_runtime import (
    _load,
    _local_object_asset,
    _run_teacher,
    _sha256,
)


SCHEMA = "npa.sim2real.policy_interchange.teacher_collection.v1"
SPLITS = ("train", "validation", "gold")


def _scenario_offset(split: str, episode: int) -> tuple[float, float]:
    """Return a deterministic split-disjoint object-layout offset in metres."""

    if split not in SPLITS or episode < 0:
        raise ValueError(
            "teacher scenario requires a known split and nonnegative index"
        )
    digest = hashlib.sha256(f"{split}:{episode}".encode()).digest()
    unit_y = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    x_ranges = {
        "train": (-0.025, -0.005),
        "validation": (0.0, 0.010),
        "gold": (0.015, 0.025),
    }
    low, high = x_ranges[split]
    unit_x = int.from_bytes(digest[8:16], "big") / float(2**64 - 1)
    return low + (high - low) * unit_x, -0.025 + 0.05 * unit_y


def _scenario_prop(prop: Any, split: str, episode: int) -> tuple[Any, dict[str, Any]]:
    offset_x, offset_y = _scenario_offset(split, episode)
    translation = prop.world_from_root.translation_m
    world_from_root = Transform(
        (
            translation[0] + offset_x,
            translation[1] + offset_y,
            translation[2],
        ),
        prop.world_from_root.quaternion_xyzw,
    )
    scenario = {
        "split": split,
        "episode_index": episode,
        "object_translation_offset_m": [offset_x, offset_y, 0.0],
        "world_from_root": world_from_root.to_dict(),
    }
    scenario["identity"] = hashlib.sha256(
        json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return replace(prop, world_from_root=world_from_root), scenario


def _asset_binding(
    preparation: dict[str, Any], manifest: dict[str, Any], path: Path
) -> None:
    asset = preparation.get("robot_asset") or {}
    contract = asset.get("robot_spec") if isinstance(asset, dict) else None
    if not isinstance(contract, dict):
        raise RuntimeError("preparation lacks normalized RobotSpec bytes")
    identity = str((preparation.get("robot") or {}).get("identity") or "")
    source_tree = str((preparation.get("robot") or {}).get("bundle_sha256") or "")
    if (
        manifest.get("embodiment_digest") != identity
        or manifest.get("source_tree_sha256") != source_tree
        or manifest.get("usd_sha256") != _sha256(path)
    ):
        raise RuntimeError("fetched robot asset manifest does not bind this profile")


def _episode_artifacts(directory: Path) -> dict[str, str]:
    names = (
        "obs_workspace.npy",
        "obs_wrist.npy",
        "state.npy",
        "actions.npy",
        "timestamps.npy",
        "metadata.json",
        "native_teacher.json",
    )
    return {name: _sha256(directory / name) for name in names}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Collect the configured split counts and reject any unsuccessful teacher."""

    import gymnasium as gym

    preparation = _load(args.preparation_path)
    manifest = _load(args.robot_asset_manifest_path)
    _asset_binding(preparation, manifest, args.robot_usd_path)
    asset = preparation["robot_asset"]
    robot = robot_sensor_profile_from_dict(preparation.get("robot") or {})
    prop = rigid_object_profile_from_dict(preparation.get("object") or {})
    task = surface_placement_task_from_dict(preparation.get("task") or {})
    policy = policy_profile_from_dict(preparation.get("policy") or {})
    object_usd_path = _local_object_asset(args.input_path, prop)
    args.output_path.mkdir(parents=True, exist_ok=False)
    counts = {
        "train": args.train_episodes,
        "validation": args.validation_episodes,
        "gold": args.gold_episodes,
    }
    episodes: list[dict[str, Any]] = []
    for split in SPLITS:
        for index in range(counts[split]):
            scenario_prop, scenario = _scenario_prop(prop, split, index)
            episode = args.output_path / "splits" / split / f"episode_{index:04d}"
            episode_args = argparse.Namespace(**vars(args))
            episode_args.seed = args.seed + SPLITS.index(split) * 1_000_000 + index
            episode_args.output_path = episode
            teacher = _run_teacher(
                gym=gym,
                args=episode_args,
                robot=robot,
                robot_spec=asset["robot_spec"],
                prop=scenario_prop,
                task=task,
                object_usd_path=object_usd_path,
                control_hz=policy.action.control_hz,
                episode_directory=episode,
            )
            episodes.append(
                {
                    "split": split,
                    "episode_index": index,
                    "seed": episode_args.seed,
                    "scenario": scenario,
                    "frames": len(teacher["controls"]),
                    "ordered_event_steps": teacher["ordered_event_steps"],
                    "task_success": teacher["task_success"],
                    "artifacts": _episode_artifacts(episode),
                }
            )
    report = {
        "schema": SCHEMA,
        "status": "passed",
        "robot_identity": robot.identity,
        "robot_source_tree_sha256": robot.bundle_sha256,
        "robot_usd_sha256": manifest["usd_sha256"],
        "object_identity": prop.identity,
        "object_source_sha256": prop.source_sha256,
        "task_identity": task.identity,
        "policy_backend": policy.backend,
        "counts": counts,
        "episodes": episodes,
        "all_teacher_episodes_successful": all(row["task_success"] for row in episodes),
        "train_only_normalization_required": True,
        "privileged_object_state_policy_input": False,
    }
    write_json(args.output_path / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--preparation-path", type=Path, required=True)
    parser.add_argument("--robot-usd-path", type=Path, required=True)
    parser.add_argument("--robot-asset-manifest-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, required=True)
    parser.add_argument("--validation-episodes", type=int, required=True)
    parser.add_argument("--gold-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.train_episodes, args.validation_episodes, args.gold_episodes) < 1:
        raise ValueError("every teacher split requires at least one episode")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    os.environ["OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"] = "1"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    try:
        started = time.monotonic()
        result = run(args)
        result["duration_seconds"] = time.monotonic() - started
        write_json(args.output_path / "report.json", result)
    finally:
        app.close()
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
