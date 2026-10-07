"""Export physics-verified pi0.5 episodes to OpenPI NPZ and LeRobot v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from npa.workflows.byof.openpi_pipeline import (
    _sample_hash,
    deterministic_npz,
)
from npa.workflows.sim2real.pi05_contract import (
    ACTION_HORIZON,
    DATASET_SCHEMA,
    Pi05ContractError,
    validate_dense_episode,
)


OPENPI_SCHEMA = "npa.workbench.openpi.pi05-surface-pick-place-dataset.v1"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_successes(root: Path, split: str) -> list[tuple[Path, dict[str, Any]]]:
    episodes: list[tuple[Path, dict[str, Any]]] = []
    for metadata_path in sorted(root.glob("episode-*/episode.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        try:
            validate_dense_episode(metadata)
        except Pi05ContractError:
            # Failed attempts are intentionally retained by collection.json but
            # can never become supervised targets.
            continue
        episodes.append((metadata_path.parent, metadata))
    if not episodes:
        raise Pi05ContractError(f"{split} has no physics-verified successful episode")
    return episodes


def _verify_array_metadata(arrays: dict[str, Any], metadata: dict[str, Any]) -> None:
    import numpy as np

    rows = metadata["steps"]
    expected_state = np.asarray(
        [row["joint_position"] + row["gripper_position"] for row in rows],
        dtype=np.float32,
    )
    expected_actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
    expected_times = np.asarray([row["timestamp_s"] for row in rows], dtype=np.float64)
    if not np.array_equal(arrays["state"], expected_state):
        raise Pi05ContractError("state array does not match causal episode rows")
    if not np.array_equal(arrays["actions"], expected_actions):
        raise Pi05ContractError("action array does not match causal episode rows")
    if not np.array_equal(arrays["timestamps"], expected_times):
        raise Pi05ContractError("timestamp array does not match causal episode rows")


def _first_true(mask: Any, start: int, label: str) -> int:
    import numpy as np

    indices = np.flatnonzero(np.asarray(mask)[start:])
    if not len(indices):
        raise Pi05ContractError(f"physics trace does not prove ordered {label}")
    return start + int(indices[0])


def _load_physics_trace(episode_dir: Path, length: int) -> dict[str, Any]:
    import numpy as np

    with np.load(episode_dir / "physics_trace.npz", allow_pickle=False) as loaded:
        trace = {key: loaded[key] for key in loaded.files}
    required = {
        "object_position_m",
        "object_speed_m_s",
        "left_finger_object_force_n",
        "right_finger_object_force_n",
        "object_support_force_n",
        "gripper_width_m",
        "hand_object_distance_m",
    }
    if set(trace) < required or any(len(trace[key]) != length for key in required):
        raise Pi05ContractError("physics trace is incomplete or not action-aligned")
    return trace


def _verify_physics_evidence(
    episode_dir: Path, arrays: dict[str, Any], metadata: dict[str, Any]
) -> None:
    import numpy as np

    trace = _load_physics_trace(episode_dir, len(arrays["actions"]))
    position = trace["object_position_m"]
    initial = np.asarray(metadata["initial_object_position_m"])
    target = np.asarray(metadata["target_position_m"])
    bilateral = (trace["left_finger_object_force_n"] > 1.0e-3) & (
        trace["right_finger_object_force_n"] > 1.0e-3
    )
    contact = _first_true(bilateral, 0, "contact")
    motion = np.linalg.norm(np.diff(np.vstack([initial, position]), axis=0), axis=1)
    width_ok = np.abs(trace["gripper_width_m"] - metadata["object_width_m"]) <= 0.015
    grasped = bilateral & (arrays["actions"][:, 7] > 0.5) & width_ok & (motion > 0.005)
    grasp = _first_true(grasped, contact, "grasp with object motion")
    lifted = position[:, 2] - initial[2] >= 0.05
    lift = _first_true(lifted, grasp, "lift")
    xy_error = np.linalg.norm(position[:, :2] - target[:2], axis=1)
    transport = _first_true(lifted & (xy_error < 0.08), lift, "transport")
    released = (trace["gripper_width_m"] >= 0.06) & ~bilateral
    release = _first_true(released, transport, "release")
    stable = released & (trace["object_support_force_n"] > 1.0e-3)
    stable &= (xy_error < 0.05) & (trace["object_speed_m_s"] < 0.03)
    stable &= trace["hand_object_distance_m"] >= 0.10
    runs = np.convolve(
        stable[release:].astype(np.int8), np.ones(3, dtype=np.int8), "valid"
    )
    if not len(runs) or int(runs.max()) < 3:
        raise Pi05ContractError(
            "physics trace lacks three stable post-retreat support steps"
        )


def _episode_arrays(episode_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    arrays = {
        "exterior": np.load(episode_dir / "obs_workspace.npy", allow_pickle=False),
        "wrist": np.load(episode_dir / "obs_wrist.npy", allow_pickle=False),
        "state": np.load(episode_dir / "state.npy", allow_pickle=False),
        "actions": np.load(episode_dir / "actions.npy", allow_pickle=False),
        "timestamps": np.load(episode_dir / "timestamps.npy", allow_pickle=False),
    }
    length = int(arrays["state"].shape[0])
    expected = {
        "exterior": (length, 224, 224, 3),
        "wrist": (length, 224, 224, 3),
        "state": (length, 8),
        "actions": (length, 8),
        "timestamps": (length,),
    }
    if {key: value.shape for key, value in arrays.items()} != expected:
        raise Pi05ContractError("dense episode arrays do not match metadata")
    if arrays["exterior"].dtype != np.uint8 or arrays["wrist"].dtype != np.uint8:
        raise Pi05ContractError("OpenPI camera arrays must be uint8")
    if arrays["state"].dtype != np.float32 or arrays["actions"].dtype != np.float32:
        raise Pi05ContractError("OpenPI state/actions must be float32")
    _verify_array_metadata(arrays, metadata)
    _verify_physics_evidence(episode_dir, arrays, metadata)
    return arrays


def _append_episode_samples(
    output: dict[str, list[Any]],
    lineage: list[dict[str, str]],
    arrays: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    length = int(arrays["state"].shape[0])
    for index in range(length - ACTION_HORIZON + 1):
        output["exterior_image"].append(arrays["exterior"][index])
        output["wrist_image"].append(arrays["wrist"][index])
        output["joint_position"].append(arrays["state"][index, :7])
        output["gripper_position"].append(arrays["state"][index, 7:8])
        output["actions"].append(arrays["actions"][index : index + ACTION_HORIZON])
        output["prompts"].append(str(metadata["instruction"]))
        sample_id = f"{metadata['episode_id']}:{index:06d}"
        output["sample_ids"].append(sample_id)
        lineage.append(
            {
                "sample_id": sample_id,
                "episode_id": str(metadata["episode_id"]),
                "object_identity": str(metadata["object_identity"]),
                "scene_configuration_digest": str(
                    metadata["scene_configuration_digest"]
                ),
            }
        )


def _samples(
    episodes: list[tuple[Path, dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    import numpy as np

    keys = (
        "exterior_image",
        "wrist_image",
        "joint_position",
        "gripper_position",
        "actions",
        "prompts",
        "sample_ids",
    )
    output: dict[str, list[Any]] = {key: [] for key in keys}
    lineage: list[dict[str, str]] = []
    for episode_dir, metadata in episodes:
        _append_episode_samples(
            output, lineage, _episode_arrays(episode_dir, metadata), metadata
        )
    if not output["sample_ids"]:
        raise Pi05ContractError(
            "successful episodes contain no complete action horizon"
        )
    arrays = {key: np.asarray(value) for key, value in output.items()}
    arrays["exterior_image"] = np.stack(output["exterior_image"]).astype(np.uint8)
    arrays["wrist_image"] = np.stack(output["wrist_image"]).astype(np.uint8)
    for key in ("joint_position", "gripper_position", "actions"):
        arrays[key] = np.stack(output[key]).astype(np.float32)
    return arrays, lineage


def _normalization(train: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    values = np.concatenate(
        [
            np.asarray(train["joint_position"], dtype=np.float64),
            np.asarray(train["gripper_position"], dtype=np.float64),
        ],
        axis=1,
    )
    action = np.asarray(train["actions"], dtype=np.float64).reshape(-1, 8)
    payload = {
        "source_split": "train",
        "state_mean": values.mean(axis=0).tolist(),
        "state_std": np.maximum(values.std(axis=0), 1.0e-6).tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": np.maximum(action.std(axis=0), 1.0e-6).tolist(),
    }
    payload["sha256"] = _sha_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    return payload


def _split_episodes(
    train_root: Path, validation_root: Path, gold_root: Path
) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    return {
        "train": _load_successes(train_root, "train"),
        "validation": _load_successes(validation_root, "validation"),
        "gold": _load_successes(gold_root, "gold"),
    }


def _split_rows(
    split_episodes: dict[str, list[tuple[Path, dict[str, Any]]]],
) -> dict[str, Any]:
    seen_objects: set[str] = set()
    seen_scenes: set[str] = set()
    split_rows: dict[str, list[dict[str, str]]] = {}
    for split, episodes in split_episodes.items():
        rows = [
            {
                "episode_id": str(metadata["episode_id"]),
                "object_identity": str(metadata["object_identity"]),
                "scene_configuration_digest": str(
                    metadata["scene_configuration_digest"]
                ),
            }
            for _, metadata in episodes
        ]
        objects = {row["object_identity"] for row in rows}
        scenes = {row["scene_configuration_digest"] for row in rows}
        if objects & seen_objects or scenes & seen_scenes:
            raise Pi05ContractError(
                "object identity or scene configuration leaked across splits"
            )
        seen_objects |= objects
        seen_scenes |= scenes
        split_rows[split] = rows
    return split_rows


def _sample_hashes(values: dict[str, Any]) -> list[str]:
    return [
        _sample_hash(
            {
                "exterior_image": values["exterior_image"][index],
                "wrist_image": values["wrist_image"][index],
                "joint_position": values["joint_position"][index],
                "gripper_position": values["gripper_position"][index],
                "actions": values["actions"][index],
                "prompt": str(values["prompts"][index]),
            }
        )
        for index in range(len(values["sample_ids"]))
    ]


def _openpi_archive(
    split_episodes: dict[str, Any], output: Path
) -> tuple[dict[str, Any], ...]:
    train_arrays, train_lineage = _samples(split_episodes["train"])
    heldout_arrays, validation_lineage = _samples(split_episodes["validation"])
    arrays: dict[str, Any] = {}
    for split_name, values in (("train", train_arrays), ("heldout", heldout_arrays)):
        arrays.update({f"{split_name}_{key}": value for key, value in values.items()})
    hashes = {
        "train": _sample_hashes(train_arrays),
        "heldout": _sample_hashes(heldout_arrays),
    }
    if set(hashes["train"]) & set(hashes["heldout"]):
        raise Pi05ContractError("train and validation content hashes overlap")
    archive = deterministic_npz(arrays)
    (output / "openpi-dataset.npz").write_bytes(archive)
    return (
        train_arrays,
        heldout_arrays,
        train_lineage,
        validation_lineage,
        hashes,
        archive,
    )


def _openpi_contract() -> dict[str, Any]:
    return {
        "exterior_image": {"shape": [224, 224, 3], "dtype": "uint8"},
        "wrist_image": {
            "shape": [224, 224, 3],
            "dtype": "uint8",
            "mount": "panda_hand",
        },
        "joint_position": {"shape": [7], "dtype": "float32", "units": "rad"},
        "gripper_position": {
            "shape": [1],
            "dtype": "float32",
            "convention": "DROID 0=open 1=closed",
        },
        "actions": {
            "shape": [15, 8],
            "dtype": "float32",
            "semantics": "absolute joint radians plus DROID gripper",
        },
    }


def _split_manifest(
    arrays: dict[str, Any], hashes: list[str], lineage: list[dict[str, str]]
) -> dict[str, Any]:
    return {
        "count": len(arrays["sample_ids"]),
        "sample_ids": [str(value) for value in arrays["sample_ids"]],
        "sample_hashes": hashes,
        "lineage": lineage,
    }


def _write_openpi_manifest(
    output: Path,
    archive: bytes,
    train: dict[str, Any],
    heldout: dict[str, Any],
    lineages: tuple[list[Any], list[Any]],
    hashes: dict[str, list[str]],
    gold_rows: list[dict[str, str]],
    normalization: dict[str, Any],
) -> None:
    manifest = {
        "schema": OPENPI_SCHEMA,
        "contract": _openpi_contract(),
        "archive_sha256": _sha_bytes(archive),
        "archive_size_bytes": len(archive),
        "splits": {
            "train": _split_manifest(train, hashes["train"], lineages[0]),
            "heldout": _split_manifest(heldout, hashes["heldout"], lineages[1]),
        },
        "gold": {"episodes": gold_rows, "used_for_training_or_selection": False},
        "split_isolation": {
            "sample_id_intersection": [],
            "sample_hash_intersection": [],
            "disjoint": True,
            "object_identity_disjoint": True,
            "scene_configuration_disjoint": True,
        },
        "normalization": normalization,
        "limitations": [
            "simulation_demonstrations",
            "physical_robot_validation_not_performed",
        ],
    }
    (output / "openpi-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _convert_lerobot(
    episodes: list[tuple[Path, dict[str, Any]]], output: Path
) -> dict[str, Any]:
    lerobot_input = output / "lerobot-input"
    for index, (episode_dir, _) in enumerate(episodes):
        destination = lerobot_input / f"episode_{index:06d}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in (
            "obs_workspace.npy",
            "obs_wrist.npy",
            "state.npy",
            "actions.npy",
            "timestamps.npy",
        ):
            shutil.copy2(episode_dir / name, destination / name)
    from npa.adapter.sim_to_lerobot import convert

    root = convert(
        lerobot_input,
        output / "lerobot-v3",
        fps=15,
        robot_type="franka_panda",
        task="Pick up the object and release it inside the target area on the table.",
    )
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("total_episodes") != len(episodes):
        raise Pi05ContractError("LeRobot readback episode count mismatch")
    if info.get("timestamp_source") != "input_episode_relative":
        raise Pi05ContractError("LeRobot did not preserve source capture timestamps")
    return info


def _verify_openpi_readback(output: Path) -> None:
    import numpy as np

    with np.load(output / "openpi-dataset.npz", allow_pickle=False) as readback:
        if readback["train_actions"].shape[1:] != (15, 8):
            raise Pi05ContractError("OpenPI readback action contract mismatch")


def _dataset_report(
    split_rows: dict[str, Any],
    normalization: dict[str, Any],
    train: dict[str, Any],
    heldout: dict[str, Any],
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": DATASET_SCHEMA,
        "splits": split_rows,
        "normalization": normalization,
        "openpi": {
            "archive": "openpi-dataset.npz",
            "manifest": "openpi-manifest.json",
            "train_samples": len(train["sample_ids"]),
            "validation_samples": len(heldout["sample_ids"]),
        },
        "lerobot": {
            "root": "lerobot-v3",
            "codebase_version": info.get("codebase_version"),
            "episodes": info.get("total_episodes"),
            "frames": info.get("total_frames"),
            "timestamp_source": info.get("timestamp_source"),
        },
    }


def export_local(
    *, train_root: Path, validation_root: Path, gold_root: Path, output: Path
) -> dict[str, Any]:
    """Export native data. Args: split roots/output. Returns: Report. Raises: Pi05ContractError."""

    split_episodes = _split_episodes(train_root, validation_root, gold_root)
    split_rows = _split_rows(split_episodes)
    output.mkdir(parents=True)
    values = _openpi_archive(split_episodes, output)
    train_arrays, heldout_arrays, train_lineage, validation_lineage, hashes, archive = (
        values
    )
    normalization = _normalization(train_arrays)
    _write_openpi_manifest(
        output,
        archive,
        train_arrays,
        heldout_arrays,
        (train_lineage, validation_lineage),
        hashes,
        split_rows["gold"],
        normalization,
    )
    info = _convert_lerobot(split_episodes["train"], output)
    _verify_openpi_readback(output)
    report = _dataset_report(
        split_rows, normalization, train_arrays, heldout_arrays, info
    )
    (output / "dataset-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI. Args: None. Returns: Parser. Raises: None."""
    parser = argparse.ArgumentParser()
    for split in ("train", "validation", "gold"):
        parser.add_argument(f"--{split}-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--component-root-uri", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Args: argv. Returns: Exit status. Raises: Pi05ContractError."""
    args = build_parser().parse_args(argv)

    from npa.clients.storage import StorageClient

    storage = StorageClient.from_environment()
    work = Path(tempfile.mkdtemp(prefix="npa-pi05-data-"))
    roots: dict[str, Path] = {}
    for split in ("train", "validation", "gold"):
        root = work / split
        storage.download_directory(getattr(args, f"{split}_uri"), str(root))
        roots[split] = root
    output = work / "output"
    report = export_local(
        train_root=roots["train"],
        validation_root=roots["validation"],
        gold_root=roots["gold"],
        output=output,
    )
    storage.upload_directory(str(output), args.output_uri, require_empty=True)
    from npa.workflows.sim2real.workflow_io import publish_component_record

    publish_component_record(
        root_uri=args.component_root_uri,
        stage=3,
        name="pi05_dense_dataset",
        tier="WORKS",
        evidence="Exported split-safe dense OpenPI and native LeRobot data.",
        artifacts={
            "dataset": args.output_uri,
            "normalization_sha256": report["normalization"]["sha256"],
        },
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
