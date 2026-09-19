"""Validate policy inputs and preserve replay commands in the runtime's pose format."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .action_evidence import action_sequence_evidence
from .errors import IsaacArenaError
from .hashing import file_sha256 as _sha256
from .replay_quaternions import dataset_format_version, normalize_pose_representation
from .replay_target_poses import validate_recorded_target_poses

if TYPE_CHECKING:
    from .runtime import IsaacArenaRequest


def _replay_dependencies(operation: str) -> tuple[Any, Any]:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:  # pragma: no cover - supplied by the Isaac runtime
        raise IsaacArenaError(f"replay {operation} requires h5py and numpy") from exc
    return h5py, np


def _checkpoint_input(local_input: Path) -> tuple[Path, Path]:
    checkpoint = local_input
    if checkpoint.is_dir():
        candidates = sorted(checkpoint.glob("model*.pt"))
        if not candidates:
            candidates = sorted(checkpoint.rglob("model*.pt"))
        if len(candidates) != 1:
            raise IsaacArenaError(
                "rsl_rl input directory must contain exactly one model*.pt checkpoint"
            )
        checkpoint = candidates[0]
    if checkpoint.suffix != ".pt" or not checkpoint.is_file():
        raise IsaacArenaError("rsl_rl input_path must resolve to a .pt checkpoint")
    agent_config = checkpoint.parent / "params" / "agent.yaml"
    if not agent_config.is_file():
        raise IsaacArenaError("rsl_rl checkpoint requires sibling params/agent.yaml")
    return checkpoint, agent_config


def _replay_action_statistics(actions: Any, np: Any) -> dict[str, Any]:
    if (
        actions.ndim < 2
        or actions.shape[0] < 2
        or not actions.size
        or actions.dtype.kind not in "biuf"
        or not np.isfinite(actions).all()
    ):
        raise IsaacArenaError("replay actions must be a finite multi-step tensor")
    measured_actions = actions.astype(np.float64)
    action_abs_max = float(np.max(np.abs(measured_actions)))
    action_nonzero_fraction = float(np.mean(np.abs(measured_actions) > 1e-6))
    return {
        "steps": int(actions.shape[0]),
        "action_dimensions": int(actions.shape[-1]),
        "action_abs_max": action_abs_max,
        "action_abs_mean": float(np.mean(np.abs(measured_actions))),
        "action_nonzero_fraction": action_nonzero_fraction,
        "nonzero_actions": action_abs_max >= 1e-4 and action_nonzero_fraction >= 0.001,
        "action_step_delta_mean": float(
            np.mean(np.abs(np.diff(measured_actions, axis=0)))
        ),
    }


def _recorded_state_ranges(
    episode: Any,
    steps: int,
    h5py: Any,
    np: Any,
) -> list[tuple[str, float]]:
    state_ranges: list[tuple[str, float]] = []

    def inspect_state(name: str, item: Any) -> None:
        if not isinstance(item, h5py.Dataset) or not name.startswith("states/"):
            return
        if len(item.shape) < 1 or item.shape[0] != steps:
            return
        values = np.asarray(item)
        if (
            not values.size
            or values.dtype.kind not in "biuf"
            or not np.isfinite(values).all()
        ):
            return
        state_ranges.append((name, float(np.max(np.ptp(values.astype(float), axis=0)))))

    episode.visititems(inspect_state)
    state_ranges.sort(key=lambda item: (-item[1], item[0]))
    return state_ranges


def _validate_initial_state(episode: Any, h5py: Any, np: Any) -> None:
    initial = episode.get("initial_state")
    if not isinstance(initial, h5py.Group):
        raise IsaacArenaError("replay HDF5 episode contains no initial_state group")
    tensors = []

    def inspect(name: str, item: Any) -> None:
        if not isinstance(item, h5py.Dataset):
            return
        values = np.asarray(item)
        if (
            not values.size
            or values.ndim < 1
            or values.dtype.kind not in "biuf"
            or not np.isfinite(values).all()
        ):
            raise IsaacArenaError(
                "replay initial_state must contain finite real state tensors"
            )
        tensors.append(name)

    initial.visititems(inspect)
    if not tensors:
        raise IsaacArenaError("replay initial_state contains no state tensors")


def _episode_input_evidence(dataset: Any, h5py: Any, np: Any) -> dict[str, Any]:
    episodes = sorted(str(name) for name in dataset.get("data", {}))
    if not episodes:
        raise IsaacArenaError("replay HDF5 contains no episodes under data/")
    episode_name = episodes[0]
    episode = dataset["data"][episode_name]
    if "actions" not in episode:
        raise IsaacArenaError("replay HDF5 episode contains no actions")
    _validate_initial_state(episode, h5py, np)
    actions = np.asarray(episode["actions"])
    statistics = _replay_action_statistics(actions, np)
    state_ranges = _recorded_state_ranges(episode, actions.shape[0], h5py, np)
    return {
        "episode": episode_name,
        "dataset_format_version": dataset_format_version(dataset),
        "source_recorded_success": (
            bool(episode.attrs["success"]) if "success" in episode.attrs else None
        ),
        **statistics,
        "state_max_range": state_ranges[0][1] if state_ranges else None,
        "recorded_state_available": bool(state_ranges),
        "recorded_state_changes": bool(state_ranges and state_ranges[0][1] >= 1e-4),
        "largest_state_ranges": [
            {"dataset": name, "range": value} for name, value in state_ranges[:5]
        ],
        "runtime_outcome_claim": False,
    }


def _replay_input_evidence(path: Path) -> dict[str, Any]:
    """Validate upstream replay inputs and report optional behavior diagnostics."""
    h5py, np = _replay_dependencies("validation")
    try:
        with h5py.File(path, "r") as dataset:
            return _episode_input_evidence(dataset, h5py, np)
    except OSError as exc:
        raise IsaacArenaError("replay input is not a readable HDF5 file") from exc


def _copy_execution_episode(
    source_file: Any,
    output_file: Any,
    episode_name: str,
    actions: Any,
) -> None:
    source_episode = source_file["data"][episode_name]
    source_steps = int(actions.shape[0])
    for key, value in source_file.attrs.items():
        output_file.attrs[key] = value
    output_data = output_file.create_group("data")
    for key, value in source_file["data"].attrs.items():
        output_data.attrs[key] = value
    output_data.attrs["total"] = source_steps
    output_episode = output_data.create_group(episode_name)
    for key, value in source_episode.attrs.items():
        output_episode.attrs[key] = value
    output_episode.attrs["num_samples"] = source_steps
    output_episode.create_dataset("actions", data=actions, compression="gzip")
    if "initial_state" not in source_episode:
        raise IsaacArenaError("replay HDF5 episode contains no initial_state")
    source_episode.copy("initial_state", output_episode)


def _write_execution_input(
    source: Path, destination: Path, episode_name: str, embodiment: str
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    h5py, np = _replay_dependencies("normalization")
    try:
        with h5py.File(source, "r") as source_file:
            target_poses = validate_recorded_target_poses(
                source_file, episode_name, embodiment
            )
            actions = np.asarray(source_file["data"][episode_name]["actions"])
            with h5py.File(destination, "w") as output_file:
                _copy_execution_episode(source_file, output_file, episode_name, actions)
                representation = normalize_pose_representation(
                    source_file, output_file, episode_name, embodiment
                )
                if target_poses is not None:
                    representation["recorded_target_poses"] = target_poses
                prepared_actions = np.asarray(
                    output_file[f"data/{episode_name}/actions"]
                )
                prepared_action_sequence = action_sequence_evidence(prepared_actions)
    except (KeyError, OSError) as exc:
        raise IsaacArenaError(
            "replay input cannot be normalized for execution"
        ) from exc
    return int(actions.shape[0]), representation, prepared_action_sequence


def _prepare_replay_execution_input(
    source: Path,
    private_dir: Path,
    *,
    source_evidence: dict[str, Any],
    embodiment: str = "",
) -> tuple[Path, dict[str, Any]]:
    """Keep the replay sequence, translating declared legacy Pink pose coordinates.

    Upstream loads every field onto the execution device but only consumes
    actions and initial state. Both source and execution input remain private.
    """
    episode_name = str((source_evidence.get("trajectory") or {}).get("episode") or "")
    if not episode_name:
        raise IsaacArenaError("replay evidence does not identify an episode")
    destination = private_dir / "replay-execution.hdf5"
    private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_steps, representation, prepared_action_sequence = _write_execution_input(
        source, destination, episode_name, embodiment
    )
    destination.chmod(0o600)
    return destination, {
        "strategy": "actions_initial_state_exact_replay",
        "pose_representation": representation,
        "source_sha256": str(source_evidence.get("sha256") or ""),
        "prepared_sha256": _sha256(destination),
        "source_steps": source_steps,
        "prepared_steps": source_steps,
        "prepared_action_sequence": prepared_action_sequence,
        "runtime_outcome_claim": False,
        "action_padding_steps": 0,
        "initial_state_application": "isaac_lab_reset_to_relative",
        "fields": ["actions", "initial_state"],
        "published": False,
    }


def _input_evidence(
    request: IsaacArenaRequest,
    local_input: Path | None,
) -> dict[str, Any] | None:
    if local_input is None:
        return None
    if request.policy_type == "replay":
        if not local_input.is_file():
            raise IsaacArenaError("replay input_path must resolve to one HDF5 file")
        return {
            "kind": "replay_hdf5",
            "bytes": local_input.stat().st_size,
            "sha256": _sha256(local_input),
            "trajectory": _replay_input_evidence(local_input),
        }
    checkpoint, agent_config = _checkpoint_input(local_input)
    return {
        "kind": "rsl_rl_checkpoint",
        "bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
        "agent_config_sha256": _sha256(agent_config),
    }
