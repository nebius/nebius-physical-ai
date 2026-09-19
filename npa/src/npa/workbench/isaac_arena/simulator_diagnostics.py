"""Retain unfinished simulator data without creating a scored Arena episode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import file_sha256


def _plain_state(value: Any) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        torch = None

    if isinstance(value, dict):
        return {key: _plain_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_state(item) for item in value]
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("unsupported simulator diagnostic state type")


def _write_diagnostic(directory: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = directory / "simulator-unscored-episode.json"
    path.write_text(
        json.dumps(_plain_state(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return {"path": path.name, "sha256": file_sha256(path), "scored": False}


def retain_unscored_episode(
    env: Any, directory: Path, captured_action_steps: int
) -> dict[str, Any] | None:
    """Persist the remaining recorder buffer separately from scored results.

    Args:
        env: Live, single-environment Arena instance before recorder teardown.
        directory: Current run's upstream artifact directory.
        captured_action_steps: Total source actions actually captured.
    Returns:
        Diagnostic path and hash, or None when the recorder buffer is empty.
    Raises:
        ValueError: The count, environment, or measured data is invalid.
        OSError: The diagnostic cannot be written.
    """
    if (
        env.num_envs != 1
        or type(captured_action_steps) is not int
        or captured_action_steps < 0
    ):
        raise ValueError(
            "unscored video diagnostics require one environment and a valid action count"
        )
    episode = env.recorder_manager.get_episode(0)
    if not episode.data:
        return None
    payload = {
        "schema": "npa.isaac-arena.unscored-diagnostic.v1",
        "scored": False,
        "task_success_claimed": False,
        "scope": "remaining recorder buffer after the last completed episode export",
        "captured_action_steps": captured_action_steps,
        "initial_physics_state": getattr(env, "_npa_video_initial_physics_state", None),
        "final_scene_state": env.scene.get_state(is_relative=True),
        "remaining_recorder_buffer": episode.data,
    }
    return _write_diagnostic(directory, payload)
