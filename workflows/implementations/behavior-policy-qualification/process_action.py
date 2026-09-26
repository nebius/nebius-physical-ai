"""Load one pinned Comet profile and record its first action in a fresh process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import comet_policy
import comet_server
import numpy as np

VERSIONS = {
    "jax": "0.5.3",
    "jaxlib": "0.5.3",
    "flax": "0.10.2",
    "numpy": "1.26.4",
    "orbax-checkpoint": "0.11.13",
    "torch": "2.7.1",
    "opencv-python": "4.11.0.86",
    "websockets": "15.0.1",
}


def _runtime_probe() -> tuple[dict, object]:
    """Verify the pinned interpreter and return its one visible GPU."""
    actual = {name: importlib.metadata.version(name) for name in VERSIONS}
    comparable = {**actual, "torch": actual["torch"].split("+", 1)[0]}
    if comparable != VERSIONS:
        raise ValueError(f"Retained OpenPI runtime versions differ: {actual}")
    import jax

    devices = list(jax.devices())
    if platform.python_implementation() != "CPython" or tuple(
        map(int, platform.python_version_tuple()[:2])
    ) != (3, 11):
        raise ValueError("Comet qualification requires CPython 3.11")
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("Comet qualification requires exactly one visible GPU")
    result = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": actual,
        "jax_devices": [
            {"platform": device.platform, "kind": device.device_kind}
            for device in devices
        ],
    }
    return result, devices[0]


def _allocator_memory(device: object) -> dict:
    """Read current-process JAX allocator evidence when the backend exposes it."""
    try:
        values = device.memory_stats()
    except (AttributeError, RuntimeError) as error:
        return _unavailable_memory(f"memory_stats_error:{type(error).__name__}")
    if not isinstance(values, dict) or type(values.get("peak_bytes_in_use")) is not int:
        return _unavailable_memory("peak_bytes_in_use_unavailable")
    return {
        "available": True,
        "source": "jax_device_memory_stats",
        "identity": "current_policy_process_and_visible_jax_device",
        "peak_bytes_in_use": values["peak_bytes_in_use"],
        "bytes_in_use": values.get("bytes_in_use"),
        "bytes_limit": values.get("bytes_limit"),
    }


def _unavailable_memory(reason: str) -> dict:
    return {
        "available": False,
        "source": "jax_device_memory_stats",
        "identity": "current_policy_process_and_visible_jax_device",
        "reason": reason,
        "peak_bytes_in_use": None,
    }


def _load_observation(path: Path) -> dict[str, np.ndarray]:
    """Load the frozen NPZ fixture and exercise the exact observation adapter."""
    with np.load(path, allow_pickle=False) as source:
        observation = {name: source[name] for name in source.files}
    comet_policy.policy_observation(observation)
    return observation


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=tuple(comet_policy.COMET_PROFILES), required=True
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _action_identity(action: np.ndarray, path: Path) -> dict:
    """Describe exact in-memory and NPY action bytes."""
    return {
        "shape": list(action.shape),
        "dtype": str(action.dtype),
        "raw_sha256": hashlib.sha256(action.tobytes(order="C")).hexdigest(),
        "npy_sha256": comet_policy._digest(path),
        "bytes": path.stat().st_size,
    }


def _infer(args, profile, observation, device):
    with tempfile.TemporaryDirectory(
        prefix=f"npa-{profile.kind}-qualification-"
    ) as temporary:
        overlay = comet_policy.build_source_overlay(args.source_root, Path(temporary))
        policy = comet_server._load_policy(args, overlay)
        after_load = _allocator_memory(device)
        loaded = time.monotonic()
        policy.reset()
        action = comet_policy.validate_action(
            policy.act(comet_policy.policy_observation(observation))
        )
        after_action = _allocator_memory(device)
    return action, after_load, after_action, loaded


def _result(args, profile, runtime, source_files, memory, action, timing):
    return {
        "schema": "npa.behavior.comet-fresh-process-action.v1",
        "status": "first_synthetic_action_complete",
        "profile": profile.kind,
        "task_id": 1,
        "task_name": "picking_up_trash",
        "runtime": runtime,
        "source_commit": comet_policy.SOURCE_COMMIT,
        "source_files": source_files,
        "checkpoint_revision": profile.revision,
        "policy_process_allocator_memory": memory,
        "action": _action_identity(action, args.action),
        "timing_seconds": timing,
    }


def _execute(args) -> dict:
    started = time.monotonic()
    runtime, device = _runtime_probe()
    profile = comet_policy.get_profile(args.profile)
    source_files = comet_policy.verify_source(args.source_root)
    comet_policy.task_identity(args.source_root, 1, "picking_up_trash", profile)
    observation = _load_observation(args.observation)
    args.task_id, args.task_name, args.profile = 1, "picking_up_trash", profile
    action, after_load, after_action, loaded = _infer(
        args, profile, observation, device
    )
    args.action.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.action, action, allow_pickle=False)
    return _result(
        args,
        profile,
        runtime,
        source_files,
        {"after_load": after_load, "after_first_action": after_action},
        action,
        {
            "environment_source_and_load": loaded - started,
            "first_action_including_jit": time.monotonic() - loaded,
        },
    )


def main() -> None:
    """Load one profile and record its first action.

    Args:
        None.
    Returns:
        None.
    Raises:
        ValueError: Runtime, source, task, observation, or action validation fails.
    """
    args = _arguments()
    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        WANDB_MODE="disabled",
        PYTHONUNBUFFERED="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    result = _execute(args)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
