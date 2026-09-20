"""Real RoboCasa capability operations.

This module is the single source of truth for RoboCasa capability behavior. The
FastAPI service, the CLI, and the SDK all call into it. It exercises the real
upstream RoboCasa surface: Gymnasium task registration, kitchen asset
availability, headless EGL environment reset, and a random rollout with a video
artifact.

GPU-heavy imports (robocasa, robosuite, mujoco, gymnasium) are deferred to call
time so that importing this module on a client without the simulation stack
never fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
from pathlib import Path
from typing import Any, Callable

from npa.clients.storage import safe_s3_download_target
from npa.workbench.robocasa.schemas import (
    DEFAULT_ENV_ID,
    RoboCasaRunRequest,
    RoboCasaSystemInfo,
)

LOGGER = logging.getLogger(__name__)

#: Capabilities this tool can exercise, keyed by the upstream capability id.
SUPPORTED_CAPABILITIES = {
    "kitchen_task_registration",
    "kitchen_asset_availability",
    "kitchen_egl_env_reset",
    "kitchen_random_rollout",
    "kitchen_trajectory_export",
    "kitchen_policy_eval",
}

ROBOCASA_EMBODIMENT = "PandaOmron"
ROBOCASA_OBJECT_REGISTRIES = ("objaverse",)
ROBOCASA_STATE_KEYS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)
_SOURCE_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class RoboCasaError(RuntimeError):
    """Raised when a RoboCasa capability operation fails."""


@dataclass(frozen=True)
class _PreparedAction:
    """Keep the raw policy action distinct from the action applied to the env."""

    value: Any
    raw_flat: np.ndarray
    applied_flat: np.ndarray
    max_bound_violation: float


@dataclass
class _ActionTrace:
    """Hash raw/applied actions and count action-bound corrections."""

    raw_digest: Any = field(default_factory=hashlib.sha256)
    applied_digest: Any = field(default_factory=hashlib.sha256)
    out_of_bounds_steps: int = 0
    max_bound_violation: float = 0.0

    def update(self, prepared: _PreparedAction) -> None:
        self.raw_digest.update(prepared.raw_flat.tobytes())
        self.applied_digest.update(prepared.applied_flat.tobytes())
        if prepared.max_bound_violation > 0:
            self.out_of_bounds_steps += 1
        self.max_bound_violation = max(
            self.max_bound_violation, prepared.max_bound_violation
        )


def make_run_id(capability: str, manifest: str) -> str:
    """Build a deterministic run id from a capability and request manifest."""
    digest = hashlib.sha256(f"{capability}:{manifest}".encode("utf-8")).hexdigest()[:12]
    return f"robocasa-{capability}-{digest}"


def compute_manifest_sha256(capability: str, payload: dict[str, Any]) -> str:
    """Compute a content hash over a request payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{capability}:{canonical}".encode("utf-8")).hexdigest()


def _import_robocasa() -> Any:
    """Import the real robocasa package, raising a clear error if absent."""
    try:
        import robocasa  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError(
            "robocasa is not installed in this environment; run inside the "
            "npa-robocasa image"
        ) from exc
    return robocasa


def _import_gymnasium() -> Any:
    try:
        import gymnasium as gym
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError("gymnasium is not installed in this environment") from exc
    return gym


def _assets_root() -> Path:
    """Locate RoboCasa assets without importing its eager object catalog.

    Importing :mod:`robocasa` before the runtime-fetched object assets exist
    permanently caches empty ``mjcf_paths`` in the service process.  Keep
    read-only system-info and asset checks from changing later simulation
    behavior.
    """
    import importlib.util

    loaded = sys.modules.get("robocasa")
    loaded_path = getattr(loaded, "__file__", None)
    if loaded_path:
        return Path(loaded_path).resolve().parent / "models" / "assets"
    try:
        robocasa_spec = importlib.util.find_spec("robocasa")
    except ValueError:
        robocasa_spec = None
    if robocasa_spec is None or robocasa_spec.origin is None:
        raise RoboCasaError("robocasa package not found")
    return Path(robocasa_spec.origin).resolve().parent / "models" / "assets"


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # pragma: no cover - best effort.
        return ""


def _runtime_source_identity() -> tuple[str, str]:
    """Return the immutable NPA source identity carried by the runtime image."""
    source_sha = os.environ.get("NPA_IMAGE_SOURCE_SHA", "").strip().lower()
    required = os.environ.get("ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA", "").strip().lower()
    if required not in {"", "0", "false", "1", "true"}:
        raise RoboCasaError("ROBOCASA_REQUIRE_IMAGE_SOURCE_SHA must be boolean")
    if source_sha and not _SOURCE_SHA_PATTERN.fullmatch(source_sha):
        raise RoboCasaError("NPA_IMAGE_SOURCE_SHA must be a 40-character git SHA")
    if required in {"1", "true"} and not source_sha:
        raise RoboCasaError("RoboCasa image is missing required NPA_IMAGE_SOURCE_SHA")
    identity = "container_image" if source_sha else "local_unbound"
    return identity, source_sha


def system_info() -> RoboCasaSystemInfo:
    """Collect system and RoboCasa stack information."""
    source_identity, image_source_sha = _runtime_source_identity()
    info = RoboCasaSystemInfo(
        status="ok",
        python=platform.python_version(),
        platform=platform.platform(),
        robocasa_version=_package_version("robocasa"),
        robosuite_version=_package_version("robosuite"),
        mujoco_version=_package_version("mujoco"),
        gymnasium_version=_package_version("gymnasium"),
        source_identity=source_identity,
        image_source_sha=image_source_sha,
    )
    try:
        import torch

        info.cuda_available = bool(torch.cuda.is_available())
        info.cuda_device_count = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        )
        if torch.cuda.is_available():
            info.cuda_device_name = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - torch optional on client.
        LOGGER.debug("torch unavailable: %s", exc)
    try:
        gym = _import_gymnasium()
        robocasa_envs = sorted(
            env for env in gym.envs.registry.keys() if env.startswith("robocasa/")
        )
        info.registered_env_count = len(robocasa_envs)
    except Exception as exc:  # pragma: no cover - depends on the container.
        LOGGER.debug("gymnasium unavailable: %s", exc)
    try:
        info.assets_root_exists = _assets_root().exists()
    except Exception as exc:  # pragma: no cover - depends on the container.
        LOGGER.debug("assets root unavailable: %s", exc)
    return info


def _download_assets() -> None:
    """Download the RoboCasa kitchen assets (textures, fixtures, objects).

    Assets are NOT baked into the image and download at runtime from the
    operator's entitled Hugging Face identity. This mirrors the upstream
    ``download_kitchen_assets.py`` registry but skips its interactive prompt so
    it can run inside the service. Missing assets are the usual cause of a
    ``model.xml`` FileNotFoundError on the first real rollout.

    The standard fixtures (stoves, windows, sinks, ...) live in
    ``robocasa/robocasa-assets/fixtures.zip``; the lightwheel variants are
    published as individual ``fixtures_lightwheel/<name>.zip`` files in
    ``nvidia/PhysicalAI-Kitchen-Assets``.
    """
    try:
        from huggingface_hub import hf_hub_download
        from zipfile import ZipFile
        from pathlib import Path as _Path

        # Locate the package WITHOUT importing its eager object catalog.
        assets_root = _Path(_assets_root())
        # Standard (non-additive) assets: skip when the target directory already
        # has content. (repo_id, filename, extract_to, marker_dir)
        standard = [
            ("robocasa/robocasa-assets", "textures.zip", ".", "textures"),
            (
                "robocasa/robocasa-assets",
                "generative_textures.zip",
                ".",
                "generative_textures",
            ),
            ("robocasa/robocasa-assets", "fixtures.zip", ".", "fixtures/accessories"),
            (
                "robocasa/robocasa-assets",
                "objaverse.zip",
                "objects",
                "objects/objaverse",
            ),
            (
                "robocasa/robocasa-assets",
                "aigen_objs.zip",
                "objects",
                "objects/aigen_objs",
            ),
        ]
        # Lightwheel fixtures are one zip per fixture family, each extracting a
        # top-level folder (e.g. stoves/) that must land under fixtures/. They are
        # additive on top of baked directories, so track completion with a marker.
        lightwheel_fixtures = [
            "blenders",
            "cabinets",
            "coffee_machines",
            "dishwashers",
            "electric_kettles",
            "fridges",
            "handles",
            "hoods",
            "microwaves",
            "ovens",
            "sinks",
            "stand_mixers",
            "stoves",
            "stovetops",
            "toaster_ovens",
            "toasters",
            "windows",
        ]
        # Lightwheel objects are one zip per object family, each extracting a
        # top-level folder (e.g. stool/) that must land under objects/lightwheel/.
        lightwheel_objects = [
            "aluminum_foil",
            "basket",
            "blender_jug",
            "cheese_grater",
            "chicken_drumstick",
            "cinnamon",
            "colander",
            "cookie_dough_ball",
            "cream_cheese_stick",
            "digital_scale",
            "dish_brush",
            "dish_rack",
            "flour_bag",
            "flower_vase",
            "fruit_bowl",
            "glass_cup",
            "honey_bottle",
            "hotdog_bun",
            "ice_cube",
            "ice_cube_tray",
            "jar",
            "juice",
            "kebab_skewer",
            "kettle",
            "knife_block",
            "lemon_wedge",
            "lettuce",
            "marshmallow",
            "mayonnaise",
            "measuring_cup",
            "mug_tree",
            "mustard",
            "oil_and_vinegar_bottle",
            "oven_tray",
            "pancake",
            "paper_towel_holder",
            "paprika",
            "peeler",
            "pickle_slice",
            "pitcher",
            "pizza",
            "pizza_cutter",
            "placemat",
            "plant",
            "pot",
            "reamer",
            "salt_and_pepper_shaker",
            "sandwich_bread",
            "saucepan",
            "shrimp",
            "soap_dispenser",
            "spray",
            "stool",
            "strainer",
            "straw",
            "sugar_cube",
            "syrup_bottle",
            "tiered_basket",
            "tiered_shelf",
            "tomato_slice",
            "tongs",
            "tray",
            "tupperware",
            "turkey_slice",
            "turmeric",
            "utensil_rack",
            "utensil_set",
            "whisk",
            "wooden_spoon",
        ]
        lightwheel = [
            (
                "nvidia/PhysicalAI-Kitchen-Assets",
                f"fixtures_lightwheel/{name}.zip",
                "fixtures",
                f"fixtures/{name}",
            )
            for name in lightwheel_fixtures
        ] + [
            (
                "nvidia/PhysicalAI-Kitchen-Assets",
                f"objects_lightwheel/{name}.zip",
                "objects/lightwheel",
                f"objects/lightwheel/{name}",
            )
            for name in lightwheel_objects
        ]

        def _extract(repo_id: str, filename: str, extract_to: str) -> None:
            zip_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                revision="main",
            )
            dest = assets_root if extract_to == "." else assets_root / extract_to
            dest.mkdir(parents=True, exist_ok=True)
            with ZipFile(zip_path, "r") as zf:
                zf.extractall(path=dest)

        for repo_id, filename, extract_to, marker_dir in standard:
            marker_path = assets_root / marker_dir
            if marker_path.exists() and any(marker_path.iterdir()):
                continue
            try:
                _extract(repo_id, filename, extract_to)
                LOGGER.info("downloaded robocasa assets %s from %s", filename, repo_id)
            except Exception as exc:  # pragma: no cover - network/entitlement.
                LOGGER.warning(
                    "failed to download robocasa assets %s: %s", filename, exc
                )
        for repo_id, filename, extract_to, marker_dir in lightwheel:
            marker_path = assets_root / marker_dir
            done_marker = marker_path / ".npa_lightwheel_done"
            if done_marker.exists():
                continue
            try:
                _extract(repo_id, filename, extract_to)
                marker_path.mkdir(parents=True, exist_ok=True)
                done_marker.write_text("done\n")
                LOGGER.info("downloaded robocasa assets %s from %s", filename, repo_id)
            except Exception as exc:  # pragma: no cover - network/entitlement.
                LOGGER.warning(
                    "failed to download robocasa assets %s: %s", filename, exc
                )
    except Exception as exc:  # pragma: no cover - client without the stack.
        LOGGER.warning("robocasa asset download unavailable: %s", exc)


def _make_env(env_id: str, *, download_assets: bool = True) -> Any:
    """Create a headless EGL RoboCasa env, downloading assets when requested.

    The upstream gym wrapper defaults ``split="test"``, which the pinned
    ``create_env`` rejects; pass ``split="all"`` so real rollouts can run.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if download_assets:
        _download_assets()
    # Import robocasa AFTER assets are downloaded so OBJ_CATEGORIES are
    # populated with real object paths. This also registers the gymnasium
    # environments.
    _import_robocasa()
    gym = _import_gymnasium()
    try:
        return gym.make(
            env_id,
            split="all",
            obj_registries=ROBOCASA_OBJECT_REGISTRIES,
        )
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError(f"failed to create RoboCasa env {env_id}: {exc}") from exc


def kitchen_task_registration(*, env_id: str = DEFAULT_ENV_ID) -> dict[str, Any]:
    """Verify Gymnasium task registration for a RoboCasa env id."""
    gym = _import_gymnasium()
    if env_id not in gym.envs.registry:
        raise RoboCasaError(f"RoboCasa env id not registered: {env_id}")
    spec = gym.envs.registry[env_id]
    robocasa_envs = sorted(
        env for env in gym.envs.registry.keys() if env.startswith("robocasa/")
    )
    return {
        "env_id": env_id,
        "entry_point": str(spec.entry_point),
        "registered_env_count": len(robocasa_envs),
        "sample_registered_envs": robocasa_envs[:10],
    }


def kitchen_asset_availability() -> dict[str, Any]:
    """Verify the kitchen assets root exists and is populated."""
    assets_root = _assets_root()
    if not assets_root.exists():
        raise RoboCasaError(f"RoboCasa assets root does not exist: {assets_root}")
    subdirs = sorted(p.name for p in assets_root.iterdir() if p.is_dir())
    return {
        "assets_root": str(assets_root),
        "assets_root_exists": True,
        "subdirs": subdirs,
    }


def kitchen_egl_env_reset(
    *,
    env_id: str = DEFAULT_ENV_ID,
    seed: int | None = None,
    download_assets: bool = True,
) -> dict[str, Any]:
    """Create a headless EGL RoboCasa env and reset it."""
    env = _make_env(env_id, download_assets=download_assets)
    try:
        obs, info = env.reset(seed=seed)
        return {
            "env_id": env_id,
            "reset_ok": True,
            "observation_keys": sorted(obs.keys()) if isinstance(obs, dict) else [],
            "info_keys": sorted(info.keys()) if isinstance(info, dict) else [],
            "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        }
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError(f"failed to reset RoboCasa env {env_id}: {exc}") from exc
    finally:
        try:
            env.close()
        except Exception as exc:  # pragma: no cover - best effort.
            LOGGER.debug("env close failed: %s", exc)


def kitchen_random_rollout(
    *,
    env_id: str = DEFAULT_ENV_ID,
    iterations: int = 1,
    seed: int | None = None,
    output_dir: Path | None = None,
    download_assets: bool = True,
) -> dict[str, Any]:
    """Run a real random rollout and write a video artifact."""
    env = _make_env(env_id, download_assets=download_assets)
    video_path: Path | None = None
    try:
        obs, _ = env.reset(seed=seed)
        frames: list[Any] = []
        for _ in range(iterations):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            try:
                frames.append(env.render())
            except Exception as exc:  # pragma: no cover - render may be unavailable.
                LOGGER.debug("render unavailable: %s", exc)
            if terminated or truncated:
                break
        result: dict[str, Any] = {
            "env_id": env_id,
            "rollout_ok": True,
            "iterations": iterations,
            "final_reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "observation_keys": sorted(obs.keys()) if isinstance(obs, dict) else [],
        }
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = _write_video(frames, output_dir / "rollout.mp4")
            if video_path is not None:
                result["video_exists"] = True
                result["video_bytes"] = video_path.stat().st_size
                result["video_sha256"] = _sha256_file(video_path)
        return result
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError(f"failed to run RoboCasa rollout {env_id}: {exc}") from exc
    finally:
        try:
            env.close()
        except Exception as exc:  # pragma: no cover - best effort.
            LOGGER.debug("env close failed: %s", exc)


def kitchen_trajectory_export(
    *,
    env_id: str = DEFAULT_ENV_ID,
    iterations: int = 1,
    num_envs: int = 1,
    seed: int | None = None,
    output_dir: Path | None = None,
    download_assets: bool = True,
) -> dict[str, Any]:
    """Run real RoboCasa rollouts and export trajectories for LeRobotDataset.

    Writes one ``episode_NNNN/`` directory per rollout, each containing the
    numpy arrays the ``npa adapter convert`` adapter consumes:

      obs_workspace.npy  (T, H, W, 3) uint8   workspace camera
      obs_wrist.npy      (T, H, W, 3) uint8   wrist camera
      state.npy          (T, n_joints) float32
      actions.npy        (T, n_actions) float32

    plus a per-episode ``rollout.mp4`` and run-level ``metadata.json`` /
    ``metrics.json``. This is the real trajectory export seam between RoboCasa
    simulation and LeRobotDataset policy training.
    """
    try:
        env_ids = _parse_env_ids(env_id)
        episodes = _collect_trajectory_episodes(
            env_ids=env_ids,
            iterations=iterations,
            num_envs=num_envs,
            seed=seed,
            output_dir=output_dir,
            download_assets=download_assets,
        )
        result = _trajectory_export_result(env_id, env_ids, iterations, episodes)
        if output_dir is not None:
            _write_run_metadata(output_dir, env_id, episodes)
            result["output_dir"] = str(output_dir)
        return result
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError(
            f"failed to run RoboCasa trajectory export {env_id}: {exc}"
        ) from exc


def _collect_trajectory_episodes(
    *,
    env_ids: list[str],
    iterations: int,
    num_envs: int,
    seed: int | None,
    output_dir: Path | None,
    download_assets: bool,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    env = None
    try:
        for episode_index in range(num_envs):
            if env is not None:
                env.close()
            episode_env_id = env_ids[episode_index % len(env_ids)]
            env = _make_env(
                episode_env_id,
                download_assets=download_assets and episode_index == 0,
            )
            episode_seed = seed + episode_index if seed is not None else episode_index
            arrays, outcome, video_frames = _collect_trajectory_episode(
                env, iterations=iterations, seed=episode_seed
            )
            if output_dir is not None:
                episode_dir = output_dir / f"episode_{episode_index:04d}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                _write_trajectory_artifacts(episode_dir, arrays, video_frames)
            episodes.append(
                _trajectory_episode_identity(episode_index, episode_env_id, outcome)
            )
        return episodes
    finally:
        try:
            if env is not None:
                env.close()
        except Exception as exc:  # pragma: no cover - best effort.
            LOGGER.debug("env close failed: %s", exc)


def _trajectory_episode_identity(
    episode_index: int, env_id: str, outcome: dict[str, Any]
) -> dict[str, Any]:
    return {
        "episode_index": episode_index,
        "env_id": env_id,
        "task": env_id.removeprefix("robocasa/"),
        "embodiment": ROBOCASA_EMBODIMENT,
        **outcome,
    }


def _trajectory_export_result(
    env_id: str,
    env_ids: list[str],
    iterations: int,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "npa.robocasa.trajectory_export.v1",
        "env_id": env_id,
        "env_ids": env_ids,
        "embodiment": ROBOCASA_EMBODIMENT,
        "trajectory_export_ok": True,
        "temporal_alignment": "observation_before_action",
        "policy": "random_action_baseline",
        "num_episodes": len(episodes),
        "iterations": iterations,
        "successful_episodes": sum(int(episode["success"]) for episode in episodes),
        "episodes": episodes,
    }


def _collect_trajectory_episode(
    env: Any, *, iterations: int, seed: int
) -> tuple[dict[str, list[Any]], dict[str, Any], list[np.ndarray]]:
    """Collect causally aligned observation/action rows from one random rollout."""
    _seed_action_space(env.action_space, seed)
    observation, _ = env.reset(seed=seed)
    arrays: dict[str, list[Any]] = {
        "workspace": [],
        "wrist": [],
        "state": [],
        "actions": [],
    }
    outcome = _empty_episode_outcome()
    for _ in range(iterations):
        action = env.action_space.sample()
        _append_observation_action(arrays, observation, action)
        observation, reward, terminated, truncated, info = env.step(action)
        _update_episode_outcome(outcome, env, reward, terminated, truncated, info)
        if terminated or truncated:
            break
    terminal_frame = _obs_image(observation, "video.robot0_agentview_left")
    _validate_trajectory_arrays(arrays)
    video_frames = [*arrays["workspace"], terminal_frame]
    return (
        arrays,
        _trajectory_episode_record(arrays, outcome, terminal_frame, seed=seed),
        video_frames,
    )


def _append_observation_action(
    arrays: dict[str, list[Any]], observation: dict[str, Any], action: Any
) -> None:
    arrays["workspace"].append(_obs_image(observation, "video.robot0_agentview_left"))
    arrays["wrist"].append(_obs_image(observation, "video.robot0_eye_in_hand"))
    arrays["state"].append(_validated_finite("robot state", _obs_state(observation)))
    arrays["actions"].append(_validated_action(action))


def _empty_episode_outcome() -> dict[str, Any]:
    return {
        "reward_sum": 0.0,
        "final_reward": 0.0,
        "max_reward": float("-inf"),
        "success": False,
        "success_sources": set(),
        "terminated": False,
        "truncated": False,
    }


def _update_episode_outcome(
    outcome: dict[str, Any],
    env: Any,
    reward: Any,
    terminated: Any,
    truncated: Any,
    info: Any,
) -> None:
    reward_value = float(reward)
    if not np.isfinite(reward_value):
        raise RoboCasaError("RoboCasa reward contains a non-finite value")
    success, sources = _native_task_success(env, info, reward_value)
    outcome["reward_sum"] += reward_value
    outcome["final_reward"] = reward_value
    outcome["max_reward"] = max(outcome["max_reward"], reward_value)
    outcome["success"] = bool(outcome["success"] or success)
    outcome["success_sources"].update(sources)
    outcome["terminated"] = bool(terminated)
    outcome["truncated"] = bool(truncated)


def _trajectory_episode_record(
    arrays: dict[str, list[Any]],
    outcome: dict[str, Any],
    terminal_frame: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    workspace = arrays["workspace"]
    return {
        "seed": seed,
        "length": len(arrays["actions"]),
        "reward_sum": outcome["reward_sum"],
        "final_reward": outcome["final_reward"],
        "max_reward": outcome["max_reward"],
        "success": outcome["success"],
        "success_sources": sorted(outcome["success_sources"]),
        "terminated": outcome["terminated"],
        "truncated": outcome["truncated"],
        "state_keys": list(ROBOCASA_STATE_KEYS),
        "state_dim": int(np.asarray(arrays["state"][0]).size),
        "initial_workspace_sha256": _sha256_array(workspace[0]),
        "terminal_workspace_sha256": _sha256_array(terminal_frame),
        "video_frames": len(workspace) + 1,
    }


def _write_trajectory_artifacts(
    episode_dir: Path,
    arrays: dict[str, list[Any]],
    video_frames: list[np.ndarray],
) -> None:
    np.save(episode_dir / "obs_workspace.npy", np.stack(arrays["workspace"]))
    np.save(episode_dir / "obs_wrist.npy", np.stack(arrays["wrist"]))
    np.save(episode_dir / "state.npy", np.stack(arrays["state"]))
    np.save(episode_dir / "actions.npy", np.stack(arrays["actions"]))
    _write_required_video(video_frames, episode_dir / "rollout.mp4")


def _validate_trajectory_arrays(arrays: dict[str, list[Any]]) -> None:
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1 or not next(iter(lengths.values()), 0):
        raise RoboCasaError(f"trajectory arrays are not aligned: {lengths}")


def _flatten_action(action: Any) -> np.ndarray:
    """Flatten a RoboCasa action (OrderedDict or array) into a float32 vector."""
    if isinstance(action, dict):
        parts = []
        for key in sorted(action.keys()):
            value = action[key]
            if isinstance(value, dict):
                for sub_key in sorted(value.keys()):
                    parts.append(
                        np.asarray(value[sub_key], dtype=np.float32).reshape(-1)
                    )
            else:
                parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
        return np.concatenate(parts)
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _obs_image(obs: dict[str, Any], key: str) -> Any:
    """Return a uint8 (H, W, 3) image frame for a RoboCasa observation key."""
    frame = obs.get(key)
    if frame is None:
        raise RoboCasaError(f"RoboCasa observation missing image key: {key}")
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.uint8)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0].astype(np.uint8)
    raise RoboCasaError(f"RoboCasa image key {key!r} has unexpected shape {arr.shape}")


def _obs_state(obs: dict[str, Any]) -> np.ndarray:
    """Build a float32 robot-state vector from a RoboCasa observation."""
    parts: list[np.ndarray] = []
    missing: list[str] = []
    for key in ROBOCASA_STATE_KEYS:
        value = obs.get(key)
        if value is None:
            missing.append(key)
            continue
        parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
    if missing:
        raise RoboCasaError(f"RoboCasa observation missing robot state keys: {missing}")
    return np.concatenate(parts)


def _write_run_metadata(
    output_dir: Path, env_id: str, episodes: list[dict[str, Any]]
) -> None:
    """Write run-level metadata.json and metrics.json for the trajectory export."""
    metadata = {
        "env_id": env_id,
        "num_episodes": len(episodes),
        "episodes": episodes,
        "format": "lerobot-adapter-input",
        "schema": "npa.robocasa.trajectory_export.v1",
        "temporal_alignment": "observation_before_action",
        "policy": "random_action_baseline",
        "embodiment": ROBOCASA_EMBODIMENT,
        "robot_type": "panda_omron",
        "state_keys": list(ROBOCASA_STATE_KEYS),
        "state_dim": int(episodes[0]["state_dim"]) if episodes else 0,
        "task_env_ids": sorted({str(ep["env_id"]) for ep in episodes}),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    metrics = {
        "num_episodes": len(episodes),
        "total_steps": sum(int(ep["length"]) for ep in episodes),
        "mean_episode_length": (
            sum(int(ep["length"]) for ep in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True)
    )


def _parse_env_ids(value: str) -> list[str]:
    env_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not env_ids:
        raise RoboCasaError("at least one RoboCasa env id is required")
    if len(set(env_ids)) != len(env_ids):
        raise RoboCasaError("RoboCasa env ids must be unique")
    if any(not item.startswith("robocasa/") for item in env_ids):
        raise RoboCasaError("all RoboCasa env ids must start with 'robocasa/'")
    return env_ids


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _download_s3_tree(uri: str, destination: Path) -> Path:
    if not uri.startswith("s3://"):
        raise RoboCasaError(
            "policy evaluation requires an exact s3:// checkpoint prefix"
        )
    import boto3

    bucket, prefix = uri[5:].split("/", 1)
    client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("NEBIUS_S3_ENDPOINT")
        or None,
    )
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = str(item["Key"])
            if key.endswith("/"):
                continue
            target = safe_s3_download_target(destination, key, prefix.rstrip("/") + "/")
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            count += 1
    if count == 0:
        raise RoboCasaError("checkpoint prefix contains no objects")
    return destination


def _resolve_pretrained_dir(root: Path) -> Path:
    def is_loadable(candidate: Path) -> bool:
        return (candidate / "config.json").is_file() and any(
            (candidate / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        )

    preferred = root / "checkpoints" / "last" / "pretrained_model"
    if is_loadable(preferred):
        return preferred

    nested = sorted(
        candidate
        for candidate in root.rglob("pretrained_model")
        if candidate != preferred and is_loadable(candidate)
    )
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        choices = [candidate.relative_to(root).as_posix() for candidate in nested]
        raise RoboCasaError(
            "exact checkpoint prefix contains multiple loadable pretrained_model "
            f"directories without checkpoints/last: {choices}"
        )
    if is_loadable(root):
        return root
    raise RoboCasaError("exact checkpoint contains no loadable pretrained_model")


def _checkpoint_identity(checkpoint_root: Path) -> tuple[Path, str, str]:
    """Resolve and hash the exact loadable policy separately from run artifacts."""
    pretrained = _resolve_pretrained_dir(checkpoint_root)
    return pretrained, _sha256_tree(pretrained), _sha256_tree(checkpoint_root)


def _policy_observation(obs: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    def image_tensor(key: str) -> Any:
        array = _obs_image(obs, key)
        return (
            torch.from_numpy(array.copy())
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(device)
        )

    return {
        "observation.images.workspace": image_tensor("video.robot0_agentview_left"),
        "observation.images.wrist": image_tensor("video.robot0_eye_in_hand"),
        "observation.state": torch.from_numpy(_obs_state(obs).copy())
        .float()
        .unsqueeze(0)
        .to(device),
    }


def _unflatten_action(space: Any, values: np.ndarray) -> Any:
    if hasattr(space, "spaces"):
        offset = 0
        result: dict[str, Any] = {}
        for key in sorted(space.spaces):
            child = space.spaces[key]
            size = int(np.prod(child.shape))
            result[key] = values[offset : offset + size].reshape(child.shape)
            offset += size
        if offset != len(values):
            raise RoboCasaError(
                f"policy action dimension {len(values)} does not match RoboCasa action space {offset}"
            )
        return result
    expected = int(np.prod(space.shape))
    if expected != len(values):
        raise RoboCasaError(
            f"policy action dimension {len(values)} does not match RoboCasa action space {expected}"
        )
    return values.reshape(space.shape)


def _seed_action_space(space: Any, seed: int) -> None:
    seed_method = getattr(space, "seed", None)
    if callable(seed_method):
        seed_method(seed)


def _validated_finite(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise RoboCasaError(f"RoboCasa {name} contains non-finite values")
    return array


def _validated_action(action: Any, space: Any | None = None) -> np.ndarray:
    flat = _validated_finite("action", _flatten_action(action))
    contains = getattr(space, "contains", None)
    if callable(contains) and not bool(contains(action)):
        raise RoboCasaError("policy action is outside the RoboCasa action space")
    return flat


def _action_bounds(space: Any) -> tuple[np.ndarray, np.ndarray] | None:
    children = getattr(space, "spaces", None)
    if children is not None:
        child_bounds = [_action_bounds(children[key]) for key in sorted(children)]
        if any(bounds is None for bounds in child_bounds):
            return None
        lows = [bounds[0] for bounds in child_bounds if bounds is not None]
        highs = [bounds[1] for bounds in child_bounds if bounds is not None]
        return np.concatenate(lows), np.concatenate(highs)
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is None or high is None:
        return None
    return np.asarray(low).reshape(-1), np.asarray(high).reshape(-1)


def _prepare_eval_action(action: Any, space: Any) -> _PreparedAction:
    raw_flat = _validated_action(action)
    contains = getattr(space, "contains", None)
    if not callable(contains) or bool(contains(action)):
        return _PreparedAction(action, raw_flat, raw_flat, 0.0)
    bounds = _action_bounds(space)
    if bounds is None or any(len(bound) != len(raw_flat) for bound in bounds):
        _validated_action(action, space)
        raise AssertionError("unreachable")
    applied_flat = np.clip(raw_flat, bounds[0], bounds[1]).astype(np.float32)
    violation = float(np.max(np.abs(raw_flat - applied_flat)))
    prepared = _unflatten_action(space, applied_flat)
    if not bool(contains(prepared)):
        raise RoboCasaError("policy action is outside the RoboCasa action space")
    return _PreparedAction(prepared, raw_flat, applied_flat, violation)


def _native_task_success(env: Any, info: Any, reward: float) -> tuple[bool, list[str]]:
    signals: dict[str, bool] = {}
    if isinstance(info, dict):
        for key in ("success", "is_success", "goal_reached"):
            if key in info:
                signals[f"info.{key}"] = bool(info[key])
    unwrapped = getattr(env, "unwrapped", env)
    checker = getattr(unwrapped, "_check_success", None)
    if callable(checker):
        signals["environment._check_success"] = bool(checker())
    if reward in {0.0, 1.0}:
        signals["binary_reward"] = reward == 1.0
    if not signals:
        raise RoboCasaError("RoboCasa native task-success signal is unavailable")
    if len(set(signals.values())) != 1:
        raise RoboCasaError(f"RoboCasa native task-success signals disagree: {signals}")
    return next(iter(signals.values())), sorted(signals)


def _sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_act_policy(pretrained: Path) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(pretrained))
    policy.eval()
    device = next(policy.parameters()).device
    config = PreTrainedConfig.from_pretrained(str(pretrained))
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(pretrained)
    )
    return policy, device, preprocessor, postprocessor, torch


def _act_action_selector(
    runtime: tuple[Any, Any, Any, Any, Any],
) -> Callable[[Any, dict[str, Any]], Any]:
    policy, device, preprocessor, postprocessor, torch = runtime

    def select(_env: Any, observation: dict[str, Any]) -> Any:
        model_observation = preprocessor(_policy_observation(observation, device))
        with torch.inference_mode():
            action = postprocessor(policy.select_action(model_observation))
        flat = np.asarray(action.squeeze(0).detach().cpu(), dtype=np.float32)
        return _unflatten_action(_env.action_space, flat)

    return select


def _random_action_selector(env: Any, _observation: dict[str, Any]) -> Any:
    return env.action_space.sample()


def _rollout_eval_episode(
    env: Any,
    *,
    iterations: int,
    selector: Callable[[Any, dict[str, Any]], Any],
    observation: dict[str, Any],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    frames = [_obs_image(observation, "video.robot0_agentview_left")]
    initial_state = _validated_finite("robot state", _obs_state(observation))
    terminal_state = initial_state
    outcome = _empty_episode_outcome()
    action_trace = _ActionTrace()
    steps = 0
    for _ in range(iterations):
        prepared = _prepare_eval_action(selector(env, observation), env.action_space)
        action_trace.update(prepared)
        observation, reward, terminated, truncated, info = env.step(prepared.value)
        frames.append(_obs_image(observation, "video.robot0_agentview_left"))
        terminal_state = _validated_finite("robot state", _obs_state(observation))
        _update_episode_outcome(outcome, env, reward, terminated, truncated, info)
        steps += 1
        if terminated or truncated:
            break
    return frames, _eval_episode_record(
        frames,
        outcome,
        steps,
        action_trace,
        initial_state=initial_state,
        terminal_state=terminal_state,
    )


def _eval_episode_record(
    frames: list[np.ndarray],
    outcome: dict[str, Any],
    steps: int,
    action_trace: _ActionTrace,
    *,
    initial_state: np.ndarray,
    terminal_state: np.ndarray,
) -> dict[str, Any]:
    return {
        "steps": steps,
        "reward_sum": outcome["reward_sum"],
        "max_reward": outcome["max_reward"],
        "success": outcome["success"],
        "success_sources": sorted(outcome["success_sources"]),
        "terminated": outcome["terminated"],
        "truncated": outcome["truncated"],
        "state_keys": list(ROBOCASA_STATE_KEYS),
        "state_dim": int(initial_state.size),
        "initial_state_sha256": _sha256_array(initial_state),
        "terminal_state_sha256": _sha256_array(terminal_state),
        "initial_workspace_sha256": _sha256_array(frames[0]),
        "terminal_workspace_sha256": _sha256_array(frames[-1]),
        "action_sha256": action_trace.applied_digest.hexdigest(),
        "raw_action_sha256": action_trace.raw_digest.hexdigest(),
        "action_count": steps,
        "action_clipping_applied": action_trace.out_of_bounds_steps > 0,
        "action_out_of_bounds_steps": action_trace.out_of_bounds_steps,
        "max_action_bound_violation": action_trace.max_bound_violation,
    }


def _run_eval_episode(
    task_id: str,
    *,
    seed: int,
    iterations: int,
    selector: Callable[[Any, dict[str, Any]], Any],
    video_path: Path,
    download_assets: bool,
) -> dict[str, Any]:
    env = _make_env(task_id, download_assets=download_assets)
    try:
        _seed_action_space(env.action_space, seed)
        observation, _ = env.reset(seed=seed)
        frames, result = _rollout_eval_episode(
            env, iterations=iterations, selector=selector, observation=observation
        )
        video = _write_required_video(frames, video_path)
        result.update(
            {
                "video_sha256": _sha256_file(video),
                "video_bytes": video.stat().st_size,
                "video_frames": len(frames),
            }
        )
        return result
    finally:
        try:
            env.close()
        except Exception as exc:  # pragma: no cover - best effort.
            LOGGER.debug("env close failed: %s", exc)


def _write_eval_manifest(
    output_dir: Path,
    *,
    train_ids: list[str],
    heldout_ids: list[str],
    num_envs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], str]:
    episodes = [
        {
            "episode_index": index,
            "env_id": heldout_ids[index % len(heldout_ids)],
            "seed": seed + index,
        }
        for index in range(num_envs)
    ]
    manifest = {
        "schema": "npa.robocasa.eval_manifest.v1",
        "train_env_ids": train_ids,
        "heldout_env_ids": heldout_ids,
        "episodes": episodes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "eval_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return episodes, _sha256_file(path)


def _paired_outcome_counts(pairs: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"policy_wins": 0, "baseline_wins": 0, "ties": 0}
    for pair in pairs:
        policy_success = bool(pair["policy"]["success"])
        baseline_success = bool(pair["random_baseline"]["success"])
        if policy_success and not baseline_success:
            counts["policy_wins"] += 1
        elif baseline_success and not policy_success:
            counts["baseline_wins"] += 1
        else:
            counts["ties"] += 1
    return counts


def kitchen_policy_eval(
    *,
    checkpoint_uri: str,
    train_env_ids: str,
    heldout_env_ids: str,
    iterations: int,
    num_envs: int,
    seed: int | None,
    output_dir: Path,
    download_assets: bool = True,
) -> dict[str, Any]:
    """Evaluate ACT and random actions on matched, disjoint RoboCasa tasks."""
    train_ids = _parse_env_ids(train_env_ids)
    heldout_ids = _parse_env_ids(heldout_env_ids)
    overlap = sorted(set(train_ids) & set(heldout_ids))
    if overlap:
        raise RoboCasaError(f"train/held-out RoboCasa task overlap: {overlap}")

    base_seed = seed if seed is not None else 42
    episode_manifest, manifest_sha256 = _write_eval_manifest(
        output_dir,
        train_ids=train_ids,
        heldout_ids=heldout_ids,
        num_envs=num_envs,
        seed=base_seed,
    )
    execution = _evaluate_policy_checkpoint(
        checkpoint_uri,
        episode_manifest=episode_manifest,
        iterations=iterations,
        output_dir=output_dir,
        download_assets=download_assets,
    )
    result = _policy_eval_result(
        checkpoint_uri=checkpoint_uri,
        base_seed=base_seed,
        train_ids=train_ids,
        heldout_ids=heldout_ids,
        manifest_sha256=manifest_sha256,
        execution=execution,
    )
    _write_policy_eval_outputs(output_dir, result)
    return result


def _evaluate_policy_checkpoint(
    checkpoint_uri: str,
    *,
    episode_manifest: list[dict[str, Any]],
    iterations: int,
    output_dir: Path,
    download_assets: bool,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="robocasa-checkpoint-") as tmp:
        checkpoint_root = _download_s3_tree(checkpoint_uri, Path(tmp))
        pretrained, checkpoint_sha256, artifact_tree_sha256 = _checkpoint_identity(
            checkpoint_root
        )
        checkpoint_selection = pretrained.relative_to(checkpoint_root).as_posix()
        runtime = _load_act_policy(pretrained)
        policy, *_ = runtime
        selector = _act_action_selector(runtime)
        pairs = _run_matched_eval_pairs(
            episode_manifest,
            policy=policy,
            selector=selector,
            iterations=iterations,
            output_dir=output_dir,
            download_assets=download_assets,
        )
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_selection": checkpoint_selection,
        "artifact_tree_sha256": artifact_tree_sha256,
        "pairs": pairs,
    }


def _policy_eval_split_proof(
    train_ids: list[str], heldout_ids: list[str], manifest_sha256: str
) -> dict[str, Any]:
    task_sets_disjoint = set(train_ids).isdisjoint(heldout_ids)
    return {
        "train_env_ids": train_ids,
        "heldout_env_ids": heldout_ids,
        "configured_task_sets_disjoint": task_sets_disjoint,
        "basis": "caller-declared task ids plus held-out evaluation manifest",
        "checkpoint_training_tasks_verified": False,
        "declared_train_task_set_sha256": hashlib.sha256(
            json.dumps(sorted(train_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "heldout_task_set_sha256": hashlib.sha256(
            json.dumps(sorted(heldout_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "heldout_episode_manifest_sha256": manifest_sha256,
    }


def _policy_eval_result(
    *,
    checkpoint_uri: str,
    base_seed: int,
    train_ids: list[str],
    heldout_ids: list[str],
    manifest_sha256: str,
    execution: dict[str, Any],
) -> dict[str, Any]:
    pairs = execution["pairs"]
    episodes = [pair["policy"] for pair in pairs]
    baselines = [pair["random_baseline"] for pair in pairs]
    policy_success_rate = _success_rate(episodes)
    baseline_success_rate = _success_rate(baselines)
    return {
        "schema": "npa.robocasa.policy_eval.v1",
        "embodiment": ROBOCASA_EMBODIMENT,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": execution["checkpoint_sha256"],
        "checkpoint_selection": execution["checkpoint_selection"],
        "training_artifact_tree_sha256": execution["artifact_tree_sha256"],
        "checkpoint_loadable": True,
        "split_proof": _policy_eval_split_proof(
            train_ids, heldout_ids, manifest_sha256
        ),
        "num_episodes": len(episodes),
        "base_seed": base_seed,
        "success_rate": policy_success_rate,
        "baseline_success_rate": baseline_success_rate,
        "success_rate_delta": policy_success_rate - baseline_success_rate,
        "paired_outcomes": _paired_outcome_counts(pairs),
        "mean_reward": _mean_reward(episodes),
        "baseline_mean_reward": _mean_reward(baselines),
        "episodes": episodes,
        "baseline_episodes": baselines,
        "paired_episodes": pairs,
    }


def _write_policy_eval_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "eval.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "success_rate": result["success_rate"],
                "baseline_success_rate": result["baseline_success_rate"],
                "success_rate_delta": result["success_rate_delta"],
                "mean_reward": result["mean_reward"],
                "baseline_mean_reward": result["baseline_mean_reward"],
                **result["paired_outcomes"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_matched_eval_pairs(
    manifest: list[dict[str, Any]],
    *,
    policy: Any,
    selector: Callable[[Any, dict[str, Any]], Any],
    iterations: int,
    output_dir: Path,
    download_assets: bool,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for item in manifest:
        policy.reset()
        episode_dir = output_dir / f"episode_{item['episode_index']:04d}"
        policy_result = _run_eval_episode(
            item["env_id"],
            seed=item["seed"],
            iterations=iterations,
            selector=selector,
            video_path=episode_dir / "rollout.mp4",
            download_assets=download_assets and not pairs,
        )
        baseline = _run_eval_episode(
            item["env_id"],
            seed=item["seed"],
            iterations=iterations,
            selector=_random_action_selector,
            video_path=episode_dir / "random_baseline.mp4",
            download_assets=False,
        )
        _require_matched_initial_state(policy_result, baseline)
        identity = {key: item[key] for key in ("episode_index", "env_id", "seed")}
        policy_result.update(identity)
        baseline.update(identity)
        pairs.append({**identity, "policy": policy_result, "random_baseline": baseline})
    return pairs


def _require_matched_initial_state(
    policy_result: dict[str, Any], baseline_result: dict[str, Any]
) -> None:
    if (
        policy_result["initial_workspace_sha256"]
        != baseline_result["initial_workspace_sha256"]
    ):
        raise RoboCasaError(
            "policy and random baseline initial workspace frames do not match"
        )
    if policy_result["initial_state_sha256"] != baseline_result["initial_state_sha256"]:
        raise RoboCasaError(
            "policy and random baseline initial robot states do not match"
        )


def _success_rate(episodes: list[dict[str, Any]]) -> float:
    return sum(int(episode["success"]) for episode in episodes) / len(episodes)


def _mean_reward(episodes: list[dict[str, Any]]) -> float:
    return sum(float(episode["reward_sum"]) for episode in episodes) / len(episodes)


def _write_required_video(frames: list[Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = _write_video(frames, path)
    if written is None or not written.is_file() or written.stat().st_size <= 0:
        raise RoboCasaError(f"RoboCasa video was not written: {path}")
    return written


def _write_video(frames: list[Any], path: Path) -> Path | None:
    """Write frames to an MP4 using imageio's ffmpeg backend when available."""
    if not frames:
        return None
    try:
        import imageio

        imageio.mimsave(path, frames, fps=20)
        return path
    except Exception:  # pragma: no cover - ffmpeg backend may be absent.
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_capability(
    request: RoboCasaRunRequest,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Dispatch a RoboCasa capability request to the real implementation."""
    if request.capability == "kitchen_task_registration":
        return kitchen_task_registration(env_id=request.env_id)
    if request.capability == "kitchen_asset_availability":
        return kitchen_asset_availability()
    if request.capability == "kitchen_egl_env_reset":
        return kitchen_egl_env_reset(
            env_id=request.env_id,
            seed=request.seed,
            download_assets=request.download_assets,
        )
    if request.capability == "kitchen_random_rollout":
        return kitchen_random_rollout(
            env_id=request.env_id,
            iterations=request.iterations,
            seed=request.seed,
            output_dir=output_dir,
            download_assets=request.download_assets,
        )
    if request.capability == "kitchen_trajectory_export":
        return kitchen_trajectory_export(
            env_id=request.env_id,
            iterations=request.iterations,
            num_envs=request.num_envs,
            seed=request.seed,
            output_dir=output_dir,
            download_assets=request.download_assets,
        )
    if request.capability == "kitchen_policy_eval":
        if output_dir is None:
            raise RoboCasaError("policy evaluation requires an output directory")
        return kitchen_policy_eval(
            checkpoint_uri=request.checkpoint_uri,
            train_env_ids=request.train_env_ids,
            heldout_env_ids=request.heldout_env_ids,
            iterations=request.iterations,
            num_envs=request.num_envs,
            seed=request.seed,
            output_dir=output_dir,
            download_assets=request.download_assets,
        )
    raise RoboCasaError(f"unsupported robocasa capability: {request.capability}")


def run_capability_with_output(
    request: RoboCasaRunRequest,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a capability and persist/upload its output truthfully.

    Creates a temporary output directory when none is provided, runs the
    capability, and uploads any produced artifacts to ``request.output_uri``
    when set. Returns the capability result dict.

    This is the single entrypoint used by both the FastAPI service and the SDK
    local path so that local execution persists and uploads output exactly like
    a service run, instead of silently dropping artifacts.
    """
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="robocasa_") as tmp:
            return run_capability_with_output(request, output_dir=Path(tmp))
    result = run_capability(request, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _execution_provenance(request, output_dir, result)
    result["execution_provenance"] = provenance
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    if request.output_uri:
        upload_output(output_dir, request.output_uri, result)
    return result


def _provenance_environment_ids(
    request: RoboCasaRunRequest, result: dict[str, Any]
) -> list[str]:
    env_ids = result.get("env_ids")
    if not isinstance(env_ids, list):
        heldout = result.get("split_proof", {}).get("heldout_env_ids", [])
        env_ids = heldout if isinstance(heldout, list) and heldout else [request.env_id]
    return [str(item) for item in env_ids]


def _provenance_mp4_artifacts(output_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*.mp4"))
    ]


def _execution_provenance(
    request: RoboCasaRunRequest,
    output_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Describe how run artifacts were produced, without overstating validation."""
    rollout_capabilities = {
        "kitchen_random_rollout",
        "kitchen_trajectory_export",
        "kitchen_policy_eval",
    }
    source_identity, image_source_sha = _runtime_source_identity()
    mp4_artifacts = _provenance_mp4_artifacts(output_dir)
    return {
        "schema": "npa.robocasa.execution_provenance.v2",
        "generator": "robocasa",
        "simulator": "mujoco",
        "source_identity": source_identity,
        "image_source_sha": image_source_sha,
        "capability": request.capability,
        "environment_ids": _provenance_environment_ids(request, result),
        "execution_path": (
            "gymnasium.make(robocasa/*)->RoboCasa->MuJoCo->step/render"
            if request.capability in rollout_capabilities
            else "RoboCasa runtime capability probe"
        ),
        "capture_source": (
            "runtime_environment_observation_or_render"
            if request.capability in rollout_capabilities
            else "none"
        ),
        "stock_or_copied_fixture": False,
        "runtime_result_sha256": hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "recording_formats": {
            "mp4": bool(mp4_artifacts),
            "rrd": False,
            "mcap": False,
        },
        "mp4_artifacts": mp4_artifacts,
        "validation_scope": "runtime artifact provenance; no GPU validation claim",
    }


def upload_output(local_dir: Path, output_uri: str, result: dict[str, Any]) -> None:
    """Upload a capability's local output tree to S3 when the run produced one.

    Capabilities that write artifacts (rollouts, trajectory exports, policy
    evaluation) publish their output directory to ``output_uri`` so downstream
    workflow stages can read it from S3. Capabilities that only return a result
    dict (task registration, asset availability) have nothing to upload.
    """
    if not output_uri:
        return
    root = Path(local_dir)
    if not root.exists() or not any(root.iterdir()):
        return
    import boto3

    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get(
        "NEBIUS_S3_ENDPOINT", ""
    )
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
    )
    bucket, prefix = parse_s3_uri(output_uri)
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file():
            rel = file_path.relative_to(root)
            s3.upload_file(str(file_path), bucket, f"{prefix}/{rel}")
    result["output_uri"] = output_uri


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an s3:// URI into (bucket, prefix)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri}")
    rest = uri[len("s3://") :]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


__all__ = [
    "SUPPORTED_CAPABILITIES",
    "RoboCasaError",
    "compute_manifest_sha256",
    "kitchen_asset_availability",
    "kitchen_egl_env_reset",
    "kitchen_random_rollout",
    "kitchen_task_registration",
    "kitchen_trajectory_export",
    "kitchen_policy_eval",
    "make_run_id",
    "parse_s3_uri",
    "run_capability",
    "run_capability_with_output",
    "system_info",
    "upload_output",
]
