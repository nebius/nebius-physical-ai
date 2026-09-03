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
import sys
import tempfile

import numpy as np
from pathlib import Path
from typing import Any

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


class RoboCasaError(RuntimeError):
    """Raised when a RoboCasa capability operation fails."""


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


def system_info() -> RoboCasaSystemInfo:
    """Collect system and RoboCasa stack information."""
    info = RoboCasaSystemInfo(
        status="ok",
        python=platform.python_version(),
        platform=platform.platform(),
        robocasa_version=_package_version("robocasa"),
        robosuite_version=_package_version("robosuite"),
        mujoco_version=_package_version("mujoco"),
        gymnasium_version=_package_version("gymnasium"),
    )
    try:
        import torch

        info.cuda_available = bool(torch.cuda.is_available())
        info.cuda_device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
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
            ("robocasa/robocasa-assets", "generative_textures.zip", ".", "generative_textures"),
            ("robocasa/robocasa-assets", "fixtures.zip", ".", "fixtures/accessories"),
            ("robocasa/robocasa-assets", "objaverse.zip", "objects", "objects/objaverse"),
            ("robocasa/robocasa-assets", "aigen_objs.zip", "objects", "objects/aigen_objs"),
        ]
        # Lightwheel fixtures are one zip per fixture family, each extracting a
        # top-level folder (e.g. stoves/) that must land under fixtures/. They are
        # additive on top of baked directories, so track completion with a marker.
        lightwheel_fixtures = [
            "blenders", "cabinets", "coffee_machines", "dishwashers",
            "electric_kettles", "fridges", "handles", "hoods", "microwaves",
            "ovens", "sinks", "stand_mixers", "stoves", "stovetops",
            "toaster_ovens", "toasters", "windows",
        ]
        # Lightwheel objects are one zip per object family, each extracting a
        # top-level folder (e.g. stool/) that must land under objects/lightwheel/.
        lightwheel_objects = [
            "aluminum_foil", "basket", "blender_jug", "cheese_grater",
            "chicken_drumstick", "cinnamon", "colander", "cookie_dough_ball",
            "cream_cheese_stick", "digital_scale", "dish_brush", "dish_rack",
            "flour_bag", "flower_vase", "fruit_bowl", "glass_cup",
            "honey_bottle", "hotdog_bun", "ice_cube", "ice_cube_tray", "jar",
            "juice", "kebab_skewer", "kettle", "knife_block", "lemon_wedge",
            "lettuce", "marshmallow", "mayonnaise", "measuring_cup", "mug_tree",
            "mustard", "oil_and_vinegar_bottle", "oven_tray", "pancake",
            "paper_towel_holder", "paprika", "peeler", "pickle_slice",
            "pitcher", "pizza", "pizza_cutter", "placemat", "plant", "pot",
            "reamer", "salt_and_pepper_shaker", "sandwich_bread", "saucepan",
            "shrimp", "soap_dispenser", "spray", "stool", "strainer", "straw",
            "sugar_cube", "syrup_bottle", "tiered_basket", "tiered_shelf",
            "tomato_slice", "tongs", "tray", "tupperware", "turkey_slice",
            "turmeric", "utensil_rack", "utensil_set", "whisk", "wooden_spoon",
        ]
        lightwheel = [
            ("nvidia/PhysicalAI-Kitchen-Assets", f"fixtures_lightwheel/{name}.zip", "fixtures", f"fixtures/{name}")
            for name in lightwheel_fixtures
        ] + [
            ("nvidia/PhysicalAI-Kitchen-Assets", f"objects_lightwheel/{name}.zip", "objects/lightwheel", f"objects/lightwheel/{name}")
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
                LOGGER.warning("failed to download robocasa assets %s: %s", filename, exc)
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
                LOGGER.warning("failed to download robocasa assets %s: %s", filename, exc)
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
    subdirs = sorted(
        p.name for p in assets_root.iterdir() if p.is_dir()
    )
    return {
        "assets_root": str(assets_root),
        "assets_root_exists": True,
        "subdirs": subdirs,
    }


def kitchen_egl_env_reset(
    *, env_id: str = DEFAULT_ENV_ID, seed: int | None = None, download_assets: bool = True
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
    env_ids = _parse_env_ids(env_id)
    env = None
    try:
        episodes: list[dict[str, Any]] = []
        for ep in range(num_envs):
            episode_env_id = env_ids[ep % len(env_ids)]
            if env is not None:
                env.close()
            env = _make_env(
                episode_env_id,
                download_assets=download_assets and ep == 0,
            )
            obs, _ = env.reset(seed=(seed + ep) if seed is not None else None)
            workspace_frames: list[Any] = []
            wrist_frames: list[Any] = []
            states: list[Any] = []
            actions: list[Any] = []
            reward = 0.0
            terminated = False
            truncated = False
            for _ in range(iterations):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _info = env.step(action)
                workspace_frames.append(_obs_image(obs, "video.robot0_agentview_left"))
                wrist_frames.append(_obs_image(obs, "video.robot0_eye_in_hand"))
                states.append(_obs_state(obs))
                actions.append(_flatten_action(action))
                if terminated or truncated:
                    break
            if output_dir is not None:
                ep_dir = output_dir / f"episode_{ep:04d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                np.save(ep_dir / "obs_workspace.npy", np.stack(workspace_frames))
                np.save(ep_dir / "obs_wrist.npy", np.stack(wrist_frames))
                np.save(ep_dir / "state.npy", np.stack(states))
                np.save(ep_dir / "actions.npy", np.stack(actions))
                _write_video(workspace_frames, ep_dir / "rollout.mp4")
            episodes.append(
                {
                    "episode_index": ep,
                    "env_id": episode_env_id,
                    "task": episode_env_id.removeprefix("robocasa/"),
                    "embodiment": ROBOCASA_EMBODIMENT,
                    "length": len(actions),
                    "final_reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
        result: dict[str, Any] = {
            "env_id": env_id,
            "env_ids": env_ids,
            "embodiment": ROBOCASA_EMBODIMENT,
            "trajectory_export_ok": True,
            "num_episodes": len(episodes),
            "iterations": iterations,
            "episodes": episodes,
        }
        if output_dir is not None:
            _write_run_metadata(output_dir, env_id, episodes)
            result["output_dir"] = str(output_dir)
        return result
    except Exception as exc:  # pragma: no cover - depends on the container.
        raise RoboCasaError(
            f"failed to run RoboCasa trajectory export {env_id}: {exc}"
        ) from exc
    finally:
        try:
            if env is not None:
                env.close()
        except Exception as exc:  # pragma: no cover - best effort.
            LOGGER.debug("env close failed: %s", exc)


def _flatten_action(action: Any) -> np.ndarray:
    """Flatten a RoboCasa action (OrderedDict or array) into a float32 vector."""
    if isinstance(action, dict):
        parts = []
        for key in sorted(action.keys()):
            value = action[key]
            if isinstance(value, dict):
                for sub_key in sorted(value.keys()):
                    parts.append(np.asarray(value[sub_key], dtype=np.float32).reshape(-1))
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
    raise RoboCasaError(
        f"RoboCasa image key {key!r} has unexpected shape {arr.shape}"
    )


def _obs_state(obs: dict[str, Any]) -> np.ndarray:
    """Build a float32 robot-state vector from a RoboCasa observation."""
    parts: list[np.ndarray] = []
    for key in (
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
    ):
        value = obs.get(key)
        if value is not None:
            parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
    if not parts:
        raise RoboCasaError("RoboCasa observation has no robot state keys")
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
        "embodiment": ROBOCASA_EMBODIMENT,
        "robot_type": "panda_omron",
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
        raise RoboCasaError("policy evaluation requires an exact s3:// checkpoint prefix")
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
            target = destination / key.removeprefix(prefix.rstrip("/") + "/")
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            count += 1
    if count == 0:
        raise RoboCasaError("checkpoint prefix contains no objects")
    return destination


def _resolve_pretrained_dir(root: Path) -> Path:
    candidates = [root, *root.rglob("pretrained_model")]
    for candidate in candidates:
        if (candidate / "config.json").is_file() and any(
            (candidate / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        ):
            return candidate
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
    """Evaluate the exact trained ACT checkpoint on disjoint RoboCasa tasks."""
    train_ids = _parse_env_ids(train_env_ids)
    heldout_ids = _parse_env_ids(heldout_env_ids)
    overlap = sorted(set(train_ids) & set(heldout_ids))
    if overlap:
        raise RoboCasaError(f"train/held-out RoboCasa task overlap: {overlap}")

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    with tempfile.TemporaryDirectory(prefix="robocasa-checkpoint-") as tmp:
        checkpoint_root = _download_s3_tree(checkpoint_uri, Path(tmp))
        pretrained, checkpoint_sha256, artifact_tree_sha256 = _checkpoint_identity(
            checkpoint_root
        )
        policy = ACTPolicy.from_pretrained(str(pretrained))
        policy.eval()
        device = next(policy.parameters()).device
        cfg = PreTrainedConfig.from_pretrained(str(pretrained))
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=str(pretrained)
        )
        episodes: list[dict[str, Any]] = []
        for episode_index in range(num_envs):
            task_id = heldout_ids[episode_index % len(heldout_ids)]
            env = _make_env(task_id, download_assets=download_assets and episode_index == 0)
            frames: list[Any] = []
            try:
                obs, _ = env.reset(
                    seed=(seed + episode_index) if seed is not None else None
                )
                policy.reset()
                reward_sum = 0.0
                max_reward = float("-inf")
                success = False
                steps = 0
                for _ in range(iterations):
                    model_obs = preprocessor(_policy_observation(obs, device))
                    with torch.inference_mode():
                        action = postprocessor(policy.select_action(model_obs))
                    flat = np.asarray(action.squeeze(0).detach().cpu(), dtype=np.float32)
                    obs, reward, terminated, truncated, info = env.step(
                        _unflatten_action(env.action_space, flat)
                    )
                    frames.append(_obs_image(obs, "video.robot0_agentview_left"))
                    reward_sum += float(reward)
                    max_reward = max(max_reward, float(reward))
                    success = success or bool(info.get("success", False)) or float(reward) >= 1.0
                    steps += 1
                    if terminated or truncated:
                        break
                video = output_dir / f"episode_{episode_index:04d}" / "rollout.mp4"
                video.parent.mkdir(parents=True, exist_ok=True)
                _write_video(frames, video)
                episodes.append(
                    {
                        "episode_index": episode_index,
                        "env_id": task_id,
                        "seed": (seed + episode_index) if seed is not None else None,
                        "steps": steps,
                        "reward_sum": reward_sum,
                        "max_reward": max_reward,
                        "success": success,
                        "video_sha256": _sha256_file(video) if video.exists() else "",
                    }
                )
            finally:
                env.close()

    heldout_episode_manifest = [
        {
            "episode_index": int(ep["episode_index"]),
            "env_id": str(ep["env_id"]),
            "seed": ep["seed"],
        }
        for ep in episodes
    ]
    split_proof = {
        "train_env_ids": train_ids,
        "heldout_env_ids": heldout_ids,
        "task_sets_disjoint": True,
        "episode_sets_disjoint_by_task": True,
        "train_task_set_sha256": hashlib.sha256(
            json.dumps(sorted(train_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "heldout_task_set_sha256": hashlib.sha256(
            json.dumps(sorted(heldout_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "heldout_episode_manifest_sha256": hashlib.sha256(
            json.dumps(
                heldout_episode_manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    result = {
        "schema": "npa.robocasa.policy_eval.v1",
        "embodiment": ROBOCASA_EMBODIMENT,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": checkpoint_sha256,
        "training_artifact_tree_sha256": artifact_tree_sha256,
        "checkpoint_loadable": True,
        "split_proof": split_proof,
        "num_episodes": len(episodes),
        "success_rate": sum(int(ep["success"]) for ep in episodes) / len(episodes),
        "mean_reward": sum(float(ep["reward_sum"]) for ep in episodes) / len(episodes),
        "episodes": episodes,
    }
    (output_dir / "eval.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {"success_rate": result["success_rate"], "mean_reward": result["mean_reward"]},
            indent=2,
            sort_keys=True,
        )
    )
    return result


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
            env_id=request.env_id, seed=request.seed, download_assets=request.download_assets
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
    mp4_files = sorted(output_dir.rglob("*.mp4"))
    env_ids = result.get("env_ids")
    if not isinstance(env_ids, list):
        heldout = result.get("split_proof", {}).get("heldout_env_ids", [])
        env_ids = heldout if isinstance(heldout, list) and heldout else [request.env_id]
    return {
        "schema": "npa.robocasa.execution_provenance.v1",
        "generator": "robocasa",
        "simulator": "mujoco",
        "capability": request.capability,
        "environment_ids": [str(item) for item in env_ids],
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
            "mp4": bool(mp4_files),
            "rrd": False,
            "mcap": False,
        },
        "mp4_artifacts": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in mp4_files
        ],
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

    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT", "")
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
    rest = uri[len("s3://"):]
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
