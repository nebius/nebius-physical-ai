"""Bind the BYOF task to native Isaac, RSL-RL and a single loaded warehouse scene."""

from __future__ import annotations

import importlib
from importlib.machinery import PathFinder
from importlib.metadata import version
from pathlib import Path

from npa.workflows.navigation.artifacts import file_sha256


def task_adapter(recipe):
    """Import an explicitly selected, hash-bound operator task adapter.

    Args:
        recipe: Validated task recipe from the sealed operator bundle.
    Returns:
        Module implementing configure, reset and measure.
    Raises:
        ValueError: Source hash or required integration methods differ.
        ImportError: Operator adapter is not installed in the BYOF image.
    """
    if recipe.adapter_module == "npa.workflows.navigation.reference":
        if recipe.source_bundle_sha256 != source_bundle_digest():
            raise ValueError("installed reference source bundle SHA-256 mismatch")
    source = _module_source(recipe.adapter_module)
    if file_sha256(source) != recipe.adapter_sha256:
        raise ValueError("installed task adapter source SHA-256 mismatch")
    module = importlib.import_module(recipe.adapter_module)
    if Path(module.__file__ or "").resolve() != source.resolve():
        raise ValueError("imported task adapter differs from verified source")
    for name in ("configure", "reset", "measure", "probe_mode"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"task adapter requires callable {name}")
    if recipe.sensor_mode == "rgbd" and not callable(
        getattr(module, "visibility_paths", None)
    ):
        raise ValueError("RGB-D adapter requires visibility_paths(env)")
    return module


def source_bundle_digest() -> str:
    """Bind all built-in navigation source bytes separately from the native image.

    Args:
        None.
    Returns:
        Stable SHA-256 of relative module names and file digests.
    Raises:
        OSError: An installed source module cannot be read.
    """
    import hashlib
    import json

    root = Path(__file__).parent
    sources = {path.name: file_sha256(path) for path in sorted(root.glob("*.py"))}
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def _module_source(name):
    search_path = None
    components = name.split(".")
    for index in range(len(components)):
        spec = PathFinder.find_spec(".".join(components[: index + 1]), search_path)
        if spec is None:
            raise ImportError(
                "operator task adapter is not installed in the BYOF image"
            )
        if index < len(components) - 1:
            search_path = spec.submodule_search_locations
            if search_path is None:
                raise ImportError("task adapter parent is not a package")
    source = Path(spec.origin or "")
    if source.suffix != ".py" or not source.is_file():
        raise ValueError(
            "task adapter must have readable Python source in the pinned image"
        )
    return source


def build_runner(env, recipe, output: Path | None):
    """Use the registered task's native RSL-RL entrypoint without replacing it.

    Args:
        env: Native Isaac navigation environment.
        recipe: Validated training settings.
        output: Checkpoint/log directory or None for inference.
    Returns:
        RSL-RL wrapper, runner and resolved learner configuration.
    Raises:
        ImportError: Native runtime is unavailable.
        RuntimeError: Registered task or learner construction fails.
    """
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    config = load_cfg_from_registry(recipe.task, "rsl_rl_cfg_entry_point")
    config = handle_deprecated_rsl_rl_cfg(config, version("rsl-rl-lib"))
    config.seed = recipe.train_cases[0].seed
    config.max_iterations = recipe.iterations
    wrapped = RslRlVecEnvWrapper(env, clip_actions=config.clip_actions)
    settings = config.to_dict()
    runner = OnPolicyRunner(
        wrapped, settings, log_dir=str(output) if output else None, device="cuda:0"
    )
    return wrapped, runner, settings


def inspect_scene(env, recipe, scene_file: Path) -> dict:
    """Verify native environment identity, loaded scene and supported sensors.

    Args:
        env: Instantiated Isaac environment.
        recipe: Expected task, scene and sensor contract.
        scene_file: Hash-verified self-contained USDZ file.
    Returns:
        Native runtime and scene inventory evidence.
    Raises:
        ValueError: Environment, scene replication or sensor mode is unsupported.
    """
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
    from npa.workbench.isaac_lab.routing import validate_render_gpu_target
    import torch

    native = env.unwrapped
    _verify_sim_version()
    if not isinstance(native, (DirectRLEnv, ManagerBasedRLEnv)):
        raise ValueError("BYOF task must be a native Isaac Lab RL environment")
    if native.num_envs != recipe.num_envs or not str(native.device).startswith("cuda"):
        raise ValueError("native Isaac robot count or CUDA device differs from recipe")
    validate_render_gpu_target(torch.cuda.get_device_name(), what="Isaac navigation")
    _verify_scene_reference(native.sim.stage, recipe, scene_file)
    return {
        "isaaclab": version("isaaclab"),
        "isaacsim": version("isaacsim"),
        "rsl_rl": version("rsl-rl-lib"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "robot_population": native.num_envs,
        "scene_collision_meshes": _scene_collider_count(
            native.sim.stage, recipe.scene_prim
        ),
        "scene_instances": 1,
        "scene_sha256": recipe.scene_sha256,
        "scene_prim": recipe.scene_prim,
        "sensors": _inspect_sensors(native.scene.sensors, recipe),
    }


def _verify_sim_version():
    if not version("isaacsim").startswith("6.0.1"):
        raise ValueError("navigation adapter requires the Isaac Sim 6.0.1 runtime")


def _scene_collider_count(stage, root):
    from pxr import Sdf, Usd, UsdPhysics

    return sum(
        prim.GetPath().HasPrefix(Sdf.Path(root))
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
        for prim in stage.Traverse(Usd.TraverseInstanceProxies())
    )


def _verify_scene_reference(stage, recipe, scene_file):
    references = []
    for prim in stage.Traverse():
        authored = prim.GetMetadata("references")
        if authored is None:
            continue
        for item in authored.GetAddedOrExplicitItems():
            if (
                item.assetPath
                and Path(item.assetPath).resolve() == scene_file.resolve()
            ):
                references.append(str(prim.GetPath()))
    if references != [recipe.scene_prim]:
        raise ValueError(
            "scene must have exactly one direct USDZ reference at scene_prim"
        )
    if not stage.GetPrimAtPath(recipe.scene_prim).IsValid():
        raise ValueError("shared warehouse prim was not loaded")


def _inspect_sensors(sensors, recipe):
    from isaaclab.sensors import Camera
    from isaaclab.sensors.contact_sensor import BaseContactSensor
    from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster

    inventory = {}
    rays = cameras = 0
    for name, sensor in sensors.items():
        if isinstance(sensor, Camera) and recipe.sensor_mode == "rgbd":
            inventory[name] = {"type": "camera"}
            cameras += 1
            continue
        if isinstance(sensor, BaseContactSensor):
            inventory[name] = {"type": "contact"}
            continue
        if (
            not isinstance(sensor, BaseRayCaster)
            or "camera" in type(sensor).__name__.lower()
        ):
            raise ValueError(
                "camera or unknown sensor: perception isolation unsupported"
            )
        paths = list(sensor.cfg.mesh_prim_paths)
        if not paths or any(not p.startswith(recipe.scene_prim + "/") for p in paths):
            raise ValueError(
                "raycasts must target only static warehouse mesh descendants"
            )
        inventory[name] = {"type": "static_raycast", "mesh_prim_paths": paths}
        rays += 1
    if recipe.sensor_mode == "rgbd" and not cameras:
        raise ValueError("RGB-D mode requires task camera sensors feeding observations")
    if recipe.sensor_mode != "rgbd" and (
        recipe.sensor_mode == "static_raycast"
    ) != bool(rays):
        raise ValueError("configured sensor mode differs from native sensor inventory")
    return inventory


def parameters(runner):
    """Copy native policy parameters for finite update verification.

    Args:
        runner: Native RSL-RL runner.
    Returns:
        Detached flat CPU tensor.
    Raises:
        RuntimeError: Native policy parameters are unavailable.
    """
    import torch

    return torch.cat(
        [
            p.detach().flatten().cpu().clone()
            for p in runner.alg.get_policy().parameters()
        ]
    )


def validate_scene_package(scene_file: Path) -> None:
    """Require a loadable USDZ whose layers and asset dependencies stay in its package.

    Args:
        scene_file: Hash-verified scene package, before task configuration loads it.
    Returns:
        None.
    Raises:
        ValueError: USD cannot load the package or dependencies escape or are missing.
    """
    from pxr import Sdf, UsdUtils

    package = str(scene_file.resolve())
    layers, assets, missing = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(package))
    paths = [layer.realPath for layer in layers] + list(assets)
    if (
        not layers
        or missing
        or any(path != package and not path.startswith(package + "[") for path in paths)
    ):
        raise ValueError(
            "scene USDZ must be loadable and self-contained; external/missing dependencies rejected"
        )


def policy_state_digest(runner) -> str:
    """Hash policy parameters and buffers to detect inference-time state changes.

    Args:
        runner: Native RSL-RL runner exposing the actual policy state dictionary.
    Returns:
        SHA-256 over names, tensor layouts and exact state bytes.
    Raises:
        RuntimeError: Policy state cannot be read.
    """
    import hashlib
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(runner.alg.get_policy().state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
