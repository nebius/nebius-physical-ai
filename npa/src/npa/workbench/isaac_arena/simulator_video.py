"""Capture real Arena frames before automatic reset and retain their state binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .simulator_phases import phase_scope, record_readiness


_PLAY_SIMULATIONS = "/app/player/playSimulations"
_PHYSICS_CLOCK = "native_physx_step_events_since_capture_setup"
_STARTUP_RENDER_SETTINGS = {
    # Kit reads renderer availability while registering its render modes. These
    # settings must also be present in AppLauncher's ``--kit_args``; changing
    # them only after SimulationApp starts cannot switch the registered mode.
    "/persistent/rtx/modes/rt/enabled": True,
    "/persistent/rtx/modes/rt2/enabled": False,
    "/persistent/rtx/modes/pt/enabled": False,
}
_CAPTURE_RENDER_SETTINGS = {
    "/rtx/rendermode": "RaytracedLighting",
    # Legacy RTX maps TAA to its supported temporal anti-aliasing path. It
    # suppresses the severe single-frame grain observed with spatial-only FXAA
    # without relying on the unsupported DLAA selection, which Isaac Sim 6
    # silently remaps to TAA. The simulator is frozen while temporal history
    # settles, and the separate verifier still requires task-bound motion.
    "/rtx/post/aa/op": 1,
    "/rtx/post/dlss/execMode": 2,
    "/rtx-transient/dldenoiser/enabled": True,
    "/rtx-transient/dlssg/enabled": False,
    "/rtx/ecoMode/enabled": False,
    # Pin native lighting quality instead of inheriting the interactive
    # renderer's low sample defaults. Temporal settling alone left severe
    # grain in a completed episode and failed the unchanged motion gate.
    "/rtx/directLighting/sampledLighting/samplesPerPixel": 32,
    "/rtx/indirectDiffuse/fetchSampleCount": 32,
    "/rtx/reflections/sampledLighting/samplesPerPixel": 16,
}
_RENDER_SETTINGS = {**_STARTUP_RENDER_SETTINGS, **_CAPTURE_RENDER_SETTINGS}
_MINIMUM_SETTLING_RENDERS = 8


def legacy_rtx_kit_args() -> str:
    """Return ordered Kit startup arguments for the stable legacy RTX mode."""

    def encode(value: Any) -> str:
        return str(value).lower() if type(value) is bool else str(value)

    return " ".join(
        f"--{key}={encode(value)}" for key, value in _RENDER_SETTINGS.items()
    )


class _PhysicsStepClock:
    def __init__(self) -> None:
        import omni.physx

        self.steps = 0
        self.elapsed = 0.0
        self.invalid_delta = False
        self._callback = self._observe_step
        self._subscription = (
            omni.physx.get_physx_interface().subscribe_physics_step_events(
                self._callback
            )
        )
        if self._subscription is None:
            raise RuntimeError("Arena video could not subscribe to native PhysX steps")

    def _observe_step(self, delta: float) -> None:
        self.steps += 1
        try:
            duration = float(delta)
        except (TypeError, ValueError, OverflowError):
            self.invalid_delta = True
            return
        if not np.isfinite(duration) or duration < 0:
            self.invalid_delta = True
            return
        self.elapsed += duration
        if not np.isfinite(self.elapsed):
            self.invalid_delta = True

    def snapshot(self) -> dict[str, Any]:
        # Native callbacks may swallow raised exceptions. Persist invalid input
        # and fail synchronously at the next capture boundary instead.
        if self.invalid_delta:
            raise RuntimeError(
                "Arena video received invalid native PhysX step duration"
            )
        return {"native_physics_step": self.steps, "physics_time": self.elapsed}


def _kit_interfaces() -> tuple[Any, Any]:
    import carb.settings
    import omni.usd

    context = omni.usd.get_context()
    if context is None:
        raise RuntimeError("Arena video requires an active USD context")
    return carb.settings.get_settings(), context


def _physics_array(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise RuntimeError("Arena video received invalid direct physics state")
    return array.tolist()


def _direct_physics_state(env: Any) -> dict[str, Any]:
    # Read PhysX views directly: scene.data caches cannot detect a stray Kit step.
    getters = {
        "articulations": (
            "get_root_transforms",
            "get_root_velocities",
            "get_dof_positions",
            "get_dof_velocities",
        ),
        "rigid_objects": ("get_transforms", "get_velocities"),
        "rigid_object_collections": ("get_transforms", "get_velocities"),
    }
    if env.scene.deformable_objects:
        raise RuntimeError(
            "Arena video physics verification does not support deformables"
        )
    state = {}
    for category, methods in getters.items():
        for name, asset in getattr(env.scene, category).items():
            state[f"{category}/{name}"] = {
                method: _physics_array(getattr(asset.root_view, method)())
                for method in methods
            }
    if not state:
        raise RuntimeError("Arena video requires direct simulator physics state")
    return state


def _physics_snapshot(env: Any) -> dict[str, Any]:
    return {
        **env._npa_video_physics_clock.snapshot(),
        "physics_step": env.sim.get_physics_step_count(),
        "state": _direct_physics_state(env),
    }


def _state_hash(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot["state"], sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_physics_unchanged(before: dict, after: dict) -> None:
    if before != after:
        raise RuntimeError(
            "Arena video rendering advanced physics time, steps, or state"
        )


def _rendering_settings_readback(settings: Any) -> dict[str, Any]:
    actual = {key: settings.get(key) for key in _RENDER_SETTINGS}
    if any(
        type(actual[key]) is not type(expected) or actual[key] != expected
        for key, expected in _RENDER_SETTINGS.items()
    ):
        raise RuntimeError(
            "Arena video renderer does not match its required capture settings: "
            + json.dumps(actual, sort_keys=True, default=str)
        )
    return actual


def _apply_rendering_settings(settings: Any) -> None:
    # Isaac Sim can reapply application defaults after SimulationCfg raw
    # settings are consumed. Reassert the deterministic capture renderer at
    # the actual capture boundary and reject unsupported settings before a
    # single render is attempted.
    for key, value in _CAPTURE_RENDER_SETTINGS.items():
        settings.set(key, value)
    _rendering_settings_readback(settings)


def _rendering_evidence(settings: Any) -> dict[str, Any]:
    # Kit/Replicator may apply a deferred renderer remap only when it pumps the
    # frame. This read is intentionally non-mutating and happens after the
    # final accepted render, so retained evidence cannot describe stale state.
    actual = _rendering_settings_readback(settings)
    return {
        "mode": "RaytracedLighting",
        "legacy_mode_enabled": True,
        "rt2_enabled": False,
        "path_tracing_enabled": False,
        "antialiasing": "TAA",
        "dlss_execution_mode": "quality",
        "dl_denoiser_enabled": True,
        "frame_generation_enabled": False,
        "minimum_settling_renders": _MINIMUM_SETTLING_RENDERS,
        "stochastic_accumulation": False,
        "readback_phase": "after_final_accepted_render",
        "settings": actual,
    }


def _annotator_ready(env: Any) -> bool:
    capture = getattr(getattr(env, "video_recorder", None), "_capture", None)
    annotator = getattr(capture, "_rgb_annotator", None)
    if annotator is None:
        return False
    data = annotator.get_data()
    if isinstance(data, dict):
        data = data.get("data", [])
    return np.asarray(data).size > 0


def _stage_ready(context: Any) -> bool:
    _, loaded, total = context.get_stage_loading_status()
    return not context.get_stage_streaming_status() and loaded == total


def _render_frame(env: Any) -> np.ndarray:
    if env.sim.is_stopped():
        raise RuntimeError("Arena simulator stopped before its RGB annotator was ready")
    # The native backend pumps Kit once. Calling env.render() first would let
    # KitVisualizer restore playSimulations=True before that native update.
    env._npa_capture_render_call += 1
    with phase_scope(
        env,
        "render_call",
        env._npa_video_capture_recorder._step,
        render_call=env._npa_capture_render_call,
    ):
        frame = np.asarray(env.video_recorder.render_rgb_array())
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise RuntimeError("Arena simulator renderer returned no RGB uint8 frame")
    return frame


def _ready_frame(env: Any, context: Any, frame: np.ndarray) -> bool:
    stage_ready = _stage_ready(context)
    annotator_ready = _annotator_ready(env) if stage_ready else None
    nonblack_rgb = bool(np.any(frame)) if annotator_ready else None
    record_readiness(
        env,
        env._npa_capture_render_call,
        stage_ready=stage_ready,
        annotator_ready=annotator_ready,
        nonblack_rgb=nonblack_rgb,
    )
    return stage_ready and bool(annotator_ready) and bool(nonblack_rgb)


def _ready_capture_frame(env: Any, context: Any) -> tuple[np.ndarray, int, int]:
    renders = 0
    consecutive_ready_renders = 0
    while True:
        frame = _render_frame(env)
        renders += 1
        if _ready_frame(env, context, frame):
            consecutive_ready_renders += 1
        else:
            consecutive_ready_renders = 0
        # The first ready frame establishes the temporal-history baseline. Only
        # subsequent consecutive ready renders count as TAA settling.
        settling_renders = max(0, consecutive_ready_renders - 1)
        if settling_renders >= _MINIMUM_SETTLING_RENDERS:
            return frame.copy(), renders, settling_renders


def _freeze_evidence(
    before: dict, after: dict, renders: int, settling_renders: int
) -> dict[str, Any]:
    return {
        "physics_time_before": before["physics_time"],
        "physics_time_after": after["physics_time"],
        "physics_step_before": before["physics_step"],
        "physics_step_after": after["physics_step"],
        "native_physics_step_before": before["native_physics_step"],
        "native_physics_step_after": after["native_physics_step"],
        "state_sha256_before": _state_hash(before),
        "state_sha256_after": _state_hash(after),
        "render_calls": renders,
        "pre_settling_render_calls": renders - settling_renders,
        "settling_render_calls": settling_renders,
        "consecutive_ready_render_calls": settling_renders + 1,
        "stage_streaming_idle": True,
        "stage_assets_loaded": True,
        "nonblack_rgb": True,
    }


def _capture_frame(env: Any) -> np.ndarray:
    env._npa_capture_render_call = 0
    with phase_scope(env, "capture", env._npa_video_capture_recorder._step):
        return _capture_verified_frame(env)


def _capture_verified_frame(env: Any) -> np.ndarray:
    settings, context = _kit_interfaces()
    _apply_rendering_settings(settings)
    before = _physics_snapshot(env)
    previous = settings.get(_PLAY_SIMULATIONS)
    try:
        settings.set(_PLAY_SIMULATIONS, False)
        env.sim.physics_manager.forward()
        frame, renders, settling_renders = _ready_capture_frame(env, context)
        rendering = _rendering_evidence(settings)
    finally:
        settings.set(_PLAY_SIMULATIONS, previous)
        after = _physics_snapshot(env)
        _assert_physics_unchanged(before, after)
    env._npa_video_rendering = rendering
    env._npa_video_capture_proof = _freeze_evidence(
        before, after, renders, settling_renders
    )
    if getattr(env, "_npa_video_initial_physics_state", None) is None:
        env._npa_video_initial_physics_state = before
    return frame


def _frame_record(
    frame: np.ndarray, directory: Path, name: str, step: int
) -> dict[str, Any]:
    from PIL import Image

    path = directory / name
    Image.fromarray(frame).save(path, format="PNG")
    return {
        "path": name,
        "action_step": step,
        "source_frame_index": step - 1 if step else None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "decoded_rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        "width": frame.shape[1],
        "height": frame.shape[0],
    }


def _step_tensor(step: int, env: Any) -> Any:
    import torch

    return torch.tensor([[step]], dtype=torch.int64, device=env.device)


class _VideoCaptureMethods:
    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise RuntimeError("Arena simulator video capture requires one environment")
        self._directory = Path(env.cfg.recorders.dataset_export_dir_path)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._step = 0
        self._evidence = {
            "schema": "npa.isaac-arena.video-capture.v2",
            "capture_phase": "recorder_post_step_before_autoreset",
            "physics_clock": _PHYSICS_CLOCK,
            "initial": None,
            "terminals": [],
            "captured_action_steps": 0,
            "physics_freeze_checks": [],
        }
        env._npa_video_frame = None
        env._npa_video_physics_clock = _PhysicsStepClock()
        env._npa_video_capture_recorder = self
        self._finalized = False

    def _retain_capture_proof(self) -> None:
        proof = self._env._npa_video_capture_proof
        previous = self._evidence["physics_freeze_checks"]
        if (
            previous
            and proof["native_physics_step_before"]
            <= previous[-1]["native_physics_step_after"]
        ):
            raise RuntimeError(
                "Arena video did not observe native physics steps between actions"
            )
        self._evidence["rendering"] = self._env._npa_video_rendering
        self._evidence["physics_freeze_checks"].append(
            {"action_step": self._step, **proof}
        )

    def _write_evidence(self) -> None:
        path = self._directory / "simulator-video-evidence.json"
        path.write_text(json.dumps(self._evidence, indent=2) + "\n", encoding="utf-8")

    def record_post_reset(self, env_ids: Any) -> tuple[str | None, Any]:
        if self._step or self._evidence["initial"] is not None:
            return None, None
        frame = _capture_frame(self._env)
        self._retain_capture_proof()
        self._evidence["initial"] = _frame_record(
            frame, self._directory, "simulator-initial.png", 0
        )
        self._write_evidence()
        return "npa_video/initial_action_step", _step_tensor(0, self._env)

    def record_post_step(self) -> tuple[str, Any]:
        self._step += 1
        self._env._npa_video_frame = _capture_frame(self._env)
        self._retain_capture_proof()
        self._evidence["captured_action_steps"] = self._step
        self._write_evidence()
        return "npa_video/action_step", _step_tensor(self._step, self._env)

    def record_pre_reset(self, env_ids: Any) -> tuple[str | None, Any]:
        if not self._step:
            return None, None
        index = len(self._evidence["terminals"])
        terminal = _frame_record(
            self._env._npa_video_frame,
            self._directory,
            f"simulator-episode-{index}-terminal.png",
            self._step,
        )
        self._evidence["terminals"].append(terminal)
        self._write_evidence()
        return "npa_video/terminal_action_step", _step_tensor(self._step, self._env)

    def finalize(self) -> None:
        from .simulator_diagnostics import retain_unscored_episode

        if self._finalized:
            return
        diagnostic = retain_unscored_episode(
            self._env, self._directory, self._evidence["captured_action_steps"]
        )
        if diagnostic is not None:
            self._evidence["unscored_diagnostic"] = diagnostic
            self._write_evidence()
        self._finalized = True


def finalize_video_capture(env: Any) -> None:
    """Retain unscored state while the runner's simulator is still live.

    Args:
        env: Upstream Gym environment, optionally wrapped for recording.
    Returns:
        None; ordinary evaluations without a video recorder are unchanged.
    Raises:
        ValueError: The measured diagnostic state is invalid.
        OSError: The diagnostic cannot be retained.
    """
    recorder = getattr(env.unwrapped, "_npa_video_capture_recorder", None)
    if recorder is not None:
        recorder.finalize()


def _capture_recorder_type() -> type:
    from isaaclab.managers.recorder_manager import RecorderTerm

    class _SimulatorVideoRecorder(_VideoCaptureMethods, RecorderTerm):
        pass

    return _SimulatorVideoRecorder


def configure_video_capture(env_cfg: Any) -> None:
    """Add a pre-reset frame recorder after Isaac Sim has started.

    Args:
        env_cfg: Arena manager configuration with its real metric recorder.
    Returns:
        None; adds a recorder term to the existing configuration.
    Raises:
        RuntimeError: No metric recorder exists for the environment.
    """
    from isaaclab.managers.recorder_manager import RecorderTermCfg

    if env_cfg.recorders is None:
        raise RuntimeError("Arena video capture requires the task's metric recorder")
    env_cfg.sim.render.carb_settings.update(_CAPTURE_RENDER_SETTINGS)
    # Isaac Lab applies this native Replicator bridge after raw Carb settings;
    # make both configuration paths request the same supported temporal mode.
    env_cfg.sim.render.antialiasing_mode = "TAA"
    env_cfg.sim.render.dlss_mode = 2
    env_cfg.sim.render.enable_dl_denoiser = True
    env_cfg.sim.render.samples_per_pixel = _CAPTURE_RENDER_SETTINGS[
        "/rtx/directLighting/sampledLighting/samplesPerPixel"
    ]
    env_cfg.recorders.npa_video = RecorderTermCfg(class_type=_capture_recorder_type())


def cached_frame_env(env: Any) -> Any:
    """Wrap Gym video capture to return the real frame from before automatic reset.

    Args:
        env: Base Arena environment with configure_video_capture installed.
    Returns:
        A Gym wrapper whose render method returns the cached post-action frame.
    Raises:
        RuntimeError: A frame is requested before the first action capture.
    """
    import gymnasium as gym

    class _CachedFrameEnv(gym.Wrapper):
        def render(self) -> np.ndarray:
            frame = self.unwrapped._npa_video_frame
            if frame is None:
                raise RuntimeError(
                    "Arena video requested a frame before its first action"
                )
            return frame.copy()

    return _CachedFrameEnv(env)
