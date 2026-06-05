"""SONIC ONNX policy evaluation backends.

The eval path consumes the metadata sidecar emitted by ``sonic export``. The
reference backend prefers a gymnasium environment when one is configured and
falls back to a marked smoke rollout when no local locomotion simulator is
wireable. The container backend stages the exported ONNX and sidecar behind a
stable file contract for contributed evaluators.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np

from npa.workbench.sonic import EXPORT_METADATA_FORMAT, load_export_metadata

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient


EVAL_RESULT_FORMAT = "npa_sonic_eval_result_v1"
DEFAULT_EVAL_OUTPUT_NAME = "sonic_eval_results.json"
DEFAULT_EVAL_ENV = "smoke"
DEFAULT_CONTAINER_RUNTIME = "docker"
DEFAULT_CONTAINER_GPUS = "all"
DEFAULT_CONTAINER_DRIVER_CAPABILITIES = "all"
DEFAULT_CONTAINER_VULKAN_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"
DEFAULT_CONTAINER_GLX_VENDOR = "nvidia"
DEFAULT_CONTAINER_POLICY_PATH = "/npa/eval/input/policy.onnx"
DEFAULT_CONTAINER_METADATA_PATH = "/npa/eval/input/metadata.json"
DEFAULT_CONTAINER_OUTPUT_PATH = "/npa/eval/output/sonic_eval_results.json"
DEFAULT_CONTAINER_RENDER_FRAMES = 8
DEFAULT_CONTAINER_XDG_RUNTIME_DIR = "/tmp/xdg-runtime"
DEFAULT_CONTAINER_OMNI_USER_DIR = "/tmp/isaac-sim-cache"
DEFAULT_CONTAINER_OMNI_LOG_DIR = "/tmp/isaac-sim-cache/logs"
REFERENCE_BACKEND = "reference"
CONTAINER_BACKEND = "container"
BACKENDS = {REFERENCE_BACKEND, CONTAINER_BACKEND}
BUILTIN_REFERENCE_ENVS = {"locomotion-smoke", "sonic-locomotion-smoke"}
BUILTIN_REFERENCE_STEPS = 32

EVAL_RESULT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SONIC ONNX eval result",
    "type": "object",
    "required": [
        "format",
        "status",
        "backend",
        "mode",
        "smoke_level",
        "policy",
        "eval",
        "metrics",
        "episodes",
        "warnings",
    ],
    "properties": {
        "format": {"const": EVAL_RESULT_FORMAT},
        "status": {"type": "string"},
        "backend": {"enum": sorted(BACKENDS)},
        "mode": {"type": "string"},
        "smoke_level": {"type": "boolean"},
        "policy": {
            "type": "object",
            "required": [
                "onnx_path",
                "metadata_path",
                "input_name",
                "output_name",
                "obs_dim",
                "action_dim",
                "normalize",
            ],
        },
        "eval": {
            "type": "object",
            "required": ["env", "episodes", "generated_at"],
        },
        "metrics": {
            "type": "object",
            "required": [
                "episode_return_mean",
                "distance_mean",
                "fall_rate",
                "termination_rate",
                "episode_length_mean",
                "valid_action_rate",
            ],
        },
        "episodes": {"type": "array", "items": {"type": "object"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}


class SonicEvalError(ValueError):
    """Raised when a SONIC ONNX policy cannot be evaluated."""


class _EvalBundle:
    def __init__(
        self, *, onnx_path: Path, metadata_path: Path, metadata: dict[str, Any]
    ) -> None:
        self.onnx_path = onnx_path
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.input_name = str(metadata.get("input_name") or "obs")
        self.output_name = str(metadata.get("output_name") or "action")
        self.normalize = str(metadata.get("normalize") or "baked")
        self.normalization = (
            metadata.get("normalization")
            if isinstance(metadata.get("normalization"), dict)
            else None
        )
        self.obs_spec = (
            metadata.get("obs_spec")
            if isinstance(metadata.get("obs_spec"), dict)
            else {}
        )
        self.action_spec = (
            metadata.get("action_spec")
            if isinstance(metadata.get("action_spec"), dict)
            else {}
        )
        self.obs_dim = _required_dim(self.obs_spec, "observation")
        self.action_dim = _required_dim(self.action_spec, "action")
        control_dt = metadata.get("control_dt")
        self.control_dt = (
            float(control_dt) if isinstance(control_dt, int | float) else None
        )


class _OnnxPolicy:
    def __init__(self, bundle: _EvalBundle) -> None:
        ort = _import_onnxruntime()
        self._bundle = bundle
        self._session = ort.InferenceSession(
            str(bundle.onnx_path), providers=["CPUExecutionProvider"]
        )

    def predict(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if obs.shape[-1] != self._bundle.obs_dim:
            raise SonicEvalError(
                f"observation dimension {obs.shape[-1]} does not match exported "
                f"policy dimension {self._bundle.obs_dim}"
            )
        if self._bundle.normalize == "sidecar":
            if self._bundle.normalization is None:
                raise SonicEvalError(
                    "sidecar-normalized export is missing normalization stats"
                )
            obs = _apply_sidecar_normalization(obs, self._bundle.normalization)
        outputs = self._session.run(
            [self._bundle.output_name],
            {self._bundle.input_name: obs.astype(np.float32, copy=False)},
        )
        action = np.asarray(outputs[0], dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.shape[-1] != self._bundle.action_dim:
            raise SonicEvalError(
                f"ONNX action dimension {action.shape[-1]} does not match metadata "
                f"dimension {self._bundle.action_dim}"
            )
        if not np.all(np.isfinite(action)):
            raise SonicEvalError("ONNX policy produced non-finite actions")
        return action


def evaluate_onnx_policy(
    *,
    onnx: str,
    metadata: str | None = None,
    backend: str = REFERENCE_BACKEND,
    episodes: int = 8,
    env: str = DEFAULT_EVAL_ENV,
    output: str = "",
    container_image: str = "",
    container_runtime: str = DEFAULT_CONTAINER_RUNTIME,
    container_gpus: str = DEFAULT_CONTAINER_GPUS,
    container_driver_capabilities: str = DEFAULT_CONTAINER_DRIVER_CAPABILITIES,
    container_vulkan_icd: str = DEFAULT_CONTAINER_VULKAN_ICD,
    container_glx_vendor: str = DEFAULT_CONTAINER_GLX_VENDOR,
    container_device: list[str] | None = None,
    container_render_frames: int = DEFAULT_CONTAINER_RENDER_FRAMES,
    container_policy_path: str = DEFAULT_CONTAINER_POLICY_PATH,
    container_metadata_path: str = DEFAULT_CONTAINER_METADATA_PATH,
    container_output_path: str = DEFAULT_CONTAINER_OUTPUT_PATH,
    container_args: list[str] | None = None,
    storage_client: "StorageClient | None" = None,
) -> dict[str, Any]:
    """Evaluate an exported SONIC ONNX policy and optionally write JSON results."""

    backend = _validate_backend(backend)
    if episodes <= 0:
        raise SonicEvalError("--episodes must be positive")
    if container_render_frames <= 0:
        raise SonicEvalError("--container-render-frames must be positive")

    bundle = load_eval_bundle(onnx=onnx, metadata=metadata)
    if backend == REFERENCE_BACKEND:
        result = _run_reference_backend(bundle=bundle, episodes=episodes, env=env)
    else:
        result = _run_container_backend(
            bundle=bundle,
            episodes=episodes,
            env=env,
            container_image=container_image,
            container_runtime=container_runtime,
            container_gpus=container_gpus,
            container_driver_capabilities=container_driver_capabilities,
            container_vulkan_icd=container_vulkan_icd,
            container_glx_vendor=container_glx_vendor,
            container_devices=container_device or [],
            container_render_frames=container_render_frames,
            container_policy_path=container_policy_path,
            container_metadata_path=container_metadata_path,
            container_output_path=container_output_path,
            container_args=container_args or [],
        )

    if output:
        result["result_uri"] = write_eval_result(
            result,
            output=output,
            storage_client=storage_client,
        )
    return result


def load_eval_bundle(*, onnx: str, metadata: str | None = None) -> _EvalBundle:
    """Load an exported ONNX policy and its SONIC export metadata sidecar."""

    if not onnx:
        raise SonicEvalError("--onnx is required")
    onnx_path = Path(onnx)
    if not onnx_path.exists():
        raise SonicEvalError(f"ONNX policy not found: {onnx}")
    metadata_path = _resolve_metadata_path(onnx_path, metadata)
    payload = load_export_metadata(str(metadata_path))
    if payload.get("format") != EXPORT_METADATA_FORMAT:
        raise SonicEvalError(
            f"metadata format must be {EXPORT_METADATA_FORMAT}, got {payload.get('format')!r}"
        )
    return _EvalBundle(
        onnx_path=onnx_path, metadata_path=metadata_path, metadata=payload
    )


def result_uri_for(output: str) -> str:
    """Return the JSON result URI for a local path or S3 output target."""

    if not output:
        return ""
    if output.startswith("s3://"):
        if output.endswith(".json"):
            return output
        return output.rstrip("/") + f"/{DEFAULT_EVAL_OUTPUT_NAME}"
    path = Path(output)
    if path.suffix == ".json":
        return str(path)
    return str(path / DEFAULT_EVAL_OUTPUT_NAME)


def write_eval_result(
    payload: dict[str, Any],
    *,
    output: str,
    storage_client: "StorageClient | None" = None,
) -> str:
    """Write a SONIC eval result JSON document to local disk or S3."""

    result_uri = result_uri_for(output)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if result_uri.startswith("s3://"):
        from npa.clients.storage import StorageClient

        client = storage_client or StorageClient.from_environment()
        with tempfile.TemporaryDirectory(prefix="npa-sonic-eval-result-") as tmp:
            local_path = Path(tmp) / DEFAULT_EVAL_OUTPUT_NAME
            local_path.write_text(body, encoding="utf-8")
            return client.upload_file(str(local_path), result_uri)

    path = Path(result_uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def _run_reference_backend(
    *, bundle: _EvalBundle, episodes: int, env: str
) -> dict[str, Any]:
    warnings: list[str] = []
    policy = _OnnxPolicy(bundle)
    env_name = (env or DEFAULT_EVAL_ENV).strip() or DEFAULT_EVAL_ENV
    if env_name.lower() in BUILTIN_REFERENCE_ENVS:
        return _run_builtin_locomotion_reference(
            bundle=bundle,
            policy=policy,
            episodes=episodes,
            env=env_name,
        )
    if env_name.lower() not in {"smoke", "none"}:
        try:
            return _run_gymnasium_reference(
                bundle=bundle,
                policy=policy,
                episodes=episodes,
                env_name=env_name,
            )
        except SonicEvalError as exc:
            warnings.append(
                f"Reference simulator unavailable for env={env_name!r}; "
                f"ran smoke-level ONNX rollout instead: {exc}"
            )
    else:
        warnings.append(
            "No reference simulator env configured; ran smoke-level ONNX rollout."
        )
    return _run_smoke_reference(
        bundle=bundle,
        policy=policy,
        episodes=episodes,
        env=env_name,
        warnings=warnings,
    )


def _run_builtin_locomotion_reference(
    *,
    bundle: _EvalBundle,
    policy: _OnnxPolicy,
    episodes: int,
    env: str,
) -> dict[str, Any]:
    episode_metrics: list[dict[str, Any]] = []
    dt = float(bundle.control_dt or 0.02)
    for episode_index in range(episodes):
        x_position = 0.0
        velocity = 0.2 + 0.02 * float(episode_index)
        pitch = 0.0
        total_reward = 0.0
        energy = 0.0
        valid_actions = 0
        fall = False
        terminated = False
        action_norms: list[float] = []
        length = 0

        for step_index in range(BUILTIN_REFERENCE_STEPS):
            observation = _locomotion_observation(
                bundle=bundle,
                x_position=x_position,
                velocity=velocity,
                pitch=pitch,
                step_index=step_index,
                episode_index=episode_index,
            )
            action = policy.predict(observation)[0]
            valid_actions += 1
            length = step_index + 1

            action_norm = float(np.linalg.norm(action))
            action_norms.append(action_norm)
            action_mean = float(np.tanh(np.mean(action))) if action.size else 0.0
            acceleration = 0.15 + 0.2 * action_mean - 0.01 * min(action_norm, 5.0)
            velocity = float(np.clip(velocity + acceleration * dt, -0.25, 2.0))
            x_position += velocity * dt
            pitch = float(0.95 * pitch + 0.01 * action_mean)
            energy += action_norm * dt
            fall = abs(pitch) > 0.65 or not np.all(np.isfinite(action))
            total_reward += velocity * dt - 0.001 * action_norm

            if fall:
                terminated = True
                total_reward -= 1.0
                break

        distance = max(0.0, x_position)
        episode_metrics.append(
            {
                "episode_index": episode_index,
                "episode_return": total_reward,
                "distance": distance,
                "fall": fall,
                "terminated": terminated,
                "truncated": False,
                "episode_length": length,
                "valid_actions": valid_actions,
                "steps": length,
                "action_norm_mean": _mean(action_norms),
                "energy": energy,
            }
        )

    return _result_payload(
        bundle=bundle,
        backend=REFERENCE_BACKEND,
        mode="sim",
        smoke_level=False,
        env=env,
        episodes=episodes,
        episode_metrics=episode_metrics,
        warnings=[],
    )


def _run_gymnasium_reference(
    *,
    bundle: _EvalBundle,
    policy: _OnnxPolicy,
    episodes: int,
    env_name: str,
) -> dict[str, Any]:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise SonicEvalError("gymnasium is not installed") from exc

    try:
        sim = gym.make(env_name)
    except Exception as exc:  # noqa: BLE001
        raise SonicEvalError(f"failed to create gymnasium env: {exc}") from exc

    episode_metrics: list[dict[str, Any]] = []
    warnings: list[str] = []
    max_steps = int(
        getattr(getattr(sim, "spec", None), "max_episode_steps", None) or 1000
    )
    try:
        for episode_index in range(episodes):
            reset_output = sim.reset()
            if isinstance(reset_output, tuple) and len(reset_output) == 2:
                observation, reset_info = reset_output
            else:
                observation, reset_info = reset_output, {}
            start_position = _extract_position(observation, reset_info)
            last_position = start_position
            total_reward = 0.0
            distance = 0.0
            valid_actions = 0
            fall = False
            terminated = False
            truncated = False
            length = 0
            for step_index in range(max_steps):
                obs_vector = _observation_vector(
                    observation,
                    obs_dim=bundle.obs_dim,
                    obs_spec=bundle.obs_spec,
                    warnings=warnings,
                )
                raw_action = policy.predict(obs_vector)[0]
                valid_actions += 1
                action = _action_for_env(
                    raw_action, getattr(sim, "action_space", None), warnings
                )
                step_output = sim.step(action)
                if len(step_output) == 5:
                    observation, reward, terminated, truncated, info = step_output
                else:
                    observation, reward, done, info = step_output
                    terminated = bool(done)
                    truncated = False
                length = step_index + 1
                total_reward += float(reward)
                fall = fall or bool(
                    info.get("fall") or info.get("fallen") or info.get("is_fallen")
                )
                position = _extract_position(observation, info)
                if position is not None and last_position is not None:
                    distance += abs(float(position) - float(last_position))
                last_position = position if position is not None else last_position
                distance = float(
                    info.get("distance")
                    or info.get("forward_distance")
                    or info.get("traveled_distance")
                    or distance
                )
                if terminated or truncated:
                    break
            episode_metrics.append(
                {
                    "episode_index": episode_index,
                    "episode_return": total_reward,
                    "distance": distance,
                    "fall": fall,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "episode_length": length,
                    "valid_actions": valid_actions,
                    "steps": length,
                }
            )
    finally:
        close = getattr(sim, "close", None)
        if callable(close):
            close()

    return _result_payload(
        bundle=bundle,
        backend=REFERENCE_BACKEND,
        mode="sim",
        smoke_level=False,
        env=env_name,
        episodes=episodes,
        episode_metrics=episode_metrics,
        warnings=_dedupe(warnings),
    )


def _run_smoke_reference(
    *,
    bundle: _EvalBundle,
    policy: _OnnxPolicy,
    episodes: int,
    env: str,
    warnings: list[str],
) -> dict[str, Any]:
    episode_metrics: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        observation = _representative_observation(bundle, episode_index)
        action = policy.predict(observation)[0]
        action_norm = float(np.linalg.norm(action))
        episode_metrics.append(
            {
                "episode_index": episode_index,
                "episode_return": 0.0,
                "distance": 0.0,
                "fall": False,
                "terminated": False,
                "truncated": False,
                "episode_length": 1,
                "valid_actions": 1,
                "steps": 1,
                "action_norm": action_norm,
                "action_min": float(action.min()) if action.size else 0.0,
                "action_max": float(action.max()) if action.size else 0.0,
            }
        )

    return _result_payload(
        bundle=bundle,
        backend=REFERENCE_BACKEND,
        mode="smoke",
        smoke_level=True,
        env=env,
        episodes=episodes,
        episode_metrics=episode_metrics,
        warnings=_dedupe(warnings),
    )


def _run_container_backend(
    *,
    bundle: _EvalBundle,
    episodes: int,
    env: str,
    container_image: str,
    container_runtime: str,
    container_gpus: str,
    container_driver_capabilities: str,
    container_vulkan_icd: str,
    container_glx_vendor: str,
    container_devices: list[str],
    container_render_frames: int,
    container_policy_path: str,
    container_metadata_path: str,
    container_output_path: str,
    container_args: list[str],
) -> dict[str, Any]:
    if not container_image:
        raise SonicEvalError("--container-image is required for --backend container")
    if not container_runtime:
        raise SonicEvalError("--container-runtime is required for --backend container")

    with tempfile.TemporaryDirectory(prefix="npa-sonic-eval-container-") as tmp:
        stage = Path(tmp)
        input_dir = stage / "input"
        output_dir = stage / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        staged_policy = input_dir / Path(container_policy_path).name
        staged_metadata = input_dir / Path(container_metadata_path).name
        shutil.copy2(bundle.onnx_path, staged_policy)
        shutil.copy2(bundle.metadata_path, staged_metadata)

        command = _container_command(
            runtime=container_runtime,
            image=container_image,
            input_dir=input_dir,
            output_dir=output_dir,
            container_gpus=container_gpus,
            container_driver_capabilities=container_driver_capabilities,
            container_vulkan_icd=container_vulkan_icd,
            container_glx_vendor=container_glx_vendor,
            container_devices=container_devices,
            container_render_frames=container_render_frames,
            container_policy_path=container_policy_path,
            container_metadata_path=container_metadata_path,
            container_output_path=container_output_path,
            episodes=episodes,
            env=env,
            extra_args=container_args,
        )
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            raise SonicEvalError(
                "container eval failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )

        host_output_path = output_dir / Path(container_output_path).name
        if not host_output_path.exists():
            raise SonicEvalError(
                "container eval did not write expected result JSON: "
                f"{container_output_path}"
            )
        container_payload = json.loads(host_output_path.read_text(encoding="utf-8"))

    return _normalize_container_result(
        bundle=bundle,
        payload=container_payload,
        episodes=episodes,
        env=env,
        container_image=container_image,
        container_runtime=container_runtime,
        container_gpus=container_gpus,
        container_driver_capabilities=container_driver_capabilities,
        container_vulkan_icd=container_vulkan_icd,
        container_glx_vendor=container_glx_vendor,
        container_devices=container_devices,
        container_render_frames=container_render_frames,
        container_policy_path=container_policy_path,
        container_metadata_path=container_metadata_path,
        container_output_path=container_output_path,
    )


def _container_command(
    *,
    runtime: str,
    image: str,
    input_dir: Path,
    output_dir: Path,
    container_gpus: str,
    container_driver_capabilities: str,
    container_vulkan_icd: str,
    container_glx_vendor: str,
    container_devices: list[str],
    container_render_frames: int,
    container_policy_path: str,
    container_metadata_path: str,
    container_output_path: str,
    episodes: int,
    env: str,
    extra_args: list[str],
) -> list[str]:
    policy_parent = str(Path(container_policy_path).parent)
    metadata_parent = str(Path(container_metadata_path).parent)
    output_parent = str(Path(container_output_path).parent)
    mounts = [
        (str(input_dir), policy_parent, "ro"),
        (str(input_dir), metadata_parent, "ro"),
        (str(output_dir), output_parent, ""),
    ]
    command = [runtime, "run", "--rm"]
    if _is_docker_runtime(runtime):
        command.extend(["--runtime", "nvidia"])
        if container_gpus:
            command.extend(["--gpus", container_gpus])
        for device in container_devices:
            if device:
                command.extend(["--device", device])
    for host_path, container_path, mode in _dedupe_mounts(mounts):
        mount = f"{host_path}:{container_path}"
        if mode:
            mount = f"{mount}:{mode}"
        command.extend(["-v", mount])
    env_vars = {
        "NPA_SONIC_ONNX": container_policy_path,
        "NPA_SONIC_METADATA": container_metadata_path,
        "NPA_SONIC_OUTPUT": container_output_path,
        "NPA_SONIC_EPISODES": str(episodes),
        "NPA_SONIC_ENV": env,
        "NPA_SONIC_RESULT_FORMAT": EVAL_RESULT_FORMAT,
        "NPA_SONIC_RENDER_FRAMES": str(container_render_frames),
        "XDG_RUNTIME_DIR": DEFAULT_CONTAINER_XDG_RUNTIME_DIR,
        "OMNI_USER_DIR": DEFAULT_CONTAINER_OMNI_USER_DIR,
        "OMNI_LOG_DIR": DEFAULT_CONTAINER_OMNI_LOG_DIR,
    }
    if container_driver_capabilities:
        env_vars["NVIDIA_DRIVER_CAPABILITIES"] = container_driver_capabilities
    if container_vulkan_icd:
        env_vars["VK_ICD_FILENAMES"] = container_vulkan_icd
        env_vars["VK_DRIVER_FILES"] = container_vulkan_icd
    if container_glx_vendor:
        env_vars["__GLX_VENDOR_LIBRARY_NAME"] = container_glx_vendor
    for key, value in env_vars.items():
        command.extend(["-e", f"{key}={value}"])
    command.append(image)
    command.extend(extra_args)
    return command


def _normalize_container_result(
    *,
    bundle: _EvalBundle,
    payload: dict[str, Any],
    episodes: int,
    env: str,
    container_image: str,
    container_runtime: str,
    container_gpus: str,
    container_driver_capabilities: str,
    container_vulkan_icd: str,
    container_glx_vendor: str,
    container_devices: list[str],
    container_render_frames: int,
    container_policy_path: str,
    container_metadata_path: str,
    container_output_path: str,
) -> dict[str, Any]:
    base = _result_payload(
        bundle=bundle,
        backend=CONTAINER_BACKEND,
        mode="container",
        smoke_level=False,
        env=env,
        episodes=episodes,
        episode_metrics=payload.get("episodes")
        if isinstance(payload.get("episodes"), list)
        else [],
        warnings=payload.get("warnings")
        if isinstance(payload.get("warnings"), list)
        else [],
    )
    if isinstance(payload.get("metrics"), dict):
        base["metrics"].update(_jsonable(payload["metrics"]))
    if isinstance(payload.get("status"), str):
        base["status"] = payload["status"]
    base["container"] = {
        "image": container_image,
        "runtime": container_runtime,
        "gpus": container_gpus,
        "driver_capabilities": container_driver_capabilities,
        "vulkan_icd": container_vulkan_icd,
        "glx_vendor": container_glx_vendor,
        "devices": _jsonable(container_devices),
        "render_frames": container_render_frames,
        "policy_path": container_policy_path,
        "metadata_path": container_metadata_path,
        "output_path": container_output_path,
    }
    for key in ("render", "diagnostics", "artifacts"):
        if key in payload:
            base[key] = _jsonable(payload[key])
    if payload.get("format") != EVAL_RESULT_FORMAT:
        base["external_result"] = _jsonable(payload)
    return base


def _result_payload(
    *,
    bundle: _EvalBundle,
    backend: str,
    mode: str,
    smoke_level: bool,
    env: str,
    episodes: int,
    episode_metrics: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "format": EVAL_RESULT_FORMAT,
        "status": "completed",
        "backend": backend,
        "mode": mode,
        "smoke_level": smoke_level,
        "result_uri": "",
        "policy": {
            "onnx_path": str(bundle.onnx_path),
            "metadata_path": str(bundle.metadata_path),
            "input_name": bundle.input_name,
            "output_name": bundle.output_name,
            "obs_dim": bundle.obs_dim,
            "action_dim": bundle.action_dim,
            "normalize": bundle.normalize,
            "control_dt": bundle.control_dt,
            "opset": bundle.metadata.get("opset"),
            "obs_spec": _jsonable(bundle.obs_spec),
            "action_spec": _jsonable(bundle.action_spec),
        },
        "eval": {
            "env": env,
            "episodes": episodes,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "metrics": _aggregate_metrics(episode_metrics),
        "episodes": _jsonable(episode_metrics),
        "warnings": warnings,
    }


def _aggregate_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [_float(item.get("episode_return")) for item in episodes]
    distances = [_float(item.get("distance")) for item in episodes]
    lengths = [_float(item.get("episode_length")) for item in episodes]
    total_steps = sum(max(0, int(item.get("steps", 0) or 0)) for item in episodes)
    total_valid_actions = sum(
        max(0, int(item.get("valid_actions", 0) or 0)) for item in episodes
    )
    count = len(episodes)
    return {
        "episode_return_mean": _mean(returns),
        "episode_return_min": min(returns) if returns else 0.0,
        "episode_return_max": max(returns) if returns else 0.0,
        "distance_mean": _mean(distances),
        "distance_min": min(distances) if distances else 0.0,
        "distance_max": max(distances) if distances else 0.0,
        "fall_rate": _rate(episodes, "fall"),
        "termination_rate": _rate(episodes, "terminated"),
        "truncation_rate": _rate(episodes, "truncated"),
        "episode_length_mean": _mean(lengths),
        "valid_action_rate": (
            float(total_valid_actions) / float(total_steps) if total_steps else 0.0
        ),
        "episodes": count,
    }


def _resolve_metadata_path(onnx_path: Path, metadata: str | None) -> Path:
    candidates = [Path(metadata)] if metadata else []
    candidates.extend(
        [
            onnx_path.with_suffix(".metadata.json"),
            Path(str(onnx_path) + ".metadata.json"),
        ]
    )
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    expected = (
        str(candidates[0])
        if candidates
        else str(onnx_path.with_suffix(".metadata.json"))
    )
    raise SonicEvalError(f"metadata sidecar not found: {expected}")


def _required_dim(spec: dict[str, Any], label: str) -> int:
    dim = _dim_from_spec(spec)
    if dim <= 0:
        raise SonicEvalError(f"{label} dimension is missing from export metadata")
    return dim


def _dim_from_spec(spec: dict[str, Any]) -> int:
    if not spec:
        return 0
    dim = spec.get("dim")
    if isinstance(dim, int):
        return int(dim)
    shape = spec.get("shape")
    if isinstance(shape, list) and shape:
        product = 1
        for item in shape:
            product *= int(item)
        return product
    fields = spec.get("fields")
    if isinstance(fields, list):
        return sum(_dim_from_spec(field) for field in fields if isinstance(field, dict))
    return 0


def _representative_observation(bundle: _EvalBundle, episode_index: int) -> np.ndarray:
    if bundle.normalize == "sidecar" and bundle.normalization:
        mean = np.asarray(bundle.normalization.get("mean"), dtype=np.float32).reshape(
            -1
        )
        if mean.size == bundle.obs_dim:
            std = bundle.normalization.get("std")
            if std is None and bundle.normalization.get("var") is not None:
                std = np.sqrt(
                    np.asarray(bundle.normalization["var"], dtype=np.float32)
                    + float(bundle.normalization.get("epsilon", 1e-5))
                )
            std_array = np.ones(bundle.obs_dim, dtype=np.float32)
            if std is not None:
                candidate = np.asarray(std, dtype=np.float32).reshape(-1)
                if candidate.size == bundle.obs_dim:
                    std_array = candidate
            return mean + (0.01 * float(episode_index + 1) * std_array)

    observation = np.zeros(bundle.obs_dim, dtype=np.float32)
    if bundle.obs_dim:
        span = min(bundle.obs_dim, 8)
        observation[:span] = np.linspace(-0.05, 0.05, num=span, dtype=np.float32)
        observation = observation + np.float32(episode_index * 0.001)
    return observation


def _locomotion_observation(
    *,
    bundle: _EvalBundle,
    x_position: float,
    velocity: float,
    pitch: float,
    step_index: int,
    episode_index: int,
) -> np.ndarray:
    observation = np.zeros(bundle.obs_dim, dtype=np.float32)
    values = np.asarray(
        [
            velocity,
            pitch,
            x_position,
            float(step_index) * float(bundle.control_dt or 0.02),
            0.01 * float(episode_index + 1),
            np.sin(0.1 * float(step_index)),
            np.cos(0.1 * float(step_index)),
            1.0,
        ],
        dtype=np.float32,
    )
    span = min(bundle.obs_dim, values.size)
    observation[:span] = values[:span]
    return observation


def _observation_vector(
    observation: Any,
    *,
    obs_dim: int,
    obs_spec: dict[str, Any],
    warnings: list[str],
) -> np.ndarray:
    if isinstance(observation, dict):
        ordered = _ordered_dict_observation(observation, obs_spec)
        if ordered is not None:
            return _fit_vector(
                ordered, obs_dim=obs_dim, label="observation", warnings=warnings
            )
        values = [
            np.asarray(observation[key]).reshape(-1) for key in sorted(observation)
        ]
        return _fit_vector(
            np.concatenate(values),
            obs_dim=obs_dim,
            label="observation",
            warnings=warnings,
        )
    return _fit_vector(
        np.asarray(observation).reshape(-1),
        obs_dim=obs_dim,
        label="observation",
        warnings=warnings,
    )


def _ordered_dict_observation(
    observation: dict[str, Any], obs_spec: dict[str, Any]
) -> np.ndarray | None:
    fields = obs_spec.get("fields")
    if not isinstance(fields, list):
        return None
    parts: list[np.ndarray] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = field.get("name")
        if not isinstance(name, str) or name not in observation:
            return None
        parts.append(np.asarray(observation[name], dtype=np.float32).reshape(-1))
    if not parts:
        return None
    return np.concatenate(parts)


def _action_for_env(action: np.ndarray, action_space: Any, warnings: list[str]) -> Any:
    target_shape = getattr(action_space, "shape", None)
    if not target_shape:
        return action
    shaped = _fit_vector(
        action, obs_dim=int(np.prod(target_shape)), label="action", warnings=warnings
    )
    shaped = shaped.reshape(target_shape)
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is not None and high is not None:
        shaped = np.clip(shaped, low, high)
    return shaped


def _fit_vector(
    vector: np.ndarray, *, obs_dim: int, label: str, warnings: list[str]
) -> np.ndarray:
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    if flat.size == obs_dim:
        return flat
    warnings.append(
        f"{label} vector dim {flat.size} was adapted to expected dim {obs_dim}"
    )
    if flat.size > obs_dim:
        return flat[:obs_dim]
    padded = np.zeros(obs_dim, dtype=np.float32)
    padded[: flat.size] = flat
    return padded


def _extract_position(observation: Any, info: dict[str, Any]) -> float | None:
    for key in ("x_position", "position_x", "base_x", "root_x"):
        value = info.get(key)
        if isinstance(value, int | float):
            return float(value)
    if isinstance(observation, dict):
        for key in ("x_position", "base_position", "root_position", "position"):
            value = observation.get(key)
            if value is None:
                continue
            array = np.asarray(value).reshape(-1)
            if array.size:
                return float(array[0])
    return None


def _apply_sidecar_normalization(obs: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats.get("mean"), dtype=np.float32)
    if mean.shape[-1] != obs.shape[-1]:
        raise SonicEvalError(
            "sidecar normalization mean dimension does not match observation"
        )
    if stats.get("std") is not None:
        denom = np.asarray(stats["std"], dtype=np.float32)
    else:
        denom = np.sqrt(
            np.asarray(stats.get("var"), dtype=np.float32)
            + float(stats.get("epsilon", 1e-5))
        )
    normalized = (obs - mean) / denom
    clip = stats.get("clip", 5.0)
    if clip is not None:
        normalized = np.clip(normalized, -float(clip), float(clip))
    return normalized.astype(np.float32, copy=False)


def _validate_backend(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in BACKENDS:
        raise SonicEvalError(f"--backend must be one of: {', '.join(sorted(BACKENDS))}")
    return normalized


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SonicEvalError(
            "SONIC ONNX eval requires onnxruntime. Install the npa[sonic] extra."
        ) from exc
    return ort


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_mounts(mounts: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str, str]] = []
    for host_path, container_path, mode in mounts:
        key = (host_path, container_path)
        if key in seen:
            continue
        seen.add(key)
        result.append((host_path, container_path, mode))
    return result


def _is_docker_runtime(runtime: str) -> bool:
    return Path(runtime).name == "docker"


def _rate(episodes: list[dict[str, Any]], key: str) -> float:
    if not episodes:
        return 0.0
    return sum(1 for item in episodes if bool(item.get(key))) / float(len(episodes))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _float(value: Any) -> float:
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "BACKENDS",
    "BUILTIN_REFERENCE_ENVS",
    "CONTAINER_BACKEND",
    "DEFAULT_CONTAINER_DRIVER_CAPABILITIES",
    "DEFAULT_CONTAINER_GLX_VENDOR",
    "DEFAULT_CONTAINER_GPUS",
    "DEFAULT_CONTAINER_METADATA_PATH",
    "DEFAULT_CONTAINER_OUTPUT_PATH",
    "DEFAULT_CONTAINER_POLICY_PATH",
    "DEFAULT_CONTAINER_RENDER_FRAMES",
    "DEFAULT_CONTAINER_RUNTIME",
    "DEFAULT_CONTAINER_VULKAN_ICD",
    "DEFAULT_EVAL_ENV",
    "DEFAULT_EVAL_OUTPUT_NAME",
    "EVAL_RESULT_FORMAT",
    "EVAL_RESULT_SCHEMA",
    "REFERENCE_BACKEND",
    "SonicEvalError",
    "evaluate_onnx_policy",
    "load_eval_bundle",
    "result_uri_for",
    "write_eval_result",
]
