"""Standalone, fail-closed Isaac Lab RSL-RL checkpoint evaluator.

The workbench CLI embeds this file into its remote SSH command. Kubernetes
validation jobs can mount and execute the same file directly, which keeps live
GPU evidence aligned with the shipped evaluator instead of maintaining a
second evaluation script.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any


EVAL_FORMAT = "npa.isaac_lab.eval.v1"
SUPPORTED_SUCCESS_METRICS = {"auto", "goal-distance", "survival"}
_LOGGER = logging.getLogger(__name__)
PORTABLE_CHECKPOINT_NAMES = (
    "npa_isaac_lab_checkpoint.pt",
    "model_latest.pt",
    "model.pt",
)


@dataclass(frozen=True)
class EvalConfig:
    task: str
    checkpoint: Path
    num_episodes: int
    output_dir: Path
    success_metric: str = "auto"
    success_distance_m: float = 0.05
    min_success_rate: float = 0.0
    max_steps_per_episode: int = 200
    seed: int = 42
    capture_video: bool = False
    video_length: int = 400
    video_fps: int = 30

    @property
    def summary_path(self) -> Path:
        return self.output_dir / "npa_isaac_lab_eval_summary.json"

    @property
    def video_dir(self) -> Path:
        return self.output_dir / "video"

    @classmethod
    def from_environment(cls) -> "EvalConfig":
        return cls(
            task=_required_env("NPA_ISAAC_EVAL_TASK"),
            checkpoint=Path(_required_env("NPA_ISAAC_EVAL_CHECKPOINT")),
            num_episodes=int(os.environ.get("NPA_ISAAC_EVAL_EPISODES", "10")),
            output_dir=Path(_required_env("NPA_ISAAC_EVAL_OUTPUT_DIR")),
            success_metric=os.environ.get("NPA_ISAAC_EVAL_SUCCESS_METRIC", "auto"),
            success_distance_m=float(
                os.environ.get("NPA_ISAAC_EVAL_SUCCESS_DISTANCE_M", "0.05")
            ),
            min_success_rate=float(
                os.environ.get("NPA_ISAAC_EVAL_MIN_SUCCESS_RATE", "0.0")
            ),
            max_steps_per_episode=int(
                os.environ.get("NPA_ISAAC_EVAL_MAX_STEPS", "200")
            ),
            seed=int(os.environ.get("NPA_ISAAC_EVAL_SEED", "42")),
            capture_video=_env_flag("NPA_ISAAC_EVAL_VIDEO"),
            video_length=int(os.environ.get("NPA_ISAAC_EVAL_VIDEO_LENGTH", "400")),
            video_fps=int(os.environ.get("NPA_ISAAC_EVAL_VIDEO_FPS", "30")),
        )

    def validate(self) -> None:
        if not self.task:
            raise ValueError("task must not be empty")
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode must be positive")
        if self.success_metric not in SUPPORTED_SUCCESS_METRICS:
            raise ValueError(
                f"unsupported success metric {self.success_metric!r}; "
                f"expected one of {sorted(SUPPORTED_SUCCESS_METRICS)}"
            )
        if self.success_distance_m <= 0:
            raise ValueError("success_distance_m must be positive")
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be between 0 and 1")
        if self.video_length <= 0:
            raise ValueError("video_length must be positive")
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive")


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_checkpoint(path: Path) -> tuple[Path, str]:
    """Resolve real RSL-RL weights from a file, directory, or NPA manifest."""

    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if path.is_dir():
        candidates = [
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in {".pt", ".pth"}
        ]
        if candidates:
            return max(candidates, key=checkpoint_preference_key), "rsl_rl_checkpoint"
        raise FileNotFoundError(
            f"checkpoint directory contains no RSL-RL weights: {path}"
        )

    if path.suffix.lower() != ".json":
        return path, "rsl_rl_checkpoint"

    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"checkpoint manifest is not valid JSON: {path}") from exc
    for key in ("stable_checkpoint_path", "checkpoint_path"):
        if not info.get(key):
            continue
        declared = Path(str(info[key]))
        for candidate in (
            path.parent / "npa_isaac_lab_checkpoint.pt",
            path.parent / "model_latest.pt",
            path.parent / "model.pt",
            path.parent / declared.name,
            declared,
        ):
            if candidate.is_file():
                return candidate, str(info.get("format") or "manifest")
    raise FileNotFoundError(
        f"manifest does not resolve to local RSL-RL weights: {path}"
    )


def checkpoint_preference_key(path: Path) -> tuple[int, int, float, str]:
    """Rank portable/final weights ahead of numbered and arbitrary checkpoints."""

    try:
        modified = path.stat().st_mtime
    except OSError:
        modified = 0.0
    if path.name in PORTABLE_CHECKPOINT_NAMES:
        # Earlier entries are the strongest contract names.
        rank = len(PORTABLE_CHECKPOINT_NAMES) - PORTABLE_CHECKPOINT_NAMES.index(
            path.name
        )
        return 3, rank, modified, str(path)
    numbered = re.fullmatch(r"model_(\d+)\.(?:pt|pth)", path.name)
    if numbered:
        return 2, int(numbered.group(1)), modified, str(path)
    return 1, 0, modified, str(path)


def resolve_success_metric(
    requested: str, episode_results: list[dict[str, Any]]
) -> str:
    """Resolve ``auto`` without inventing a task-specific success predicate."""

    if requested != "auto":
        return requested
    if not episode_results:
        raise ValueError("episode_results must not be empty")
    if episode_results and all(
        item.get("native_success") is not None for item in episode_results
    ):
        return "native-success"
    if all(item.get("min_goal_distance_m") is not None for item in episode_results):
        return "goal-distance"
    return "survival"


def apply_success_metric(
    metric: str,
    episode_results: list[dict[str, Any]],
    *,
    success_distance_m: float,
    task: str,
) -> None:
    """Attach a boolean success result to every measured episode."""

    for item in episode_results:
        if metric == "native-success":
            if item.get("native_success") is None:
                raise RuntimeError(
                    "native success metric was not emitted for every episode"
                )
            item["success"] = bool(item["native_success"])
        elif metric == "goal-distance":
            if item.get("min_goal_distance_m") is None:
                raise RuntimeError(
                    f"task {task} did not expose object_pose or ee_pose goal distance; "
                    "select --success-metric survival"
                )
            item["success"] = float(item["min_goal_distance_m"]) <= success_distance_m
        elif metric == "survival":
            item["success"] = not bool(item.get("terminated"))
        else:
            raise ValueError(f"unsupported resolved success metric: {metric}")


def _as_bool(value: Any, torch: Any) -> bool | None:
    if value is None:
        return None
    try:
        return bool(torch.as_tensor(value).any().item())
    except Exception as exc:
        _LOGGER.debug("native success conversion is unavailable", exc_info=exc)
        return None


def _native_success(info: Any, torch: Any) -> bool | None:
    if not isinstance(info, dict):
        return None
    for key in ("success", "is_success", "goal_reached"):
        if key in info:
            return _as_bool(info[key], torch)
    return None


def _step_env(
    env: Any, actions: Any, torch: Any
) -> tuple[Any, Any, bool, bool, bool, bool | None]:
    """Normalize Gymnasium and RSL-RL vector-wrapper step signatures."""

    output = env.step(actions)
    if len(output) == 5:
        observation, reward, terminated, truncated, info = output
        did_terminate = bool(torch.as_tensor(terminated).any().item())
        timed_out = bool(torch.as_tensor(truncated).any().item())
        done = did_terminate or timed_out
    else:
        observation, reward, dones, info = output
        done = bool(torch.as_tensor(dones).any().item())
        timeout_value = info.get("time_outs") if isinstance(info, dict) else None
        if timeout_value is None and isinstance(info, dict):
            timeout_value = info.get("timeouts")
        timed_out = (
            bool(_as_bool(timeout_value, torch)) if timeout_value is not None else False
        )
        did_terminate = done and not timed_out
    return (
        observation,
        reward,
        done,
        did_terminate,
        timed_out,
        _native_success(info, torch),
    )


def _goal_distance(env: Any, torch: Any) -> tuple[float | None, str]:
    unwrapped = env.unwrapped
    try:
        command = unwrapped.command_manager.get_command("object_pose")
        obj = unwrapped.scene["object"].data.root_pos_w[:, :3]
        goal = command[:, :3] + unwrapped.scene.env_origins[:, :3]
        return (
            float(torch.linalg.norm(obj - goal, dim=1).min().item()),
            "object_pose",
        )
    except Exception as exc:
        _LOGGER.debug("object_pose goal distance is unavailable", exc_info=exc)
    try:
        command = unwrapped.command_manager.get_command("ee_pose")
        end_effector = unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :3]
        goal = command[:, :3] + unwrapped.scene.env_origins[:, :3]
        return (
            float(torch.linalg.norm(end_effector - goal, dim=1).min().item()),
            "ee_pose",
        )
    except Exception as exc:
        _LOGGER.debug("ee_pose goal distance is unavailable", exc_info=exc)
        return None, ""


def load_rsl_rl_policy(
    env: Any,
    *,
    task: str,
    checkpoint_file: Path,
    device: str,
) -> tuple[Any, Any]:
    """Wrap an Isaac environment and load a real RSL-RL inference policy."""

    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except Exception:
        from omni.isaac.lab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    agent_config = None
    for loader in ("isaaclab_tasks.utils", "omni.isaac.lab_tasks.utils"):
        try:
            module = __import__(loader, fromlist=["load_cfg_from_registry"])
            agent_config = module.load_cfg_from_registry(
                task, "rsl_rl_cfg_entry_point"
            )
            break
        except Exception as exc:
            _LOGGER.debug(
                "RSL-RL config loader %s did not resolve task %s",
                loader,
                task,
                exc_info=exc,
            )
    if agent_config is None:
        raise RuntimeError(f"no rsl_rl_cfg_entry_point registered for task {task}")
    runner_config = (
        agent_config.to_dict()
        if hasattr(agent_config, "to_dict")
        else dict(agent_config)
    )
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, runner_config, log_dir=None, device=device)
    runner.load(str(checkpoint_file))
    return wrapped, runner.get_inference_policy(device=device)


def write_eval_summary(
    config: EvalConfig,
    *,
    episode_results: list[dict[str, Any]],
    checkpoint_file: Path,
    checkpoint_format: str,
    device: str,
    started: float,
    captured_videos: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the shared metric contract and write a successful eval summary."""

    if not episode_results:
        raise ValueError("episode_results must not be empty")
    success_metric = resolve_success_metric(config.success_metric, episode_results)
    apply_success_metric(
        success_metric,
        episode_results,
        success_distance_m=config.success_distance_m,
        task=config.task,
    )
    reward_values = [
        float(item["reward"])
        for item in episode_results
        if item.get("reward") is not None
    ]
    mean_reward = (
        sum(reward_values) / len(reward_values)
        if len(reward_values) == len(episode_results)
        else None
    )
    distances = [
        float(item["min_goal_distance_m"])
        for item in episode_results
        if item.get("min_goal_distance_m") is not None
    ]
    success_rate = sum(1 for item in episode_results if item["success"]) / len(
        episode_results
    )
    videos = captured_videos or []
    summary = {
        "format": EVAL_FORMAT,
        "status": "success",
        "task": config.task,
        "checkpoint": str(config.checkpoint),
        "resolved_checkpoint": str(checkpoint_file),
        "checkpoint_format": checkpoint_format,
        "policy_loaded": True,
        "num_episodes": len(episode_results),
        "max_steps_per_episode": config.max_steps_per_episode,
        "seed": config.seed,
        "device": device,
        "mean_reward": mean_reward,
        "success_rate": success_rate,
        "success_metric_requested": config.success_metric,
        "success_metric": success_metric,
        "success_distance_m": (
            config.success_distance_m if success_metric == "goal-distance" else None
        ),
        "min_success_rate": config.min_success_rate,
        # A false value means evaluation completed but missed the quality bar.
        # Sim2Real Stage 11 owns the loop decision; runtime/policy-load failures
        # remain non-zero and fail closed.
        "passed": success_rate >= config.min_success_rate,
        "mean_min_goal_distance_m": (
            sum(distances) / len(distances) if distances else None
        ),
        "video": {
            "enabled": config.capture_video,
            "fps": config.video_fps,
            "length_steps": config.video_length,
            "files": videos,
        },
        "episodes": episode_results,
        "duration_seconds": round(time.time() - started, 3),
        "output_path": str(config.summary_path),
    }
    if extra:
        summary.update(extra)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_failure_summary(
    config: EvalConfig,
    *,
    stage: str,
    error: Exception,
    started: float,
    resolved_checkpoint: str = "",
) -> dict[str, Any]:
    failure = {
        "format": EVAL_FORMAT,
        "status": "failed",
        "stage": stage,
        "task": config.task,
        "checkpoint": str(config.checkpoint),
        "resolved_checkpoint": resolved_checkpoint,
        "policy_loaded": False,
        "num_episodes": config.num_episodes,
        "max_steps_per_episode": config.max_steps_per_episode,
        "seed": config.seed,
        "error_type": type(error).__name__,
        "error": str(error),
        "duration_seconds": round(time.time() - started, 3),
        "output_path": str(config.summary_path),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.summary_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
    print("ISAAC_LAB_EVAL_FAILED", flush=True)
    print(json.dumps(failure, indent=2), flush=True)
    return failure


def run_eval(config: EvalConfig) -> dict[str, Any]:
    """Load one real checkpoint and evaluate it in a headless Isaac Lab task."""

    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stage = "runtime_start"
    simulation_app = None
    env = None
    resolved_checkpoint = ""
    try:
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(
            headless=True,
            enable_cameras=config.capture_video,
        ).app

        import gymnasium as gym
        import torch

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        stage = "checkpoint_resolution"
        checkpoint_file, checkpoint_format = resolve_checkpoint(config.checkpoint)
        resolved_checkpoint = str(checkpoint_file)

        print(
            f"ISAAC_LAB_EVAL_START task={config.task} "
            f"checkpoint={checkpoint_file} episodes={config.num_episodes} "
            f"device={'cuda:0' if torch.cuda.is_available() else 'cpu'}",
            flush=True,
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(config.seed)
        env_config = parse_env_cfg(config.task, device=device, num_envs=1)
        if hasattr(env_config, "seed"):
            env_config.seed = config.seed

        stage = "environment_create"
        print("ISAAC_LAB_ENV_CREATE_START", flush=True)
        render_mode = "rgb_array" if config.capture_video else None
        env = gym.make(config.task, cfg=env_config, render_mode=render_mode)
        if config.capture_video:
            config.video_dir.mkdir(parents=True, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(config.video_dir),
                episode_trigger=lambda episode_id: episode_id == 0,
                video_length=config.video_length,
                name_prefix="isaac-lab-eval",
                fps=config.video_fps,
                disable_logger=True,
            )
            print(
                "ISAAC_LAB_EVAL_VIDEO_ENABLED "
                f"directory={config.video_dir} length={config.video_length} "
                f"fps={config.video_fps}",
                flush=True,
            )
        print("ISAAC_LAB_ENV_CREATE_COMPLETE", flush=True)

        stage = "policy_load"
        env, policy = load_rsl_rl_policy(
            env,
            task=config.task,
            checkpoint_file=checkpoint_file,
            device=device,
        )
        print("ISAAC_LAB_EVAL_POLICY_LOADED", flush=True)

        stage = "rollout"
        episode_results: list[dict[str, Any]] = []
        for episode in range(config.num_episodes):
            try:
                reset_output = env.reset(seed=config.seed + episode)
            except TypeError:
                reset_output = env.reset()
            observation = (
                reset_output[0] if isinstance(reset_output, tuple) else reset_output
            )
            episode_reward = 0.0
            steps_ran = 0
            minimum_distance = None
            distance_source = ""
            terminated = False
            timed_out = False
            native_success = None
            for step in range(config.max_steps_per_episode):
                if observation is None:
                    raise RuntimeError("policy observation is unavailable")
                with torch.inference_mode():
                    actions = policy(observation)
                (
                    observation,
                    rewards,
                    done,
                    terminated,
                    timed_out,
                    step_success,
                ) = _step_env(env, actions, torch)
                episode_reward += float(torch.as_tensor(rewards).mean().item())
                steps_ran = step + 1
                if step_success is not None:
                    native_success = step_success
                distance, source = _goal_distance(env, torch)
                if distance is not None:
                    minimum_distance = (
                        distance
                        if minimum_distance is None
                        else min(minimum_distance, distance)
                    )
                    distance_source = source
                if done:
                    break

            episode_results.append(
                {
                    "episode": episode + 1,
                    "steps": steps_ran,
                    "reward": episode_reward,
                    "mean_reward_per_step": episode_reward / max(1, steps_ran),
                    "min_goal_distance_m": minimum_distance,
                    "goal_distance_source": distance_source,
                    "terminated": terminated,
                    "timed_out": timed_out or steps_ran >= config.max_steps_per_episode,
                    "native_success": native_success,
                }
            )
            print(
                f"ISAAC_LAB_EVAL_EPISODE "
                f"episode={episode + 1}/{config.num_episodes} "
                f"steps={steps_ran} reward={episode_reward:.6f} "
                f"min_dist={minimum_distance} terminated={terminated} "
                f"timed_out={timed_out}",
                flush=True,
            )

        captured_videos: list[str] = []
        if config.capture_video:
            stage = "video_finalize"
            env.close()
            env = None
            captured_videos = [
                str(path)
                for path in sorted(config.video_dir.glob("*.mp4"))
                if path.stat().st_size > 0
            ]
            if not captured_videos:
                raise RuntimeError(
                    f"video capture was requested but no MP4 was written under {config.video_dir}"
                )
            print(
                "ISAAC_LAB_EVAL_VIDEO_COMPLETE " + " ".join(captured_videos),
                flush=True,
            )

        stage = "metric_resolution"
        summary = write_eval_summary(
            config,
            episode_results=episode_results,
            checkpoint_file=checkpoint_file,
            checkpoint_format=checkpoint_format,
            device=device,
            started=started,
            captured_videos=captured_videos,
        )
        print("ISAAC_LAB_EVAL_COMPLETE", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    except Exception as exc:
        write_failure_summary(
            config,
            stage=stage,
            error=exc,
            started=started,
            resolved_checkpoint=resolved_checkpoint,
        )
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


def main() -> int:
    try:
        run_eval(EvalConfig.from_environment())
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
