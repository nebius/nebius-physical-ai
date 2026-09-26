"""Execute the pinned upstream MJLab runtime inside a dedicated worker process."""

from __future__ import annotations

from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

from .schemas import EvalRequest, ExportRequest, MJLAB_VERSION, MjlabError, TrainRequest


def _imports():
    if importlib.metadata.version("mjlab") != MJLAB_VERSION:
        raise MjlabError(f"MJLab {MJLAB_VERSION} is required")
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks import registry

    return registry


def _configs(request, inputs, *, play=False):
    registry = _imports()
    if request.task not in registry.list_tasks():
        raise MjlabError(f"Unknown MJLab task: {request.task}; run mjlab list")
    env = registry.load_env_cfg(request.task, play=play)
    agent = registry.load_rl_cfg(request.task)
    env.seed = agent.seed = request.seed
    agent.logger = "tensorboard"
    agent.upload_model = False
    if request.num_envs is not None:
        env.scene.num_envs = request.num_envs
    if "motion" in env.commands:
        if not inputs.get("input_path"):
            raise MjlabError("Tracking requires an MJLab motion NPZ")
        env.commands["motion"].motion_file = inputs["input_path"]["path"]
    return env, agent


def _train(request, inputs, outputs):
    import torch
    from mjlab.scripts.train import TrainConfig, launch_training

    if not torch.cuda.is_available():
        raise MjlabError("MJLab training requires an NVIDIA CUDA GPU")
    if torch.cuda.device_count() < request.gpu_count:
        raise MjlabError("Requested GPU count exceeds the current node's visible GPUs")
    env, agent = _configs(request, inputs)
    if request.iterations is not None:
        agent.max_iterations = request.iterations
    if request.learning_rate is not None:
        agent.algorithm.learning_rate = request.learning_rate
    if request.checkpoint:
        resume = outputs / agent.experiment_name / "resume"
        resume.mkdir(parents=True)
        shutil.copyfile(inputs["checkpoint"]["path"], resume / "checkpoint.pt")
        agent.resume = True
        agent.load_run, agent.load_checkpoint = "resume", "checkpoint.pt"
    cfg = TrainConfig(
        env=env,
        agent=agent,
        log_root=str(outputs),
        gpu_ids=list(range(request.gpu_count)),
    )
    launch_training(request.task, cfg)
    _publish_checkpoint(outputs)
    if request.checkpoint:
        shutil.rmtree(resume)
    return {
        "status": "completed",
        "iterations": agent.max_iterations,
        "num_envs": env.scene.num_envs,
        "seed": request.seed,
        "gpu_count": request.gpu_count,
    }


def _policy_environment(request, inputs):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env_cfg, agent_cfg = _configs(request, inputs, play=True)
    # Upstream play disables the episode horizon for an interactive viewer.
    # Evaluation restores the training horizon so every measured episode completes.
    train_cfg, _ = _configs(request, inputs)
    env_cfg.episode_length_s = train_cfg.episode_length_s
    env_cfg.terminations = train_cfg.terminations
    env_cfg.scene.num_envs = min(request.num_envs or 1, getattr(request, "episodes", 1))
    raw = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=request.device,
        render_mode="rgb_array" if getattr(request, "video", False) else None,
    )
    try:
        env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(request.task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=request.device)
        runner.load(
            inputs["checkpoint"]["path"],
            load_cfg={"actor": True},
            strict=True,
            map_location=request.device,
        )
        return env, runner
    except BaseException:
        raw.close()
        raise


def _evaluate(request, inputs, outputs):
    from .rollout import measure_episodes
    from .video_report import write_video_report

    env, runner = _policy_environment(request, inputs)
    try:
        policy = runner.get_inference_policy(device=request.device)
        report = measure_episodes(env, policy, request, outputs)
        if request.video:
            report["video_frames"] = _verify_video(outputs / "rollout.mp4")
            write_video_report(outputs, request, report, inputs["checkpoint"]["sha256"])
        (outputs / "episodes.json").write_text(
            json.dumps(report["episodes"], allow_nan=False)
        )
        return report
    finally:
        env.close()


def _verify_video(path):
    import imageio_ffmpeg

    frames = imageio_ffmpeg.read_frames(str(path))
    try:
        metadata = next(frames)
        count = sum(1 for _ in frames)
    finally:
        frames.close()
    if count == 0 or metadata["fps"] <= 0:
        raise MjlabError("Video contains no decodable frames")
    return count


def _export(request, inputs, outputs):
    import onnx
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

    env, runner = _policy_environment(request, inputs)
    try:
        runner.export_policy_to_onnx(str(outputs), "policy.onnx")
        path = outputs / "policy.onnx"
        metadata = {"task": request.task, "mjlab_version": MJLAB_VERSION}
        if (
            "robot" in env.unwrapped.scene.entities
            and "joint_pos" in env.unwrapped.action_manager.active_terms
        ):
            metadata.update(get_base_metadata(env.unwrapped, "npa"))
        attach_metadata_to_onnx(str(path), metadata)
        onnx.checker.check_model(str(path))
        return {"status": "completed", "format": "onnx", "opset": 18}
    finally:
        env.close()


def main() -> None:
    """Run one private worker recipe and persist its result.

    Args:
        None; operation and recipe path come from the parent process argv.
    Returns:
        None.
    Raises:
        MjlabError: Missing runtime, unsupported task or failed execution.
    """
    operation, recipe = sys.argv[1:]
    root = Path(recipe).parent
    payload = json.loads(Path(recipe).read_text())
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    if operation == "list":
        report = {"tasks": _imports().list_tasks(), "version": MJLAB_VERSION}
    else:
        model, execute = {
            "train": (TrainRequest, _train),
            "eval": (EvalRequest, _evaluate),
            "export": (ExportRequest, _export),
        }[operation]
        request = model.model_validate(payload["request"])
        outputs = root / "outputs"
        outputs.mkdir()
        report = execute(request, payload["inputs"], outputs)
        (outputs / "request.json").write_text(request.model_dump_json(indent=2))
    (root / "result.json").write_text(json.dumps(report, allow_nan=False))


def _publish_checkpoint(outputs):
    checkpoints = list(outputs.rglob("model_*.pt"))
    if not checkpoints:
        raise MjlabError("Training produced no native RSL-RL checkpoint")
    latest = max(checkpoints, key=lambda p: int(p.stem.removeprefix("model_")))
    shutil.copyfile(latest, outputs / "checkpoint.pt")


if __name__ == "__main__":
    main()
