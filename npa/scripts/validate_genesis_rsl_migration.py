"""Validate real Genesis GPU learning, restricted loading, ONNX parity and cameras.

Run in a GPU environment with the candidate NPA source and Genesis dependencies.
This is a framework compatibility check, not a trained-policy quality benchmark.
Each stage uses a fresh process so Genesis owns its simulation lifetime.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def _write(output: Path, stage: str, values: dict) -> None:
    (output / f"{stage}.json").write_text(json.dumps(values, indent=2) + "\n")


def _runtime_metadata(torch) -> dict:
    return {
        "genesis": version("genesis-world"),
        "rsl_rl": version("rsl-rl-lib"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


def _train(output: Path) -> None:
    import torch
    from npa.genesis.train_teacher import PPOConfig, train_teacher

    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires a real CUDA GPU")
    if version("rsl-rl-lib") != "5.5.1":
        raise RuntimeError("The candidate RSL-RL version is not installed")
    started = time.monotonic()
    result = train_teacher(
        n_envs=4,
        max_iterations=2,
        output_dir=output / "teacher",
        log_dir=output / "training-logs",
        ppo_cfg=PPOConfig(
            actor_hidden_dims=[32, 16],
            critic_hidden_dims=[32, 16],
            num_steps_per_env=4,
            num_learning_epochs=1,
            num_mini_batches=2,
            empirical_normalization=True,
        ),
        env_overrides={"max_episode_steps": 16},
    )
    if result["status"] != "success":
        raise RuntimeError("Real teacher training did not succeed")
    _write(
        output,
        "train",
        {
            "environments": 4,
            "iterations": 2,
            "seconds": round(time.monotonic() - started, 3),
            "checkpoint_bytes": (output / "teacher/model.pt").stat().st_size,
            **_runtime_metadata(torch),
        },
    )


def _reference_actions(checkpoint: Path, architecture: dict, observations):
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    config = architecture["actor"]
    grouped = TensorDict({"policy": observations}, batch_size=[len(observations)])
    actor = (
        MLPModel(
            grouped,
            {"actor": ["policy"]},
            "actor",
            architecture["num_actions"],
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
            obs_normalization=config["obs_normalization"],
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
        )
        .to("cuda")
        .eval()
    )
    tensors = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor.load_state_dict(tensors["actor_state_dict"], strict=True)
    with torch.inference_mode():
        return actor(grouped)


def _export(output: Path) -> None:
    from types import SimpleNamespace
    import numpy as np
    import onnxruntime
    import torch
    from npa.genesis.generate_demos import _load_teacher_policy
    from npa.workflows.sim2real.policy_export import export_policy_onnx

    checkpoint = output / "teacher/model.pt"
    architecture = json.loads((output / "teacher/arch_config.json").read_text())
    env = SimpleNamespace(
        obs_dim=architecture["num_obs"],
        act_dim=architecture["num_actions"],
        device="cuda",
    )
    torch.manual_seed(17)
    observations = torch.randn(4, env.obs_dim, device="cuda")
    expected = _reference_actions(checkpoint, architecture, observations)
    with torch.inference_mode():
        actual = _load_teacher_policy(checkpoint, env).act_inference(observations)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    exported = export_policy_onnx(checkpoint, out_dir=str(output / "export"))
    session = onnxruntime.InferenceSession(
        exported["onnx_path"], providers=["CPUExecutionProvider"]
    )
    observed = session.run(None, {"obs": observations.cpu().numpy()})[0]
    reference = expected.cpu().numpy()
    np.testing.assert_allclose(observed, reference, rtol=1e-5, atol=1e-6)
    _write(
        output,
        "export",
        {
            "observation_width": env.obs_dim,
            "action_width": env.act_dim,
            "samples": len(observations),
            "teacher_max_absolute_error": float((actual - expected).abs().max()),
            "onnx_max_absolute_error": float(np.abs(observed - reference).max()),
        },
    )


def _camera_arrays(episode: Path) -> tuple[int, dict]:
    import numpy as np

    arrays = {
        name: np.load(episode / f"{name}.npy", allow_pickle=False)
        for name in ("obs_workspace", "obs_wrist", "state", "actions")
    }
    frames = len(arrays["actions"])
    if frames < 2 or any(len(value) != frames for value in arrays.values()):
        raise RuntimeError("Camera, state and action frame counts differ")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("The demonstration contains non-finite values")
    for name in ("obs_workspace", "obs_wrist"):
        pixels = arrays[name]
        if pixels.ndim != 4 or pixels.shape[-1] != 3 or np.ptp(pixels) == 0:
            raise RuntimeError("The camera did not produce RGB scene pixels")
    return frames, {name: list(value.shape) for name, value in arrays.items()}


def _demos(output: Path) -> None:
    from npa.genesis.generate_demos import generate_demos

    result = generate_demos(
        checkpoint_path=output / "teacher/model.pt",
        n_envs=1,
        n_episodes=0,
        output_dir=output / "demos",
        domain_randomize=False,
        allow_failure_demos=True,
    )
    episodes = sorted((output / "demos").glob("episode_*"))
    if not episodes or result["total_episodes"] != len(episodes):
        raise RuntimeError("No actual camera episode was produced")
    frames, shapes = _camera_arrays(episodes[0])
    architecture = json.loads((output / "teacher/arch_config.json").read_text())
    if shapes["actions"][-1] != architecture["num_actions"]:
        raise RuntimeError("Camera demonstration action width differs from teacher")
    _write(
        output,
        "demos",
        {
            "episodes": len(episodes),
            "frames": frames,
            "array_shapes": shapes,
            "teacher_success_rate": result["teacher_success_rate"],
            "includes_failure_demonstrations": result["includes_failures"],
            "scope": "camera and checkpoint compatibility; not policy quality",
        },
    )


def _run(output: Path) -> None:
    source = os.environ.get("NPA_VALIDATION_SOURCE_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("Set NPA_VALIDATION_SOURCE_SHA to the exact candidate commit")
    for stage in ("train", "export", "demos"):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--stage",
                stage,
                "--output-path",
                str(output),
            ],
            check=True,
        )
    report = {
        "schema": "npa.genesis.rsl-migration.v1",
        "source_sha": source,
        "status": "passed",
    }
    report.update(
        {
            stage: json.loads((output / f"{stage}.json").read_text())
            for stage in ("train", "export", "demos")
        }
    )
    _write(output, "report", report)
    print(json.dumps(report, sort_keys=True))


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("all", "train", "export", "demos"), default="all"
    )
    args = parser.parse_args()
    output = args.output_path.resolve()
    output.mkdir(parents=True, exist_ok=True)
    {"all": _run, "train": _train, "export": _export, "demos": _demos}[args.stage](
        output
    )


if __name__ == "__main__":
    _main()
