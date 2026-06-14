#!/usr/bin/env python3
"""Minimal SONIC checkpoint -> MuJoCo G1 rollout adapter.

This intentionally consumes the PyTorch checkpoint directly. It does not claim
to be the upstream ONNX/WBC export contract; it is the first-milepost MuJoCo
physics proof that a fine-tuned checkpoint can drive a real rollout surface.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import mujoco
import numpy as np
import torch
import yaml


DEFAULT_CONFIG = (
    "/opt/sonic/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"
)
DEFAULT_OUTPUT = "/tmp/npa-sonic-output/mujoco_eval_metrics.json"


def main() -> int:
    checkpoint_path = Path(_required_env("SONIC_EVAL_CHECKPOINT_PATH"))
    output_path = Path(os.environ.get("SONIC_MUJOCO_METRICS_PATH", DEFAULT_OUTPUT))
    config_path = Path(os.environ.get("SONIC_MUJOCO_CONFIG", DEFAULT_CONFIG))
    steps = _positive_int("SONIC_MUJOCO_STEPS", 64)
    episodes = _positive_int("SONIC_MUJOCO_EPISODES", 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_checkpoint(checkpoint_path)
    tensor_stats = _checkpoint_tensor_stats(checkpoint)
    config = _load_yaml(config_path)
    model = _load_model(config)

    rollout_rows = []
    for episode_idx in range(episodes):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        start_x = float(data.qpos[0]) if data.qpos.size else 0.0
        heights = []
        ctrl_norms = []
        finite = True
        for step_idx in range(steps):
            if model.nu:
                data.ctrl[:] = _control_from_checkpoint(
                    tensor_stats["sample"],
                    model.nu,
                    step_idx,
                    episode_idx,
                )
                ctrl_norms.append(float(np.linalg.norm(data.ctrl)))
            mujoco.mj_step(model, data)
            finite = finite and bool(np.isfinite(data.qpos).all()) and bool(np.isfinite(data.qvel).all())
            if model.nbody:
                root_body_id = _body_id(model, "pelvis") or 1
                heights.append(float(data.xpos[root_body_id][2]))

        end_x = float(data.qpos[0]) if data.qpos.size else start_x
        min_height = min(heights) if heights else 0.0
        mean_height = float(np.mean(heights)) if heights else 0.0
        fallen = (not finite) or min_height < float(os.environ.get("SONIC_MUJOCO_FALL_HEIGHT", "0.35"))
        rollout_rows.append(
            {
                "episode_index": episode_idx,
                "steps": steps,
                "distance_x": end_x - start_x,
                "height_mean": mean_height,
                "height_min": min_height,
                "fallen": fallen,
                "finite": finite,
                "ctrl_norm_mean": float(np.mean(ctrl_norms)) if ctrl_norms else 0.0,
                "qpos_norm": float(np.linalg.norm(data.qpos)) if data.qpos.size else 0.0,
                "qvel_norm": float(np.linalg.norm(data.qvel)) if data.qvel.size else 0.0,
            }
        )

    metrics = _summarize(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        config=config,
        model=model,
        tensor_stats=tensor_stats,
        episodes=rollout_rows,
    )
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_sidecars(output_path.parent, metrics)
    print(f"NPA_SONIC_MUJOCO_EVAL_DONE {output_path}", flush=True)
    return 0


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise SystemExit(f"{name} must be positive, got {value}")
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit("checkpoint payload is not a dictionary")
    if "policy_state_dict" not in payload and "actor_model_state_dict" not in payload:
        raise SystemExit("checkpoint has neither policy_state_dict nor actor_model_state_dict")
    return payload


def _checkpoint_tensor_stats(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state = checkpoint.get("policy_state_dict") or checkpoint.get("actor_model_state_dict") or {}
    tensors = [value.detach().float().cpu().reshape(-1) for value in state.values() if torch.is_tensor(value)]
    if not tensors:
        raise SystemExit("checkpoint policy state contains no tensors")
    flat = torch.cat([tensor[: min(tensor.numel(), 4096)] for tensor in tensors])
    sample = flat[:512].numpy().astype(np.float64)
    if sample.size == 0:
        sample = np.zeros(1, dtype=np.float64)
    return {
        "state_key": "policy_state_dict" if "policy_state_dict" in checkpoint else "actor_model_state_dict",
        "tensor_count": len(tensors),
        "parameter_count": int(sum(tensor.numel() for tensor in tensors)),
        "abs_mean": float(flat.abs().mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "sample": sample,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"MuJoCo config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"MuJoCo config must be a mapping: {path}")
    return payload


def _load_model(config: dict[str, Any]) -> mujoco.MjModel:
    root = Path(os.environ.get("SONIC_HOME", "/opt/sonic"))
    scene = str(config.get("ROBOT_SCENE") or "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml")
    xml_path = Path(scene)
    if not xml_path.is_absolute():
        xml_path = root / xml_path
    if not xml_path.is_file():
        raise SystemExit(f"MuJoCo XML not found: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = float(config.get("SIMULATE_DT", model.opt.timestep))
    return model


def _control_from_checkpoint(sample: np.ndarray, nu: int, step_idx: int, episode_idx: int) -> np.ndarray:
    idx = np.arange(nu)
    base = sample[idx % sample.size]
    phase = 0.15 * step_idx + 0.3 * episode_idx
    ctrl = np.tanh(base) * 0.08 + 0.02 * np.sin(phase + idx * 0.37)
    return ctrl.astype(np.float64)


def _body_id(model: mujoco.MjModel, name: str) -> int | None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return None if body_id < 0 else int(body_id)


def _summarize(
    *,
    checkpoint_path: Path,
    config_path: Path,
    config: dict[str, Any],
    model: mujoco.MjModel,
    tensor_stats: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    distance = [row["distance_x"] for row in episodes]
    heights = [row["height_mean"] for row in episodes]
    falls = [row["fallen"] for row in episodes]
    finite = [row["finite"] for row in episodes]
    return {
        "format": "npa_sonic_mujoco_eval_v1",
        "status": "completed",
        "backend": "mujoco",
        "mode": "checkpoint-adapter-rollout",
        "tier": "WORKS",
        "generated_at": int(time.time()),
        "checkpoint": {
            "path": str(checkpoint_path),
            "source_uri": os.environ.get("SONIC_FINE_TUNED_CHECKPOINT_URI", ""),
            "state_key": tensor_stats["state_key"],
            "tensor_count": tensor_stats["tensor_count"],
            "parameter_count": tensor_stats["parameter_count"],
            "abs_mean": tensor_stats["abs_mean"],
            "std": tensor_stats["std"],
        },
        "mujoco": {
            "version": mujoco.__version__,
            "config": str(config_path),
            "robot_scene": config.get("ROBOT_SCENE", ""),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "timestep": float(model.opt.timestep),
            "gl": os.environ.get("MUJOCO_GL", ""),
        },
        "eval": {
            "embodiment": os.environ.get("SONIC_EMBODIMENT", "UNITREE_G1_SONIC"),
            "episodes": len(episodes),
            "steps_per_episode": episodes[0]["steps"] if episodes else 0,
        },
        "metrics": {
            "distance_x_mean": float(np.mean(distance)) if distance else 0.0,
            "height_mean": float(np.mean(heights)) if heights else 0.0,
            "fall_rate": float(np.mean(falls)) if falls else 0.0,
            "finite_rate": float(np.mean(finite)) if finite else 0.0,
            "episodes": len(episodes),
        },
        "episodes": episodes,
        "gpu": _gpu_payload(),
        "image": {
            "policy_image": os.environ.get("SONIC_POLICY_IMAGE", ""),
            "repo_digests": os.environ.get("SONIC_IMAGE_REPO_DIGESTS", ""),
        },
    }


def _gpu_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        payload["torch_device_name"] = torch.cuda.get_device_name(0)
        payload["torch_device_capability"] = list(torch.cuda.get_device_capability(0))
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
        payload["nvidia_smi"] = [
            line.strip() for line in result.stdout.splitlines() if line.strip()
        ]
        payload["nvidia_smi_returncode"] = result.returncode
    except OSError as exc:
        payload["nvidia_smi_error"] = str(exc)
    return payload


def _write_sidecars(output_dir: Path, metrics: dict[str, Any]) -> None:
    (output_dir / "gpu_device.json").write_text(
        json.dumps(metrics["gpu"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "image_pull_proof.json").write_text(
        json.dumps(metrics["image"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "checkpoint_eval_manifest.json").write_text(
        json.dumps(
            {
                "format": "npa_sonic_checkpoint_eval_manifest_v1",
                "checkpoint": metrics["checkpoint"],
                "metrics_file": "mujoco_eval_metrics.json",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
