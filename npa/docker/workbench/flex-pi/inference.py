#!/usr/bin/env python3
"""Run one real flex-pi action-only inference from a pinned RoboTwin sample."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

import av
from huggingface_hub import hf_hub_download, snapshot_download
import numpy as np
import pandas as pd
import torch


MODELSCOPE_REPOSITORY = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
MODELSCOPE_BRANCH = "master"
MODELSCOPE_GIT_COMMIT = "150f75d811d51f6c7760154aa7fec371dccda529"
MODELSCOPE_GIT_URL = (
    "https://www.modelscope.cn/DiffSynth-Studio/Wan-Series-Converted-Safetensors.git"
)
MODELSCOPE_FILES = {
    "models_t5_umt5-xxl-enc-bf16.safetensors": (
        "d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23"
    ),
    "Wan2.2_VAE.safetensors": (
        "0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996"
    ),
}
UMT5_REPOSITORY = "Wan-AI/Wan2.1-T2V-1.3B"
UMT5_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"
DINO_REPOSITORY = "timm/vit_base_patch16_dinov3.lvd1689m"
DINO_REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
DINO_SHA256 = "1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(entry: dict, dataset: dict) -> Path:
    path = Path(
        hf_hub_download(
            repo_id=dataset["id"],
            repo_type="dataset",
            revision=dataset["revision"],
            filename=entry["path"],
        )
    )
    if _sha256(path) != entry["sha256"]:
        raise RuntimeError("public observation asset hash mismatch")
    return path


def _first_rgb(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0), None)
    if frame is None:
        raise RuntimeError("public observation video has no decodable frame")
    rgb = frame.to_ndarray(format="rgb24")
    if rgb.shape != (240, 320, 3):
        raise RuntimeError(f"unexpected public observation shape: {rgb.shape}")
    return rgb


def _state(path: Path, frame_index: int) -> np.ndarray:
    frame = pd.read_parquet(path)
    row = frame.loc[frame["frame_index"] == frame_index]
    if len(row) != 1:
        raise RuntimeError("public observation frame is not unique")
    state = np.asarray(row.iloc[0]["observation.state"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise RuntimeError("public observation state must be finite 14D")
    return state


def _intrinsics(path: Path, camera: str) -> np.ndarray:
    values = json.loads(path.read_text(encoding="utf-8"))[camera]
    return np.asarray(
        [
            [values["fx"], 0.0, values["cx"]],
            [0.0, values["fy"], values["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _observation(manifest: dict) -> tuple[dict, dict[str, str]]:
    dataset = manifest["dataset"]
    observation = manifest["observation"]
    paths = {
        name: _download(entry, dataset) for name, entry in observation["rgb"].items()
    }
    paths["state"] = _download(observation["state"], dataset)
    paths["intrinsics"] = _download(observation["intrinsics"], dataset)
    cameras = {}
    for model_name, obs_name in {
        "cam_high": "head_camera",
        "cam_left_wrist": "left_camera",
        "cam_right_wrist": "right_camera",
    }.items():
        cameras[obs_name] = {
            "rgb": _first_rgb(paths[model_name]),
            "depth": np.zeros((240, 320), dtype=np.uint16),
            "intrinsic_cv": _intrinsics(paths["intrinsics"], model_name),
        }
    payload = {
        "observation": cameras,
        "joint_action": {"vector": _state(paths["state"], observation["frame_index"])},
    }
    return payload, {name: _sha256(path) for name, path in paths.items()}


def _deploy_module():
    path = Path("/opt/flex-pi/experiments/robotwin/flexpi_policy/deploy_policy.py")
    spec = importlib.util.spec_from_file_location("npa_flexpi_deploy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("pinned flex-pi deploy adapter is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(
        snapshot_download(
            repo_id=args.checkpoint_id,
            revision=args.checkpoint_revision,
            allow_patterns=[
                "checkpoints/weights/*.pt",
                "config.yaml",
                "dataset_stats.json",
            ],
        )
    )
    checkpoints = list((root / "checkpoints" / "weights").glob("*.pt"))
    if len(checkpoints) != 1 or not (root / "dataset_stats.json").is_file():
        raise RuntimeError("checkpoint snapshot is incomplete or ambiguous")
    return checkpoints[0], root / "dataset_stats.json"


def _prepare_runtime_models() -> dict[str, str]:
    """Materialize ancillary weights from immutable revisions and verify bytes."""
    from modelscope import snapshot_download as modelscope_snapshot_download

    # ModelScope's public SDK accepts branch/tag names for this repository but
    # rejects raw Git object IDs.  Bind the supported branch locator to the
    # immutable Git head before asking the SDK to download, then independently
    # verify every selected large file by SHA-256 below.  A branch move or a
    # byte change therefore fails closed before model construction.
    remote = subprocess.run(
        ["git", "ls-remote", MODELSCOPE_GIT_URL, f"refs/heads/{MODELSCOPE_BRANCH}"],
        check=False,
        capture_output=True,
        text=True,
    )
    fields = remote.stdout.strip().split()
    if remote.returncode != 0 or fields[:1] != [MODELSCOPE_GIT_COMMIT]:
        raise RuntimeError("immutable ModelScope branch head verification failed")

    base = Path(os.environ["DIFFSYNTH_MODEL_BASE_PATH"])
    converted = Path(
        modelscope_snapshot_download(
            MODELSCOPE_REPOSITORY,
            revision=MODELSCOPE_BRANCH,
            local_dir=str(base / MODELSCOPE_REPOSITORY),
            allow_file_pattern=list(MODELSCOPE_FILES),
        )
    )
    hashes = {}
    for filename, expected in MODELSCOPE_FILES.items():
        path = converted / filename
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"immutable ancillary model hash mismatch: {filename}")
        hashes[filename] = actual

    tokenizer = Path(
        snapshot_download(
            repo_id=UMT5_REPOSITORY,
            revision=UMT5_REVISION,
            local_dir=str(base / UMT5_REPOSITORY),
            allow_patterns=["google/umt5-xxl/*"],
        )
    )
    tokenizer_files = sorted((tokenizer / "google" / "umt5-xxl").iterdir())
    if not tokenizer_files:
        raise RuntimeError("immutable tokenizer snapshot is empty")
    hashes["tokenizer_tree"] = hashlib.sha256(
        "".join(f"{path.name}:{_sha256(path)}\n" for path in tokenizer_files).encode()
    ).hexdigest()

    dino = Path(
        hf_hub_download(
            repo_id=DINO_REPOSITORY,
            revision=DINO_REVISION,
            filename="model.safetensors",
        )
    )
    if _sha256(dino) != DINO_SHA256:
        raise RuntimeError("immutable DINOv3 model hash mismatch")
    os.environ["FLEX_PI_DINO_CHECKPOINT"] = str(dino)
    hashes["dino_model"] = DINO_SHA256
    return hashes


def _policy(args: argparse.Namespace):
    checkpoint, stats = _checkpoint(args)
    ancillary_hashes = _prepare_runtime_models()
    deploy = _deploy_module()
    policy = deploy.get_model(
        {
            "ckpt_setting": str(checkpoint),
            "dataset_stats_path": str(stats),
            "device": "cuda",
            "mixed_precision": "bf16",
            "num_inference_steps": args.num_inference_steps,
            "seed": args.seed,
            "action_horizon": 32,
            "replan_steps": 32,
            "timing_enabled": True,
            "torch_compile": args.torch_compile,
            "use_per_cam": True,
            "infer_joint_video": False,
            "infer_joint_dino": False,
            "infer_joint_pointmap": False,
            "infer_present_video": True,
            "infer_present_dino": True,
            "infer_present_pointmap": False,
        }
    )
    return policy, checkpoint, stats, ancillary_hashes


def _run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("flex-pi acceptance requires exactly one CUDA GPU")
    manifest = json.loads(Path(args.input_manifest).read_text(encoding="utf-8"))
    observation, input_hashes = _observation(manifest)
    policy, checkpoint, stats, ancillary_hashes = _policy(args)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    actions = policy._infer_action_chunk(
        observation, manifest["observation"]["instruction"]
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (32, 14) or not np.isfinite(actions).all():
        raise RuntimeError(
            "upstream policy did not produce a finite 32x14 action chunk"
        )
    return {
        "schema": "npa.flex_pi.actions.v1",
        "regime": "action-only",
        "actions": actions.tolist(),
        "metrics": {
            "inference_seconds": elapsed,
            "action_l2_mean": float(np.linalg.norm(actions, axis=1).mean()),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "runtime": {
            "cuda": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(
                str(v) for v in torch.cuda.get_device_capability(0)
            ),
            "torch": torch.__version__,
            "source_revision": __import__("os").environ.get(
                "FLEX_PI_SOURCE_REVISION", ""
            ),
        },
        "provenance": {
            "checkpoint_sha256": _sha256(checkpoint),
            "dataset_stats_sha256": _sha256(stats),
            "ancillary_sha256": ancillary_hashes,
            "input_sha256": input_hashes,
            "dataset_revision": manifest["dataset"]["revision"],
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--torch-compile", action="store_true")
    args = parser.parse_args()
    payload = _run(args)
    if not math.isfinite(payload["metrics"]["action_l2_mean"]):
        raise RuntimeError("invalid action magnitude metric")
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("FLEX_PI_REAL_INFERENCE_PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
