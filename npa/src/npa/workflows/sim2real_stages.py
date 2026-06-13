"""Mandatory sim2real preamble stages: Cosmos augment, 10K envgen split, policy rollouts."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, TYPE_CHECKING

from npa.workflows.cosmos_split import Cosmos2TransferConfig, build_cosmos2_transfer_manifest
from npa.workflows.sim2real_envgen import (
    EnvGenConfig,
    Sim2RealEnvGenError,
    build_scene_spec,
    write_action_conditioned_envs,
    write_raw_shard,
    write_split_manifest,
)

if TYPE_CHECKING:
    from npa.workflows.sim2real_loop import Sim2RealLoopConfig

DEFAULT_ENV_COUNT = 10_000
DEFAULT_TRAIN_FRACTION = 0.8


def effective_env_count(config: Sim2RealLoopConfig) -> int:
    """Production default is 10K; unit tests pass env_count=0 for legacy sizing."""

    if config.env_count > 0:
        return config.env_count
    return config.rollout_count + config.heldout_env_count


def effective_train_count(config: Sim2RealLoopConfig) -> int:
    if config.env_count > 0:
        return int(round(effective_env_count(config) * config.train_fraction))
    return config.rollout_count


def effective_heldout_count(config: Sim2RealLoopConfig) -> int:
    if config.env_count > 0:
        total = effective_env_count(config)
        return total - effective_train_count(config)
    return config.heldout_env_count


def artifact_output_uri(config: Sim2RealLoopConfig) -> str:
    if not config.s3_bucket:
        raise Sim2RealLoopError("s3_bucket is required for production stage execution")
    prefix = config.s3_prefix.strip("/") or "sim2real-b"
    return f"s3://{config.s3_bucket}/{prefix}/{config.run_id}"


class Sim2RealStageError(RuntimeError):
    """Raised when a mandatory workflow stage fails."""


# Alias for callers that import from this module only.
Sim2RealLoopError = Sim2RealStageError


def run_augment_stage(config: Sim2RealLoopConfig, local_dir: Path) -> dict[str, Any]:
    """Stage 3: run Cosmos Transfer 2.5 (K8s sibling job when bucket set, else local reference)."""

    augment_dir = local_dir / "augment"
    augment_dir.mkdir(parents=True, exist_ok=True)
    input_uri = (config.trigger_dataset_uri or "").strip()
    if not input_uri:
        input_uri = f"local://{local_dir / 'stage_01_trigger' / 'trigger.json'}"
    if config.s3_bucket:
        output_uri = f"{artifact_output_uri(config)}/augment/"
        from npa.workflows import sim2real_loop as loop

        result = loop.run_cosmos2_transfer_component(
            config,
            input_uri=input_uri,
            output_uri=output_uri,
            local_dir=augment_dir,
        )
        manifest = result["manifest"]
        augmented_frames_uri = result["augmented_frames_uri"]
        tier = "WORKS"
        evidence = "Executed Cosmos Transfer 2.5 via sibling Kubernetes job."
    else:
        manifest, augmented_frames_uri = _reference_augment_local(
            config, local_dir, input_uri=input_uri
        )
        tier = "WORKS"
        evidence = "Executed reference Cosmos Transfer augmentation locally (no s3_bucket)."
    _write_json(augment_dir / "manifest.json", manifest)
    return {
        "manifest": manifest,
        "augmented_frames_uri": augmented_frames_uri,
        "component": {
            "name": "stage_03_augment",
            "tier": tier,
            "evidence": evidence,
            "artifacts": {"local": str(augment_dir / "manifest.json")},
        },
    }


def run_envgen_split_stage(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    augmented_frames_uri: str,
) -> dict[str, Any]:
    """Stages 4–6: generate raw envs, 80/20 split, token manifest."""

    env_count = effective_env_count(config)
    train_count = effective_train_count(config)
    heldout_count = effective_heldout_count(config)
    if train_count + heldout_count != env_count:
        raise Sim2RealStageError("train + heldout counts must equal env_count")

    scene = build_scene_spec(
        byo_mesh_uri=config.assets_uri,
        augmented_frames_uri=augmented_frames_uri,
        notes=[f"robot_preset={config.robot_preset or 'franka-stock'}"],
    )
    env_root = local_dir / "envs"
    env_root.mkdir(parents=True, exist_ok=True)

    if config.s3_bucket:
        output_uri = artifact_output_uri(config)
        envgen = EnvGenConfig(
            run_id=config.run_id,
            output_uri=output_uri,
            env_count=env_count,
            train_fraction=config.train_fraction,
            seed=config.seed,
            shard_index=0,
            shard_count=1,
            scene_spec=scene,
        )
        with tempfile.TemporaryDirectory(prefix="npa-envgen-") as tmp:
            tmp_path = Path(tmp)
            write_raw_shard(envgen, tmp_path)
            split = write_split_manifest(envgen, tmp_path / "split")
        train_envs_uri = split["uploaded_train"]
        heldout_envs_uri = split["uploaded_heldout"]
        split_manifest_uri = split["uploaded_manifest"]
        _mirror_env_manifests(config, local_dir, envgen, split)
        tier = "WORKS"
        evidence = (
            f"Generated {env_count} raw envs with {train_count}/{heldout_count} "
            "train/heldout split via sim2real_envgen on S3."
        )
    else:
        from npa.workflows.sim2real_loop import (
            _write_env_manifest,
            _write_train_heldout_split,
        )

        raw = _write_env_manifest(
            env_root / "raw",
            count=env_count,
            seed=config.seed,
        )
        train, heldout = _write_train_heldout_split(
            env_root,
            raw_envs=raw,
            train_count=train_count,
            heldout_count=heldout_count,
            seed=config.seed,
        )
        train_envs_uri = str(env_root / "train" / "manifest.json")
        heldout_envs_uri = str(env_root / "heldout" / "manifest.json")
        split_manifest_uri = ""
        tier = "WORKS"
        evidence = f"Generated {env_count} local reference env manifests with 80/20 split."
        _write_json(
            local_dir / "tokens" / "manifest.json",
            {
                "schema": "npa.sim2real.tokens.v1",
                "stage": 6,
                "train_env_count": train_count,
                "heldout_env_count": heldout_count,
                "status": "ready",
            },
        )

    return {
        "env_count": env_count,
        "train_count": train_count,
        "heldout_count": heldout_count,
        "train_envs_uri": train_envs_uri,
        "heldout_envs_uri": heldout_envs_uri,
        "split_manifest_uri": split_manifest_uri,
        "component": {
            "name": "stage_04_06_env_gen_split_tokens",
            "tier": tier,
            "evidence": evidence,
            "artifacts": {
                "train_envs": train_envs_uri,
                "heldout_envs": heldout_envs_uri,
            },
        },
    }


def run_policy_rollouts(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    actions_dir: Path,
    outer_iteration: int,
    iteration: int,
) -> list[Path]:
    """Stage 7: swappable LeRobot/Cortex policy container or local reference rollouts."""

    from npa.workflows.sim2real_loop import generate_action_rollouts

    train_uri = (config.train_envs_uri or "").strip()
    if config.s3_bucket and train_uri.startswith("s3://"):
        from npa.workflows import sim2real_loop as loop

        return loop.run_policy_rollout_component(
            config,
            local_dir=local_dir,
            actions_dir=actions_dir,
            outer_iteration=outer_iteration,
            iteration=iteration,
            train_envs_uri=train_uri,
        )
    return generate_action_rollouts(
        actions_dir,
        count=config.rollout_count,
        steps_per_rollout=config.steps_per_rollout,
        seed=config.seed + outer_iteration * 100 + iteration,
        quality=0.5,
    )


def _reference_augment_local(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    input_uri: str,
) -> tuple[dict[str, Any], str]:
    frames_dir = local_dir / "augment" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    frame_count = min(1024, max(16, config.rollout_count * 4))
    index: list[dict[str, Any]] = []
    for index_no in range(frame_count):
        frame_path = frames_dir / f"frame-{index_no:05d}.json"
        payload = {
            "schema": "npa.sim2real.augmented_frame.v1",
            "frame_id": f"frame-{index_no:05d}",
            "source_dataset_uri": input_uri,
            "perturbation": rng.choice(
                ["lighting", "texture", "background", "contrast"]
            ),
            "status": "reference_augmented",
        }
        _write_json(frame_path, payload)
        index.append({"frame_id": payload["frame_id"], "local": str(frame_path)})
    _write_json(
        frames_dir / "index.json",
        {
            "schema": "npa.sim2real.augmented_frames.v1",
            "frame_count": frame_count,
            "frames": index,
        },
    )
    output_uri = str(frames_dir)
    if config.s3_bucket:
        from npa.clients.storage import StorageClient

        client = StorageClient.from_environment()
        root = f"{artifact_output_uri(config)}/augment/frames/"
        for item in index:
            client.upload_file(item["local"], f"{root}{Path(item['local']).name}")
        client.upload_file(
            str(frames_dir / "index.json"), f"{root}index.json"
        )
        output_uri = root
    manifest = build_cosmos2_transfer_manifest(
        Cosmos2TransferConfig(
            input_uri=input_uri,
            output_uri=output_uri,
            assets_uri=config.assets_uri,
            scene_spec_uri=config.scene_spec_uri,
            image=config.augment_image,
            run_id=config.run_id,
        )
    )
    manifest["status"] = "executed_reference"
    manifest["augmented_frames_uri"] = output_uri
    manifest["frame_count"] = frame_count
    return manifest, output_uri


def _mirror_env_manifests(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    envgen: EnvGenConfig,
    split: dict[str, Any],
) -> None:
    from npa.clients.storage import StorageClient

    client = StorageClient.from_environment()
    env_root = local_dir / "envs"
    env_root.mkdir(parents=True, exist_ok=True)
    for sub, uri_key in (
        ("train", "uploaded_train"),
        ("heldout", "uploaded_heldout"),
    ):
        target = env_root / sub / "envs.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_path(split[uri_key], str(target))
    _write_json(
        env_root / "split-manifest.json",
        {
            "schema": "npa.sim2real.split_manifest.v1",
            "run_id": config.run_id,
            "train_count": split["train_count"],
            "heldout_count": split["heldout_count"],
            "raw_count": split["raw_count"],
            "train_uri": split["train_uri"],
            "heldout_uri": split["heldout_uri"],
            "remote_manifest": split.get("uploaded_manifest", ""),
        },
    )
    _write_json(
        local_dir / "tokens" / "manifest.json",
        {
            "schema": "npa.sim2real.tokens.v1",
            "stage": 6,
            "train_env_count": split["train_count"],
            "heldout_env_count": split["heldout_count"],
            "status": "ready",
        },
    )
    # Persist URIs on config object via caller (mutable dataclass? frozen - return values)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
