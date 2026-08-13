"""Mandatory sim2real preamble stages: Cosmos augment, 10K envgen split, policy rollouts."""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from npa.workflows.cosmos_split import (
    Cosmos2TransferConfig,
    build_cosmos2_transfer_manifest,
)
from npa.workflows.sim2real_envgen import (
    EnvGenConfig,
    load_raw_shards,
    write_raw_shard,
    write_split_manifest,
)

if TYPE_CHECKING:
    from npa.workflows.sim2real.models import Sim2RealLoopConfig

DEFAULT_ENV_COUNT = 10_000
DEFAULT_TRAIN_FRACTION = 0.8


def resolve_augment_frame_count(*, rollout_count: int = 0, override: int = 0) -> int:
    """Scale augment frames with rollout count; cap at 1024 for production runs."""

    if override > 0:
        return min(1024, override)
    env_override = int(os.environ.get("NPA_SIM2REAL_AUGMENT_FRAME_COUNT", "0") or "0")
    if env_override > 0:
        return min(1024, env_override)
    rollout = rollout_count or int(
        os.environ.get("NPA_SIM2REAL_ROLLOUT_COUNT", "0") or "0"
    )
    if rollout > 0:
        return min(1024, max(16, rollout * 4))
    return 1024


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


def k8s_image_ready(image: str) -> bool:
    """Return true when an image reference is registry-qualified (not a placeholder)."""

    from npa.guardrails.skypilot import unresolved_image_placeholders
    from npa.workflows.sim2real_health import _looks_registry_qualified

    ref = str(image or "").strip()
    return (
        bool(ref)
        and _looks_registry_qualified(ref)
        and not unresolved_image_placeholders(ref)
    )


def _gpu_invocation_artifacts(invocation: dict[str, Any]) -> dict[str, Any]:
    """Copy the complete public placement proof into a ComponentRecord."""

    provenance = dict(invocation.get("gpu_provenance") or {})
    return {
        "job_name": str(invocation.get("job_name") or provenance.get("job_name") or ""),
        "image": str(invocation.get("image") or provenance.get("image") or ""),
        "image_digests": invocation.get("image_digests", [])
        or provenance.get("image_digests", []),
        "gpu_request": invocation.get("gpu_request", {}),
        "gpu_candidate_order": provenance.get("candidate_order", []),
        "gpu_attempts": provenance.get("attempts", []),
        "selected_gpu_product": str(provenance.get("selected_product") or ""),
        "selected_gpu_node": str(provenance.get("selected_node") or ""),
        "allocated_gpu": provenance.get("allocated_gpu", {}),
        "duration_s": provenance.get("duration_s", ""),
    }


def run_augment_stage(config: Sim2RealLoopConfig, local_dir: Path) -> dict[str, Any]:
    """Stage 3: run Cosmos Transfer 2.5 (K8s sibling job when bucket set, else local reference)."""

    augment_dir = local_dir / "augment"
    augment_dir.mkdir(parents=True, exist_ok=True)
    input_uri = (config.trigger_dataset_uri or "").strip()
    if not input_uri:
        input_uri = f"local://{local_dir / 'stage_01_trigger' / 'trigger.json'}"
    invocation: dict[str, Any] = {}
    if config.s3_bucket and k8s_image_ready(config.augment_image):
        output_uri = f"{artifact_output_uri(config)}/augment/"
        from npa.workflows.sim2real.engine import run_cosmos2_transfer_component

        result = run_cosmos2_transfer_component(
            config,
            input_uri=input_uri,
            output_uri=output_uri,
            local_dir=augment_dir,
        )
        manifest = result["manifest"]
        mode = str(manifest.get("mode") or "").strip()
        if mode not in {"cosmos_transfer2.5", "cosmos_transfer2.5_gpu"}:
            raise Sim2RealStageError(
                "registry-qualified Cosmos Transfer stage did not emit real GPU "
                f"provenance (mode={mode or 'missing'})"
            )
        augmented_frames_uri = result["augmented_frames_uri"]
        invocation = result.get("invocation") or {}
        mirrored = _mirror_augment_frames(augmented_frames_uri, augment_dir)
        tier = "WORKS"
        evidence = (
            "Executed real Cosmos Transfer 2.5 via sibling Kubernetes GPU Job and mirrored generated frames locally."
            if mirrored
            else (
                "Executed real Cosmos Transfer 2.5 via sibling Kubernetes GPU Job; local frame mirror was unavailable, "
                "so final visualization falls back to manifest descriptors."
            )
        )
    else:
        manifest, augmented_frames_uri = _reference_augment_local(
            config, local_dir, input_uri=input_uri
        )
        if config.s3_bucket:
            tier = "SEAM"
            evidence = (
                "Augment image is an operator placeholder or bare tag; executed reference "
                "Cosmos Transfer locally until AUGMENT_IMAGE is registry-qualified."
            )
        else:
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
            "artifacts": {
                "local": str(augment_dir / "manifest.json"),
                "remote": f"{artifact_output_uri(config)}/augment/manifest.json"
                if config.s3_bucket
                else "",
                **_gpu_invocation_artifacts(invocation),
            },
        },
    }


def run_envgen_split_stage(
    config: Sim2RealLoopConfig,
    local_dir: Path,
    *,
    augmented_frames_uri: str,
    scene_spec_uri: str = "",
    robot_spec_uri: str = "",
) -> dict[str, Any]:
    """Stages 4–6: generate, curate, and split train/validation/gold scenarios."""

    stage_group_started = time.monotonic()
    env_count = effective_env_count(config)
    train_count = effective_train_count(config)
    non_train_count = effective_heldout_count(config)
    if train_count + non_train_count != env_count:
        raise Sim2RealStageError("train + heldout counts must equal env_count")

    from npa.workflows.sim2real_assets import build_envgen_scene_spec

    scene = build_envgen_scene_spec(
        config,
        scene_spec_uri=scene_spec_uri
        or str(local_dir / "stage_02_assets" / "consumed_scene_spec.json"),
        robot_spec_uri=robot_spec_uri
        or str(local_dir / "stage_02_assets" / "consumed_robot_spec.json"),
        augmented_frames_uri=augmented_frames_uri,
    )
    env_root = local_dir / "envs"
    env_root.mkdir(parents=True, exist_ok=True)

    if config.s3_bucket:
        output_uri = artifact_output_uri(config)
        shard_count = max(1, int(config.envgen_shard_count))
        envgen = EnvGenConfig(
            run_id=config.run_id,
            output_uri=output_uri,
            env_count=env_count,
            train_fraction=config.train_fraction,
            seed=config.seed,
            shard_index=0,
            shard_count=shard_count,
            scene_spec=scene,
        )
        split_envgen = envgen
        envgen_invocation: dict[str, Any] = {}
        if k8s_image_ready(config.envgen_image) and shard_count > 1:
            from npa.workflows.sim2real.engine import run_envgen_sharded_component

            envgen_result = run_envgen_sharded_component(config, envgen=envgen)
            envgen_invocation = dict(envgen_result.get("invocation") or {})
            tier = "WORKS"
            evidence = (
                f"Generated {env_count} raw envs across {shard_count} indexed GPU "
                f"shards (parallelism capped at {min(shard_count, config.k8s_max_parallel_gpus)}) "
                f"with {train_count}/{non_train_count} train/non-train partition via "
                "sim2real_envgen on S3."
            )
        else:
            envgen_single = EnvGenConfig(
                run_id=config.run_id,
                output_uri=output_uri,
                env_count=env_count,
                train_fraction=config.train_fraction,
                seed=config.seed,
                shard_index=0,
                shard_count=1,
                scene_spec=scene,
            )
            split_envgen = envgen_single
            with tempfile.TemporaryDirectory(prefix="npa-envgen-") as tmp:
                tmp_path = Path(tmp)
                write_raw_shard(envgen_single, tmp_path)
            tier = "WORKS" if k8s_image_ready(config.envgen_image) else "SEAM"
            if tier == "SEAM":
                evidence = (
                    f"Generated {env_count} raw envs with {train_count}/{non_train_count} "
                    "train/non-train partition via orchestrator in-process envgen because "
                    "ENVGEN_IMAGE is not registry-qualified."
                )
            else:
                evidence = (
                    f"Generated {env_count} raw envs with {train_count}/{non_train_count} "
                    "train/non-train partition via orchestrator in-process envgen (single shard)."
                )
        with tempfile.TemporaryDirectory(prefix="npa-envgen-split-") as tmp:
            tmp_path = Path(tmp)
            raw_rows, raw_proof = load_raw_shards(
                split_envgen, tmp_path / "stage-04-raw"
            )
            split = write_split_manifest(
                split_envgen,
                tmp_path / "split",
                raw_envs=raw_rows,
                raw_input_proof=raw_proof,
            )
        train_count = int(split["train_count"])
        validation_count = int(split["validation_count"])
        gold_heldout_count = int(split["gold_heldout_count"])
        heldout_count = int(split["heldout_count"])
        if train_count + validation_count + gold_heldout_count != env_count:
            raise Sim2RealStageError(
                "curated train + validation + gold-heldout counts must equal env_count"
            )
        if validation_count + gold_heldout_count != non_train_count:
            raise Sim2RealStageError(
                "validation + gold-heldout counts must equal the configured non-train count"
            )
        train_envs_uri = split["uploaded_train"]
        validation_envs_uri = split["uploaded_validation"]
        heldout_envs_uri = split["uploaded_heldout"]
        gold_heldout_envs_uri = split["uploaded_gold_heldout"]
        split_manifest_uri = split["uploaded_manifest"]
        curation_manifest_uri = split["uploaded_curation"]
        _mirror_env_manifests(config, local_dir, envgen, split)
    else:
        from npa.workflows.sim2real.engine import (
            _write_env_manifest,
            _write_train_heldout_split,
        )

        heldout_count = non_train_count
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
        validation_envs_uri = heldout_envs_uri
        gold_heldout_envs_uri = heldout_envs_uri
        validation_count = heldout_count
        gold_heldout_count = heldout_count
        split_manifest_uri = ""
        curation_manifest_uri = ""
        tier = "WORKS"
        evidence = (
            f"Generated {env_count} local reference env manifests with 80/20 split."
        )
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
        envgen_invocation = {}

    stage_04_job_artifacts = _gpu_invocation_artifacts(envgen_invocation)
    if not stage_04_job_artifacts.get("duration_s"):
        stage_04_job_artifacts["duration_s"] = round(
            time.monotonic() - stage_group_started, 3
        )
    stage_components = [
        {
            "name": "stage_04_envs_raw",
            "tier": tier,
            "evidence": evidence,
            "artifacts": {
                "raw_envs": f"{artifact_output_uri(config)}/envs/raw/"
                if config.s3_bucket
                else str(env_root / "raw"),
                **stage_04_job_artifacts,
            },
        },
        {
            "name": "stage_05_envs_train",
            "tier": tier,
            "evidence": (
                f"Split the {env_count} generated environments into "
                f"{train_count} train, {validation_count} validation, and "
                f"{gold_heldout_count} final gold-heldout records."
            ),
            "artifacts": {
                "train_envs": train_envs_uri,
                "validation_envs": validation_envs_uri,
                "heldout_envs": heldout_envs_uri,
                "gold_heldout_envs": gold_heldout_envs_uri,
                "split_manifest": split_manifest_uri,
                "curation_manifest": curation_manifest_uri,
                "job_name": config.run_id,
                "execution": "orchestrator_record_from_stage_04_gpu_outputs",
                "upstream_job_name": str(envgen_invocation.get("job_name") or ""),
                "duration_s": 0.0,
                "duration_scope": "record materialized from completed Stage 4 outputs",
            },
        },
        {
            "name": "stage_06_tokens",
            "tier": tier,
            "evidence": (
                "Recorded scenario features as reporting/lineage inputs for the "
                "state PPO; pixels/tokens are not falsely advertised as policy observations."
            ),
            "artifacts": {
                "tokens": f"{artifact_output_uri(config)}/tokens/manifest.json"
                if config.s3_bucket
                else str(local_dir / "tokens" / "manifest.json"),
                "job_name": config.run_id,
                "execution": "orchestrator_record_from_stage_04_gpu_outputs",
                "upstream_job_name": str(envgen_invocation.get("job_name") or ""),
                "duration_s": 0.0,
                "duration_scope": "record materialized from completed Stage 4 outputs",
            },
        },
    ]

    return {
        "env_count": env_count,
        "train_count": train_count,
        "heldout_count": heldout_count,
        "validation_count": validation_count,
        "gold_heldout_count": gold_heldout_count,
        "train_envs_uri": train_envs_uri,
        "validation_envs_uri": validation_envs_uri,
        "heldout_envs_uri": heldout_envs_uri,
        "gold_heldout_envs_uri": gold_heldout_envs_uri,
        "split_manifest_uri": split_manifest_uri,
        "curation_manifest_uri": curation_manifest_uri,
        "components": stage_components,
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
    checkpoint_uri: str = "",
) -> list[Path]:
    """Stage 7: swappable LeRobot policy container or local reference rollouts."""

    from npa.workflows.sim2real.engine import generate_action_rollouts

    train_uri = (config.train_envs_uri or "").strip()
    if (
        config.s3_bucket
        and train_uri.startswith("s3://")
        and k8s_image_ready(config.policy_image)
    ):
        from npa.workflows.sim2real import engine as loop

        return loop.run_policy_rollout_component(
            config,
            local_dir=local_dir,
            actions_dir=actions_dir,
            outer_iteration=outer_iteration,
            iteration=iteration,
            train_envs_uri=train_uri,
            checkpoint_uri=checkpoint_uri,
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
    frame_count = resolve_augment_frame_count(rollout_count=config.rollout_count)
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
    if config.s3_bucket and config.s3_endpoint.strip():
        from npa.workflows.sim2real.engine import _storage_client

        client = _storage_client(config)
        root = f"{artifact_output_uri(config)}/augment/frames/"
        for item in index:
            client.upload_file(item["local"], f"{root}{Path(item['local']).name}")
        client.upload_file(str(frames_dir / "index.json"), f"{root}index.json")
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


def _mirror_augment_frames(
    frames_uri: str, augment_dir: Path, *, attempts: int = 6
) -> bool:
    """Mirror remote augmentation descriptors/images into the orchestrator tree."""

    uri = str(frames_uri or "").strip()
    if not uri.startswith("s3://"):
        return False
    from npa.clients.storage import StorageClient, StorageError

    frames_dir = Path(augment_dir) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    last_error = ""
    client = StorageClient.from_environment()
    for attempt in range(attempts):
        try:
            client.download_directory(uri.rstrip("/") + "/", str(frames_dir))
        except (OSError, StorageError) as exc:
            last_error = repr(exc)
        if (frames_dir / "index.json").is_file() or any(frames_dir.glob("frame-*.*")):
            return True
        if attempt + 1 < attempts:
            time.sleep(2)
    _write_json(
        frames_dir / "mirror-warning.json",
        {
            "schema": "npa.sim2real.augment_mirror_warning.v1",
            "status": "mirror_unavailable",
            "frames_uri": uri,
            "last_error": last_error,
        },
    )
    return False


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
        ("validation", "uploaded_validation"),
        ("heldout", "uploaded_heldout"),
        ("gold-heldout", "uploaded_gold_heldout"),
    ):
        target = env_root / sub / "envs.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_path(split[uri_key], str(target))
    _write_json(
        env_root / "split-manifest.json",
        {
            "schema": "npa.sim2real.split_manifest.v2",
            "run_id": config.run_id,
            "train_count": split["train_count"],
            "heldout_count": split["heldout_count"],
            "validation_count": split["validation_count"],
            "gold_heldout_count": split["gold_heldout_count"],
            "raw_count": split["raw_count"],
            "train_uri": split["train_uri"],
            "heldout_uri": split["heldout_uri"],
            "validation_uri": split["validation_uri"],
            "gold_heldout_uri": split["gold_heldout_uri"],
            "config_digest_leakage": split["config_digest_leakage"],
            "coverage": split["coverage"],
            "raw_input_proof": split.get("raw_input_proof", {}),
            "remote_manifest": split.get("uploaded_manifest", ""),
        },
    )
    client.download_path(
        split["uploaded_curation"], str(env_root / "curation-manifest.json")
    )
    _write_json(
        local_dir / "tokens" / "manifest.json",
        {
            "schema": "npa.sim2real.tokens.v2",
            "stage": 6,
            "train_env_count": split["train_count"],
            "heldout_env_count": split["heldout_count"],
            "validation_env_count": split["validation_count"],
            "gold_heldout_env_count": split["gold_heldout_count"],
            "learning_consumer": "lineage_and_reporting_only_for_state_ppo",
            "policy_observation_consumer": False,
            "rollout_consumer": "scenario_config_digest",
            "status": "ready",
        },
    )
    # Persist URIs on config object via caller (mutable dataclass? frozen - return values)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
