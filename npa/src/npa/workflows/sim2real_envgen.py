"""Sim2Real environment generation, split, and action-conditioning contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient
from npa.workflows.sim2real.task_contract import (
    LIFT_DATASET_ID,
    LIFT_TASK_ID,
    TaskContractError,
    assert_contract_digest,
    build_task_contract,
)


DEFAULT_SCENE_CATALOG = ("isaac://Isaac-Lift-Cube-Franka-v0/stock-table-v1",)
DEFAULT_BYO_MESH_URI = ""


class Sim2RealEnvGenError(RuntimeError):
    """Raised when env generation inputs are invalid."""


@dataclass(frozen=True)
class SceneSpec:
    """Scene composition used for raw environment generation."""

    schema: str = "npa.sim2real.scene_spec.v1"
    simready_catalog: tuple[str, ...] = DEFAULT_SCENE_CATALOG
    byo_mesh_uri: str = DEFAULT_BYO_MESH_URI
    augmented_frames_uri: str = ""
    augmented_frames_manifest_uri: str = ""
    augmented_frame_uris: tuple[str, ...] = field(default_factory=tuple)
    scene_spec_uri: str = ""
    robot_spec_uri: str = ""
    robot_preset: str = "franka"
    sim_backend: str = "isaac"
    camera_names: tuple[str, ...] = ("primary", "side", "overhead")
    cameras: dict[str, Any] = field(default_factory=dict)
    physics_profile: str = "isaac-lift-franka"
    notes: tuple[str, ...] = field(default_factory=tuple)
    task_id: str = LIFT_TASK_ID
    dataset_id: str = LIFT_DATASET_ID
    task_contract: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnvGenConfig:
    """Configuration for raw env generation and split."""

    run_id: str
    output_uri: str
    env_count: int = 10_000
    train_fraction: float = 0.8
    seed: int = 42
    shard_index: int = 0
    shard_count: int = 1
    scene_spec: SceneSpec = field(default_factory=SceneSpec)

    @property
    def raw_uri(self) -> str:
        return f"{self.output_uri.rstrip('/')}/envs/raw/"

    @property
    def train_uri(self) -> str:
        return f"{self.output_uri.rstrip('/')}/envs/train/"

    @property
    def heldout_uri(self) -> str:
        return f"{self.output_uri.rstrip('/')}/envs/heldout/"

    @property
    def validation_uri(self) -> str:
        return f"{self.output_uri.rstrip('/')}/envs/validation/"

    @property
    def gold_heldout_uri(self) -> str:
        return f"{self.output_uri.rstrip('/')}/envs/gold-heldout/"

    @property
    def manifest_uri(self) -> str:
        return f"{self.output_uri.rstrip('/')}/envs/manifest/"

    @property
    def actions_uri(self) -> str:
        return f"{self.output_uri.rstrip('/')}/actions/train/"

    def validate(self) -> None:
        if not self.run_id:
            raise Sim2RealEnvGenError("run_id must not be empty")
        if not self.output_uri.startswith("s3://"):
            raise Sim2RealEnvGenError(
                f"output_uri must be s3://, got {self.output_uri}"
            )
        if self.env_count < 2:
            raise Sim2RealEnvGenError("env_count must be at least 2")
        if not 0.0 < self.train_fraction < 1.0:
            raise Sim2RealEnvGenError("train_fraction must be in (0, 1)")
        if self.shard_count <= 0:
            raise Sim2RealEnvGenError("shard_count must be positive")
        if not 0 <= self.shard_index < self.shard_count:
            raise Sim2RealEnvGenError("shard_index must be in [0, shard_count)")


def build_scene_spec(
    *,
    catalog: list[str] | tuple[str, ...] | None = None,
    byo_mesh_uri: str = "",
    augmented_frames_uri: str = "",
    augmented_frames_manifest_uri: str = "",
    augmented_frame_uris: list[str] | tuple[str, ...] | None = None,
    notes: list[str] | tuple[str, ...] | None = None,
) -> SceneSpec:
    """Build the full SceneSpec from SimReady, BYO mesh, and optional augment."""

    final_notes = list(notes or ())
    if (
        not augmented_frames_uri
        and not augmented_frames_manifest_uri
        and not augmented_frame_uris
    ):
        final_notes.append(
            "Cosmos augment omitted because Stage 2 did not produce approved frames."
        )
    return SceneSpec(
        simready_catalog=tuple(catalog or DEFAULT_SCENE_CATALOG),
        byo_mesh_uri=byo_mesh_uri or DEFAULT_BYO_MESH_URI,
        augmented_frames_uri=augmented_frames_uri,
        augmented_frames_manifest_uri=augmented_frames_manifest_uri,
        augmented_frame_uris=_validated_frame_uris(
            augmented_frame_uris or (), source="SceneSpec"
        ),
        notes=tuple(final_notes),
        task_contract=build_task_contract(
            task_id=LIFT_TASK_ID,
            dataset_id=LIFT_DATASET_ID,
            dataset_uri="local://isaac-generated-seed",
        ),
    )


def scene_spec_from_uri(uri: str) -> SceneSpec:
    """Load a SceneSpec JSON artifact from a local path or s3:// URI."""

    ref = str(uri or "").strip()
    if not ref:
        return build_scene_spec()
    local_path = Path("/tmp/npa-scene-spec.json")
    if ref.startswith("s3://"):
        StorageClient.from_environment().download_path(ref, str(local_path))
    else:
        local_path = Path(ref)
    payload = json.loads(local_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Sim2RealEnvGenError(f"scene spec must be a JSON object: {ref}")
    task_contract = dict(payload.get("task_contract") or {})
    if task_contract:
        try:
            contract_digest = assert_contract_digest(task_contract)
        except TaskContractError as exc:
            raise Sim2RealEnvGenError(
                f"scene spec {ref!r} contains an invalid task contract: {exc}"
            ) from exc
        task_id = str(payload.get("task_id") or task_contract.get("task_id") or "")
        dataset = task_contract.get("dataset") or {}
        dataset_id = str(payload.get("dataset_id") or dataset.get("id") or "")
        if task_id != str(task_contract.get("task_id") or ""):
            raise Sim2RealEnvGenError(
                f"scene spec {ref!r} task_id disagrees with its task contract"
            )
        if dataset_id != str(dataset.get("id") or ""):
            raise Sim2RealEnvGenError(
                f"scene spec {ref!r} dataset_id disagrees with its task contract"
            )
        declared_digest = str(payload.get("task_contract_digest") or "")
        if declared_digest and declared_digest != contract_digest:
            raise Sim2RealEnvGenError(
                f"scene spec {ref!r} task_contract_digest disagrees with its "
                "task contract"
            )
    else:
        task_id = str(payload.get("task_id") or LIFT_TASK_ID)
        dataset_id = str(payload.get("dataset_id") or LIFT_DATASET_ID)
    catalog = payload.get("simready_catalog") or DEFAULT_SCENE_CATALOG
    raw_cameras = payload.get("cameras") or {}
    camera_names = payload.get("camera_names")
    if not camera_names and isinstance(raw_cameras, list):
        camera_names = raw_cameras
    elif not camera_names and isinstance(raw_cameras, dict) and raw_cameras:
        camera_names = tuple(raw_cameras)
    camera_names = camera_names or ("primary", "side", "overhead")
    cameras = dict(raw_cameras) if isinstance(raw_cameras, dict) else {}
    notes = payload.get("notes") or ()
    return SceneSpec(
        schema=str(payload.get("schema") or "npa.sim2real.scene_spec.v1"),
        simready_catalog=tuple(catalog),
        byo_mesh_uri=str(payload.get("byo_mesh_uri") or DEFAULT_BYO_MESH_URI),
        augmented_frames_uri=str(payload.get("augmented_frames_uri") or ""),
        augmented_frames_manifest_uri=str(
            payload.get("augmented_frames_manifest_uri") or ""
        ),
        augmented_frame_uris=_validated_frame_uris(
            payload.get("augmented_frame_uris") or (), source=ref
        ),
        scene_spec_uri=str(payload.get("scene_spec_uri") or ""),
        robot_spec_uri=str(payload.get("robot_spec_uri") or ""),
        robot_preset=str(payload.get("robot_preset") or "franka"),
        sim_backend=str(payload.get("sim_backend") or "isaac"),
        camera_names=tuple(camera_names),
        cameras=cameras,
        physics_profile=str(payload.get("physics_profile") or "isaac-lift-franka"),
        notes=tuple(notes),
        task_id=task_id,
        dataset_id=dataset_id,
        task_contract=task_contract,
    )


def frame_uris_from_transfer_manifest(uri: str) -> tuple[str, ...]:
    """Load and validate the exact frame list a real Cosmos transfer published."""

    from npa.workbench.cosmos.transfer import (
        TRANSFER_MANIFEST_MODE,
        TRANSFER_MANIFEST_SCHEMA,
        TRANSFER_MANIFEST_STATUS,
    )

    ref = str(uri or "").strip()
    if not ref:
        raise Sim2RealEnvGenError("transfer manifest URI must not be empty")

    if ref.startswith("s3://"):
        with tempfile.TemporaryDirectory(prefix="npa-envgen-transfer-") as tmp:
            local_path = Path(tmp) / "manifest.json"
            StorageClient.from_environment().download_path(ref, str(local_path))
            payload = _read_json_object(local_path, source=ref)
    else:
        local_ref = ref.removeprefix("file://").removeprefix("local://")
        payload = _read_json_object(Path(local_ref), source=ref)

    if payload.get("schema") != TRANSFER_MANIFEST_SCHEMA:
        raise Sim2RealEnvGenError(
            f"transfer manifest {ref!r} must use schema {TRANSFER_MANIFEST_SCHEMA!r}"
        )
    if payload.get("mode") != TRANSFER_MANIFEST_MODE:
        raise Sim2RealEnvGenError(
            f"transfer manifest {ref!r} must use real-transfer mode "
            f"{TRANSFER_MANIFEST_MODE!r}"
        )
    if payload.get("status") != TRANSFER_MANIFEST_STATUS:
        raise Sim2RealEnvGenError(
            f"transfer manifest {ref!r} must have successful status "
            f"{TRANSFER_MANIFEST_STATUS!r}"
        )
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise Sim2RealEnvGenError(
            f"transfer manifest {ref!r} must contain a non-empty frames list"
        )

    frame_uris: list[str] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise Sim2RealEnvGenError(
                f"transfer manifest {ref!r} frames[{index}] must be an object"
            )
        frame_uri = frame.get("uri")
        if not isinstance(frame_uri, str) or not frame_uri.strip():
            raise Sim2RealEnvGenError(
                f"transfer manifest {ref!r} frames[{index}].uri must be non-empty"
            )
        frame_uris.append(frame_uri.strip())
    return tuple(frame_uris)


def frame_uris_from_augmented_index(prefix: str) -> tuple[str, ...]:
    """Resolve the exact Stage 3 frame index; never synthesize nonexistent URIs."""

    ref = str(prefix or "").rstrip("/") + "/index.json"
    with tempfile.TemporaryDirectory(prefix="npa-envgen-frame-index-") as tmp:
        path = Path(tmp) / "index.json"
        if ref.startswith("s3://"):
            StorageClient.from_environment().download_path(ref, str(path))
        else:
            path = Path(ref.removeprefix("file://").removeprefix("local://"))
        payload = _read_json_object(path, source=ref)
    if payload.get("schema") != "npa.sim2real.augmented_frames.v1":
        raise Sim2RealEnvGenError(f"invalid augmented frame index schema at {ref}")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise Sim2RealEnvGenError(f"augmented frame index is empty at {ref}")
    if not all(
        isinstance(frame, dict) and (frame.get("uri") or frame.get("local"))
        for frame in frames
    ):
        raise Sim2RealEnvGenError(f"augmented frame index has malformed rows at {ref}")
    return _validated_frame_uris(
        [frame.get("uri") or frame.get("local") for frame in frames],
        source=ref,
    )


def resolve_augmented_frames(scene: SceneSpec, reference: str = "") -> SceneSpec:
    """Resolve a manifest reference, or retain prefix semantics for legacy callers."""

    ref = str(reference or scene.augmented_frames_manifest_uri or "").strip()
    if not ref:
        return scene
    if not ref.lower().endswith(".json"):
        return replace(
            scene,
            augmented_frames_uri=ref,
            augmented_frames_manifest_uri="",
            augmented_frame_uris=(),
        )
    return replace(
        scene,
        augmented_frames_uri="",
        augmented_frames_manifest_uri=ref,
        augmented_frame_uris=frame_uris_from_transfer_manifest(ref),
    )


def build_scene_spec_for_augmented_frames(
    *, byo_mesh_uri: str = "", reference: str = ""
) -> SceneSpec:
    """Build a scene and resolve one augment reference exactly once.

    JSON references are manifests; all other non-empty references retain the
    legacy frame-prefix contract. Keeping that classification here prevents CLI
    adapters from first treating a manifest as a prefix and then deriving it a
    second time.
    """

    ref = str(reference or "").strip()
    manifest_ref = ref if ref.lower().endswith(".json") else ""
    prefix_ref = ref if ref and not manifest_ref else ""
    return resolve_augmented_frames(
        build_scene_spec(
            byo_mesh_uri=byo_mesh_uri,
            augmented_frames_uri=prefix_ref,
            augmented_frames_manifest_uri=manifest_ref,
        )
    )


def generate_raw_envs(config: EnvGenConfig) -> list[dict[str, Any]]:
    """Generate deterministic raw env specs for one shard."""

    config.validate()
    envs: list[dict[str, Any]] = []
    for index in range(config.env_count):
        if index % config.shard_count != config.shard_index:
            continue
        catalog = config.scene_spec.simready_catalog[
            index % len(config.scene_spec.simready_catalog)
        ]
        augment_uri = _augment_ref(config.scene_spec, index)
        # Stage 3 is not decorative: the exact augmented-frame lineage keys the
        # state-policy domain-randomization record that Isaac later applies.
        rng = random.Random(
            _stable_int(f"{config.seed}:{index}:{config.run_id}:{augment_uri}")
        )
        env_id = f"env-{index:05d}"
        difficulty = ("easy", "medium", "hard")[index % 3]
        # Exact, reachable point configurations.  Difficulty changes target
        # displacement/height and physics extremes while retaining the same stock
        # Isaac scene/light that can actually be applied per vector environment.
        object_x = round(rng.uniform(-0.08, 0.08), 6)
        object_y = round(rng.uniform(-0.20, 0.20), 6)
        if difficulty == "easy":
            goal_x = round(rng.uniform(0.46, 0.54), 6)
            goal_y = round(rng.uniform(-0.08, 0.08), 6)
            goal_z = round(rng.uniform(0.20, 0.27), 6)
            friction = round(rng.uniform(0.75, 1.05), 6)
            mass_scale = round(rng.uniform(0.92, 1.08), 6)
        elif difficulty == "medium":
            goal_x = round(rng.uniform(0.44, 0.56), 6)
            goal_y = round(rng.uniform(-0.15, 0.15), 6)
            goal_z = round(rng.uniform(0.27, 0.35), 6)
            friction = round(rng.uniform(0.60, 1.18), 6)
            mass_scale = round(rng.uniform(0.88, 1.12), 6)
        else:
            goal_x = round(rng.uniform(0.42, 0.58), 6)
            goal_y = round(rng.choice((-1.0, 1.0)) * rng.uniform(0.15, 0.20), 6)
            goal_z = round(rng.uniform(0.35, 0.42), 6)
            friction = round(
                rng.choice((rng.uniform(0.45, 0.60), rng.uniform(1.18, 1.25))), 6
            )
            mass_scale = round(
                rng.choice((rng.uniform(0.85, 0.90), rng.uniform(1.10, 1.15))), 6
            )
        task_contract = config.scene_spec.task_contract or build_task_contract(
            task_id=config.scene_spec.task_id,
            dataset_id=config.scene_spec.dataset_id,
            dataset_uri="local://isaac-generated-seed",
        )
        applied = {
            "task_id": config.scene_spec.task_id,
            "task_contract_digest": task_contract["task_contract_digest"],
            "scene_id": catalog,
            "object_pose_offset_m": {"x": object_x, "y": object_y, "z": 0.0},
            "goal_pose_robot_base_m": {"x": goal_x, "y": goal_y, "z": goal_z},
            "physics": {
                "friction": friction,
                "mass_scale": mass_scale,
            },
            # Isaac Lift's dome light is global, not independently instanced.  It
            # is fixed in every accepted record so the declared value equals the
            # real scene rather than becoming label-only per-env metadata.
            "lighting": {"dome_intensity": 3000.0, "mode": "global_fixed"},
            "camera_profile": "primary-side-overhead-v1",
        }
        scenario_digest = _scenario_digest(applied)
        envs.append(
            {
                "schema": "npa.sim2real.scenario.v2",
                "env_id": env_id,
                "seed": rng.randrange(1, 2**31 - 1),
                "task_id": config.scene_spec.task_id,
                "task_contract_digest": task_contract["task_contract_digest"],
                "difficulty": difficulty,
                "scenario_config_digest": scenario_digest,
                "applied_config": applied,
                "validity": {
                    "reachable": True,
                    "intersection_free": True,
                    "camera_usable": True,
                    "physics_supported": True,
                    "assets_present": True,
                    "task_schema_match": True,
                },
                "scene": {
                    "simready_asset": catalog,
                    "byo_mesh_uri": config.scene_spec.byo_mesh_uri,
                    "scene_spec_uri": config.scene_spec.scene_spec_uri,
                    "augmented_frame_uri": augment_uri,
                },
                "embodiment": {
                    "robot_preset": config.scene_spec.robot_preset,
                    "robot_spec_uri": config.scene_spec.robot_spec_uri,
                    "sim_backend": config.scene_spec.sim_backend,
                },
                "cameras": config.scene_spec.cameras
                or {
                    name: {
                        "runtime_source": "isaac_tiled_camera_stage_07_and_10",
                        "materialized_in_scenario_record": False,
                        "policy_observation": False,
                        "shape": [480, 640, 3],
                        "dtype": "uint8",
                    }
                    for name in config.scene_spec.camera_names
                },
                "physics": {
                    "engine": config.scene_spec.sim_backend,
                    "profile": config.scene_spec.physics_profile,
                    "friction": friction,
                    "mass_scale": mass_scale,
                    "lighting_lux": 3000.0,
                },
                "object_placement": applied["object_pose_offset_m"],
                "goal_placement": applied["goal_pose_robot_base_m"],
                "source_augmentation": {
                    "frame_uri": augment_uri,
                    "lineage_id": hashlib.sha256(
                        augment_uri.encode("utf-8")
                    ).hexdigest()
                    if augment_uri
                    else "",
                    "consumer": (
                        "scenario_parameter_generation_stratification_and_"
                        "cosmos_reason_context"
                    ),
                    "direct_state_policy_pixels": False,
                },
                "camera_obs": {
                    name: {
                        "runtime_source": "isaac_tiled_camera_stage_07_and_10",
                        "materialized_in_scenario_record": False,
                        "consumer": "cosmos_reason_and_rerun_not_state_ppo",
                        "shape": [480, 640, 3],
                        "dtype": "uint8",
                    }
                    for name in config.scene_spec.camera_names
                },
                "actions": None,
            }
        )
    return envs


def write_raw_shard(config: EnvGenConfig, output_dir: Path) -> dict[str, Any]:
    """Write one raw shard and upload it to S3."""

    envs = generate_raw_envs(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "scene-spec.json"
    shard_path = (
        output_dir
        / f"raw-shard-{config.shard_index:02d}-of-{config.shard_count:02d}.jsonl"
    )
    summary_path = output_dir / f"raw-shard-{config.shard_index:02d}-summary.json"
    _write_json(scene_path, config.scene_spec.to_dict())
    _write_jsonl(shard_path, envs)
    summary = {
        "schema": "npa.sim2real.raw_env_shard_summary.v1",
        "run_id": config.run_id,
        "env_count": config.env_count,
        "shard_index": config.shard_index,
        "shard_count": config.shard_count,
        "raw_count": len(envs),
        "raw_uri": config.raw_uri,
        "scene_spec": str(scene_path),
    }
    _write_json(summary_path, summary)
    client = StorageClient.from_environment()
    uploaded_shard = client.upload_file(
        str(shard_path), f"{config.raw_uri}{shard_path.name}"
    )
    uploaded_scene = client.upload_file(
        str(scene_path), f"{config.manifest_uri}scene-spec.json"
    )
    uploaded_summary = client.upload_file(
        str(summary_path), f"{config.raw_uri}{summary_path.name}"
    )
    return {
        **summary,
        "uploaded_shard": uploaded_shard,
        "uploaded_scene_spec": uploaded_scene,
        "uploaded_summary": uploaded_summary,
    }


def load_raw_shards(
    config: EnvGenConfig,
    download_dir: Path,
    *,
    storage: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Download and verify the exact Stage 4 raw shards consumed by Stage 5.

    This is deliberately stricter than a lineage-only S3 reference: every expected
    indexed shard must exist, the aggregate row count must match ``env_count``, and
    each generated environment ID must occur exactly once.  The returned hashes are
    persisted in the split/curation manifests so a live run can prove which Stage 4
    bytes Stage 5 consumed.
    """

    config.validate()
    download_dir.mkdir(parents=True, exist_ok=True)
    client = storage or StorageClient.from_environment()
    client.download_directory(config.raw_uri, str(download_dir))
    expected_names = {
        f"raw-shard-{index:02d}-of-{config.shard_count:02d}.jsonl"
        for index in range(config.shard_count)
    }
    shard_paths = sorted(download_dir.rglob("raw-shard-*-of-*.jsonl"))
    paths_by_name = {path.name: path for path in shard_paths}
    actual_names = set(paths_by_name)
    if actual_names != expected_names:
        raise Sim2RealEnvGenError(
            "Stage 4 raw shard set mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )

    rows: list[dict[str, Any]] = []
    shard_proof: list[dict[str, Any]] = []
    for name in sorted(expected_names):
        path = paths_by_name[name]
        shard_rows = _read_jsonl(path)
        rows.extend(shard_rows)
        shard_proof.append(
            {
                "name": name,
                "uri": f"{config.raw_uri}{name}",
                "row_count": len(shard_rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if len(rows) != config.env_count:
        raise Sim2RealEnvGenError(
            f"Stage 4 raw rows mismatch: expected {config.env_count}, got {len(rows)}"
        )
    env_ids = [str(row.get("env_id") or "") for row in rows]
    expected_ids = {f"env-{index:05d}" for index in range(config.env_count)}
    if len(set(env_ids)) != len(env_ids) or set(env_ids) != expected_ids:
        raise Sim2RealEnvGenError(
            "Stage 4 raw environment IDs are missing, duplicated, or unexpected"
        )
    proof = {
        "mode": "downloaded_stage_04_raw_shards",
        "raw_uri": config.raw_uri,
        "shard_count": len(shard_proof),
        "row_count": len(rows),
        "shards": shard_proof,
    }
    return rows, proof


def write_split_manifest(
    config: EnvGenConfig,
    output_dir: Path,
    *,
    raw_envs: list[dict[str, Any]] | None = None,
    raw_input_proof: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Curate and write deterministic train/validation/gold-heldout splits."""

    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_rows = list(raw_envs) if raw_envs is not None else _all_envs(config)
    input_proof = dict(
        raw_input_proof
        or {
            "mode": "deterministic_in_process_generation",
            "raw_uri": config.raw_uri,
            "shard_count": 1,
            "row_count": len(input_rows),
        }
    )
    if int(input_proof.get("row_count", -1)) != len(input_rows):
        raise Sim2RealEnvGenError(
            "raw input proof row_count does not match supplied rows"
        )
    accepted, rejected, rejection_reasons = curate_envs(input_rows)
    strata: dict[str, list[dict[str, Any]]] = {
        difficulty: [row for row in accepted if row["difficulty"] == difficulty]
        for difficulty in ("easy", "medium", "hard")
    }
    if any(len(rows) < 3 for rows in strata.values()):
        raise Sim2RealEnvGenError(
            "curation must leave at least three usable scenarios in every difficulty stratum"
        )

    train_envs: list[dict[str, Any]] = []
    validation_envs: list[dict[str, Any]] = []
    gold_envs: list[dict[str, Any]] = []
    # Stratify by difficulty; a digest-based order is deterministic and immune to
    # input file ordering.  The non-train remainder is divided evenly between
    # checkpoint validation and final gold heldout.
    requested_train = int(round(len(accepted) * config.train_fraction))
    # Every difficulty stratum must retain one validation and one gold
    # scenario.  On small integration runs, rounding the requested fraction can
    # otherwise demand more train rows than that sealed coverage permits (for
    # example 19/24 at 0.8 while three strata must retain six rows).  Clamp the
    # requested count to the explicit bounds and record both values below.  The
    # allocator remains exact and fail-closed for internally inconsistent
    # bounds.
    minimum_train = len(strata)
    maximum_train = sum(len(rows) - 2 for rows in strata.values())
    target_train = min(maximum_train, max(minimum_train, requested_train))
    train_quotas = _bounded_stratified_quotas(
        {difficulty: len(rows) for difficulty, rows in strata.items()},
        target=target_train,
        lower={difficulty: 1 for difficulty in strata},
        upper={difficulty: len(rows) - 2 for difficulty, rows in strata.items()},
    )
    remaining_sizes = {
        difficulty: len(rows) - train_quotas[difficulty]
        for difficulty, rows in strata.items()
    }
    target_validation = (len(accepted) - target_train) // 2
    validation_quotas = _bounded_stratified_quotas(
        remaining_sizes,
        target=target_validation,
        lower={difficulty: 1 for difficulty in strata},
        upper={difficulty: size - 1 for difficulty, size in remaining_sizes.items()},
    )
    for difficulty in ("easy", "medium", "hard"):
        stratum = strata[difficulty]
        stratum.sort(
            key=lambda row: hashlib.sha256(
                f"{config.seed}:{row['scenario_config_digest']}".encode("utf-8")
            ).hexdigest()
        )
        n_train = train_quotas[difficulty]
        n_validation = validation_quotas[difficulty]
        train_envs.extend(stratum[:n_train])
        validation_envs.extend(stratum[n_train : n_train + n_validation])
        gold_envs.extend(stratum[n_train + n_validation :])

    split_sets = {
        "train": {row["scenario_config_digest"] for row in train_envs},
        "validation": {row["scenario_config_digest"] for row in validation_envs},
        "gold_heldout": {row["scenario_config_digest"] for row in gold_envs},
    }
    leakage = {
        "train_validation": sorted(split_sets["train"] & split_sets["validation"]),
        "train_gold_heldout": sorted(split_sets["train"] & split_sets["gold_heldout"]),
        "validation_gold_heldout": sorted(
            split_sets["validation"] & split_sets["gold_heldout"]
        ),
    }
    if any(leakage.values()):
        raise Sim2RealEnvGenError(f"scenario config leakage across splits: {leakage}")

    train_path = output_dir / "train-envs.jsonl"
    heldout_path = output_dir / "heldout-envs.jsonl"
    validation_path = output_dir / "validation-envs.jsonl"
    gold_path = output_dir / "gold-heldout-envs.jsonl"
    manifest_path = output_dir / "split-manifest.json"
    curation_path = output_dir / "curation-manifest.json"
    _write_jsonl(train_path, train_envs)
    _write_jsonl(validation_path, validation_envs)
    _write_jsonl(gold_path, gold_envs)
    # Compatibility alias: heldout means the final gold set, never validation.
    _write_jsonl(heldout_path, gold_envs)
    coverage = {
        split: _coverage(rows)
        for split, rows in {
            "train": train_envs,
            "validation": validation_envs,
            "gold_heldout": gold_envs,
        }.items()
    }
    augmentation_records_consumed = sum(
        1 for row in accepted if row["source_augmentation"]["frame_uri"]
    )
    if config.scene_spec.augmented_frame_uris and augmentation_records_consumed != len(
        accepted
    ):
        raise Sim2RealEnvGenError(
            "Stage 3 produced indexed frames but some curated scenarios lack augmentation lineage"
        )
    curation_manifest = {
        "schema": "npa.sim2real.curation_manifest.v1",
        "run_id": config.run_id,
        "input_count": len(input_rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "rejection_reasons": rejection_reasons,
        "deduplicated": rejection_reasons.get("duplicate_config_digest", 0),
        "coverage": _coverage(accepted),
        "task_contract_digest": config.scene_spec.task_contract.get(
            "task_contract_digest", ""
        ),
        "augmentation_records_consumed": augmentation_records_consumed,
        "augmentation_consumer_contract": {
            "scenario_field": "source_augmentation.frame_uri",
            "cosmos_reason_context": True,
            "direct_state_policy_pixels": False,
        },
        "raw_input_proof": input_proof,
    }
    _write_json(curation_path, curation_manifest)
    manifest = {
        "schema": "npa.sim2real.split_manifest.v2",
        "run_id": config.run_id,
        "seed": config.seed,
        "train_fraction": config.train_fraction,
        "requested_train_count": requested_train,
        "effective_train_fraction": len(train_envs) / len(accepted),
        "split_count_adjusted_for_stratification": target_train != requested_train,
        "raw_count": len(input_rows),
        "accepted_count": len(accepted),
        "train_count": len(train_envs),
        "validation_count": len(validation_envs),
        "gold_heldout_count": len(gold_envs),
        "heldout_count": len(gold_envs),
        "disjoint": True,
        "config_digest_leakage": leakage,
        "coverage": coverage,
        "raw_uri": config.raw_uri,
        "train_uri": config.train_uri,
        "heldout_uri": config.heldout_uri,
        "validation_uri": config.validation_uri,
        "gold_heldout_uri": config.gold_heldout_uri,
        "curation_manifest_uri": f"{config.manifest_uri}curation-manifest.json",
        "raw_input_proof": input_proof,
    }
    _write_json(manifest_path, manifest)
    client = StorageClient.from_environment()
    uploaded_manifest = client.upload_file(
        str(manifest_path), f"{config.manifest_uri}split-manifest.json"
    )
    uploaded_curation = client.upload_file(
        str(curation_path), f"{config.manifest_uri}curation-manifest.json"
    )
    uploaded_train = client.upload_file(
        str(train_path), f"{config.train_uri}envs.jsonl"
    )
    uploaded_heldout = client.upload_file(
        str(heldout_path), f"{config.heldout_uri}envs.jsonl"
    )
    uploaded_validation = client.upload_file(
        str(validation_path), f"{config.validation_uri}envs.jsonl"
    )
    uploaded_gold = client.upload_file(
        str(gold_path), f"{config.gold_heldout_uri}envs.jsonl"
    )
    return {
        **manifest,
        "uploaded_manifest": uploaded_manifest,
        "uploaded_curation": uploaded_curation,
        "uploaded_train": uploaded_train,
        "uploaded_heldout": uploaded_heldout,
        "uploaded_validation": uploaded_validation,
        "uploaded_gold_heldout": uploaded_gold,
    }


def curate_envs(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Reject invalid records and deduplicate exact applied configurations."""

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    seen: set[str] = set()
    for row in rows:
        reason = _invalid_reason(row)
        digest = str(row.get("scenario_config_digest") or "")
        if not reason and digest in seen:
            reason = "duplicate_config_digest"
        if reason:
            rejected.append({"env_id": row.get("env_id"), "reason": reason})
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        seen.add(digest)
        accepted.append(row)
    return accepted, rejected, reasons


def _bounded_stratified_quotas(
    sizes: dict[str, int],
    *,
    target: int,
    lower: dict[str, int],
    upper: dict[str, int],
) -> dict[str, int]:
    """Allocate an exact total proportionally while honoring per-stratum bounds."""

    names = sorted(sizes)
    total = sum(sizes.values())
    if total <= 0:
        raise Sim2RealEnvGenError("cannot allocate a split from zero scenarios")
    if target < sum(lower.values()) or target > sum(upper.values()):
        raise Sim2RealEnvGenError(
            f"split target {target} is incompatible with stratum bounds"
        )
    ideals = {name: target * sizes[name] / total for name in names}
    quotas = {
        name: min(upper[name], max(lower[name], int(ideals[name]))) for name in names
    }
    while sum(quotas.values()) < target:
        candidates = [name for name in names if quotas[name] < upper[name]]
        if not candidates:  # pragma: no cover - bounds gate above makes this impossible
            raise Sim2RealEnvGenError("unable to increase split quotas to target")
        chosen = max(candidates, key=lambda name: (ideals[name] - quotas[name], name))
        quotas[chosen] += 1
    while sum(quotas.values()) > target:
        candidates = [name for name in names if quotas[name] > lower[name]]
        if not candidates:  # pragma: no cover - bounds gate above makes this impossible
            raise Sim2RealEnvGenError("unable to decrease split quotas to target")
        chosen = max(candidates, key=lambda name: (quotas[name] - ideals[name], name))
        quotas[chosen] -= 1
    return quotas


def _invalid_reason(row: dict[str, Any]) -> str:
    validity = row.get("validity") or {}
    for validity_field, reason in (
        ("reachable", "unreachable_placement"),
        ("intersection_free", "invalid_intersection"),
        ("camera_usable", "unusable_camera"),
        ("physics_supported", "impossible_physics"),
        ("assets_present", "missing_asset"),
        ("task_schema_match", "task_schema_mismatch"),
    ):
        if not validity.get(validity_field):
            return reason
    applied = row.get("applied_config") or {}
    if row.get("task_id") != LIFT_TASK_ID:
        return "task_schema_mismatch"
    if not row.get("task_contract_digest"):
        return "missing_task_contract_digest"
    if _scenario_digest(applied) != row.get("scenario_config_digest"):
        return "scenario_digest_mismatch"
    goal = row.get("goal_placement") or {}
    if not (0.40 <= float(goal.get("x", -1)) <= 0.60):
        return "unreachable_placement"
    if abs(float(goal.get("y", 9))) > 0.22 or not (
        0.18 <= float(goal.get("z", -1)) <= 0.45
    ):
        return "unreachable_placement"
    physics = row.get("physics") or {}
    if not (0.1 <= float(physics.get("friction", -1)) <= 2.0):
        return "impossible_physics"
    if not (0.2 <= float(physics.get("mass_scale", -1)) <= 3.0):
        return "impossible_physics"
    cameras = row.get("cameras") or {}
    if not all(name in cameras for name in ("primary", "side", "overhead")):
        return "unusable_camera"
    if any(
        not isinstance(spec, dict)
        or spec.get("materialized_in_scenario_record") is not False
        or spec.get("runtime_source") != "isaac_tiled_camera_stage_07_and_10"
        for spec in cameras.values()
    ):
        return "unusable_camera"
    source = row.get("source_augmentation") or {}
    if not source.get("frame_uri") or not source.get("lineage_id"):
        return "missing_augmentation_lineage"
    return ""


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    difficulties = {name: 0 for name in ("easy", "medium", "hard")}
    physics_bins = {name: 0 for name in ("low", "nominal", "high")}
    lineages: set[str] = set()
    for row in rows:
        difficulties[str(row.get("difficulty") or "unknown")] = (
            difficulties.get(str(row.get("difficulty") or "unknown"), 0) + 1
        )
        friction = float((row.get("physics") or {}).get("friction", 1.0))
        physics_bins[
            "low" if friction < 0.7 else "high" if friction > 1.1 else "nominal"
        ] += 1
        lineage = str((row.get("source_augmentation") or {}).get("lineage_id") or "")
        if lineage:
            lineages.add(lineage)
    return {
        "count": len(rows),
        "difficulty": difficulties,
        "friction_bins": physics_bins,
        "augmentation_lineage_count": len(lineages),
    }


def _policy_action_amplitude() -> float:
    """Return action delta scale for the configured policy image variant."""

    variant = os.environ.get("NPA_SIM2REAL_POLICY_VARIANT", "reference").strip().lower()
    if variant in {"explore", "alt", "aggressive"}:
        return 0.085
    return 0.025


def write_action_conditioned_envs(
    config: EnvGenConfig,
    output_dir: Path,
    *,
    policy_image: str,
    limit: int,
    train_envs_uri: str = "",
    actions_uri: str = "",
) -> dict[str, Any]:
    """Write reference action-conditioned envs for a train slice."""

    if limit <= 0:
        raise Sim2RealEnvGenError("limit must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    client = StorageClient.from_environment()
    if train_envs_uri:
        train_path = output_dir / "input" / "train-envs.jsonl"
        client.download_path(train_envs_uri, str(train_path))
        input_train_uri = train_envs_uri
    else:
        split = write_split_manifest(config, output_dir / "split")
        train_path = output_dir / "split" / "train-envs.jsonl"
        input_train_uri = split["uploaded_train"]
    output_actions_uri = (
        actions_uri.rstrip("/") + "/" if actions_uri else config.actions_uri
    )
    amplitude = _policy_action_amplitude()
    variant = os.environ.get("NPA_SIM2REAL_POLICY_VARIANT", "reference").strip().lower()
    conditioned: list[dict[str, Any]] = []
    for env in _read_jsonl(train_path)[:limit]:
        seed = _stable_int(f"{config.seed}:{env['env_id']}:{policy_image}")
        rng = random.Random(seed)
        env = dict(env)
        env["actions"] = {
            "schema": "npa.sim2real.reference_actions.v1",
            "policy_image": policy_image,
            "policy_variant": variant,
            "action_space": "cartesian_delta_xyz_gripper",
            "timesteps": 16,
            "values": [
                [round(rng.uniform(-amplitude, amplitude), 6) for _ in range(3)]
                + [round(rng.uniform(0.0, 1.0), 6)]
                for _ in range(16)
            ],
        }
        conditioned.append(env)
    action_path = output_dir / "action-conditioned-train-envs.jsonl"
    summary_path = output_dir / "actions-summary.json"
    _write_jsonl(action_path, conditioned)
    summary = {
        "schema": "npa.sim2real.actions_summary.v1",
        "run_id": config.run_id,
        "policy_image": policy_image,
        "input_train_uri": input_train_uri,
        "actions_uri": output_actions_uri,
        "action_conditioned_count": len(conditioned),
    }
    _write_json(summary_path, summary)
    uploaded_actions = client.upload_file(
        str(action_path), f"{output_actions_uri}envs.jsonl"
    )
    uploaded_summary = client.upload_file(
        str(summary_path), f"{output_actions_uri}actions-summary.json"
    )
    return {
        **summary,
        "uploaded_actions": uploaded_actions,
        "uploaded_summary": uploaded_summary,
    }


def build_policy_image_contract(
    *, train_envs_uri: str, output_uri: str, default_policy_image: str
) -> dict[str, Any]:
    """Return the BYO policy-image contract for action generation."""

    return {
        "schema": "npa.sim2real.policy_image_contract.v1",
        "input": {
            "train_envs_uri": train_envs_uri,
            "camera_obs": {
                "workspace": {"dtype": "uint8", "shape": [480, 640, 3]},
                "wrist": {"dtype": "uint8", "shape": [480, 640, 3]},
            },
        },
        "output": {
            "action_conditioned_envs_uri": output_uri,
            "action_schema": {"dtype": "float32", "shape": ["T", 4]},
        },
        "defaults": {"policy_image": default_policy_image},
        "overrides": ["--policy-image", "--train-envs-uri", "--actions-uri"],
    }


def build_parser() -> argparse.ArgumentParser:
    """Return this module's CLI parser.

    Exposed separately from :func:`main` so a guardrail can check that a catalog
    ``toolRef`` argv this module is invoked with actually parses. ``--run-id`` is required,
    and the ``raw-shard`` toolRef used to omit it — a defect no test could see, because the
    flag audit only understands Typer CLIs invoked as ``npa …``.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    raw = sub.add_parser("raw-shard")
    _add_common(raw)
    raw.add_argument(
        "--shard-index",
        type=int,
        default=int(os.environ.get("JOB_COMPLETION_INDEX", "0")),
    )
    raw.add_argument(
        "--shard-count", type=int, default=int(os.environ.get("NPA_SHARD_COUNT", "1"))
    )
    split = sub.add_parser("split")
    _add_common(split)
    split.add_argument(
        "--shard-count",
        type=int,
        default=int(os.environ.get("NPA_SHARD_COUNT", "1")),
    )
    actions = sub.add_parser("actions")
    _add_common(actions)
    actions.add_argument(
        "--policy-image",
        default=os.environ.get("POLICY_IMAGE", "npa-reference-policy:local"),
    )
    actions.add_argument(
        "--limit", type=int, default=int(os.environ.get("ACTION_ENV_LIMIT", "256"))
    )
    actions.add_argument(
        "--train-envs-uri", default=os.environ.get("NPA_TRAIN_ENVS_URI", "")
    )
    actions.add_argument("--actions-uri", default=os.environ.get("NPA_ACTIONS_URI", ""))
    contract = sub.add_parser("policy-contract")
    contract.add_argument("--train-envs-uri", required=True)
    contract.add_argument("--actions-uri", required=True)
    contract.add_argument("--policy-image", default="npa-reference-policy:local")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "policy-contract":
        print(
            json.dumps(
                build_policy_image_contract(
                    train_envs_uri=args.train_envs_uri,
                    output_uri=args.actions_uri,
                    default_policy_image=args.policy_image,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if getattr(args, "scene_spec_uri", "").strip():
        scene = resolve_augmented_frames(
            scene_spec_from_uri(args.scene_spec_uri), args.augmented_frames_uri
        )
    else:
        scene = build_scene_spec_for_augmented_frames(
            byo_mesh_uri=args.byo_mesh_uri,
            reference=args.augmented_frames_uri,
        )
    config = EnvGenConfig(
        run_id=args.run_id,
        output_uri=args.output_uri,
        env_count=args.env_count,
        train_fraction=args.train_fraction,
        seed=args.seed,
        shard_index=getattr(args, "shard_index", 0),
        shard_count=getattr(args, "shard_count", 1),
        scene_spec=scene,
    )
    output_dir = Path(args.output_dir)
    if args.command == "raw-shard":
        result = write_raw_shard(config, output_dir)
    elif args.command == "split":
        raw_envs, raw_proof = load_raw_shards(config, output_dir / "stage-04-raw")
        result = write_split_manifest(
            config,
            output_dir,
            raw_envs=raw_envs,
            raw_input_proof=raw_proof,
        )
    elif args.command == "actions":
        result = write_action_conditioned_envs(
            config,
            output_dir,
            policy_image=args.policy_image,
            limit=args.limit,
            train_envs_uri=args.train_envs_uri,
            actions_uri=args.actions_uri,
        )
    else:  # pragma: no cover - argparse enforces choices
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--env-count", type=int, default=10_000)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--byo-mesh-uri", default=DEFAULT_BYO_MESH_URI)
    parser.add_argument(
        "--augmented-frames-uri",
        default="",
        help=(
            "Legacy frame prefix, or a transfer manifest.json whose exact non-empty "
            "frames[].uri list will be resolved before environment generation."
        ),
    )
    parser.add_argument(
        "--scene-spec-uri", default=os.environ.get("NPA_SIM2REAL_SCENE_SPEC_URI", "")
    )
    parser.add_argument("--output-dir", default="/tmp/npa-envgen")


def _all_envs(config: EnvGenConfig) -> list[dict[str, Any]]:
    single = EnvGenConfig(
        run_id=config.run_id,
        output_uri=config.output_uri,
        env_count=config.env_count,
        train_fraction=config.train_fraction,
        seed=config.seed,
        shard_index=0,
        shard_count=1,
        scene_spec=config.scene_spec,
    )
    return generate_raw_envs(single)


def _augment_ref(scene: SceneSpec, index: int) -> str:
    if scene.augmented_frame_uris:
        return scene.augmented_frame_uris[index % len(scene.augmented_frame_uris)]
    if not scene.augmented_frames_uri:
        return ""
    return f"{scene.augmented_frames_uri.rstrip('/')}/frame-{index % 1024:05d}.png"


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _scenario_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_json_object(path: Path, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Sim2RealEnvGenError(
            f"could not read transfer manifest {source!r}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise Sim2RealEnvGenError(f"transfer manifest {source!r} must be a JSON object")
    return payload


def _validated_frame_uris(values: Any, *, source: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise Sim2RealEnvGenError(f"augmented_frame_uris in {source!r} must be a list")
    frame_uris: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise Sim2RealEnvGenError(
                f"augmented_frame_uris[{index}] in {source!r} must be non-empty"
            )
        frame_uris.append(value.strip())
    return tuple(frame_uris)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
