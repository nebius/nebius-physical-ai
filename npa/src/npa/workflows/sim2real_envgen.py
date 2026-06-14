"""Sim2Real environment generation, split, and action-conditioning contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient


DEFAULT_SCENE_CATALOG = (
    "simready://warehouse/tabletop_v1",
    "simready://lab/bin-picking_v1",
    "simready://factory/conveyor_cell_v1",
)
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
    camera_names: tuple[str, ...] = ("workspace", "wrist")
    physics_profile: str = "genesis-franka-pick-place"
    notes: tuple[str, ...] = field(default_factory=tuple)

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
    def manifest_uri(self) -> str:
        return f"{self.output_uri.rstrip('/')}/envs/manifest/"

    @property
    def actions_uri(self) -> str:
        return f"{self.output_uri.rstrip('/')}/actions/train/"

    def validate(self) -> None:
        if not self.run_id:
            raise Sim2RealEnvGenError("run_id must not be empty")
        if not self.output_uri.startswith("s3://"):
            raise Sim2RealEnvGenError(f"output_uri must be s3://, got {self.output_uri}")
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
    notes: list[str] | tuple[str, ...] | None = None,
) -> SceneSpec:
    """Build the full SceneSpec from SimReady, BYO mesh, and optional augment."""

    final_notes = list(notes or ())
    if not augmented_frames_uri:
        final_notes.append("Cosmos augment omitted because Stage 2 did not produce approved frames.")
    return SceneSpec(
        simready_catalog=tuple(catalog or DEFAULT_SCENE_CATALOG),
        byo_mesh_uri=byo_mesh_uri or DEFAULT_BYO_MESH_URI,
        augmented_frames_uri=augmented_frames_uri,
        notes=tuple(final_notes),
    )


def generate_raw_envs(config: EnvGenConfig) -> list[dict[str, Any]]:
    """Generate deterministic raw env specs for one shard."""

    config.validate()
    envs: list[dict[str, Any]] = []
    for index in range(config.env_count):
        if index % config.shard_count != config.shard_index:
            continue
        rng = random.Random(_stable_int(f"{config.seed}:{index}:{config.run_id}"))
        catalog = config.scene_spec.simready_catalog[index % len(config.scene_spec.simready_catalog)]
        env_id = f"env-{index:05d}"
        envs.append(
            {
                "schema": "npa.sim2real.raw_env.v1",
                "env_id": env_id,
                "seed": rng.randrange(1, 2**31 - 1),
                "scene": {
                    "simready_asset": catalog,
                    "byo_mesh_uri": config.scene_spec.byo_mesh_uri,
                    "augmented_frame_uri": _augment_ref(config.scene_spec.augmented_frames_uri, index),
                },
                "physics": {
                    "engine": "genesis",
                    "profile": config.scene_spec.physics_profile,
                    "friction": round(rng.uniform(0.45, 1.25), 5),
                    "mass_scale": round(rng.uniform(0.85, 1.15), 5),
                    "lighting_lux": round(rng.uniform(350.0, 1200.0), 3),
                },
                "camera_obs": {
                    name: {
                        "uri": f"{config.raw_uri}camera/{env_id}/{name}.png",
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
    shard_path = output_dir / f"raw-shard-{config.shard_index:02d}-of-{config.shard_count:02d}.jsonl"
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
    uploaded_shard = client.upload_file(str(shard_path), f"{config.raw_uri}{shard_path.name}")
    uploaded_scene = client.upload_file(str(scene_path), f"{config.manifest_uri}scene-spec.json")
    uploaded_summary = client.upload_file(str(summary_path), f"{config.raw_uri}{summary_path.name}")
    return {**summary, "uploaded_shard": uploaded_shard, "uploaded_scene_spec": uploaded_scene, "uploaded_summary": uploaded_summary}


def write_split_manifest(config: EnvGenConfig, output_dir: Path) -> dict[str, Any]:
    """Write deterministic disjoint train/heldout split manifests."""

    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = [f"env-{index:05d}" for index in range(config.env_count)]
    shuffled = ids[:]
    random.Random(config.seed).shuffle(shuffled)
    train_count = int(round(config.env_count * config.train_fraction))
    train_ids = set(shuffled[:train_count])
    heldout_ids = set(shuffled[train_count:])
    train_envs = [env for env in _all_envs(config) if env["env_id"] in train_ids]
    heldout_envs = [env for env in _all_envs(config) if env["env_id"] in heldout_ids]
    if set(item["env_id"] for item in train_envs) & set(item["env_id"] for item in heldout_envs):
        raise Sim2RealEnvGenError("train and heldout splits overlap")

    train_path = output_dir / "train-envs.jsonl"
    heldout_path = output_dir / "heldout-envs.jsonl"
    manifest_path = output_dir / "split-manifest.json"
    _write_jsonl(train_path, train_envs)
    _write_jsonl(heldout_path, heldout_envs)
    manifest = {
        "schema": "npa.sim2real.split_manifest.v1",
        "run_id": config.run_id,
        "seed": config.seed,
        "train_fraction": config.train_fraction,
        "raw_count": config.env_count,
        "train_count": len(train_envs),
        "heldout_count": len(heldout_envs),
        "disjoint": True,
        "raw_uri": config.raw_uri,
        "train_uri": config.train_uri,
        "heldout_uri": config.heldout_uri,
    }
    _write_json(manifest_path, manifest)
    client = StorageClient.from_environment()
    uploaded_manifest = client.upload_file(str(manifest_path), f"{config.manifest_uri}split-manifest.json")
    uploaded_train = client.upload_file(str(train_path), f"{config.train_uri}envs.jsonl")
    uploaded_heldout = client.upload_file(str(heldout_path), f"{config.heldout_uri}envs.jsonl")
    return {
        **manifest,
        "uploaded_manifest": uploaded_manifest,
        "uploaded_train": uploaded_train,
        "uploaded_heldout": uploaded_heldout,
    }


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
    output_actions_uri = actions_uri.rstrip("/") + "/" if actions_uri else config.actions_uri
    conditioned: list[dict[str, Any]] = []
    for env in _read_jsonl(train_path)[:limit]:
        seed = _stable_int(f"{config.seed}:{env['env_id']}:{policy_image}")
        rng = random.Random(seed)
        env = dict(env)
        env["actions"] = {
            "schema": "npa.sim2real.reference_actions.v1",
            "policy_image": policy_image,
            "action_space": "cartesian_delta_xyz_gripper",
            "timesteps": 16,
            "values": [
                [round(rng.uniform(-0.025, 0.025), 6) for _ in range(3)]
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
    uploaded_actions = client.upload_file(str(action_path), f"{output_actions_uri}envs.jsonl")
    uploaded_summary = client.upload_file(str(summary_path), f"{output_actions_uri}actions-summary.json")
    return {**summary, "uploaded_actions": uploaded_actions, "uploaded_summary": uploaded_summary}


def build_policy_image_contract(*, train_envs_uri: str, output_uri: str, default_policy_image: str) -> dict[str, Any]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    raw = sub.add_parser("raw-shard")
    _add_common(raw)
    raw.add_argument("--shard-index", type=int, default=int(os.environ.get("JOB_COMPLETION_INDEX", "0")))
    raw.add_argument("--shard-count", type=int, default=int(os.environ.get("NPA_SHARD_COUNT", "1")))
    split = sub.add_parser("split")
    _add_common(split)
    actions = sub.add_parser("actions")
    _add_common(actions)
    actions.add_argument("--policy-image", default=os.environ.get("POLICY_IMAGE", "npa-sim2real-reference-policy:local"))
    actions.add_argument("--limit", type=int, default=int(os.environ.get("ACTION_ENV_LIMIT", "256")))
    actions.add_argument("--train-envs-uri", default=os.environ.get("NPA_TRAIN_ENVS_URI", ""))
    actions.add_argument("--actions-uri", default=os.environ.get("NPA_ACTIONS_URI", ""))
    contract = sub.add_parser("policy-contract")
    contract.add_argument("--train-envs-uri", required=True)
    contract.add_argument("--actions-uri", required=True)
    contract.add_argument("--policy-image", default="npa-sim2real-reference-policy:local")
    args = parser.parse_args(argv)

    if args.command == "policy-contract":
        print(json.dumps(build_policy_image_contract(train_envs_uri=args.train_envs_uri, output_uri=args.actions_uri, default_policy_image=args.policy_image), indent=2, sort_keys=True))
        return 0

    scene = build_scene_spec(
        byo_mesh_uri=args.byo_mesh_uri,
        augmented_frames_uri=args.augmented_frames_uri,
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
        result = write_split_manifest(config, output_dir)
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
    parser.add_argument("--augmented-frames-uri", default="")
    parser.add_argument("--output-dir", default="/tmp/npa-sim2real-envgen")


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


def _augment_ref(base_uri: str, index: int) -> str:
    if not base_uri:
        return ""
    return f"{base_uri.rstrip('/')}/frame-{index % 1024:05d}.png"


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
