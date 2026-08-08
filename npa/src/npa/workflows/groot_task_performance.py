"""Closed-loop GR00T N1.7 task-performance evaluation for canonical PushT.

Unlike :mod:`npa.workflows.groot_learning`, this module never scores actions
against recorded expert actions.  Every accepted action is produced by a
loaded GR00T checkpoint and advances a live ``gym_pusht/PushT-v0`` physics
environment.  The module intentionally fails closed on provenance, model,
shape, simulator-frame, pairing, or statistical-evidence mismatches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from npa.workbench.foxglove.inspect import summarize_mcap
from npa.workbench.foxglove.mcap_writer import FrameInput, LogInput, MetricsInput, write_run_mcap
from npa.workflows.groot_learning import _checkpoint_identity, _download_prefix, _json_bytes
from npa.workflows.groot_visualization import (
    GrootVisualizationError,
    _download,
    _head_artifact,
    _list_objects,
    _put_bytes,
    _put_json,
    _read_s3_json,
    _s3_client,
    _split_s3,
    inspect_rrd,
)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


CONTRACT_SCHEMA = "npa.groot.task_contract.v1"
EVALUATION_SCHEMA = "npa.groot.closed_loop_eval.v1"
REPORT_SCHEMA = "npa.groot.task_performance.v1"
RENDER_SCHEMA = "npa.groot.task_rollouts.v1"
PUBLISH_SCHEMA = "npa.groot.task_performance_publish.v1"
CHECKPOINT_REF_SCHEMA = "npa.groot.checkpoint_ref.v1"
SELECTION_SCHEMA = "npa.groot.checkpoint_selection.v1"
DATASET_ID = "lerobot/pusht"
DATASET_REVISION = "7628202a2180972f291ba1bc6723834921e72c19"
DATASET_TASK = "Push the T-shaped block onto the T-shaped target."
ENVIRONMENT_ID = "gym_pusht/PushT-v0"
ENVIRONMENT_VERSION = "0.1.6"
ENVIRONMENT_REVISION = "8227842037637f92cf7e2a7199db7570d04de8a1"
ENVIRONMENT_SOURCE_SHA256 = "becc03430fb2a1f8d7f2d2a0d483198dd3e3c4da89cc23ed5cbde2f93d412ddf"
ENVIRONMENT_WHEEL_SHA256 = "c0785a29795f17c97c58b00ffaed9e45be5996dbc864d863a39a857403010206"
HORIZON = 300
FPS = 10.0
SUCCESS_THRESHOLD = 0.95
SEMANTIC_PHASES = [
    "resolve_task_contract",
    "prepare_retraining_split",
    "retrain_task_policy",
    "resolve_trained_checkpoint",
    "evaluate_validation_baseline",
    "evaluate_validation_candidate",
    "analyze_validation_outcomes",
    "select_checkpoint",
    "evaluate_baseline_closed_loop",
    "evaluate_trained_closed_loop",
    "analyze_paired_outcomes",
    "render_task_rollouts",
    "emit_mcap",
    "emit_rrd",
    "publish",
]
REQUIRED_MCAP_TOPICS = {
    "/rollout/baseline/camera",
    "/rollout/trained/camera",
    "/rollout/baseline/action",
    "/rollout/trained/action",
    "/rollout/object_pose",
    "/rollout/goal_pose",
    "/rollout/task_progress",
    "/rollout/success",
    "/metrics/baseline_success_rate",
    "/metrics/trained_success_rate",
    "/metrics/paired_delta",
    "/log",
}
REQUIRED_RRD_ENTITIES = [
    "rollout/baseline/camera",
    "rollout/trained/camera",
    "task/object_pose",
    "task/goal_pose",
    "task/progress/baseline",
    "task/progress/trained",
    "task/outcome",
    "aggregate/baseline_success_rate",
    "aggregate/trained_success_rate",
    "aggregate/paired_delta",
    "aggregate/confidence_interval",
    "provenance",
]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def deterministic_seeds(namespace: str, count: int) -> list[int]:
    """Derive a stable, explicit final-test seed set without global RNG state."""

    if count < 20:
        raise GrootVisualizationError("closed-loop evaluation requires at least 20 paired episodes")
    seeds = [
        int.from_bytes(hashlib.sha256(f"{namespace}:{index}".encode()).digest()[:4], "big")
        for index in range(count)
    ]
    if len(set(seeds)) != count:
        raise GrootVisualizationError("deterministic seed derivation produced a collision")
    return seeds


def _logical_rows_hash(table: Any, columns: Sequence[str]) -> str:
    """Hash Arrow values by logical content rather than Parquet byte encoding."""

    digest = hashlib.sha256()
    for column in columns:
        if column not in table.column_names:
            raise GrootVisualizationError(f"provenance table lacks canonical column {column}")
        digest.update(column.encode())
        for value in table[column].to_pylist():
            digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _hf_url(path: str) -> str:
    return (
        "https://huggingface.co/datasets/lerobot/pusht/resolve/"
        f"{DATASET_REVISION}/{path}?download=true"
    )


def _fetch(url: str, target: Path) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "npa-task-contract/1"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - immutable URL
        payload = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return payload


def _source_metadata(client: Any, source_uri: str, root: Path) -> tuple[dict[str, Any], list[Path]]:
    """Materialize only metadata and Parquet rows, never source replay videos."""

    ref = _split_s3(source_uri, require_key=False)
    objects = _list_objects(client, source_uri)
    prefix = ref.key.rstrip("/") + "/"
    wanted = []
    for item in objects:
        key = str(item["key"])
        relative = key[len(prefix) :] if key.startswith(prefix) else key
        if relative in {"meta/info.json", "meta/episodes.jsonl"} or (
            relative.startswith("data/") and relative.endswith(".parquet")
        ):
            wanted.append(item)
    parquet_paths: list[Path] = []
    for item in wanted:
        key = str(item["key"])
        relative = key[len(prefix) :]
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(ref.bucket, key, str(target))
        if target.suffix == ".parquet" and "/data/" in ("/" + relative):
            parquet_paths.append(target)
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file() or not parquet_paths:
        raise GrootVisualizationError("converted source lacks info, episode metadata, or data rows")
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    return {"info": json.loads(info_path.read_text()), "episodes": episodes}, sorted(parquet_paths)


def resolve_task_contract(
    source_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Prove the dataset/environment contract from immutable upstream bytes."""

    import importlib.metadata
    import pyarrow as pa
    import pyarrow.parquet as pq

    client = _s3_client(s3_client)
    if importlib.metadata.version("gym-pusht") != ENVIRONMENT_VERSION:
        raise GrootVisualizationError(f"evaluator must be gym-pusht=={ENVIRONMENT_VERSION}")
    with tempfile.TemporaryDirectory(prefix="npa-groot-task-contract-") as tmp:
        root = Path(tmp)
        converted, converted_paths = _source_metadata(client, source_uri, root / "converted")
        upstream_info_bytes = _fetch(_hf_url("meta/info.json"), root / "upstream/info.json")
        upstream_data_bytes = _fetch(
            _hf_url("data/chunk-000/file-000.parquet"), root / "upstream/data.parquet"
        )
        upstream_episodes_bytes = _fetch(
            _hf_url("meta/episodes/chunk-000/file-000.parquet"),
            root / "upstream/episodes.parquet",
        )
        upstream_tasks_bytes = _fetch(
            _hf_url("meta/tasks.parquet"), root / "upstream/tasks.parquet"
        )
        upstream_info = json.loads(upstream_info_bytes)
        canonical = pq.read_table(root / "upstream/data.parquet")
        converted_table = pa.concat_tables([pq.read_table(path) for path in converted_paths])
        columns = [
            "observation.state",
            "action",
            "episode_index",
            "frame_index",
            "index",
            "next.done",
            "next.reward",
            "next.success",
            "task_index",
            "timestamp",
        ]
        canonical_hash = _logical_rows_hash(canonical, columns)
        converted_hash = _logical_rows_hash(converted_table, columns)
        if canonical.num_rows != 25_650 or converted_table.num_rows != canonical.num_rows:
            raise GrootVisualizationError("converted source row count does not match canonical PushT")
        if canonical_hash != converted_hash:
            raise GrootVisualizationError("converted source rows do not equal canonical lerobot/pusht")
        tasks = pq.read_table(root / "upstream/tasks.parquet").to_pylist()
        canonical_task = str(tasks[0].get("task") or tasks[0].get("__index_level_0__") or "")
        converted_tasks = {str(item.get("tasks", [""])[0]) for item in converted["episodes"]}
        if canonical_task != DATASET_TASK or converted_tasks != {DATASET_TASK}:
            raise GrootVisualizationError("task descriptions do not prove the canonical PushT task")
        info = converted["info"]
        feature = info.get("features", {})
        if (
            int(info.get("total_episodes", -1)) != 206
            or int(info.get("total_frames", -1)) != 25_650
            or float(info.get("fps", -1)) != FPS
            or list(feature.get("observation.image", {}).get("shape", [])) != [96, 96, 3]
            or list(feature.get("observation.state", {}).get("shape", [])) != [2]
            or list(feature.get("action", {}).get("shape", [])) != [2]
        ):
            raise GrootVisualizationError("converted observation/action metadata is not PushT-compatible")

        import gymnasium as gym
        import gym_pusht  # noqa: F401

        env = gym.make(
            ENVIRONMENT_ID,
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            observation_width=96,
            observation_height=96,
            visualization_width=680,
            visualization_height=680,
        )
        observation, _ = env.reset(seed=1701)
        first_frame = env.render()
        initial_hash = _sha256(_json_bytes(_environment_state(env)) + first_frame.tobytes())
        observation2, _, _, _, _ = env.step(env.action_space.sample())
        second_frame = env.render()
        env.close()
        if (
            observation["pixels"].shape != (96, 96, 3)
            or observation["agent_pos"].shape != (2,)
            or observation2["pixels"].shape != (96, 96, 3)
            or first_frame.shape != (680, 680, 3)
            or _sha256(first_frame.tobytes()) == _sha256(second_frame.tobytes())
        ):
            raise GrootVisualizationError("official simulator did not produce live compatible frames")

    result = {
        "schema": CONTRACT_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "task": {"name": "PushT", "goal": DATASET_TASK},
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "source_uri": source_uri,
            "canonical_rows": 25_650,
            "episodes": 206,
            "logical_rows_sha256": canonical_hash,
            "canonical_object_sha256": {
                "info.json": _sha256(upstream_info_bytes),
                "data.parquet": _sha256(upstream_data_bytes),
                "episodes.parquet": _sha256(upstream_episodes_bytes),
                "tasks.parquet": _sha256(upstream_tasks_bytes),
            },
            "converted_logically_equal": True,
            "canonical_info": {
                "codebase_version": upstream_info.get("codebase_version"),
                "total_episodes": upstream_info.get("total_episodes"),
                "total_frames": upstream_info.get("total_frames"),
            },
        },
        "embodiment": {
            "name": "2D simulated circular pusher",
            "checkpoint_tag": "NEW_EMBODIMENT",
            "physical_robot": False,
            "simulation": True,
        },
        "observation": {
            "video": "96x96 RGB simulator observation, uint8, current environment state",
            "state": "pusher x/y position, float32",
            "frequency_hz": FPS,
        },
        "action": {
            "dimensions": 2,
            "semantics": ["absolute target pusher x", "absolute target pusher y"],
            "units": ["workspace pixels", "workspace pixels"],
            "range": [[0.0, 512.0], [0.0, 512.0]],
            "postprocessing": "finite check then component-wise clip to [0,512]",
        },
        "environment": {
            "id": ENVIRONMENT_ID,
            "package": "gym-pusht",
            "version": ENVIRONMENT_VERSION,
            "source_revision": ENVIRONMENT_REVISION,
            "source_file_sha256": ENVIRONMENT_SOURCE_SHA256,
            "wheel_sha256": ENVIRONMENT_WHEEL_SHA256,
            "physics": ["PyMunk", "pygame", "Shapely"],
            "horizon": HORIZON,
            "render_resolution": [680, 680],
            "simulator_frame_probe_sha256": initial_hash,
        },
        "success": {
            "predicate": "block/goal intersection coverage > 0.95",
            "threshold": SUCCESS_THRESHOLD,
            "task_native_score": "maximum block/goal coverage over episode",
            "higher_is_better": True,
        },
        "proven": True,
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@dataclass
class _EpisodeWriter:
    path: Path
    container: Any
    stream: Any

    @classmethod
    def create(cls, path: Path) -> "_EpisodeWriter":
        import av

        path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(path), mode="w")
        try:
            stream = container.add_stream("libx264", rate=int(FPS))
        except Exception:  # noqa: BLE001 - runtime codec fallback
            stream = container.add_stream("mpeg4", rate=int(FPS))
        stream.width = 680
        stream.height = 680
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        return cls(path, container, stream)

    def add(self, rgb: Any) -> None:
        import av

        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def _upload_file(client: Any, path: Path, uri: str) -> dict[str, Any]:
    ref = _split_s3(uri)
    client.upload_file(str(path), ref.bucket, ref.key)
    return _head_artifact(client, uri)


def _verify_checkpoint_identity(identity: Mapping[str, Any], expected_sha256: str) -> None:
    """Fail closed unless the canonical checkpoint-directory digest matches."""

    actual = str(identity.get("sha256") or "")
    if actual != expected_sha256:
        raise GrootVisualizationError(
            f"checkpoint immutable identity mismatch: {actual or 'missing'} != {expected_sha256}"
        )


def _policy_observation(policy: Any, observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    modality = policy.get_modality_config()
    video_keys = list(modality["video"].modality_keys)
    state_keys = list(modality["state"].modality_keys)
    language_keys = list(modality["language"].modality_keys)
    if video_keys != ["front"] or state_keys != ["single_arm", "gripper"]:
        raise GrootVisualizationError(
            f"checkpoint modality mismatch: video={video_keys}, state={state_keys}"
        )
    pixels = np.stack([np.asarray(item["pixels"], dtype=np.uint8) for item in observations])
    positions = np.stack([np.asarray(item["agent_pos"], dtype=np.float32) for item in observations])
    if pixels.shape[1:] != (96, 96, 3) or positions.shape[1:] != (2,):
        raise GrootVisualizationError("live simulator observation shape mismatch")
    language = [[DATASET_TASK] for _ in observations]
    return {
        "video": {"front": pixels[:, None, ...]},
        "state": {
            "single_arm": positions[:, None, 0:1],
            "gripper": positions[:, None, 1:2],
        },
        "language": {key: language for key in language_keys},
    }


def _policy_action_chunk(policy: Any, observations: Sequence[Mapping[str, Any]]) -> Any:
    import numpy as np

    raw, _ = policy.get_action(_policy_observation(policy, observations))
    if set(raw) != {"single_arm", "gripper"}:
        raise GrootVisualizationError(f"checkpoint action keys mismatch: {sorted(raw)}")
    x = np.asarray(raw["single_arm"], dtype=np.float32)
    y = np.asarray(raw["gripper"], dtype=np.float32)
    if x.ndim != 3 or y.ndim != 3 or x.shape[0] != len(observations) or y.shape != x.shape:
        raise GrootVisualizationError(f"checkpoint action tensor mismatch: x={x.shape}, y={y.shape}")
    actions = np.concatenate([x, y], axis=-1)
    if actions.shape[-1] != 2 or not np.all(np.isfinite(actions)):
        raise GrootVisualizationError("checkpoint emitted non-finite or non-2D actions")
    return actions


def _environment_state(env: Any) -> dict[str, Any]:
    raw = env.unwrapped
    return {
        "pusher_xy": [float(raw.agent.position.x), float(raw.agent.position.y)],
        "object_xy": [float(raw.block.position.x), float(raw.block.position.y)],
        "object_angle_rad": float(raw.block.angle),
        "goal_xy": [float(raw.goal_pose[0]), float(raw.goal_pose[1])],
        "goal_angle_rad": float(raw.goal_pose[2]),
        "coverage": float(raw._get_coverage()),
    }


def _canonical_initial_state(step: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize physical reset state, excluding derived GEOS coverage noise."""

    return {
        "pusher_xy": [float(value) for value in step["pusher_xy"]],
        "object_xy": [float(value) for value in step["object_xy"]],
        "object_angle_rad": float(step["object_angle_rad"]),
        "goal_xy": [float(value) for value in step["goal_xy"]],
        "goal_angle_rad": float(step["goal_angle_rad"]),
    }


def _paired_initial_state_hash(client: Any, before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    """Prove paired reset state even if derived polygon coverage differs by machine epsilon."""

    before_doc = _read_s3_json(client, str(before["trajectory"]["uri"]))
    after_doc = _read_s3_json(client, str(after["trajectory"]["uri"]))
    before_steps = before_doc.get("steps") or []
    after_steps = after_doc.get("steps") or []
    if not before_steps or not after_steps:
        raise GrootVisualizationError("paired rollout lacks an initial trajectory state")
    before_state = _canonical_initial_state(before_steps[0])
    after_state = _canonical_initial_state(after_steps[0])
    if before_state != after_state:
        raise GrootVisualizationError("paired rollouts did not start from the same physical state")
    return _sha256(_json_bytes(before_state))


def resolve_trained_checkpoint(
    training_manifest_uri: str,
    split_manifest_uri: str,
    checkpoint_uri: str,
    output_uri: str,
    run_id: str,
    *,
    expected_gpu_count: int = 7,
    expected_max_steps: int = 0,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Resolve an immutable identity for the exact uploaded trainer output."""

    client = _s3_client(s3_client)
    manifest = _read_s3_json(client, training_manifest_uri)
    split = _read_s3_json(client, split_manifest_uri)
    if (
        manifest.get("schema") != "npa.groot.finetune.v1"
        or manifest.get("status") != "completed"
        or manifest.get("run_id") != run_id
        or manifest.get("checkpoint_uri") != checkpoint_uri
        or manifest.get("optimizer_step_ok") is not True
        or manifest.get("collective_ok") is not True
    ):
        raise GrootVisualizationError("trainer manifest lacks completed immutable-run evidence")
    gpu_count = int(manifest.get("num_gpus") or 0)
    world_size = int(manifest.get("world_size") or 0)
    distinct_gpu_count = int(manifest.get("distinct_gpu_count") or 0)
    if {gpu_count, world_size, distinct_gpu_count} != {int(expected_gpu_count)}:
        raise GrootVisualizationError("trainer did not use the required distinct GPU world")
    configured_steps = int(manifest.get("max_steps") or 0)
    completed_steps = int(manifest.get("training_step") or 0)
    training_plan = split.get("training_plan") or {}
    if (
        split.get("schema") != "npa.groot.episode_split.v1"
        or split.get("run_id") != run_id
        or split.get("status") != "prepared"
        or int(training_plan.get("configured_max_steps") or 0) != int(expected_max_steps)
        or int(training_plan.get("effective_max_steps") or 0) != int(expected_max_steps)
        or int(training_plan.get("global_batch_size") or 0)
        != int(manifest.get("global_batch_size") or 0)
    ):
        raise GrootVisualizationError("trainer and split preflight step contracts differ")
    if int(expected_max_steps) <= 0 or configured_steps != int(expected_max_steps):
        raise GrootVisualizationError("trainer max_steps differs from the preflight contract")
    if completed_steps != configured_steps:
        raise GrootVisualizationError("trainer did not complete the configured optimizer steps")

    with tempfile.TemporaryDirectory(prefix="npa-groot-checkpoint-ref-") as tmp:
        checkpoint_path = Path(tmp) / "checkpoint"
        _download_prefix(client, checkpoint_uri, checkpoint_path)
        identity = _checkpoint_identity(checkpoint_path)
    result = {
        "schema": CHECKPOINT_REF_SCHEMA,
        "status": "resolved",
        "run_id": run_id,
        "checkpoint": {"uri": checkpoint_uri, **identity},
        "training_manifest_uri": training_manifest_uri,
        "split_manifest_uri": split_manifest_uri,
        "training": {
            "base_model": manifest.get("base_model"),
            "base_model_revision": manifest.get("base_model_revision"),
            "groot_repo_ref": manifest.get("groot_repo_ref"),
            "gpu_model": manifest.get("gpu_model"),
            "gpu_count": gpu_count,
            "world_size": world_size,
            "distinct_gpu_count": distinct_gpu_count,
            "optimizer_steps": completed_steps,
            "global_batch_size": int(manifest.get("global_batch_size") or 0),
            "training_examples": int(manifest.get("training_examples") or 0),
            "aggregate_train_loss": manifest.get("aggregate_train_loss"),
            "final_step_loss": manifest.get("final_step_loss", manifest.get("final_loss")),
        },
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _checkpoint_from_ref(
    client: Any, checkpoint_ref_uri: str, *, run_id: str, allow_selection: bool
) -> tuple[str, str]:
    reference = _read_s3_json(client, checkpoint_ref_uri)
    permitted = {CHECKPOINT_REF_SCHEMA}
    if allow_selection:
        permitted.add(SELECTION_SCHEMA)
    if (
        reference.get("schema") not in permitted
        or reference.get("run_id") != run_id
        or reference.get("status") not in {"resolved", "selected"}
    ):
        raise GrootVisualizationError("checkpoint reference does not belong to this run")
    checkpoint = reference.get("checkpoint") or {}
    uri = str(checkpoint.get("uri") or "")
    sha256 = str(checkpoint.get("sha256") or "")
    if not uri.startswith("s3://") or len(sha256) != 64:
        raise GrootVisualizationError("checkpoint reference lacks an immutable S3 identity")
    return uri, sha256


def evaluate_closed_loop(
    contract_uri: str,
    checkpoint_uri: str | None,
    output_uri: str,
    rollout_prefix_uri: str,
    run_id: str,
    phase: str,
    checkpoint_sha256: str | None,
    seed_namespace: str,
    *,
    checkpoint_ref_uri: str | None = None,
    allow_selection_ref: bool = False,
    episodes: int = 20,
    horizon: int = HORIZON,
    policy_batch_size: int = 4,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Execute real GR00T outputs in live paired PushT environments."""

    import numpy as np
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    if phase not in {"baseline", "trained"}:
        raise GrootVisualizationError("closed-loop phase must be baseline or trained")
    if int(horizon) != HORIZON:
        raise GrootVisualizationError(f"PushT evaluation horizon must remain {HORIZON}")
    if int(policy_batch_size) < 1 or int(policy_batch_size) > 8:
        raise GrootVisualizationError("policy batch size must be in [1,8] for bounded GPU inference")
    if not torch.cuda.is_available():
        raise GrootVisualizationError("closed-loop GR00T evaluation requires CUDA")
    client = _s3_client(s3_client)
    if checkpoint_ref_uri:
        if checkpoint_uri or checkpoint_sha256:
            raise GrootVisualizationError("checkpoint URI and reference are mutually exclusive")
        checkpoint_uri, checkpoint_sha256 = _checkpoint_from_ref(
            client,
            checkpoint_ref_uri,
            run_id=run_id,
            allow_selection=allow_selection_ref,
        )
    if not checkpoint_uri or not checkpoint_sha256:
        raise GrootVisualizationError("evaluation requires an immutable checkpoint identity")
    contract = _read_s3_json(client, contract_uri)
    if contract.get("schema") != CONTRACT_SCHEMA or contract.get("proven") is not True:
        raise GrootVisualizationError("evaluation refuses an unproven task contract")
    if contract.get("run_id") != run_id:
        raise GrootVisualizationError("task contract belongs to another run")
    seeds = deterministic_seeds(seed_namespace, int(episodes))

    import gymnasium as gym
    import gym_pusht  # noqa: F401

    with tempfile.TemporaryDirectory(prefix=f"npa-groot-pusht-{phase}-") as tmp:
        root = Path(tmp)
        checkpoint_path = root / "checkpoint"
        _download_prefix(client, checkpoint_uri, checkpoint_path)
        identity = _checkpoint_identity(checkpoint_path)
        _verify_checkpoint_identity(identity, checkpoint_sha256)
        try:
            policy = Gr00tPolicy(
                embodiment_tag=EmbodimentTag.resolve("NEW_EMBODIMENT"),
                model_path=str(checkpoint_path),
                device="cuda:0",
            )
        except Exception as exc:  # noqa: BLE001 - fail closed with checkpoint provenance
            raise GrootVisualizationError(f"{phase} checkpoint failed to load: {exc}") from exc

        envs = [
            gym.make(
                ENVIRONMENT_ID,
                obs_type="pixels_agent_pos",
                render_mode="rgb_array",
                observation_width=96,
                observation_height=96,
                visualization_width=680,
                visualization_height=680,
            )
            for _ in seeds
        ]
        observations: list[Any] = []
        initial_hashes: list[str] = []
        writers: list[_EpisodeWriter] = []
        trajectories: list[list[dict[str, Any]]] = []
        completed = [False] * len(seeds)
        termination_reasons = [""] * len(seeds)
        returns = [0.0] * len(seeds)
        success = [False] * len(seeds)
        forward_calls = 0
        model_action_steps = 0
        try:
            for index, (env, seed) in enumerate(zip(envs, seeds, strict=True)):
                obs, _ = env.reset(seed=seed)
                state = _environment_state(env)
                initial_hashes.append(_sha256(_json_bytes(state)))
                observations.append(obs)
                writer = _EpisodeWriter.create(root / f"seed-{seed}.mp4")
                writer.add(env.render())
                writers.append(writer)
                trajectories.append(
                    [{"step": 0, **state, "reward": 0.0, "success": False, "action": None}]
                )

            while not all(completed):
                active = [index for index, done in enumerate(completed) if not done]
                action_chunks: dict[int, Any] = {}
                chunk_horizon = -1
                for start in range(0, len(active), int(policy_batch_size)):
                    group = active[start : start + int(policy_batch_size)]
                    chunks = _policy_action_chunk(
                        policy, [observations[index] for index in group]
                    )
                    forward_calls += 1
                    if chunk_horizon not in {-1, int(chunks.shape[1])}:
                        raise GrootVisualizationError("model action horizon changed between batches")
                    chunk_horizon = int(chunks.shape[1])
                    for batch_index, episode_index in enumerate(group):
                        action_chunks[episode_index] = chunks[batch_index]
                if chunk_horizon < 1:
                    raise GrootVisualizationError("checkpoint emitted an empty action horizon")
                for chunk_step in range(chunk_horizon):
                    for episode_index in active:
                        if completed[episode_index]:
                            continue
                        env = envs[episode_index]
                        raw_action = action_chunks[episode_index][chunk_step].astype(np.float32)
                        applied_action = np.clip(raw_action, 0.0, 512.0).astype(np.float32)
                        obs, reward, terminated, truncated, info = env.step(applied_action)
                        observations[episode_index] = obs
                        returns[episode_index] += float(reward)
                        model_action_steps += 1
                        state = _environment_state(env)
                        step = len(trajectories[episode_index])
                        is_success = bool(info.get("is_success", False))
                        trajectories[episode_index].append(
                            {
                                "step": step,
                                **state,
                                "reward": float(reward),
                                "success": is_success,
                                "action": applied_action.tolist(),
                                "raw_model_action": raw_action.tolist(),
                            }
                        )
                        writers[episode_index].add(env.render())
                        reached_horizon = step >= int(horizon)
                        if terminated or truncated or reached_horizon:
                            completed[episode_index] = True
                            success[episode_index] = is_success
                            termination_reasons[episode_index] = (
                                "success"
                                if is_success
                                else "environment_terminated"
                                if terminated
                                else "environment_truncated"
                                if truncated
                                else "horizon"
                            )
                    if all(completed):
                        break
        finally:
            for writer in writers:
                writer.close()
            for env in envs:
                env.close()

        episodes_result: list[dict[str, Any]] = []
        for index, seed in enumerate(seeds):
            trajectory_path = root / f"seed-{seed}.json"
            trajectory_path.write_bytes(_json_bytes({"steps": trajectories[index]}))
            video_uri = rollout_prefix_uri.rstrip("/") + f"/seed-{seed}.mp4"
            trajectory_uri = rollout_prefix_uri.rstrip("/") + f"/seed-{seed}.json"
            video_artifact = _upload_file(client, root / f"seed-{seed}.mp4", video_uri)
            trajectory_artifact = _upload_file(client, trajectory_path, trajectory_uri)
            scores = [float(row["coverage"]) for row in trajectories[index]]
            final = trajectories[index][-1]
            episodes_result.append(
                {
                    "seed": seed,
                    "initial_state_sha256": initial_hashes[index],
                    "success": success[index],
                    "task_native_score": max(scores),
                    "final_coverage": float(final["coverage"]),
                    "return": returns[index],
                    "steps": int(final["step"]),
                    "termination_reason": termination_reasons[index],
                    "failure_category": "none" if success[index] else "goal_not_reached",
                    "final_goal_distance": float(
                        math.dist(final["object_xy"], final["goal_xy"])
                    ),
                    "video": video_artifact,
                    "trajectory": trajectory_artifact,
                }
            )
    if forward_calls <= 0 or model_action_steps != sum(item["steps"] for item in episodes_result):
        raise GrootVisualizationError("model-action accounting failed")
    result = {
        "schema": EVALUATION_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "phase": phase,
        "task_contract_uri": contract_uri,
        "checkpoint": {"uri": checkpoint_uri, **identity},
        "checkpoint_expected_sha256": checkpoint_sha256,
        "checkpoint_loaded": True,
        "closed_loop": True,
        "offline_replay": False,
        "physical_robot": False,
        "simulation": True,
        "environment": contract["environment"],
        "seed_namespace": seed_namespace,
        "seed_set_sha256": _sha256(_json_bytes({"seeds": seeds})),
        "seeds": seeds,
        "horizon": int(horizon),
        "policy_batch_size": int(policy_batch_size),
        "observation_preprocessing": (
            "current 96x96 simulator RGB + current pusher x/y + canonical PushT task text"
        ),
        "action_postprocessing": "finite check then component-wise clip to [0,512]",
        "action_source": "Gr00tPolicy.get_action model output on every environment step",
        "scripted_controller_used": False,
        "simulator_frames_live": True,
        "simulator_frame_source": "env.render() after each current gym-pusht physics transition",
        "model_forward_calls": forward_calls,
        "model_action_steps": model_action_steps,
        "gpu": torch.cuda.get_device_name(0),
        "episodes": episodes_result,
        "summary": {
            "episode_count": len(episodes_result),
            "successes": sum(bool(item["success"]) for item in episodes_result),
            "success_rate": sum(bool(item["success"]) for item in episodes_result)
            / len(episodes_result),
            "mean_task_native_score": sum(item["task_native_score"] for item in episodes_result)
            / len(episodes_result),
            "mean_return": sum(item["return"] for item in episodes_result) / len(episodes_result),
        },
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def select_checkpoint(
    validation_report_uri: str,
    checkpoint_ref_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Select only a candidate that passed the separate paired validation gate."""

    client = _s3_client(s3_client)
    report = _read_s3_json(client, validation_report_uri)
    reference = _read_s3_json(client, checkpoint_ref_uri)
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("run_id") != run_id
        or report.get("status") != "passed"
        or report.get("performance", {}).get("improvement_gate_passed") is not True
    ):
        raise GrootVisualizationError("candidate selection requires passed validation outcomes")
    if (
        reference.get("schema") != CHECKPOINT_REF_SCHEMA
        or reference.get("run_id") != run_id
        or reference.get("status") != "resolved"
    ):
        raise GrootVisualizationError("candidate checkpoint reference is invalid")
    checkpoint = dict(reference.get("checkpoint") or {})
    if len(str(checkpoint.get("sha256") or "")) != 64:
        raise GrootVisualizationError("candidate checkpoint is not immutable")
    result = {
        "schema": SELECTION_SCHEMA,
        "status": "selected",
        "run_id": run_id,
        "checkpoint": checkpoint,
        "checkpoint_ref_uri": checkpoint_ref_uri,
        "validation_report_uri": validation_report_uri,
        "validation_seed_set_sha256": report.get("paired_evaluation", {}).get(
            "seed_set_sha256"
        ),
        "selection_predicate": "paired closed-loop validation improvement gate passed",
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def paired_bootstrap(
    deltas: Sequence[float], *, samples: int = 50_000, seed: int = 17
) -> dict[str, float | int | str]:
    """Paired percentile bootstrap CI plus a one-sided sign-randomization test."""

    import numpy as np

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) < 20 or not np.all(np.isfinite(values)):
        raise GrootVisualizationError("paired evidence requires at least 20 finite deltas")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    randomization_samples = max(100_000, int(samples))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(randomization_samples, len(values)))
    randomized = (signs * values).mean(axis=1)
    observed = float(values.mean())
    p_value = float((1 + np.count_nonzero(randomized >= observed)) / (randomization_samples + 1))
    return {
        "method": "paired percentile bootstrap",
        "confidence_level": 0.95,
        "bootstrap_samples": int(samples),
        "ci_low": float(low),
        "ci_high": float(high),
        "mean_delta": observed,
        "paired_test": "one-sided paired sign-randomization Monte Carlo",
        "paired_test_samples": randomization_samples,
        "p_value": p_value,
    }


def analyze_paired_outcomes(
    contract_uri: str,
    baseline_uri: str,
    trained_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Pair identical initial conditions and enforce a real outcome improvement."""

    client = _s3_client(s3_client)
    contract = _read_s3_json(client, contract_uri)
    baseline = _read_s3_json(client, baseline_uri)
    trained = _read_s3_json(client, trained_uri)
    for document, phase in ((baseline, "baseline"), (trained, "trained")):
        if (
            document.get("schema") != EVALUATION_SCHEMA
            or document.get("run_id") != run_id
            or document.get("phase") != phase
            or document.get("closed_loop") is not True
            or document.get("offline_replay") is not False
            or document.get("checkpoint_loaded") is not True
            or document.get("scripted_controller_used") is not False
            or document.get("simulator_frames_live") is not True
            or not str(document.get("action_source", "")).startswith("Gr00tPolicy.get_action")
        ):
            raise GrootVisualizationError(f"{phase} evaluation lacks closed-loop truth evidence")
    if baseline.get("seeds") != trained.get("seeds") or baseline.get("seed_set_sha256") != trained.get(
        "seed_set_sha256"
    ):
        raise GrootVisualizationError("baseline and trained seed sets differ")
    if (
        baseline.get("horizon") != trained.get("horizon")
        or baseline.get("observation_preprocessing") != trained.get("observation_preprocessing")
        or baseline.get("action_postprocessing") != trained.get("action_postprocessing")
    ):
        raise GrootVisualizationError("paired evaluation conditions differ")
    baseline_by_seed = {int(item["seed"]): item for item in baseline["episodes"]}
    trained_by_seed = {int(item["seed"]): item for item in trained["episodes"]}
    if set(baseline_by_seed) != set(trained_by_seed) or len(baseline_by_seed) < 20:
        raise GrootVisualizationError("paired episode set is incomplete")
    paired: list[dict[str, Any]] = []
    for seed in baseline["seeds"]:
        before = baseline_by_seed[int(seed)]
        after = trained_by_seed[int(seed)]
        physical_initial_hash = _paired_initial_state_hash(client, before, after)
        paired.append(
            {
                "seed": int(seed),
                "initial_state_sha256": physical_initial_hash,
                "recorded_initial_hashes_equal": (
                    before["initial_state_sha256"] == after["initial_state_sha256"]
                ),
                "baseline_success": bool(before["success"]),
                "trained_success": bool(after["success"]),
                "success_delta": int(bool(after["success"])) - int(bool(before["success"])),
                "baseline_task_score": float(before["task_native_score"]),
                "trained_task_score": float(after["task_native_score"]),
                "task_score_delta": float(after["task_native_score"])
                - float(before["task_native_score"]),
                "baseline_return": float(before["return"]),
                "trained_return": float(after["return"]),
                "baseline_steps": int(before["steps"]),
                "trained_steps": int(after["steps"]),
                "baseline_termination_reason": before["termination_reason"],
                "trained_termination_reason": after["termination_reason"],
                "baseline_failure_category": before["failure_category"],
                "trained_failure_category": after["failure_category"],
                "baseline_final_coverage": float(before["final_coverage"]),
                "trained_final_coverage": float(after["final_coverage"]),
                "baseline_final_goal_distance": float(before["final_goal_distance"]),
                "trained_final_goal_distance": float(after["final_goal_distance"]),
                "baseline_video_uri": before["video"]["uri"],
                "trained_video_uri": after["video"]["uri"],
                "baseline_trajectory_uri": before["trajectory"]["uri"],
                "trained_trajectory_uri": after["trajectory"]["uri"],
                "outcome_category": (
                    "trained_win"
                    if after["success"] and not before["success"]
                    else "baseline_win"
                    if before["success"] and not after["success"]
                    else "both_success"
                    if before["success"] and after["success"]
                    else "both_failure"
                ),
            }
        )
    score_evidence = paired_bootstrap(
        [item["task_score_delta"] for item in paired],
        seed=int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16),
    )
    success_evidence = paired_bootstrap(
        [item["success_delta"] for item in paired],
        seed=int(hashlib.sha256((run_id + ":success").encode()).hexdigest()[:8], 16),
    )
    baseline_success_rate = sum(item["baseline_success"] for item in paired) / len(paired)
    trained_success_rate = sum(item["trained_success"] for item in paired) / len(paired)
    baseline_score = sum(item["baseline_task_score"] for item in paired) / len(paired)
    trained_score = sum(item["trained_task_score"] for item in paired) / len(paired)
    success_supported = (
        trained_success_rate > baseline_success_rate
        and float(success_evidence["ci_low"]) > 0.0
        and float(success_evidence["p_value"]) < 0.05
    )
    score_supported = (
        trained_score > baseline_score
        and float(score_evidence["ci_low"]) > 0.0
        and float(score_evidence["p_value"]) < 0.05
    )
    improvement_supported = success_supported or score_supported
    primary = "success_rate" if success_supported else "maximum_goal_coverage"
    primary_evidence = success_evidence if success_supported else score_evidence
    result = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if improvement_supported else "failed",
        "run_id": run_id,
        "task": contract["task"],
        "platform": {
            "label": "Simulated",
            "physical_robot": False,
            "simulation": True,
            "environment": contract["environment"],
            "embodiment": contract["embodiment"],
        },
        "observation": contract["observation"],
        "action": contract["action"],
        "success_definition": contract["success"],
        "dataset": contract["dataset"],
        "paired_evaluation": {
            "episode_count": len(paired),
            "same_initial_conditions": True,
            "seed_namespace": baseline["seed_namespace"],
            "seed_set_sha256": baseline["seed_set_sha256"],
            "horizon": baseline["horizon"],
            "observation_preprocessing": baseline["observation_preprocessing"],
            "action_postprocessing": baseline["action_postprocessing"],
            "baseline_checkpoint": baseline["checkpoint"],
            "trained_checkpoint": trained["checkpoint"],
            "baseline_evaluation_uri": baseline_uri,
            "trained_evaluation_uri": trained_uri,
            "episodes": paired,
        },
        "performance": {
            "primary_metric": primary,
            "higher_is_better": True,
            "baseline_success_rate": baseline_success_rate,
            "trained_success_rate": trained_success_rate,
            "success_rate_delta": trained_success_rate - baseline_success_rate,
            "baseline_task_score": baseline_score,
            "trained_task_score": trained_score,
            "task_score_delta": trained_score - baseline_score,
            "success_rate_evidence": success_evidence,
            "task_score_evidence": score_evidence,
            "primary_evidence": primary_evidence,
            "improvement_supported": improvement_supported,
            "improvement_gate_passed": improvement_supported,
            "conclusion": (
                "PASS: trained checkpoint improved closed-loop PushT outcomes"
                if improvement_supported
                else "FAIL: trained checkpoint lacks statistically supported closed-loop improvement"
            ),
        },
        "execution_truth": {
            "closed_loop": True,
            "offline_replay": False,
            "baseline_policy_loaded": True,
            "trained_policy_loaded": True,
            "actions_from_model": True,
            "scripted_controller_used": False,
            "simulator_frames_live": True,
        },
        "semantic_phases": SEMANTIC_PHASES,
        "diagnostics": {
            "training_and_offline_metrics": "secondary; see prior learning report",
        },
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not improvement_supported:
        raise GrootVisualizationError(result["performance"]["conclusion"])
    return result


def _decode_frames(path: Path):
    import av

    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def _selected_pairs(report: Mapping[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    episodes = list(report["paired_evaluation"]["episodes"])
    selected: list[dict[str, Any]] = []
    for category in ("trained_win", "both_success", "both_failure", "baseline_win"):
        candidates = [item for item in episodes if item["outcome_category"] == category]
        if candidates:
            selected.append(
                max(candidates, key=lambda item: abs(float(item["task_score_delta"])))
            )
    if not selected:
        raise GrootVisualizationError("paired report contains no rollout episodes")
    for item in sorted(episodes, key=lambda row: abs(float(row["task_score_delta"])), reverse=True):
        if item not in selected and len(selected) < limit:
            selected.append(item)
    return selected[:limit]


def _burn_comparison_frame(
    baseline: Any,
    trained: Any,
    *,
    pair: Mapping[str, Any],
    step: int,
    horizon: int,
) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1360, 760), "#0b1020")
    canvas.paste(Image.fromarray(np.asarray(baseline)), (0, 80))
    canvas.paste(Image.fromarray(np.asarray(trained)), (680, 80))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    draw.text((18, 10), f"PushT · Simulated · seed {pair['seed']} · step {step}/{horizon}", fill="white", font=font)
    draw.text(
        (18, 39),
        "Goal: push the T-shaped block onto the green T target (>95% coverage)",
        fill="#cbd5e1",
        font=small,
    )
    draw.rectangle((10, 705, 670, 750), fill="#162033")
    draw.rectangle((690, 705, 1350, 750), fill="#162033")
    draw.text(
        (22, 715),
        f"BASELINE · score {pair['baseline_task_score']:.3f} · {str(pair['baseline_success']).upper()} · {pair['baseline_termination_reason']}",
        fill="#fbbf24",
        font=small,
    )
    draw.text(
        (702, 715),
        f"TRAINED · score {pair['trained_task_score']:.3f} · {str(pair['trained_success']).upper()} · {pair['trained_termination_reason']}",
        fill="#34d399",
        font=small,
    )
    return np.asarray(canvas)


def render_task_rollouts(
    report_uri: str,
    video_uri: str,
    manifest_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Render synchronized same-seed comparisons from actual simulator videos."""

    import av

    client = _s3_client(s3_client)
    report = _read_s3_json(client, report_uri)
    if report.get("schema") != REPORT_SCHEMA or report.get("run_id") != run_id:
        raise GrootVisualizationError("task report identity mismatch")
    if report.get("performance", {}).get("improvement_gate_passed") is not True:
        raise GrootVisualizationError("rendering refuses a failed performance gate")
    selected = _selected_pairs(report)
    with tempfile.TemporaryDirectory(prefix="npa-groot-task-video-") as tmp:
        root = Path(tmp)
        output = root / "paired-task-performance.mp4"
        container = av.open(str(output), mode="w")
        try:
            stream = container.add_stream("libx264", rate=int(FPS))
        except Exception:  # noqa: BLE001
            stream = container.add_stream("mpeg4", rate=int(FPS))
        stream.width = 1360
        stream.height = 760
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "22", "preset": "veryfast"}
        total_frames = 0
        for pair in selected:
            seed = int(pair["seed"])
            baseline_path = root / f"baseline-{seed}.mp4"
            trained_path = root / f"trained-{seed}.mp4"
            _download(client, pair["baseline_video_uri"], baseline_path)
            _download(client, pair["trained_video_uri"], trained_path)
            baseline_iter = iter(_decode_frames(baseline_path))
            trained_iter = iter(_decode_frames(trained_path))
            baseline_last = next(baseline_iter, None)
            trained_last = next(trained_iter, None)
            if baseline_last is None or trained_last is None:
                raise GrootVisualizationError(f"seed {seed} rollout video is blank")
            step = 0
            while baseline_last is not None and trained_last is not None:
                composed = _burn_comparison_frame(
                    baseline_last,
                    trained_last,
                    pair=pair,
                    step=step,
                    horizon=int(report["paired_evaluation"]["horizon"]),
                )
                frame = av.VideoFrame.from_ndarray(composed, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                total_frames += 1
                step += 1
                before_next = next(baseline_iter, None)
                trained_next = next(trained_iter, None)
                if before_next is None and trained_next is None:
                    break
                if before_next is not None:
                    baseline_last = before_next
                if trained_next is not None:
                    trained_last = trained_next
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        if total_frames <= 0:
            raise GrootVisualizationError("comparison video contains no simulator frames")
        artifact = _upload_file(client, output, video_uri)
    categories = sorted({str(item["outcome_category"]) for item in selected})
    has_success = any(item["baseline_success"] or item["trained_success"] for item in selected)
    has_failure = any(not item["baseline_success"] or not item["trained_success"] for item in selected)
    result = {
        "schema": RENDER_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "artifact": artifact,
        "resolution": "1360x760",
        "fps": FPS,
        "encoding": "H.264/yuv420p (MPEG-4 fallback only if H.264 unavailable)",
        "frame_count": total_frames,
        "actual_rollout_frames": True,
        "simulator_native_panel_resolution": "680x680 per policy",
        "simulation_label_present": True,
        "selected_seeds": [int(item["seed"]) for item in selected],
        "outcome_categories": categories,
        "representative_successes_and_failures": has_success and has_failure,
        "selection_rule": "one largest-delta example per outcome category, then largest absolute paired deltas",
    }
    _put_json(client, manifest_uri, result)
    report["visualizations"] = {"comparison_video": result, "render_manifest_uri": manifest_uri}
    _put_json(client, report_uri, report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _materialize_replay_bundle(
    client: Any, report: Mapping[str, Any], render: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    selected_seed = int(render["selected_seeds"][0])
    pair = next(
        item
        for item in report["paired_evaluation"]["episodes"]
        if int(item["seed"]) == selected_seed
    )
    baseline_video = root / "baseline.mp4"
    trained_video = root / "trained.mp4"
    baseline_trajectory = root / "baseline.json"
    trained_trajectory = root / "trained.json"
    _download(client, pair["baseline_video_uri"], baseline_video)
    _download(client, pair["trained_video_uri"], trained_video)
    _download(client, pair["baseline_trajectory_uri"], baseline_trajectory)
    _download(client, pair["trained_trajectory_uri"], trained_trajectory)
    return pair, baseline_video, trained_video, baseline_trajectory, trained_trajectory


def _paired_replay_timebase(
    pair: Mapping[str, Any],
    baseline_steps: Sequence[Mapping[str, Any]],
    trained_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One explicit selected-episode clock shared by MCAP and RRD."""

    core = {
        "semantics": "selected-paired-rollout-step-at-simulator-fps",
        "seed": int(pair["seed"]),
        "fps": FPS,
        "baseline_step_count": len(baseline_steps),
        "trained_step_count": len(trained_steps),
        "sample_count": max(len(baseline_steps), len(trained_steps)),
        "episode_boundaries": [
            {
                "seed": int(pair["seed"]),
                "start_sample": 0,
                "end_sample_exclusive": max(len(baseline_steps), len(trained_steps)),
            }
        ],
    }
    return {**core, "id": "sha256:" + _sha256(_json_bytes(core))}


def emit_task_mcap(
    report_uri: str,
    render_manifest_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Emit named Lichtblick topics from live rollout frames and state."""

    from PIL import Image

    client = _s3_client(s3_client)
    report = _read_s3_json(client, report_uri)
    render = _read_s3_json(client, render_manifest_uri)
    if report.get("run_id") != run_id or render.get("run_id") != run_id:
        raise GrootVisualizationError("MCAP inputs belong to another run")
    with tempfile.TemporaryDirectory(prefix="npa-groot-task-mcap-") as tmp:
        root = Path(tmp)
        pair, baseline_video, trained_video, baseline_path, trained_path = _materialize_replay_bundle(
            client, report, render, root
        )
        baseline_steps = json.loads(baseline_path.read_text())["steps"]
        trained_steps = json.loads(trained_path.read_text())["steps"]
        timebase = _paired_replay_timebase(pair, baseline_steps, trained_steps)
        base_ns = 1
        step_ns = int(1_000_000_000 / FPS)
        frames: list[FrameInput] = []
        for phase, video in (("baseline", baseline_video), ("trained", trained_video)):
            phase_dir = root / phase
            phase_dir.mkdir()
            for index, frame in enumerate(_decode_frames(video)):
                frame_path = phase_dir / f"{index:04d}.jpg"
                Image.fromarray(frame).save(frame_path, format="JPEG", quality=88)
                frames.append(
                    FrameInput(
                        path=frame_path,
                        camera=phase,
                        timestamp_ns=base_ns + index * step_ns,
                        topic=f"/rollout/{phase}/camera",
                    )
                )
        metric_specs: list[tuple[str, str, Any]] = [
            (
                "baseline_action",
                "/rollout/baseline/action",
                [
                    {
                        "_timestamp_ns": base_ns + int(row["step"]) * step_ns,
                        "seed": pair["seed"],
                        "step": row["step"],
                        "x": row["action"][0],
                        "y": row["action"][1],
                    }
                    for row in baseline_steps
                    if row["action"] is not None
                ],
            ),
            (
                "trained_action",
                "/rollout/trained/action",
                [
                    {
                        "_timestamp_ns": base_ns + int(row["step"]) * step_ns,
                        "seed": pair["seed"],
                        "step": row["step"],
                        "x": row["action"][0],
                        "y": row["action"][1],
                    }
                    for row in trained_steps
                    if row["action"] is not None
                ],
            ),
            (
                "object_pose",
                "/rollout/object_pose",
                [
                    {
                        "phase": phase,
                        "_timestamp_ns": base_ns + int(row["step"]) * step_ns,
                        "seed": pair["seed"],
                        "step": row["step"],
                        "x": row["object_xy"][0],
                        "y": row["object_xy"][1],
                        "angle_rad": row["object_angle_rad"],
                    }
                    for phase, rows in (("baseline", baseline_steps), ("trained", trained_steps))
                    for row in rows
                ],
            ),
            (
                "goal_pose",
                "/rollout/goal_pose",
                [
                    {
                        "phase": phase,
                        "_timestamp_ns": base_ns + int(row["step"]) * step_ns,
                        "seed": pair["seed"],
                        "step": row["step"],
                        "x": row["goal_xy"][0],
                        "y": row["goal_xy"][1],
                        "angle_rad": row["goal_angle_rad"],
                    }
                    for phase, rows in (("baseline", baseline_steps), ("trained", trained_steps))
                    for row in rows
                ],
            ),
            (
                "task_progress",
                "/rollout/task_progress",
                [
                    {
                        "phase": phase,
                        "_timestamp_ns": base_ns + int(row["step"]) * step_ns,
                        "seed": pair["seed"],
                        "step": row["step"],
                        "coverage": row["coverage"],
                        "reward": row["reward"],
                    }
                    for phase, rows in (("baseline", baseline_steps), ("trained", trained_steps))
                    for row in rows
                ],
            ),
            (
                "success",
                "/rollout/success",
                [
                    {
                        "phase": "baseline",
                        "_timestamp_ns": base_ns
                        + int(pair["baseline_steps"]) * step_ns,
                        "seed": pair["seed"],
                        "success": pair["baseline_success"],
                        "termination_reason": pair["baseline_termination_reason"],
                    },
                    {
                        "phase": "trained",
                        "_timestamp_ns": base_ns
                        + int(pair["trained_steps"]) * step_ns,
                        "seed": pair["seed"],
                        "success": pair["trained_success"],
                        "termination_reason": pair["trained_termination_reason"],
                    },
                ],
            ),
            (
                "baseline_success_rate",
                "/metrics/baseline_success_rate",
                {
                    "_timestamp_ns": base_ns,
                    "value": report["performance"]["baseline_success_rate"],
                },
            ),
            (
                "trained_success_rate",
                "/metrics/trained_success_rate",
                {
                    "_timestamp_ns": base_ns,
                    "value": report["performance"]["trained_success_rate"],
                },
            ),
            (
                "paired_delta",
                "/metrics/paired_delta",
                [
                    {
                        "_timestamp_ns": base_ns
                        + round(
                            index
                            * max(timebase["sample_count"] - 1, 0)
                            / max(len(report["paired_evaluation"]["episodes"]) - 1, 1)
                        )
                        * step_ns,
                        "seed": item["seed"],
                        "success_delta": item["success_delta"],
                        "task_score_delta": item["task_score_delta"],
                    }
                    for index, item in enumerate(report["paired_evaluation"]["episodes"])
                ],
            ),
        ]
        metrics: list[MetricsInput] = []
        for name, topic, payload in metric_specs:
            path = root / f"{name}.json"
            path.write_bytes((json.dumps(payload, sort_keys=True) + "\n").encode())
            metrics.append(MetricsInput(path=path, name=name, topic=topic, timestamp_ns=base_ns))
        log_path = root / "rollout.log"
        log_path.write_text(
            "\n".join(
                [
                    f"PushT Simulated paired seed {pair['seed']}",
                    f"baseline outcome={pair['baseline_success']} reason={pair['baseline_termination_reason']}",
                    f"trained outcome={pair['trained_success']} reason={pair['trained_termination_reason']}",
                    report["performance"]["conclusion"],
                ]
            )
            + "\n"
        )
        output = root / "task-performance.mcap"
        summary = write_run_mcap(
            output=output,
            frames=frames,
            metrics=metrics,
            logs=[LogInput(path=log_path, name="groot_task_performance")],
            fps=FPS,
            start_time_ns=base_ns,
            run_id=run_id,
            metadata={
                "task": "PushT",
                "platform": "Simulated",
                "frame_source": "live gym-pusht env.render() rollout",
                "producer": "npa.groot.task-performance",
                "timestamps": str(timebase["semantics"]),
                "timebase_id": str(timebase["id"]),
                "episode_boundaries_sha256": _sha256(
                    _json_bytes({"episode_boundaries": timebase["episode_boundaries"]})
                ),
            },
        )
        inspection = summarize_mcap(output).to_dict()
        topics = set(summary.channels)
        missing = sorted(REQUIRED_MCAP_TOPICS - topics)
        if missing or summary.frames < 2:
            raise GrootVisualizationError(f"task MCAP missing required live topics: {missing}")
        artifact = _upload_file(client, output, output_uri)
    result = {
        "status": "written",
        "artifact": artifact,
        "summary": summary.to_dict(),
        "inspect": inspection,
        "required_topics": sorted(REQUIRED_MCAP_TOPICS),
        "timebase": timebase,
        "browser_compatible": True,
    }
    report.setdefault("visualizations", {})["mcap"] = result
    _put_json(client, report_uri, report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _set_rerun_time(rr: Any, recording: Any, seconds: float) -> None:
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("rollout_time", seconds, recording=recording)
    else:
        rr.set_time("rollout_time", duration=seconds, recording=recording)


def _task_blueprint(rrb: Any) -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="rollout/baseline", name="BASELINE · Simulated rollout"),
            rrb.Spatial2DView(origin="rollout/trained", name="TRAINED · Simulated rollout"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="task", contents="task/**", name="Object / goal trajectory"),
                rrb.TimeSeriesView(
                    origin="task/progress", contents="task/progress/**", name="Task-native coverage"
                ),
                rrb.TimeSeriesView(
                    origin="aggregate", contents="aggregate/**", name="Paired outcomes and 95% CI"
                ),
                rrb.TextDocumentView(origin="task/outcome", name="Performance card and outcomes"),
            ),
            column_shares=[2.2, 2.2, 1.8],
        ),
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TimePanel(state=rrb.PanelState.Expanded, timeline="rollout_time"),
        auto_layout=False,
    )


def emit_task_rrd(
    report_uri: str,
    render_manifest_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Emit a Rerun recording whose primary layout is task performance."""

    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb

    client = _s3_client(s3_client)
    report = _read_s3_json(client, report_uri)
    render = _read_s3_json(client, render_manifest_uri)
    if report.get("run_id") != run_id or render.get("run_id") != run_id:
        raise GrootVisualizationError("RRD inputs belong to another run")
    with tempfile.TemporaryDirectory(prefix="npa-groot-task-rrd-") as tmp:
        root = Path(tmp)
        pair, baseline_video, trained_video, baseline_path, trained_path = _materialize_replay_bundle(
            client, report, render, root
        )
        baseline_steps = json.loads(baseline_path.read_text())["steps"]
        trained_steps = json.loads(trained_path.read_text())["steps"]
        timebase = _paired_replay_timebase(pair, baseline_steps, trained_steps)
        rrd = root / "task-performance.rrd"
        blueprint = _task_blueprint(rrb)
        recording = rr.RecordingStream("npa_groot_task_performance", recording_id=run_id)
        rr.save(rrd, default_blueprint=blueprint, recording=recording)
        if hasattr(rr, "send_blueprint"):
            rr.send_blueprint(blueprint, recording=recording)
        performance = report["performance"]
        evidence = performance["primary_evidence"]
        performance_card = (
            "# PushT task performance\n\n"
            "**Platform: Simulated** (`gym_pusht/PushT-v0`, not physical hardware)\n\n"
            f"Goal: {report['task']['goal']}\n\n"
            f"Paired episodes: {report['paired_evaluation']['episode_count']}\n\n"
            f"Baseline success: {performance['baseline_success_rate']:.1%}  \n"
            f"Trained success: {performance['trained_success_rate']:.1%}  \n"
            f"Absolute delta: {performance['success_rate_delta']:+.1%}  \n"
            f"Task-score delta: {performance['task_score_delta']:+.4f}  \n"
            f"95% paired CI: [{evidence['ci_low']:+.4f}, {evidence['ci_high']:+.4f}]  \n"
            f"Test: {evidence['paired_test']}, p={evidence['p_value']:.4g}\n\n"
            f"**{performance['conclusion']}**\n\n"
            f"Replay seed: {pair['seed']} · baseline {pair['baseline_termination_reason']} · "
            f"trained {pair['trained_termination_reason']}\n\n"
            "Actions: absolute target pusher x/y in workspace pixels [0,512].\n\n"
            "Training loss and offline action MSE are secondary diagnostics, not task evidence."
        )
        rr.log("task/outcome", rr.TextDocument(performance_card), static=True, recording=recording)
        rr.log(
            "provenance",
            rr.TextDocument(
                f"Dataset `{DATASET_ID}@{DATASET_REVISION}`; environment "
                f"`{ENVIRONMENT_ID}` package {ENVIRONMENT_VERSION}, source `{ENVIRONMENT_REVISION}`; "
                f"frames are current env.render() output; timebase `{timebase['id']}`."
            ),
            static=True,
            recording=recording,
        )
        baseline_iter = iter(_decode_frames(baseline_video))
        trained_iter = iter(_decode_frames(trained_video))
        baseline_last = next(baseline_iter, None)
        trained_last = next(trained_iter, None)
        if baseline_last is None or trained_last is None:
            raise GrootVisualizationError("RRD source rollout video is blank")
        index = 0
        baseline_track: list[list[float]] = []
        trained_track: list[list[float]] = []
        while baseline_last is not None and trained_last is not None:
            _set_rerun_time(rr, recording, index / FPS)
            rr.log("rollout/baseline/camera", rr.Image(baseline_last, color_model="RGB"), recording=recording)
            rr.log("rollout/trained/camera", rr.Image(trained_last, color_model="RGB"), recording=recording)
            b_row = baseline_steps[min(index, len(baseline_steps) - 1)]
            t_row = trained_steps[min(index, len(trained_steps) - 1)]
            baseline_track.append(b_row["object_xy"])
            trained_track.append(t_row["object_xy"])
            rr.log(
                "task/object_pose",
                rr.Points2D([b_row["object_xy"], t_row["object_xy"]], colors=[[251, 191, 36], [52, 211, 153]], radii=5),
                recording=recording,
            )
            rr.log(
                "task/goal_pose",
                rr.Points2D([b_row["goal_xy"]], colors=[[34, 197, 94]], radii=8),
                recording=recording,
            )
            rr.log(
                "task/trajectory/baseline",
                rr.LineStrips2D([np.asarray(baseline_track)], colors=[[251, 191, 36]]),
                recording=recording,
            )
            rr.log(
                "task/trajectory/trained",
                rr.LineStrips2D([np.asarray(trained_track)], colors=[[52, 211, 153]]),
                recording=recording,
            )
            rr.log("task/progress/baseline", rr.Scalars(float(b_row["coverage"])), recording=recording)
            rr.log("task/progress/trained", rr.Scalars(float(t_row["coverage"])), recording=recording)
            rr.log(
                "task/success/baseline",
                rr.Scalars(float(bool(b_row["success"]))),
                recording=recording,
            )
            rr.log(
                "task/success/trained",
                rr.Scalars(float(bool(t_row["success"]))),
                recording=recording,
            )
            before_next = next(baseline_iter, None)
            trained_next = next(trained_iter, None)
            if before_next is None and trained_next is None:
                break
            if before_next is not None:
                baseline_last = before_next
            if trained_next is not None:
                trained_last = trained_next
            index += 1
        replay_duration = max(index / FPS, 1.0)
        paired = report["paired_evaluation"]["episodes"]
        for paired_index, item in enumerate(paired):
            _set_rerun_time(rr, recording, replay_duration * paired_index / max(1, len(paired) - 1))
            rr.log(
                "aggregate/paired_delta",
                rr.Scalars(float(item["task_score_delta"])),
                recording=recording,
            )
            rr.log(
                "aggregate/baseline_success_rate",
                rr.Scalars(float(performance["baseline_success_rate"])),
                recording=recording,
            )
            rr.log(
                "aggregate/trained_success_rate",
                rr.Scalars(float(performance["trained_success_rate"])),
                recording=recording,
            )
            rr.log(
                "aggregate/confidence_interval",
                rr.Scalars([float(evidence["ci_low"]), float(evidence["ci_high"])]),
                recording=recording,
            )
        recording.flush(timeout_sec=60.0)
        recording.disconnect()
        inspection = inspect_rrd(
            rrd,
            application_id="npa_groot_task_performance",
            recording_id=run_id,
            expected_entities=REQUIRED_RRD_ENTITIES,
            timeline="rollout_time",
        )
        artifact = _put_bytes(client, output_uri, rrd.read_bytes())
    result = {
        "status": "written",
        "artifact": artifact,
        "inspect": inspection,
        "outcome_panels": [
            "side-by-side baseline/trained live camera",
            "object/goal trajectory",
            "task-native coverage",
            "success/termination outcome",
            "aggregate success rates",
            "paired delta and 95% confidence interval",
        ],
        "timebase": timebase,
        "browser_compatible": True,
    }
    report.setdefault("visualizations", {})["rrd"] = result
    _put_json(client, report_uri, report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def publish_task_performance(
    report_uri: str,
    render_manifest_uri: str,
    video_uri: str,
    mcap_uri: str,
    rrd_uri: str,
    workflow_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Independently parse and gate every user-facing task artifact."""

    import av

    client = _s3_client(s3_client)
    report = _read_s3_json(client, report_uri)
    render = _read_s3_json(client, render_manifest_uri)
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("run_id") != run_id
        or report.get("performance", {}).get("improvement_gate_passed") is not True
        or report.get("execution_truth", {}).get("closed_loop") is not True
        or report.get("execution_truth", {}).get("actions_from_model") is not True
        or render.get("actual_rollout_frames") is not True
        or render.get("simulation_label_present") is not True
        or render.get("representative_successes_and_failures") is not True
    ):
        raise GrootVisualizationError("publish refuses an incomplete task-performance truth gate")
    mcap_timebase = report.get("visualizations", {}).get("mcap", {}).get("timebase", {})
    rrd_timebase = report.get("visualizations", {}).get("rrd", {}).get("timebase", {})
    if not mcap_timebase.get("id") or mcap_timebase.get("id") != rrd_timebase.get("id"):
        raise GrootVisualizationError("MCAP and RRD do not share the same replay timebase")
    with tempfile.TemporaryDirectory(prefix="npa-groot-task-publish-") as tmp:
        root = Path(tmp)
        video = root / "task-performance.mp4"
        mcap = root / "task-performance.mcap"
        rrd = root / "task-performance.rrd"
        workflow = root / "workflow.yaml"
        for uri, path in ((video_uri, video), (mcap_uri, mcap), (rrd_uri, rrd), (workflow_uri, workflow)):
            _download(client, uri, path)
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            decoded = sum(1 for _ in container.decode(video=0))
            resolution = f"{stream.width}x{stream.height}"
        if decoded <= 0 or resolution != "1360x760":
            raise GrootVisualizationError("comparison video is blank or has the wrong resolution")
        mcap_inspection = summarize_mcap(mcap).to_dict()
        mcap_topics = set(mcap_inspection.get("channels", {}))
        missing_topics = sorted(REQUIRED_MCAP_TOPICS - mcap_topics)
        if missing_topics:
            raise GrootVisualizationError(f"published MCAP lacks topics: {missing_topics}")
        rrd_inspection = inspect_rrd(
            rrd,
            application_id="npa_groot_task_performance",
            recording_id=run_id,
            expected_entities=REQUIRED_RRD_ENTITIES,
            timeline="rollout_time",
        )
        workflow_text = workflow.read_text()
        for phase in SEMANTIC_PHASES:
            if f"  {phase}:" not in workflow_text:
                raise GrootVisualizationError(f"submitted workflow omits semantic phase {phase}")
        artifacts = {
            "report": _head_artifact(client, report_uri),
            "comparison_video": _head_artifact(client, video_uri),
            "mcap": _head_artifact(client, mcap_uri),
            "rrd": _head_artifact(client, rrd_uri),
            "workflow": _head_artifact(client, workflow_uri),
        }
    result = {
        "schema": PUBLISH_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "performance_gate_passed": True,
        "closed_loop": True,
        "physical_robot": False,
        "simulation": True,
        "artifacts": artifacts,
        "video_inspection": {
            "frames": decoded,
            "resolution": resolution,
            "browser_playable": True,
            "actual_rollout_frames": True,
        },
        "mcap_inspection": mcap_inspection,
        "rrd_inspection": rrd_inspection,
        "semantic_phases": SEMANTIC_PHASES,
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    contract = commands.add_parser("resolve-task-contract")
    contract.add_argument("--source-uri", required=True)
    contract.add_argument("--output-uri", required=True)
    contract.add_argument("--run-id", required=True)

    evaluate = commands.add_parser("evaluate-closed-loop")
    evaluate.add_argument("--contract-uri", required=True)
    evaluate.add_argument("--checkpoint-uri")
    evaluate.add_argument("--checkpoint-ref-uri")
    evaluate.add_argument("--output-uri", required=True)
    evaluate.add_argument("--rollout-prefix-uri", required=True)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--phase", choices=("baseline", "trained"), required=True)
    evaluate.add_argument("--checkpoint-sha256")
    evaluate.add_argument("--allow-selection-ref", action="store_true")
    evaluate.add_argument("--seed-namespace", required=True)
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--horizon", type=int, default=HORIZON)
    evaluate.add_argument("--policy-batch-size", type=int, default=4)

    analyze = commands.add_parser("analyze-paired-outcomes")
    analyze.add_argument("--contract-uri", required=True)
    analyze.add_argument("--baseline-uri", required=True)
    analyze.add_argument("--trained-uri", required=True)
    analyze.add_argument("--output-uri", required=True)
    analyze.add_argument("--run-id", required=True)

    checkpoint = commands.add_parser("resolve-trained-checkpoint")
    checkpoint.add_argument("--training-manifest-uri", required=True)
    checkpoint.add_argument("--split-manifest-uri", required=True)
    checkpoint.add_argument("--checkpoint-uri", required=True)
    checkpoint.add_argument("--output-uri", required=True)
    checkpoint.add_argument("--run-id", required=True)
    checkpoint.add_argument("--expected-gpu-count", type=int, required=True)
    checkpoint.add_argument("--expected-max-steps", type=int, required=True)

    select = commands.add_parser("select-checkpoint")
    select.add_argument("--validation-report-uri", required=True)
    select.add_argument("--checkpoint-ref-uri", required=True)
    select.add_argument("--output-uri", required=True)
    select.add_argument("--run-id", required=True)

    render = commands.add_parser("render-task-rollouts")
    render.add_argument("--report-uri", required=True)
    render.add_argument("--video-uri", required=True)
    render.add_argument("--manifest-uri", required=True)
    render.add_argument("--run-id", required=True)

    mcap = commands.add_parser("emit-mcap")
    mcap.add_argument("--report-uri", required=True)
    mcap.add_argument("--render-manifest-uri", required=True)
    mcap.add_argument("--output-uri", required=True)
    mcap.add_argument("--run-id", required=True)

    rrd = commands.add_parser("emit-rrd")
    rrd.add_argument("--report-uri", required=True)
    rrd.add_argument("--render-manifest-uri", required=True)
    rrd.add_argument("--output-uri", required=True)
    rrd.add_argument("--run-id", required=True)

    publish = commands.add_parser("publish")
    publish.add_argument("--report-uri", required=True)
    publish.add_argument("--render-manifest-uri", required=True)
    publish.add_argument("--video-uri", required=True)
    publish.add_argument("--mcap-uri", required=True)
    publish.add_argument("--rrd-uri", required=True)
    publish.add_argument("--workflow-uri", required=True)
    publish.add_argument("--output-uri", required=True)
    publish.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values = vars(args).copy()
    command = values.pop("command")
    if command == "resolve-task-contract":
        resolve_task_contract(**values)
    elif command == "evaluate-closed-loop":
        evaluate_closed_loop(**values)
    elif command == "analyze-paired-outcomes":
        analyze_paired_outcomes(**values)
    elif command == "resolve-trained-checkpoint":
        resolve_trained_checkpoint(**values)
    elif command == "select-checkpoint":
        select_checkpoint(**values)
    elif command == "render-task-rollouts":
        render_task_rollouts(**values)
    elif command == "emit-mcap":
        emit_task_mcap(**values)
    elif command == "emit-rrd":
        emit_task_rrd(**values)
    elif command == "publish":
        publish_task_performance(**values)
    else:  # pragma: no cover
        raise GrootVisualizationError(f"unknown command: {command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
