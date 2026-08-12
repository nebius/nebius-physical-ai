"""Leakage-free GR00T N1.7 learning, evaluation, and replay workflow.

This module deliberately fails closed.  Evaluation artifacts are accepted only
when they were produced by real ``Gr00tPolicy.get_action`` calls over a held-out
episode split.  The synchronized MCAP, RRD, and comparison video are labelled
as offline held-out evaluation; none of them is a closed-loop rollout.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from npa.workbench.foxglove.inspect import summarize_mcap
from npa.workbench.foxglove.mcap_writer import (
    FrameInput,
    LogInput,
    MetricsInput,
    write_run_mcap,
)
from npa.workflows.groot_visualization import (
    GrootVisualizationError,
    _download,
    _head_artifact,
    _list_objects,
    _put_bytes,
    _put_json,
    _read_s3_bytes,
    _read_s3_json,
    _s3_client,
    _split_s3,
    inspect_rrd,
)


SPLIT_SCHEMA = "npa.groot.episode_split.v1"
EVAL_SCHEMA = "npa.groot.offline_eval.v1"
REPORT_SCHEMA = "npa.groot.learning.v1"
PUBLISH_SCHEMA = "npa.groot.learning_publish.v1"
CHECKPOINT_REF_SCHEMA = "npa.groot.checkpoint_ref.v1"
EVALUATION_KIND = "offline held-out policy evaluation"
TIMESTAMP_SEMANTICS = "dataset-index-at-recorded-fps"
RERUN_APPLICATION_ID = "npa_groot_offline_learning"
RERUN_TIMELINE = "dataset_time"
SEMANTIC_PHASES = [
    "access_capacity_preflight",
    "prepare_deterministic_split",
    "baseline_inference_evaluation",
    "distributed_training",
    "trained_checkpoint_resolution",
    "post_training_inference_evaluation",
    "classify_learning_outcome",
    "generate_rrd",
    "generate_mcap",
    "publish_artifacts_run_summary",
    "agent_ui_load_viewer_verification",
]
REQUIRED_MCAP_TOPICS = {
    "/camera/front": "foxglove.CompressedImage",
    "/policy/predicted_action": "npa.RunMetrics.policy/predicted_action",
    "/expert/action": "npa.RunMetrics.expert/action",
    "/metrics/action_error": "npa.RunMetrics.metrics/action_error",
    "/metrics/heldout_before": "npa.RunMetrics.metrics/heldout_before",
    "/metrics/heldout_after": "npa.RunMetrics.metrics/heldout_after",
    "/metrics/train_loss": "npa.RunMetrics.metrics/train_loss",
    "/metrics/per_horizon_error": "npa.RunMetrics.metrics/per_horizon_error",
    "/metrics/checkpoint_curve": "npa.RunMetrics.metrics/checkpoint_curve",
    "/log": "foxglove.Log",
}
REQUIRED_RRD_ENTITIES = [
    "heldout/camera/front",
    "actions/expert/dim_0",
    "actions/predicted_before/dim_0",
    "actions/predicted_after/dim_0",
    "error/before/absolute",
    "error/after/absolute",
    "metrics/heldout_before/mse",
    "metrics/heldout_after/mse",
    "train/loss",
    "metrics/per_horizon/baseline_mse",
    "metrics/per_horizon/posttrain_mse",
    "validation/checkpoint_mse",
    "provenance",
]
CUSTOM_DATASET_METADATA = (
    "npa_groot_adapter.json",
    "npa_groot_modality_config.py",
)

# Every surface that constructs or loads the PushT policy must agree on these
# values.  In particular, PushT actions are absolute workspace targets; the
# prior continuation accidentally enabled the global relative-action mode and
# produced a model that was materially worse than a train-mean predictor.
GROOT_MODEL_CONFIG_CONTRACT: dict[str, Any] = {
    "tune_projector": True,
    "tune_diffusion_model": True,
    "load_bf16": False,
    "reproject_vision": False,
    "backbone_trainable_params_fp32": True,
    "use_relative_action": False,
    "action_representation": "ABSOLUTE",
}

DEFAULT_MINIMUM_SKILL_SCORE = 0.05
DEFAULT_MINIMUM_RELATIVE_IMPROVEMENT = 0.10
DEFAULT_REPEAT_NOISE_MULTIPLE = 3.0
DEFAULT_MAX_DIMENSION_REGRESSION = 0.0
DEFAULT_MINIMUM_EPOCHS = 2.0
DEFAULT_MINIMUM_EFFECTIVE_GLOBAL_BATCH = 128
WEIGHT_FILE_PATTERNS = (
    re.compile(r"(?:^|/)model(?:-\d+-of-\d+)?\.safetensors$"),
    re.compile(r"(?:^|/)pytorch_model(?:-\d+-of-\d+)?\.bin$"),
)


def preflight_rigor_contract(
    output_uri: str,
    run_id: str,
    *,
    gpu_type: str,
    gpu_count: int,
    global_batch_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    train_episodes: int,
    validation_episodes: int,
    final_episodes: int,
    max_steps: int,
    save_steps: int,
    save_total_limit: int,
    minimum_epochs: float,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Fail before GPU scheduling when the operational training contract is incoherent.

    The reference workflow is a plumbing validation, not a statistical-learning
    experiment.  It therefore requires real optimizer work and a checkpoint, but
    does not manufacture a 10k-step or multi-epoch minimum.  Coverage is still
    derived from the materialized split and reported exactly.
    """

    effective = (
        int(gpu_count) * int(per_device_batch_size) * int(gradient_accumulation_steps)
    )
    if int(gpu_count) < 1:
        raise GrootVisualizationError("GPU count must be at least one")
    if effective != int(global_batch_size) or effective < 1:
        raise GrootVisualizationError(
            "declared effective global batch contract is invalid"
        )
    if int(train_episodes) < 1 or int(validation_episodes) < 1:
        raise GrootVisualizationError("train and held-out cohorts must be non-empty")
    if int(final_episodes) < 0:
        raise GrootVisualizationError("final episode count cannot be negative")
    if int(max_steps) < 2 or float(minimum_epochs) <= 0.0:
        raise GrootVisualizationError(
            "pipeline validation requires at least two optimizer steps and positive coverage"
        )
    if int(save_steps) != int(max_steps) or int(save_total_limit) < 1:
        raise GrootVisualizationError(
            "the final optimizer step must be saved as an available checkpoint"
        )
    result = {
        "schema": "npa.groot.rigor_preflight.v1",
        "status": "passed",
        "run_id": run_id,
        "gpu": {"type": gpu_type, "count": int(gpu_count)},
        "batch": {
            "per_device": int(per_device_batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "effective_global": effective,
        },
        "episodes": {
            "train": int(train_episodes),
            "validation": int(validation_episodes),
            "final": int(final_episodes),
        },
        "max_steps": int(max_steps),
        "checkpoint_schedule": {
            "save_steps": int(save_steps),
            "save_total_limit": int(save_total_limit),
            "final_step_checkpoint_required": True,
        },
        "minimum_epochs": float(minimum_epochs),
        "validation_mode": "operational_pipeline_smoke",
        "statistical_learning_claim": False,
        "absolute_actions": True,
        "use_relative_action": False,
        "control_plane_capacity": (
            "enforced separately by workflow planner/scheduler before trainer provisioning"
        ),
    }
    _put_json(_s3_client(s3_client), output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()


def deterministic_episode_split(
    episode_count: int,
    *,
    train_episodes: int,
    heldout_episodes: int,
    seed: str,
) -> dict[str, list[int]]:
    """Select a stable disjoint experiment cohort at episode granularity."""

    if episode_count < train_episodes + heldout_episodes:
        raise GrootVisualizationError(
            "dataset has too few episodes for requested split"
        )
    if train_episodes < 1 or heldout_episodes < 1:
        raise GrootVisualizationError(
            "train and held-out episode counts must be positive"
        )
    order = sorted(
        range(episode_count),
        key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).hexdigest(),
    )
    heldout = order[:heldout_episodes]
    train = order[heldout_episodes : heldout_episodes + train_episodes]
    if set(train) & set(heldout):
        raise GrootVisualizationError("episode leakage detected in deterministic split")
    return {
        "train": train,
        "heldout": heldout,
        "excluded": order[len(train) + len(heldout) :],
    }


def deterministic_experiment_split(
    episode_count: int,
    *,
    train_episodes: int,
    validation_episodes: int,
    final_episodes: int,
    seed: str,
) -> dict[str, list[int]]:
    """Create a stable train/validation/final split without episode leakage."""

    requested = int(train_episodes) + int(validation_episodes) + int(final_episodes)
    if min(train_episodes, validation_episodes, final_episodes) < 1:
        raise GrootVisualizationError(
            "train, validation, and final episode counts must be positive"
        )
    if requested > int(episode_count):
        raise GrootVisualizationError(
            "dataset has too few episodes for train/validation/final split"
        )
    order = sorted(
        range(int(episode_count)),
        key=lambda index: hashlib.sha256(f"{seed}:{index}".encode()).hexdigest(),
    )
    validation_end = int(validation_episodes)
    final_end = validation_end + int(final_episodes)
    train_end = final_end + int(train_episodes)
    result = {
        "validation": order[:validation_end],
        "final": order[validation_end:final_end],
        "train": order[final_end:train_end],
        "excluded": order[train_end:],
    }
    selected = result["train"] + result["validation"] + result["final"]
    if len(selected) != len(set(selected)):
        raise GrootVisualizationError(
            "episode leakage detected in deterministic experiment split"
        )
    return result


def derive_training_step_contract(
    *,
    train_samples: int,
    global_batch_size: int,
    configured_max_steps: int | None = None,
    minimum_epochs: float = 1.0,
    minimum_effective_global_batch: int = 1,
    gpu_count: int | None = None,
    per_device_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> dict[str, int | float | None]:
    """Derive the billed GPU step budget from the materialized train split."""

    if int(train_samples) <= 0 or int(global_batch_size) <= 0:
        raise GrootVisualizationError(
            "training step derivation requires positive train samples and global batch"
        )
    if float(minimum_epochs) <= 0:
        raise GrootVisualizationError("minimum training epochs must be positive")
    if int(global_batch_size) < int(minimum_effective_global_batch):
        raise GrootVisualizationError(
            f"effective global batch {global_batch_size} is below required "
            f"{minimum_effective_global_batch}"
        )
    if (
        gpu_count is not None
        or per_device_batch_size is not None
        or gradient_accumulation_steps is not None
    ):
        if (
            min(
                int(gpu_count or 0),
                int(per_device_batch_size or 0),
                int(gradient_accumulation_steps or 0),
            )
            <= 0
        ):
            raise GrootVisualizationError(
                "GPU, per-device batch, and accumulation must be positive"
            )
        derived_global = (
            int(gpu_count or 0)
            * int(per_device_batch_size or 0)
            * int(gradient_accumulation_steps or 0)
        )
        if derived_global != int(global_batch_size):
            raise GrootVisualizationError(
                "effective global batch differs from gpu_count * per_device_batch * "
                "gradient_accumulation_steps"
            )
    required = math.ceil(
        int(train_samples) * float(minimum_epochs) / int(global_batch_size)
    )
    configured = int(configured_max_steps or 0)
    if configured < 0:
        raise GrootVisualizationError("configured max_steps cannot be negative")
    if configured and configured < required:
        raise GrootVisualizationError(
            f"configured max_steps={configured} is insufficient for {train_samples} "
            f"samples at global batch {global_batch_size}; at least {required} steps "
            "are required before any GPU training is submitted"
        )
    effective = configured or required
    return {
        "train_samples": int(train_samples),
        "global_batch_size": int(global_batch_size),
        "required_optimizer_steps": required,
        "configured_max_steps": configured or None,
        "effective_max_steps": effective,
        "minimum_epochs": float(minimum_epochs),
        "epoch_equivalent": effective * int(global_batch_size) / int(train_samples),
        "gpu_count": int(gpu_count) if gpu_count is not None else None,
        "per_device_batch_size": (
            int(per_device_batch_size) if per_device_batch_size is not None else None
        ),
        "gradient_accumulation_steps": (
            int(gradient_accumulation_steps)
            if gradient_accumulation_steps is not None
            else None
        ),
    }


def _source_object_uri(source_uri: str, relative: str) -> str:
    ref = _split_s3(source_uri, require_key=False)
    key = "/".join(part for part in (ref.key.rstrip("/"), relative.lstrip("/")) if part)
    return f"s3://{ref.bucket}/{key}"


def _camera_contract(
    info: Mapping[str, Any], modality: Mapping[str, Any] | None
) -> list[dict[str, str]]:
    """Derive display camera names and original keys from dataset metadata."""

    video_features = {
        str(key)
        for key, value in (info.get("features") or {}).items()
        if isinstance(value, Mapping) and value.get("dtype") == "video"
    }
    if not video_features:
        raise GrootVisualizationError("dataset metadata declares no video observation")
    cameras: list[dict[str, str]] = []
    video_modality = (modality or {}).get("video") or {}
    if isinstance(video_modality, Mapping):
        for name, value in video_modality.items():
            if not isinstance(value, Mapping):
                continue
            original = str(value.get("original_key") or name)
            if original in video_features:
                cameras.append({"name": str(name), "original_key": original})
    mapped = {item["original_key"] for item in cameras}
    for original in sorted(video_features - mapped):
        fallback = original.removeprefix("observation.").replace(".", "_")
        cameras.append({"name": fallback, "original_key": original})
    names = [item["name"] for item in cameras]
    if len(names) != len(set(names)):
        raise GrootVisualizationError(
            "dataset camera metadata resolves duplicate names"
        )
    return cameras


def _episode_timebase(
    alignment: Sequence[Mapping[str, Any]], *, fps: float, camera_names: Sequence[str]
) -> dict[str, Any]:
    """Build the common camera/action timebase and explicit episode boundaries."""

    if float(fps) <= 0 or not alignment:
        raise GrootVisualizationError(
            "timebase requires aligned samples at positive FPS"
        )
    entries: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    current_episode: int | None = None
    for expected_index, row in enumerate(alignment):
        sample_index = int(row.get("sample_index", -1))
        if sample_index != expected_index:
            raise GrootVisualizationError("sample alignment is not contiguous")
        episode_index = int(row.get("episode_index", -1))
        frame_index = int(row.get("frame_index", -1))
        if episode_index < 0 or frame_index < 0:
            raise GrootVisualizationError(
                "sample alignment lacks episode/frame indices"
            )
        entry = {
            "sample_index": sample_index,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "time_seconds": sample_index / float(fps),
        }
        entries.append(entry)
        if episode_index != current_episode:
            boundaries.append(
                {
                    "episode_index": episode_index,
                    "start_sample": sample_index,
                    "end_sample_exclusive": sample_index,
                    "sample_count": 0,
                }
            )
            current_episode = episode_index
        boundaries[-1]["end_sample_exclusive"] = sample_index + 1
        boundaries[-1]["sample_count"] = int(boundaries[-1]["sample_count"]) + 1
    core = {
        "semantics": "global-heldout-sample-index-at-recorded-fps",
        "fps": float(fps),
        "camera_names": list(camera_names),
        "sample_count": len(entries),
        "episode_boundaries": boundaries,
    }
    return {
        **core,
        "id": "sha256:" + _sha256_bytes(_json_bytes(core)),
        "entries": entries,
    }


def _object_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def _write_dataset_stats(parquet_paths: Sequence[Path]) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq

    columns: dict[str, list[Any]] = {}
    for path in parquet_paths:
        table = pq.read_table(path)
        for name in table.column_names:
            if name == "observation.image":
                continue
            values = table[name].to_pylist()
            if not values:
                continue
            probe = np.asarray(values[0])
            if probe.dtype.kind not in "biuf":
                continue
            columns.setdefault(name, []).extend(values)
    result: dict[str, Any] = {}
    for name, values in columns.items():
        data = np.vstack([np.asarray(value).reshape(-1) for value in values]).astype(
            np.float64
        )
        result[name] = {
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q10": np.quantile(data, 0.10, axis=0).tolist(),
            "q50": np.quantile(data, 0.50, axis=0).tolist(),
            "q90": np.quantile(data, 0.90, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
            "count": [int(data.shape[0])],
        }
    if "action" not in result or "observation.state" not in result:
        raise GrootVisualizationError(
            "split statistics lack required state/action tensors"
        )
    return result


def _rewrite_episode_parquet(
    source: Path,
    target: Path,
    *,
    episode_index: int,
    global_index_start: int,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(source)
    rows = table.num_rows
    if rows <= 0:
        raise GrootVisualizationError(f"source episode is empty: {source.name}")
    replacements = {
        "episode_index": [episode_index] * rows,
        "index": list(range(global_index_start, global_index_start + rows)),
    }
    for name, values in replacements.items():
        position = table.schema.get_field_index(name)
        if position < 0:
            raise GrootVisualizationError(f"episode parquet has no {name} column")
        array = pa.array(values, type=table.schema.field(position).type)
        table = table.set_column(position, name, array)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)
    return rows


def _put_file(client: Any, uri: str, path: Path, content_type: str) -> dict[str, Any]:
    return _put_bytes(client, uri, path.read_bytes(), content_type=content_type)


def _materialize_split(
    client: Any,
    *,
    source_uri: str,
    output_uri: str,
    source_info: Mapping[str, Any],
    source_episodes: Sequence[Mapping[str, Any]],
    selected: Sequence[int],
    split_name: str,
    root: Path,
    action_representation: str = "source",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local = root / split_name
    data_pattern = str(source_info["data_path"])
    video_pattern = str(source_info["video_path"])
    video_keys = sorted(
        str(key)
        for key, feature in (source_info.get("features") or {}).items()
        if isinstance(feature, Mapping) and feature.get("dtype") == "video"
    )
    if not video_keys:
        raise GrootVisualizationError("dataset has no video observation feature")
    content_records: list[dict[str, Any]] = []
    rewritten_episodes: list[dict[str, Any]] = []
    parquet_paths: list[Path] = []
    global_index = 0
    for new_index, source_index in enumerate(selected):
        episode_chunk = source_index // int(source_info["chunks_size"])
        relative = data_pattern.format(
            episode_chunk=episode_chunk, episode_index=source_index
        )
        source_parquet_uri = _source_object_uri(source_uri, relative)
        raw = _read_s3_bytes(client, source_parquet_uri)
        source_path = local / "source" / f"episode_{source_index:06d}.parquet"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(raw)
        target_relative = data_pattern.format(episode_chunk=0, episode_index=new_index)
        target_path = local / target_relative
        row_count = _rewrite_episode_parquet(
            source_path,
            target_path,
            episode_index=new_index,
            global_index_start=global_index,
        )
        global_index += row_count
        parquet_paths.append(target_path)
        uploaded = _put_file(
            client,
            _source_object_uri(output_uri, target_relative),
            target_path,
            "application/vnd.apache.parquet",
        )
        content_records.append(
            {
                "kind": "parquet",
                "source_episode": source_index,
                "episode": new_index,
                "source_uri": source_parquet_uri,
                "source_sha256": _sha256_bytes(raw),
                "output_sha256": uploaded["sha256"],
                "rows": row_count,
            }
        )
        episode_meta = dict(source_episodes[source_index])
        episode_meta["episode_index"] = new_index
        episode_meta["length"] = row_count
        episode_meta["source_episode_index"] = source_index
        rewritten_episodes.append(episode_meta)
        for original_video_key in video_keys:
            camera = (
                "front"
                if original_video_key == "observation.image"
                else original_video_key
            )
            source_video_relative = video_pattern.format(
                episode_chunk=episode_chunk,
                episode_index=source_index,
                video_key=original_video_key,
            )
            output_video_relative = video_pattern.format(
                episode_chunk=0,
                episode_index=new_index,
                video_key=original_video_key,
            )
            source_video_uri = _source_object_uri(source_uri, source_video_relative)
            video = _read_s3_bytes(client, source_video_uri)
            uploaded_video = _put_bytes(
                client,
                _source_object_uri(output_uri, output_video_relative),
                video,
                content_type="video/mp4",
            )
            content_records.append(
                {
                    "kind": "video",
                    "camera": camera,
                    "source_episode": source_index,
                    "episode": new_index,
                    "source_uri": source_video_uri,
                    "source_sha256": _sha256_bytes(video),
                    "output_sha256": uploaded_video["sha256"],
                }
            )
    stats = _write_dataset_stats(parquet_paths)
    info = dict(source_info)
    info["total_episodes"] = len(selected)
    info["total_frames"] = global_index
    info["splits"] = {"train": f"0:{len(selected)}"}
    info["chunks_size"] = max(int(source_info["chunks_size"]), len(selected))
    info["npa_split_role"] = split_name
    info["npa_stats_source"] = (
        "train-only" if split_name == "train" else "train-only-copied"
    )
    _put_json(client, _source_object_uri(output_uri, "meta/info.json"), info)
    episodes_body = b"".join(
        (json.dumps(item, sort_keys=True) + "\n").encode()
        for item in rewritten_episodes
    )
    _put_bytes(
        client,
        _source_object_uri(output_uri, "meta/episodes.jsonl"),
        episodes_body,
        content_type="application/x-ndjson",
    )
    _put_json(client, _source_object_uri(output_uri, "meta/stats.json"), stats)
    for name in ("tasks.jsonl", "modality.json", *CUSTOM_DATASET_METADATA):
        payload = _read_s3_bytes(client, _source_object_uri(source_uri, f"meta/{name}"))
        source_payload_sha256 = _sha256_bytes(payload)
        if (
            action_representation == "absolute"
            and name == "npa_groot_modality_config.py"
        ):
            text = payload.decode("utf-8")
            if "ActionRepresentation." not in text:
                raise GrootVisualizationError(
                    "generated modality config has no action representation"
                )
            payload = (
                text.replace(
                    "ActionRepresentation.RELATIVE", "ActionRepresentation.ABSOLUTE"
                )
                .replace(
                    "Joint commands use relative single-arm deltas and an absolute gripper target, "
                    "matching Isaac-GR00T's SO100 custom-embodiment contract.",
                    "All action dimensions are absolute task-space targets; relative-action mode is disabled.",
                )
                .encode("utf-8")
            )
        elif action_representation == "absolute" and name == "npa_groot_adapter.json":
            adapter = json.loads(payload)
            adapter["action_representation"] = "ABSOLUTE"
            adapter["use_relative_action"] = False
            payload = _json_bytes(adapter)
        if name.endswith(".py"):
            content_type = "text/x-python"
        elif name.endswith("jsonl"):
            content_type = "application/x-ndjson"
        else:
            content_type = "application/json"
        uploaded_metadata = _put_bytes(
            client,
            _source_object_uri(output_uri, f"meta/{name}"),
            payload,
            content_type=content_type,
        )
        content_records.append(
            {
                "kind": "metadata",
                "name": name,
                "source_uri": _source_object_uri(source_uri, f"meta/{name}"),
                "source_sha256": source_payload_sha256,
                "output_sha256": uploaded_metadata["sha256"],
            }
        )
    return {
        "uri": output_uri,
        "episodes": len(selected),
        "samples": global_index,
        "stats_sha256": _sha256_bytes(_json_bytes(stats)),
    }, content_records


def prepare_split(
    source_uri: str,
    train_uri: str,
    heldout_uri: str,
    output_uri: str,
    run_id: str,
    *,
    train_episodes: int = 24,
    heldout_episodes: int = 6,
    final_uri: str = "",
    final_episodes: int = 0,
    seed: str = "groot17-learning-v1",
    global_batch_size: int = 8,
    max_steps: int = 0,
    minimum_epochs: float = 1.0,
    minimum_effective_global_batch: int = 1,
    gpu_count: int | None = None,
    per_device_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    action_representation: str = "source",
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Materialize a deterministic train/held-out split with train-only stats."""

    client = _s3_client(s3_client)
    info = _read_s3_json(client, _source_object_uri(source_uri, "meta/info.json"))
    modality = _read_s3_json(
        client, _source_object_uri(source_uri, "meta/modality.json")
    )
    cameras = _camera_contract(info, modality)
    episode_lines = _read_s3_bytes(
        client, _source_object_uri(source_uri, "meta/episodes.jsonl")
    ).decode()
    episodes = [json.loads(line) for line in episode_lines.splitlines() if line.strip()]
    count = int(info.get("total_episodes") or 0)
    if count != len(episodes):
        raise GrootVisualizationError("dataset episode metadata count mismatch")
    if action_representation not in {"source", "absolute"}:
        raise GrootVisualizationError(
            "action representation must be source or absolute"
        )
    if final_episodes:
        if not final_uri:
            raise GrootVisualizationError("final episode split requires a final URI")
        experiment_split = deterministic_experiment_split(
            count,
            train_episodes=train_episodes,
            validation_episodes=heldout_episodes,
            final_episodes=final_episodes,
            seed=seed,
        )
        split = {
            "train": experiment_split["train"],
            "heldout": experiment_split["validation"],
            "final": experiment_split["final"],
            "excluded": experiment_split["excluded"],
        }
    else:
        split = deterministic_episode_split(
            count,
            train_episodes=train_episodes,
            heldout_episodes=heldout_episodes,
            seed=seed,
        )
    with tempfile.TemporaryDirectory(prefix="npa-groot-split-") as tmp:
        train, train_content = _materialize_split(
            client,
            source_uri=source_uri,
            output_uri=train_uri,
            source_info=info,
            source_episodes=episodes,
            selected=split["train"],
            split_name="train",
            root=Path(tmp),
            action_representation=action_representation,
        )
        heldout, heldout_content = _materialize_split(
            client,
            source_uri=source_uri,
            output_uri=heldout_uri,
            source_info=info,
            source_episodes=episodes,
            selected=split["heldout"],
            split_name="heldout",
            root=Path(tmp),
            action_representation=action_representation,
        )
        final: dict[str, Any] | None = None
        final_content: list[dict[str, Any]] = []
        if final_episodes:
            final, final_content = _materialize_split(
                client,
                source_uri=source_uri,
                output_uri=final_uri,
                source_info=info,
                source_episodes=episodes,
                selected=split["final"],
                split_name="final",
                root=Path(tmp),
                action_representation=action_representation,
            )
    # Held-out normalization must come from the train split, never held-out values.
    train_stats = _read_s3_bytes(
        client, _source_object_uri(train_uri, "meta/stats.json")
    )
    _put_bytes(
        client,
        _source_object_uri(heldout_uri, "meta/stats.json"),
        train_stats,
        content_type="application/json",
    )
    if final is not None:
        _put_bytes(
            client,
            _source_object_uri(final_uri, "meta/stats.json"),
            train_stats,
            content_type="application/json",
        )
    train["stats_sha256"] = _sha256_bytes(train_stats)
    train["stats_source"] = "train split only"
    heldout["stats_sha256"] = _sha256_bytes(train_stats)
    heldout["stats_source"] = "train split only (copied; held-out values unused)"
    heldout["stats_byte_identical_to_train"] = True
    if final is not None:
        final["stats_sha256"] = _sha256_bytes(train_stats)
        final["stats_source"] = "train split only (copied; final values unused)"
        final["stats_byte_identical_to_train"] = True
    split_core = {
        "seed": seed,
        "source_uri": source_uri,
        "source_episode_count": count,
        "train_source_episode_ids": split["train"],
        "heldout_source_episode_ids": split["heldout"],
        "excluded_source_episode_ids": split["excluded"],
        "train_content": train_content,
        "heldout_content": heldout_content,
        "final_content": final_content,
    }
    split_hash = _sha256_bytes(_json_bytes(split_core))
    action_shape = (info.get("features") or {}).get("action", {}).get("shape") or []
    video_features = [
        feature
        for feature, value in (info.get("features") or {}).items()
        if isinstance(value, Mapping) and value.get("dtype") == "video"
    ]
    resolution = (info.get("features") or {}).get(video_features[0], {}).get(
        "shape"
    ) or []
    step_contract = derive_training_step_contract(
        train_samples=int(train["samples"]),
        global_batch_size=int(global_batch_size),
        configured_max_steps=max_steps,
        minimum_epochs=minimum_epochs,
        minimum_effective_global_batch=minimum_effective_global_batch,
        gpu_count=gpu_count,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    optimizer_steps = int(step_contract["effective_max_steps"] or 0)
    result = {
        "schema": SPLIT_SCHEMA,
        "status": "prepared",
        "run_id": run_id,
        "split_hash": split_hash,
        "seed": seed,
        "source": {
            "uri": source_uri,
            "episodes": count,
            "samples": int(info.get("total_frames") or 0),
            "embodiment": str(info.get("robot_type") or ""),
            "fps": float(info.get("fps") or 0),
            "camera_names": [item["name"] for item in cameras],
            "cameras": cameras,
            "source_resolution": f"{resolution[1]}x{resolution[0]}"
            if len(resolution) >= 2
            else "unknown",
            "action_dimensions": int(action_shape[0]) if action_shape else 0,
        },
        "train": {**train, "source_episode_ids": split["train"]},
        "heldout": {**heldout, "source_episode_ids": split["heldout"]},
        "final": (
            {**final, "source_episode_ids": split["final"]}
            if final is not None
            else None
        ),
        "excluded": {
            "episodes": len(split["excluded"]),
            "source_episode_ids": split["excluded"],
            "reason": "outside the deterministic experiment cohort",
        },
        "integrity": {
            "episode_overlap": [],
            "leakage_free": True,
            "statistics_source": "train split only",
            "statistics_byte_identity": (
                {"train_equals_validation": True, "train_equals_final": True}
                if final is not None
                else {"train_equals_heldout": True}
            ),
            "final_split_untouched": final is not None,
            "action_representation": action_representation.upper(),
            "content_hash_algorithm": "sha256",
        },
        "training_plan": {
            "global_batch_size": int(global_batch_size),
            "optimizer_steps": optimizer_steps,
            "required_optimizer_steps": int(
                step_contract["required_optimizer_steps"] or 0
            ),
            "configured_max_steps": step_contract["configured_max_steps"],
            "effective_max_steps": optimizer_steps,
            "training_examples": optimizer_steps * int(global_batch_size),
            "epoch_equivalent": optimizer_steps
            * int(global_batch_size)
            / int(train["samples"]),
            "criterion": (
                f"at least {float(step_contract['minimum_epochs'] or 0):g} complete "
                "passes over the deterministic train cohort"
            ),
            "minimum_epochs": float(step_contract["minimum_epochs"] or 0),
            "per_device_batch_size": step_contract["per_device_batch_size"],
            "gradient_accumulation_steps": step_contract["gradient_accumulation_steps"],
            "gpu_count": step_contract["gpu_count"],
        },
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _download_prefix(client: Any, uri: str, destination: Path) -> list[Path]:
    ref = _split_s3(uri, require_key=False)
    prefix = ref.key.rstrip("/") + "/" if ref.key else ""
    downloaded: list[Path] = []
    for item in _list_objects(client, uri):
        if int(item["size"]) <= 0 or not item["key"].startswith(prefix):
            continue
        relative = item["key"][len(prefix) :]
        target = destination / relative
        _download(client, _object_uri(ref.bucket, item["key"]), target)
        downloaded.append(target)
    if not downloaded:
        raise GrootVisualizationError(
            f"S3 prefix contains no material artifacts: {uri}"
        )
    return downloaded


def _upload_directory(client: Any, directory: Path, uri: str) -> dict[str, Any]:
    ref = _split_s3(uri, require_key=False)
    prefix = ref.key.rstrip("/")
    objects: list[dict[str, Any]] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        key = "/".join(part for part in (prefix, relative) if part)
        payload = path.read_bytes()
        record = _put_bytes(client, _object_uri(ref.bucket, key), payload)
        record["relative_path"] = relative
        objects.append(record)
    if not objects:
        raise GrootVisualizationError(f"checkpoint directory is empty: {directory}")
    local_identity = _checkpoint_identity(directory)
    return {
        "uri": uri,
        "objects": len(objects),
        "bytes": sum(int(item["bytes"]) for item in objects),
        **{
            key: value
            for key, value in local_identity.items()
            if key
            in {
                "sha256",
                "artifact_sha256",
                "weight_objects",
                "weight_bytes",
                "weights_sha256",
            }
        },
    }


def _checkpoint_digest(files: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_json_bytes({"files": list(files)}))


def _is_model_weight_path(relative_path: str) -> bool:
    return any(pattern.search(relative_path) for pattern in WEIGHT_FILE_PATTERNS)


def _checkpoint_identity(directory: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    if not files:
        raise GrootVisualizationError("checkpoint contains no files")
    weight_files = [item for item in files if _is_model_weight_path(str(item["path"]))]
    if not weight_files:
        raise GrootVisualizationError("checkpoint contains no model weight files")
    return {
        "objects": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
        "sha256": _checkpoint_digest(files),
        "artifact_sha256": _checkpoint_digest(files),
        "weight_objects": len(weight_files),
        "weight_bytes": sum(int(item["bytes"]) for item in weight_files),
        "weights_sha256": _checkpoint_digest(weight_files),
    }


def _resolve_highest_checkpoint_directory(directory: Path) -> tuple[Path, int | None]:
    """Resolve exactly the highest valid checkpoint-N directory, never a prefix root."""

    candidates: list[tuple[int, Path]] = []
    for path in directory.rglob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit() and any(item.is_file() for item in path.rglob("*")):
            candidates.append((int(suffix), path))
    if candidates:
        step, path = max(candidates, key=lambda item: item[0])
        return path, step
    if any(item.is_file() for item in directory.rglob("*")):
        return directory, None
    raise GrootVisualizationError(
        "checkpoint prefix contains no valid checkpoint directory"
    )


def require_distinct_trained_weights(
    baseline_identity: Mapping[str, Any], candidate_identity: Mapping[str, Any]
) -> None:
    baseline_digest = str(baseline_identity.get("weights_sha256") or "")
    candidate_digest = str(candidate_identity.get("weights_sha256") or "")
    if len(baseline_digest) != 64 or len(candidate_digest) != 64:
        raise GrootVisualizationError(
            "baseline/candidate weight-only identity is missing"
        )
    if baseline_digest == candidate_digest:
        raise GrootVisualizationError(
            "trained checkpoint model weights equal baseline weights"
        )


def checkpoint_model_config_contract(directory: Path) -> dict[str, Any]:
    """Read factual model/action settings from checkpoint artifacts."""

    documents: list[Mapping[str, Any]] = []
    for relative in (
        "experiment_cfg/final_model_config.json",
        "config.json",
        "npa_baseline_checkpoint.json",
    ):
        path = directory / relative
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_bytes())
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(value, Mapping):
            nested = value.get("model_config_contract")
            documents.append(nested if isinstance(nested, Mapping) else value)
    resolved: dict[str, Any] = {}
    for key in GROOT_MODEL_CONFIG_CONTRACT:
        if key == "action_representation":
            continue
        for document in documents:
            if key in document:
                resolved[key] = document[key]
                break
    processor_paths = list(directory.rglob("processor_config.json"))
    processor_contracts: list[tuple[bool, list[str]]] = []
    for path in processor_paths:
        try:
            processor = json.loads(path.read_bytes())
        except (OSError, ValueError, TypeError):
            continue
        kwargs = processor.get("processor_kwargs")
        if not isinstance(kwargs, Mapping):
            continue
        use_relative_action = kwargs.get("use_relative_action")
        modalities = kwargs.get("modality_configs")
        custom = (
            modalities.get("new_embodiment")
            if isinstance(modalities, Mapping)
            else None
        )
        action = custom.get("action") if isinstance(custom, Mapping) else None
        configs = action.get("action_configs") if isinstance(action, Mapping) else None
        if not isinstance(use_relative_action, bool) or not isinstance(configs, list):
            continue
        reps = [
            str(item.get("rep") or "") for item in configs if isinstance(item, Mapping)
        ]
        if reps and len(reps) == len(configs):
            processor_contracts.append((use_relative_action, reps))
    if not processor_contracts:
        raise GrootVisualizationError(
            "checkpoint processor omits factual custom action configuration"
        )
    if any(relative for relative, _ in processor_contracts):
        resolved["use_relative_action"] = True
    if all(
        reps and all(rep == "ABSOLUTE" for rep in reps)
        for _, reps in processor_contracts
    ):
        resolved["action_representation"] = "ABSOLUTE"
    else:
        resolved["action_representation"] = "MIXED_OR_RELATIVE"
    if resolved != GROOT_MODEL_CONFIG_CONTRACT:
        raise GrootVisualizationError(
            f"checkpoint model/action configuration mismatch: {resolved!r}"
        )
    return resolved


def _load_custom_modality_config(dataset_path: Path) -> Path:
    """Load the converter-authored custom embodiment registration or fail closed."""

    path = dataset_path / "meta" / "npa_groot_modality_config.py"
    if not path.is_file():
        raise GrootVisualizationError(f"custom GR00T modality config is absent: {path}")
    module_name = (
        f"_npa_groot_modality_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise GrootVisualizationError(
            f"could not load custom GR00T modality config: {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return path


def _initialize_baseline_checkpoint(
    *,
    train_path: Path,
    output_path: Path,
    base_model: str,
    embodiment: str,
) -> None:
    """Initialize a custom-embodiment checkpoint without optimizer updates."""

    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    from huggingface_hub import snapshot_download
    from npa.cli.groot import HF_MODEL_REVISIONS
    from transformers import set_seed

    set_seed(42)
    _load_custom_modality_config(train_path)
    tag = EmbodimentTag.resolve(embodiment)
    base_revision = HF_MODEL_REVISIONS.get(base_model)
    if not base_revision:
        raise GrootVisualizationError(
            f"starting GR00T checkpoint has no immutable revision pin: {base_model}"
        )
    # The workbench pins a separate revision for the nested Cosmos backbone.
    # Snapshot the outer GR00T checkpoint first so that nested revision cannot
    # be interpreted as a revision of the outer repository by Transformers.
    resolved_base = snapshot_download(repo_id=base_model, revision=base_revision)
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "episode_sampling_rate": 1.0,
                "datasets": [
                    {
                        "dataset_paths": [str(train_path)],
                        "mix_ratio": 1.0,
                        "embodiment_tag": tag.value,
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = bool(GROOT_MODEL_CONFIG_CONTRACT["tune_projector"])
    config.model.tune_diffusion_model = bool(
        GROOT_MODEL_CONFIG_CONTRACT["tune_diffusion_model"]
    )
    config.model.state_dropout_prob = 0.0
    config.model.load_bf16 = bool(GROOT_MODEL_CONFIG_CONTRACT["load_bf16"])
    config.model.reproject_vision = bool(
        GROOT_MODEL_CONFIG_CONTRACT["reproject_vision"]
    )
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.model_revision = None
    config.model.backbone_trainable_params_fp32 = bool(
        GROOT_MODEL_CONFIG_CONTRACT["backbone_trainable_params_fp32"]
    )
    config.model.use_relative_action = bool(
        GROOT_MODEL_CONFIG_CONTRACT["use_relative_action"]
    )
    config.training.start_from_checkpoint = resolved_base
    config.training.num_gpus = 1
    config.training.global_batch_size = 1
    config.validate()
    experiment = output_path / "experiment_cfg"
    experiment.mkdir(parents=True, exist_ok=True)
    pipeline = Gr00tN1d7Pipeline(config, experiment)
    pipeline.setup()
    output_path.mkdir(parents=True, exist_ok=True)
    pipeline.return_model().save_pretrained(output_path)
    processor = pipeline.return_processor()
    processor.save_pretrained(output_path)
    processor.save_pretrained(output_path / "processor")
    marker = {
        "schema": "npa.groot.baseline_checkpoint.v1",
        "optimizer_steps": 0,
        "base_model": base_model,
        "base_model_revision": base_revision,
        "embodiment": embodiment,
        "statistics_source": "train split only",
        "model_config_contract": GROOT_MODEL_CONFIG_CONTRACT,
    }
    (output_path / "npa_baseline_checkpoint.json").write_bytes(_json_bytes(marker))


def _extract_arrays(
    trajectory: Any, columns: Sequence[str], *, count: int | None = None
) -> Any:
    import numpy as np

    arrays = [
        np.vstack([np.asarray(item) for item in trajectory[column]])
        for column in columns
    ]
    result = np.concatenate(arrays, axis=-1)
    return result if count is None else result[:count]


def _seed_stochastic_sources(seed: int) -> None:
    """Reset every stochastic source used by GR00T inference."""

    import numpy as np
    import torch

    # PyTorch's deterministic-algorithm guard rejects CuBLAS matmuls unless a
    # reproducible workspace policy is configured.  Set it before the first
    # policy forward (this helper runs before Gr00tPolicy construction) so
    # seeded evaluation is deterministic in fact instead of failing on CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        from transformers import set_seed

        set_seed(int(seed), deterministic=True)
    except TypeError:
        set_seed(int(seed))


def _prediction_sha256(predicted: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(predicted)
    return _sha256_bytes(array.view(np.uint8).tobytes())


def trivial_predictor_metrics(
    expert: Any, predicted: Any, train_action_mean: Sequence[float]
) -> dict[str, Any]:
    """Measure a checkpoint against zero and leakage-free train-mean floors."""

    import numpy as np

    expert_array = np.asarray(expert, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    validate_action_alignment(expert_array, predicted_array, label="trivial-floor")
    mean = np.asarray(train_action_mean, dtype=np.float64).reshape(1, -1)
    if mean.shape[1] != expert_array.shape[1] or not np.isfinite(mean).all():
        raise GrootVisualizationError(
            "train-split mean predictor shape/value is invalid"
        )
    model_mse = float(np.mean(np.square(expert_array - predicted_array)))
    zero_mse = float(np.mean(np.square(expert_array)))
    mean_mse = float(np.mean(np.square(expert_array - mean)))
    if mean_mse <= 0 or not all(
        math.isfinite(value) for value in (model_mse, zero_mse, mean_mse)
    ):
        raise GrootVisualizationError("trivial predictor MSE is invalid")
    return {
        "zero_predictor_mse": zero_mse,
        "train_mean_predictor_mse": mean_mse,
        "train_action_mean": mean.reshape(-1).tolist(),
        "model_mse": model_mse,
        "skill_score": 1.0 - model_mse / mean_mse,
        "skill_score_definition": "1 - model_mse / train_mean_predictor_mse",
    }


def _evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    heldout_path: Path,
    embodiment: str,
    action_horizon: int,
    seed: int,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run upstream GR00T inference and retain aligned expert/predicted arrays."""

    import numpy as np
    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    if runtime is None:
        if not torch.cuda.is_available():
            raise GrootVisualizationError("real GR00T evaluation requires a CUDA GPU")
        _load_custom_modality_config(heldout_path)
        tag = EmbodimentTag.resolve(embodiment)
        _seed_stochastic_sources(seed)
        policy = Gr00tPolicy(
            embodiment_tag=tag,
            model_path=str(checkpoint_path),
            device="cuda:0",
        )
        modality = policy.get_modality_config()
        loader = LeRobotEpisodeLoader(
            dataset_path=heldout_path,
            modality_configs=modality,
            video_backend="torchcodec",
            video_backend_kwargs=None,
        )
    else:
        policy = runtime["policy"]
        loader = runtime["loader"]
        tag = runtime["tag"]
    state_keys = list(loader.modality_configs["state"].modality_keys)
    action_keys = list(loader.modality_configs["action"].modality_keys)
    model_modality = {
        key: value for key, value in loader.modality_configs.items() if key != "action"
    }
    expert_parts: list[Any] = []
    predicted_parts: list[Any] = []
    state_parts: list[Any] = []
    sample_rows: list[dict[str, Any]] = []
    horizon_parts: list[Any] = []
    episode_metrics: list[dict[str, Any]] = []
    forward_calls = 0
    offset = 0
    for episode_index in range(len(loader)):
        trajectory = loader[episode_index]
        episode_steps = len(trajectory)
        predicted_rows: list[Any] = []
        episode_horizons: list[int] = []
        for step in range(0, episode_steps, action_horizon):
            point = extract_step_data(trajectory, step, model_modality, tag)
            observation: dict[str, Any] = {}
            for key, value in point.states.items():
                observation[f"state.{key}"] = value
            for key, value in point.images.items():
                observation[f"video.{key}"] = np.asarray(value)
            for language_key in loader.modality_configs["language"].modality_keys:
                observation[language_key] = point.text
            parsed = parse_observation_gr00t(observation, loader.modality_configs)
            _seed_stochastic_sources(int(seed) + forward_calls)
            raw_action, _ = policy.get_action(parsed)
            action = parse_action_gr00t(raw_action)
            forward_calls += 1
            available = min(action_horizon, episode_steps - step)
            for horizon_index in range(available):
                episode_horizons.append(horizon_index)
                predicted_rows.append(
                    np.concatenate(
                        [
                            np.atleast_1d(action[f"action.{key}"][horizon_index])
                            for key in action_keys
                        ]
                    )
                )
        predicted = np.asarray(predicted_rows, dtype=np.float32)
        expert = _extract_arrays(
            trajectory, [f"action.{key}" for key in action_keys], count=episode_steps
        ).astype(np.float32)
        states = _extract_arrays(
            trajectory, [f"state.{key}" for key in state_keys], count=episode_steps
        ).astype(np.float32)
        if predicted.shape != expert.shape or predicted.shape[0] != episode_steps:
            raise GrootVisualizationError(
                f"predicted/expert alignment failed for episode {episode_index}: "
                f"{predicted.shape} != {expert.shape}"
            )
        squared = np.square(expert - predicted)
        absolute = np.abs(expert - predicted)
        episode_metrics.append(
            {
                "episode_index": episode_index,
                "samples": episode_steps,
                "mse": float(np.mean(squared)),
                "mae": float(np.mean(absolute)),
            }
        )
        for frame_index in range(episode_steps):
            sample_rows.append(
                {
                    "sample_index": offset + frame_index,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "dataset_time_seconds": frame_index / float(loader.fps),
                }
            )
        offset += episode_steps
        expert_parts.append(expert)
        predicted_parts.append(predicted)
        state_parts.append(states)
        horizon_parts.append(np.asarray(episode_horizons, dtype=np.int64))
    expert_all = np.concatenate(expert_parts)
    predicted_all = np.concatenate(predicted_parts)
    states_all = np.concatenate(state_parts)
    horizon_all = np.concatenate(horizon_parts)
    if forward_calls <= 0 or expert_all.size <= 0:
        raise GrootVisualizationError(
            "real evaluation emitted no model forwards or actions"
        )
    squared = np.square(expert_all - predicted_all)
    absolute = np.abs(expert_all - predicted_all)
    per_horizon_mse: list[float] = []
    per_horizon_counts: list[int] = []
    for horizon_index in range(int(action_horizon)):
        selected = horizon_all == horizon_index
        per_horizon_counts.append(int(np.count_nonzero(selected)))
        if not np.any(selected):
            raise GrootVisualizationError(
                f"evaluation has no samples for action horizon {horizon_index}"
            )
        per_horizon_mse.append(float(np.mean(squared[selected])))
    return {
        "expert": expert_all,
        "predicted": predicted_all,
        "states": states_all,
        "horizon_indices": horizon_all,
        "samples": sample_rows,
        "metrics": {
            "mse": float(np.mean(squared)),
            "mae": float(np.mean(absolute)),
            "per_dimension_mse": np.mean(squared, axis=0).tolist(),
            "per_dimension_mae": np.mean(absolute, axis=0).tolist(),
            "per_dimension_max_abs_error": np.max(absolute, axis=0).tolist(),
            "per_horizon_mse": per_horizon_mse,
            "per_horizon_counts": per_horizon_counts,
        },
        "episode_metrics": episode_metrics,
        "episode_count": len(loader),
        "sample_count": int(expert_all.shape[0]),
        "action_dimensions": int(expert_all.shape[1]),
        "forward_calls": forward_calls,
        "fps": float(loader.fps),
        "action_horizon": int(action_horizon),
        "evaluation_seed": int(seed),
        "prediction_sha256": _prediction_sha256(predicted_all),
        "gpu_name": torch.cuda.get_device_name(0),
        "_runtime": {"policy": policy, "loader": loader, "tag": tag},
    }


def evaluate(
    split_manifest_uri: str,
    checkpoint_uri: str,
    output_uri: str,
    arrays_uri: str,
    run_id: str,
    phase: str,
    *,
    base_model: str = "",
    baseline_checkpoint_uri: str = "",
    action_horizon: int = 16,
    evaluation_seeds: Sequence[int] = (1701, 1701, 2718, 3141, 5772),
    evaluation_repeats: int = 5,
    expected_checkpoint_sha256: str = "",
    expected_checkpoint_step: int = 0,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Initialize (baseline only) and evaluate a real checkpoint on held-out data."""

    import numpy as np

    if phase not in {"baseline", "posttrain"}:
        raise GrootVisualizationError("evaluation phase must be baseline or posttrain")
    client = _s3_client(s3_client)
    split = _read_s3_json(client, split_manifest_uri)
    if split.get("schema") != SPLIT_SCHEMA or split.get("run_id") != run_id:
        raise GrootVisualizationError("split manifest does not belong to this run")
    if split.get("integrity", {}).get("leakage_free") is not True:
        raise GrootVisualizationError(
            "evaluation refuses a split without leakage proof"
        )
    embodiment = str(split.get("source", {}).get("embodiment") or "")
    with tempfile.TemporaryDirectory(prefix=f"npa-groot-{phase}-eval-") as tmp:
        root = Path(tmp)
        train_path = root / "train"
        heldout_path = root / "heldout"
        _download_prefix(client, str(split["train"]["uri"]), train_path)
        _download_prefix(client, str(split["heldout"]["uri"]), heldout_path)
        checkpoint_path = root / "checkpoint"
        checkpoint_artifact: dict[str, Any]
        resolved_checkpoint_uri = checkpoint_uri
        if phase == "baseline":
            if not base_model or not baseline_checkpoint_uri:
                raise GrootVisualizationError(
                    "baseline evaluation requires base model and baseline checkpoint URI"
                )
            _initialize_baseline_checkpoint(
                train_path=train_path,
                output_path=checkpoint_path,
                base_model=base_model,
                embodiment=embodiment,
            )
            checkpoint_artifact = _upload_directory(
                client, checkpoint_path, baseline_checkpoint_uri
            )
            resolved_checkpoint_uri = baseline_checkpoint_uri
        else:
            _download_prefix(client, checkpoint_uri, checkpoint_path)
            checkpoint_artifact = {
                "uri": checkpoint_uri,
                **_checkpoint_identity(checkpoint_path),
            }
        if phase == "posttrain":
            checkpoint_path, checkpoint_step = _resolve_highest_checkpoint_directory(
                checkpoint_path
            )
            if checkpoint_step is None:
                match = re.search(r"(?:^|/)checkpoint-(\d+)/?$", checkpoint_uri)
                checkpoint_step = int(match.group(1)) if match else None
            checkpoint_artifact = {
                "uri": checkpoint_uri.rstrip("/")
                + (
                    "/"
                    if re.search(r"(?:^|/)checkpoint-\d+/?$", checkpoint_uri)
                    else f"/checkpoint-{checkpoint_step}/"
                    if checkpoint_step is not None
                    else "/"
                ),
                "resolved_checkpoint_step": checkpoint_step,
                **_checkpoint_identity(checkpoint_path),
            }
            resolved_checkpoint_uri = str(checkpoint_artifact["uri"])
            if (
                len(expected_checkpoint_sha256) != 64
                or checkpoint_artifact.get("sha256") != expected_checkpoint_sha256
                or int(checkpoint_step or 0) != int(expected_checkpoint_step)
            ):
                raise GrootVisualizationError(
                    "post-training checkpoint differs from the resolved immutable reference"
                )
        resolved_model_contract = checkpoint_model_config_contract(checkpoint_path)
        repeats_requested = int(evaluation_repeats)
        if repeats_requested < 4:
            raise GrootVisualizationError(
                "evaluation_repeats must preserve one duplicate and two independent seeds"
            )
        seed_pool = [int(value) for value in evaluation_seeds]
        while len(seed_pool) < repeats_requested:
            seed_pool.append(5772 + 7919 * (len(seed_pool) - 4))
        seeds = seed_pool[:repeats_requested]
        if len(seeds) < 4 or len(seeds) == len(set(seeds)):
            raise GrootVisualizationError(
                "evaluation requires a duplicated determinism seed and at least two independent seeds"
            )
        # Load one immutable checkpoint/policy and one held-out dataset per
        # evaluation stage. Each forward is independently reseeded below, so
        # reuse avoids five redundant multi-GB model constructions without
        # weakening deterministic seed isolation.
        first_runtime_result = _evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            heldout_path=heldout_path,
            embodiment=embodiment,
            action_horizon=int(action_horizon),
            seed=seeds[0],
        )
        runtime_objects = first_runtime_result.pop("_runtime", None)
        repeats = [first_runtime_result] + [
            _evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                heldout_path=heldout_path,
                embodiment=embodiment,
                action_horizon=int(action_horizon),
                seed=seed,
                runtime=runtime_objects,
            )
            for seed in seeds[1:]
        ]
        raw = repeats[0]
        same_seed = [item for item in repeats if item["evaluation_seed"] == seeds[0]]
        deterministic = (
            len(same_seed) >= 2
            and len({str(item["prediction_sha256"]) for item in same_seed}) == 1
        )
        if not deterministic:
            raise GrootVisualizationError(
                "identical checkpoint evaluation was not deterministic for the same seed"
            )
        train_stats = json.loads((train_path / "meta" / "stats.json").read_bytes())
        action_stats = train_stats.get("action") or {}
        floors = trivial_predictor_metrics(
            raw["expert"], raw["predicted"], action_stats.get("mean") or []
        )
        repeat_mses = [float(item["metrics"]["mse"]) for item in repeats]
        repeat_spread = max(repeat_mses) - min(repeat_mses)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            expert=raw["expert"],
            predicted=raw["predicted"],
            states=raw["states"],
            horizon_indices=raw["horizon_indices"],
        )
        arrays_artifact = _put_bytes(client, arrays_uri, buffer.getvalue())
    result = {
        "schema": EVAL_SCHEMA,
        "status": "completed",
        "run_id": run_id,
        "phase": phase,
        "evaluation_kind": EVALUATION_KIND,
        "closed_loop": False,
        "offline_label": "Offline held-out policy evaluation",
        "split_manifest_uri": split_manifest_uri,
        "split_hash": split["split_hash"],
        "heldout_data_uri": split["heldout"]["uri"],
        "checkpoint": checkpoint_artifact,
        "checkpoint_uri": resolved_checkpoint_uri,
        "engine": "NVIDIA Isaac-GR00T Gr00tPolicy.get_action",
        "real_model_forward": True,
        "model_forward_calls": raw["forward_calls"],
        "episodes": raw["episode_count"],
        "samples": raw["sample_count"],
        "action_dimensions": raw["action_dimensions"],
        "action_horizon": raw["action_horizon"],
        "denoising_steps_contract": (
            "not exposed: pinned Gr00tPolicy.get_action does not consume an inference-step option"
        ),
        "fps": raw["fps"],
        "metrics": {**raw["metrics"], **floors},
        "repeat_evaluation": {
            "configured_repeats": repeats_requested,
            "policy_constructions": 1,
            "cost_note": "GPU model-forward cost scales linearly with evaluation_repeats",
            "seeds": seeds,
            "same_seed_deterministic": deterministic,
            "prediction_sha256": [item["prediction_sha256"] for item in repeats],
            "repeat_mses": repeat_mses,
            "repeat_spread": repeat_spread,
            "independent_seed_count": len(set(seeds)),
            "effective_sample_unit": "model-forward/action-chunk",
            "effective_model_forwards": sum(
                int(item["forward_calls"]) for item in repeats
            ),
        },
        "episode_metrics": raw["episode_metrics"],
        "sample_alignment": raw["samples"],
        "arrays": arrays_artifact,
        "accelerator": {"gpu_count": 1, "gpu_name": raw["gpu_name"]},
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "is_robot_capture_time": False,
        "model_config_contract": resolved_model_contract,
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def baseline_eval(
    split_manifest_uri: str,
    output_uri: str,
    arrays_uri: str,
    baseline_checkpoint_uri: str,
    run_id: str,
    base_model: str,
    **kwargs: Any,
) -> dict[str, Any]:
    return evaluate(
        split_manifest_uri,
        "",
        output_uri,
        arrays_uri,
        run_id,
        "baseline",
        base_model=base_model,
        baseline_checkpoint_uri=baseline_checkpoint_uri,
        **kwargs,
    )


def posttrain_eval(
    split_manifest_uri: str,
    checkpoint_ref_uri: str,
    output_uri: str,
    arrays_uri: str,
    run_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    client = _s3_client(kwargs.get("s3_client"))
    reference = _read_s3_json(client, checkpoint_ref_uri)
    checkpoint = reference.get("checkpoint")
    training = reference.get("training")
    if (
        reference.get("schema") != CHECKPOINT_REF_SCHEMA
        or reference.get("status") != "resolved"
        or reference.get("run_id") != run_id
        or not isinstance(checkpoint, Mapping)
        or not isinstance(training, Mapping)
    ):
        raise GrootVisualizationError(
            "post-training evaluation requires this run's resolved checkpoint reference"
        )
    checkpoint_uri = str(checkpoint.get("uri") or "")
    checkpoint_sha256 = str(checkpoint.get("sha256") or "")
    checkpoint_step = int(checkpoint.get("resolved_checkpoint_step") or 0)
    if (
        not re.search(rf"/checkpoint-{checkpoint_step}/?$", checkpoint_uri)
        or len(checkpoint_sha256) != 64
        or checkpoint_step <= 0
        or int(training.get("optimizer_steps") or 0) != checkpoint_step
    ):
        raise GrootVisualizationError("resolved checkpoint reference is internally inconsistent")
    return evaluate(
        split_manifest_uri,
        checkpoint_uri,
        output_uri,
        arrays_uri,
        run_id,
        "posttrain",
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_checkpoint_step=checkpoint_step,
        **kwargs,
    )


def validate_evaluation(payload: Mapping[str, Any], *, phase: str, run_id: str) -> None:
    """Reject JSON-only shims, misaligned samples, and non-finite metrics."""

    if payload.get("schema") != EVAL_SCHEMA or payload.get("status") != "completed":
        raise GrootVisualizationError(f"{phase} evaluation schema/status is invalid")
    if payload.get("run_id") != run_id or payload.get("phase") != phase:
        raise GrootVisualizationError(f"{phase} evaluation identity mismatch")
    if payload.get("real_model_forward") is not True:
        raise GrootVisualizationError(
            f"{phase} evaluation has no real model forward proof"
        )
    forwards = int(payload.get("model_forward_calls") or 0)
    episodes = int(payload.get("episodes") or 0)
    samples = int(payload.get("samples") or 0)
    dimensions = int(payload.get("action_dimensions") or 0)
    if min(forwards, episodes, samples, dimensions) <= 0:
        raise GrootVisualizationError(f"{phase} evaluation counts are empty")
    alignment = payload.get("sample_alignment") or []
    if len(alignment) != samples:
        raise GrootVisualizationError(
            f"{phase} predicted/expert sample alignment is incomplete"
        )
    metrics = payload.get("metrics") or {}
    for name in ("mse", "mae"):
        value = float(metrics.get(name, math.nan))
        if not math.isfinite(value) or value < 0:
            raise GrootVisualizationError(f"{phase} metric {name} is invalid")
    for name in ("per_dimension_mse", "per_dimension_mae"):
        values = metrics.get(name) or []
        if len(values) != dimensions or not all(
            math.isfinite(float(value)) for value in values
        ):
            raise GrootVisualizationError(f"{phase} metric {name} is invalid")
    horizon = int(payload.get("action_horizon") or 0)
    horizon_values = metrics.get("per_horizon_mse") or []
    horizon_counts = metrics.get("per_horizon_counts") or []
    if (
        horizon <= 0
        or len(horizon_values) != horizon
        or len(horizon_counts) != horizon
        or not all(
            math.isfinite(float(value)) and float(value) >= 0
            for value in horizon_values
        )
        or not all(int(value) > 0 for value in horizon_counts)
    ):
        raise GrootVisualizationError(f"{phase} per-horizon metric contract is invalid")
    repeat = payload.get("repeat_evaluation") or {}
    if repeat.get("same_seed_deterministic") is not True:
        raise GrootVisualizationError(
            f"{phase} evaluation lacks deterministic repeat proof"
        )
    if int(repeat.get("independent_seed_count") or 0) < 2:
        raise GrootVisualizationError(
            f"{phase} evaluation lacks independent repeat seeds"
        )
    if payload.get("model_config_contract") != GROOT_MODEL_CONFIG_CONTRACT:
        raise GrootVisualizationError(
            f"{phase} model/action configuration contract differs"
        )


def validate_action_alignment(expert: Any, predicted: Any, *, label: str) -> None:
    """Require non-empty, finite, sample-for-sample action tensors."""

    import numpy as np

    if expert.ndim != 2 or predicted.shape != expert.shape or expert.size <= 0:
        raise GrootVisualizationError(
            f"{label} predicted/expert action alignment is invalid: "
            f"{predicted.shape} != {expert.shape}"
        )
    if not np.isfinite(expert).all() or not np.isfinite(predicted).all():
        raise GrootVisualizationError(
            f"{label} action tensors contain non-finite values"
        )


def require_learning_improvement(comparison: Mapping[str, Any]) -> None:
    if comparison.get("improved") is not True:
        raise GrootVisualizationError(
            "learning gate failed: post-training held-out action MSE did not improve"
        )
    failures = list(comparison.get("gate_failures") or [])
    if failures:
        reason = "; ".join(str(item) for item in failures)
        raise GrootVisualizationError(f"learning gate failed: {reason}")


def operational_learning_decision(
    comparison: Mapping[str, Any], loss_trend: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep operational completion, observed learning, and promotion independent."""

    improved = (
        comparison.get("gate_passed") is True
        and loss_trend.get("loss_decreased") is True
    )
    return {
        "pipeline_status": "succeeded",
        "learning_outcome": "improved" if improved else "not_improved",
        # This smoke is not a statistically powered candidate-selection run.
        "candidate_promoted": False,
    }


def compare_metrics(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    minimum_relative_improvement: float = 0.0,
    repeat_noise_spread: float = 0.0,
    repeat_noise_multiple: float = 0.0,
    minimum_skill_score: float = -math.inf,
    max_dimension_regression: float = math.inf,
) -> dict[str, Any]:
    """Calculate a transparent before/after comparison and regression list."""

    baseline = float(before["mse"])
    posttrain = float(after["mse"])
    absolute = baseline - posttrain
    relative = (absolute / baseline * 100.0) if baseline > 0 else 0.0
    before_dims = [float(value) for value in before["per_dimension_mse"]]
    after_dims = [float(value) for value in after["per_dimension_mse"]]
    if len(before_dims) != len(after_dims):
        raise GrootVisualizationError("per-dimension metric widths differ")
    dimensions = []
    regressions = []
    for index, (old, new) in enumerate(zip(before_dims, after_dims)):
        item = {
            "dimension": index,
            "baseline_mse": old,
            "posttrain_mse": new,
            "absolute_improvement": old - new,
            "improved": new < old,
        }
        dimensions.append(item)
        if new > old:
            regressions.append(item)
    relative_fraction = absolute / baseline if baseline > 0 else 0.0
    candidate_skill = float(after.get("skill_score", -math.inf))
    noise_requirement = float(repeat_noise_spread) * float(repeat_noise_multiple)
    failures: list[str] = []
    if relative_fraction < float(minimum_relative_improvement):
        failures.append(
            f"relative improvement {relative_fraction:.6f} is below "
            f"{float(minimum_relative_improvement):.6f}"
        )
    if absolute <= noise_requirement:
        failures.append(
            f"absolute improvement {absolute:.6f} does not exceed repeat-noise "
            f"requirement {noise_requirement:.6f}"
        )
    if candidate_skill < float(minimum_skill_score):
        failures.append(
            f"candidate skill score {candidate_skill:.6f} is below "
            f"{float(minimum_skill_score):.6f}"
        )
    blocking_regressions = [
        item
        for item in regressions
        if float(item["posttrain_mse"]) - float(item["baseline_mse"])
        > float(max_dimension_regression)
    ]
    if blocking_regressions:
        failures.append(
            "per-dimension regression exceeds configured tolerance on dimensions "
            + ",".join(str(item["dimension"]) for item in blocking_regressions)
        )
    return {
        "metric_name": "action_mse",
        "baseline_value": baseline,
        "posttrain_value": posttrain,
        "absolute_improvement": absolute,
        "relative_improvement_percent": relative,
        "relative_improvement": relative_fraction,
        "improved": posttrain < baseline,
        "per_dimension": dimensions,
        "regressions": regressions,
        "blocking_regressions": blocking_regressions,
        "candidate_skill_score": candidate_skill,
        "gate": {
            "minimum_relative_improvement": float(minimum_relative_improvement),
            "repeat_noise_spread": float(repeat_noise_spread),
            "repeat_noise_multiple": float(repeat_noise_multiple),
            "required_absolute_improvement_over_noise": noise_requirement,
            "minimum_skill_score": float(minimum_skill_score),
            "max_dimension_regression": float(max_dimension_regression),
        },
        "gate_failures": failures,
        "gate_passed": posttrain < baseline and not failures,
    }


def robust_loss_decrease(
    loss_history: Sequence[Mapping[str, Any]],
    *,
    window: int = 5,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Require a robust early/late median decrease without value deduplication."""

    import statistics

    rows = [
        (int(item["optimizer_step"]), float(item["loss"]))
        for item in loss_history
        if isinstance(item.get("optimizer_step"), (int, float))
        and isinstance(item.get("loss"), (int, float))
        and math.isfinite(float(item["loss"]))
    ]
    if len(rows) < 4:
        raise GrootVisualizationError(
            "robust loss evidence has too few real step records"
        )
    steps = [item[0] for item in rows]
    if steps != sorted(steps) or len(steps) != len(set(steps)):
        raise GrootVisualizationError(
            "loss evidence optimizer steps are not unique and ordered"
        )
    width = min(int(window), len(rows) // 2)
    early = float(statistics.median(item[1] for item in rows[:width]))
    late = float(statistics.median(item[1] for item in rows[-width:]))
    required = early * (1.0 - float(tolerance))
    decreased = late < required
    return {
        "window": width,
        "tolerance": float(tolerance),
        "robust_early_loss": early,
        "robust_late_loss": late,
        "required_late_below": required,
        "loss_decreased": decreased,
    }


def calculate_training_coverage(
    *, optimizer_steps: int, global_batch_size: int, train_samples: int
) -> dict[str, Any]:
    """Calculate coverage from the trainer's factual batch, not a planned allocation."""

    training_examples = int(optimizer_steps) * int(global_batch_size)
    if optimizer_steps <= 1 or global_batch_size <= 0 or train_samples <= 0:
        raise GrootVisualizationError("training coverage inputs are invalid")
    return {
        "training_examples": training_examples,
        "epoch_equivalent": training_examples / int(train_samples),
    }


def _read_npz(client: Any, uri: str) -> dict[str, Any]:
    import numpy as np

    with np.load(io.BytesIO(_read_s3_bytes(client, uri))) as arrays:
        required = {"expert", "predicted", "states"}
        if not required.issubset(arrays.files):
            raise GrootVisualizationError(
                "evaluation arrays omit expert/predicted/state data"
            )
        return {name: arrays[name].copy() for name in required}


def _decode_video(path: Path, *, max_frames: int = 0) -> tuple[list[Any], float]:
    frames: list[Any] = []
    try:
        import av
    except ImportError:
        from PIL import Image

        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(probe.stdout).get("streams") or []
        if not streams:
            raise GrootVisualizationError("held-out source has no video stream")
        rate = (
            streams[0].get("avg_frame_rate") or streams[0].get("r_frame_rate") or "0/1"
        )
        numerator, denominator = (int(value) for value in str(rate).split("/", 1))
        fps = numerator / denominator if denominator else 0.0
        with tempfile.TemporaryDirectory(prefix="npa-groot-video-decode-") as tmp:
            output_pattern = str(Path(tmp) / "%06d.png")
            command = ["ffmpeg", "-v", "error", "-i", str(path)]
            if max_frames:
                command.extend(["-frames:v", str(max_frames)])
            subprocess.run([*command, "-vsync", "0", output_pattern], check=True)
            for frame_path in sorted(Path(tmp).glob("*.png")):
                with Image.open(frame_path) as image:
                    frames.append(image.convert("RGB").copy())
    else:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise GrootVisualizationError("held-out source has no video stream")
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            fps = float(rate) if rate else 0.0
            for frame in container.decode(stream):
                frames.append(frame.to_image().convert("RGB"))
                if max_frames and len(frames) >= max_frames:
                    break
    if not frames or fps <= 0:
        raise GrootVisualizationError("held-out source video decoded no timed frames")
    return frames, fps


def _heldout_video_inventory(
    client: Any,
    heldout_uri: str,
    *,
    cameras: Sequence[Mapping[str, str]],
    episode_count: int,
) -> list[dict[str, Any]]:
    """Inventory every camera/episode video declared by held-out metadata."""

    import re

    by_original = {str(item["original_key"]): str(item["name"]) for item in cameras}
    inventory: list[dict[str, Any]] = []
    pattern = re.compile(r"/([^/]+)/episode_(\d+)\.mp4$")
    for item in _list_objects(client, heldout_uri):
        if int(item["size"]) <= 0 or not str(item["key"]).lower().endswith(".mp4"):
            continue
        match = pattern.search("/" + str(item["key"]).lstrip("/"))
        if not match or match.group(1) not in by_original:
            continue
        inventory.append(
            {
                "episode_index": int(match.group(2)),
                "camera_name": by_original[match.group(1)],
                "original_key": match.group(1),
                "uri": _object_uri(item["bucket"], item["key"]),
                "bytes": int(item["size"]),
            }
        )
    expected = {
        (episode, str(camera["name"]))
        for episode in range(int(episode_count))
        for camera in cameras
    }
    actual = {(item["episode_index"], item["camera_name"]) for item in inventory}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise GrootVisualizationError(
            f"held-out camera inventory mismatch; missing={missing}, extra={extra}"
        )
    return sorted(
        inventory, key=lambda item: (item["episode_index"], item["camera_name"])
    )


def _decode_synchronized_camera(
    client: Any,
    inventory: Sequence[Mapping[str, Any]],
    alignment: Sequence[Mapping[str, Any]],
    *,
    camera_name: str,
    root: Path,
) -> tuple[list[Any], float]:
    """Decode every represented episode and align it to action/state samples."""

    frames_by_episode: dict[int, list[Any]] = {}
    fps_values: set[float] = set()
    for item in inventory:
        if str(item["camera_name"]) != camera_name:
            continue
        episode = int(item["episode_index"])
        path = root / f"camera-{camera_name}-episode-{episode:06d}.mp4"
        _download(client, str(item["uri"]), path)
        frames, fps = _decode_video(path)
        frames_by_episode[episode] = frames
        fps_values.add(round(float(fps), 9))
    if len(fps_values) != 1:
        raise GrootVisualizationError("held-out camera episodes do not share one FPS")
    synchronized: list[Any] = []
    for row in alignment:
        episode = int(row["episode_index"])
        frame = int(row["frame_index"])
        episode_frames = frames_by_episode.get(episode) or []
        if frame >= len(episode_frames):
            raise GrootVisualizationError(
                f"camera {camera_name!r} episode {episode} has {len(episode_frames)} "
                f"frames but action/state alignment requires frame {frame}"
            )
        synchronized.append(episode_frames[frame])
    if len(synchronized) != len(alignment):
        raise GrootVisualizationError(
            "camera/action synchronized sample count mismatch"
        )
    return synchronized, next(iter(fps_values))


def _draw_series(
    draw: Any,
    values: Sequence[float],
    *,
    bounds: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    if len(values) < 2:
        return
    left, top, right, bottom = bounds
    low, high = min(values), max(values)
    spread = max(high - low, 1e-6)
    points = [
        (
            left + int(index * (right - left) / max(len(values) - 1, 1)),
            bottom - int((value - low) * (bottom - top) / spread),
        )
        for index, value in enumerate(values)
    ]
    draw.line(points, fill=color, width=2)


def _comparison_video(
    source_video: Path,
    destination: Path,
    *,
    expert: Any,
    baseline: Any,
    posttrain: Any,
    baseline_mse: float,
    posttrain_mse: float,
) -> dict[str, Any]:
    images, fps = _decode_video(source_video)
    return _comparison_video_frames(
        images,
        fps,
        destination,
        expert=expert,
        baseline=baseline,
        posttrain=posttrain,
        baseline_mse=baseline_mse,
        posttrain_mse=posttrain_mse,
    )


def _comparison_video_frames(
    images: Sequence[Any],
    fps: float,
    destination: Path,
    *,
    expert: Any,
    baseline: Any,
    posttrain: Any,
    baseline_mse: float,
    posttrain_mse: float,
) -> dict[str, Any]:
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    count = min(len(images), len(expert), len(baseline), len(posttrain))
    if count != len(expert):
        raise GrootVisualizationError(
            f"comparison camera/action synchronization mismatch: {len(images)} "
            f"frames for {len(expert)} action samples"
        )
    if count <= 0:
        raise GrootVisualizationError("comparison video has no aligned held-out frames")
    source_width = max(int(image.width) for image in images[:count])
    source_height = max(int(image.height) for image in images[:count])
    camera_x, camera_y = 24, 76
    plot_x = camera_x + source_width + 32
    canvas_width = max(640, plot_x + 488)
    canvas_height = max(360, camera_y + source_height + 52)
    # yuv420p requires even dimensions; padding is truthful and never scales or
    # crops the native camera pixels.
    canvas_width += canvas_width % 2
    canvas_height += canvas_height % 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(destination), mode="w")
    try:
        try:
            stream = container.add_stream("libx264", rate=max(1, round(fps)))
        except Exception:  # noqa: BLE001 - image may only ship the MPEG-4 encoder
            stream = container.add_stream("mpeg4", rate=max(1, round(fps)))
        stream.width = canvas_width
        stream.height = canvas_height
        stream.pix_fmt = "yuv420p"
        font = ImageFont.load_default()
        history = min(50, count)
        for index in range(count):
            canvas = Image.new("RGB", (canvas_width, canvas_height), (15, 20, 28))
            draw = ImageDraw.Draw(canvas)
            native = images[index]
            canvas.paste(native, (camera_x, camera_y))
            draw.rectangle(
                (
                    camera_x - 1,
                    camera_y - 1,
                    camera_x + native.width,
                    camera_y + native.height,
                ),
                outline=(120, 160, 190),
            )
            draw.text(
                (20, 18),
                "Offline held-out evaluation — not a rollout",
                fill=(244, 248, 252),
                font=font,
            )
            draw.text(
                (20, 40),
                f"camera: {native.width}x{native.height} native pixels at 1:1",
                fill=(164, 190, 214),
                font=font,
            )
            draw.text(
                (plot_x, 76),
                f"sample {index}   expert / baseline / posttrain",
                fill=(235, 240, 245),
                font=font,
            )
            start = max(0, index - history + 1)
            plot = (plot_x + 2, 105, plot_x + 463, 245)
            draw.rectangle(plot, outline=(70, 85, 105))
            _draw_series(
                draw,
                expert[start : index + 1, 0].tolist(),
                bounds=plot,
                color=(90, 210, 130),
            )
            _draw_series(
                draw,
                baseline[start : index + 1, 0].tolist(),
                bounds=plot,
                color=(240, 110, 100),
            )
            _draw_series(
                draw,
                posttrain[start : index + 1, 0].tolist(),
                bounds=plot,
                color=(95, 160, 250),
            )
            error_before = float(np.mean(np.abs(expert[index] - baseline[index])))
            error_after = float(np.mean(np.abs(expert[index] - posttrain[index])))
            draw.text(
                (plot_x, 265),
                f"expert action: {np.round(expert[index], 3).tolist()}",
                fill=(90, 210, 130),
                font=font,
            )
            draw.text(
                (plot_x, 283),
                f"baseline: {np.round(baseline[index], 3).tolist()}  abs err {error_before:.4f}",
                fill=(240, 110, 100),
                font=font,
            )
            draw.text(
                (plot_x, 301),
                f"posttrain: {np.round(posttrain[index], 3).tolist()}  abs err {error_after:.4f}",
                fill=(95, 160, 250),
                font=font,
            )
            draw.text(
                (20, canvas_height - 28),
                f"held-out MSE {baseline_mse:.6f} -> {posttrain_mse:.6f}; dataset-index time at {fps:g} FPS",
                fill=(220, 225, 230),
                font=font,
            )
            frame = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise GrootVisualizationError("comparison video encoder produced no artifact")
    return {
        "resolution": f"{canvas_width}x{canvas_height}",
        "source_resolution": f"{images[0].width}x{images[0].height}",
        "native_camera_scale": "1:1",
        "native_resolution_preserved": True,
        "camera_region": {
            "x": camera_x,
            "y": camera_y,
            "width": source_width,
            "height": source_height,
        },
        "frames": count,
        "fps": fps,
        "label": EVALUATION_KIND,
    }


def compare_learning(
    split_manifest_uri: str,
    baseline_uri: str,
    posttrain_uri: str,
    training_manifest_uri: str,
    output_uri: str,
    video_uri: str,
    run_id: str,
    *,
    minimum_relative_improvement: float = DEFAULT_MINIMUM_RELATIVE_IMPROVEMENT,
    minimum_skill_score: float = DEFAULT_MINIMUM_SKILL_SCORE,
    repeat_noise_multiple: float = DEFAULT_REPEAT_NOISE_MULTIPLE,
    max_dimension_regression: float = DEFAULT_MAX_DIMENSION_REGRESSION,
    loss_decrease_tolerance: float = 0.01,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Compare aligned evaluations without conflating learning with pipeline health.

    Structural/data/checkpoint failures still raise.  A valid smoke checkpoint
    that does not improve is recorded as ``not_improved`` so diagnostic media and
    publication continue, while promotion remains fail-closed.
    """

    import numpy as np

    client = _s3_client(s3_client)
    split = _read_s3_json(client, split_manifest_uri)
    baseline = _read_s3_json(client, baseline_uri)
    posttrain = _read_s3_json(client, posttrain_uri)
    training = _read_s3_json(client, training_manifest_uri)
    validate_evaluation(baseline, phase="baseline", run_id=run_id)
    validate_evaluation(posttrain, phase="posttrain", run_id=run_id)
    if baseline["split_hash"] != posttrain["split_hash"] or baseline[
        "split_hash"
    ] != split.get("split_hash"):
        raise GrootVisualizationError("baseline/posttrain evaluation split mismatch")
    if baseline["sample_alignment"] != posttrain["sample_alignment"]:
        raise GrootVisualizationError("baseline/posttrain sample alignment mismatch")
    if baseline.get("model_config_contract") != posttrain.get("model_config_contract"):
        raise GrootVisualizationError("baseline/posttrain model configuration differs")
    require_distinct_trained_weights(baseline["checkpoint"], posttrain["checkpoint"])
    before_arrays = _read_npz(client, str(baseline["arrays"]["uri"]))
    after_arrays = _read_npz(client, str(posttrain["arrays"]["uri"]))
    if before_arrays["expert"].shape != after_arrays[
        "expert"
    ].shape or not np.array_equal(before_arrays["expert"], after_arrays["expert"]):
        raise GrootVisualizationError(
            "expert action tensors differ between evaluations"
        )
    validate_action_alignment(
        before_arrays["expert"], before_arrays["predicted"], label="baseline"
    )
    validate_action_alignment(
        after_arrays["expert"], after_arrays["predicted"], label="posttrain"
    )
    repeat_noise_spread = float(
        (baseline.get("repeat_evaluation") or {}).get("repeat_spread") or 0.0
    )
    comparison = compare_metrics(
        baseline["metrics"],
        posttrain["metrics"],
        minimum_relative_improvement=minimum_relative_improvement,
        repeat_noise_spread=repeat_noise_spread,
        repeat_noise_multiple=repeat_noise_multiple,
        minimum_skill_score=minimum_skill_score,
        max_dimension_regression=max_dimension_regression,
    )
    if (
        training.get("schema") != "npa.groot.finetune.v1"
        or training.get("status") != "completed"
    ):
        raise GrootVisualizationError(
            "training manifest is not a completed real GR00T run"
        )
    for key in (
        "collective_ok",
        "optimizer_step_ok",
        "loss_finite",
        "loss_steps_real",
        "rank_zero_checkpoint_only",
    ):
        if training.get(key) is not True:
            raise GrootVisualizationError(f"training evidence requires {key}=true")
    expected_world_size = int(split.get("training_plan", {}).get("gpu_count") or 0)
    expected_ranks = list(range(expected_world_size))
    observed_ranks = sorted(int(value) for value in training.get("observed_ranks") or [])
    training_observed_ranks = sorted(
        int(value) for value in training.get("training_observed_ranks") or []
    )
    if (
        expected_world_size < 1
        or int(training.get("num_gpus") or 0) != expected_world_size
        or int(training.get("world_size") or 0) != expected_world_size
        or int(training.get("distinct_gpu_count") or 0) != expected_world_size
        or observed_ranks != expected_ranks
        or training_observed_ranks != expected_ranks
        or int(training.get("checkpoint_upload_invocations") or 0) != 1
    ):
        raise GrootVisualizationError(
            "training manifest lacks the declared GPU world, rank participation, or single-upload evidence"
        )
    expected_steps = int(split.get("training_plan", {}).get("optimizer_steps") or 0)
    actual_steps = int(training.get("training_step") or 0)
    global_batch = int(training.get("global_batch_size") or 0)
    if expected_steps < 2 or actual_steps < expected_steps:
        raise GrootVisualizationError(
            f"meaningful training coverage missing: {actual_steps} < {expected_steps} steps"
        )
    train_samples = int(split["train"]["samples"])
    coverage = calculate_training_coverage(
        optimizer_steps=actual_steps,
        global_batch_size=global_batch,
        train_samples=train_samples,
    )
    minimum_epochs = float(split.get("training_plan", {}).get("minimum_epochs") or 0)
    if minimum_epochs <= 0.0 or float(coverage["epoch_equivalent"]) < minimum_epochs:
        raise GrootVisualizationError(
            "training coverage did not satisfy the declared smoke contract"
        )
    loss_history = training.get("loss_history") or []
    loss_trend = robust_loss_decrease(
        loss_history, tolerance=float(loss_decrease_tolerance)
    )
    outcome_failures = list(comparison.get("gate_failures") or [])
    if loss_trend["loss_decreased"] is not True:
        outcome_failures.append("robust late-window training loss did not decrease")
    decision = operational_learning_decision(comparison, loss_trend)
    learning_outcome = str(decision["learning_outcome"])
    comparison["gate_failures"] = outcome_failures
    comparison["gate_passed"] = learning_outcome == "improved"
    cameras = list(split["source"].get("cameras") or [])
    if not cameras:
        raise GrootVisualizationError("split manifest lacks derived camera metadata")
    inventory = _heldout_video_inventory(
        client,
        str(split["heldout"]["uri"]),
        cameras=cameras,
        episode_count=int(split["heldout"]["episodes"]),
    )
    timebase = _episode_timebase(
        baseline["sample_alignment"],
        fps=float(split["source"]["fps"]),
        camera_names=[str(item["name"]) for item in cameras],
    )
    primary_camera = str(cameras[0]["name"])
    with tempfile.TemporaryDirectory(prefix="npa-groot-compare-") as tmp:
        root = Path(tmp)
        comparison_video = root / "offline-heldout-comparison.mp4"
        images, camera_fps = _decode_synchronized_camera(
            client,
            inventory,
            baseline["sample_alignment"],
            camera_name=primary_camera,
            root=root,
        )
        if abs(float(camera_fps) - float(split["source"]["fps"])) > 1e-6:
            raise GrootVisualizationError(
                "decoded camera FPS differs from dataset metadata"
            )
        video_meta = _comparison_video_frames(
            images,
            camera_fps,
            comparison_video,
            expert=before_arrays["expert"],
            baseline=before_arrays["predicted"],
            posttrain=after_arrays["predicted"],
            baseline_mse=comparison["baseline_value"],
            posttrain_mse=comparison["posttrain_value"],
        )
        video_artifact = _put_bytes(
            client, video_uri, comparison_video.read_bytes(), content_type="video/mp4"
        )
    result = {
        "schema": REPORT_SCHEMA,
        "status": "completed",
        **decision,
        "run_id": run_id,
        "evaluation_kind": EVALUATION_KIND,
        "badge": "Offline held-out policy evaluation",
        "closed_loop": False,
        "offline_label_present": True,
        "semantic_phases": SEMANTIC_PHASES,
        "dataset": {
            "embodiment": split["source"]["embodiment"],
            "camera_names": split["source"]["camera_names"],
            "cameras": cameras,
            "source_resolution": split["source"]["source_resolution"],
            "fps": split["source"]["fps"],
            "source_episodes": split["source"]["episodes"],
            "train_episodes": split["train"]["episodes"],
            "train_samples": train_samples,
            "heldout_episodes": split["heldout"]["episodes"],
            "heldout_samples": split["heldout"]["samples"],
            "excluded_episodes": split["excluded"]["episodes"],
            "action_dimensions": baseline["action_dimensions"],
            "split_hash": split["split_hash"],
            "leakage_free": True,
            "statistics_source": "train split only",
        },
        "training": {
            "accelerator": training.get("gpu_model")
            or training.get("accelerator")
            or "GPU",
            "gpu_count": int(training.get("num_gpus") or 0),
            "distinct_gpu_count": int(training.get("distinct_gpu_count") or 0),
            "world_size": int(training.get("world_size") or 0),
            "observed_ranks": [
                int(value) for value in training.get("observed_ranks") or []
            ],
            "training_observed_ranks": training_observed_ranks,
            "all_ranks_trained": training_observed_ranks == expected_ranks,
            "both_ranks_trained": training.get("both_ranks_trained") is True,
            "collective_ok": training.get("collective_ok") is True,
            "rank_zero_checkpoint_only": training.get("rank_zero_checkpoint_only")
            is True,
            "checkpoint_upload_invocations": int(
                training.get("checkpoint_upload_invocations") or 0
            ),
            "optimizer_steps": actual_steps,
            "global_batch_size": global_batch,
            **coverage,
            "coverage_criterion": split["training_plan"]["criterion"],
            "initial_loss": float(loss_history[0]["loss"]),
            "final_loss": float(loss_history[-1]["loss"]),
            "loss_history": loss_history,
            "checkpoint_uri": posttrain["checkpoint_uri"],
            "checkpoint_sha256": posttrain["checkpoint"]["sha256"],
            "checkpoint_weights_sha256": posttrain["checkpoint"]["weights_sha256"],
            "baseline_weights_sha256": baseline["checkpoint"]["weights_sha256"],
            "weights_differ": True,
            "resolved_checkpoint_step": posttrain["checkpoint"].get(
                "resolved_checkpoint_step"
            ),
            "model_config_contract": GROOT_MODEL_CONFIG_CONTRACT,
            "loss_trend": loss_trend,
        },
        "evaluation": {
            **comparison,
            "episodes": baseline["episodes"],
            "samples": baseline["samples"],
            "real_model_forward": True,
            "baseline_forward_calls": baseline["model_forward_calls"],
            "posttrain_forward_calls": posttrain["model_forward_calls"],
            "baseline_mae": baseline["metrics"]["mae"],
            "posttrain_mae": posttrain["metrics"]["mae"],
            "zero_predictor_mse": posttrain["metrics"]["zero_predictor_mse"],
            "train_mean_predictor_mse": posttrain["metrics"][
                "train_mean_predictor_mse"
            ],
            "baseline_skill_score": baseline["metrics"]["skill_score"],
            "posttrain_skill_score": posttrain["metrics"]["skill_score"],
            "baseline_repeat_evaluation": baseline["repeat_evaluation"],
            "posttrain_repeat_evaluation": posttrain["repeat_evaluation"],
            "per_horizon_mse": {
                "baseline": baseline["metrics"]["per_horizon_mse"],
                "posttrain": posttrain["metrics"]["per_horizon_mse"],
                "counts": posttrain["metrics"]["per_horizon_counts"],
                "action_horizon": posttrain["action_horizon"],
            },
            "baseline_uri": baseline_uri,
            "posttrain_uri": posttrain_uri,
        },
        "visualizations": {
            "comparison_video": {**video_artifact, **video_meta},
            "mcap_uri": "pending emit_mcap",
            "rrd_uri": "pending emit_rrd",
            "timestamp_semantics": TIMESTAMP_SEMANTICS,
            "timebase": timebase,
            "is_robot_capture_time": False,
            "native_resolution_preserved": True,
        },
        "provenance": {
            "split_manifest_uri": split_manifest_uri,
            "training_manifest_uri": training_manifest_uri,
            "baseline_checkpoint_uri": baseline["checkpoint_uri"],
            "posttrain_checkpoint_uri": posttrain["checkpoint_uri"],
            "heldout_source_videos": inventory,
            "primary_camera": primary_camera,
            "synchronized_camera_samples": len(baseline["sample_alignment"]),
        },
        "limitations": [
            "This report is offline action-matching evidence, not a robot rollout.",
            "The short optimizer smoke is an operational pipeline validation and does not establish statistical learning.",
        ],
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _evaluation_bundle(
    client: Any, report_uri: str
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    report = _read_s3_json(client, report_uri)
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status") not in {"passed", "completed"}
        or report.get("pipeline_status") not in {None, "succeeded"}
        or report.get("learning_outcome") not in {None, "improved", "not_improved"}
    ):
        raise GrootVisualizationError(
            "learning report has not completed its operational evaluation"
        )
    baseline = _read_s3_json(client, str(report["evaluation"]["baseline_uri"]))
    posttrain = _read_s3_json(client, str(report["evaluation"]["posttrain_uri"]))
    validate_evaluation(baseline, phase="baseline", run_id=str(report["run_id"]))
    validate_evaluation(posttrain, phase="posttrain", run_id=str(report["run_id"]))
    before_arrays = _read_npz(client, str(baseline["arrays"]["uri"]))
    after_arrays = _read_npz(client, str(posttrain["arrays"]["uri"]))
    return report, baseline, posttrain, before_arrays, after_arrays


def _evaluated_checkpoint_curve(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the one factual checkpoint evaluated by the shipped workflow."""

    evaluation = report.get("evaluation")
    training = report.get("training")
    if not isinstance(evaluation, Mapping) or not isinstance(training, Mapping):
        raise GrootVisualizationError(
            "learning report lacks evaluation/training evidence"
        )
    step = int(training.get("resolved_checkpoint_step") or 0)
    mse_value = evaluation.get("posttrain_value")
    skill_score_value = evaluation.get("candidate_skill_score")
    if not isinstance(mse_value, (int, float)) or not isinstance(
        skill_score_value, (int, float)
    ):
        raise GrootVisualizationError(
            "final evaluated checkpoint omits numeric MSE/skill score"
        )
    mse = float(mse_value)
    skill_score = float(skill_score_value)
    if step <= 0 or not math.isfinite(mse) or not math.isfinite(skill_score):
        raise GrootVisualizationError(
            "final evaluated checkpoint lacks a valid step/MSE/skill score"
        )
    return [
        {
            "optimizer_step": step,
            "mse": mse,
            "skill_score": skill_score,
            "source": "final_evaluated_smoke_checkpoint",
        }
    ]


def _validate_learning_mcap(
    path: Path, *, run_id: str, camera_name: str = "front"
) -> dict[str, Any]:
    info = summarize_mcap(path)
    if not info.valid_magic or info.size_bytes <= 0 or info.message_count <= 0:
        raise GrootVisualizationError("learning MCAP is empty or invalid")
    if not info.timestamps_in_int64_domain:
        raise GrootVisualizationError(
            "learning MCAP timestamps exceed the nonnegative int64 domain"
        )
    if not info.channels_monotonic:
        raise GrootVisualizationError(
            "learning MCAP timestamps are not monotonic within each channel"
        )
    if info.start_time_ns != 1:
        raise GrootVisualizationError(
            "learning MCAP must use one relative dataset-index clock, not wall time"
        )
    if info.end_time_ns < info.start_time_ns:
        raise GrootVisualizationError("learning MCAP relative timeline is invalid")
    required_topics = {
        topic.replace("/camera/front", f"/camera/{camera_name}"): schema
        for topic, schema in REQUIRED_MCAP_TOPICS.items()
    }
    for topic, schema in required_topics.items():
        if info.channels.get(topic, 0) <= 0:
            raise GrootVisualizationError(f"learning MCAP lacks required topic {topic}")
        if info.schemas.get(topic) != schema:
            raise GrootVisualizationError(
                f"learning MCAP schema mismatch on {topic}: {info.schemas.get(topic)!r}"
            )
    metadata = info.metadata.get("npa") or {}
    if metadata.get("run_id") != run_id:
        raise GrootVisualizationError("learning MCAP run identity mismatch")
    if metadata.get("evaluation_kind") != EVALUATION_KIND:
        raise GrootVisualizationError(
            "learning MCAP is not truthfully labelled offline"
        )
    if metadata.get("timestamps") != TIMESTAMP_SEMANTICS:
        raise GrootVisualizationError("learning MCAP timestamp semantics are missing")
    if metadata.get("timeline_origin") != "relative-zero-plus-1ns":
        raise GrootVisualizationError("learning MCAP timeline origin is missing")
    if metadata.get("training_loss_clock") != "optimizer_step-as-seconds":
        raise GrootVisualizationError("learning MCAP training clock is missing")
    if metadata.get("is_robot_capture_time") != "false":
        raise GrootVisualizationError("learning MCAP capture-time semantics are invalid")
    if metadata.get("primary_camera") != camera_name:
        raise GrootVisualizationError("learning MCAP primary-camera provenance mismatch")
    if not str(metadata.get("timebase_id") or "").startswith("sha256:"):
        raise GrootVisualizationError("learning MCAP timebase identity is missing")
    try:
        dataset_sample_count = int(metadata.get("dataset_sample_count") or 0)
        dataset_end_time_ns = int(metadata.get("dataset_end_time_ns") or -1)
        declared_end_time_ns = int(metadata.get("declared_end_time_ns") or -1)
        dataset_fps = float(metadata.get("fps") or 0.0)
    except (TypeError, ValueError) as exc:
        raise GrootVisualizationError(
            "learning MCAP timeline metadata is malformed"
        ) from exc
    if (
        dataset_sample_count <= 0
        or not math.isfinite(dataset_fps)
        or dataset_fps <= 0
        or dataset_end_time_ns < info.start_time_ns
        or declared_end_time_ns < dataset_end_time_ns
        or declared_end_time_ns != info.end_time_ns
    ):
        raise GrootVisualizationError(
            "learning MCAP metadata/timeline relationship is invalid"
        )
    aligned_topics = (
        f"/camera/{camera_name}",
        "/policy/predicted_action",
        "/expert/action",
        "/metrics/action_error",
    )
    if any(info.channels.get(topic) != dataset_sample_count for topic in aligned_topics):
        raise GrootVisualizationError(
            "learning MCAP aligned channel counts differ from the dataset timebase"
        )
    camera_range = info.channel_time_ranges.get(f"/camera/{camera_name}") or {}
    if camera_range != {
        "start_time_ns": info.start_time_ns,
        "end_time_ns": dataset_end_time_ns,
    }:
        raise GrootVisualizationError(
            "learning MCAP camera range differs from the dataset timebase"
        )
    if any(
        value["start_time_ns"] < info.start_time_ns
        or value["end_time_ns"] > declared_end_time_ns
        for value in info.channel_time_ranges.values()
    ):
        raise GrootVisualizationError(
            "learning MCAP channel range exceeds the declared relative timeline"
        )
    return info.to_dict()


def emit_learning_mcap(
    report_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Emit synchronized held-out camera, action, error, and metric replay."""

    client = _s3_client(s3_client)
    report, baseline, _posttrain, before_arrays, after_arrays = _evaluation_bundle(
        client, report_uri
    )
    if report["run_id"] != run_id:
        raise GrootVisualizationError("learning report run identity mismatch")
    expert = before_arrays["expert"]
    predicted_before = before_arrays["predicted"]
    predicted_after = after_arrays["predicted"]
    dimensions = int(expert.shape[1])
    timebase = dict(report["visualizations"]["timebase"])
    entries = list(timebase.get("entries") or [])
    if len(entries) != len(expert):
        raise GrootVisualizationError("MCAP timebase/action sample mismatch")
    primary_camera = str(report["provenance"]["primary_camera"])
    with tempfile.TemporaryDirectory(prefix="npa-groot-learning-mcap-") as tmp:
        root = Path(tmp)
        images, fps = _decode_synchronized_camera(
            client,
            report["provenance"]["heldout_source_videos"],
            baseline["sample_alignment"],
            camera_name=primary_camera,
            root=root,
        )
        frame_inputs: list[FrameInput] = []
        timeline_origin_ns = 1
        for index, (image, time_entry) in enumerate(zip(images, entries, strict=True)):
            path = root / "frames" / f"{index:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, "PNG")
            frame_inputs.append(
                FrameInput(
                    path=path,
                    camera=primary_camera,
                    timestamp_ns=timeline_origin_ns
                    + round(float(time_entry["time_seconds"]) * 1_000_000_000),
                )
            )
        predicted_records = []
        expert_records = []
        error_records = []
        for index in range(expert.shape[0]):
            entry = entries[index]
            timestamp_ns = timeline_origin_ns + round(
                float(entry["time_seconds"]) * 1_000_000_000
            )
            identity = {
                "_timestamp_ns": timestamp_ns,
                "sample_index": index,
                "episode_index": int(entry["episode_index"]),
                "frame_index": int(entry["frame_index"]),
                "timebase_id": str(timebase["id"]),
            }
            predicted_record: dict[str, Any] = dict(identity)
            expert_record: dict[str, Any] = dict(identity)
            error_record: dict[str, Any] = dict(identity)
            for dimension in range(dimensions):
                predicted_record[f"baseline_dim_{dimension}"] = float(
                    predicted_before[index, dimension]
                )
                predicted_record[f"posttrain_dim_{dimension}"] = float(
                    predicted_after[index, dimension]
                )
                expert_record[f"dim_{dimension}"] = float(expert[index, dimension])
                error_record[f"baseline_abs_dim_{dimension}"] = float(
                    abs(expert[index, dimension] - predicted_before[index, dimension])
                )
                error_record[f"posttrain_abs_dim_{dimension}"] = float(
                    abs(expert[index, dimension] - predicted_after[index, dimension])
                )
            before_mae, before_mse = _per_sample_action_errors(
                expert[index], predicted_before[index]
            )
            after_mae, after_mse = _per_sample_action_errors(
                expert[index], predicted_after[index]
            )
            error_record["baseline_mae"] = before_mae
            error_record["posttrain_mae"] = after_mae
            error_record["baseline_mse"] = before_mse
            error_record["posttrain_mse"] = after_mse
            predicted_records.append(predicted_record)
            expert_records.append(expert_record)
            error_records.append(error_record)
        checkpoint_curve = _evaluated_checkpoint_curve(report)
        metric_documents = {
            "policy/predicted_action": predicted_records,
            "expert/action": expert_records,
            "metrics/action_error": error_records,
            "metrics/heldout_before": [
                {
                    "mse": report["evaluation"]["baseline_value"],
                    "mae": report["evaluation"]["baseline_mae"],
                    "samples": report["evaluation"]["samples"],
                }
            ],
            "metrics/heldout_after": [
                {
                    "mse": report["evaluation"]["posttrain_value"],
                    "mae": report["evaluation"]["posttrain_mae"],
                    "samples": report["evaluation"]["samples"],
                }
            ],
            "metrics/train_loss": [
                {
                    **item,
                    "_timestamp_ns": timeline_origin_ns
                    + int(item["optimizer_step"]) * 1_000_000_000,
                    "clock_domain": "optimizer_step",
                }
                for item in report["training"]["loss_history"]
            ],
            "metrics/per_horizon_error": [
                {
                    "horizon_index": index,
                    "baseline_mse": float(before),
                    "posttrain_mse": float(after),
                    "effective_count": int(count),
                    "action_horizon": int(
                        report["evaluation"]["per_horizon_mse"]["action_horizon"]
                    ),
                    "_timestamp_ns": timeline_origin_ns + index * 1_000_000_000,
                    "clock_domain": "action_horizon_index",
                }
                for index, (before, after, count) in enumerate(
                    zip(
                        report["evaluation"]["per_horizon_mse"]["baseline"],
                        report["evaluation"]["per_horizon_mse"]["posttrain"],
                        report["evaluation"]["per_horizon_mse"]["counts"],
                        strict=True,
                    )
                )
            ],
            "metrics/checkpoint_curve": [
                {
                    "optimizer_step": int(item["optimizer_step"]),
                    "validation_mse": float(item["mse"]),
                    "skill_score": float(item["skill_score"]),
                    "_timestamp_ns": timeline_origin_ns
                    + int(item["optimizer_step"]) * 1_000_000_000,
                    "clock_domain": "optimizer_step",
                }
                for item in checkpoint_curve
            ],
        }
        metric_inputs: list[MetricsInput] = []
        for name, records in metric_documents.items():
            path = root / "metrics" / f"{name.replace('/', '_')}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(records), encoding="utf-8")
            metric_inputs.append(MetricsInput(path=path, name=name))
        log_path = root / "offline-evaluation.log"
        log_text = (
            "Offline held-out policy evaluation; not a closed-loop rollout.\n"
            f"split_hash={report['dataset']['split_hash']}\n"
            f"checkpoint={report['training']['checkpoint_uri']}\n"
            f"action_mse={report['evaluation']['baseline_value']} -> "
            f"{report['evaluation']['posttrain_value']}\n"
        )
        log_path.write_text(log_text, encoding="utf-8")
        step_ns = int(1_000_000_000 / float(baseline["fps"]))
        dataset_end_time_ns = max(
            int(frame.timestamp_ns or timeline_origin_ns) for frame in frame_inputs
        )
        declared_timestamps = [
            int(
                record.get("_timestamp_ns")
                or timeline_origin_ns
                + (metric_index if len(records) == 1 else offset) * step_ns
            )
            for metric_index, records in enumerate(metric_documents.values())
            for offset, record in enumerate(records)
        ]
        declared_timestamps.extend(
            timeline_origin_ns + index * step_ns
            for index, line in enumerate(log_text.splitlines())
            if line.strip()
        )
        declared_end_time_ns = max(dataset_end_time_ns, *declared_timestamps)
        output = root / "groot-learning.mcap"
        write_run_mcap(
            output=output,
            frames=frame_inputs,
            metrics=metric_inputs,
            logs=[LogInput(path=log_path, name="groot_offline_learning")],
            fps=float(baseline["fps"]),
            start_time_ns=timeline_origin_ns,
            run_id=run_id,
            camera_topic_prefix="/camera",
            metrics_topic_prefix="",
            metadata={
                "evaluation_kind": EVALUATION_KIND,
                "closed_loop": "false",
                "split_hash": str(report["dataset"]["split_hash"]),
                "timestamps": TIMESTAMP_SEMANTICS,
                "timebase_id": str(timebase["id"]),
                "dataset_sample_count": str(len(entries)),
                "dataset_end_time_ns": str(dataset_end_time_ns),
                "declared_end_time_ns": str(declared_end_time_ns),
                "episode_boundaries_sha256": _sha256_bytes(
                    _json_bytes({"episode_boundaries": timebase["episode_boundaries"]})
                ),
                "training_loss_clock": "optimizer_step-as-seconds",
                "timeline_origin": "relative-zero-plus-1ns",
                "is_robot_capture_time": "false",
                "source_resolution": str(report["dataset"]["source_resolution"]),
                "primary_camera": primary_camera,
                "producer": "npa.groot.offline-learning",
            },
        )
        inspection = _validate_learning_mcap(
            output, run_id=run_id, camera_name=primary_camera
        )
        artifact = _put_bytes(client, output_uri, output.read_bytes())
    result = {"status": "written", "artifact": artifact, "inspect": inspection}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _set_rerun_time(rr: Any, recording: Any, timeline: str, seconds: float) -> None:
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds(timeline, seconds, recording=recording)
    else:
        rr.set_time(timeline, duration=seconds, recording=recording)


def _per_sample_action_errors(expert: Any, predicted: Any) -> tuple[float, float]:
    """Return MAE and true MSE for one sample's action residual vector."""

    import numpy as np

    residual = np.asarray(expert, dtype=np.float64) - np.asarray(
        predicted, dtype=np.float64
    )
    return float(np.mean(np.abs(residual))), float(np.mean(np.square(residual)))


def _learning_blueprint(rrb: Any, camera_name: str = "front") -> Any:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                origin=f"heldout/camera/{camera_name}",
                name="Held-out camera (native source pixels)",
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="actions",
                    contents="actions/**",
                    name="Expert vs predicted actions",
                ),
                rrb.TimeSeriesView(
                    origin="error",
                    contents="error/**",
                    name="Per-sample action error",
                ),
                rrb.TimeSeriesView(
                    origin="metrics",
                    # Per-horizon diagnostics intentionally use their own
                    # action_horizon_index clock. Keep them out of this default
                    # dataset_time view so Rerun never opens an empty/error panel.
                    contents=[
                        "metrics/heldout_before/**",
                        "metrics/heldout_after/**",
                    ],
                    name="Held-out before / after",
                ),
            ),
            rrb.TextDocumentView(origin="provenance", name="Evaluation provenance"),
            column_shares=[2.1, 1.7, 1.2],
        ),
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TimePanel(state=rrb.PanelState.Expanded, timeline=RERUN_TIMELINE),
        auto_layout=False,
    )


def emit_learning_rrd(
    report_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Write native Rerun archetypes with a replay-first learning blueprint."""

    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb

    client = _s3_client(s3_client)
    report, baseline, _posttrain, before_arrays, after_arrays = _evaluation_bundle(
        client, report_uri
    )
    if report["run_id"] != run_id:
        raise GrootVisualizationError("learning report run identity mismatch")
    with tempfile.TemporaryDirectory(prefix="npa-groot-learning-rrd-") as tmp:
        root = Path(tmp)
        rrd = root / "groot-learning.rrd"
        primary_camera = str(report["provenance"]["primary_camera"])
        images, fps = _decode_synchronized_camera(
            client,
            report["provenance"]["heldout_source_videos"],
            baseline["sample_alignment"],
            camera_name=primary_camera,
            root=root,
        )
        timebase = dict(report["visualizations"]["timebase"])
        entries = list(timebase.get("entries") or [])
        blueprint = _learning_blueprint(rrb, primary_camera)
        recording = rr.RecordingStream(RERUN_APPLICATION_ID, recording_id=run_id)
        rr.save(rrd, default_blueprint=blueprint, recording=recording)
        if hasattr(rr, "send_blueprint"):
            rr.send_blueprint(blueprint, recording=recording)
        provenance = (
            "# Offline held-out policy evaluation\n\n"
            "This is not a closed-loop rollout.  Times are dataset indices at "
            f"{fps:g} FPS, not wall-clock robot sensor timestamps.\n\n"
            f"- split: `{report['dataset']['split_hash']}`\n"
            f"- train: {report['dataset']['train_episodes']} episodes\n"
            f"- held-out: {report['dataset']['heldout_episodes']} episodes / "
            f"{report['dataset']['heldout_samples']} samples\n"
            f"- checkpoint: `{report['training']['checkpoint_uri']}`\n"
            f"- source resolution: {report['dataset']['source_resolution']}\n"
            f"- cameras: {report['dataset']['camera_names']}\n"
            f"- synchronized samples: {len(entries)}\n"
            f"- timebase: `{timebase['id']}`"
        )
        rr.log(
            "provenance", rr.TextDocument(provenance), static=True, recording=recording
        )
        # Training loss has its own factual optimizer-step clock.  It is logged
        # before dataset time is ever set so Rerun does not attach a fabricated,
        # compressed replay timestamp to optimizer evidence.
        for item in report["training"]["loss_history"]:
            optimizer_step = float(item["optimizer_step"])
            _set_rerun_time(rr, recording, "optimizer_step", optimizer_step)
            rr.log("train/loss", rr.Scalars(float(item["loss"])), recording=recording)
        for item in _evaluated_checkpoint_curve(report):
            _set_rerun_time(
                rr, recording, "optimizer_step", float(item["optimizer_step"])
            )
            rr.log(
                "validation/checkpoint_mse",
                rr.Scalars(float(item["mse"])),
                recording=recording,
            )
        horizon = report["evaluation"].get("per_horizon_mse") or {}
        for index, (before, after) in enumerate(
            zip(
                horizon.get("baseline") or [],
                horizon.get("posttrain") or [],
                strict=True,
            )
        ):
            _set_rerun_time(rr, recording, "action_horizon_index", float(index))
            rr.log(
                "metrics/per_horizon/baseline_mse",
                rr.Scalars(float(before)),
                recording=recording,
            )
            rr.log(
                "metrics/per_horizon/posttrain_mse",
                rr.Scalars(float(after)),
                recording=recording,
            )
        expert = before_arrays["expert"]
        predicted_before = before_arrays["predicted"]
        predicted_after = after_arrays["predicted"]
        if len(images) != len(expert) or len(entries) != len(expert):
            raise GrootVisualizationError("RRD camera/action/timebase sample mismatch")
        for index in range(len(expert)):
            _set_rerun_time(
                rr, recording, RERUN_TIMELINE, float(entries[index]["time_seconds"])
            )
            rr.log(
                f"heldout/camera/{primary_camera}",
                rr.Image(np.asarray(images[index]), color_model="RGB"),
                recording=recording,
            )
            for dimension in range(expert.shape[1]):
                rr.log(
                    f"actions/expert/dim_{dimension}",
                    rr.Scalars(float(expert[index, dimension])),
                    recording=recording,
                )
                rr.log(
                    f"actions/predicted_before/dim_{dimension}",
                    rr.Scalars(float(predicted_before[index, dimension])),
                    recording=recording,
                )
                rr.log(
                    f"actions/predicted_after/dim_{dimension}",
                    rr.Scalars(float(predicted_after[index, dimension])),
                    recording=recording,
                )
            before_abs, before_squared = _per_sample_action_errors(
                expert[index], predicted_before[index]
            )
            after_abs, after_squared = _per_sample_action_errors(
                expert[index], predicted_after[index]
            )
            rr.log("error/before/absolute", rr.Scalars(before_abs), recording=recording)
            rr.log("error/after/absolute", rr.Scalars(after_abs), recording=recording)
            rr.log(
                "error/before/squared", rr.Scalars(before_squared), recording=recording
            )
            rr.log(
                "error/after/squared", rr.Scalars(after_squared), recording=recording
            )
        replay_duration = max(
            (len(expert) - 1) / float(baseline["fps"]),
            1.0,
        )
        for metric_time in (0.0, replay_duration):
            _set_rerun_time(rr, recording, RERUN_TIMELINE, metric_time)
            rr.log(
                "metrics/heldout_before/mse",
                rr.Scalars(float(report["evaluation"]["baseline_value"])),
                recording=recording,
            )
            rr.log(
                "metrics/heldout_after/mse",
                rr.Scalars(float(report["evaluation"]["posttrain_value"])),
                recording=recording,
            )
        # Rerun writes the footer, manifests, and trailing chunks when the file
        # sink is disconnected.  Inspecting an attached sink races the batching
        # pipeline and yields a truncated, non-replayable RRD.
        recording.flush(timeout_sec=60.0)
        recording.disconnect()
        inspection = inspect_rrd(
            rrd,
            application_id=RERUN_APPLICATION_ID,
            recording_id=run_id,
            expected_entities=[
                entity.replace(
                    "heldout/camera/front", f"heldout/camera/{primary_camera}"
                )
                for entity in REQUIRED_RRD_ENTITIES
            ],
            timeline=RERUN_TIMELINE,
        )
        artifact = _put_bytes(client, output_uri, rrd.read_bytes())
    result = {"status": "written", "artifact": artifact, "inspect": inspection}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def publish_learning(
    report_uri: str,
    mcap_uri: str,
    rrd_uri: str,
    video_uri: str,
    workflow_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Independently validate, hash, and index all learning outputs."""

    import yaml

    client = _s3_client(s3_client)
    report = _read_s3_json(client, report_uri)
    if report.get("schema") != REPORT_SCHEMA or report.get("run_id") != run_id:
        raise GrootVisualizationError("publish report identity mismatch")
    workflow = yaml.safe_load(_read_s3_bytes(client, workflow_uri))
    if (
        not isinstance(workflow, dict)
        or workflow.get("apiVersion") != "npa.workflow/v0.0.1"
    ):
        raise GrootVisualizationError("submitted workflow is not v0.0.1")
    provenance = report.get("provenance")
    dataset = report.get("dataset")
    visualizations = report.get("visualizations")
    if not all(isinstance(value, Mapping) for value in (provenance, dataset, visualizations)):
        raise GrootVisualizationError("learning report lacks modality provenance")
    primary_camera = str(provenance.get("primary_camera") or "").strip()
    camera_names = [str(value) for value in dataset.get("camera_names") or []]
    timebase_cameras = [
        str(value)
        for value in (visualizations.get("timebase") or {}).get("camera_names") or []
    ]
    video_cameras = {
        str(item.get("camera_name") or "")
        for item in provenance.get("heldout_source_videos") or []
        if isinstance(item, Mapping)
    }
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+", primary_camera)
        or primary_camera not in camera_names
        or primary_camera not in timebase_cameras
        or primary_camera not in video_cameras
    ):
        raise GrootVisualizationError(
            "primary camera is missing or inconsistent across report modality provenance"
        )
    expected_rrd_entities = [
        entity.replace("heldout/camera/front", f"heldout/camera/{primary_camera}")
        for entity in REQUIRED_RRD_ENTITIES
    ]
    with tempfile.TemporaryDirectory(prefix="npa-groot-learning-publish-") as tmp:
        root = Path(tmp)
        mcap_path = root / "groot-learning.mcap"
        rrd_path = root / "groot-learning.rrd"
        video_path = root / "offline-heldout-comparison.mp4"
        _download(client, mcap_uri, mcap_path)
        _download(client, rrd_uri, rrd_path)
        _download(client, video_uri, video_path)
        mcap_inspection = _validate_learning_mcap(
            mcap_path, run_id=run_id, camera_name=primary_camera
        )
        rrd_inspection = inspect_rrd(
            rrd_path,
            application_id=RERUN_APPLICATION_ID,
            recording_id=run_id,
            expected_entities=expected_rrd_entities,
            timeline=RERUN_TIMELINE,
        )
        images, _fps = _decode_video(video_path, max_frames=1)
        expected_resolution = str(
            (visualizations.get("comparison_video") or {}).get("resolution")
            or visualizations.get("comparison_video_resolution")
            or ""
        )
        match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", expected_resolution)
        if match is None or images[0].size != (int(match.group(1)), int(match.group(2))):
            raise GrootVisualizationError(
                "comparison video resolution does not match report provenance"
            )
        hashes = {
            "mcap": _sha256_bytes(mcap_path.read_bytes()),
            "rrd": _sha256_bytes(rrd_path.read_bytes()),
            "comparison_video": _sha256_bytes(video_path.read_bytes()),
            "workflow": _sha256_bytes(_read_s3_bytes(client, workflow_uri)),
        }
    report["visualizations"].update(
        {
            "mcap_uri": mcap_uri,
            "mcap_bytes": mcap_inspection["size_bytes"],
            "mcap_topics": sorted(mcap_inspection["channels"]),
            "rrd_uri": rrd_uri,
            "rrd_bytes": rrd_inspection["bytes"],
            "rrd_entities": expected_rrd_entities,
            "comparison_video_uri": video_uri,
            "comparison_video_resolution": expected_resolution,
            "primary_camera": primary_camera,
        }
    )
    _put_json(client, report_uri, report)
    result = {
        "schema": PUBLISH_SCHEMA,
        "status": "published",
        "pipeline_status": "succeeded",
        "learning_outcome": str(report.get("learning_outcome") or "inconclusive"),
        "candidate_promoted": report.get("candidate_promoted") is True,
        "run_id": run_id,
        "evaluation_kind": EVALUATION_KIND,
        "closed_loop": False,
        "workflow": {
            **_head_artifact(client, workflow_uri),
            "sha256": hashes["workflow"],
        },
        "learning_report": _head_artifact(client, report_uri),
        "artifacts": {
            "mcap": {**_head_artifact(client, mcap_uri), "sha256": hashes["mcap"]},
            "rrd": {**_head_artifact(client, rrd_uri), "sha256": hashes["rrd"]},
            "comparison_video": {
                **_head_artifact(client, video_uri),
                "sha256": hashes["comparison_video"],
            },
        },
        "mcap_inspection": mcap_inspection,
        "rrd_inspection": rrd_inspection,
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def verify_agent_ui_handoff(
    agent_url: str,
    report_uri: str,
    rrd_uri: str,
    mcap_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Exercise the deployed agent's run/artifact/viewer API for this exact run.

    Pixel-level browser validation remains an external E2E gate, but this stage
    makes the workflow fail if the deployed agent cannot discover the run, list
    the required artifacts, load both native viewers, or serve byte ranges.
    Authentication is read only from ``NPA_AGENT_BASIC_AUTH`` and is never
    persisted or printed.
    """

    client = _s3_client(s3_client)
    origin = str(agent_url or "").strip().rstrip("/")
    if not origin.startswith(("http://", "https://")):
        raise GrootVisualizationError("agent_url must be an HTTP(S) origin")
    credentials = os.environ.get("NPA_AGENT_BASIC_AUTH", "").strip()
    if not credentials or ":" not in credentials:
        raise GrootVisualizationError("NPA_AGENT_BASIC_AUTH is required for UI verification")
    authorization = "Basic " + base64.b64encode(credentials.encode()).decode()
    context = None
    if origin.startswith("https://") and os.environ.get("NPA_AGENT_TLS_VERIFY", "1") == "0":
        import ssl

        context = ssl._create_unverified_context()  # noqa: S323 - explicit operator setting

    def request_json(path: str, *, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = None if body is None else _json_bytes(body)
        request = urllib.request.Request(
            origin + path,
            data=payload,
            headers={
                "Authorization": authorization,
                **({"Content-Type": "application/json"} if payload is not None else {}),
            },
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=60, context=context) as response:
                parsed = json.loads(response.read().decode())
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise GrootVisualizationError(f"agent API request failed for {path.split('?', 1)[0]}") from exc
        if not isinstance(parsed, dict):
            raise GrootVisualizationError("agent API returned a non-object response")
        return parsed

    def range_ok(path: str) -> bool:
        request = urllib.request.Request(
            urllib.parse.urljoin(origin + "/", str(path).lstrip("/")),
            headers={"Authorization": authorization, "Range": "bytes=0-63"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60, context=context) as response:
                body = response.read()
                return response.status in {200, 206} and 0 < len(body) <= 64
        except (OSError, urllib.error.URLError):
            return False

    health = request_json("/api/health")
    selector = urllib.parse.quote(run_id, safe="")
    inventory = request_json(f"/api/artifacts/run/{selector}")
    if str(inventory.get("run_id") or "") != run_id:
        raise GrootVisualizationError("agent loaded a different run identity")
    artifacts = [item for item in inventory.get("artifacts") or [] if isinstance(item, dict)]
    required = {
        "report": report_uri,
        "rrd": rrd_uri,
        "mcap": mcap_uri,
    }
    selected: dict[str, dict[str, Any]] = {}
    for label, uri in required.items():
        matches = [item for item in artifacts if str(item.get("s3_uri") or "") == uri]
        if len(matches) != 1:
            raise GrootVisualizationError(f"agent inventory lacks unique {label} artifact")
        selected[label] = matches[0]

    loads: dict[str, dict[str, Any]] = {}
    ranges: dict[str, bool] = {}
    for label in ("rrd", "mcap"):
        item = selected[label]
        loaded = request_json(
            "/api/sim-viz/load-artifact",
            body={
                "run_id": run_id,
                "run_ref": str(inventory.get("run_ref") or ""),
                "key": str(item.get("key") or ""),
                "s3_uri": str(item.get("s3_uri") or ""),
            },
        )
        sim_viz_value = loaded.get("sim_viz")
        sim_viz: dict[str, Any] = (
            dict(sim_viz_value) if isinstance(sim_viz_value, dict) else {}
        )
        ready = (
            sim_viz.get("rerun_ready") is True
            if label == "rrd"
            else sim_viz.get("lichtblick_ready") is True
        )
        if not ready:
            raise GrootVisualizationError(f"agent {label} viewer did not become ready")
        download_path = str(sim_viz.get("artifact_download_url") or "")
        ranges[label] = bool(download_path) and range_ok(download_path)
        if not ranges[label]:
            raise GrootVisualizationError(f"agent {label} byte-range endpoint failed")
        loads[label] = {
            "artifact_key": str(sim_viz.get("artifact_key") or ""),
            "viewer_ready": True,
            "range_download_verified": True,
        }

    result = {
        "schema": "npa.groot.agent_ui_handoff.v1",
        "status": "passed",
        "pipeline_status": "succeeded",
        "run_id": run_id,
        "agent_url": origin,
        "health_ok": bool(health.get("ok", True)),
        "run_discovered": True,
        "artifact_count": len(artifacts),
        "required_artifacts": sorted(required),
        "viewer_loads": loads,
        "range_downloads": ranges,
        "browser_validation": "required externally after workflow completion",
    }
    _put_json(client, output_uri, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight-rigor")
    preflight.add_argument("--output-uri", required=True)
    preflight.add_argument("--run-id", required=True)
    preflight.add_argument("--gpu-type", required=True)
    preflight.add_argument("--gpu-count", type=int, required=True)
    preflight.add_argument("--global-batch-size", type=int, required=True)
    preflight.add_argument("--per-device-batch-size", type=int, required=True)
    preflight.add_argument("--gradient-accumulation-steps", type=int, required=True)
    preflight.add_argument("--train-episodes", type=int, required=True)
    preflight.add_argument("--validation-episodes", type=int, required=True)
    preflight.add_argument("--final-episodes", type=int, required=True)
    preflight.add_argument("--max-steps", type=int, required=True)
    preflight.add_argument("--save-steps", type=int, required=True)
    preflight.add_argument("--save-total-limit", type=int, required=True)
    preflight.add_argument("--minimum-epochs", type=float, required=True)

    split = subparsers.add_parser("prepare-split")
    split.add_argument("--source-uri", required=True)
    split.add_argument("--train-uri", required=True)
    split.add_argument("--heldout-uri", required=True)
    split.add_argument("--output-uri", required=True)
    split.add_argument("--run-id", required=True)
    split.add_argument("--train-episodes", type=int, default=24)
    split.add_argument("--heldout-episodes", type=int, default=6)
    split.add_argument("--final-uri", default="")
    split.add_argument("--final-episodes", type=int, default=0)
    split.add_argument("--seed", default="groot17-learning-v1")
    split.add_argument("--global-batch-size", type=int, default=8)
    split.add_argument("--minimum-epochs", type=float, default=1.0)
    split.add_argument("--minimum-effective-global-batch", type=int, default=1)
    split.add_argument("--gpu-count", type=int)
    split.add_argument("--per-device-batch-size", type=int)
    split.add_argument("--gradient-accumulation-steps", type=int)
    split.add_argument(
        "--action-representation", choices=("source", "absolute"), default="source"
    )
    split.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Explicit GPU step budget; zero derives exactly one complete train pass.",
    )

    baseline = subparsers.add_parser("baseline-eval")
    baseline.add_argument("--split-manifest-uri", required=True)
    baseline.add_argument("--output-uri", required=True)
    baseline.add_argument("--arrays-uri", required=True)
    baseline.add_argument("--baseline-checkpoint-uri", required=True)
    baseline.add_argument("--base-model", required=True)
    baseline.add_argument("--run-id", required=True)
    baseline.add_argument("--action-horizon", type=int, default=16)
    baseline.add_argument("--evaluation-repeats", type=int, default=5)

    posttrain = subparsers.add_parser("posttrain-eval")
    posttrain.add_argument("--split-manifest-uri", required=True)
    posttrain.add_argument("--checkpoint-ref-uri", required=True)
    posttrain.add_argument("--output-uri", required=True)
    posttrain.add_argument("--arrays-uri", required=True)
    posttrain.add_argument("--run-id", required=True)
    posttrain.add_argument("--action-horizon", type=int, default=16)
    posttrain.add_argument("--evaluation-repeats", type=int, default=5)

    compare = subparsers.add_parser("compare-learning")
    compare.add_argument("--split-manifest-uri", required=True)
    compare.add_argument("--baseline-uri", required=True)
    compare.add_argument("--posttrain-uri", required=True)
    compare.add_argument("--training-manifest-uri", required=True)
    compare.add_argument("--output-uri", required=True)
    compare.add_argument("--video-uri", required=True)
    compare.add_argument("--run-id", required=True)
    compare.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=DEFAULT_MINIMUM_RELATIVE_IMPROVEMENT,
    )
    compare.add_argument(
        "--minimum-skill-score", type=float, default=DEFAULT_MINIMUM_SKILL_SCORE
    )
    compare.add_argument(
        "--repeat-noise-multiple", type=float, default=DEFAULT_REPEAT_NOISE_MULTIPLE
    )
    compare.add_argument(
        "--max-dimension-regression",
        type=float,
        default=DEFAULT_MAX_DIMENSION_REGRESSION,
    )
    compare.add_argument("--loss-decrease-tolerance", type=float, default=0.01)

    mcap = subparsers.add_parser("emit-mcap")
    mcap.add_argument("--report-uri", required=True)
    mcap.add_argument("--output-uri", required=True)
    mcap.add_argument("--run-id", required=True)

    rrd = subparsers.add_parser("emit-rrd")
    rrd.add_argument("--report-uri", required=True)
    rrd.add_argument("--output-uri", required=True)
    rrd.add_argument("--run-id", required=True)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--report-uri", required=True)
    publish.add_argument("--mcap-uri", required=True)
    publish.add_argument("--rrd-uri", required=True)
    publish.add_argument("--video-uri", required=True)
    publish.add_argument("--workflow-uri", required=True)
    publish.add_argument("--output-uri", required=True)
    publish.add_argument("--run-id", required=True)

    agent_ui = subparsers.add_parser("verify-agent-ui")
    agent_ui.add_argument("--agent-url", required=True)
    agent_ui.add_argument("--report-uri", required=True)
    agent_ui.add_argument("--rrd-uri", required=True)
    agent_ui.add_argument("--mcap-uri", required=True)
    agent_ui.add_argument("--output-uri", required=True)
    agent_ui.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values = vars(args).copy()
    command = values.pop("command")
    if command == "preflight-rigor":
        preflight_rigor_contract(**values)
    elif command == "prepare-split":
        prepare_split(**values)
    elif command == "baseline-eval":
        baseline_eval(**values)
    elif command == "posttrain-eval":
        posttrain_eval(**values)
    elif command == "compare-learning":
        compare_learning(**values)
    elif command == "emit-mcap":
        emit_learning_mcap(**values)
    elif command == "emit-rrd":
        emit_learning_rrd(**values)
    elif command == "publish":
        publish_learning(**values)
    elif command == "verify-agent-ui":
        verify_agent_ui_handoff(**values)
    else:  # pragma: no cover - argparse rejects unknown commands
        raise GrootVisualizationError(f"unknown command: {command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
