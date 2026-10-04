"""Pinned LingBot-VA LIBERO-Long workflow stages.

This adapter intentionally keeps models and data out of image layers. The
preparation stage consumes a caller-staged copy of the exact public
HuggingFaceVLA/libero CC-BY-4.0 LeRobot v2.1 revision, selects its real
LIBERO-Long records, and creates fresh run-scoped Wan latents. It never
consumes Robbyant's separate CC-BY-NC-SA latent dataset.

The executable model path is upstream-native: ``wan_va.train`` (Flex Attention)
and ``wan_va_server.VA_Server`` (Torch SDPA).  The narrow configuration overlay
only replaces upstream placeholder paths and disables the example W&B telemetry
configuration; it does not replace training, inference, or LIBERO rollout code.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any


SOURCE_REPO = "https://github.com/Robbyant/lingbot-va"
SOURCE_REF = "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb"
BASE_MODEL_ID = "robbyant/lingbot-va-base"
BASE_MODEL_REF = "68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c"
POSTTRAIN_MODEL_ID = "robbyant/lingbot-va-posttrain-libero-long"
POSTTRAIN_MODEL_REF = "0e89d1e753019988aba484e8da2dc0810e264d9f"
RAW_DATASET_ID = "HuggingFaceVLA/libero"
RAW_DATASET_REF = "affa19c0de0f6bce2a7edd26dddef8a532e7e6f6"
RAW_DATASET_LICENSE = "CC-BY-4.0"
RAW_DATASET_FORMAT = "LeRobot v2.1"
RAW_SOURCE_MANIFEST = "npa-lingbot-va-source.json"
LIBERO_REF = "8f1084e3132a39270c3a13ebe37270a43ece2a01"

TRAINING_ATTENTION = "flex"
INFERENCE_ATTENTION = "torch"
LIBERO_BENCHMARK = "libero_90"
UPSTREAM_TRAIN_STEPS = 5000
UPSTREAM_TRAIN_GPUS = 8

ACTION_CONTRACT: dict[str, Any] = {
    "action_dim": 30,
    "used_action_channel_ids": list(range(7)),
    "action_per_frame": 4,
    "action_norm_method": "quantiles",
    "snr_shift": 5.0,
    "action_snr_shift": 0.05,
}
CAMERAS = (
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
)
RAW_CAMERA_FIELDS = (
    "observation.images.image",
    "observation.images.image2",
)
# The authoritative HuggingFaceVLA conversion maps the original LIBERO-Long
# tasks to IDs 0 through 9. Some of the multi-object instructions do not use
# the literal word "and", so pin their exact reviewed labels rather than apply
# a brittle natural-language heuristic or silently substitute shorter tasks.
LIBERO_LONG_TASKS = {
    0: "put the white mug on the left plate and put the yellow and white mug on the right plate",
    1: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    2: "put the yellow and white mug in the microwave and close it",
    3: "turn on the stove and put the moka pot on it",
    4: "put both the alphabet soup and the cream cheese box in the basket",
    5: "put both the alphabet soup and the tomato sauce in the basket",
    6: "put both moka pots on the stove",
    7: "put both the cream cheese box and the butter in the basket",
    8: "put the black bowl in the bottom drawer of the cabinet and close it",
    9: "pick up the book and place it in the back compartment of the caddy",
}
LIBERO_LONG_TASK_IDS = tuple(LIBERO_LONG_TASKS)
LATENT_FPS = 10
LATENT_HEIGHT = 128
LATENT_WIDTH = 128


def _is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _local_uri_path(uri: str) -> Path:
    return Path(uri.removeprefix("file://"))


def _read_json(uri: str, destination: Path) -> dict[str, Any]:
    _download_file(uri, destination)
    return json.loads(destination.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _download_file(uri: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_s3(uri):
        from npa.clients.storage import StorageClient

        StorageClient.from_environment().download_file(uri, str(destination))
    else:
        source = _local_uri_path(uri)
        if not source.is_file():
            raise FileNotFoundError(f"Expected file input at {uri}")
        shutil.copy2(source, destination)
    return destination


def _download_tree(uri: str, destination: Path) -> Path:
    if _is_s3(uri):
        from npa.clients.storage import StorageClient

        StorageClient.from_environment().download_directory(uri, str(destination))
    else:
        source = _local_uri_path(uri)
        if not source.is_dir():
            raise FileNotFoundError(f"Expected directory input at {uri}")
        shutil.copytree(source, destination, dirs_exist_ok=False)
    return destination


def _upload_file(path: Path, uri: str) -> str:
    if _is_s3(uri):
        from npa.clients.storage import StorageClient

        return StorageClient.from_environment().upload_file(str(path), uri)
    destination = _local_uri_path(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return f"file://{destination}"


def _upload_tree(path: Path, uri: str) -> str:
    if _is_s3(uri):
        from npa.clients.storage import StorageClient

        return StorageClient.from_environment().upload_directory(str(path), uri)
    destination = _local_uri_path(uri)
    shutil.copytree(path, destination, dirs_exist_ok=False)
    return f"file://{destination}"


def _tree_inventory_digest(root: Path) -> str:
    """Hash a stable filename/size inventory without reading data or weights twice."""
    digest = hashlib.sha256()
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _uri_join(uri: str, relative_path: str) -> str:
    """Append a repository-relative path to a staged local or S3 directory URI."""
    return f"{uri.rstrip('/')}/{relative_path.lstrip('/')}"


def _action_contract_from_quantiles(
    q01: list[float], q99: list[float]
) -> dict[str, Any]:
    """Build the exact seven-active-channel LingBot contract from raw LIBERO data."""
    if len(q01) != 7 or len(q99) != 7:
        raise ValueError("LIBERO action quantiles must contain exactly seven channels")
    contract = dict(ACTION_CONTRACT)
    contract["q01"] = [float(value) for value in q01] + [0.0] * 23
    contract["q99"] = [float(value) for value in q99] + [0.0] * 23
    _assert_action_contract(contract)
    return contract


def _assert_action_contract(contract: dict[str, Any]) -> None:
    """Reject a changed action shape, schedule, or raw-data normalization contract."""
    for key, expected in ACTION_CONTRACT.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"LingBot-VA action contract has incompatible {key}: {contract.get(key)!r}"
            )
    q01 = contract.get("q01")
    q99 = contract.get("q99")
    if not isinstance(q01, list) or not isinstance(q99, list):
        raise ValueError("LingBot-VA action contract requires list quantiles")
    if len(q01) != 30 or len(q99) != 30:
        raise ValueError("LingBot-VA action quantiles must have all 30 padded channels")
    if any(float(lower) >= float(upper) for lower, upper in zip(q01[:7], q99[:7])):
        raise ValueError("Every active LIBERO action quantile range must be non-empty")
    if any(float(value) != 0.0 for value in [*q01[7:], *q99[7:]]):
        raise ValueError("Inactive LingBot-VA action channels must remain zero-padded")


def _episode_records(root: Path) -> list[dict[str, Any]]:
    episodes = root / "meta" / "episodes.jsonl"
    info = root / "meta" / "info.json"
    if not episodes.is_file() or not info.is_file():
        raise ValueError("Expected LeRobot meta/info.json and meta/episodes.jsonl")
    if not (root / "empty_emb.pt").is_file():
        raise ValueError("LingBot-VA training requires dataset-local empty_emb.pt")
    records = [
        json.loads(line)
        for line in episodes.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not records:
        raise ValueError("LeRobot episodes.jsonl is empty")
    for record in records:
        index = record.get("episode_index")
        length = record.get("length")
        configs = record.get("action_config")
        if type(index) is not int or type(length) is not int or length <= 0:
            raise ValueError(
                "Every episode needs integer episode_index and positive length"
            )
        if not isinstance(configs, list) or not configs:
            raise ValueError(f"Episode {index} lacks required action_config")
        for config in configs:
            start, end, text = (
                config.get("start_frame"),
                config.get("end_frame"),
                config.get("action_text"),
            )
            if (
                type(start) is not int
                or type(end) is not int
                or not (0 <= start < end <= length)
            ):
                raise ValueError(
                    f"Episode {index} has invalid action_config frame bounds"
                )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Episode {index} has action_config without action_text"
                )
            for camera in CAMERAS:
                pattern = (
                    f"latents/chunk-*/{camera}/episode_{index:06d}_{start}_{end}.pth"
                )
                if not list(root.glob(pattern)):
                    raise ValueError(
                        f"Episode {index} is missing expected latent {pattern}"
                    )
                video_pattern = f"videos/chunk-*/{camera}/episode_{index:06d}.mp4"
                videos = list(root.glob(video_pattern))
                if not videos or any(video.stat().st_size == 0 for video in videos):
                    raise ValueError(
                        f"Episode {index} is missing expected non-empty MP4 {video_pattern}"
                    )
    return records


def _load_raw_hfvla_metadata(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, str]]:
    """Read and validate the exact CC-BY raw LeRobot v2.1 metadata files."""
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.jsonl"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not all(path.is_file() for path in (info_path, tasks_path, episodes_path)):
        raise ValueError("Raw HuggingFaceVLA input lacks LeRobot v2.1 metadata")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            "LingBot-VA preparation requires the pinned LeRobot v2.1 input"
        )
    if info.get("fps") != LATENT_FPS:
        raise ValueError("HuggingFaceVLA LIBERO source must retain its 10 Hz cadence")
    features = info.get("features")
    if not isinstance(features, dict) or any(
        field not in features for field in (*RAW_CAMERA_FIELDS, "action")
    ):
        raise ValueError(
            "Raw HuggingFaceVLA input lacks its two image fields or 7D action"
        )
    action = features["action"]
    if action.get("shape") != [7]:
        raise ValueError("Raw HuggingFaceVLA LIBERO action must be exactly 7D")
    tasks = {
        row["task_index"]: row["task"]
        for row in (
            json.loads(line)
            for line in tasks_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    if set(LIBERO_LONG_TASK_IDS).difference(tasks):
        raise ValueError(
            "Pinned HuggingFaceVLA metadata lacks the LIBERO-Long task IDs"
        )
    if {
        task_id: tasks[task_id] for task_id in LIBERO_LONG_TASK_IDS
    } != LIBERO_LONG_TASKS:
        raise ValueError(
            "Pinned LIBERO-Long task labels do not match the reviewed source"
        )
    task_id_by_text = {text: task_id for task_id, text in tasks.items()}
    selected: list[dict[str, Any]] = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        labels = record.get("tasks")
        if not isinstance(labels, list) or len(labels) != 1:
            raise ValueError(
                "Every raw LIBERO episode must have one authoritative task label"
            )
        task_id = task_id_by_text.get(labels[0])
        if task_id in LIBERO_LONG_TASK_IDS:
            if (
                type(record.get("episode_index")) is not int
                or record.get("length", 0) <= 0
            ):
                raise ValueError("Raw LIBERO episode has invalid index or length")
            selected.append({**record, "task_index": task_id})
    if len(selected) < 20:
        raise ValueError(
            "Pinned LIBERO source contains too few actual LIBERO-Long episodes"
        )
    return info, sorted(selected, key=lambda record: record["episode_index"]), tasks


def _load_raw_source_manifest(source_uri: str, destination: Path) -> dict[str, Any]:
    """Require the immutable source/terms receipt written before data staging.

    A public URI alone is not enough to establish that this run is receiving
    the reviewed source revision. The private staging helper writes this small
    receipt only after every selected raw parquet and metadata object has
    uploaded; preparation consumes it before downloading any raw episode.
    """
    manifest_path = _download_file(
        _uri_join(source_uri, RAW_SOURCE_MANIFEST), destination / RAW_SOURCE_MANIFEST
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("Raw LIBERO source receipt is not complete")
    expected = {
        "dataset_id": RAW_DATASET_ID,
        "dataset_revision": RAW_DATASET_REF,
        "dataset_license": RAW_DATASET_LICENSE,
        "dataset_format": RAW_DATASET_FORMAT,
        "selected_task_ids": list(LIBERO_LONG_TASK_IDS),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Raw LIBERO source receipt has incompatible {key}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Raw LIBERO source receipt lacks per-file provenance")
    return manifest


def _split_long_episodes(
    records: list[dict[str, Any]], heldout_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make a deterministic per-task split so every long task remains trainable."""
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task_index"]].append(record)
    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for task_id in LIBERO_LONG_TASK_IDS:
        task_records = sorted(
            by_task[task_id], key=lambda record: record["episode_index"]
        )
        heldout_count = max(1, round(len(task_records) * heldout_fraction))
        if heldout_count >= len(task_records):
            raise ValueError(f"LIBERO-Long task {task_id} needs at least two episodes")
        train.extend(task_records[:-heldout_count])
        heldout.extend(task_records[-heldout_count:])
    return train, heldout


def _raw_episode_relative_path(record: dict[str, Any], chunk_size: int) -> str:
    """Return the authoritative v2.1 parquet location for one raw episode."""
    episode_index = record.get("source_episode_index", record["episode_index"])
    return f"data/chunk-{episode_index // chunk_size:03d}/episode_{episode_index:06d}.parquet"


def _derived_episode_relative_path(record: dict[str, Any], chunk_size: int) -> str:
    """Return the contiguous derived v2.1 parquet location for one output episode."""
    episode_index = record["episode_index"]
    return f"data/chunk-{episode_index // chunk_size:03d}/episode_{episode_index:06d}.parquet"


def _derive_records(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reindex sparse source records for LeRobot 0.3.3's positional frame lookup.

    The ten long tasks are interleaved with thirty shorter tasks in the source.
    Their source episode IDs are consequently sparse; preserving them would
    cause LeRobot's cumulative frame table to use an episode ID as an array
    position. The derivative uses contiguous IDs and records the source map in
    its manifest provenance.
    """
    return [
        {
            **record,
            "source_episode_index": record["episode_index"],
            "episode_index": new_index,
        }
        for new_index, record in enumerate(
            sorted(raw_records, key=lambda record: record["episode_index"])
        )
    ]


def _stage_raw_long_dataset(
    source_uri: str,
    destination: Path,
    records: list[dict[str, Any]],
    chunk_size: int,
) -> None:
    """Fetch only selected public raw records, never precomputed NC latent artifacts."""
    for relative_path in (
        RAW_SOURCE_MANIFEST,
        "README.md",
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
    ):
        _download_file(
            _uri_join(source_uri, relative_path), destination / relative_path
        )
    for record in records:
        relative_path = _raw_episode_relative_path(record, chunk_size)
        _download_file(
            _uri_join(source_uri, relative_path), destination / relative_path
        )


def _raw_action_quantiles(
    root: Path, records: list[dict[str, Any]], chunk_size: int
) -> tuple[list[float], list[float]]:
    """Recompute 1st/99th-percentile normalization from training-only 7D actions."""
    import numpy as np
    import pyarrow.parquet as pq

    actions: list[Any] = []
    for record in records:
        table = pq.read_table(
            root / _raw_episode_relative_path(record, chunk_size), columns=["action"]
        )
        values = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
            raise ValueError(
                "Raw LIBERO parquet has non-finite or non-7D action values"
            )
        actions.append(values)
    merged = np.concatenate(actions, axis=0)
    q01, q99 = np.quantile(merged, [0.01, 0.99], axis=0)
    if not np.all(q01 < q99):
        raise ValueError("Raw LIBERO training actions have an empty quantile range")
    return q01.astype(float).tolist(), q99.astype(float).tolist()


def _write_processed_metadata(
    raw_root: Path,
    destination: Path,
    info: dict[str, Any],
    tasks: dict[int, str],
    records: list[dict[str, Any]],
    chunk_size: int,
) -> None:
    """Create the derived LeRobot tree consumed by the upstream latent loader."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    destination.mkdir(parents=True, exist_ok=False)
    global_frame_index = 0
    for record in records:
        relative_path = _raw_episode_relative_path(record, chunk_size)
        source = raw_root / relative_path
        target = destination / _derived_episode_relative_path(record, chunk_size)
        target.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(source)
        row_count = len(table)
        if row_count != record["length"]:
            raise ValueError("Raw LIBERO parquet length does not match its metadata")
        for name, values in {
            "episode_index": [record["episode_index"]] * row_count,
            "index": list(range(global_frame_index, global_frame_index + row_count)),
        }.items():
            field_index = table.schema.get_field_index(name)
            if field_index < 0:
                raise ValueError(f"Raw LIBERO parquet lacks required {name} column")
            table = table.set_column(
                field_index,
                name,
                pa.array(values, type=table.schema.field(field_index).type),
            )
        pq.write_table(table, target)
        global_frame_index += row_count
    output_info = dict(info)
    output_info["total_episodes"] = len(records)
    output_info["total_frames"] = sum(record["length"] for record in records)
    output_info["total_tasks"] = len(LIBERO_LONG_TASK_IDS)
    output_info["total_videos"] = len(records) * len(CAMERAS)
    output_info["total_chunks"] = (len(records) + chunk_size - 1) // chunk_size
    output_info["fps"] = LATENT_FPS
    output_info["video_path"] = (
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    _write_json(destination / "meta" / "info.json", output_info)
    selected_source_ids = {record["source_episode_index"] for record in records}
    derived_by_source = {
        record["source_episode_index"]: record["episode_index"] for record in records
    }
    (destination / "meta" / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": task_id, "task": tasks[task_id]}, sort_keys=True)
            + "\n"
            for task_id in LIBERO_LONG_TASK_IDS
        ),
        encoding="utf-8",
    )
    prepared_records = [
        {
            "episode_index": record["episode_index"],
            "source_episode_index": record["source_episode_index"],
            "tasks": record["tasks"],
            "length": record["length"],
            "action_config": [
                {
                    "start_frame": 0,
                    "end_frame": record["length"],
                    "action_text": record["tasks"][0],
                }
            ],
        }
        for record in records
    ]
    (destination / "meta" / "episodes.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in prepared_records
        ),
        encoding="utf-8",
    )
    source_stats = raw_root / "meta" / "episodes_stats.jsonl"
    if source_stats.is_file():
        selected_stats = [
            line
            for line in source_stats.read_text(encoding="utf-8").splitlines()
            if line and json.loads(line).get("episode_index") in selected_source_ids
        ]
        selected_stats = [
            json.dumps(
                {
                    **json.loads(line),
                    "episode_index": derived_by_source[
                        json.loads(line)["episode_index"]
                    ],
                },
                sort_keys=True,
            )
            for line in selected_stats
        ]
        (destination / "meta" / "episodes_stats.jsonl").write_text(
            "\n".join(selected_stats) + ("\n" if selected_stats else ""),
            encoding="utf-8",
        )


def _read_raw_episode_images(
    dataset: Path, record: dict[str, Any], chunk_size: int
) -> dict[str, list[Any]]:
    """Decode embedded PNG/JPEG fields from a pinned raw v2.1 parquet episode."""
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(
        dataset / _derived_episode_relative_path(record, chunk_size),
        columns=list(RAW_CAMERA_FIELDS),
    )
    images: dict[str, list[Any]] = {}
    for raw_key, camera in zip(RAW_CAMERA_FIELDS, CAMERAS, strict=True):
        decoded: list[Any] = []
        for item in table.column(raw_key).to_pylist():
            encoded = item.get("bytes") if isinstance(item, dict) else None
            if not isinstance(encoded, bytes):
                raise ValueError(
                    f"Raw LIBERO {raw_key} frame has no embedded image bytes"
                )
            with Image.open(BytesIO(encoded)) as image:
                decoded.append(
                    np.asarray(
                        image.convert("RGB").resize((LATENT_WIDTH, LATENT_HEIGHT))
                    )
                )
        if len(decoded) != record["length"]:
            raise ValueError(f"Raw LIBERO {raw_key} length does not match metadata")
        images[camera] = decoded
    return images


def _provenance() -> dict[str, Any]:
    return {
        "lingbot_va": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REF,
            "license": "Apache-2.0",
        },
        "base_checkpoint": {
            "id": BASE_MODEL_ID,
            "revision": BASE_MODEL_REF,
            "license": "Apache-2.0",
        },
        "official_libero_long_checkpoint": {
            "id": POSTTRAIN_MODEL_ID,
            "revision": POSTTRAIN_MODEL_REF,
            "license": "Apache-2.0",
        },
        "libero": {
            "repo": "https://github.com/Lifelong-Robot-Learning/LIBERO",
            "revision": LIBERO_REF,
        },
        "dataset": {
            "id": RAW_DATASET_ID,
            "revision": RAW_DATASET_REF,
            "license": RAW_DATASET_LICENSE,
            "format": RAW_DATASET_FORMAT,
            "delivery": "operator-staged raw source; fresh derived videos and latents are run-scoped",
            "selected_task_ids": list(LIBERO_LONG_TASK_IDS),
        },
        "attribution": ["Robbyant Team", "Wan-Video", "Mixture-of-Transformers (MoT)"],
    }


def _encode_text_embedding(
    tokenizer: Any, text_encoder: Any, text: str, device: Any
) -> Any:
    """Use the native Wan tokenizer/text encoder convention for one action label."""
    import torch
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    max_sequence_length = 512
    inputs = tokenizer(
        [prompt_clean(text)],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    sequence_length = inputs.attention_mask.gt(0).sum(dim=1).long()
    hidden = text_encoder(
        inputs.input_ids.to(device), inputs.attention_mask.to(device)
    ).last_hidden_state
    trimmed = hidden[0, : sequence_length[0]].to(dtype=torch.bfloat16)
    return torch.cat(
        [
            trimmed,
            trimmed.new_zeros(max_sequence_length - trimmed.shape[0], trimmed.shape[1]),
        ]
    ).cpu()


def _write_video(path: Path, frames: list[Any]) -> None:
    """Write the decoded source camera frames as a real 10 Hz MP4 artifact."""
    import cv2

    if not frames:
        raise ValueError("Cannot write a video with no raw LIBERO frames")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), LATENT_FPS, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create derived MP4 at {path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("Raw LIBERO image shape changed within one episode")
            writer.write(frame[..., ::-1])
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Derived MP4 is empty at {path}")


def _encode_video_latent(
    streaming_vae: Any, vae: Any, frames: list[Any], device: Any
) -> tuple[Any, int, int, int]:
    """Encode one decoded camera video through the upstream Wan2.2 VAE path."""
    import numpy as np
    import torch

    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    video = video.to(device=device, dtype=torch.bfloat16) / 255.0 * 2.0 - 1.0
    streaming_vae.clear_cache()
    encoded = streaming_vae.encode_chunk(video)
    streaming_vae.clear_cache()
    mean, _ = torch.chunk(encoded, 2, dim=1)
    latent_mean = torch.as_tensor(vae.config.latents_mean, device=device).view(
        1, -1, 1, 1, 1
    )
    latent_std = torch.as_tensor(vae.config.latents_std, device=device).view(
        1, -1, 1, 1, 1
    )
    normalized = (mean.float() - latent_mean) * latent_std.reciprocal()
    latent_num_frames, latent_height, latent_width = normalized.shape[-3:]
    flattened = normalized[0].permute(1, 2, 3, 0).reshape(-1, normalized.shape[1])
    return (
        flattened.to(dtype=torch.bfloat16, device="cpu"),
        latent_num_frames,
        latent_height,
        latent_width,
    )


def _materialize_native_latents(
    dataset: Path, records: list[dict[str, Any]], chunk_size: int, model_dir: Path
) -> None:
    """Generate videos, action-text embeddings, and Wan2.2 VAE latents on a GPU."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Fresh Wan2.2 latent preparation requires a CUDA GPU")
    _enable_upstream()
    from wan_va.modules.utils import (
        WanVAEStreamingWrapper,
        load_text_encoder,
        load_tokenizer,
        load_vae,
    )

    device = torch.device("cuda")
    vae = load_vae(str(model_dir / "vae"), torch.bfloat16, device).eval()
    text_encoder = load_text_encoder(
        str(model_dir / "text_encoder"), torch.bfloat16, device
    ).eval()
    tokenizer = load_tokenizer(str(model_dir / "tokenizer"))
    streaming_vae = WanVAEStreamingWrapper(vae)
    with torch.inference_mode():
        empty_embedding = _encode_text_embedding(tokenizer, text_encoder, "", device)
        torch.save(empty_embedding, dataset / "empty_emb.pt")
        for record in records:
            episode_index = record["episode_index"]
            start_frame = 0
            end_frame = record["length"]
            images = _read_raw_episode_images(dataset, record, chunk_size)
            text_embedding = _encode_text_embedding(
                tokenizer, text_encoder, record["tasks"][0], device
            )
            for camera in CAMERAS:
                frames = images[camera]
                chunk = episode_index // chunk_size
                video_path = (
                    dataset
                    / "videos"
                    / f"chunk-{chunk:03d}"
                    / camera
                    / f"episode_{episode_index:06d}.mp4"
                )
                _write_video(video_path, frames)
                latent, latent_num_frames, latent_height, latent_width = (
                    _encode_video_latent(streaming_vae, vae, frames, device)
                )
                latent_path = (
                    dataset
                    / "latents"
                    / f"chunk-{chunk:03d}"
                    / camera
                    / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                )
                latent_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "latent": latent,
                        "latent_num_frames": latent_num_frames,
                        "latent_height": latent_height,
                        "latent_width": latent_width,
                        "video_num_frames": len(frames),
                        "video_height": LATENT_HEIGHT,
                        "video_width": LATENT_WIDTH,
                        "text_emb": text_embedding,
                        "text": record["tasks"][0],
                        "frame_ids": list(range(len(frames))),
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "fps": LATENT_FPS,
                        "ori_fps": LATENT_FPS,
                    },
                    latent_path,
                )
    del text_encoder
    del vae
    torch.cuda.empty_cache()


def prepare(
    source_uri: str,
    prepared_dataset_uri: str,
    output_uri: str,
    *,
    heldout_fraction: float = 0.2,
) -> dict[str, Any]:
    """Create the native LingBot latent dataset from public raw LIBERO-Long data.

    The stage reads only the selected raw v2.1 parquet episodes, decodes their
    embedded cameras, adds source-task action segments, writes 10 Hz MP4s, and
    uses the upstream Wan2.2 VAE/text encoder for fresh latents.  It deliberately
    rejects precomputed Robbyant latent trees.
    """
    if not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction must be strictly between zero and one")
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-va-prepare-") as temporary:
        work = Path(temporary)
        source_receipt = _load_raw_source_manifest(source_uri, work / "source-receipt")
        metadata = work / "metadata"
        for relative_path in (
            "meta/info.json",
            "meta/tasks.jsonl",
            "meta/episodes.jsonl",
        ):
            _download_file(
                _uri_join(source_uri, relative_path), metadata / relative_path
            )
        info, raw_records, tasks = _load_raw_hfvla_metadata(metadata)
        records = _derive_records(raw_records)
        chunk_size = info.get("chunks_size")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("Raw HuggingFaceVLA metadata has no valid chunk size")
        train_records, heldout_records = _split_long_episodes(records, heldout_fraction)
        raw = work / "raw-long"
        _stage_raw_long_dataset(source_uri, raw, records, chunk_size)
        q01, q99 = _raw_action_quantiles(raw, train_records, chunk_size)
        action_contract = _action_contract_from_quantiles(q01, q99)
        prepared = work / "prepared-long"
        _write_processed_metadata(raw, prepared, info, tasks, records, chunk_size)
        model = _snapshot_base_checkpoint(work / "base-model")
        _materialize_native_latents(prepared, records, chunk_size, model)
        _episode_records(prepared)
        prepared_dataset_uri = _upload_tree(prepared, prepared_dataset_uri)
        manifest = {
            "schema": "npa.lingbot_va.prepared.v1",
            "stage": "prepare_hfvla_libero_long_with_native_wan_latents",
            "source_dataset_uri": source_uri,
            "source_dataset_id": RAW_DATASET_ID,
            "source_dataset_revision": RAW_DATASET_REF,
            "source_dataset_license": RAW_DATASET_LICENSE,
            "source_staging_manifest_sha256": hashlib.sha256(
                json.dumps(source_receipt, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "source_dataset_inventory_sha256": _tree_inventory_digest(raw),
            "prepared_dataset_uri": prepared_dataset_uri,
            "prepared_dataset_inventory_sha256": _tree_inventory_digest(prepared),
            "episode_count": len(records),
            "selected_libero_long_task_ids": list(LIBERO_LONG_TASK_IDS),
            "source_episode_mapping": [
                {
                    "prepared_episode_index": record["episode_index"],
                    "source_episode_index": record["source_episode_index"],
                    "task_index": record["task_index"],
                }
                for record in records
            ],
            "train_episode_indices": [
                record["episode_index"] for record in train_records
            ],
            "heldout_episode_indices": [
                record["episode_index"] for record in heldout_records
            ],
            "attention": {
                "training": TRAINING_ATTENTION,
                "inference": INFERENCE_ATTENTION,
            },
            "action_contract": action_contract,
            "latent_preparation": {
                "source_camera_fields": list(RAW_CAMERA_FIELDS),
                "derived_camera_fields": list(CAMERAS),
                "fps": LATENT_FPS,
                "height": LATENT_HEIGHT,
                "width": LATENT_WIDTH,
                "encoder": "upstream Wan2.2 VAE and text encoder from the pinned base checkpoint",
                "action_segments": "one source task-label segment spanning each actual episode",
                "action_mapping": "raw 7D LIBERO actions occupy channels 0..6; channels 7..29 remain zero-padded",
                "statistics": "training-split raw 7D q01/q99, then padded to 30 channels",
            },
            "provenance": _provenance(),
            "data_authorization": "CC-BY-4.0 source revision is operator-staged; no separate NPA acceptance is recorded",
        }
        local_manifest = work / "prepared.json"
        _write_json(local_manifest, manifest)
        manifest["prepared_manifest_uri"] = _upload_file(local_manifest, output_uri)
        return manifest


def _snapshot_base_checkpoint(destination: Path) -> Path:
    """Fetch the public pinned base checkpoint only in the private job runtime."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            BASE_MODEL_ID, revision=BASE_MODEL_REF, local_dir=str(destination)
        )
    )


def _snapshot_checkpoint(destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            POSTTRAIN_MODEL_ID, revision=POSTTRAIN_MODEL_REF, local_dir=str(destination)
        )
    )


def _replace_training_episodes(dataset: Path, train_indices: list[int]) -> None:
    episode_file = dataset / "meta" / "episodes.jsonl"
    records = [
        json.loads(line)
        for line in episode_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = [
        record for record in records if record["episode_index"] in set(train_indices)
    ]
    if not selected:
        raise ValueError("Prepared manifest selects no training episodes")
    episode_file.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in selected),
        encoding="utf-8",
    )


def _apply_action_contract(config: Any, action_contract: dict[str, Any]) -> None:
    """Apply source-derived normalization while retaining upstream LingBot modes."""
    _assert_action_contract(action_contract)
    config.action_dim = action_contract["action_dim"]
    config.used_action_channel_ids = list(action_contract["used_action_channel_ids"])
    config.action_per_frame = action_contract["action_per_frame"]
    config.action_norm_method = action_contract["action_norm_method"]
    config.snr_shift = action_contract["snr_shift"]
    config.action_snr_shift = action_contract["action_snr_shift"]
    config.norm_stat = {
        "q01": list(action_contract["q01"]),
        "q99": list(action_contract["q99"]),
    }
    inverse = [len(config.used_action_channel_ids)] * config.action_dim
    for position, channel_id in enumerate(config.used_action_channel_ids):
        inverse[channel_id] = position
    config.inverse_used_action_channel_ids = inverse


def _native_train(
    dataset_dir: Path,
    model_dir: Path,
    save_root: Path,
    action_contract: dict[str, Any],
) -> None:
    """Invoke the upstream FSDP trainer with only documented path/telemetry overrides."""
    import torch

    if torch.cuda.device_count() != UPSTREAM_TRAIN_GPUS:
        raise RuntimeError(
            f"Upstream LIBERO recipe requires exactly {UPSTREAM_TRAIN_GPUS} GPUs"
        )
    _enable_upstream()
    from wan_va.configs import VA_CONFIGS
    import wan_va.train as upstream_train

    config = VA_CONFIGS["libero_train"]
    config.dataset_path = str(dataset_dir.parent)
    config.empty_emb_path = str(dataset_dir / "empty_emb.pt")
    config.wan22_pretrained_model_name_or_path = str(model_dir)
    config.resume_from = str(model_dir)
    config.enable_wandb = False
    config.save_root = str(save_root)
    _apply_action_contract(config, action_contract)
    _assert_upstream_config(
        config, attention=TRAINING_ATTENTION, action_contract=action_contract
    )
    os.environ["WANDB_DISABLED"] = "true"
    upstream_train.run(
        argparse.Namespace(config_name="libero_train", save_root=str(save_root))
    )


def _assert_upstream_config(
    config: Any, *, attention: str, action_contract: dict[str, Any]
) -> None:
    applied = {
        "action_dim": config.action_dim,
        "used_action_channel_ids": list(config.used_action_channel_ids),
        "action_per_frame": config.action_per_frame,
        "action_norm_method": config.action_norm_method,
        "snr_shift": config.snr_shift,
        "action_snr_shift": config.action_snr_shift,
        "q01": list(config.norm_stat["q01"]),
        "q99": list(config.norm_stat["q99"]),
    }
    _assert_action_contract(applied)
    if applied != action_contract:
        raise ValueError(
            "Native LingBot configuration did not retain prepared action statistics"
        )
    if attention == TRAINING_ATTENTION and config.num_steps != UPSTREAM_TRAIN_STEPS:
        raise ValueError(
            "The LingBot-VA LIBERO recipe must retain upstream 5000 training steps"
        )


def _enable_upstream() -> None:
    source = os.environ.get("NPA_LINGBOT_VA_SOURCE", "/opt/lingbot-va")
    if source not in sys.path:
        sys.path.insert(0, source)


def train(prepared_uri: str, checkpoint_uri: str, training_uri: str) -> dict[str, Any]:
    """Continue the official checkpoint using the upstream eight-GPU LIBERO recipe."""
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-va-train-") as temporary:
        work = Path(temporary)
        prepared = _read_json(prepared_uri, work / "prepared.json")
        _assert_action_contract(prepared["action_contract"])
        dataset = _download_tree(prepared["prepared_dataset_uri"], work / "dataset")
        _episode_records(dataset)
        _replace_training_episodes(dataset, list(prepared["train_episode_indices"]))
        # Upstream LeRobot 0.3.3 resolves repo ids below HF_LEROBOT_HOME and
        # separately resolves latent paths relative to cwd.  The symlink keeps
        # one data copy while satisfying both native path contracts.
        hf_root = work / "hf-home"
        repo_id = RAW_DATASET_ID
        repo_root = hf_root / repo_id
        repo_root.parent.mkdir(parents=True)
        repo_root.symlink_to(dataset, target_is_directory=True)
        model = _snapshot_checkpoint(work / "official-posttrain")
        output = work / "output"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(UPSTREAM_TRAIN_GPUS),
            "-m",
            "npa.solutions.lingbot_va",
            "native-train",
            "--dataset-dir",
            str(repo_root),
            "--model-dir",
            str(model),
            "--save-root",
            str(output),
            "--action-contract-json",
            str(work / "action-contract.json"),
        ]
        _write_json(work / "action-contract.json", prepared["action_contract"])
        environment = dict(
            os.environ,
            HF_LEROBOT_HOME=str(hf_root),
            TOKENIZERS_PARALLELISM="false",
            WANDB_DISABLED="true",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        subprocess.run(command, check=True, cwd=hf_root, env=environment)
        checkpoints = output / "checkpoints"
        if not list(checkpoints.glob("checkpoint_step_*/transformer/config.json")):
            raise RuntimeError(
                "Upstream trainer completed without a usable transformer checkpoint"
            )
        _upload_tree(checkpoints, checkpoint_uri)
        summary = {
            "schema": "npa.lingbot_va.training.v1",
            "stage": "upstream_libero_long_posttrain",
            "prepared_manifest_uri": prepared_uri,
            "checkpoint_uri": checkpoint_uri,
            "training_attention": TRAINING_ATTENTION,
            "world_size": UPSTREAM_TRAIN_GPUS,
            "num_steps": UPSTREAM_TRAIN_STEPS,
            "telemetry": "disabled; no W&B API key, team, project, or consent is configured",
            "provenance": _provenance(),
        }
        local = work / "training.json"
        _write_json(local, summary)
        summary["training_manifest_uri"] = _upload_file(local, training_uri)
        return summary


def _overlay_latest_transformer(model_root: Path, checkpoints: Path) -> Path:
    candidates = sorted(checkpoints.glob("checkpoint_step_*/transformer"))
    if not candidates:
        raise ValueError(
            "Training checkpoint artifact has no upstream transformer checkpoint"
        )
    selected = candidates[-1]
    shutil.copytree(selected, model_root / "transformer", dirs_exist_ok=True)
    return selected


def _native_server(
    model_dir: Path, output: Path, port: int, action_contract: dict[str, Any]
) -> None:
    _enable_upstream()
    from wan_va.configs import VA_CONFIGS
    import wan_va.wan_va_server as upstream_server

    config = VA_CONFIGS["libero"]
    config.wan22_pretrained_model_name_or_path = str(model_dir)
    config.save_root = str(output)
    config.port = port
    _apply_action_contract(config, action_contract)
    _assert_upstream_config(
        config, attention=INFERENCE_ATTENTION, action_contract=action_contract
    )
    # The authoritative upstream server itself calls load_transformer(...,
    # attn_mode="torch").  Do not offer Flex Attention as an eval fallback.
    upstream_server.run(
        argparse.Namespace(config_name="libero", port=port, save_root=str(output))
    )


def _wait_for_port(
    port: int, process: subprocess.Popen[str], timeout: float = 180.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"LingBot-VA server stopped before becoming ready ({process.returncode})"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError("LingBot-VA websocket server did not become ready")


def rollout(
    prepared_uri: str,
    checkpoint_uri: str,
    output_uri: str,
    rollout_root_uri: str,
    *,
    test_num: int = 50,
    task_start: int = 0,
    task_end: int = 10,
) -> dict[str, Any]:
    """Run the actual upstream websocket policy and LIBERO-90 closed-loop client."""
    if test_num < 1 or task_start < 0 or task_end <= task_start:
        raise ValueError("Invalid LIBERO rollout range")
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-va-rollout-") as temporary:
        work = Path(temporary)
        prepared = _read_json(prepared_uri, work / "prepared.json")
        _assert_action_contract(prepared["action_contract"])
        checkpoints = _download_tree(checkpoint_uri, work / "checkpoints")
        model = _snapshot_checkpoint(work / "official-posttrain")
        selected = _overlay_latest_transformer(model, checkpoints)
        output = work / "rollout"
        output.mkdir()
        port = 23908
        server_command = [
            sys.executable,
            "-m",
            "npa.solutions.lingbot_va",
            "native-server",
            "--model-dir",
            str(model),
            "--output-dir",
            str(output),
            "--port",
            str(port),
            "--action-contract-json",
            str(work / "action-contract.json"),
        ]
        _write_json(work / "action-contract.json", prepared["action_contract"])
        environment = dict(os.environ, TOKENIZERS_PARALLELISM="false")
        with (output / "server.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                server_command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
            )
            try:
                _wait_for_port(port, server)
                _enable_upstream()
                from evaluation.libero.client import run as upstream_client_run

                # The upstream CLI omits libero_90 from choices, but its native
                # run function accepts every benchmark registered by LIBERO.
                upstream_client_run(
                    LIBERO_BENCHMARK,
                    port,
                    str(output / "libero"),
                    test_num,
                    [task_start, task_end],
                )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
        result_files = sorted(output.glob(f"libero/{LIBERO_BENCHMARK}_*.json"))
        if len(result_files) != task_end - task_start:
            raise RuntimeError(
                "LIBERO client did not publish a success result for every requested long-horizon task"
            )
        mp4s = sorted(output.glob(f"libero/{LIBERO_BENCHMARK}/**/*.mp4"))
        action_tensors = sorted(output.glob("real/**/actions_*.pt"))
        latent_tensors = sorted(output.glob("real/**/latents_*.pt"))
        if not mp4s or not action_tensors or not latent_tensors:
            raise RuntimeError(
                "Closed-loop rollout lacks real video, action, or predicted-latent artifacts"
            )
        _upload_tree(output, rollout_root_uri)
        manifest = {
            "schema": "npa.lingbot_va.rollout.v1",
            "stage": "libero_90_closed_loop_rollout",
            "prepared_manifest_uri": prepared_uri,
            "checkpoint_uri": checkpoint_uri,
            "checkpoint_transformer": selected.name,
            "rollout_root_uri": rollout_root_uri,
            "benchmark": LIBERO_BENCHMARK,
            "test_episodes_per_task": test_num,
            "task_range": [task_start, task_end],
            "inference_attention": INFERENCE_ATTENTION,
            "result_json_count": len(result_files),
            "rollout_mp4_count": len(mp4s),
            "predicted_action_chunk_count": len(action_tensors),
            "predicted_latent_chunk_count": len(latent_tensors),
            "provenance": _provenance(),
        }
        local = work / "rollout.json"
        _write_json(local, manifest)
        manifest["rollout_manifest_uri"] = _upload_file(local, output_uri)
        return manifest


def evaluate(
    rollout_uri: str, rollout_root_uri: str, output_uri: str
) -> dict[str, Any]:
    """Aggregate actual LIBERO success and action-prediction validity evidence."""
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-va-evaluate-") as temporary:
        work = Path(temporary)
        rollout_manifest = _read_json(rollout_uri, work / "rollout.json")
        if rollout_manifest.get("benchmark") != LIBERO_BENCHMARK:
            raise ValueError("Evaluation refuses a rollout from another benchmark")
        rollout = _download_tree(rollout_root_uri, work / "rollout")
        rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(rollout.glob(f"libero/{LIBERO_BENCHMARK}_*.json"))
        ]
        if not rows:
            raise RuntimeError(
                "No upstream LIBERO long-horizon success JSON artifacts found"
            )
        success_count = sum(float(row["succ_num"]) for row in rows)
        trial_count = sum(float(row["total_num"]) for row in rows)
        if trial_count <= 0:
            raise RuntimeError("LIBERO success artifacts reported no episodes")
        import torch

        action_tensors = sorted(rollout.glob("real/**/actions_*.pt"))
        if not action_tensors:
            raise RuntimeError("No LingBot-VA action-prediction tensors found")
        values = [
            torch.load(path, map_location="cpu", weights_only=True).float()[:, :7]
            for path in action_tensors
        ]
        actions = torch.cat([value.flatten() for value in values])
        finite = torch.isfinite(actions)
        finite_values = actions[finite]
        if not len(finite_values):
            raise RuntimeError("All predicted action values are non-finite")
        # This is a declared validity metric of emitted normalized predictions,
        # not an accuracy claim against unavailable aligned ground truth.
        prediction_validity = {
            "scope": "normalized predicted-action validity; not ground-truth action accuracy or calibrated video quality",
            "finite_fraction": float(finite.float().mean().item()),
            "within_normalized_range_fraction": float(
                ((finite_values >= -1.0) & (finite_values <= 1.0)).float().mean().item()
            ),
            "mean_absolute_normalized_action": float(finite_values.abs().mean().item()),
            "predicted_action_chunk_count": len(action_tensors),
        }
        metrics = {
            "schema": "npa.lingbot_va.evaluation.v1",
            "stage": "libero_90_long_horizon_evaluation",
            "rollout_manifest_uri": rollout_uri,
            "benchmark": LIBERO_BENCHMARK,
            "task_count": len(rows),
            "success_count": success_count,
            "trial_count": trial_count,
            "long_horizon_success_rate": success_count / trial_count,
            "action_prediction_validity": prediction_validity,
            "prediction_quality_limit": "No aligned held-out action/video target is fabricated. Add an explicitly aligned comparison artifact before making an accuracy or visual-quality claim.",
            "provenance": _provenance(),
        }
        local = work / "evaluation.json"
        _write_json(local, metrics)
        metrics["evaluation_uri"] = _upload_file(local, output_uri)
        return metrics


def visualize(
    rollout_uri: str,
    rollout_root_uri: str,
    evaluation_uri: str,
    rrd_uri: str,
    mp4_uri: str,
    output_uri: str,
) -> dict[str, Any]:
    """Emit a workflow-bound Rerun recording and MP4 from a completed rollout.

    Args:
        rollout_uri: Run-scoped URI of the native closed-loop rollout manifest.
        evaluation_uri: Run-scoped URI of the numerical evaluation manifest.
        output_uri: Run-scoped URI where visualization artifacts are written.

    Returns:
        Manifest identifying the generated Rerun recording, MP4, and provenance.

    Raises:
        RuntimeError: If the workflow did not supply its scoped run identifier.
    """
    run_id = os.environ.get("NPA_WORKFLOW_RUN_ID", "").strip()
    if not run_id:
        raise RuntimeError(
            "Visualization requires the workflow-scoped NPA_WORKFLOW_RUN_ID"
        )
    with tempfile.TemporaryDirectory(prefix="npa-lingbot-va-viz-") as temporary:
        work = Path(temporary)
        rollout_manifest = _read_json(rollout_uri, work / "rollout.json")
        metrics = _read_json(evaluation_uri, work / "evaluation.json")
        rollout = _download_tree(rollout_root_uri, work / "rollout")
        videos = sorted(rollout.glob(f"libero/{LIBERO_BENCHMARK}/**/*.mp4"))
        if not videos:
            raise RuntimeError("No real LIBERO rollout MP4 available for visualization")
        preview = work / "libero-rollout-preview.mp4"
        shutil.copy2(videos[0], preview)
        import imageio.v3 as iio
        import rerun as rr

        rrd = work / "libero-long.rrd"
        rr.init("npa.lingbot_va", recording_id=run_id, spawn=False)
        rr.save(str(rrd))
        rr.log(
            "provenance/run",
            rr.TextLog(
                json.dumps(
                    {
                        "run_id": run_id,
                        "producer": "npa.solutions.lingbot_va",
                        "provenance": _provenance(),
                        "limitation": (
                            "Observed rollout camera frames and model-emitted action "
                            "validity are recorded; this run does not claim aligned "
                            "ground-truth video or action prediction accuracy."
                        ),
                    },
                    sort_keys=True,
                )
            ),
        )
        frame_count = 0
        for frame in iio.imiter(preview, plugin="pyav"):
            rr.set_time_sequence("rollout_frame", frame_count)
            rr.log("rollout/cameras", rr.Image(frame))
            frame_count += 1
        rr.log(
            "metrics/long_horizon_success_rate",
            rr.Scalar(metrics["long_horizon_success_rate"]),
        )
        rr.log(
            "metrics/action_finite_fraction",
            rr.Scalar(metrics["action_prediction_validity"]["finite_fraction"]),
        )
        rr.disconnect()
        if not rrd.is_file() or rrd.stat().st_size == 0 or frame_count == 0:
            raise RuntimeError("Rerun recording or copied real MP4 is empty")
        manifest = {
            "schema": "npa.lingbot_va.visualization.v1",
            "stage": "synchronized_rrd_mp4_provenance",
            "rollout_manifest_uri": rollout_uri,
            "evaluation_uri": evaluation_uri,
            "rrd_uri": _upload_file(rrd, rrd_uri),
            "mp4_uri": _upload_file(preview, mp4_uri),
            "source_rollout_mp4": str(videos[0].relative_to(rollout)),
            "decoded_frame_count": frame_count,
            "rollout_benchmark": rollout_manifest["benchmark"],
            "recording_id": run_id,
            "provenance": _provenance(),
        }
        local = work / "visualization.json"
        _write_json(local, manifest)
        manifest["visualization_manifest_uri"] = _upload_file(local, output_uri)
        return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NPA LingBot-VA LIBERO-Long stages")
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("prepare")
    command.add_argument("--source-uri", required=True)
    command.add_argument("--prepared-dataset-uri", required=True)
    command.add_argument("--output-uri", required=True)
    command.add_argument("--heldout-fraction", type=float, default=0.2)
    command = subcommands.add_parser("train")
    command.add_argument("--prepared-uri", required=True)
    command.add_argument("--checkpoint-uri", required=True)
    command.add_argument("--training-uri", required=True)
    command = subcommands.add_parser("rollout")
    command.add_argument("--prepared-uri", required=True)
    command.add_argument("--checkpoint-uri", required=True)
    command.add_argument("--output-uri", required=True)
    command.add_argument("--rollout-root-uri", required=True)
    command.add_argument("--test-num", type=int, default=50)
    command.add_argument("--task-start", type=int, default=0)
    command.add_argument("--task-end", type=int, default=10)
    command = subcommands.add_parser("evaluate")
    command.add_argument("--rollout-uri", required=True)
    command.add_argument("--rollout-root-uri", required=True)
    command.add_argument("--output-uri", required=True)
    command = subcommands.add_parser("visualize")
    command.add_argument("--rollout-uri", required=True)
    command.add_argument("--rollout-root-uri", required=True)
    command.add_argument("--evaluation-uri", required=True)
    command.add_argument("--rrd-uri", required=True)
    command.add_argument("--mp4-uri", required=True)
    command.add_argument("--output-uri", required=True)
    command = subcommands.add_parser("native-train")
    command.add_argument("--dataset-dir", type=Path, required=True)
    command.add_argument("--model-dir", type=Path, required=True)
    command.add_argument("--save-root", type=Path, required=True)
    command.add_argument("--action-contract-json", type=Path, required=True)
    command = subcommands.add_parser("native-server")
    command.add_argument("--model-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--port", type=int, required=True)
    command.add_argument("--action-contract-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            args.source_uri,
            args.prepared_dataset_uri,
            args.output_uri,
            heldout_fraction=args.heldout_fraction,
        )
    elif args.command == "train":
        result = train(args.prepared_uri, args.checkpoint_uri, args.training_uri)
    elif args.command == "rollout":
        result = rollout(
            args.prepared_uri,
            args.checkpoint_uri,
            args.output_uri,
            args.rollout_root_uri,
            test_num=args.test_num,
            task_start=args.task_start,
            task_end=args.task_end,
        )
    elif args.command == "evaluate":
        result = evaluate(args.rollout_uri, args.rollout_root_uri, args.output_uri)
    elif args.command == "visualize":
        result = visualize(
            args.rollout_uri,
            args.rollout_root_uri,
            args.evaluation_uri,
            args.rrd_uri,
            args.mp4_uri,
            args.output_uri,
        )
    elif args.command == "native-train":
        _native_train(
            args.dataset_dir,
            args.model_dir,
            args.save_root,
            json.loads(args.action_contract_json.read_text(encoding="utf-8")),
        )
        return
    elif args.command == "native-server":
        _native_server(
            args.model_dir,
            args.output_dir,
            args.port,
            json.loads(args.action_contract_json.read_text(encoding="utf-8")),
        )
        return
    else:  # pragma: no cover - argparse already constrains this
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
