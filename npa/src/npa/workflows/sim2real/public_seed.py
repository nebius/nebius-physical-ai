"""Runtime staging for the public Franka-lift Sim2Real seed preset.

Dataset bytes are fetched from the immutable public Hugging Face revision only
when an operator explicitly stages the preset. Nothing is vendored or baked.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

import av
import pyarrow.parquet as pq

from npa.clients.storage import StorageClient
from npa.orchestration.npa_workflow.presets import (
    PUBLIC_FRANKA_LIFT,
    PUBLIC_FRANKA_LIFT_CANONICAL_TASK_ID,
    PUBLIC_FRANKA_LIFT_DATASET_ID,
    PUBLIC_FRANKA_LIFT_DATASET_REPOSITORY,
    PUBLIC_FRANKA_LIFT_DATASET_REVISION,
    PUBLIC_FRANKA_LIFT_SOURCE_TASK_ID,
)
from npa.workflows.sim2real.task_contract import (
    SEED_DATASET_SCHEMA,
    build_task_contract,
)


HF_BASE = "https://huggingface.co"
DECLARED_LICENSE = "apache-2.0"
SOURCE_PATHS = {
    "metadata": "final_dataset/meta/info.json",
    "tasks": "final_dataset/meta/tasks.jsonl",
    "episodes": "final_dataset/meta/episodes.jsonl",
    "actions": "final_dataset/data/chunk-000/episode_000000.parquet",
    "main_video": "final_dataset/videos/chunk-000/image/episode_000000.mp4",
    "wrist_video": (
        "final_dataset/videos/chunk-000/wrist_image/episode_000000.mp4"
    ),
}


class PublicSeedError(RuntimeError):
    """The public seed could not be verified, decoded, or staged truthfully."""


Fetch = Callable[[str], bytes]


def _fetch(url: str) -> bytes:
    try:
        with urlopen(url, timeout=60) as response:  # noqa: S310 - pinned public host
            payload = response.read()
    except Exception as exc:  # noqa: BLE001 - normalize network failures
        raise PublicSeedError(f"public seed fetch failed for {url}: {exc}") from exc
    if not payload:
        raise PublicSeedError(f"public seed fetch returned no bytes for {url}")
    return payload


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicSeedError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise PublicSeedError(f"{label} must be a JSON object")
    return parsed


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_url(path: str) -> str:
    encoded = "/".join(quote(part, safe="") for part in path.split("/"))
    return (
        f"{HF_BASE}/datasets/{PUBLIC_FRANKA_LIFT_DATASET_REPOSITORY}/resolve/"
        f"{PUBLIC_FRANKA_LIFT_DATASET_REVISION}/{encoded}?download=true"
    )


def _verify_repository(fetch: Fetch) -> dict[str, Any]:
    api_url = (
        f"{HF_BASE}/api/datasets/{PUBLIC_FRANKA_LIFT_DATASET_REPOSITORY}/revision/"
        f"{PUBLIC_FRANKA_LIFT_DATASET_REVISION}"
    )
    metadata = _json_object(fetch(api_url), label="Hugging Face repository metadata")
    license_value = str((metadata.get("cardData") or {}).get("license") or "").lower()
    if metadata.get("sha") != PUBLIC_FRANKA_LIFT_DATASET_REVISION:
        raise PublicSeedError("public seed revision drifted from its immutable pin")
    if metadata.get("private") is not False or metadata.get("gated") not in {False, None}:
        raise PublicSeedError("public seed is no longer anonymously accessible")
    if license_value != DECLARED_LICENSE:
        raise PublicSeedError(
            f"public seed declared license mismatch: expected {DECLARED_LICENSE}, "
            f"got {license_value or 'missing'}"
        )
    siblings = {
        str(item.get("rfilename") or "") for item in metadata.get("siblings") or []
    }
    missing = sorted(set(SOURCE_PATHS.values()) - siblings)
    if missing:
        raise PublicSeedError(f"public seed revision is missing required paths: {missing}")
    return metadata


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _decode_frames(
    video_path: Path, *, frame_count: int, output_dir: Path
) -> list[Path]:
    decoded: list[Path] = []
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if index >= frame_count:
                    break
                # Stage 3 deliberately consumes the canonical Isaac frame naming
                # contract (camera-<integer>.png).  Keep source-camera identity in
                # the manifest rather than encoding it into the filename, because
                # names such as camera-image-000.png are not valid transfer input.
                target = output_dir / f"camera-{index:03d}.png"
                frame.to_image().save(target, format="PNG")
                if target.stat().st_size <= 0:
                    raise PublicSeedError(f"decoded empty frame from {video_path.name}")
                decoded.append(target)
    except PublicSeedError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize decoder failures
        raise PublicSeedError(f"could not decode {video_path.name}: {exc}") from exc
    if len(decoded) != frame_count:
        raise PublicSeedError(
            f"{video_path.name} yielded {len(decoded)} frames; "
            f"expected at least {frame_count}"
        )
    return decoded


def _read_actions(parquet_path: Path) -> list[list[float]]:
    try:
        table = pq.read_table(parquet_path, columns=["actions"])
        values = table.column("actions").to_pylist()
    except Exception as exc:  # noqa: BLE001 - normalize parquet/schema failures
        raise PublicSeedError(f"could not read source actions: {exc}") from exc
    actions: list[list[float]] = []
    for index, value in enumerate(values):
        row = list(value or [])
        if len(row) != 7:
            raise PublicSeedError(
                f"source action row {index} has {len(row)} dimensions; expected 7"
            )
        actions.append([float(item) for item in row])
    if not actions:
        raise PublicSeedError("source episode contains no actions")
    return actions


def _upload(
    client: StorageClient, *, local: Path, uri: str, source_path: str = ""
) -> dict[str, Any]:
    payload = local.read_bytes()
    try:
        uploaded = client.upload_file(str(local), uri)
    except Exception as exc:  # noqa: BLE001 - upload is a fail-closed boundary
        raise PublicSeedError(f"upload failed for {uri}: {exc}") from exc
    if uploaded != uri:
        raise PublicSeedError(f"upload destination mismatch: expected {uri}, got {uploaded}")
    result: dict[str, Any] = {
        "uri": uploaded,
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }
    if source_path:
        result["source_path"] = source_path
    return result


def stage_public_franka_lift(
    *,
    bucket: str,
    run_id: str,
    client: StorageClient,
    fetch: Fetch = _fetch,
) -> dict[str, Any]:
    """Stage one verified episode and truthful Stage-1 evidence to run-scoped S3."""

    clean_bucket = bucket.strip()
    clean_run = run_id.strip()
    if not clean_bucket or "/" in clean_bucket or clean_bucket == "example-bucket":
        raise PublicSeedError("a real S3 bucket name is required")
    if not clean_run or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in clean_run):
        raise PublicSeedError("run_id must contain only letters, digits, '-' and '_'")

    repository_metadata = _verify_repository(fetch)
    prefix = f"s3://{clean_bucket}/sim2real-triggers/{clean_run}/{PUBLIC_FRANKA_LIFT}/"
    with tempfile.TemporaryDirectory(prefix="npa-public-franka-lift-") as temp:
        work = Path(temp)
        source_files: dict[str, Path] = {}
        for label, source_path in SOURCE_PATHS.items():
            payload = fetch(_source_url(source_path))
            target = work / "source" / source_path
            _write(target, payload)
            source_files[label] = target

        info = _json_object(source_files["metadata"].read_bytes(), label="info.json")
        source_meta = info.get("source_rollout_meta") or {}
        features = info.get("features") or {}
        if (
            info.get("robot_type") != "franka"
            or source_meta.get("task") != PUBLIC_FRANKA_LIFT_SOURCE_TASK_ID
            or (features.get("actions") or {}).get("shape") != [7]
            or not all((features.get(name) or {}).get("dtype") == "video" for name in ("image", "wrist_image"))
        ):
            raise PublicSeedError("public seed metadata no longer matches the pinned Franka/7D/two-camera contract")
        task_rows = [json.loads(line) for line in source_files["tasks"].read_text().splitlines() if line.strip()]
        episode_rows = [json.loads(line) for line in source_files["episodes"].read_text().splitlines() if line.strip()]
        if task_rows != [{"task_index": 0, "task": "lift the cube"}]:
            raise PublicSeedError("public seed task metadata is missing or changed")
        episode = next((row for row in episode_rows if row.get("episode_index") == 0), None)
        if not episode or int(episode.get("length") or 0) <= 0:
            raise PublicSeedError("public seed episode metadata is missing episode 0")
        selected_episodes = [episode]

        actions = _read_actions(source_files["actions"])
        if len(actions) != int(episode["length"]):
            raise PublicSeedError("source action rows do not match episode metadata length")
        actions_path = work / "actions.json"
        actions_payload = {
            "schema": "npa.sim2real.seed_actions.v1",
            "dimensions": 7,
            "representation": "IK-relative (6D end-effector delta plus gripper)",
            "source_only": True,
            "count": len(actions),
            "actions": actions,
        }
        _write(actions_path, (json.dumps(actions_payload, sort_keys=True) + "\n").encode())

        frames_dir = work / "frames"
        frames_dir.mkdir()
        main_frames = _decode_frames(
            source_files["main_video"], frame_count=4, output_dir=frames_dir
        )
        frame_sources = [
            ("image", index, frame) for index, frame in enumerate(main_frames)
        ]
        frames = [frame for _camera, _index, frame in frame_sources]
        if len(frames) < 4:
            raise PublicSeedError("public seed decode produced fewer than four camera frames")

        staged: list[dict[str, Any]] = []
        for label, local in source_files.items():
            source_path = SOURCE_PATHS[label]
            staged.append(
                _upload(
                    client,
                    local=local,
                    uri=prefix + "source/" + source_path,
                    source_path=source_path,
                )
            )
        actions_evidence = _upload(
            client, local=actions_path, uri=prefix + "evidence/actions.json"
        )
        staged.append(actions_evidence)
        frame_evidence = []
        for source_camera, source_frame_index, frame in frame_sources:
            record = _upload(
                client, local=frame, uri=prefix + "frames/" + frame.name
            )
            record.update(
                source_camera=source_camera,
                source_frame_index=source_frame_index,
            )
            staged.append(record)
            frame_evidence.append(record)

        contract = build_task_contract(
            task_id=PUBLIC_FRANKA_LIFT_CANONICAL_TASK_ID,
            dataset_id=PUBLIC_FRANKA_LIFT_DATASET_ID,
            dataset_uri=prefix,
        )
        sample_uri = prefix + "sample-rollout-manifest.json"
        sample = {
            "schema": "npa.sim2real.sample_seed_rollout.v1",
            "episode_index": int(episode["episode_index"]),
            "source_task_id": PUBLIC_FRANKA_LIFT_SOURCE_TASK_ID,
            "action_evidence": actions_evidence,
            "camera_evidence": frame_evidence,
            "action_count": len(actions),
            "camera_observation_count": len(frames),
        }
        sample_path = work / "sample-rollout-manifest.json"
        _write(sample_path, (json.dumps(sample, indent=2, sort_keys=True) + "\n").encode())
        sample_record = _upload(client, local=sample_path, uri=sample_uri)
        staged.append(sample_record)

        manifest_uri = prefix + "dataset-manifest.json"
        manifest = {
            "schema": SEED_DATASET_SCHEMA,
            "preset": PUBLIC_FRANKA_LIFT,
            "task_id": PUBLIC_FRANKA_LIFT_CANONICAL_TASK_ID,
            "dataset_id": PUBLIC_FRANKA_LIFT_DATASET_ID,
            "task_contract_digest": contract["task_contract_digest"],
            "source_backend": "isaac",
            "source_run_id": f"hf-episode-{int(episode['episode_index']):06d}",
            "relabeled_from_another_task": False,
            "trajectory_count": len(selected_episodes),
            "action_count": len(actions),
            "camera_observation_count": len(frames),
            "sample_rollout_manifest_uri": sample_uri,
            "source_provenance": {
                "provider": "huggingface",
                "repository": PUBLIC_FRANKA_LIFT_DATASET_REPOSITORY,
                "revision": PUBLIC_FRANKA_LIFT_DATASET_REVISION,
                "declared_license": DECLARED_LICENSE,
                "anonymous_public": repository_metadata.get("private") is False
                and repository_metadata.get("gated") in {False, None},
                "source_paths": [SOURCE_PATHS[key] for key in SOURCE_PATHS],
                "objects": staged,
            },
            "source_contract": {
                "task_id": PUBLIC_FRANKA_LIFT_SOURCE_TASK_ID,
                "robot": "franka",
                "action": {"dimensions": 7, "representation": "IK-relative"},
                "cameras": ["image", "wrist_image"],
            },
            "compatibility_boundary": {
                "classification": "task-family seed/conditioning evidence",
                "source_actions_reused_as_canonical_ppo_actions": False,
                "canonical_action": {
                    "dimensions": 8,
                    "representation": "joint position delta plus binary gripper",
                },
                "canonical_cameras": ["primary", "side", "overhead"],
                "strict_success_distance_m": 0.05,
                "placement_stability_required": True,
            },
        }
        manifest_path = work / "dataset-manifest.json"
        _write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        manifest_record = _upload(client, local=manifest_path, uri=manifest_uri)

    return {
        "status": "staged",
        "preset": PUBLIC_FRANKA_LIFT,
        "dataset_id": PUBLIC_FRANKA_LIFT_DATASET_ID,
        "task_id": PUBLIC_FRANKA_LIFT_CANONICAL_TASK_ID,
        "trigger_uri": prefix,
        "seed_manifest_uri": manifest_uri,
        "manifest": manifest_record,
        "trajectory_count": manifest["trajectory_count"],
        "action_count": manifest["action_count"],
        "camera_observation_count": manifest["camera_observation_count"],
    }
