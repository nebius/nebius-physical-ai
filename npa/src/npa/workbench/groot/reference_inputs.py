"""Shipped public reference inputs for the GR00T N1.5 offline workflow.

This module deliberately contains mappings and immutable source identities, not
dataset bytes.  The workflow downloads the two public sources at their pinned
revisions inside the run, copies a small selected subset into a native N1.5
LeRobot v2.1 derivative, and writes the provenance, feedback, and recorded
recovery-gallery documents itself.  That makes the checked-in reference usable
without a task-private adapter archive while retaining the source's real labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


REFERENCE_PROFILE = "vifailback-wrist-hybrid-v1"
VIFAILBACK_REPOSITORY = "sii-rhos-ai/vifailback-dataset-lerobot"
VIFAILBACK_REVISION = "619791b94f61667333fe179d471051e0bca0e06f"
RECOVERY_REPOSITORY = "BodenAI/RW-RL-HIL-Dataset"
RECOVERY_REVISION = "9290f5a96dc91ea58256e81d9ce2dd9621890b51"
RECOVERY_TASK = "plug_charger_into_power_strip"
RECOVERY_EPISODE = 0
REFERENCE_EPISODES = tuple(range(20))
REFERENCE_EVAL_EPISODES = frozenset({14, 15, 16, 17, 19})
RECOVERY_VIDEO_SHA256 = (
    "47b58da6f45e2c055eec6ab8f0f6345f65022fc9679dc5c4623fb9e0cb03d879"
)
RECOVERY_PARQUET_SHA256 = (
    "a739f55e8ceb1874d0bbe560062527c012e174aa6d75b316497e69f647e7a2ca"
)
RECOVERY_FRAME_COUNT = 719
RECOVERY_FPS = 15
RECOVERY_RANGE = (454, 574)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _phase(length: int, avoid: list[int], correct: list[int]) -> list[float]:
    """Map source keyframes to an explicit, non-monotonic feedback signal."""

    if length < 1:
        raise ValueError("reference episode has no recorded frames")
    if not avoid and not correct:
        return [0.5] * length
    avoid_frame = min(avoid) if avoid else length
    correct_frame = min(correct) if correct else length
    if not (0 <= avoid_frame < length and 0 <= correct_frame < length):
        raise ValueError("reference keyframe lies outside its episode")
    if correct_frame <= avoid_frame:
        raise ValueError("reference correction must follow avoidance")
    return [
        0.5
        if frame < avoid_frame
        else min(1.0, (frame - avoid_frame) / (correct_frame - avoid_frame))
        for frame in range(length)
    ]


def _download_public_inputs(destination: Path) -> tuple[Path, Path]:
    """Fetch exactly the public files used by this pinned profile."""

    from huggingface_hub import snapshot_download

    source_patterns = [
        "README.md",
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/vifailback_annotations.jsonl",
    ]
    for episode in REFERENCE_EPISODES:
        source_patterns.extend(
            (
                f"data/chunk-000/episode_{episode:06d}.parquet",
                "videos/chunk-000/observation.images.cam_right_wrist/"
                f"episode_{episode:06d}.mp4",
            )
        )
    source = destination / "vifailback"
    snapshot_download(
        VIFAILBACK_REPOSITORY,
        repo_type="dataset",
        revision=VIFAILBACK_REVISION,
        allow_patterns=source_patterns,
        local_dir=source,
    )
    recovery = destination / "recovery"
    prefix = RECOVERY_TASK
    snapshot_download(
        RECOVERY_REPOSITORY,
        repo_type="dataset",
        revision=RECOVERY_REVISION,
        allow_patterns=[
            f"{prefix}/meta/info.json",
            f"{prefix}/meta/episodes.jsonl",
            f"{prefix}/data/chunk-000/episode_{RECOVERY_EPISODE:06d}.parquet",
            f"{prefix}/videos/chunk-000/observation.images.cam_right_wrist/"
            f"episode_{RECOVERY_EPISODE:06d}.mp4",
        ],
        local_dir=recovery,
    )
    return source, recovery / prefix


def _annotations(source: Path) -> dict[int, dict[str, Any]]:
    path = source / "meta" / "vifailback_annotations.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {int(row["episode_index"]): row for row in rows}
    if not all(episode in result for episode in REFERENCE_EPISODES):
        raise ValueError("pinned reference source lacks selected episode annotations")
    return result


def _write_recovery_gallery(source: Path, gallery: Path) -> dict[str, Any]:
    """Copy and verify one public recovery reference without assigning success."""

    import pandas as pd

    video = (
        source
        / "videos/chunk-000/observation.images.cam_right_wrist"
        / f"episode_{RECOVERY_EPISODE:06d}.mp4"
    )
    parquet = source / "data/chunk-000" / f"episode_{RECOVERY_EPISODE:06d}.parquet"
    if (
        _sha256(video) != RECOVERY_VIDEO_SHA256
        or _sha256(parquet) != RECOVERY_PARQUET_SHA256
    ):
        raise ValueError(
            "pinned recovery source bytes differ from the reference profile"
        )
    table = pd.read_parquet(parquet)
    intervention = table["intervention"].astype(bool).to_numpy()
    start, end = RECOVERY_RANGE
    if (
        len(table) != RECOVERY_FRAME_COUNT
        or not intervention[start : end + 1].all()
        or intervention[:start].any()
        or intervention[end + 1 :].any()
    ):
        raise ValueError("recovery intervention range differs from the pinned source")
    gallery.mkdir(parents=True)
    media = gallery / "recovery-right-wrist.mp4"
    shutil.copyfile(video, media)
    item = {
        "id": "rw-rl-hil-recovery-episode-0",
        "media_path": media.name,
        "media_sha256": _sha256(media),
        "source": {
            "repository": RECOVERY_REPOSITORY,
            "revision": RECOVERY_REVISION,
            "task_directory": RECOVERY_TASK,
        },
        "license": "CC-BY-4.0",
        "privacy_classification": "public",
        "source_episode_index": RECOVERY_EPISODE,
        "camera_key": "observation.images.cam_right_wrist",
        "recorded_fps": RECOVERY_FPS,
        "recorded_frame_count": RECOVERY_FRAME_COUNT,
        "frame_start": start,
        "frame_end": end,
        "outcome": "unknown",
        "collection_kind": "operator_recovery",
        "tags": ["human_intervention", "operator_recovery"],
        "label_source": "upstream per-frame intervention/control_source fields",
        "action_predictions_available": False,
        "limitation": (
            "Human intervention documents recovery control authority, not an observed "
            "task-success outcome. It is a gallery reference only and is not mixed "
            "into the ViFailback training schema."
        ),
    }
    _write_json(
        gallery / "gallery.json",
        {"schema_version": "npa.groot-recorded-gallery-reference/v1", "items": [item]},
    )
    return item


@dataclass(frozen=True)
class PreparedReference:
    """Run-local materials returned by :func:`prepare_reference` ."""

    dataset_root: Path
    feedback_path: Path
    gallery_root: Path
    provenance: dict[str, Any]
    action_contract: dict[str, Any]
    gallery_item: dict[str, Any]


def _reference_video_feature(source_info: Mapping[str, Any]) -> dict[str, Any]:
    """Return the native-reader video schema from the pinned source metadata.

    Args:
        source_info: Parsed ``meta/info.json`` from the pinned public source.

    Returns:
        A detached camera feature schema with named height and width axes.

    Raises:
        ValueError: The source lacks a native-compatible wrist-camera schema.
    """
    features = source_info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("reference source metadata has no feature mapping")
    feature = features.get("observation.images.cam_right_wrist")
    if not isinstance(feature, Mapping) or feature.get("dtype") != "video":
        raise ValueError("reference source wrist camera is not a video feature")
    shape = feature.get("shape")
    names = feature.get("names")
    if (
        not isinstance(shape, list)
        or not isinstance(names, list)
        or len(shape) != len(names)
        or not {"height", "width"}.issubset(names)
    ):
        raise ValueError(
            "reference source wrist video feature needs named height and width axes"
        )
    return dict(feature)


def prepare_reference(stage: Path) -> PreparedReference:
    """Create the checked-in reference derivative and its source-bound labels."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    source, recovery = _download_public_inputs(stage / "public-source")
    annotations = _annotations(source)
    output = stage / "dataset"
    (output / "meta").mkdir(parents=True)
    (output / "data/chunk-000").mkdir(parents=True)
    video_dir = output / "videos/chunk-000/observation.images.cam_right_wrist"
    video_dir.mkdir(parents=True)
    episodes: list[dict[str, int]] = []
    feedback: list[dict[str, Any]] = []
    index_offset = 0
    for output_episode, source_episode in enumerate(REFERENCE_EPISODES):
        path = source / "data/chunk-000" / f"episode_{source_episode:06d}.parquet"
        table = pq.read_table(path)
        length = len(table)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        flags = np.asarray(
            table["complementary_info.vifailback_flags"].to_pylist(), dtype=bool
        )
        if state.shape != (length, 14) or action.shape != (length, 14):
            raise ValueError("reference source state/action dimensions changed")
        if flags.shape != (length, 3):
            raise ValueError("reference source flags changed")
        annotation = annotations[source_episode]
        success = annotation.get("episode_success") == "success"
        if success == bool(flags[:, 0].any()):
            raise ValueError("reference outcome conflicts with its frame flags")
        task_index = table["task_index"].to_pylist()
        timestamps = table["timestamp"].to_pylist()
        converted = pa.table(
            {
                "observation.state": state.tolist(),
                "action": action.tolist(),
                "signal.corrective_phase": pa.array(
                    _phase(
                        length,
                        [
                            int(value)
                            for value in annotation.get("avoid_keyframe_indices", [])
                        ],
                        [
                            int(value)
                            for value in annotation.get("correct_keyframe_indices", [])
                        ],
                    ),
                    type=pa.float32(),
                ),
                "feedback.failure_flag": pa.array(
                    flags[:, 0].tolist(), type=pa.bool_()
                ),
                "feedback.avoid_keyframe": pa.array(
                    flags[:, 1].tolist(), type=pa.bool_()
                ),
                "feedback.correct_keyframe": pa.array(
                    flags[:, 2].tolist(), type=pa.bool_()
                ),
                "source.episode_index": pa.array(
                    [source_episode] * length, type=pa.int64()
                ),
                "source.timestamp": pa.array(timestamps, type=pa.float32()),
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "frame_index": pa.array(list(range(length)), type=pa.int64()),
                "episode_index": pa.array([output_episode] * length, type=pa.int64()),
                "index": pa.array(
                    list(range(index_offset, index_offset + length)), type=pa.int64()
                ),
                "task_index": pa.array(task_index, type=pa.int64()),
            }
        )
        pq.write_table(
            converted,
            output / "data/chunk-000" / f"episode_{output_episode:06d}.parquet",
        )
        shutil.copyfile(
            source
            / "videos/chunk-000/observation.images.cam_right_wrist"
            / f"episode_{source_episode:06d}.mp4",
            video_dir / f"episode_{output_episode:06d}.mp4",
        )
        split = "eval" if source_episode in REFERENCE_EVAL_EPISODES else "train"
        feedback.append(
            {
                "episode_index": output_episode,
                "split": split,
                "success": success,
                "group": f"vifailback-source-episode-{source_episode:06d}",
                "collection_kind": "policy_rollout",
                "tags": ["human_corrective_guidance"] if flags[:, 2].any() else [],
                "source_episode_id": f"vifailback:{source_episode}",
                "label_source": "upstream ViFailback episode annotation",
                "progress": (
                    _phase(
                        length,
                        [
                            int(value)
                            for value in annotation.get("avoid_keyframe_indices", [])
                        ],
                        [
                            int(value)
                            for value in annotation.get("correct_keyframe_indices", [])
                        ],
                    )
                    if not success
                    else None
                ),
                "progress_source": "annotation_derived" if not success else None,
            }
        )
        episodes.append({"episode_index": output_episode, "length": length})
        index_offset += length
    source_info = json.loads((source / "meta/info.json").read_text())
    video_feature = _reference_video_feature(source_info)
    source_info.update(
        {
            "codebase_version": "v2.1",
            "robot_type": "aloha",
            "total_episodes": len(episodes),
            "total_frames": index_offset,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {"dtype": "float32", "shape": [14]},
                "action": {"dtype": "float32", "shape": [14]},
                "observation.images.cam_right_wrist": video_feature,
            },
        }
    )
    _write_json(output / "meta/info.json", source_info)
    (output / "meta/episodes.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in episodes) + "\n"
    )
    shutil.copyfile(source / "meta/tasks.jsonl", output / "meta/tasks.jsonl")
    _write_json(
        output / "meta/modality.json",
        {
            "state": {
                "left_arm": {
                    "start": 0,
                    "end": 7,
                    "dtype": "float32",
                    "original_key": "observation.state",
                },
                "right_arm": {
                    "start": 7,
                    "end": 14,
                    "dtype": "float32",
                    "original_key": "observation.state",
                },
            },
            "action": {
                "left_arm": {
                    "start": 0,
                    "end": 7,
                    "dtype": "float32",
                    "original_key": "action",
                },
                "right_arm": {
                    "start": 7,
                    "end": 14,
                    "dtype": "float32",
                    "original_key": "action",
                },
            },
            "video": {"wrist": {"original_key": "observation.images.cam_right_wrist"}},
            "annotation": {"human.task_description": {"original_key": "task_index"}},
        },
    )
    feedback_path = stage / "feedback.json"
    _write_json(
        feedback_path,
        {
            "schema_version": "npa.groot-feedback/v2",
            "label_protocol": "upstream per-episode success/failure annotation; source corrective/avoidance event phase for failed policy rollouts",
            "label_provenance": "pinned ViFailback source annotations; phase is annotation-derived and not a measured robot-success metric",
            "privacy_classification": "public",
            "progress_contract": {
                "units": "annotation-derived phase in [0,1]",
                "direction": "0.5 before avoidance, linearly recovers from 0.0 at avoidance to 1.0 at the source correction keyframe; regressions are retained",
                "alignment": "one value per converted recorded frame",
            },
            "episodes": feedback,
        },
    )
    gallery = stage / "recorded-gallery-reference"
    item = _write_recovery_gallery(recovery, gallery)
    provenance = {
        "schema_version": "npa.groot-input-provenance/v1",
        "source_kind": "huggingface",
        "source_identifier": VIFAILBACK_REPOSITORY,
        "source_revision": VIFAILBACK_REVISION,
        "license": "MIT",
        "privacy_classification": "public",
        "redistributable_media": True,
        "label_provenance": "upstream ViFailback per-episode outcome and annotated avoidance/correction keyframes",
        "reference_profile": REFERENCE_PROFILE,
        "recovery_gallery_source": {
            "repository": RECOVERY_REPOSITORY,
            "revision": RECOVERY_REVISION,
            "license": "CC-BY-4.0",
            "outcome": "unknown",
        },
    }
    action_contract = {
        "schema_version": "npa.groot-action-contract/v1",
        "kind": "joint_position",
        "target_keys": ["action.left_arm", "action.right_arm"],
        "frame": "ALOHA joint coordinate order; source-defined",
        "translation_unit": "not-applicable",
        "rotation_representation": "not-applicable",
        "rotation_unit": "not-applicable",
        "gripper_fields": ["left_arm[6]", "right_arm[6]"],
        "transform_identity": "native-min-max-training-split-only-v1",
    }
    return PreparedReference(
        output, feedback_path, gallery, provenance, action_contract, item
    )


class SingleWristAlohaConfig:
    """A shipped N1.5 profile for the public one-wrist ALOHA reference."""

    video_keys = ["video.wrist"]
    state_keys = ["state.left_arm", "state.right_arm"]
    action_keys = ["action.left_arm", "action.right_arm"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        from gr00t.data.dataset import ModalityConfig

        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices, modality_keys=self.video_keys
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices, modality_keys=self.state_keys
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices, modality_keys=self.action_keys
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices, modality_keys=self.language_keys
            ),
        }

    def transform(self):
        from gr00t.data.transform.base import ComposedModalityTransform
        from gr00t.data.transform.concat import ConcatTransform
        from gr00t.data.transform.state_action import (
            StateActionToTensor,
            StateActionTransform,
        )
        from gr00t.data.transform.video import VideoResize, VideoToNumpy, VideoToTensor
        from gr00t.model.transforms import GR00TTransform

        return ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=self.video_keys),
                VideoResize(
                    apply_to=self.video_keys,
                    height=224,
                    width=224,
                    interpolation="linear",
                ),
                VideoToNumpy(apply_to=self.video_keys),
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "min_max" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "min_max" for key in self.action_keys},
                ),
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
                GR00TTransform(
                    state_horizon=len(self.observation_indices),
                    action_horizon=len(self.action_indices),
                    max_state_dim=64,
                    max_action_dim=32,
                ),
            ]
        )
