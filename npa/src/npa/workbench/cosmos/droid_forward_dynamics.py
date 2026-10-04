"""DROID action-conditioned Cosmos3 forward-dynamics workflow stages.

The public ``jere-mybao/cosmos3-nano-droid-forward-dynamics`` checkpoint is a
post-train of Cosmos3-Nano, not a generic video model.  These stages preserve
the checkpoint's 17-frame/15-fps, three-camera, 16-action DROID contract and
make an action-sensitivity control a first-class artifact.  They deliberately
do not turn the card's published action probe into a benchmark claim.

All stage handoffs use the existing content-hashed policy artifact transport.
Weights and DROID inputs are runtime materialized only; neither is copied into
an NPA image or repository checkout.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from npa.workbench.cosmos.policy_artifacts import (
    file_digest,
    materialize_bundle,
    policy_workspace,
    publish_bundle,
    write_local_json,
)
from npa.workbench.dataset.storage import read_bytes_uri, read_json_uri

SELECTION_SCHEMA = "npa.cosmos3.droid-fd.selection.v1"
PREPARED_SCHEMA = "npa.cosmos3.droid-fd.prepared.v1"
PREDICTION_SCHEMA = "npa.cosmos3.droid-fd.prediction.v1"
CONTROLS_SCHEMA = "npa.cosmos3.droid-fd.controls.v1"
EVALUATION_SCHEMA = "npa.cosmos3.droid-fd.evaluation.v1"
VISUALIZATION_SCHEMA = "npa.cosmos3.droid-fd.visualization.v1"

CHECKPOINT_REPOSITORY = "jere-mybao/cosmos3-nano-droid-forward-dynamics"
CHECKPOINT_REVISION = "1dfff3cc3b86548b208341bb123d1c4f71043114"
CARD_FRAMEWORK_REVISION = "9cbd0841b50a1e667577292be1a4ad79cbc8e3d9"
NATIVE_FRAMEWORK_REPOSITORY = "https://github.com/NVIDIA/cosmos-framework.git"
# The original export revision above has been pruned from the public remote.  This
# is the independently available OpenMDW-1.1 pin used by npa-cosmos3; it retains
# the native forward-dynamics input schema and DROID domain mapping.  The workflow
# records both identities and never describes this as a bit-identical rebuild.
NATIVE_FRAMEWORK_REVISION = "5e67049cd94acb667786f1e6dd0dab821cb90c97"
NATIVE_FRAMEWORK_LICENSE = "OpenMDW-1.1"
NATIVE_SOURCE_MARKER = ".npa_source_revision"
COSMOS3_DROID_REPOSITORY = "nvidia/Cosmos3-DROID"
COSMOS3_DROID_DATASET_VERSION = "droid_plus_lerobot_640x360_20260412"
OPENMDW_LICENSE_URL = "https://openmdw.ai/license/1-1/"
CARD_URL = f"https://huggingface.co/{CHECKPOINT_REPOSITORY}"
FRAMEWORK_URL = "https://github.com/NVIDIA/cosmos-framework"
DROID_CITATION_URL = "https://arxiv.org/abs/2403.12945"

FRAME_COUNT = 17
PREDICTION_FRAME_COUNT = 16
FPS = 15
RAW_ACTION_DIM = 10
PADDED_ACTION_DIM = 64
DOMAIN_NAME = "droid_lerobot"
DOMAIN_ID = 8
COMPOSITE_WIDTH = 640
COMPOSITE_HEIGHT = 540
SOURCE_VIEW_WIDTH = 640
SOURCE_VIEW_HEIGHT = 360


class DroidForwardDynamicsError(RuntimeError):
    """Raised when a DROID forward-dynamics artifact is incomplete or invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _expect_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DroidForwardDynamicsError(f"{name} must be an object")
    return value


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise DroidForwardDynamicsError(f"{name} must contain exactly {length} values")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise DroidForwardDynamicsError(f"{name} must be numeric") from exc
    if not all(math.isfinite(item) for item in result):
        raise DroidForwardDynamicsError(f"{name} must contain only finite values")
    return result


def _matrix3(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3:
        raise DroidForwardDynamicsError(f"{name} must be a 3x3 rotation matrix")
    rows = [
        _finite_vector(row, 3, f"{name}[{index}]") for index, row in enumerate(value)
    ]
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=1e-4
    ):
        raise DroidForwardDynamicsError(f"{name} must be a proper rotation matrix")
    return matrix


def _card_holdout_metadata(selection: dict[str, Any]) -> dict[str, Any]:
    """Validate the documented split or a stronger scene/building separation."""

    split = _expect_mapping(selection.get("heldout_split"), "heldout_split")
    method = str(split.get("method") or "")
    group_key = str(split.get("group_key") or "")
    if method == "scene_or_building" and group_key in {"scene_id", "building_id"}:
        group_id = str(split.get("group_id") or "")
        if not group_id:
            raise DroidForwardDynamicsError(
                "scene/building held-out selection needs group_id"
            )
        return {
            "scope": "scene_or_building_heldout",
            "method": method,
            "group_key": group_key,
            "group_id": group_id,
            "note": "Stronger than the checkpoint card's episode-level split.",
        }
    if method != "card_episode_3pct":
        raise DroidForwardDynamicsError(
            "heldout_split must use scene_or_building with scene_id/building_id, "
            "or the documented card_episode_3pct method"
        )
    if int(split.get("seed", -1)) != 42 or float(split.get("val_ratio", -1)) != 0.03:
        raise DroidForwardDynamicsError(
            "card_episode_3pct must retain the card's seed=42 and val_ratio=0.03"
        )
    subset = str(split.get("subset") or "")
    expected_total = {"success": 57639, "failure": 14268}.get(subset)
    if (
        expected_total is None
        or int(split.get("subset_episode_count", -1)) != expected_total
    ):
        raise DroidForwardDynamicsError(
            "card_episode_3pct must name success/57639 or failure/14268 exactly"
        )
    episode_index = split.get("episode_index")
    if not isinstance(episode_index, int) or not 0 <= episode_index < expected_total:
        raise DroidForwardDynamicsError(
            "card_episode_3pct needs an in-range episode_index"
        )
    try:
        import torch
    except ImportError as exc:
        raise DroidForwardDynamicsError(
            "card_episode_3pct verification requires PyTorch to reproduce the card's split"
        ) from exc
    generator = torch.Generator().manual_seed(42)
    held_out = torch.randperm(expected_total, generator=generator).tolist()[
        : int(round(expected_total * 0.03))
    ]
    if episode_index not in set(held_out):
        raise DroidForwardDynamicsError(
            "episode_index is not in the documented seed=42 3% held-out split"
        )
    return {
        "scope": "episode_heldout",
        "method": method,
        "seed": 42,
        "val_ratio": 0.03,
        "subset": subset,
        "subset_episode_count": expected_total,
        "episode_index": episode_index,
        "heldout_episode_count": len(held_out),
        "membership_reproduced_with": "torch.randperm(seed=42)[:round(total*0.03)]",
        "note": (
            "The card's 3% split is held out by episode only; it can retain scene "
            "correlation and is not a scene-separated generalization result."
        ),
    }


def _read_hashed_image(value: Any, label: str) -> Image.Image:
    entry = _expect_mapping(value, label)
    uri = str(entry.get("uri") or "")
    expected_digest = str(entry.get("sha256") or "")
    expected_bytes = entry.get("bytes")
    if not uri or len(expected_digest) != 64 or not isinstance(expected_bytes, int):
        raise DroidForwardDynamicsError(f"{label} must declare uri, bytes, and sha256")
    payload = read_bytes_uri(uri)
    if len(payload) != expected_bytes or _sha256_bytes(payload) != expected_digest:
        raise DroidForwardDynamicsError(
            f"{label} content does not match its source manifest"
        )
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:  # Pillow reports a family of decoder exceptions.
        raise DroidForwardDynamicsError(
            f"{label} is not a decodable RGB image"
        ) from exc
    if image.size != (SOURCE_VIEW_WIDTH, SOURCE_VIEW_HEIGHT):
        raise DroidForwardDynamicsError(
            f"{label} must be {SOURCE_VIEW_WIDTH}x{SOURCE_VIEW_HEIGHT}; got {image.size}"
        )
    return image


def _composite_frame(frame: dict[str, Any], frame_index: int) -> Image.Image:
    views = _expect_mapping(frame.get("views"), f"frames[{frame_index}].views")
    wrist = _read_hashed_image(
        views.get("wrist_image_left"), f"frame {frame_index} wrist"
    )
    exterior_one = _read_hashed_image(
        views.get("exterior_image_1_left"), f"frame {frame_index} exterior 1"
    )
    exterior_two = _read_hashed_image(
        views.get("exterior_image_2_left"), f"frame {frame_index} exterior 2"
    )
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    lower_size = (SOURCE_VIEW_WIDTH // 2, SOURCE_VIEW_HEIGHT // 2)
    canvas = Image.new("RGB", (COMPOSITE_WIDTH, COMPOSITE_HEIGHT))
    canvas.paste(wrist, (0, 0))
    canvas.paste(exterior_one.resize(lower_size, resampling), (0, SOURCE_VIEW_HEIGHT))
    canvas.paste(
        exterior_two.resize(lower_size, resampling),
        (SOURCE_VIEW_WIDTH // 2, SOURCE_VIEW_HEIGHT),
    )
    return canvas


def _write_video(frames: Iterable[Image.Image], destination: Path) -> None:
    try:
        import av
    except ImportError as exc:
        raise DroidForwardDynamicsError(
            "PyAV is required to encode DROID evidence video"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame_list = list(frames)
    if not frame_list:
        raise DroidForwardDynamicsError("cannot encode an empty video")
    try:
        with av.open(str(destination), mode="w") as container:
            stream = container.add_stream("libx264", rate=FPS)
            stream.width = COMPOSITE_WIDTH
            stream.height = COMPOSITE_HEIGHT
            stream.pix_fmt = "yuv420p"
            for image in frame_list:
                for packet in stream.encode(av.VideoFrame.from_image(image)):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as exc:
        raise DroidForwardDynamicsError(
            "could not encode the synchronized evidence MP4"
        ) from exc


def _decode_video(path: Path) -> list[np.ndarray]:
    try:
        import av
    except ImportError as exc:
        raise DroidForwardDynamicsError(
            "PyAV is required to evaluate generated video"
        ) from exc
    try:
        with av.open(str(path), mode="r") as container:
            frames = [
                frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
            ]
    except Exception as exc:
        raise DroidForwardDynamicsError(f"could not decode video {path.name}") from exc
    if not frames:
        raise DroidForwardDynamicsError(f"decoded video {path.name} contains no frames")
    if any(frame.shape != frames[0].shape for frame in frames):
        raise DroidForwardDynamicsError(
            f"decoded video {path.name} has inconsistent frame geometry"
        )
    return frames


def _relative_actions(
    poses: list[dict[str, Any]], raw_gripper_actions: list[float]
) -> list[list[float]]:
    if len(poses) != FRAME_COUNT:
        raise DroidForwardDynamicsError(
            f"poses_abs must contain exactly {FRAME_COUNT} entries"
        )
    if len(raw_gripper_actions) != PREDICTION_FRAME_COUNT:
        raise DroidForwardDynamicsError(
            "gripper_actions_raw must contain exactly 16 source action values"
        )
    if any(not 0.0 <= value <= 1.0 for value in raw_gripper_actions):
        raise DroidForwardDynamicsError(
            "gripper_actions_raw values must be within [0, 1]"
        )
    decoded: list[tuple[np.ndarray, np.ndarray]] = []
    for index, raw_pose in enumerate(poses):
        pose = _expect_mapping(raw_pose, f"poses_abs[{index}]")
        position = np.asarray(
            _finite_vector(pose.get("position_m"), 3, f"pose {index} position")
        )
        rotation = _matrix3(pose.get("rotation_matrix"), f"pose {index} rotation")
        decoded.append((position, rotation))
    actions: list[list[float]] = []
    for index, ((position, rotation), (next_position, next_rotation)) in enumerate(
        zip(decoded, decoded[1:])
    ):
        translation = next_position - position
        relative_rotation = rotation.T @ next_rotation
        # Cosmos' rot6d convention stores the first two *columns*, column-major.
        rot6d = relative_rotation[:, :2].reshape(-1, order="F")
        # The DROID post-train reads action.gripper_position, not the gripper
        # state embedded in an observation/pose record.  It flips that source
        # field for this exact dataset version before concatenating it with the
        # relative end-effector pose action.
        raw = [*translation.tolist(), *rot6d.tolist(), 1.0 - raw_gripper_actions[index]]
        if len(raw) != RAW_ACTION_DIM or not all(math.isfinite(value) for value in raw):
            raise DroidForwardDynamicsError(
                f"action {index} does not satisfy the raw 10-D contract"
            )
        actions.append([*raw, *([0.0] * (PADDED_ACTION_DIM - RAW_ACTION_DIM))])
    return actions


def _action_payload(
    actions: list[list[float]], *, control: str = "true"
) -> dict[str, Any]:
    if len(actions) != PREDICTION_FRAME_COUNT or any(
        len(action) != PADDED_ACTION_DIM for action in actions
    ):
        raise DroidForwardDynamicsError("action payload must be [16, 64]")
    return {
        "schema": "npa.cosmos3.droid-fd.actions.v1",
        "action_chunk_size": PREDICTION_FRAME_COUNT,
        "action_dim": PADDED_ACTION_DIM,
        "raw_action_dim": RAW_ACTION_DIM,
        "domain_name": DOMAIN_NAME,
        "domain_id": DOMAIN_ID,
        "control": control,
        "actions": actions,
    }


def _native_raw_actions(actions: list[list[float]]) -> list[list[float]]:
    """Convert the card's padded audit tensor to Cosmos Framework's raw JSON input.

    ``cosmos_framework.inference.action._load_actions`` takes a bare ``[T, raw_dim]``
    JSON array and pads it to the model's 64 action channels itself.  Keeping the
    64-D tensor as the handoff and generating this 10-D file only at invocation
    preserves both the checkpoint-card contract and the native CLI contract.
    """

    payload = _action_payload(actions)
    if any(
        any(value != 0.0 for value in action[RAW_ACTION_DIM:]) for action in actions
    ):
        raise DroidForwardDynamicsError(
            "DROID's padded 64-D action suffix must be zero"
        )
    raw_actions = [action[:RAW_ACTION_DIM] for action in payload["actions"]]
    if any(len(action) != RAW_ACTION_DIM for action in raw_actions):
        raise DroidForwardDynamicsError("native action JSON must be exactly [16, 10]")
    return raw_actions


def prepare_droid_forward_dynamics(
    *, input_path: str, output_path: str
) -> dict[str, Any]:
    """Prepare one selected held-out DROID window and its exact action tensor."""

    selection = read_json_uri(input_path)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise DroidForwardDynamicsError(f"expected {SELECTION_SCHEMA} input")
    source = _expect_mapping(selection.get("source"), "source")
    if str(source.get("dataset_repository") or "") != COSMOS3_DROID_REPOSITORY:
        raise DroidForwardDynamicsError(
            "selection must identify the Cosmos3-DROID source dataset"
        )
    if str(source.get("dataset_version") or "") != COSMOS3_DROID_DATASET_VERSION:
        raise DroidForwardDynamicsError(
            "selection must identify the checkpoint card's exact DROID conversion version"
        )
    episode_id = str(selection.get("episode_id") or "")
    if not episode_id:
        raise DroidForwardDynamicsError("selection must identify its source episode")
    heldout = _card_holdout_metadata(selection)
    frames = selection.get("frames")
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise DroidForwardDynamicsError(
            f"selection frames must contain exactly {FRAME_COUNT} observations"
        )
    poses = selection.get("poses_abs")
    if not isinstance(poses, list):
        raise DroidForwardDynamicsError("selection must provide absolute DROID poses")
    raw_gripper_actions = _finite_vector(
        selection.get("gripper_actions_raw"),
        PREDICTION_FRAME_COUNT,
        "gripper_actions_raw",
    )
    actions = _relative_actions(poses, raw_gripper_actions)

    with policy_workspace(output_path, "droid-fd-prepare") as root:
        artifacts = root / "artifacts"
        artifacts.mkdir()
        composite = [
            _composite_frame(_expect_mapping(frame, f"frames[{index}]"), index)
            for index, frame in enumerate(frames)
        ]
        reference_video = artifacts / "reference_composite.mp4"
        _write_video(composite, reference_video)
        initial_frame = artifacts / "initial_composite.png"
        composite[0].save(initial_frame, format="PNG")
        action_file = artifacts / "actions_true.json"
        action_file.write_bytes(_canonical_json(_action_payload(actions)) + b"\n")
        selection_copy = artifacts / "source_selection.json"
        selection_copy.write_bytes(_canonical_json(selection) + b"\n")
        report = {
            "schema": PREPARED_SCHEMA,
            "status": "succeeded",
            "episode_id": episode_id,
            "source": source,
            "heldout": heldout,
            "frame_contract": {
                "observation_count": FRAME_COUNT,
                "prediction_count": PREDICTION_FRAME_COUNT,
                "fps": FPS,
                "camera_layout": "concat_view(wrist top; exterior_1/exterior_2 bottom)",
                "composite_size": [COMPOSITE_WIDTH, COMPOSITE_HEIGHT],
            },
            "action_contract": {
                "relative_translation": True,
                "rotation": "rot6d_first_two_rotation_columns",
                "gripper": "1.0 - source.action.gripper_position",
                "normalization": "none",
                "shape": [PREDICTION_FRAME_COUNT, PADDED_ACTION_DIM],
                "domain_name": DOMAIN_NAME,
                "domain_id": DOMAIN_ID,
            },
            "source_selection_sha256": _sha256_bytes(_canonical_json(selection)),
        }
        return publish_bundle(artifacts, output_path, report, "prepared.json")


def _download_checkpoint(revision: str) -> tuple[Path, dict[str, Any]]:
    """Fetch and independently bind the exported checkpoint to an immutable HF SHA."""

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise DroidForwardDynamicsError(
            "huggingface_hub is required for checkpoint runtime fetch"
        ) from exc
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise DroidForwardDynamicsError(
            "checkpoint revision must be a full immutable Git SHA"
        )
    info = HfApi().model_info(CHECKPOINT_REPOSITORY, revision=revision)
    if str(getattr(info, "sha", "")) != revision:
        raise DroidForwardDynamicsError(
            "Hugging Face did not resolve the requested checkpoint SHA"
        )
    allow_patterns = [
        "LICENSE",
        "README.md",
        "config.json",
        "checkpoint.json",
        "export_manifest.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
    ]
    # Hugging Face's immutable revision cache provides process-safe locks and ready
    # markers.  Do not copy 30GB of weights into a per-stage directory.
    local_dir = Path(
        snapshot_download(
            repo_id=CHECKPOINT_REPOSITORY,
            revision=revision,
            allow_patterns=allow_patterns,
        )
    )
    expected = {
        "LICENSE",
        "README.md",
        "config.json",
        "checkpoint.json",
        "export_manifest.json",
        "model.safetensors.index.json",
    }
    missing = [name for name in sorted(expected) if not (local_dir / name).is_file()]
    weights = sorted(local_dir.glob("model-*.safetensors"))
    if missing or len(weights) != 7:
        raise DroidForwardDynamicsError(
            "checkpoint snapshot lacks the expected config/provenance/7 shards"
        )
    export_manifest = json.loads(
        (local_dir / "export_manifest.json").read_text(encoding="utf-8")
    )
    if export_manifest.get("framework_commit") != CARD_FRAMEWORK_REVISION:
        raise DroidForwardDynamicsError(
            "export manifest no longer identifies the pinned card framework commit"
        )
    return local_dir, {
        "repository": CHECKPOINT_REPOSITORY,
        "revision": revision,
        "model_info_sha": str(info.sha),
        "export_framework_commit": CARD_FRAMEWORK_REVISION,
        "metadata_sha256": {
            name: file_digest(local_dir / name) for name in sorted(expected)
        },
        # This reads each full shard separately from model loading so the exported
        # identity is independently checked rather than inferred from a loader.
        "weight_sha256": {path.name: file_digest(path) for path in weights},
    }


def _runtime_framework_revision(repo: Path) -> str:
    marker = repo / NATIVE_SOURCE_MARKER
    if marker.is_file():
        revision = marker.read_text(encoding="utf-8").strip()
        if len(revision) == 40 and all(
            character in "0123456789abcdef" for character in revision
        ):
            return revision
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _require_pinned_runtime_framework(repo: Path) -> str:
    revision = _runtime_framework_revision(repo)
    if revision != NATIVE_FRAMEWORK_REVISION:
        raise DroidForwardDynamicsError(
            "native cosmos-framework revision does not match the qualified DROID runtime pin"
        )
    return revision


def _native_inference_argv(
    *, repo: Path, checkpoint: Path, input_json: Path, output_dir: Path, seed: int
) -> list[str]:
    """Invoke upstream's generic inference entrypoint with a native action sample.

    Cosmos Framework deliberately keeps action fields in the per-sample JSON
    schema, rather than exposing them as top-level process flags.  This retains
    the upstream action loader, padding, embodiment lookup, and output layout.
    """

    python = repo / ".venv" / "bin" / "python"
    if not python.is_file():
        raise DroidForwardDynamicsError(
            "the npa-cosmos3 image lacks its framework interpreter"
        )
    return [
        str(python),
        "-m",
        "cosmos_framework.scripts.inference",
        "--parallelism-preset=latency",
        "--checkpoint-path",
        str(checkpoint),
        "--seed",
        str(seed),
        "-i",
        str(input_json),
        "-o",
        str(output_dir),
    ]


def _native_forward_dynamics_input(
    *, prepared: Path, action_path: Path, control: str, seed: int
) -> dict[str, Any]:
    """Build the exact upstream per-sample JSON for DROID forward dynamics."""

    reference_video = prepared / "reference_composite.mp4"
    if not reference_video.is_file():
        raise DroidForwardDynamicsError(
            "prepared handoff lacks the 17-frame reference composite"
        )
    return {
        "model_mode": "forward_dynamics",
        "name": f"droid-{control}",
        "vision_path": str(reference_video),
        "action_path": str(action_path),
        "domain_name": DOMAIN_NAME,
        "action_chunk_size": PREDICTION_FRAME_COUNT,
        "image_size": 480,
        "fps": FPS,
        "view_point": "ego_view",
        "prompt": "DROID robot manipulation observation sequence.",
        "seed": seed,
    }


def _run_native_inference(
    *,
    root: Path,
    prepared: Path,
    actions: list[list[float]],
    control: str,
    seed: int,
    revision: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    action_path = root / f"actions_{control}.json"
    action_payload = _action_payload(actions, control=control)
    native_actions = _native_raw_actions(actions)
    action_path.write_bytes(_canonical_json(native_actions) + b"\n")
    repo = Path(os.environ.get("COSMOS3_REPO") or "/opt/cosmos3/cosmos-framework")
    if not (repo / "cosmos_framework" / "scripts" / "inference.py").is_file():
        raise DroidForwardDynamicsError(
            "native cosmos-framework inference is absent from this image"
        )
    runtime_framework_revision = _require_pinned_runtime_framework(repo)
    checkpoint, checkpoint_identity = _download_checkpoint(revision)
    input_json = root / "forward_dynamics_input.json"
    input_json.write_bytes(
        _canonical_json(
            _native_forward_dynamics_input(
                prepared=prepared,
                action_path=action_path,
                control=control,
                seed=seed,
            )
        )
        + b"\n"
    )
    output_dir = root / "native-output"
    output_dir.mkdir()
    argv = _native_inference_argv(
        repo=repo,
        checkpoint=checkpoint,
        input_json=input_json,
        output_dir=output_dir,
        seed=seed,
    )
    log = root / "native-inference.log"
    with log.open("wb") as stream:
        result = subprocess.run(
            argv, cwd=repo, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise DroidForwardDynamicsError(
            f"native forward-dynamics inference failed (exit {result.returncode})"
        )
    videos = sorted(path for path in output_dir.rglob("*.mp4") if path.is_file())
    if len(videos) != 1:
        raise DroidForwardDynamicsError(
            f"native forward-dynamics inference must produce exactly one MP4; found {len(videos)}"
        )
    decoded = _decode_video(videos[0])
    if len(decoded) not in {PREDICTION_FRAME_COUNT, FRAME_COUNT}:
        raise DroidForwardDynamicsError(
            "native output must contain either 16 future frames or the 17-frame window"
        )
    provenance = {
        "native_argv": argv,
        "runtime_framework": {
            "repository": NATIVE_FRAMEWORK_REPOSITORY,
            "revision": runtime_framework_revision,
            "license": NATIVE_FRAMEWORK_LICENSE,
        },
        "checkpoint_identity": checkpoint_identity,
        "action_payload_sha256": _sha256_bytes(_canonical_json(action_payload)),
        "native_raw_action_sha256": _sha256_bytes(_canonical_json(native_actions)),
        "native_raw_action_shape": [PREDICTION_FRAME_COUNT, RAW_ACTION_DIM],
        "decoded_frame_count": len(decoded),
    }
    return videos[0], provenance, action_payload


def _materialize_prepared(input_path: str, target: Path) -> tuple[Path, dict[str, Any]]:
    report = materialize_bundle(input_path, target, PREPARED_SCHEMA)
    expected = {"reference_composite.mp4", "initial_composite.png", "actions_true.json"}
    missing = [name for name in sorted(expected) if not (target / name).is_file()]
    if missing:
        raise DroidForwardDynamicsError(
            f"prepared handoff is missing {', '.join(missing)}"
        )
    return target, report


def predict_droid_forward_dynamics(
    *,
    input_path: str,
    output_path: str,
    seed: int,
    checkpoint_revision: str = CHECKPOINT_REVISION,
) -> dict[str, Any]:
    """Run native forward dynamics with the held-out sample's true actions."""

    with policy_workspace(output_path, "droid-fd-predict") as root:
        prepared, prepared_report = _materialize_prepared(input_path, root / "prepared")
        true_actions = json.loads((prepared / "actions_true.json").read_text())[
            "actions"
        ]
        video, native, _ = _run_native_inference(
            root=root,
            prepared=prepared,
            actions=true_actions,
            control="true",
            seed=seed,
            revision=checkpoint_revision,
        )
        artifacts = root / "artifacts"
        artifacts.mkdir()
        shutil.copyfile(video, artifacts / "prediction_true.mp4")
        shutil.copyfile(
            root / "native-inference.log", artifacts / "native-inference.log"
        )
        report = {
            "schema": PREDICTION_SCHEMA,
            "status": "succeeded",
            "prepared_manifest_sha256": _sha256_bytes(_canonical_json(prepared_report)),
            "control": "true",
            "seed": seed,
            "native": native,
        }
        return publish_bundle(artifacts, output_path, report, "prediction.json")


def _permuted_actions(
    actions: list[list[float]], seed: int
) -> tuple[list[list[float]], list[int]]:
    permutation = list(range(len(actions)))
    random.Random(seed).shuffle(permutation)
    if permutation == list(range(len(actions))):
        permutation = permutation[1:] + permutation[:1]
    shuffled = [actions[index] for index in permutation]
    if shuffled == actions:
        raise DroidForwardDynamicsError(
            "matched shuffle is identical to true actions; select a nonconstant held-out window"
        )
    return shuffled, permutation


def controls_droid_forward_dynamics(
    *,
    input_path: str,
    output_path: str,
    seed: int,
    checkpoint_revision: str = CHECKPOINT_REVISION,
) -> dict[str, Any]:
    """Generate matched temporal-shuffle and zero-action native controls."""

    with policy_workspace(output_path, "droid-fd-controls") as root:
        prepared, prepared_report = _materialize_prepared(input_path, root / "prepared")
        true_actions = json.loads((prepared / "actions_true.json").read_text())[
            "actions"
        ]
        permuted, permutation = _permuted_actions(true_actions, seed)
        zero_actions = [
            [0.0] * PADDED_ACTION_DIM for _ in range(PREDICTION_FRAME_COUNT)
        ]
        perm_video, perm_native, _ = _run_native_inference(
            root=root / "perm",
            prepared=prepared,
            actions=permuted,
            control="temporal_permutation",
            seed=seed,
            revision=checkpoint_revision,
        )
        zero_video, zero_native, _ = _run_native_inference(
            root=root / "zero",
            prepared=prepared,
            actions=zero_actions,
            control="zero",
            seed=seed,
            revision=checkpoint_revision,
        )
        artifacts = root / "artifacts"
        artifacts.mkdir()
        shutil.copyfile(perm_video, artifacts / "prediction_permuted.mp4")
        shutil.copyfile(zero_video, artifacts / "prediction_zero.mp4")
        shutil.copyfile(
            root / "perm" / "native-inference.log", artifacts / "permuted-inference.log"
        )
        shutil.copyfile(
            root / "zero" / "native-inference.log", artifacts / "zero-inference.log"
        )
        report = {
            "schema": CONTROLS_SCHEMA,
            "status": "succeeded",
            "prepared_manifest_sha256": _sha256_bytes(_canonical_json(prepared_report)),
            "seed": seed,
            "temporal_permutation": permutation,
            "controls": {"permuted": perm_native, "zero": zero_native},
        }
        return publish_bundle(artifacts, output_path, report, "controls.json")


def _future_frames(frames: list[np.ndarray], label: str) -> list[np.ndarray]:
    if len(frames) == FRAME_COUNT:
        return frames[1:]
    if len(frames) == PREDICTION_FRAME_COUNT:
        return frames
    raise DroidForwardDynamicsError(f"{label} must decode to 16 or 17 frames")


def _frame_metrics(
    prediction: list[np.ndarray], reference: list[np.ndarray]
) -> list[dict[str, float]]:
    if len(prediction) != PREDICTION_FRAME_COUNT or len(reference) != FRAME_COUNT:
        raise DroidForwardDynamicsError(
            "evaluation requires 16 predictions and a 17-frame reference"
        )
    target = reference[1:]
    if any(pred.shape != ref.shape for pred, ref in zip(prediction, target)):
        raise DroidForwardDynamicsError(
            "prediction and held-out reference frame geometry differ"
        )
    metrics: list[dict[str, float]] = []
    for index, (predicted, expected) in enumerate(zip(prediction, target), start=1):
        mse = float(
            np.mean((predicted.astype(np.float32) - expected.astype(np.float32)) ** 2)
        )
        psnr = float("inf") if mse == 0 else 10.0 * math.log10((255.0**2) / mse)
        metrics.append({"frame": index, "mse": mse, "psnr_db": psnr})
    return metrics


def _mean_frame_difference(
    left: list[np.ndarray], right: list[np.ndarray], label: str
) -> float:
    if len(left) != len(right) or any(a.shape != b.shape for a, b in zip(left, right)):
        raise DroidForwardDynamicsError(
            f"{label} videos do not have matching decoded geometry"
        )
    return float(
        np.mean(
            [
                np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
                for a, b in zip(left, right)
            ]
        )
    )


def evaluate_droid_forward_dynamics(
    *, prepared_path: str, prediction_path: str, controls_path: str, output_path: str
) -> dict[str, Any]:
    """Measure held-out visual error and true-vs-control action sensitivity."""

    with policy_workspace(output_path, "droid-fd-evaluate") as root:
        prepared, prepared_report = _materialize_prepared(
            prepared_path, root / "prepared"
        )
        prediction_report = materialize_bundle(
            prediction_path, root / "prediction", PREDICTION_SCHEMA
        )
        controls_report = materialize_bundle(
            controls_path, root / "controls", CONTROLS_SCHEMA
        )
        true_video = root / "prediction" / "prediction_true.mp4"
        perm_video = root / "controls" / "prediction_permuted.mp4"
        zero_video = root / "controls" / "prediction_zero.mp4"
        if not all(path.is_file() for path in (true_video, perm_video, zero_video)):
            raise DroidForwardDynamicsError(
                "prediction handoffs are missing their decoded-video evidence"
            )
        reference = _decode_video(prepared / "reference_composite.mp4")
        true = _future_frames(_decode_video(true_video), "true prediction")
        permuted = _future_frames(_decode_video(perm_video), "permuted control")
        zero = _future_frames(_decode_video(zero_video), "zero control")
        visual = _frame_metrics(true, reference)
        action_sensitivity = {
            "true_vs_temporal_permutation_mse": _mean_frame_difference(
                true, permuted, "true/permuted"
            ),
            "true_vs_zero_mse": _mean_frame_difference(true, zero, "true/zero"),
            "interpretation": (
                "One held-out window diagnostic only; positive visual differences prove neither "
                "a benchmark result nor a physical-robot outcome."
            ),
        }
        artifacts = root / "artifacts"
        artifacts.mkdir()
        for source, name in (
            (prepared / "reference_composite.mp4", "reference_composite.mp4"),
            (true_video, "prediction_true.mp4"),
            (perm_video, "prediction_permuted.mp4"),
            (zero_video, "prediction_zero.mp4"),
        ):
            shutil.copyfile(source, artifacts / name)
        report = {
            "schema": EVALUATION_SCHEMA,
            "status": "succeeded",
            "prepared_manifest_sha256": _sha256_bytes(_canonical_json(prepared_report)),
            "prediction_manifest_sha256": _sha256_bytes(
                _canonical_json(prediction_report)
            ),
            "controls_manifest_sha256": _sha256_bytes(_canonical_json(controls_report)),
            "evaluation_scope": prepared_report["heldout"],
            "visual_error": {
                "metric": "decoded_rgb24_pixel_mse_and_psnr",
                "frame_metrics": visual,
                "mean_mse": float(np.mean([item["mse"] for item in visual])),
                "mean_psnr_db": float(np.mean([item["psnr_db"] for item in visual])),
            },
            "action_sensitivity": action_sensitivity,
            "claims": {
                "heldout_benchmark": False,
                "published_action_probe_reproduced": False,
                "physical_robot_success": False,
            },
        }
        write_local_json(artifacts / "metrics.json", report)
        return publish_bundle(artifacts, output_path, report, "evaluation.json")


def _rerun_binary() -> str:
    candidate = Path(sys.executable).with_name("rerun")
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("rerun")
    if not found:
        raise DroidForwardDynamicsError(
            "rerun CLI is unavailable; cannot verify a recording"
        )
    return found


def _record_rrd(
    *,
    destination: Path,
    reference: list[np.ndarray],
    true: list[np.ndarray],
    permuted: list[np.ndarray],
    zero: list[np.ndarray],
    evaluation: dict[str, Any],
) -> None:
    try:
        import rerun as rr
    except ImportError as exc:
        raise DroidForwardDynamicsError(
            "rerun-sdk is required for synchronized evidence"
        ) from exc
    recording = rr.RecordingStream("npa_cosmos3_droid_forward_dynamics")
    recording.save(str(destination))
    for index, source in enumerate(reference):
        try:
            rr.set_time("frame", sequence=index, recording=recording)
        except TypeError:
            rr.set_time_sequence("frame", index, recording=recording)
        rr.log("observations/reference", rr.Image(source), recording=recording)
        if index == 0:
            continue
        future_index = index - 1
        rr.log("predictions/true", rr.Image(true[future_index]), recording=recording)
        rr.log(
            "predictions/temporal_permutation",
            rr.Image(permuted[future_index]),
            recording=recording,
        )
        rr.log(
            "predictions/zero_action", rr.Image(zero[future_index]), recording=recording
        )
        metrics = evaluation["visual_error"]["frame_metrics"][future_index]
        rr.log("metrics/visual_mse", rr.Scalars(metrics["mse"]), recording=recording)
        rr.log(
            "metrics/visual_psnr_db",
            rr.Scalars(metrics["psnr_db"]),
            recording=recording,
        )
    summary = json.dumps(
        {
            "schema": VISUALIZATION_SCHEMA,
            "action_sensitivity": evaluation["action_sensitivity"],
            "claims": evaluation["claims"],
        },
        indent=2,
        sort_keys=True,
    )
    if hasattr(rr, "TextDocument"):
        rr.log(
            "provenance/summary",
            rr.TextDocument(summary, media_type="application/json"),
            static=True,
            recording=recording,
        )
    flush = getattr(recording, "flush", None)
    if callable(flush):
        try:
            flush(blocking=True)
        except TypeError:
            flush()
    # Finalize the file sink before independent verification; without a
    # disconnect the stream can retain an unsealed footer in a live process.
    recording.disconnect()
    if not destination.is_file() or destination.stat().st_size <= 64:
        raise DroidForwardDynamicsError("Rerun did not produce a nonempty recording")
    verified = subprocess.run(
        [_rerun_binary(), "rrd", "verify", str(destination)],
        capture_output=True,
        check=False,
    )
    if verified.returncode:
        detail = verified.stderr.decode("utf-8", errors="replace").strip()
        raise DroidForwardDynamicsError(
            "Rerun rejected the synchronized recording"
            + (f": {detail}" if detail else "")
        )


def visualize_droid_forward_dynamics(
    *,
    prepared_path: str,
    prediction_path: str,
    controls_path: str,
    evaluation_path: str,
    output_path: str,
) -> dict[str, Any]:
    """Emit a verified RRD from actual observation, prediction, and metric bytes."""

    with policy_workspace(output_path, "droid-fd-visualize") as root:
        prepared, prepared_report = _materialize_prepared(
            prepared_path, root / "prepared"
        )
        prediction_report = materialize_bundle(
            prediction_path, root / "prediction", PREDICTION_SCHEMA
        )
        controls_report = materialize_bundle(
            controls_path, root / "controls", CONTROLS_SCHEMA
        )
        evaluation_report = materialize_bundle(
            evaluation_path, root / "evaluation", EVALUATION_SCHEMA
        )
        reference = _decode_video(prepared / "reference_composite.mp4")
        true = _future_frames(
            _decode_video(root / "prediction" / "prediction_true.mp4"),
            "true prediction",
        )
        permuted = _future_frames(
            _decode_video(root / "controls" / "prediction_permuted.mp4"),
            "permuted control",
        )
        zero = _future_frames(
            _decode_video(root / "controls" / "prediction_zero.mp4"), "zero control"
        )
        artifacts = root / "artifacts"
        artifacts.mkdir()
        rrd = artifacts / "droid_forward_dynamics.rrd"
        _record_rrd(
            destination=rrd,
            reference=reference,
            true=true,
            permuted=permuted,
            zero=zero,
            evaluation=evaluation_report,
        )
        provenance = {
            "schema": "npa.cosmos3.droid-fd.provenance.v1",
            "checkpoint": {
                "repository": CHECKPOINT_REPOSITORY,
                "revision": prediction_report["native"]["checkpoint_identity"][
                    "revision"
                ],
                "export_framework_commit": CARD_FRAMEWORK_REVISION,
                "card_url": CARD_URL,
            },
            "upstream": {
                "derivative_checkpoint": {
                    "repository": CHECKPOINT_REPOSITORY,
                    "publisher": "jere-mybao",
                    "license_url": f"{CARD_URL}/blob/main/LICENSE",
                },
                "framework": {
                    "repository": FRAMEWORK_URL,
                    "license_url": f"{FRAMEWORK_URL}/blob/main/LICENSE",
                    "notice_url": f"{FRAMEWORK_URL}/blob/main/NOTICE",
                    "copyright": "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.",
                },
                "dataset": {
                    "repository": COSMOS3_DROID_REPOSITORY,
                    "droid_citation": DROID_CITATION_URL,
                    "license_url": OPENMDW_LICENSE_URL,
                },
            },
            "npa_modifications": [
                "Hash-verified staged DROID selection and three-view composition.",
                "Card-specified relative rot6d and inverted-gripper action encoding.",
                "Native true/permuted/zero inference orchestration, decoded metrics, and Rerun provenance.",
            ],
            "artifact_manifests": {
                "prepared": _sha256_bytes(_canonical_json(prepared_report)),
                "prediction": _sha256_bytes(_canonical_json(prediction_report)),
                "controls": _sha256_bytes(_canonical_json(controls_report)),
                "evaluation": _sha256_bytes(_canonical_json(evaluation_report)),
            },
            "rrd_sha256": file_digest(rrd),
        }
        write_local_json(artifacts / "provenance.json", provenance)
        report = {
            "schema": VISUALIZATION_SCHEMA,
            "status": "succeeded",
            "rrd_frames": FRAME_COUNT,
            "rrd_sha256": provenance["rrd_sha256"],
            "provenance": provenance,
        }
        return publish_bundle(artifacts, output_path, report, "visualization.json")


__all__ = [
    "CHECKPOINT_REPOSITORY",
    "CHECKPOINT_REVISION",
    "CONTROLS_SCHEMA",
    "DroidForwardDynamicsError",
    "EVALUATION_SCHEMA",
    "PREDICTION_SCHEMA",
    "PREPARED_SCHEMA",
    "SELECTION_SCHEMA",
    "VISUALIZATION_SCHEMA",
    "controls_droid_forward_dynamics",
    "evaluate_droid_forward_dynamics",
    "predict_droid_forward_dynamics",
    "prepare_droid_forward_dynamics",
    "visualize_droid_forward_dynamics",
]
