"""Real OpenPI pi0.5 data, inference, training, and artifact-proof stages."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlencode, urlparse
import urllib.request
import zipfile

SOURCE_REF = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
SOURCE_REPOSITORY = "https://github.com/Physical-Intelligence/openpi"
SOURCE_LICENSE = "Apache-2.0"
DEFAULT_CONFIG_NAME = "pi05_droid_jointpos_polaris"
DEFAULT_CHECKPOINT_URI = (
    "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris"
)
DEFAULT_CHECKPOINT_MANIFEST_SHA256 = (
    "8b97388a0bbe5913210255d9e77c8ff925562ec7b40213b4525226c6d5885218"
)
REQUIRED_TOKENIZER_URI = "gs://big_vision/paligemma_tokenizer.model"
REQUIRED_TOKENIZER_GENERATION = "1711547605575873"
REQUIRED_TOKENIZER_SIZE = 4_264_023
REQUIRED_TOKENIZER_MD5 = "FCCtyYVnIKVZ6KhyhLGV4g=="
REQUIRED_TOKENIZER_SHA256 = (
    "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
)
ACTION_HORIZON = 15
ACTION_DIM = 8
JOINT_DIM = 7
IMAGE_SHAPE = (224, 224, 3)
PROMPT = "pick up the fork"
RUNTIME_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


class OpenPIPipelineError(RuntimeError):
    """Raised when a four-mode OpenPI acceptance invariant is not met."""


def _redistribution_evidence(*, trained_checkpoint: bool = False) -> dict[str, str]:
    evidence = {
        "runtime_image": "restricted_private_operator_registry",
        "openpi_source": SOURCE_LICENSE,
        "base_checkpoint": "runtime_only_not_redistributed",
        "dataset": "private_operator_object_storage_only",
    }
    if trained_checkpoint:
        evidence["trained_checkpoint"] = "private_operator_object_storage_only"
    return evidence


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_s3_uri(uri: str, *, require_key: bool = True) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise OpenPIPipelineError(f"expected an s3:// URI, got {uri!r}")
    key = parsed.path.lstrip("/")
    if require_key and not key:
        raise OpenPIPipelineError(f"S3 URI must include an object key: {uri!r}")
    return parsed.netloc, key


def _s3_client():
    import boto3
    from botocore.config import Config

    kwargs: dict[str, object] = {
        "config": Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-central1"),
    }
    endpoint = (os.environ.get("AWS_ENDPOINT_URL") or "").strip()
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def _write_bytes_uri(uri: str, payload: bytes, *, content_type: str) -> None:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        bucket, key = _parse_s3_uri(uri)
        _s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
            IfNoneMatch="*",
        )
        return
    path = Path(parsed.path if parsed.scheme == "file" else uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json_uri(uri: str, payload: Mapping[str, object]) -> None:
    _write_bytes_uri(
        uri,
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        content_type="application/json",
    )


def _read_bytes_uri(uri: str) -> bytes:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        bucket, key = _parse_s3_uri(uri)
        return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return Path(parsed.path if parsed.scheme == "file" else uri).read_bytes()


def _read_json_uri(uri: str) -> dict[str, Any]:
    value = json.loads(_read_bytes_uri(uri))
    if not isinstance(value, dict):
        raise OpenPIPipelineError(f"JSON artifact at {uri!r} is not an object")
    return value


def _uri_exists(uri: str) -> bool:
    """Return whether an owned file/S3 artifact exists without hiding access errors."""

    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        bucket, key = _parse_s3_uri(uri)
        try:
            _s3_client().head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            response = getattr(exc, "response", {}) or {}
            code = str((response.get("Error") or {}).get("Code") or "")
            status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
            if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                return False
            raise
        return True
    path = Path(parsed.path if parsed.scheme == "file" else uri)
    return path.exists()


def validate_actions(actions: object, *, label: str) -> dict[str, object]:
    """Validate the exact consumer trajectory contract and return evidence."""

    import numpy as np

    value = np.asarray(actions)
    if value.ndim != 2 or value.shape[0] < 5 or value.shape[1] != ACTION_DIM:
        raise OpenPIPipelineError(
            f"{label} actions must be shaped float64[T>=5,{ACTION_DIM}], got {value.shape}"
        )
    if value.dtype != np.dtype("float64"):
        raise OpenPIPipelineError(f"{label} actions must be float64, got {value.dtype}")
    if not np.isfinite(value).all():
        raise OpenPIPipelineError(f"{label} actions contain non-finite values")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": True,
        "minimum_horizon_satisfied": True,
    }


def _array_sha256(value: object) -> str:
    """Hash array identity, including dtype and shape, without logging content."""

    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _droid_input_receipt(observation: Mapping[str, object]) -> dict[str, object]:
    import numpy as np
    from openpi.models import model as openpi_model
    from openpi.policies import droid_policy

    source_wrist = np.asarray(observation["observation/wrist_image_left"])
    source_exterior = np.asarray(observation["observation/exterior_image_1_left"])
    droid = droid_policy.DroidInputs(model_type=openpi_model.ModelType.PI05)(
        dict(observation)
    )
    images = droid["image"]
    masks = droid["image_mask"]
    if _array_sha256(images["left_wrist_0_rgb"]) != _array_sha256(source_wrist):
        raise OpenPIPipelineError("DroidInputs changed or swapped the left wrist image")
    if _array_sha256(images["base_0_rgb"]) != _array_sha256(source_exterior):
        raise OpenPIPipelineError("DroidInputs changed or swapped the exterior image")
    if bool(masks["left_wrist_0_rgb"]) is not True:
        raise OpenPIPipelineError("DroidInputs masked the real left wrist image")
    if bool(masks["right_wrist_0_rgb"]) is not False:
        raise OpenPIPipelineError("DroidInputs exposed the padded right wrist image")
    if np.any(images["right_wrist_0_rgb"]):
        raise OpenPIPipelineError("DroidInputs right wrist padding is not zero")
    return {
        "base_0_rgb_sha256": _array_sha256(images["base_0_rgb"]),
        "left_wrist_0_rgb_sha256": _array_sha256(images["left_wrist_0_rgb"]),
        "right_wrist_0_rgb_sha256": _array_sha256(images["right_wrist_0_rgb"]),
        "masks": {name: bool(value) for name, value in masks.items()},
        "left_wrist_identity_preserved": True,
    }


def _composed_input_receipt(
    policy: object, observation: Mapping[str, object]
) -> dict[str, object]:
    import numpy as np

    transformed = policy._input_transform(dict(observation))  # type: ignore[attr-defined]
    transformed_images = transformed["image"]
    transformed_masks = transformed["image_mask"]
    if bool(np.asarray(transformed_masks["left_wrist_0_rgb"]).all()) is not True:
        raise OpenPIPipelineError(
            "composed policy transform masked the left wrist input"
        )
    if bool(np.asarray(transformed_masks["right_wrist_0_rgb"]).any()) is not False:
        raise OpenPIPipelineError(
            "composed policy transform unmasked right wrist padding"
        )
    if not np.asarray(transformed_images["left_wrist_0_rgb"]).any():
        raise OpenPIPipelineError(
            "composed policy transform replaced wrist RGB with zeros"
        )
    return {
        "left_wrist_0_rgb_sha256": _array_sha256(
            transformed_images["left_wrist_0_rgb"]
        ),
        "right_wrist_0_rgb_sha256": _array_sha256(
            transformed_images["right_wrist_0_rgb"]
        ),
        "masks": {
            name: bool(np.asarray(value).all())
            for name, value in transformed_masks.items()
        },
        "left_wrist_nonzero": True,
    }


def _wrist_transform_receipt(
    policy: object, observation: Mapping[str, object]
) -> dict[str, object]:
    """Prove pinned DROID and composed transforms consume left wrist RGB."""

    return {
        "source": {
            "exterior_sha256": _array_sha256(
                observation["observation/exterior_image_1_left"]
            ),
            "wrist_sha256": _array_sha256(observation["observation/wrist_image_left"]),
        },
        "droid_inputs": _droid_input_receipt(observation),
        "composed_model_input": _composed_input_receipt(policy, observation),
    }


def _infer_wrist_variants(
    policy: object,
    base: Mapping[str, object],
    variants: Mapping[str, object],
    noise: object,
) -> tuple[dict[str, object], dict[str, object]]:
    import numpy as np

    actions, receipts = {}, {}
    for name, wrist in variants.items():
        observation = dict(base)
        observation["observation/wrist_image_left"] = wrist
        receipts[name] = _wrist_transform_receipt(policy, observation)
        value = np.asarray(policy.infer(observation, noise=noise)["actions"])
        validate_actions(value, label=f"wrist_sensitivity.{name}")
        actions[name] = value
    return actions, receipts


def _wrist_action_deltas(actions: Mapping[str, object]) -> dict[str, object]:
    import numpy as np

    fresh = np.asarray(actions["fresh"])
    deltas = {}
    for name in ("frozen_previous_heldout_frame", "pixel_channel_shuffle"):
        delta = np.asarray(actions[name]) - fresh
        deltas[name] = {
            "l2": float(np.linalg.norm(delta)),
            "mean_absolute": float(np.mean(np.abs(delta))),
            "per_dimension_mean_absolute": np.mean(np.abs(delta), axis=0).tolist(),
            "action_sha256": _array_sha256(actions[name]),
        }
    return deltas


def _paired_wrist_sensitivity(
    policy: object,
    *,
    fresh_observation: Mapping[str, object],
    frozen_wrist: object,
    action_horizon: int,
    action_dim: int,
    seed: int,
) -> dict[str, object]:
    import numpy as np

    fresh_wrist = np.asarray(fresh_observation["observation/wrist_image_left"])
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((action_horizon, action_dim), dtype=np.float32)
    permutation = rng.permutation(fresh_wrist.size)
    shuffled_wrist = fresh_wrist.reshape(-1)[permutation].reshape(fresh_wrist.shape)
    variants = {
        "fresh": fresh_wrist,
        "frozen_previous_heldout_frame": np.asarray(frozen_wrist),
        "pixel_channel_shuffle": shuffled_wrist,
    }
    actions, receipts = _infer_wrist_variants(
        policy, fresh_observation, variants, noise
    )
    return {
        "schema": "npa.workbench.openpi.pi05-wrist-action-sensitivity.v1",
        "scope": "heldout_action_sensitivity_not_functional_effectiveness",
        "same_exterior_state_prompt": True,
        "same_explicit_sampling_noise": True,
        "noise_sha256": _array_sha256(noise),
        "fresh_action_sha256": _array_sha256(actions["fresh"]),
        "variant_receipts": receipts,
        "action_deltas_from_fresh": _wrist_action_deltas(actions),
        "limitations": [
            "action_difference_does_not_establish_helpful_wrist_use",
            "pixel_channel_shuffle_is_out_of_distribution_corruption_not_a_real_frame",
            "matched_closed_loop_task_ablation_is_required_for_effectiveness",
        ],
    }


def _sample_hash(sample: Mapping[str, Any]) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for key in sorted(sample):
        value = sample[key]
        digest.update(key.encode("utf-8") + b"\0")
        if isinstance(value, str):
            digest.update(value.encode("utf-8"))
            continue
        array = np.asarray(value)
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(_canonical_json(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _split_arrays(*, count: int, seed: int, split: str) -> dict[str, Any]:
    import numpy as np

    if count < 1:
        raise OpenPIPipelineError(f"{split} sample count must be positive")
    rng = np.random.default_rng(seed)
    exterior: Any = np.empty((count, *IMAGE_SHAPE), dtype=np.uint8)
    wrist: Any = np.empty_like(exterior)
    joint: Any = np.empty((count, JOINT_DIM), dtype=np.float32)
    gripper: Any = np.empty((count, 1), dtype=np.float32)
    actions: Any = np.empty((count, ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    prompts: list[str] = []
    sample_ids: list[str] = []
    sample_hashes: list[str] = []
    base_joint = np.asarray(
        [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398],
        dtype=np.float32,
    )
    yy, xx = np.indices(IMAGE_SHAPE[:2], dtype=np.uint32)
    for index in range(count):
        identity = f"{split}-{seed:08x}-{index:04d}"
        sample_ids.append(identity)
        phase = (index + 1) * (0.17 if split == "train" else 0.29)
        noise = rng.normal(0.0, 0.012, size=JOINT_DIM).astype(np.float32)
        joint[index] = base_joint + noise + np.float32(phase * 0.01)
        gripper[index, 0] = np.float32(0.035 + 0.004 * ((index + seed) % 3))
        for channel in range(3):
            exterior[index, ..., channel] = (
                xx * (channel + 1) + yy * (index + 2) + seed + channel * 31
            ) % 251
            wrist[index, ..., channel] = (
                np.flip(xx, axis=1) * (index + 1)
                + yy * (channel + 2)
                + seed * 3
                + channel * 17
            ) % 253
        horizon: Any = np.arange(1, ACTION_HORIZON + 1, dtype=np.float32)[:, None]
        direction = np.asarray(
            [0.003, -0.002, 0.0025, -0.0015, 0.002, 0.001, -0.0025],
            dtype=np.float32,
        )[None, :]
        actions[index, :, :JOINT_DIM] = joint[index][None, :] + horizon * direction * (
            1.0 + np.float32(phase)
        )
        actions[index, :, 7] = np.clip(
            gripper[index, 0] - horizon[:, 0] * np.float32(0.0007 + phase * 0.0001),
            0.0,
            0.08,
        )
        prompts.append(PROMPT if index % 2 == 0 else "place the fork on the plate")
        sample_hashes.append(
            _sample_hash(
                {
                    "exterior_image": exterior[index],
                    "wrist_image": wrist[index],
                    "joint_position": joint[index],
                    "gripper_position": gripper[index],
                    "actions": actions[index],
                    "prompt": prompts[-1],
                }
            )
        )
    return {
        "exterior_image": exterior,
        "wrist_image": wrist,
        "joint_position": joint,
        "gripper_position": gripper,
        "actions": actions,
        "prompts": np.asarray(prompts),
        "sample_ids": np.asarray(sample_ids),
        "sample_hashes": sample_hashes,
    }


def build_mini_dataset(
    *, train_samples: int, heldout_samples: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return deterministic disjoint arrays and their machine-verifiable manifest."""

    train = _split_arrays(count=train_samples, seed=seed, split="train")
    heldout = _split_arrays(
        count=heldout_samples, seed=seed + 1_000_003, split="heldout"
    )
    train_ids = [str(item) for item in train["sample_ids"]]
    heldout_ids = [str(item) for item in heldout["sample_ids"]]
    intersection = sorted(set(train_ids) & set(heldout_ids))
    if intersection:
        raise OpenPIPipelineError(f"train/held-out sample leakage: {intersection}")
    arrays: dict[str, Any] = {}
    for split, values in (("train", train), ("heldout", heldout)):
        for key, value in values.items():
            if key != "sample_hashes":
                arrays[f"{split}_{key}"] = value
    manifest: dict[str, Any] = {
        "schema": "npa.workbench.openpi.mini-franka-dataset.v1",
        "generator": {
            "source": "npa.workflows.byof.openpi_pipeline",
            "seed": seed,
            "deterministic_archive": True,
        },
        "contract": {
            "exterior_image": {"shape": list(IMAGE_SHAPE), "dtype": "uint8"},
            "wrist_image": {"shape": list(IMAGE_SHAPE), "dtype": "uint8"},
            "joint_position": {"shape": [JOINT_DIM], "dtype": "float32"},
            "gripper_position": {"shape": [1], "dtype": "float32"},
            "actions": {
                "shape": [ACTION_HORIZON, ACTION_DIM],
                "dtype": "float32",
                "semantics": "absolute_joint_position_targets_dims_0_6_radians;parallel_jaw_gripper_dim_7",
            },
            "prompt": "utf8_string",
        },
        "splits": {
            "train": {
                "count": train_samples,
                "sample_ids": train_ids,
                "sample_hashes": train["sample_hashes"],
            },
            "heldout": {
                "count": heldout_samples,
                "sample_ids": heldout_ids,
                "sample_hashes": heldout["sample_hashes"],
            },
        },
        "split_isolation": {
            "sample_id_intersection": intersection,
            "sample_hash_intersection": sorted(
                set(train["sample_hashes"]) & set(heldout["sample_hashes"])
            ),
            "disjoint": not intersection
            and not (set(train["sample_hashes"]) & set(heldout["sample_hashes"])),
        },
        "limitations": [
            "tiny_deterministic_optimizer_smoke_dataset",
            "not_a_policy_convergence_dataset",
            "not_physical_robot_evidence",
        ],
        "redistribution": _redistribution_evidence(),
    }
    if not manifest["split_isolation"]["disjoint"]:  # type: ignore[index]
        raise OpenPIPipelineError("train/held-out content hashes overlap")
    return arrays, manifest


def deterministic_npz(arrays: Mapping[str, Any]) -> bytes:
    """Serialize arrays as an NPZ with stable ordering and ZIP timestamps."""

    import numpy as np

    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for key in sorted(arrays):
            array_bytes = io.BytesIO()
            np.lib.format.write_array(
                array_bytes, np.asarray(arrays[key]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info, array_bytes.getvalue(), compress_type=zipfile.ZIP_DEFLATED
            )
    return output.getvalue()


def _load_dataset(
    dataset_uri: str, manifest_uri: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    archive = _read_bytes_uri(dataset_uri)
    manifest = _read_json_uri(manifest_uri)
    supported_schemas = {
        "npa.workbench.openpi.mini-franka-dataset.v1",
        "npa.workbench.openpi.pi05-surface-pick-place-dataset.v1",
    }
    if manifest.get("schema") not in supported_schemas:
        raise OpenPIPipelineError("unexpected OpenPI dataset schema")
    if manifest.get("archive_sha256") != _sha256_bytes(archive):
        raise OpenPIPipelineError("miniature dataset archive hash mismatch")
    with np.load(io.BytesIO(archive), allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    _validate_dataset_arrays(arrays, manifest)
    if (
        manifest.get("schema")
        == "npa.workbench.openpi.pi05-surface-pick-place-dataset.v1"
    ):
        _validate_pi05_normalization(arrays, manifest)
    return arrays, manifest


def _validate_pi05_normalization(
    arrays: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    import numpy as np

    reported = manifest.get("normalization")
    if not isinstance(reported, Mapping) or reported.get("source_split") != "train":
        raise OpenPIPipelineError("pi0.5 normalization must come from train only")
    state = np.concatenate(
        [arrays["train_joint_position"], arrays["train_gripper_position"]], axis=1
    ).astype(np.float64)
    action = np.asarray(arrays["train_actions"], dtype=np.float64).reshape(-1, 8)
    expected = {
        "source_split": "train",
        "state_mean": state.mean(axis=0).tolist(),
        "state_std": np.maximum(state.std(axis=0), 1.0e-6).tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": np.maximum(action.std(axis=0), 1.0e-6).tolist(),
    }
    digest = _sha256_bytes(_canonical_json(expected))
    for key in ("state_mean", "state_std", "action_mean", "action_std"):
        if not np.array_equal(np.asarray(reported.get(key)), np.asarray(expected[key])):
            raise OpenPIPipelineError(f"pi0.5 normalization {key} is not reproducible")
    if reported.get("sha256") != digest:
        raise OpenPIPipelineError("pi0.5 normalization provenance hash is invalid")


def _validate_dataset_arrays(
    arrays: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    import numpy as np

    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise OpenPIPipelineError("dataset manifest has no split map")
    verified_hashes: dict[str, set[str]] = {}
    verified_ids: dict[str, set[str]] = {}
    for split in ("train", "heldout"):
        split_manifest = splits.get(split)
        if not isinstance(split_manifest, Mapping):
            raise OpenPIPipelineError(f"dataset manifest has no {split} split")
        count = int(split_manifest.get("count", 0))
        expected: dict[str, tuple[tuple[int, ...], Any]] = {
            "exterior_image": ((count, *IMAGE_SHAPE), np.dtype("uint8")),
            "wrist_image": ((count, *IMAGE_SHAPE), np.dtype("uint8")),
            "joint_position": ((count, JOINT_DIM), np.dtype("float32")),
            "gripper_position": ((count, 1), np.dtype("float32")),
            "actions": ((count, ACTION_HORIZON, ACTION_DIM), np.dtype("float32")),
        }
        required: dict[str, Any] = {}
        for key in (*expected, "sample_ids", "prompts"):
            array_key = f"{split}_{key}"
            if array_key not in arrays:
                raise OpenPIPipelineError(
                    "miniature dataset archive schema is missing required array "
                    f"{array_key!r} for the {split!r} split"
                )
            required[key] = arrays[array_key]
        for key, (shape, dtype) in expected.items():
            value = np.asarray(required[key])
            if (
                value.shape != shape
                or value.dtype != dtype
                or not np.isfinite(value).all()
            ):
                raise OpenPIPipelineError(
                    f"{split}_{key} violates schema: {value.shape} {value.dtype}"
                )
        ids = [str(value) for value in np.asarray(required["sample_ids"])]
        if ids != list(split_manifest.get("sample_ids", [])):
            raise OpenPIPipelineError(f"{split} sample IDs do not match manifest")
        if len(set(ids)) != len(ids):
            raise OpenPIPipelineError(f"{split} sample IDs are not unique")
        hashes = [
            _sample_hash(
                {
                    "exterior_image": required["exterior_image"][index],
                    "wrist_image": required["wrist_image"][index],
                    "joint_position": required["joint_position"][index],
                    "gripper_position": required["gripper_position"][index],
                    "actions": required["actions"][index],
                    "prompt": str(required["prompts"][index]),
                }
            )
            for index in range(count)
        ]
        if hashes != list(split_manifest.get("sample_hashes", [])):
            raise OpenPIPipelineError(
                f"{split} sample content hashes do not match manifest"
            )
        verified_hashes[split] = set(hashes)
        verified_ids[split] = set(ids)
    train_ids = verified_ids["train"]
    heldout_ids = verified_ids["heldout"]
    id_intersection = sorted(train_ids & heldout_ids)
    hash_intersection = sorted(verified_hashes["train"] & verified_hashes["heldout"])
    if id_intersection:
        raise OpenPIPipelineError("dataset split IDs overlap")
    if hash_intersection:
        raise OpenPIPipelineError("dataset split content hashes overlap")
    isolation = manifest.get("split_isolation")
    if not isinstance(isolation, Mapping) or any(
        isolation.get(key) != expected
        for key, expected in {
            "sample_id_intersection": id_intersection,
            "sample_hash_intersection": hash_intersection,
            "disjoint": True,
        }.items()
    ):
        raise OpenPIPipelineError(
            "dataset split-isolation evidence does not match verified content"
        )


def _prepare_data(args: argparse.Namespace) -> int:
    arrays, manifest = build_mini_dataset(
        train_samples=args.train_samples,
        heldout_samples=args.heldout_samples,
        seed=args.seed,
    )
    archive = deterministic_npz(arrays)
    manifest["archive_sha256"] = _sha256_bytes(archive)
    manifest["archive_size_bytes"] = len(archive)
    manifest["dataset_uri"] = args.dataset_uri
    manifest["manifest_uri"] = args.manifest_uri
    _write_bytes_uri(args.dataset_uri, archive, content_type="application/octet-stream")
    _write_json_uri(args.manifest_uri, manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


def _validate_runtime_image(runtime_image: str) -> None:
    if not RUNTIME_IMAGE_RE.fullmatch(runtime_image):
        raise OpenPIPipelineError(
            f"OpenPI runtime image must be immutable and digest-pinned, got {runtime_image!r}"
        )


def _source_build_evidence(repo_root: Path) -> dict[str, object]:
    import importlib.metadata

    source_path = repo_root / "npa_source_metadata.json"
    build_path = repo_root / "npa_build_metadata.json"
    if not source_path.is_file() or not build_path.is_file():
        raise OpenPIPipelineError(
            "BYOF source/build metadata is missing from the runtime image"
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    build = json.loads(build_path.read_text(encoding="utf-8"))
    if source.get("ref") != SOURCE_REF:
        raise OpenPIPipelineError(
            f"unexpected OpenPI source ref: {source.get('ref')!r}"
        )
    if (
        build.get("schema") != "npa.byof.build.v1"
        or build.get("build_command_executed") is not True
        or not re.fullmatch(r"[0-9a-f]{64}", str(build.get("build_command_sha256", "")))
    ):
        raise OpenPIPipelineError(
            "BYOF build metadata does not prove the hashed build command execution"
        )
    editable_install: dict[str, object] | None = None
    expected_url = f"file://{repo_root.resolve()}"
    for distribution in importlib.metadata.distributions():
        direct_url_text = distribution.read_text("direct_url.json")
        if not direct_url_text:
            continue
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            continue
        if (
            str(distribution.metadata["Name"]).lower() == "openpi"
            and direct_url.get("dir_info", {}).get("editable") is True
            and direct_url.get("url") == expected_url
        ):
            editable_install = {
                "distribution": str(distribution.metadata["Name"]),
                "editable": True,
                "url": expected_url,
            }
            break
    if editable_install is None:
        raise OpenPIPipelineError(
            f"pinned OpenPI source is not installed editable from {expected_url}"
        )
    return {
        "source_metadata": source,
        "build_metadata": build,
        "editable_install": editable_install,
    }


def _set_runtime_cache(work_dir: str, mode: str) -> Path:
    """Keep runtime-fetched model material on the declared ephemeral scratch disk."""

    cache = Path(work_dir) / "runtime-cache" / mode
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["OPENPI_DATA_HOME"] = str(cache)
    return cache


def _observation(index: int = 0) -> dict[str, Any]:
    arrays = _split_arrays(count=index + 1, seed=20_260_815, split="request")
    return {
        "observation/exterior_image_1_left": arrays["exterior_image"][index],
        "observation/wrist_image_left": arrays["wrist_image"][index],
        "observation/joint_position": arrays["joint_position"][index],
        "observation/gripper_position": arrays["gripper_position"][index],
        "prompt": PROMPT,
    }


def _public_gcs_records(checkpoint_uri: str) -> tuple[str, str, list[dict[str, Any]]]:
    parsed = urlparse(checkpoint_uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise OpenPIPipelineError("public checkpoint preflight requires a gs:// URI")
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    page_token = ""
    records: list[dict[str, Any]] = []
    while True:
        payload = _public_gcs_page(parsed.netloc, prefix, page_token)
        for item in payload.get("items", []):
            records.append(
                {
                    "name": item["name"],
                    "generation": item.get("generation", ""),
                    "size": int(item.get("size", 0)),
                    "md5Hash": item.get("md5Hash", ""),
                    "crc32c": item.get("crc32c", ""),
                }
            )
        page_token = payload.get("nextPageToken", "")
        if not page_token:
            break
    records.sort(key=lambda item: str(item["name"]))
    if not records or sum(int(item["size"]) for item in records) <= 0:
        raise OpenPIPipelineError("public checkpoint listing contains no payload bytes")
    if any(
        not item["generation"] or not item["crc32c"] or not item["md5Hash"]
        for item in records
    ):
        raise OpenPIPipelineError(
            "public checkpoint lacks generation, MD5, or CRC32C metadata"
        )
    return parsed.netloc, prefix, records


def _public_gcs_page(bucket: str, prefix: str, page_token: str) -> dict[str, Any]:
    query = {"prefix": prefix, "maxResults": "1000"}
    if page_token:
        query["pageToken"] = page_token
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?" + urlencode(query)
    with urllib.request.urlopen(url) as response:  # noqa: S310 - public GCS API
        return json.load(response)


def _probe_public_gcs_payload(
    bucket: str, records: Sequence[Mapping[str, Any]]
) -> dict[str, object]:
    weight_shards = [
        item
        for item in records
        if "/params/" in str(item["name"]) and int(item["size"]) >= 1024 * 1024
    ]
    if not weight_shards:
        raise OpenPIPipelineError(
            "public checkpoint has no probeable params weight shard"
        )
    record = max(weight_shards, key=lambda item: int(item["size"]))
    name = str(record["name"])
    generation = str(record["generation"])
    url = (
        f"https://storage.googleapis.com/download/storage/v1/b/{quote(bucket, safe='')}/o/"
        f"{quote(name, safe='')}?alt=media&generation={quote(generation, safe='')}"
    )
    payload = _read_public_gcs_payload(url, 2, headers={"Range": "bytes=0-0"})
    if len(payload) != 1:
        raise OpenPIPipelineError("public checkpoint payload-byte probe was not exact")
    return {
        "object_kind": "params_weight_shard",
        "object_name_sha256": _sha256_bytes(name.encode("utf-8")),
        "generation": generation,
        "object_size_bytes": int(record["size"]),
        "object_crc32c": str(record["crc32c"]),
        "bytes_read": len(payload),
        "generation_bound": True,
        "scope": "access_probe_not_full_payload_integrity_or_model_execution",
        "authentication": "anonymous",
    }


def _read_public_gcs_payload(
    url: str, maximum_bytes: int, *, headers: Mapping[str, str] | None = None
) -> bytes:
    """Read bounded anonymous bytes from the exact HTTPS GCS origin."""
    import httpx

    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "storage.googleapis.com"
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or maximum_bytes < 1
    ):
        raise OpenPIPipelineError("public payload requires the exact HTTPS GCS origin")
    payload = bytearray()
    with httpx.Client(timeout=None, follow_redirects=False, trust_env=False) as client:
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(chunk_size=min(maximum_bytes, 64 * 1024)):
                payload.extend(chunk[: maximum_bytes - len(payload)])
                if len(payload) == maximum_bytes:
                    break
    return bytes(payload)


def _checkpoint_provenance(checkpoint_uri: str) -> dict[str, object]:
    parsed = urlparse(checkpoint_uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        return {"scheme": parsed.scheme or "local", "uri": checkpoint_uri}
    bucket, prefix, records = _public_gcs_records(checkpoint_uri)
    manifest_sha256 = _sha256_bytes(_canonical_json(records))
    if (
        checkpoint_uri == DEFAULT_CHECKPOINT_URI
        and manifest_sha256 != DEFAULT_CHECKPOINT_MANIFEST_SHA256
    ):
        raise OpenPIPipelineError(
            "public Polaris generation manifest differs from the pinned revision"
        )
    provenance = {
        "scheme": "gcs_object_generation_manifest_sha256",
        "sha256": manifest_sha256,
        "expected_sha256": (
            DEFAULT_CHECKPOINT_MANIFEST_SHA256
            if checkpoint_uri == DEFAULT_CHECKPOINT_URI
            else None
        ),
        "object_count": len(records),
        "total_size_bytes": sum(int(item["size"]) for item in records),
        "objects": [
            {
                "path": str(item["name"])[len(prefix) :],
                "generation": str(item["generation"]),
                "size": int(item["size"]),
                "md5Hash": str(item["md5Hash"]),
            }
            for item in records
        ],
        "payload_probe": _probe_public_gcs_payload(bucket, records),
    }
    if checkpoint_uri == DEFAULT_CHECKPOINT_URI:
        provenance["required_tokenizer"] = _tokenizer_provenance()
    return provenance


def _tokenizer_provenance() -> dict[str, object]:
    """Download and bind the separate tokenizer payload to its exact generation."""

    import base64

    parsed = urlparse(REQUIRED_TOKENIZER_URI)
    bucket, name = parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme != "gs" or not bucket or not name:
        raise OpenPIPipelineError("required tokenizer must be an exact gs:// object")
    url = (
        f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/"
        f"{quote(name, safe='')}?alt=media&generation={REQUIRED_TOKENIZER_GENERATION}"
    )
    payload = _read_public_gcs_payload(url, REQUIRED_TOKENIZER_SIZE + 1)
    md5 = base64.b64encode(
        hashlib.md5(payload, usedforsecurity=False).digest()  # noqa: S324
    ).decode("ascii")
    sha256 = hashlib.sha256(payload).hexdigest()
    if len(payload) != REQUIRED_TOKENIZER_SIZE:
        raise OpenPIPipelineError("required tokenizer size differs from pinned payload")
    if md5 != REQUIRED_TOKENIZER_MD5 or sha256 != REQUIRED_TOKENIZER_SHA256:
        raise OpenPIPipelineError("required tokenizer bytes differ from pinned payload")
    return {
        "uri": REQUIRED_TOKENIZER_URI,
        "generation": REQUIRED_TOKENIZER_GENERATION,
        "size": len(payload),
        "md5Hash": md5,
        "sha256": sha256,
        "access": "public_anonymous_generation_bound_full_download",
        "checkpoint_object_count_excludes_this_payload": True,
    }


def _verify_public_checkpoint_download(
    checkpoint_uri: str, checkpoint_dir: Path, provenance: Mapping[str, object]
) -> dict[str, object]:
    """Bind all downloaded public GCS bytes to the preflight object manifest."""

    import base64

    if not checkpoint_uri.startswith("gs://"):
        return {"scope": "non_public_gcs_checkpoint"}
    expected = provenance.get("objects")
    if not isinstance(expected, list) or not expected:
        raise OpenPIPipelineError("public checkpoint provenance lacks object records")
    actual: list[dict[str, object]] = []
    for path in sorted(item for item in checkpoint_dir.rglob("*") if item.is_file()):
        md5 = hashlib.md5(usedforsecurity=False)  # noqa: S324
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                md5.update(chunk)
        actual.append(
            {
                "path": path.relative_to(checkpoint_dir).as_posix(),
                "size": path.stat().st_size,
                "md5Hash": base64.b64encode(md5.digest()).decode("ascii"),
            }
        )
    comparable = [
        {"path": row["path"], "size": row["size"], "md5Hash": row["md5Hash"]}
        for row in expected
    ]
    if actual != comparable:
        raise OpenPIPipelineError(
            "downloaded checkpoint bytes differ from generation-bound GCS manifest"
        )
    return {
        "verified": True,
        "object_count": len(actual),
        "total_size_bytes": sum(int(row["size"]) for row in actual),
        "generation_manifest_sha256": provenance["sha256"],
    }


def _validated_cuda_probe_output(output: str, *, expected_cc: str) -> str:
    expected = f"devices=1 cc={expected_cc}"
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    evidence = [line for line in lines if line.startswith("devices=")]
    if evidence != [expected]:
        raise OpenPIPipelineError(
            f"CUDA probe returned unexpected SM{expected_cc.replace('.', '')} "
            f"evidence: {evidence!r}"
        )
    return expected


def _single_host_runtime_evidence(
    jax_module: Any, environment: Mapping[str, str] | None = None
) -> dict[str, int]:
    source = os.environ if environment is None else environment
    raw_nodes = str(source.get("SKYPILOT_NUM_NODES", "1"))
    try:
        sky_nodes = int(raw_nodes)
    except ValueError as exc:
        raise OpenPIPipelineError("SKYPILOT_NUM_NODES must be an integer") from exc
    process_count = int(jax_module.process_count())
    process_index = int(jax_module.process_index())
    local_devices = len(jax_module.local_devices())
    global_devices = len(jax_module.devices())
    if sky_nodes != 1 or process_count != 1 or process_index != 0:
        raise OpenPIPipelineError(
            "this pi0.5 variant supports one host only; use the full-DROID recipe "
            "for initialized multi-host JAX"
        )
    if local_devices != global_devices:
        raise OpenPIPipelineError("single-host JAX device visibility is inconsistent")
    return {
        "skypilot_nodes": sky_nodes,
        "jax_process_count": process_count,
        "jax_process_index": process_index,
        "local_device_count": local_devices,
        "global_device_count": global_devices,
    }


def _device_capability(device: Any) -> str:
    raw = getattr(device, "compute_capability", "")
    raw = raw() if callable(raw) else raw
    major, minor = getattr(raw, "major", None), getattr(raw, "minor", None)
    if major is not None and minor is not None:
        return f"{int(major)}.{int(minor)}"
    match = re.search(r"(\d{1,2})[.,](\d)", str(raw))
    if match:
        return f"{int(match.group(1))}.{int(match.group(2))}"
    compact = re.search(r"(?:sm_)?(\d{2,3})", str(raw), re.IGNORECASE)
    if not compact:
        raise OpenPIPipelineError(f"cannot parse compute capability {raw!r}")
    number = int(compact.group(1))
    return f"{number // 10}.{number % 10}"


def _compiled_sm_probe(compute_capability: str) -> dict[str, object]:
    required = compute_capability in {"10.0", "12.0"}
    result: dict[str, object] = {"required": required}
    if not required:
        return result
    compact = compute_capability.replace(".", "")
    probe = Path(f"/usr/local/bin/npa-openpi-sm{compact}-probe")
    if not probe.is_file():
        raise OpenPIPipelineError(f"compiled SM{compact} CUDA probe is missing")
    output = subprocess.run(
        [str(probe)], check=True, capture_output=True, text=True
    ).stdout
    elf = subprocess.run(
        ["cuobjdump", "--list-elf", str(probe)],
        check=True,
        capture_output=True,
        text=True,
    )
    if f"sm_{compact}" not in f"{elf.stdout}\n{elf.stderr}".lower():
        raise OpenPIPipelineError(f"CUDA probe has no sm_{compact} ELF")
    result.update(
        {
            "passed": True,
            "output": _validated_cuda_probe_output(
                output, expected_cc=compute_capability
            ),
            f"elf_contains_sm{compact}": True,
        }
    )
    return result


def _nvidia_smi_rows(expected_gpu_count: int) -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = result.stdout.strip().splitlines()
    if len(rows) != expected_gpu_count:
        raise OpenPIPipelineError("nvidia-smi visible GPU count disagrees with JAX")
    return rows


def _validated_gpu_identity(
    devices: Sequence[Any], expected_gpu_type: str, expected_compute_capability: str
) -> tuple[list[str], list[str]]:
    kinds = [str(getattr(device, "device_kind", "")) for device in devices]
    if expected_gpu_type and any(
        expected_gpu_type.upper() not in k.upper() for k in kinds
    ):
        raise OpenPIPipelineError(
            f"expected GPU type {expected_gpu_type!r}, got {kinds!r}"
        )
    capabilities = [_device_capability(device) for device in devices]
    if expected_compute_capability and any(
        c != expected_compute_capability for c in capabilities
    ):
        raise OpenPIPipelineError(
            f"expected compute capability {expected_compute_capability}, got {capabilities}"
        )
    return kinds, capabilities


def _hardware_evidence(
    *, expected_gpu_type: str, expected_gpu_count: int, expected_compute_capability: str
) -> dict[str, object]:
    import importlib.metadata
    import jax
    from jax.extend import backend as jax_backend

    devices = jax.devices()
    if len(devices) != expected_gpu_count:
        raise OpenPIPipelineError(
            f"expected {expected_gpu_count} visible GPU(s), got {len(devices)}"
        )
    kinds, capabilities = _validated_gpu_identity(
        devices, expected_gpu_type, expected_compute_capability
    )
    return {
        "gpu_count_allocated": len(devices),
        "device_kinds": kinds,
        "compute_capabilities": capabilities,
        "nvidia_smi": _nvidia_smi_rows(expected_gpu_count),
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "xla_platform_version": str(jax_backend.get_backend().platform_version),
        "sm100_probe": _compiled_sm_probe("10.0")
        if expected_compute_capability == "10.0"
        else {"required": False},
        "sm120_probe": _compiled_sm_probe("12.0")
        if expected_compute_capability == "12.0"
        else {"required": False},
    }


def _load_direct_runtime(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import numpy as np
    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as openpi_config

    hardware = _hardware_evidence(
        expected_gpu_type=args.expected_gpu_type,
        expected_gpu_count=args.expected_gpu_count,
        expected_compute_capability=args.expected_compute_capability,
    )
    hardware["topology"] = _single_host_runtime_evidence(jax)
    provenance = _checkpoint_provenance(args.checkpoint_uri)
    fetch_started = time.perf_counter()
    checkpoint_dir = Path(download.maybe_download(args.checkpoint_uri, token="anon"))
    payload_binding = _verify_public_checkpoint_download(
        args.checkpoint_uri, checkpoint_dir, provenance
    )
    checkpoint_seconds = time.perf_counter() - fetch_started
    config = openpi_config.get_config(args.config_name)
    if config.model.action_horizon != ACTION_HORIZON:
        raise OpenPIPipelineError("unexpected Polaris action horizon")
    load_started = time.perf_counter()
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    load_seconds = time.perf_counter() - load_started
    return {
        "jax": jax,
        "numpy": np,
        "hardware": hardware,
        "provenance": provenance,
        "payload_binding": payload_binding,
        "fetch_seconds": checkpoint_seconds,
        "policy": policy,
        "load_seconds": load_seconds,
    }


def _direct_trajectory(runtime: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    policy, np = runtime["policy"], runtime["numpy"]
    infer_started = time.perf_counter()
    response = policy.infer(_observation())
    infer_seconds = time.perf_counter() - infer_started
    actions = np.asarray(response["actions"])
    trajectory = validate_actions(actions, label="direct")
    trajectory.update(
        {
            "inference_ms": round(infer_seconds * 1000, 3),
            "first_five_targets": actions[:5].tolist(),
            "action_semantics": "joint_position_targets_dims_0_6_radians;parallel_jaw_gripper_dim_7",
            "consumer_guidance": "execute_about_5_targets_at_15_hz_then_requery",
        }
    )
    return trajectory, actions


def _direct_observation_schema() -> dict[str, Any]:
    return {
        "exterior_image": {"shape": list(IMAGE_SHAPE), "dtype": "uint8"},
        "wrist_image": {"shape": list(IMAGE_SHAPE), "dtype": "uint8"},
        "joint_position": {"shape": [JOINT_DIM], "dtype": "float32"},
        "gripper_position": {"shape": [1], "dtype": "float32"},
        "prompt": PROMPT,
    }


def _public_artifact_access() -> dict[str, Any]:
    return {
        "checkpoint_source": "public_anonymous_gcs",
        "npa_terms_attestation_required": False,
        "credential_persisted": False,
    }


def _direct_result(
    args: argparse.Namespace, build: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, object]:
    trajectory, _actions = _direct_trajectory(runtime)
    return {
        "schema": "npa.workbench.openpi.pi05-direct-inference.v2",
        "status": "passed",
        "mode": "direct",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "ref": SOURCE_REF,
            "license": SOURCE_LICENSE,
            **build,
        },
        "redistribution": _redistribution_evidence(),
        "runtime_image": args.runtime_image,
        "checkpoint": {
            "uri": args.checkpoint_uri,
            "provenance": runtime["provenance"],
            "downloaded_payload_binding": runtime["payload_binding"],
            "fetch_seconds": round(runtime["fetch_seconds"], 3),
            "weights_baked": False,
        },
        "artifact_access": _public_artifact_access(),
        "observation_schema": _direct_observation_schema(),
        "trajectory": trajectory,
        "policy_load_seconds": round(runtime["load_seconds"], 3),
        "memory": _memory_stats(runtime["jax"].devices()[0]),
        "hardware": runtime["hardware"],
        "software": {
            "numpy": runtime["numpy"].__version__,
            "openpi_distribution": "editable_pinned_source",
        },
        "limitations": ["no_physical_franka_task_success_claim"],
    }


def _direct(args: argparse.Namespace) -> int:
    _validate_runtime_image(args.runtime_image)
    _set_runtime_cache(args.work_dir, "direct")
    build = _source_build_evidence(Path(args.repo_root))
    result = _direct_result(args, build, _load_direct_runtime(args))
    _write_json_uri(args.output_uri, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


class _ArrayDataset:
    def __init__(self, arrays: Mapping[str, Any], split: str):
        import numpy as np

        self._arrays = arrays
        self._split = split
        self._count = int(np.asarray(arrays[f"{split}_actions"]).shape[0])

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np

        prefix = self._split
        return {
            "observation": {
                "exterior_image_1_left": np.asarray(
                    self._arrays[f"{prefix}_exterior_image"][index]
                ).copy(),
                "wrist_image_left": np.asarray(
                    self._arrays[f"{prefix}_wrist_image"][index]
                ).copy(),
                "joint_position": np.asarray(
                    self._arrays[f"{prefix}_joint_position"][index]
                ).copy(),
                "gripper_position": np.asarray(
                    self._arrays[f"{prefix}_gripper_position"][index]
                ).copy(),
            },
            "actions": np.asarray(self._arrays[f"{prefix}_actions"][index]).copy(),
            "prompt": str(self._arrays[f"{prefix}_prompts"][index]),
        }


class _StaticDataFactory:
    def __init__(self, data_config: object):
        self._data_config = data_config

    def create(self, _assets_dirs: Path, _model_config: object) -> object:
        return self._data_config


def _training_configuration(
    *,
    arrays: Mapping[str, Any],
    checkpoint_base_dir: Path,
    train_steps: int,
    batch_size: int,
    seed: int,
    base_checkpoint_location: str,
    fsdp_devices: int,
) -> tuple[Any, Any]:
    """Build the supported upstream pi0.5 LoRA config and data adapter."""

    import numpy as np
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.policies import droid_policy
    from openpi.training import config as openpi_config
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    model = pi0_config.Pi0Config(
        action_horizon=ACTION_HORIZON,
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "observation/exterior_image_1_left": "observation/exterior_image_1_left",
                    "observation/wrist_image_left": "observation/wrist_image_left",
                    "observation/joint_position": "observation/joint_position",
                    "observation/gripper_position": "observation/gripper_position",
                    "actions": "actions",
                    "prompt": "prompt",
                }
            )
        ]
    )
    delta_mask = transforms.make_bool_mask(JOINT_DIM, -1)
    data_transforms = transforms.Group(
        inputs=[
            droid_policy.DroidInputs(model_type=model.model_type),
            transforms.DeltaActions(delta_mask),
        ],
        outputs=[
            transforms.AbsoluteActions(delta_mask),
            droid_policy.DroidOutputs(),
        ],
    )
    states = np.concatenate(
        [
            np.asarray(arrays["train_joint_position"], dtype=np.float32),
            np.asarray(arrays["train_gripper_position"], dtype=np.float32),
        ],
        axis=-1,
    )
    absolute_actions = np.asarray(arrays["train_actions"], dtype=np.float32)
    normalized_actions = absolute_actions.copy()
    normalized_actions[..., :JOINT_DIM] -= states[:, None, :JOINT_DIM]

    def stats(value: object):
        array = np.asarray(value, dtype=np.float32).reshape(
            -1, np.asarray(value).shape[-1]
        )
        return transforms.NormStats(
            mean=array.mean(axis=0),
            std=array.std(axis=0),
            q01=np.quantile(array, 0.01, axis=0),
            q99=np.quantile(array, 0.99, axis=0),
        )

    norm_stats = {"state": stats(states), "actions": stats(normalized_actions)}
    data_config = openpi_config.DataConfig(
        repo_id="npa/openpi-mini-franka",
        asset_id="droid",
        norm_stats=norm_stats,
        repack_transforms=repack,
        data_transforms=data_transforms,
        model_transforms=openpi_config.ModelTransformFactory()(model),
        use_quantile_norm=True,
    )
    config = openpi_config.TrainConfig(
        name="npa_pi05_franka_mini_lora",
        exp_name="live-smoke",
        model=model,
        data=_StaticDataFactory(data_config),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            f"{base_checkpoint_location.rstrip('/')}/params"
        ),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1,
            peak_lr=1e-5,
            decay_steps=max(2, train_steps + 1),
            decay_lr=1e-5,
        ),
        optimizer=optimizer.AdamW(),
        ema_decay=None,
        freeze_filter=model.get_freeze_filter(),
        assets_base_dir=str(checkpoint_base_dir / "assets-source"),
        checkpoint_base_dir=str(checkpoint_base_dir),
        seed=seed,
        batch_size=batch_size,
        num_workers=0,
        num_train_steps=train_steps,
        log_interval=1,
        save_interval=max(1, train_steps),
        keep_period=None,
        overwrite=True,
        wandb_enabled=False,
        fsdp_devices=fsdp_devices,
        policy_metadata={
            "schema": "npa.workbench.openpi.pi05-mini-lora.v1",
            "embodiment": "franka_panda_joint_position",
        },
    )
    return config, data_config


def _make_data_loader(
    *,
    arrays: Mapping[str, Any],
    split: str,
    config: Any,
    data_config: Any,
    sharding: Any,
    batch_size: int,
    seed: int,
    num_batches: int | None,
) -> Any:
    from openpi import transforms
    from openpi.training import data_loader

    dataset = data_loader.TransformedDataset(
        _ArrayDataset(arrays, split),
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
    )
    torch_loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        sharding=sharding,
        shuffle=split == "train",
        num_batches=num_batches,
        num_workers=0,
        seed=seed,
        framework="jax",
    )
    return data_loader.DataLoaderImpl(data_config, torch_loader)


def _load_upstream_train_module(repo_root: Path):
    import importlib.util

    path = repo_root / "scripts" / "train.py"
    if not path.is_file():
        raise OpenPIPipelineError(f"upstream train.py is missing from {repo_root}")
    spec = importlib.util.spec_from_file_location("npa_pinned_openpi_train", path)
    if spec is None or spec.loader is None:
        raise OpenPIPipelineError("could not load pinned upstream train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pure_trainable(state: Any, trainable_filter: Any) -> Any:
    return state.params.filter(trainable_filter).to_pure_dict()


def _tree_sha256(tree: object) -> str:
    import jax
    import numpy as np

    digest = hashlib.sha256()
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(index).encode("ascii") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(_canonical_json(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _tree_to_host(tree: object) -> object:
    import jax
    import numpy as np

    return jax.tree.map(lambda leaf: np.asarray(jax.device_get(leaf)).copy(), tree)


def _tree_update_l2(before: object, after: object) -> float:
    import jax
    import numpy as np

    before_leaves = jax.tree_util.tree_leaves(before)
    after_leaves = jax.tree_util.tree_leaves(after)
    if len(before_leaves) != len(after_leaves):
        raise OpenPIPipelineError("trainable parameter structure changed unexpectedly")
    total = 0.0
    for old, new in zip(before_leaves, after_leaves, strict=True):
        delta = np.asarray(jax.device_get(new), dtype=np.float64) - np.asarray(
            jax.device_get(old), dtype=np.float64
        )
        total += float(np.sum(delta * delta))
    return float(total**0.5)


def _optimizer_step_rng(random_module: Any, train_rng: Any, step: int) -> Any:
    """Derive a deterministic, distinct noise key for each optimizer step."""

    return random_module.fold_in(train_rng, step)


def _memory_stats(device: Any) -> dict[str, object]:
    getter = getattr(device, "memory_stats", None)
    if not callable(getter):
        return {}
    raw = getter() or {}
    result: dict[str, object] = {}
    for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit"):
        if key in raw:
            result[key] = int(raw[key])
    return result


def _publish_pi05_component(
    args: argparse.Namespace, stage: int, name: str, evidence: str
) -> None:
    root = str(getattr(args, "component_root_uri", "") or "")
    if not root:
        return
    from npa.workflows.sim2real.workflow_io import publish_component_record

    publish_component_record(
        root_uri=root,
        stage=stage,
        name=name,
        tier="WORKS",
        require_gpu=True,
        evidence=evidence,
        artifacts={"result": args.output_uri, "runtime_image": args.runtime_image},
    )


def _upload_checkpoint(local_root: Path, output_uri: str) -> dict[str, object]:
    bucket, prefix = _parse_s3_uri(output_uri.rstrip("/") + "/")
    prefix = prefix.rstrip("/") + "/"
    client = _s3_client()
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in local_root.rglob("*") if item.is_file()):
        relative = path.relative_to(local_root).as_posix()
        record = {
            "path": relative,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        client.upload_file(str(path), bucket, prefix + relative)
        records.append(record)
    content_hash = _sha256_bytes(_canonical_json(records))
    manifest: dict[str, object] = {
        "schema": "npa.workbench.openpi.checkpoint-manifest.v1",
        "root_uri": output_uri.rstrip("/") + "/",
        "content_manifest_sha256": content_hash,
        "file_count": len(records),
        "total_size_bytes": sum(int(item["size"]) for item in records),
        "files": records,
    }
    client.put_object(
        Bucket=bucket,
        Key=prefix + "manifest.json",
        Body=json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        ContentType="application/json",
        IfNoneMatch="*",
    )
    return manifest


def _download_checkpoint(output_uri: str, local_root: Path) -> dict[str, object]:
    manifest_uri = output_uri.rstrip("/") + "/manifest.json"
    manifest = _read_json_uri(manifest_uri)
    if manifest.get("schema") != "npa.workbench.openpi.checkpoint-manifest.v1":
        raise OpenPIPipelineError("unexpected trained checkpoint manifest schema")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise OpenPIPipelineError("trained checkpoint manifest has no files")
    if manifest.get("content_manifest_sha256") != _sha256_bytes(_canonical_json(files)):
        raise OpenPIPipelineError("trained checkpoint content manifest hash is invalid")
    bucket, prefix = _parse_s3_uri(output_uri.rstrip("/") + "/")
    prefix = prefix.rstrip("/") + "/"
    client = _s3_client()
    for record in files:
        if not isinstance(record, Mapping):
            raise OpenPIPipelineError("invalid checkpoint file record")
        relative = str(record.get("path", ""))
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise OpenPIPipelineError(f"unsafe checkpoint path {relative!r}")
        path = local_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, prefix + relative, str(path))
        if path.stat().st_size != int(record.get("size", -1)):
            raise OpenPIPipelineError(f"checkpoint size mismatch for {relative}")
        if _sha256_file(path) != record.get("sha256"):
            raise OpenPIPipelineError(f"checkpoint hash mismatch for {relative}")
    return manifest


def _training_base_checkpoint(
    args: argparse.Namespace, download: Any
) -> dict[str, Any]:
    provenance = _checkpoint_provenance(args.base_checkpoint_uri)
    started = time.perf_counter()
    checkpoint_dir = Path(
        download.maybe_download(args.base_checkpoint_uri, token="anon")
    )
    binding = _verify_public_checkpoint_download(
        args.base_checkpoint_uri, checkpoint_dir, provenance
    )
    return {
        "dir": checkpoint_dir,
        "provenance": provenance,
        "binding": binding,
        "fetch_seconds": time.perf_counter() - started,
    }


def _validate_training_request(
    args: argparse.Namespace, dataset_manifest: Mapping[str, Any], device_count: int
) -> int:
    train_count = int(dataset_manifest["splits"]["train"]["count"])
    if args.batch_size > train_count:
        raise OpenPIPipelineError("batch size exceeds miniature training split")
    if args.train_steps < 1:
        raise OpenPIPipelineError("at least one optimizer step is required")
    if args.batch_size % device_count != 0:
        raise OpenPIPipelineError(
            "training batch size must be divisible by the allocated GPU count"
        )
    if args.fsdp_devices < 1 or device_count % args.fsdp_devices != 0:
        raise OpenPIPipelineError(
            "FSDP device count must be positive and divide the allocated GPU count"
        )
    return train_count


def _training_data_runtime(
    args: argparse.Namespace, arrays: Mapping[str, Any], base_dir: Path, repo_root: Path
) -> dict[str, Any]:
    from openpi.training import checkpoints
    from openpi.training import sharding as openpi_sharding

    work_root = Path(args.work_dir) / f"train-{os.getpid()}"
    work_root.mkdir(parents=True, exist_ok=False)
    config, data_config = _training_configuration(
        arrays=arrays,
        checkpoint_base_dir=work_root / "checkpoints",
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        base_checkpoint_location=str(base_dir),
        fsdp_devices=args.fsdp_devices,
    )
    mesh = openpi_sharding.make_mesh(args.fsdp_devices)
    loader = _training_loader(args, arrays, config, data_config, mesh, openpi_sharding)
    manager = _fresh_checkpoint_manager(checkpoints, config.checkpoint_dir)
    return {
        "config": config,
        "mesh": mesh,
        "loader": loader,
        "manager": manager,
        "checkpoints": checkpoints,
        "sharding": openpi_sharding,
        "upstream": _load_upstream_train_module(repo_root),
    }


def _training_loader(
    args: argparse.Namespace,
    arrays: Mapping[str, Any],
    config: Any,
    data_config: Any,
    mesh: Any,
    openpi_sharding: Any,
) -> Any:
    import jax

    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    return _make_data_loader(
        arrays=arrays,
        split="train",
        config=config,
        data_config=data_config,
        sharding=sharding,
        batch_size=args.batch_size,
        seed=args.seed,
        num_batches=args.train_steps,
    )


def _fresh_checkpoint_manager(checkpoints: Any, checkpoint_dir: Path) -> Any:
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        checkpoint_dir, keep_period=None, overwrite=True, resume=False
    )
    if resuming:
        raise OpenPIPipelineError(
            "fresh optimizer smoke unexpectedly entered resume mode"
        )
    return manager


def _initialize_training_state(
    args: argparse.Namespace, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    import jax

    config, mesh = runtime["config"], runtime["mesh"]
    train_rng, init_rng = jax.random.split(jax.random.key(args.seed))
    started = time.perf_counter()
    state, state_sharding = runtime["upstream"].init_train_state(
        config, init_rng, mesh, resume=False
    )
    jax.block_until_ready(state)
    before_tree = _tree_to_host(_pure_trainable(state, config.trainable_filter))
    return {
        "state": state,
        "state_sharding": state_sharding,
        "train_rng": train_rng,
        "before_tree": before_tree,
        "before_hash": _tree_sha256(before_tree),
        "init_seconds": time.perf_counter() - started,
    }


def _run_training_steps(
    args: argparse.Namespace, runtime: Mapping[str, Any], initialized: Mapping[str, Any]
) -> dict[str, Any]:
    import functools
    import jax
    import numpy as np

    mesh, loader = runtime["mesh"], runtime["loader"]
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(runtime["sharding"].DATA_AXIS)
    )
    step_fn = jax.jit(
        functools.partial(runtime["upstream"].train_step, runtime["config"]),
        in_shardings=(replicated, initialized["state_sharding"], data),
        out_shardings=(initialized["state_sharding"], replicated),
        donate_argnums=(1,),
    )
    iterator, state, metrics = iter(loader), initialized["state"], []
    batch, started = next(iterator), time.perf_counter()
    with runtime["sharding"].set_mesh(mesh):
        for step in range(args.train_steps):
            rng = _optimizer_step_rng(jax.random, initialized["train_rng"], step)
            state, info = step_fn(rng, state, batch)
            jax.block_until_ready(state)
            row = _finite_optimizer_metrics(jax, np, info, step)
            metrics.append({**row, "optimizer_step": float(step + 1)})
            if step + 1 < args.train_steps:
                batch = next(iterator)
    return {
        "state": state,
        "metrics": metrics,
        "seconds": time.perf_counter() - started,
    }


def _finite_optimizer_metrics(
    jax: Any, numpy: Any, info: Mapping[str, Any], step: int
) -> dict[str, float]:
    row = {
        key: float(numpy.asarray(jax.device_get(value))) for key, value in info.items()
    }
    if not all(numpy.isfinite(value) for value in row.values()):
        raise OpenPIPipelineError(f"non-finite optimizer metrics at step {step}: {row}")
    return row


def _training_evidence(
    build: Mapping[str, Any],
    base: Mapping[str, Any],
    dataset: Mapping[str, Any],
    train_count: int,
    initialized: Mapping[str, Any],
    optimized: Mapping[str, Any],
    update: Mapping[str, Any],
    saved: Mapping[str, Any],
    hardware: Mapping[str, Any],
    memory: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    return {
        "build": build,
        "base": base,
        "dataset": dataset,
        "train_count": train_count,
        "initialized": initialized,
        "optimized": optimized,
        "update": update,
        "saved": saved,
        "hardware": hardware,
        "memory": memory,
        "timings": {
            "initialization": round(initialized["init_seconds"], 3),
            "optimization": round(optimized["seconds"], 3),
            "checkpoint_save": round(saved["seconds"], 3),
            "total": round(time.perf_counter() - started, 3),
        },
    }


def _publish_training_result(
    args: argparse.Namespace, result: Mapping[str, Any]
) -> int:
    _write_json_uri(args.output_uri, result)
    _publish_pi05_component(
        args,
        8,
        "pi05_lora_training",
        "Ran upstream-native pi0.5 LoRA optimization and published a reloadable checkpoint.",
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _verify_training_update(
    args: argparse.Namespace,
    runtime: Mapping[str, Any],
    initialized: Mapping[str, Any],
    state: Any,
) -> dict[str, Any]:
    import jax
    import numpy as np

    after_tree = _tree_to_host(
        _pure_trainable(state, runtime["config"].trainable_filter)
    )
    after_hash = _tree_sha256(after_tree)
    update_l2 = _tree_update_l2(initialized["before_tree"], after_tree)
    if (
        initialized["before_hash"] == after_hash
        or not np.isfinite(update_l2)
        or update_l2 <= 0
    ):
        raise OpenPIPipelineError("optimizer did not change trainable OpenPI state")
    checkpoint_step = int(np.asarray(jax.device_get(state.step)))
    if checkpoint_step != args.train_steps:
        raise OpenPIPipelineError(
            f"optimizer step mismatch: state={checkpoint_step} requested={args.train_steps}"
        )
    return {"after_hash": after_hash, "update_l2": update_l2, "step": checkpoint_step}


def _save_training_state(
    args: argparse.Namespace,
    runtime: Mapping[str, Any],
    state: Any,
    update: Mapping[str, Any],
) -> dict[str, Any]:
    import jax
    import numpy as np
    from openpi.models import model as openpi_model

    started = time.perf_counter()
    manager, loader, step = runtime["manager"], runtime["loader"], update["step"]
    runtime["checkpoints"].save_state(manager, state, loader, step)
    manager.wait_until_finished()
    save_seconds = time.perf_counter() - started
    step_dir = Path(runtime["config"].checkpoint_dir) / str(step)
    params_hash = _tree_sha256(openpi_model.restore_params(step_dir / "params"))
    restored = runtime["checkpoints"].restore_state(manager, state, loader, step)
    restored_step = int(np.asarray(jax.device_get(restored.step)))
    manager.close()
    if restored_step != step:
        raise OpenPIPipelineError("reloaded train state step does not match saved step")
    manifest = _upload_checkpoint(
        Path(runtime["config"].checkpoint_dir), args.checkpoint_uri
    )
    return {
        "manifest": manifest,
        "params_hash": params_hash,
        "restored_step": restored_step,
        "seconds": save_seconds,
    }


def _training_result(
    args: argparse.Namespace, evidence: Mapping[str, Any]
) -> dict[str, object]:
    dataset = evidence["dataset"]
    return {
        "schema": "npa.workbench.openpi.pi05-training.v1",
        "status": "passed",
        "mode": "train",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "ref": SOURCE_REF,
            "license": SOURCE_LICENSE,
            **evidence["build"],
        },
        "redistribution": _redistribution_evidence(trained_checkpoint=True),
        "runtime_image": args.runtime_image,
        "base_checkpoint": _training_base_evidence(args, evidence["base"]),
        "dataset": _training_dataset_evidence(args, dataset, evidence["train_count"]),
        "optimization": _training_optimization_evidence(args, evidence),
        "checkpoint": _training_checkpoint_evidence(args, evidence["saved"]),
        "timings_seconds": evidence["timings"],
        "memory": evidence["memory"],
        "hardware": evidence["hardware"],
        "artifact_access": _public_artifact_access(),
        "limitations": [
            "tiny_data_optimizer_smoke_is_not_convergence",
            "no_physical_franka_task_success_claim",
        ],
    }


def _training_base_evidence(
    args: argparse.Namespace, base: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "uri": args.base_checkpoint_uri,
        "weight_loader": "upstream CheckpointWeightLoader",
        "provenance": base["provenance"],
        "downloaded_payload_binding": base["binding"],
        "fetch_seconds": round(base["fetch_seconds"], 3),
        "weights_baked": False,
    }


def _training_dataset_evidence(
    args: argparse.Namespace, manifest: Mapping[str, Any], train_count: int
) -> dict[str, Any]:
    return {
        "uri": args.dataset_uri,
        "manifest_uri": args.dataset_manifest_uri,
        "archive_sha256": str(manifest["archive_sha256"]),
        "train_sample_count": train_count,
        "heldout_sample_count": int(manifest["splits"]["heldout"]["count"]),
        "split_disjoint": manifest["split_isolation"]["disjoint"],
    }


def _training_optimization_evidence(
    args: argparse.Namespace, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "implementation": "pinned upstream scripts/train.py:init_train_state+train_step",
        "parameterization": "pi0.5_polaris_compatible_lora",
        "forward": True,
        "backward": True,
        "optimizer": "upstream optax AdamW",
        "optimizer_steps": args.train_steps,
        "metrics": evidence["optimized"]["metrics"],
        "all_metrics_finite": True,
        "trainable_state_sha256_before": evidence["initialized"]["before_hash"],
        "trainable_state_sha256_after": evidence["update"]["after_hash"],
        "trainable_state_changed": True,
        "trainable_update_l2": evidence["update"]["update_l2"],
    }


def _training_checkpoint_evidence(
    args: argparse.Namespace, saved: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = saved["manifest"]
    return {
        "uri": args.checkpoint_uri.rstrip("/") + "/",
        "manifest_uri": args.checkpoint_uri.rstrip("/") + "/manifest.json",
        "content_manifest_sha256": str(manifest["content_manifest_sha256"]),
        "file_count": manifest["file_count"],
        "total_size_bytes": manifest["total_size_bytes"],
        "saved_step": saved["restored_step"],
        "reloadable_params_sha256": saved["params_hash"],
        "reloaded_train_state_step": saved["restored_step"],
        "reload_passed": True,
    }


def _train(args: argparse.Namespace) -> int:
    _validate_runtime_image(args.runtime_image)
    _set_runtime_cache(args.work_dir, "train")
    repo_root = Path(args.repo_root)
    build = _source_build_evidence(repo_root)
    arrays, dataset_manifest = _load_dataset(
        args.dataset_uri, args.dataset_manifest_uri
    )
    import jax
    from openpi.shared import download

    started = time.perf_counter()
    hardware = _hardware_evidence(
        expected_gpu_type=args.expected_gpu_type,
        expected_gpu_count=args.expected_gpu_count,
        expected_compute_capability=args.expected_compute_capability,
    )
    hardware["topology"] = _single_host_runtime_evidence(jax)
    train_count = _validate_training_request(args, dataset_manifest, len(jax.devices()))
    base = _training_base_checkpoint(args, download)
    runtime = _training_data_runtime(args, arrays, base["dir"], repo_root)
    initialized = _initialize_training_state(args, runtime)
    optimized = _run_training_steps(args, runtime, initialized)
    update = _verify_training_update(args, runtime, initialized, optimized["state"])
    saved = _save_training_state(args, runtime, optimized["state"], update)
    evidence = _training_evidence(
        build,
        base,
        dataset_manifest,
        train_count,
        initialized,
        optimized,
        update,
        saved,
        hardware,
        _memory_stats(jax.devices()[0]),
        started,
    )
    return _publish_training_result(args, _training_result(args, evidence))


def _evaluate(args: argparse.Namespace) -> int:
    _validate_runtime_image(args.runtime_image)
    _set_runtime_cache(args.work_dir, "evaluate")
    repo_root = Path(args.repo_root)
    build = _source_build_evidence(repo_root)
    arrays, dataset_manifest = _load_dataset(
        args.dataset_uri, args.dataset_manifest_uri
    )
    train_artifact = _read_json_uri(args.training_artifact_uri)
    if train_artifact.get("schema") != "npa.workbench.openpi.pi05-training.v1":
        raise OpenPIPipelineError("evaluation input is not an OpenPI training artifact")
    checkpoint_info = train_artifact.get("checkpoint")
    if not isinstance(checkpoint_info, Mapping):
        raise OpenPIPipelineError("training artifact has no checkpoint lineage")
    if checkpoint_info.get("uri") != args.checkpoint_uri.rstrip("/") + "/":
        raise OpenPIPipelineError(
            "evaluation checkpoint URI differs from training output"
        )
    train_ids = set(dataset_manifest["splits"]["train"]["sample_ids"])  # type: ignore[index]
    heldout_ids = set(dataset_manifest["splits"]["heldout"]["sample_ids"])  # type: ignore[index]
    if train_ids & heldout_ids:
        raise OpenPIPipelineError("held-out evaluation sample IDs overlap training")

    import jax
    import numpy as np
    from openpi.models import model as openpi_model
    from openpi.policies import policy_config
    from openpi.training import sharding as openpi_sharding

    hardware = _hardware_evidence(
        expected_gpu_type=args.expected_gpu_type,
        expected_gpu_count=args.expected_gpu_count,
        expected_compute_capability=args.expected_compute_capability,
    )
    hardware["topology"] = _single_host_runtime_evidence(jax)
    heldout_count = len(heldout_ids)
    if args.batch_size % len(jax.devices()) != 0:
        raise OpenPIPipelineError(
            "evaluation batch size must be divisible by the allocated GPU count"
        )
    if heldout_count % args.batch_size != 0:
        raise OpenPIPipelineError(
            "held-out sample count must be divisible by evaluation batch size"
        )
    if args.fsdp_devices < 1 or len(jax.devices()) % args.fsdp_devices != 0:
        raise OpenPIPipelineError(
            "FSDP device count must be positive and divide the allocated GPU count"
        )
    started = time.perf_counter()
    work_root = Path(args.work_dir) / f"eval-{os.getpid()}"
    work_root.mkdir(parents=True, exist_ok=False)
    checkpoint_root = work_root / "checkpoint"
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    checkpoint_manifest = _download_checkpoint(args.checkpoint_uri, checkpoint_root)
    expected_manifest_hash = checkpoint_info.get("content_manifest_sha256")
    if checkpoint_manifest.get("content_manifest_sha256") != expected_manifest_hash:
        raise OpenPIPipelineError(
            "evaluation did not consume the exact trained checkpoint"
        )
    step = int(checkpoint_info.get("saved_step", -1))
    step_dir = checkpoint_root / str(step)
    if not step_dir.is_dir():
        raise OpenPIPipelineError(f"trained checkpoint step {step} is missing")
    config, data_config = _training_configuration(
        arrays=arrays,
        checkpoint_base_dir=work_root / "unused",
        train_steps=max(1, step),
        batch_size=args.batch_size,
        seed=args.seed,
        # The loader is not invoked during evaluation: the exact trained
        # checkpoint supplies model parameters. Keep the public base URI in the
        # reconstructed config for complete lineage.
        base_checkpoint_location=args.base_checkpoint_uri,
        fsdp_devices=args.fsdp_devices,
    )
    mesh = openpi_sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    heldout_loader = _make_data_loader(
        arrays=arrays,
        split="heldout",
        config=config,
        data_config=data_config,
        sharding=data_sharding,
        batch_size=args.batch_size,
        seed=args.seed,
        num_batches=heldout_count // args.batch_size,
    )
    restored_params = openpi_model.restore_params(step_dir / "params")
    model = config.model.load(restored_params)
    model.eval()
    losses: list[float] = []
    with openpi_sharding.set_mesh(mesh):
        for index, (observation, target_actions) in enumerate(heldout_loader):
            chunked = model.compute_loss(
                jax.random.fold_in(jax.random.key(args.seed), index),
                observation,
                target_actions,
                train=False,
            )
            values = (
                np.asarray(jax.device_get(chunked))
                .reshape(args.batch_size, -1)
                .mean(axis=1)
            )
            if not np.isfinite(values).all():
                raise OpenPIPipelineError("held-out OpenPI model loss is non-finite")
            losses.extend(float(value) for value in values)
    if len(losses) != heldout_count:
        raise OpenPIPipelineError("held-out model loss sample count is incorrect")
    del model, restored_params
    policy = policy_config.create_trained_policy(config, step_dir)
    if heldout_count < 2:
        raise OpenPIPipelineError(
            "wrist sensitivity requires at least two disjoint held-out samples"
        )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    trajectory_evidence: dict[str, object] | None = None
    for index in range(heldout_count):
        raw = _ArrayDataset(arrays, "heldout")[index]
        observation = {
            "observation/exterior_image_1_left": raw["observation"][
                "exterior_image_1_left"
            ],
            "observation/wrist_image_left": raw["observation"]["wrist_image_left"],
            "observation/joint_position": raw["observation"]["joint_position"],
            "observation/gripper_position": raw["observation"]["gripper_position"],
            "prompt": raw["prompt"],
        }
        prediction = np.asarray(policy.infer(observation)["actions"])
        evidence = validate_actions(prediction, label=f"heldout[{index}]")
        if trajectory_evidence is None:
            trajectory_evidence = {
                **evidence,
                "first_five_targets": prediction[:5].tolist(),
            }
        predictions.append(prediction)
        targets.append(np.asarray(raw["actions"], dtype=np.float64))
    sensitivity_raw = _ArrayDataset(arrays, "heldout")[1]
    sensitivity_observation = {
        "observation/exterior_image_1_left": sensitivity_raw["observation"][
            "exterior_image_1_left"
        ],
        "observation/wrist_image_left": sensitivity_raw["observation"][
            "wrist_image_left"
        ],
        "observation/joint_position": sensitivity_raw["observation"]["joint_position"],
        "observation/gripper_position": sensitivity_raw["observation"][
            "gripper_position"
        ],
        "prompt": sensitivity_raw["prompt"],
    }
    frozen_wrist = _ArrayDataset(arrays, "heldout")[0]["observation"][
        "wrist_image_left"
    ]
    wrist_sensitivity = _paired_wrist_sensitivity(
        policy,
        fresh_observation=sensitivity_observation,
        frozen_wrist=frozen_wrist,
        action_horizon=config.model.action_horizon,
        action_dim=config.model.action_dim,
        seed=args.seed,
    )
    predicted = np.stack(predictions)
    target = np.stack(targets)
    error = predicted - target
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error**2))
    per_dimension_mae = np.mean(np.abs(error), axis=(0, 1)).tolist()
    if not all(
        np.isfinite(value)
        for value in [mae, mse, *losses, *[float(v) for v in per_dimension_mae]]
    ):
        raise OpenPIPipelineError("held-out quantitative metrics are non-finite")
    result: dict[str, object] = {
        "schema": "npa.workbench.openpi.pi05-heldout-evaluation.v1",
        "status": "passed",
        "mode": "evaluate",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "ref": SOURCE_REF,
            "license": SOURCE_LICENSE,
            **build,
        },
        "redistribution": _redistribution_evidence(trained_checkpoint=True),
        "runtime_image": args.runtime_image,
        "lineage": {
            "training_artifact_uri": args.training_artifact_uri,
            "dataset_archive_sha256": dataset_manifest["archive_sha256"],
            "trained_checkpoint_uri": args.checkpoint_uri.rstrip("/") + "/",
            "trained_checkpoint_manifest_sha256": checkpoint_manifest[
                "content_manifest_sha256"
            ],
            "saved_step": step,
            "exact_training_checkpoint_consumed": True,
        },
        "split_isolation": {
            "train_sample_count": len(train_ids),
            "heldout_sample_count": heldout_count,
            "sample_id_intersection": sorted(train_ids & heldout_ids),
            "disjoint": True,
            "normalization_source": "training_split_only",
        },
        "schema_checks": {
            "two_uint8_224x224_rgb_cameras": True,
            "seven_float32_franka_joints": True,
            "one_float32_parallel_jaw_gripper": True,
            "absolute_action_target_shape": [ACTION_HORIZON, ACTION_DIM],
            "all_passed": True,
        },
        "metrics": {
            "definition": {
                "model_loss": "mean upstream OpenPI flow-matching compute_loss(train=False) over held-out samples",
                "action_mae": "mean absolute error between policy trajectories and held-out absolute joint/gripper targets",
                "action_mse": "mean squared error between policy trajectories and held-out absolute joint/gripper targets",
            },
            "heldout_model_loss_mean": float(np.mean(losses)),
            "heldout_model_loss_per_sample": losses,
            "heldout_action_mae": mae,
            "heldout_action_mse": mse,
            "heldout_action_mae_per_dimension": per_dimension_mae,
            "sample_count": heldout_count,
            "all_finite": True,
        },
        "reloaded_trajectory": trajectory_evidence,
        "wrist_action_sensitivity": wrist_sensitivity,
        "timings_seconds": {"total": round(time.perf_counter() - started, 3)},
        "memory": _memory_stats(jax.devices()[0]),
        "hardware": hardware,
        "artifact_access": {
            "base_checkpoint_source": "public_anonymous_gcs",
            "npa_terms_attestation_required": False,
            "credential_persisted": False,
        },
        "limitations": [
            "offline_heldout_evaluation_is_not_robot_success",
            "tiny_dataset_metrics_are_operational_not_statistical",
            "no_physical_franka_task_success_claim",
        ],
    }
    _write_json_uri(args.output_uri, result)
    _publish_pi05_component(
        args,
        9,
        "pi05_checkpoint_reload",
        "Reloaded the exact selected checkpoint and measured held-out OpenPI outputs.",
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _artifact_preflight(args: argparse.Namespace) -> int:
    """Prove anonymous access and immutable metadata for the public checkpoint."""

    _validate_runtime_image(args.runtime_image)
    if args.checkpoint_uri != DEFAULT_CHECKPOINT_URI:
        raise OpenPIPipelineError("artifact preflight requires the pinned Polaris URI")
    build = _source_build_evidence(Path(args.repo_root))
    provenance = _checkpoint_provenance(args.checkpoint_uri)
    result: dict[str, object] = {
        "schema": "npa.workbench.openpi.public-artifact-preflight.v1",
        "status": "passed",
        "execution_scope": "live_kubernetes_workload",
        "runtime_image": args.runtime_image,
        "source": build,
        "checkpoint": {
            "uri": args.checkpoint_uri,
            "access": "public_anonymous_gcs",
            "provenance": provenance,
        },
        "required_runtime_payloads": {
            "tokenizer": provenance["required_tokenizer"],
        },
        "npa_terms_attestation_required": False,
        "credential_required": False,
        "credential_persisted": False,
    }
    _write_json_uri(args.output_uri, result)
    _publish_pi05_component(
        args,
        4,
        "pi05_public_artifact_preflight",
        "Proved anonymous Polaris payload access and generation/checksum provenance.",
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("prepare-data")
    data.add_argument("--dataset-uri", required=True)
    data.add_argument("--manifest-uri", required=True)
    data.add_argument("--train-samples", type=int, default=4)
    data.add_argument("--heldout-samples", type=int, default=2)
    data.add_argument("--seed", type=int, default=20_260_815)
    data.set_defaults(func=_prepare_data)

    preflight = commands.add_parser("artifact-preflight")
    preflight.add_argument("--output-uri", required=True)
    preflight.add_argument("--runtime-image", required=True)
    preflight.add_argument("--repo-root", default="/opt/byof")
    preflight.add_argument("--checkpoint-uri", default=DEFAULT_CHECKPOINT_URI)
    preflight.add_argument("--component-root-uri", default="")
    preflight.set_defaults(func=_artifact_preflight)

    def add_runtime_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--output-uri", required=True)
        command.add_argument("--runtime-image", required=True)
        command.add_argument("--repo-root", default="/opt/byof")
        command.add_argument("--expected-gpu-type", default="")
        command.add_argument("--expected-gpu-count", type=int, default=1)
        command.add_argument("--expected-compute-capability", default="")
        command.add_argument("--component-root-uri", default="")

    direct = commands.add_parser("direct")
    add_runtime_args(direct)
    direct.add_argument("--checkpoint-uri", default=DEFAULT_CHECKPOINT_URI)
    direct.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    direct.add_argument("--work-dir", default="/workspace/openpi-four-mode")
    direct.set_defaults(func=_direct)

    train = commands.add_parser("train")
    add_runtime_args(train)
    train.add_argument("--dataset-uri", required=True)
    train.add_argument("--dataset-manifest-uri", required=True)
    train.add_argument("--checkpoint-uri", required=True)
    train.add_argument("--base-checkpoint-uri", default=DEFAULT_CHECKPOINT_URI)
    train.add_argument("--train-steps", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--fsdp-devices", type=int, default=1)
    train.add_argument("--seed", type=int, default=20_260_815)
    train.add_argument("--work-dir", default="/workspace/openpi-four-mode")
    train.set_defaults(func=_train)

    evaluate = commands.add_parser("evaluate")
    add_runtime_args(evaluate)
    evaluate.add_argument("--dataset-uri", required=True)
    evaluate.add_argument("--dataset-manifest-uri", required=True)
    evaluate.add_argument("--checkpoint-uri", required=True)
    evaluate.add_argument("--training-artifact-uri", required=True)
    evaluate.add_argument("--base-checkpoint-uri", default=DEFAULT_CHECKPOINT_URI)
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--fsdp-devices", type=int, default=1)
    evaluate.add_argument("--seed", type=int, default=20_260_816)
    evaluate.add_argument("--work-dir", default="/workspace/openpi-four-mode")
    evaluate.set_defaults(func=_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
