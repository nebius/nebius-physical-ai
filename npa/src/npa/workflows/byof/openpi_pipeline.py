"""Real OpenPI pi0.5 four-mode workflow stages.

The module deliberately imports OpenPI/JAX only *after* the scoped Gemma terms
gate.  It supplies a deterministic, tiny Franka data adapter, then delegates
model construction, forward/backward/optimizer work, checkpointing, policy
inference, and loss evaluation to the pinned upstream OpenPI implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode, urlparse
import urllib.request
import uuid
import zipfile

from npa.workflows.byof.openpi import (
    OPENPI_TERMS_ACCEPTED_VALUE,
    OPENPI_TERMS_ENV,
)

SOURCE_REF = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
SOURCE_REPOSITORY = "https://github.com/Physical-Intelligence/openpi"
SOURCE_LICENSE = "Apache-2.0"
DEFAULT_CONFIG_NAME = "pi05_droid_jointpos_polaris"
DEFAULT_CHECKPOINT_URI = (
    "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris"
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


def _terms_refusal() -> dict[str, object]:
    return {
        "schema": "npa.workbench.openpi.terms-gate.v1",
        "status": "refused",
        "exit_code": 64,
        "checkpoint_fetch_started": False,
        "model_import_started": False,
        "required_env": OPENPI_TERMS_ENV,
        "accepted_value": OPENPI_TERMS_ACCEPTED_VALUE,
        "acceptance_persisted": False,
    }


def _terms_diagnostic_uri(
    output_uri: str,
    *,
    diagnostic_root_uri: str,
    stage: str,
    attempt_id: str,
) -> str:
    """Return a sibling, attempt-scoped URI that can never poison success output."""

    safe_stage = re.sub(r"[^a-z0-9-]+", "-", stage.strip().lower()).strip("-")
    if not safe_stage:
        safe_stage = "unknown-stage"
    filename = f"{safe_stage}-{attempt_id}.json"
    root = diagnostic_root_uri.strip()
    if root:
        return f"{root.rstrip('/')}/{filename}"
    parsed = urlparse(output_uri)
    if parsed.scheme == "s3":
        bucket, key = _parse_s3_uri(output_uri)
        parent = key.rsplit("/", 1)[0] if "/" in key else ""
        prefix = f"{parent}/terms-refusals" if parent else "terms-refusals"
        return f"s3://{bucket}/{prefix}/{filename}"
    path = Path(parsed.path if parsed.scheme == "file" else output_uri)
    diagnostic = path.parent / "terms-refusals" / filename
    return f"file://{diagnostic}" if parsed.scheme == "file" else str(diagnostic)


def _write_terms_refusal_diagnostic(
    output_uri: str,
    *,
    diagnostic_root_uri: str = "",
    stage: str,
    attempt_id: str | None = None,
) -> tuple[str, dict[str, object]]:
    """Publish refusal evidence separately while preserving write-once success."""

    resolved_attempt = attempt_id or uuid.uuid4().hex
    diagnostic_uri = _terms_diagnostic_uri(
        output_uri,
        diagnostic_root_uri=diagnostic_root_uri,
        stage=stage,
        attempt_id=resolved_attempt,
    )
    refusal = {
        **_terms_refusal(),
        "attempt_id": resolved_attempt,
        "stage": stage,
        "declared_success_output_uri": output_uri,
        "diagnostic_uri": diagnostic_uri,
    }
    _write_json_uri(diagnostic_uri, refusal)
    return diagnostic_uri, refusal


def _gate_or_exit(
    output_uri: str, *, diagnostic_root_uri: str = "", stage: str
) -> None:
    """Exit 64 before any OpenPI/JAX import or checkpoint access."""

    if os.environ.get(OPENPI_TERMS_ENV) == OPENPI_TERMS_ACCEPTED_VALUE:
        return
    _diagnostic_uri, refusal = _write_terms_refusal_diagnostic(
        output_uri,
        diagnostic_root_uri=diagnostic_root_uri,
        stage=stage,
    )
    print(json.dumps(refusal, sort_keys=True), flush=True)
    raise SystemExit(64)


def _require_parent_acceptance() -> None:
    if os.environ.get(OPENPI_TERMS_ENV) != OPENPI_TERMS_ACCEPTED_VALUE:
        raise OpenPIPipelineError(
            f"{OPENPI_TERMS_ENV}={OPENPI_TERMS_ACCEPTED_VALUE} is required for this accepted run"
        )


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
    if manifest.get("schema") != "npa.workbench.openpi.mini-franka-dataset.v1":
        raise OpenPIPipelineError("unexpected OpenPI miniature dataset schema")
    if manifest.get("archive_sha256") != _sha256_bytes(archive):
        raise OpenPIPipelineError("miniature dataset archive hash mismatch")
    with np.load(io.BytesIO(archive), allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    _validate_dataset_arrays(arrays, manifest)
    return arrays, manifest


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
    if manifest.get("split_isolation") != {
        "sample_id_intersection": id_intersection,
        "sample_hash_intersection": hash_intersection,
        "disjoint": True,
    }:
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


def _checkpoint_provenance(checkpoint_uri: str) -> dict[str, object]:
    parsed = urlparse(checkpoint_uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        return {"scheme": parsed.scheme or "local", "uri": checkpoint_uri}
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    page_token = ""
    records: list[dict[str, Any]] = []
    while True:
        query = {"prefix": prefix, "maxResults": "1000"}
        if page_token:
            query["pageToken"] = page_token
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{parsed.netloc}/o?"
            + urlencode(query)
        )
        with urllib.request.urlopen(url) as response:  # noqa: S310 - pinned public GCS API
            payload = json.load(response)
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
    return {
        "scheme": "gcs_object_generation_manifest_sha256",
        "sha256": _sha256_bytes(_canonical_json(records)),
        "object_count": len(records),
        "total_size_bytes": sum(int(item["size"]) for item in records),
    }


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
    kinds = [str(getattr(device, "device_kind", "")) for device in devices]
    if expected_gpu_type and any(
        expected_gpu_type.upper() not in item.upper() for item in kinds
    ):
        raise OpenPIPipelineError(
            f"expected GPU type {expected_gpu_type!r}, got {kinds!r}"
        )
    capabilities: list[str] = []
    for device in devices:
        raw = getattr(device, "compute_capability", "")
        raw = raw() if callable(raw) else raw
        major = getattr(raw, "major", None)
        minor = getattr(raw, "minor", None)
        if major is not None and minor is not None:
            text = f"{int(major)}.{int(minor)}"
        else:
            match = re.search(r"(\d{1,2})[.,](\d)", str(raw))
            if match:
                text = f"{int(match.group(1))}.{int(match.group(2))}"
            else:
                compact = re.search(r"(?:sm_)?(\d{2,3})", str(raw), re.IGNORECASE)
                if not compact:
                    raise OpenPIPipelineError(
                        f"cannot parse compute capability {raw!r}"
                    )
                number = int(compact.group(1))
                text = f"{number // 10}.{number % 10}"
        capabilities.append(text)
    if expected_compute_capability and any(
        value != expected_compute_capability for value in capabilities
    ):
        raise OpenPIPipelineError(
            f"expected compute capability {expected_compute_capability}, got {capabilities}"
        )
    sm100_probe: dict[str, object] = {"required": expected_compute_capability == "10.0"}
    if sm100_probe["required"]:
        probe = Path("/usr/local/bin/npa-openpi-sm100-probe")
        if not probe.is_file():
            raise OpenPIPipelineError("compiled SM100 CUDA probe is missing")
        output = subprocess.run(
            [str(probe)], check=True, capture_output=True, text=True
        ).stdout.strip()
        elf = subprocess.run(
            ["cuobjdump", "--list-elf", str(probe)],
            check=True,
            capture_output=True,
            text=True,
        )
        elf_text = "\n".join((elf.stdout, elf.stderr)).strip()
        if "sm_100" not in elf_text.lower():
            raise OpenPIPipelineError("CUDA probe has no sm_100 ELF")
        sm100_probe.update(
            {"passed": True, "output": output, "elf_contains_sm100": True}
        )
    nvidia_smi = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    if len(nvidia_smi) != expected_gpu_count:
        raise OpenPIPipelineError("nvidia-smi visible GPU count disagrees with JAX")
    return {
        "gpu_count_allocated": len(devices),
        "device_kinds": kinds,
        "compute_capabilities": capabilities,
        "nvidia_smi": nvidia_smi,
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "xla_platform_version": str(jax_backend.get_backend().platform_version),
        "sm100_probe": sm100_probe,
    }


def _direct(args: argparse.Namespace) -> int:
    _gate_or_exit(
        args.output_uri,
        diagnostic_root_uri=args.terms_diagnostic_root_uri,
        stage="direct",
    )
    _validate_runtime_image(args.runtime_image)
    _set_runtime_cache(args.work_dir, "direct")
    repo_root = Path(args.repo_root)
    build = _source_build_evidence(repo_root)

    # All model/runtime imports stay below the fail-closed terms gate.
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
    provenance = _checkpoint_provenance(args.checkpoint_uri)
    fetch_started = time.perf_counter()
    checkpoint_dir = Path(download.maybe_download(args.checkpoint_uri, token="anon"))
    checkpoint_seconds = time.perf_counter() - fetch_started
    config = openpi_config.get_config(args.config_name)
    if config.model.action_horizon != ACTION_HORIZON:
        raise OpenPIPipelineError("unexpected Polaris action horizon")
    load_started = time.perf_counter()
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    load_seconds = time.perf_counter() - load_started
    observation = _observation()
    infer_started = time.perf_counter()
    response = policy.infer(observation)
    infer_seconds = time.perf_counter() - infer_started
    actions = np.asarray(response["actions"])
    trajectory = validate_actions(actions, label="direct")
    result: dict[str, object] = {
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
            "provenance": provenance,
            "fetch_seconds": round(checkpoint_seconds, 3),
            "weights_baked": False,
        },
        "terms": {"forwarded": True, "persisted": False, "scope": "this_run_only"},
        "observation_schema": {
            "exterior_image": {"shape": list(IMAGE_SHAPE), "dtype": "uint8"},
            "wrist_image": {"shape": list(IMAGE_SHAPE), "dtype": "uint8"},
            "joint_position": {"shape": [JOINT_DIM], "dtype": "float32"},
            "gripper_position": {"shape": [1], "dtype": "float32"},
            "prompt": PROMPT,
        },
        "trajectory": {
            **trajectory,
            "inference_ms": round(infer_seconds * 1000, 3),
            "first_five_targets": actions[:5].tolist(),
            "action_semantics": "joint_position_targets_dims_0_6_radians;parallel_jaw_gripper_dim_7",
            "consumer_guidance": "execute_about_5_targets_at_15_hz_then_requery",
        },
        "policy_load_seconds": round(load_seconds, 3),
        "memory": _memory_stats(jax.devices()[0]),
        "hardware": hardware,
        "software": {
            "numpy": np.__version__,
            "openpi_distribution": "editable_pinned_source",
        },
        "limitations": ["no_physical_franka_task_success_claim"],
    }
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


def _train(args: argparse.Namespace) -> int:
    _gate_or_exit(
        args.output_uri,
        diagnostic_root_uri=args.terms_diagnostic_root_uri,
        stage="train",
    )
    _validate_runtime_image(args.runtime_image)
    _set_runtime_cache(args.work_dir, "train")
    repo_root = Path(args.repo_root)
    build = _source_build_evidence(repo_root)
    arrays, dataset_manifest = _load_dataset(
        args.dataset_uri, args.dataset_manifest_uri
    )
    train_count = int(dataset_manifest["splits"]["train"]["count"])  # type: ignore[index]
    if args.batch_size > train_count:
        raise OpenPIPipelineError("batch size exceeds miniature training split")
    if args.train_steps < 1:
        raise OpenPIPipelineError("at least one optimizer step is required")

    # OpenPI/JAX imports remain below the terms gate.
    import functools
    import jax
    import numpy as np
    from openpi.models import model as openpi_model
    from openpi.shared import download
    from openpi.training import checkpoints
    from openpi.training import sharding as openpi_sharding

    hardware = _hardware_evidence(
        expected_gpu_type=args.expected_gpu_type,
        expected_gpu_count=args.expected_gpu_count,
        expected_compute_capability=args.expected_compute_capability,
    )
    if args.batch_size % len(jax.devices()) != 0:
        raise OpenPIPipelineError(
            "training batch size must be divisible by the allocated GPU count"
        )
    if args.fsdp_devices < 1 or len(jax.devices()) % args.fsdp_devices != 0:
        raise OpenPIPipelineError(
            "FSDP device count must be positive and divide the allocated GPU count"
        )
    started = time.perf_counter()
    base_provenance = _checkpoint_provenance(args.base_checkpoint_uri)
    base_fetch_started = time.perf_counter()
    base_checkpoint_dir = Path(
        download.maybe_download(args.base_checkpoint_uri, token="anon")
    )
    base_fetch_seconds = time.perf_counter() - base_fetch_started
    work_root = Path(args.work_dir) / f"train-{os.getpid()}"
    work_root.mkdir(parents=True, exist_ok=False)
    config, data_config = _training_configuration(
        arrays=arrays,
        checkpoint_base_dir=work_root / "checkpoints",
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        base_checkpoint_location=str(base_checkpoint_dir),
        fsdp_devices=args.fsdp_devices,
    )
    upstream_train = _load_upstream_train_module(repo_root)
    mesh = openpi_sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(openpi_sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = _make_data_loader(
        arrays=arrays,
        split="train",
        config=config,
        data_config=data_config,
        sharding=data_sharding,
        batch_size=args.batch_size,
        seed=args.seed,
        num_batches=args.train_steps,
    )
    iterator = iter(loader)
    batch = next(iterator)
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=None,
        overwrite=True,
        resume=False,
    )
    if resuming:
        raise OpenPIPipelineError(
            "fresh optimizer smoke unexpectedly entered resume mode"
        )
    rng = jax.random.key(args.seed)
    train_rng, init_rng = jax.random.split(rng)
    init_started = time.perf_counter()
    state, state_sharding = upstream_train.init_train_state(
        config, init_rng, mesh, resume=False
    )
    jax.block_until_ready(state)
    init_seconds = time.perf_counter() - init_started
    before_tree = _tree_to_host(_pure_trainable(state, config.trainable_filter))
    before_hash = _tree_sha256(before_tree)
    train_step = jax.jit(
        functools.partial(upstream_train.train_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    metrics: list[dict[str, float]] = []
    optimize_started = time.perf_counter()
    with openpi_sharding.set_mesh(mesh):
        for step in range(args.train_steps):
            step_rng = _optimizer_step_rng(jax.random, train_rng, step)
            state, info = train_step(step_rng, state, batch)
            jax.block_until_ready(state)
            row = {
                key: float(np.asarray(jax.device_get(value)))
                for key, value in info.items()
            }
            if not all(np.isfinite(value) for value in row.values()):
                raise OpenPIPipelineError(
                    f"non-finite optimizer metrics at step {step}: {row}"
                )
            row["optimizer_step"] = float(step + 1)
            metrics.append(row)
            if step + 1 < args.train_steps:
                batch = next(iterator)
    optimize_seconds = time.perf_counter() - optimize_started
    after_tree = _tree_to_host(_pure_trainable(state, config.trainable_filter))
    after_hash = _tree_sha256(after_tree)
    update_l2 = _tree_update_l2(before_tree, after_tree)
    if before_hash == after_hash or not np.isfinite(update_l2) or update_l2 <= 0.0:
        raise OpenPIPipelineError("optimizer did not change trainable OpenPI state")
    checkpoint_step = int(np.asarray(jax.device_get(state.step)))
    if checkpoint_step != args.train_steps:
        raise OpenPIPipelineError(
            f"optimizer step mismatch: state={checkpoint_step} requested={args.train_steps}"
        )
    save_started = time.perf_counter()
    checkpoints.save_state(manager, state, loader, checkpoint_step)
    manager.wait_until_finished()
    save_seconds = time.perf_counter() - save_started
    step_dir = Path(config.checkpoint_dir) / str(checkpoint_step)
    restored_params = openpi_model.restore_params(step_dir / "params")
    restored_param_hash = _tree_sha256(restored_params)
    restored_state = checkpoints.restore_state(manager, state, loader, checkpoint_step)
    restored_step = int(np.asarray(jax.device_get(restored_state.step)))
    if restored_step != checkpoint_step:
        raise OpenPIPipelineError("reloaded train state step does not match saved step")
    manager.close()
    checkpoint_manifest = _upload_checkpoint(
        Path(config.checkpoint_dir), args.checkpoint_uri
    )
    checkpoint_manifest_hash = str(checkpoint_manifest["content_manifest_sha256"])
    dataset_hash = str(dataset_manifest["archive_sha256"])
    elapsed_seconds = time.perf_counter() - started
    result: dict[str, object] = {
        "schema": "npa.workbench.openpi.pi05-training.v1",
        "status": "passed",
        "mode": "train",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "ref": SOURCE_REF,
            "license": SOURCE_LICENSE,
            **build,
        },
        "redistribution": _redistribution_evidence(trained_checkpoint=True),
        "runtime_image": args.runtime_image,
        "base_checkpoint": {
            "uri": args.base_checkpoint_uri,
            "weight_loader": "upstream CheckpointWeightLoader",
            "provenance": base_provenance,
            "fetch_seconds": round(base_fetch_seconds, 3),
            "weights_baked": False,
        },
        "dataset": {
            "uri": args.dataset_uri,
            "manifest_uri": args.dataset_manifest_uri,
            "archive_sha256": dataset_hash,
            "train_sample_count": train_count,
            "heldout_sample_count": int(
                dataset_manifest["splits"]["heldout"]["count"]  # type: ignore[index]
            ),
            "split_disjoint": dataset_manifest["split_isolation"]["disjoint"],  # type: ignore[index]
        },
        "optimization": {
            "implementation": "pinned upstream scripts/train.py:init_train_state+train_step",
            "parameterization": "pi0.5_polaris_compatible_lora",
            "forward": True,
            "backward": True,
            "optimizer": "upstream optax AdamW",
            "optimizer_steps": args.train_steps,
            "metrics": metrics,
            "all_metrics_finite": True,
            "trainable_state_sha256_before": before_hash,
            "trainable_state_sha256_after": after_hash,
            "trainable_state_changed": True,
            "trainable_update_l2": update_l2,
        },
        "checkpoint": {
            "uri": args.checkpoint_uri.rstrip("/") + "/",
            "manifest_uri": args.checkpoint_uri.rstrip("/") + "/manifest.json",
            "content_manifest_sha256": checkpoint_manifest_hash,
            "file_count": checkpoint_manifest["file_count"],
            "total_size_bytes": checkpoint_manifest["total_size_bytes"],
            "saved_step": checkpoint_step,
            "reloadable_params_sha256": restored_param_hash,
            "reloaded_train_state_step": restored_step,
            "reload_passed": True,
        },
        "timings_seconds": {
            "initialization": round(init_seconds, 3),
            "optimization": round(optimize_seconds, 3),
            "checkpoint_save": round(save_seconds, 3),
            "total": round(elapsed_seconds, 3),
        },
        "memory": _memory_stats(jax.devices()[0]),
        "hardware": hardware,
        "terms": {"forwarded": True, "persisted": False},
        "limitations": [
            "tiny_data_optimizer_smoke_is_not_convergence",
            "no_physical_franka_task_success_claim",
        ],
    }
    _write_json_uri(args.output_uri, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    _gate_or_exit(
        args.output_uri,
        diagnostic_root_uri=args.terms_diagnostic_root_uri,
        stage="evaluate",
    )
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
        "timings_seconds": {"total": round(time.perf_counter() - started, 3)},
        "memory": _memory_stats(jax.devices()[0]),
        "hardware": hardware,
        "terms": {"forwarded": True, "persisted": False},
        "limitations": [
            "offline_heldout_evaluation_is_not_robot_success",
            "tiny_dataset_metrics_are_operational_not_statistical",
            "no_physical_franka_task_success_claim",
        ],
    }
    _write_json_uri(args.output_uri, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _negative_gate(args: argparse.Namespace) -> int:
    """Live-probe the missing-acceptance child before any accepted checkpoint fetch."""

    _require_parent_acceptance()
    _validate_runtime_image(args.runtime_image)
    child_env = dict(os.environ)
    child_env.pop(OPENPI_TERMS_ENV, None)
    command = [
        sys.executable,
        "-m",
        "npa.workflows.byof.openpi_pipeline",
        "direct",
        "--output-uri",
        args.output_uri,
        "--terms-diagnostic-root-uri",
        args.terms_diagnostic_root_uri,
        "--runtime-image",
        args.runtime_image,
        "--repo-root",
        args.repo_root,
    ]
    completed = subprocess.run(
        command, env=child_env, check=False, capture_output=True, text=True
    )
    try:
        refusal = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise OpenPIPipelineError(
            "negative terms child emitted no valid refusal evidence: "
            f"rc={completed.returncode}; stderr={completed.stderr[-2000:]}"
        ) from exc
    diagnostic_uri = str(refusal.get("diagnostic_uri", ""))
    if completed.returncode != 64 or not diagnostic_uri:
        raise OpenPIPipelineError(
            f"negative terms child did not fail closed: rc={completed.returncode}"
        )
    persisted_refusal = _read_json_uri(diagnostic_uri)
    if persisted_refusal != refusal:
        raise OpenPIPipelineError(
            "negative terms child diagnostic readback differs from emitted evidence"
        )
    if _uri_exists(args.output_uri):
        raise OpenPIPipelineError(
            "negative terms child poisoned its declared success output URI"
        )
    if any(refusal.get(key) != value for key, value in _terms_refusal().items()):
        raise OpenPIPipelineError(f"unexpected negative terms artifact: {refusal}")
    result: dict[str, object] = {
        **_terms_refusal(),
        "schema": "npa.workbench.openpi.live-negative-terms-gate.v1",
        "status": "passed",
        "tested_child_status": "refused",
        "tested_child_exit_code": 64,
        "execution_scope": "live_kubernetes_workload",
        "runtime_image": args.runtime_image,
        "accepted_checkpoint_fetch_started_after_probe": False,
        "accepted_retry_same_logical_output_uri": True,
        # Preserve attribution in the accepted parent while the complete refusal
        # remains independently durable at its attempt-scoped diagnostic URI.
        "tested_child_refusal": {
            **refusal,
            "diagnostic_persistence": "separate_attempt_scoped_uri",
            "declared_success_output_uri_untouched": True,
        },
    }
    _write_json_uri(args.output_uri, result)
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

    negative = commands.add_parser("negative-gate")
    negative.add_argument("--output-uri", required=True)
    negative.add_argument("--terms-diagnostic-root-uri", default="")
    negative.add_argument("--runtime-image", required=True)
    negative.add_argument("--repo-root", default="/opt/byof")
    negative.set_defaults(func=_negative_gate)

    def add_runtime_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--output-uri", required=True)
        command.add_argument("--terms-diagnostic-root-uri", default="")
        command.add_argument("--runtime-image", required=True)
        command.add_argument("--repo-root", default="/opt/byof")
        command.add_argument("--expected-gpu-type", default="")
        command.add_argument("--expected-gpu-count", type=int, default=1)
        command.add_argument("--expected-compute-capability", default="")

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
