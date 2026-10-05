"""Run the source-pinned SwitchWorld adapter workflow on real media artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import urlsplit

from npa.clients.storage import StorageClient


SWITCHWORLD_REPOSITORY = "https://github.com/yizhiqianbi/SwitchWorld.git"
SWITCHWORLD_REVISION = "44d21478f2c15dcccf6423983cb78beee695db1c"
CANONICAL_ADAPTER_REPOSITORY = "PencilHu/SwitchWorld"
CANONICAL_ADAPTER_REVISION = "5a01361ae1f9c9b1cfe115bef1c4d0377d922c1f"
ADAPTER_FILES = {
    "joint_high_rank128.pt": (
        "a5a88105cecf09ec54955db46311b631f79aa8040402b4d2f4c8216511e1b743",
        6533976443,
    ),
    "joint_low_rank128.pt": (
        "8bb55401569f7462e9f4cd684dd34fc6a1e475ed004f9fa61ad10b6890d68aee",
        6533975921,
    ),
}
WAN_REPOSITORY = "https://github.com/Wan-Video/Wan2.2.git"
WAN_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
PREPARED_FILES = (
    "reference.png",
    "target.mp4",
    "controls.json",
    "latents/case.pt",
    "conditions/case.pt",
    "contexts.pt",
    "context_before.png",
    "context_after.png",
    "prepared.json",
)
PAIR_FRAME_RATE = 16


class SwitchWorldError(RuntimeError):
    """Raised when a SwitchWorld artifact is incomplete, inconsistent, or unsafe."""


def _runtime_lineage() -> dict[str, Any]:
    """Return pinned source and checkpoint lineage for produced artifacts."""

    from npa.solutions.lingbot_camera import (
        MODEL_ID,
        MODEL_REF,
        SOURCE_REF as LINGBOT_WORLD_REVISION,
        SOURCE_REPO as LINGBOT_WORLD_REPOSITORY,
        TEXT_ENCODER_REF,
    )

    return {
        "switchworld": {
            "repository": SWITCHWORLD_REPOSITORY,
            "revision": SWITCHWORLD_REVISION,
            "license": "Apache-2.0",
        },
        "lingbot_world": {
            "repository": LINGBOT_WORLD_REPOSITORY,
            "revision": LINGBOT_WORLD_REVISION,
            "license": "Apache-2.0",
        },
        "lingbot_base_checkpoint": {
            "repository": MODEL_ID,
            "revision": MODEL_REF,
            "license": "Apache-2.0",
        },
        "umt5_tokenizer": {
            "repository": "google/umt5-xxl",
            "revision": TEXT_ENCODER_REF,
            "license": "Apache-2.0",
        },
        "wan": {
            "repository": WAN_REPOSITORY,
            "revision": WAN_REVISION,
            "license": "Apache-2.0",
        },
    }


def _require_s3_uri(uri: str, *, directory: bool = False) -> str:
    """Validate and normalize one object-storage URI."""

    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise SwitchWorldError("expected an S3 URI without query or fragment")
    if any(char in parsed.netloc for char in "@:%\\") or ".." in parsed.path.split("/"):
        raise SwitchWorldError("invalid S3 URI")
    normalized = uri.rstrip("/")
    if directory:
        return normalized + "/"
    if not parsed.path or parsed.path.endswith("/"):
        raise SwitchWorldError("expected an S3 object URI")
    return normalized


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write canonical JSON to a local output artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _download(storage: StorageClient, uri: str, destination: Path) -> None:
    """Download one declared S3 object into a contained local path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    storage.download_file(_require_s3_uri(uri), str(destination))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise SwitchWorldError(f"downloaded object is empty: {uri}")


def _upload(storage: StorageClient, source: Path, uri: str) -> None:
    """Upload one nonempty produced artifact to its declared S3 destination."""

    if not source.is_file() or source.stat().st_size == 0:
        raise SwitchWorldError(
            f"refusing to upload missing or empty artifact: {source}"
        )
    storage.upload_file(str(source), _require_s3_uri(uri))


def _download_prepared(storage: StorageClient, uri: str, root: Path) -> dict[str, Path]:
    """Download the fixed, content-addressed prepared-case artifact set."""

    prefix = _require_s3_uri(uri, directory=True)
    files = {name: root / name for name in PREPARED_FILES}
    for name, destination in files.items():
        _download(storage, prefix + name, destination)
    return files


def _upload_tree(
    storage: StorageClient, root: Path, uri: str, names: tuple[str, ...]
) -> None:
    """Upload a declared, closed set of files under a run-scoped prefix."""

    prefix = _require_s3_uri(uri, directory=True)
    for name in names:
        _upload(storage, root / name, prefix + name)


def _decoded_frames(path: Path) -> tuple[list[Any], float]:
    """Decode all real video frames and retain their presentation-rate evidence."""

    import av

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            frames = [
                frame.to_ndarray(format="rgb24") for frame in container.decode(stream)
            ]
    except Exception as exc:  # noqa: BLE001 - normalize decoder details
        raise SwitchWorldError(f"could not decode video {path.name}: {exc}") from exc
    if not frames or fps <= 0:
        raise SwitchWorldError(
            f"video has no decodable frames or frame rate: {path.name}"
        )
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise SwitchWorldError(f"video changes frame geometry: {path.name}")
    return frames, fps


def _video_evidence(path: Path) -> dict[str, Any]:
    """Measure a video from decoded pixels rather than producer-reported values."""

    import numpy as np

    frames, fps = _decoded_frames(path)
    sampled = [frame[::8, ::8].astype(np.float32) for frame in frames]
    deltas = [
        float(np.abs(current - prior).mean())
        for prior, current in zip(sampled, sampled[1:], strict=False)
    ]
    return {
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "frame_count": len(frames),
        "width": int(frames[0].shape[1]),
        "height": int(frames[0].shape[0]),
        "fps": fps,
        "max_spatial_std": max(float(frame.std()) for frame in sampled),
        "mean_temporal_abs_delta": sum(deltas) / max(1, len(deltas)),
    }


def _load_controls(path: Path, target_frames: int) -> dict[str, Any]:
    """Load and validate supplied viewpoint controls against real target frames."""

    try:
        controls = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SwitchWorldError("controls must be valid JSON") from exc
    required = {
        "schema",
        "prompt",
        "view_ids",
        "switch_frame",
        "camera_poses",
        "intrinsics",
    }
    if not isinstance(controls, dict) or not required <= controls.keys():
        raise SwitchWorldError(f"controls require {sorted(required)}")
    views = controls["view_ids"]
    poses = controls["camera_poses"]
    intrinsics = controls["intrinsics"]
    if (
        controls["schema"] != "npa.switchworld.controls.v1"
        or not str(controls["prompt"]).strip()
    ):
        raise SwitchWorldError("controls require the SwitchWorld schema and a prompt")
    if (
        not isinstance(views, list)
        or len(views) < 2
        or any(view not in (1, 2) for view in views)
    ):
        raise SwitchWorldError(
            "view_ids must contain at least two first/third-person labels"
        )
    if len(poses) != len(views) or len(intrinsics) != len(views):
        raise SwitchWorldError(
            "each viewpoint control needs pose and intrinsic evidence"
        )
    switch = controls["switch_frame"]
    transitions = [
        index for index in range(1, len(views)) if views[index - 1] != views[index]
    ]
    if type(switch) is not int or switch not in transitions:
        raise SwitchWorldError(
            "switch_frame must identify an actual viewpoint transition"
        )
    unit = controls.get("switch_frame_unit")
    source_switches = controls.get("source_video_switch_frames")
    if unit is None and source_switches is None:
        # Older, frame-for-frame sidecars have no causal resampling boundary.
        # Do not infer one when their video and conditioning schedules differ.
        if target_frames != len(views):
            raise SwitchWorldError(
                "controls with a resampled latent schedule require source_video_switch_frames"
            )
        resolved_switches = transitions
        resolved_unit = "video_frame_index"
    elif unit == "latent_frame_index":
        if (
            not isinstance(source_switches, list)
            or len(source_switches) != len(transitions)
            or any(type(frame) is not int for frame in source_switches)
            or any(frame <= 0 or frame >= target_frames for frame in source_switches)
            or source_switches != sorted(set(source_switches))
        ):
            raise SwitchWorldError(
                "latent-frame controls require ordered in-range source video transitions"
            )
        resolved_switches = source_switches
        resolved_unit = "source_video_frame_index"
    else:
        raise SwitchWorldError(
            "switch_frame_unit must be omitted or latent_frame_index"
        )
    primary_index = transitions.index(switch)
    controls["_npa_transition_latent_frames"] = transitions
    controls["_npa_transition_video_frames"] = resolved_switches
    controls["_npa_primary_video_switch_frame"] = resolved_switches[primary_index]
    controls["_npa_transition_video_frame_unit"] = resolved_unit
    return controls


def _validate_native_condition(path: Path, controls: dict[str, Any]) -> None:
    """Verify that the native LingBot condition carries the supplied controls."""

    import torch

    condition = torch.load(path, map_location="cpu", weights_only=True)
    view_ids = condition.get("view_id") if isinstance(condition, dict) else None
    switches = condition.get("switch_mask") if isinstance(condition, dict) else None
    metadata = condition.get("meta") if isinstance(condition, dict) else None
    if view_ids is None or switches is None or not isinstance(metadata, dict):
        raise SwitchWorldError("native condition lacks view_id or switch_mask")
    if metadata.get("condition_schema") != "physicalworld_lingbot_v3_causal_multihot":
        raise SwitchWorldError("native condition is not a strict LingBot v3 sidecar")
    observed_views = [int(value) for value in view_ids.tolist()]
    observed_switches = [int(value) for value in switches.tolist()]
    if observed_views != controls["view_ids"]:
        raise SwitchWorldError(
            "native condition view_ids differ from supplied controls"
        )
    expected_transitions = [
        index
        for index in range(1, len(observed_views))
        if observed_views[index - 1] != observed_views[index]
    ]
    marked_transitions = [
        index for index, value in enumerate(observed_switches) if value == 1
    ]
    if marked_transitions != expected_transitions:
        raise SwitchWorldError("native condition switch_mask differs from view changes")
    if int(controls["switch_frame"]) not in marked_transitions:
        raise SwitchWorldError("native condition does not mark the supplied transition")

    metadata_switches = metadata.get("switch_frames")
    expected_video_switches = controls.get("_npa_transition_video_frames")
    if expected_video_switches is None:
        expected_video_switches = controls.get("source_video_switch_frames")
    if expected_video_switches is not None:
        if metadata_switches != expected_video_switches:
            raise SwitchWorldError(
                "native condition source-video transitions differ from controls"
            )

    camera = condition.get("camera_c2w")
    intrinsics = condition.get("camera_intrinsics")
    if camera is None or intrinsics is None:
        raise SwitchWorldError(
            "native condition lacks camera pose or intrinsic tensors"
        )
    expected_camera = torch.as_tensor(controls["camera_poses"], dtype=camera.dtype)
    expected_intrinsics = torch.as_tensor(
        controls["intrinsics"], dtype=intrinsics.dtype
    )
    if tuple(camera.shape) != tuple(expected_camera.shape) or not torch.allclose(
        camera.cpu(), expected_camera.cpu(), rtol=1e-4, atol=1e-5
    ):
        raise SwitchWorldError("native condition camera poses differ from controls")
    if tuple(intrinsics.shape) != tuple(
        expected_intrinsics.shape
    ) or not torch.allclose(
        intrinsics.cpu(), expected_intrinsics.cpu(), rtol=1e-4, atol=1e-5
    ):
        raise SwitchWorldError("native condition intrinsics differ from controls")


def _validate_native_case(
    latent_path: Path, contexts_path: Path, condition_path: Path
) -> None:
    """Reject incomplete native tensors before a GPU worker starts inference."""

    import torch

    latent = torch.load(latent_path, map_location="cpu", weights_only=True)
    if not isinstance(latent, dict):
        raise SwitchWorldError("native latent cache must be a mapping")
    expected_latent_shapes = {
        "latent": (16, 15, 44, 80),
        "fp": (16, 1, 44, 80),
        "tp": (16, 1, 44, 80),
    }
    for key, shape in expected_latent_shapes.items():
        value = latent.get(key)
        if value is None or tuple(value.shape) != shape:
            raise SwitchWorldError(f"native latent cache has invalid {key} shape")

    condition = torch.load(condition_path, map_location="cpu", weights_only=True)
    if not isinstance(condition, dict):
        raise SwitchWorldError("native condition must be a mapping")
    expected_condition_shapes = {
        "view_id": (15,),
        "switch_mask": (15,),
        "camera_c2w": (15, 4, 4),
        "camera_intrinsics": (15, 4),
        "keyboard_condition": (57, 6),
        "mouse_condition": (57, 2),
    }
    for key, shape in expected_condition_shapes.items():
        value = condition.get(key)
        if value is None or tuple(value.shape) != shape:
            raise SwitchWorldError(f"native condition has invalid {key} shape")
    gravity = condition.get("meta", {}).get("gravity")
    contexts = torch.load(contexts_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(contexts, dict)
        or not isinstance(gravity, (str, int, float))
        or str(gravity) not in contexts
        or not isinstance(contexts[str(gravity)], dict)
        or "context" not in contexts[str(gravity)]
    ):
        raise SwitchWorldError("native prompt contexts lack the case gravity context")


def prepare_case(
    *,
    reference_uri: str,
    target_uri: str,
    controls_uri: str,
    latent_uri: str,
    condition_uri: str,
    contexts_uri: str,
    output_uri: str,
) -> dict[str, Any]:
    """Prepare real anchors, controls, and native tensors for downstream inference.

    Args:
        reference_uri: Exact source image object URI.
        target_uri: Exact held-out target video object URI.
        controls_uri: Camera and viewpoint schedule sidecar URI.
        latent_uri: Native SwitchWorld latent-tensor URI.
        condition_uri: Native SwitchWorld condition-tensor URI.
        contexts_uri: Native SwitchWorld prompt-context tensor URI.
        output_uri: Fresh run-scoped S3 prefix for the prepared case.

    Returns:
        A hash-bound prepared-case manifest.

    Raises:
        SwitchWorldError: Input media, controls, or native tensors are invalid.
    """

    from PIL import Image

    storage = StorageClient.from_environment()
    source_uris = {
        "reference.png": reference_uri,
        "target.mp4": target_uri,
        "controls.json": controls_uri,
        "latents/case.pt": latent_uri,
        "conditions/case.pt": condition_uri,
        "contexts.pt": contexts_uri,
    }
    with tempfile.TemporaryDirectory(prefix="npa-switchworld-prepare-") as temporary:
        root = Path(temporary)
        for name, uri in source_uris.items():
            _download(storage, uri, root / name)
        with Image.open(root / "reference.png") as image:
            normalized_reference = image.convert("RGB")
        normalized_reference.save(root / "reference.png")
        frames, _ = _decoded_frames(root / "target.mp4")
        controls = _load_controls(root / "controls.json", len(frames))
        _validate_native_condition(root / "conditions/case.pt", controls)
        _validate_native_case(
            root / "latents/case.pt",
            root / "contexts.pt",
            root / "conditions/case.pt",
        )
        switch = int(controls["_npa_primary_video_switch_frame"])
        Image.fromarray(frames[max(0, switch - 1)]).save(root / "context_before.png")
        Image.fromarray(frames[switch]).save(root / "context_after.png")
        manifest = _prepared_manifest(root, source_uris, controls)
        _write_json(root / "prepared.json", manifest)
        _upload_tree(storage, root, output_uri, PREPARED_FILES)
        return manifest


def _prepared_manifest(
    root: Path, source_uris: dict[str, str], controls: dict[str, Any]
) -> dict[str, Any]:
    """Build a provenance manifest for a fully decoded native case bundle."""

    return {
        "schema": "npa.switchworld.prepared_case.v1",
        "upstream": {
            "repository": SWITCHWORLD_REPOSITORY,
            "revision": SWITCHWORLD_REVISION,
        },
        "source_uris": source_uris,
        "files": {
            name: {
                "sha256": _sha256(root / name),
                "size_bytes": (root / name).stat().st_size,
            }
            for name in source_uris
        },
        "target": _video_evidence(root / "target.mp4"),
        "controls": {
            "sha256": _sha256(root / "controls.json"),
            "switch_frame": controls["switch_frame"],
            "view_count": len(controls["view_ids"]),
        },
        "transitions": {
            "latent_frame_indices": controls["_npa_transition_latent_frames"],
            "source_video_frame_indices": controls["_npa_transition_video_frames"],
            "primary_latent_frame_index": controls["switch_frame"],
            "primary_source_video_frame_index": controls[
                "_npa_primary_video_switch_frame"
            ],
            "source_video_frame_unit": controls["_npa_transition_video_frame_unit"],
            "comparison_alignment": "decoded_frame_index",
        },
    }


def _checkout_source(root: Path) -> Path:
    """Checkout the immutable upstream source without relying on host paths."""

    checkout = root / "SwitchWorld"
    subprocess.run(
        ["git", "clone", "--filter=blob:none", SWITCHWORLD_REPOSITORY, str(checkout)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--detach", SWITCHWORLD_REVISION],
        check=True,
    )
    resolved = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != SWITCHWORLD_REVISION:
        raise SwitchWorldError(
            f"upstream checkout mismatch: expected {SWITCHWORLD_REVISION}, got {resolved}"
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(checkout),
        ],
        check=True,
    )
    return checkout


def _source_overlay_attention_fallback(checkout: Path) -> dict[str, Any]:
    """Route the one adapter-only attention call through the strict SDPA shim.

    SwitchWorld imports LingBot's optional FlashAttention entry point directly
    in its perspective adapter, bypassing the LingBot model-level compatibility
    import.  This scoped source overlay retains the pinned upstream checkout and
    changes only that call site.  :func:`lingbot_attention` delegates to the
    upstream FlashAttention implementation when it is present; otherwise it
    uses a length-preserving PyTorch SDPA implementation.
    """

    target = checkout / "models" / "lingbot_perspective_molora.py"
    before = target.read_text(encoding="utf-8")
    import_line = "from wan.modules.attention import flash_attention"
    call = "output = flash_attention("
    if before.count(import_line) != 1 or before.count(call) != 1:
        raise SwitchWorldError(
            "pinned SwitchWorld adapter attention call site did not match the "
            "reviewed source"
        )
    after = before.replace(
        import_line, "from npa.workflows.switchworld import lingbot_attention"
    ).replace(call, "output = lingbot_attention(")
    if after == before:
        raise SwitchWorldError("could not apply SwitchWorld attention source overlay")
    target.write_text(after, encoding="utf-8")
    return {
        "schema": "npa.switchworld.source_modification.v1",
        "kind": "adapter_attention_compatibility",
        "target": str(target.relative_to(checkout)),
        "upstream_revision": SWITCHWORLD_REVISION,
        "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "upstream_import": import_line,
        "replacement_import": "from npa.workflows.switchworld import lingbot_attention",
        "behavior": (
            "delegates to pinned LingBot FlashAttention when available; otherwise "
            "uses per-sequence PyTorch SDPA with explicit padding and local masks"
        ),
    }


def _attention_lengths(
    lengths: Any, *, batch: int, maximum: int, name: str
) -> list[int]:
    """Validate one optional native attention-length vector."""

    if lengths is None:
        return [maximum] * batch
    try:
        values = lengths.detach().cpu().tolist()
    except AttributeError:
        values = list(lengths)
    if (
        not isinstance(values, list)
        or len(values) != batch
        or any(
            type(value) is not int or value < 0 or value > maximum for value in values
        )
    ):
        raise SwitchWorldError(
            f"{name} must contain exactly {batch} lengths in the range 0..{maximum}"
        )
    return values


def _local_attention_mask(
    *,
    query_length: int,
    key_length: int,
    causal: bool,
    window_size: tuple[int, int],
    device: Any,
) -> Any | None:
    """Build FlashAttention-compatible causal/local access for one sequence."""

    import torch

    if len(window_size) != 2 or any(
        type(value) is not int or value < -1 for value in window_size
    ):
        raise SwitchWorldError(
            "window_size must contain two integers greater than or equal to -1"
        )
    left, right = window_size
    if not causal and (left, right) == (-1, -1):
        return None
    query_positions = torch.arange(query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    # FlashAttention aligns unequal causal query/key lengths at their right
    # edges.  That is important for image-to-video attention where K can carry
    # an additional prefix.
    center = query_positions[:, None] + (key_length - query_length)
    allowed = torch.ones((query_length, key_length), device=device, dtype=torch.bool)
    if causal:
        allowed &= key_positions[None, :] <= center
    if left != -1:
        allowed &= key_positions[None, :] >= center - left
    if right != -1:
        allowed &= key_positions[None, :] <= center + right
    return allowed


def _lingbot_sdpa_attention(
    q: Any,
    k: Any,
    v: Any,
    q_lens: Any = None,
    k_lens: Any = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: Any = None,
    **_: Any,
) -> Any:
    """Run strict, per-sequence SDPA when optional FlashAttention is absent.

    The upstream generic fallback discards ``q_lens`` and ``k_lens``.  This
    implementation slices each native sequence before calling SDPA, preserves
    output padding as zeros, applies ``q_scale`` and ``softmax_scale``, and
    constructs causal/local masks without exposing invalid padded keys.  The
    accepted ``deterministic`` argument preserves the upstream call signature;
    PyTorch's configured deterministic-kernel policy remains authoritative.
    """

    import torch
    import torch.nn.functional as functional

    if any(tensor.ndim != 4 for tensor in (q, k, v)):
        raise SwitchWorldError(
            "native attention expects [batch, length, heads, channels]"
        )
    batch, query_maximum, query_heads, query_channels = q.shape
    key_batch, key_maximum, key_heads, key_channels = k.shape
    value_batch, value_maximum, value_heads, _ = v.shape
    if (
        key_batch != batch
        or value_batch != batch
        or key_maximum != value_maximum
        or key_heads != value_heads
        or query_channels != key_channels
        or query_heads % key_heads
        or q.device != k.device
        or q.device != v.device
    ):
        raise SwitchWorldError(
            "native attention tensors have incompatible batch, head, or device shapes"
        )
    if dtype is None:
        dtype = torch.bfloat16
    if dtype not in (torch.float16, torch.bfloat16):
        raise SwitchWorldError("native attention dtype must be float16 or bfloat16")
    output_dtype = q.dtype
    compute_dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else dtype
    query_lengths = _attention_lengths(
        q_lens, batch=batch, maximum=query_maximum, name="q_lens"
    )
    key_lengths = _attention_lengths(
        k_lens, batch=batch, maximum=key_maximum, name="k_lens"
    )
    output = torch.zeros(
        (batch, query_maximum, query_heads, v.shape[-1]),
        device=q.device,
        dtype=output_dtype,
    )
    for index, (query_length, key_length) in enumerate(
        zip(query_lengths, key_lengths, strict=True)
    ):
        if query_length == 0:
            continue
        if key_length == 0:
            raise SwitchWorldError("a nonempty attention query sequence requires keys")
        query = q[index, :query_length].transpose(0, 1).unsqueeze(0).to(compute_dtype)
        keys = k[index, :key_length].transpose(0, 1).unsqueeze(0).to(compute_dtype)
        values = v[index, :key_length].transpose(0, 1).unsqueeze(0).to(compute_dtype)
        if q_scale is not None:
            query = query * q_scale
        if query_heads != key_heads:
            repeats = query_heads // key_heads
            keys = keys.repeat_interleave(repeats, dim=1)
            values = values.repeat_interleave(repeats, dim=1)
        mask = _local_attention_mask(
            query_length=query_length,
            key_length=key_length,
            causal=causal,
            window_size=window_size,
            device=q.device,
        )
        attended = functional.scaled_dot_product_attention(
            query,
            keys,
            values,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=softmax_scale,
        )
        output[index, :query_length] = (
            attended.squeeze(0).transpose(0, 1).to(output_dtype)
        )
    # The native flag controls FlashAttention's backward implementation.  This
    # inference fallback deliberately leaves PyTorch's global deterministic
    # algorithm policy untouched.
    del deterministic
    return output


def lingbot_attention(*args: Any, **kwargs: Any) -> Any:
    """Use pinned LingBot FlashAttention when present, else strict PyTorch SDPA."""

    from wan.modules import attention as native_attention

    if (
        native_attention.FLASH_ATTN_2_AVAILABLE
        or native_attention.FLASH_ATTN_3_AVAILABLE
    ):
        return native_attention.flash_attention(*args, **kwargs)
    return _lingbot_sdpa_attention(*args, **kwargs)


def _lingbot_checkpoint(root: Path) -> Path:
    """Materialize the exact LingBot checkpoint layout through its existing runtime fetcher."""

    from npa.solutions.lingbot_camera import _checkpoint

    return _checkpoint(root)


def _native_environment() -> dict[str, str]:
    """Return the minimal environment that exposes the pinned LingBot runtime."""

    byof = "/opt/byof"
    if not Path(byof).is_dir():
        raise SwitchWorldError("the selected LingBot image does not contain /opt/byof")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = byof + os.pathsep + environment.get("PYTHONPATH", "")
    environment["LINGBOT_WORLD_CODE"] = byof
    return environment


def _native_command(
    checkout: Path,
    root: Path,
    model_dir: Path,
    output: Path,
    seed: int,
    adapters: dict[str, Path] | None,
) -> list[str]:
    """Build the native upstream inference argv for baseline or adapter execution."""

    controls = json.loads((root / "controls.json").read_text(encoding="utf-8"))
    command = [
        sys.executable,
        str(checkout / "inference" / "infer_lingbot_switchmolora_case.py"),
        "--case-id",
        "case",
        "--latents-root",
        str(root / "latents"),
        "--conditions-root",
        str(root / "conditions"),
        "--contexts",
        str(root / "contexts.pt"),
        "--model-dir",
        str(model_dir),
        "--reference",
        str(root / "reference.png"),
        "--output",
        str(output),
        "--steps",
        str(controls.get("sampling_steps", 70)),
        "--seed",
        str(seed),
    ]
    if adapters:
        command.extend(
            [
                "--high-adapter",
                str(adapters["joint_high_rank128.pt"]),
                "--low-adapter",
                str(adapters["joint_low_rank128.pt"]),
                "--visual-history-mode",
                "anchors_only",
            ]
        )
    return command


def _stage_output(
    root: Path, output_uri: str, name: str, evidence: dict[str, Any]
) -> dict[str, Any]:
    """Upload one decoded video and its factual manifest."""

    storage = StorageClient.from_environment()
    _write_json(root / f"{name}.json", evidence)
    _upload_tree(storage, root, output_uri, (f"{name}.mp4", f"{name}.json"))
    return evidence


def run_baseline(*, prepared_uri: str, output_uri: str, seed: int) -> dict[str, Any]:
    """Run the native LingBot-World baseline on the prepared real case.

    Args:
        prepared_uri: S3 prefix emitted by :func:`prepare_case`.
        output_uri: Fresh S3 prefix for native baseline outputs.
        seed: Nonnegative native generation seed.

    Returns:
        Independent decoded-media and upstream-command evidence.

    Raises:
        SwitchWorldError: The native generation cannot complete or decode.
    """

    if type(seed) is not int or seed < 0:
        raise SwitchWorldError("seed must be a nonnegative integer")
    storage = StorageClient.from_environment()
    with tempfile.TemporaryDirectory(prefix="npa-switchworld-baseline-") as temporary:
        root = Path(temporary)
        _download_prepared(storage, prepared_uri, root)
        checkout = _checkout_source(root)
        model_dir = _lingbot_checkpoint(root / "model-cache")
        output = root / "baseline.mp4"
        command = _native_command(checkout, root, model_dir, output, seed, None)
        subprocess.run(
            command, check=True, cwd=str(checkout), env=_native_environment()
        )
        evidence = {
            "schema": "npa.switchworld.baseline.v1",
            "mode": "lingbot-world-baseline",
            "lineage": _runtime_lineage(),
            "command": command,
            "prepared_sha256": _sha256(root / "prepared.json"),
            "video": _video_evidence(output),
        }
        return _stage_output(root, output_uri, "baseline", evidence)


def _download_adapters(root: Path) -> dict[str, Path]:
    """Runtime-fetch and hash-verify exactly the two canonical final adapters."""

    from huggingface_hub import hf_hub_download

    resolved: dict[str, Path] = {}
    for name, (digest, size) in ADAPTER_FILES.items():
        path = Path(
            hf_hub_download(
                CANONICAL_ADAPTER_REPOSITORY, name, revision=CANONICAL_ADAPTER_REVISION
            )
        )
        if _sha256(path) != digest or path.stat().st_size != size:
            raise SwitchWorldError(f"canonical adapter digest or size mismatch: {name}")
        target = root / name
        shutil.copyfile(path, target)
        resolved[name] = target
    return resolved


def run_adapter(*, prepared_uri: str, output_uri: str, seed: int) -> dict[str, Any]:
    """Run the native SwitchWorld adapter with canonical runtime-fetched checkpoints.

    Args:
        prepared_uri: S3 prefix emitted by :func:`prepare_case`.
        output_uri: Fresh S3 prefix for adapter outputs.
        seed: Nonnegative native generation seed.

    Returns:
        Adapter lineage, native command, and decoded-media evidence.

    Raises:
        SwitchWorldError: A canonical adapter or native generation is invalid.
    """

    if type(seed) is not int or seed < 0:
        raise SwitchWorldError("seed must be a nonnegative integer")
    storage = StorageClient.from_environment()
    with tempfile.TemporaryDirectory(prefix="npa-switchworld-adapter-") as temporary:
        root = Path(temporary)
        _download_prepared(storage, prepared_uri, root)
        checkout = _checkout_source(root)
        source_modifications = _source_overlay_attention_fallback(checkout)
        adapters = _download_adapters(root)
        model_dir = _lingbot_checkpoint(root / "model-cache")
        output = root / "adapted.mp4"
        command = _native_command(checkout, root, model_dir, output, seed, adapters)
        subprocess.run(
            command, check=True, cwd=str(checkout), env=_native_environment()
        )
        evidence = {
            "schema": "npa.switchworld.adapter.v1",
            "mode": "switchworld-joint-adapter",
            "lineage": _runtime_lineage(),
            "command": command,
            "prepared_sha256": _sha256(root / "prepared.json"),
            "source_modifications": [source_modifications],
            "adapter": {
                "repository": CANONICAL_ADAPTER_REPOSITORY,
                "revision": CANONICAL_ADAPTER_REVISION,
                "license": "Apache-2.0",
                "files": {
                    name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
                    for name, path in adapters.items()
                },
            },
            "video": _video_evidence(output),
        }
        return _stage_output(root, output_uri, "adapted", evidence)


def _run_pair_evaluation(
    checkout: Path, target: Path, prediction: Path, output: Path, label: str
) -> dict[str, Any]:
    """Invoke and independently cross-check the upstream real-frame evaluator."""

    command = [
        sys.executable,
        str(checkout / "inference" / "evaluate_video_pair.py"),
        "--target",
        str(target),
        "--prediction",
        str(prediction),
        "--output-dir",
        str(output),
        "--label",
        label,
    ]
    subprocess.run(command, check=True, cwd=str(checkout), env=_native_environment())
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    # The upstream evaluator decodes with OpenCV. Cross-check its reported values
    # with an independently implemented OpenCV reader, then retain a second PyAV
    # decode as the format-independent real-frame evidence. PyAV and OpenCV can
    # apply different YUV conversion rounding, so their numerical values are not
    # required to be bitwise identical to one another.
    upstream_decoder = _independent_cv2_pixel_measurement(target, prediction)
    decoded = _independent_pixel_measurement(target, prediction)
    report["npa_upstream_cv2_pixel_cross_check"] = _verify_pixel_measurement(
        report, upstream_decoder
    )
    report["npa_independent_decoded_pixel_check"] = _decoded_pixel_evidence(decoded)
    return report


def _independent_pixel_measurement(target: Path, prediction: Path) -> dict[str, Any]:
    """Recompute PSNR and MAE from separately decoded real video pixels.

    This deliberately does not import SwitchWorld's evaluator. It makes the
    acceptance path reject an evaluator report that is disconnected from the
    two MP4 inputs, while leaving SwitchWorld's upstream SSIM calculation as
    the source-of-record value.
    """

    import numpy as np
    from PIL import Image

    target_frames, _ = _decoded_frames(target)
    prediction_frames, _ = _decoded_frames(prediction)
    frame_count = min(len(target_frames), len(prediction_frames))
    if frame_count == 0:
        raise SwitchWorldError("cannot measure empty decoded video inputs")

    per_frame: list[dict[str, float]] = []
    for target_frame, prediction_frame in zip(
        target_frames[:frame_count], prediction_frames[:frame_count], strict=True
    ):
        if prediction_frame.shape != target_frame.shape:
            prediction_frame = np.asarray(
                Image.fromarray(prediction_frame).resize(
                    (target_frame.shape[1], target_frame.shape[0]),
                    resample=Image.Resampling.BILINEAR,
                )
            )
        error = target_frame.astype(np.float64) - prediction_frame.astype(np.float64)
        mse = float(np.mean(error * error))
        per_frame.append(
            {
                "psnr_db": float("inf")
                if mse == 0.0
                else 10.0 * math.log10(255.0**2 / mse),
                "mae": float(np.mean(np.abs(error))),
            }
        )

    def summarize(indices: list[int]) -> dict[str, float | int]:
        """Summarize the independently decoded measurements for frame indices."""

        values = [per_frame[index] for index in indices]
        return {
            "frames": len(values),
            "psnr_db": sum(item["psnr_db"] for item in values) / len(values),
            "mae": sum(item["mae"] for item in values) / len(values),
        }

    future = list(range(1, frame_count)) if frame_count > 1 else [0]
    return {
        "engine": "npa.switchworld.independent_decoded_pixels.v1",
        "target_frame_count": len(target_frames),
        "prediction_frame_count": len(prediction_frames),
        "evaluated_frames": frame_count,
        "per_frame": per_frame,
        "all_frames": summarize(list(range(frame_count))),
        "future_frames_excluding_reference": summarize(future),
    }


def _independent_cv2_pixel_measurement(
    target: Path, prediction: Path
) -> dict[str, Any]:
    """Recompute upstream PSNR/MAE from a separate OpenCV MP4 decode.

    This does not import the upstream evaluator: it owns its decode loop, shape
    alignment, and numerical implementation. The matching decoder validates the
    upstream evaluator's exact numerical contract. The PyAV function above
    remains the separately decoded evidence that the measured artifacts are real
    MP4 frames rather than evaluator-owned arrays or a manifest.
    """

    import cv2
    import numpy as np

    def decode(path: Path) -> list[Any]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise SwitchWorldError(f"OpenCV could not decode video: {path}")
        frames: list[Any] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        if not frames:
            raise SwitchWorldError(f"OpenCV decoded no frames from: {path}")
        return frames

    target_frames = decode(target)
    prediction_frames = decode(prediction)
    frame_count = min(len(target_frames), len(prediction_frames))
    if frame_count == 0:
        raise SwitchWorldError("cannot measure empty OpenCV-decoded video inputs")
    height, width = target_frames[0].shape[:2]
    per_frame: list[dict[str, float]] = []
    for target_frame, prediction_frame in zip(
        target_frames[:frame_count], prediction_frames[:frame_count], strict=True
    ):
        if prediction_frame.shape[:2] != (height, width):
            prediction_frame = cv2.resize(prediction_frame, (width, height))
        error = target_frame.astype(np.float64) - prediction_frame.astype(np.float64)
        mse = float(np.mean(error * error))
        per_frame.append(
            {
                "psnr_db": float("inf")
                if mse == 0.0
                else 10.0 * math.log10(255.0**2 / mse),
                "mae": float(np.mean(np.abs(error))),
            }
        )

    def summarize(indices: list[int]) -> dict[str, float | int]:
        values = [per_frame[index] for index in indices]
        return {
            "frames": len(values),
            "psnr_db": sum(item["psnr_db"] for item in values) / len(values),
            "mae": sum(item["mae"] for item in values) / len(values),
        }

    future = list(range(1, frame_count)) if frame_count > 1 else [0]
    return {
        "engine": "npa.switchworld.independent_cv2_pixels.v1",
        "target_frame_count": len(target_frames),
        "prediction_frame_count": len(prediction_frames),
        "evaluated_frames": frame_count,
        "per_frame": per_frame,
        "all_frames": summarize(list(range(frame_count))),
        "future_frames_excluding_reference": summarize(future),
    }


def _decoded_pixel_evidence(decoded: dict[str, Any]) -> dict[str, Any]:
    """Return validated PyAV-derived evidence from actual target/prediction MP4s."""

    frame_count = decoded.get("evaluated_frames")
    if not isinstance(frame_count, int) or frame_count <= 0:
        raise SwitchWorldError("independent decoded-pixel evidence has no frames")
    if (
        decoded.get("target_frame_count") != frame_count
        or decoded.get("prediction_frame_count") != frame_count
    ):
        raise SwitchWorldError("independent decoded-pixel frame counts disagree")
    per_frame = decoded.get("per_frame")
    if not isinstance(per_frame, list) or len(per_frame) != frame_count:
        raise SwitchWorldError("independent decoded-pixel evidence lacks all frames")
    return {
        "status": "passed",
        "engine": decoded["engine"],
        "checked_metrics": ["psnr_db", "mae"],
        "target_frame_count": decoded["target_frame_count"],
        "prediction_frame_count": decoded["prediction_frame_count"],
        "evaluated_frames": frame_count,
        "all_frames": decoded["all_frames"],
        "future_frames_excluding_reference": decoded[
            "future_frames_excluding_reference"
        ],
    }


def _same_measurement_value(observed: Any, expected: float) -> bool:
    """Return whether one upstream number agrees with the independent decode."""

    if not isinstance(observed, (int, float)):
        return False
    if math.isinf(expected):
        return math.isinf(float(observed)) and (float(observed) > 0) == (expected > 0)
    return math.isclose(float(observed), expected, rel_tol=1e-4, abs_tol=1e-3)


def _verify_pixel_measurement(
    upstream: dict[str, Any], independent: dict[str, Any]
) -> dict[str, Any]:
    """Reject an upstream report that does not match decoded input-frame pixels."""

    frame_count = independent["evaluated_frames"]
    if (
        upstream.get("evaluated_frames") != frame_count
        or upstream.get("target_frames") != frame_count
        or upstream.get("prediction_frames") != frame_count
    ):
        raise SwitchWorldError("upstream evaluator frame counts disagree with decoding")
    per_frame = upstream.get("per_frame")
    if not isinstance(per_frame, list) or len(per_frame) != frame_count:
        raise SwitchWorldError("upstream evaluator lacks decoded per-frame metrics")
    expected_frames = independent["per_frame"]
    for index, (observed, expected) in enumerate(
        zip(per_frame, expected_frames, strict=True)
    ):
        if not isinstance(observed, dict) or any(
            not _same_measurement_value(observed.get(metric), expected[metric])
            for metric in ("psnr_db", "mae")
        ):
            raise SwitchWorldError(
                f"upstream evaluator disagrees with decoded pixels at frame {index}"
            )
    for summary_name in ("all_frames", "future_frames_excluding_reference"):
        observed_summary = upstream.get(summary_name)
        expected_summary = independent[summary_name]
        if (
            not isinstance(observed_summary, dict)
            or observed_summary.get("frames") != expected_summary["frames"]
            or any(
                not _same_measurement_value(
                    observed_summary.get(metric), expected_summary[metric]
                )
                for metric in ("psnr_db", "mae")
            )
        ):
            raise SwitchWorldError(
                f"upstream evaluator {summary_name} disagrees with decoded pixels"
            )
    return {
        "status": "passed",
        "engine": independent["engine"],
        "checked_metrics": ["psnr_db", "mae"],
        "target_frame_count": independent["target_frame_count"],
        "prediction_frame_count": independent["prediction_frame_count"],
        "evaluated_frames": frame_count,
        "all_frames": independent["all_frames"],
        "future_frames_excluding_reference": independent[
            "future_frames_excluding_reference"
        ],
    }


def _switch_window(report: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    """Summarize native per-frame measurements only around the declared transition."""

    frames = report["per_frame"]
    switch = int(controls["_npa_primary_video_switch_frame"])
    if switch >= len(frames):
        raise SwitchWorldError(
            "decoded evaluator output ends before the declared source-video transition"
        )
    start, stop = max(0, switch - 1), min(len(frames), switch + 2)
    values = frames[start:stop]
    return {
        "first_frame": start,
        "last_frame": stop - 1,
        "frames": len(values),
        "psnr_db": sum(item["psnr_db"] for item in values) / len(values),
        "ssim": sum(item["ssim"] for item in values) / len(values),
        "mae": sum(item["mae"] for item in values) / len(values),
    }


def measure(
    *, prepared_uri: str, baseline_uri: str, adapted_uri: str, output_uri: str
) -> dict[str, Any]:
    """Measure frame-decoded target and baseline comparisons without mock backends.

    Args:
        prepared_uri: S3 prefix emitted by :func:`prepare_case`.
        baseline_uri: Exact baseline MP4 emitted by :func:`run_baseline`.
        adapted_uri: Exact adapter MP4 emitted by :func:`run_adapter`.
        output_uri: Fresh S3 prefix for measured reports and comparison media.

    Returns:
        Native real-frame metrics and declared transition-window summaries.

    Raises:
        SwitchWorldError: Upstream evaluation output is missing or inconsistent.
    """

    storage = StorageClient.from_environment()
    with tempfile.TemporaryDirectory(prefix="npa-switchworld-measure-") as temporary:
        root = Path(temporary)
        files = _download_prepared(storage, prepared_uri, root)
        _download(storage, baseline_uri, root / "baseline.mp4")
        _download(storage, adapted_uri, root / "adapted.mp4")
        checkout = _checkout_source(root)
        target_vs_adapter = _run_pair_evaluation(
            checkout,
            files["target.mp4"],
            root / "adapted.mp4",
            root / "target_vs_adapter",
            "SWITCHWORLD",
        )
        baseline_vs_adapter = _run_pair_evaluation(
            checkout,
            root / "baseline.mp4",
            root / "adapted.mp4",
            root / "baseline_vs_adapter",
            "SWITCHWORLD",
        )
        controls = _load_controls(
            files["controls.json"], _video_evidence(files["target.mp4"])["frame_count"]
        )
        report = {
            "schema": "npa.switchworld.real_frame_metrics.v1",
            "engine": "SwitchWorld inference/evaluate_video_pair.py",
            "mock_metrics_excluded": [
                "switch_fidelity",
                "cross_view_geo",
                "identity",
                "action_follow",
                "offscreen_acc",
                "fvd",
            ],
            "target_vs_adapter": {
                "all_frames": target_vs_adapter["all_frames"],
                "future_frames_excluding_reference": target_vs_adapter[
                    "future_frames_excluding_reference"
                ],
                "switch_window": _switch_window(target_vs_adapter, controls),
                "upstream_cv2_pixel_cross_check": target_vs_adapter[
                    "npa_upstream_cv2_pixel_cross_check"
                ],
                "independent_decoded_pixel_check": target_vs_adapter[
                    "npa_independent_decoded_pixel_check"
                ],
            },
            "baseline_vs_adapter": {
                "all_frames": baseline_vs_adapter["all_frames"],
                "switch_window": _switch_window(baseline_vs_adapter, controls),
                "upstream_cv2_pixel_cross_check": baseline_vs_adapter[
                    "npa_upstream_cv2_pixel_cross_check"
                ],
                "independent_decoded_pixel_check": baseline_vs_adapter[
                    "npa_independent_decoded_pixel_check"
                ],
            },
            "decoded_inputs": {
                "target": _video_evidence(files["target.mp4"]),
                "baseline": _video_evidence(root / "baseline.mp4"),
                "adapted": _video_evidence(root / "adapted.mp4"),
            },
        }
        shutil.copyfile(
            root / "target_vs_adapter" / "comparison.mp4",
            root / "target_vs_adapter_comparison.mp4",
        )
        _write_json(root / "metrics.json", report)
        _upload_tree(
            storage,
            root,
            output_uri,
            ("metrics.json", "target_vs_adapter_comparison.mp4"),
        )
        return report


def _pair_filter() -> str:
    """Return the frame-index-aligned filter graph for paired model media."""

    # Target media may retain its 40 fps cache presentation rate while model
    # outputs are emitted at 16 fps.  Pair by decoded frame index, not source PTS.
    return (
        f"[0:v]settb=AVTB,setpts=N/({PAIR_FRAME_RATE}*TB)[left];"
        f"[1:v]settb=AVTB,setpts=N/({PAIR_FRAME_RATE}*TB)[right];"
        "[left][right]hstack=inputs=2,setsar=1[paired]"
    )


def _pair_videos(left: Path, right: Path, output: Path) -> dict[str, Any]:
    """Produce a frame-index-aligned H.264 side-by-side MP4 from decoded media."""

    left_evidence, right_evidence = _video_evidence(left), _video_evidence(right)
    if left_evidence["frame_count"] != right_evidence["frame_count"]:
        raise SwitchWorldError("paired videos must have equal decoded frame counts")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            _pair_filter(),
            "-map",
            "[paired]",
            "-r",
            str(PAIR_FRAME_RATE),
            "-vsync",
            "0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    evidence = _video_evidence(output)
    if evidence["frame_count"] != left_evidence["frame_count"]:
        raise SwitchWorldError("paired MP4 lost decoded frames")
    if evidence["fps"] != PAIR_FRAME_RATE:
        raise SwitchWorldError("paired MP4 did not retain the indexed 16 fps timeline")
    return evidence


def _build_rrd(
    baseline: Path,
    adapted: Path,
    target: Path,
    metrics: dict[str, Any],
    output: Path,
    run_id: str,
) -> dict[str, Any]:
    """Write and independently verify a Rerun recording from decoded pixel arrays."""

    import rerun as rr

    baseline_frames, _ = _decoded_frames(baseline)
    adapted_frames, _ = _decoded_frames(adapted)
    target_frames, _ = _decoded_frames(target)
    count = min(len(baseline_frames), len(adapted_frames), len(target_frames))
    if count == 0:
        raise SwitchWorldError("cannot create an RRD from empty videos")
    if not run_id.strip():
        raise SwitchWorldError("RRD recording requires the workflow run ID")
    rr.init("switchworld", recording_id=f"switchworld-{run_id}")
    rr.save(str(output))
    rr.log(
        "switchworld/provenance",
        rr.TextDocument(
            json.dumps(metrics, sort_keys=True), media_type="application/json"
        ),
    )
    for index in range(count):
        rr.set_time("frame", sequence=index)
        rr.log("switchworld/baseline", rr.Image(baseline_frames[index]))
        rr.log("switchworld/adapted", rr.Image(adapted_frames[index]))
        rr.log("switchworld/target", rr.Image(target_frames[index]))
    rr.disconnect()
    if not output.is_file() or output.stat().st_size == 0:
        raise SwitchWorldError("Rerun did not write a recording")
    verified = subprocess.run(
        [sys.executable, "-m", "rerun", "rrd", "verify", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    if verified.returncode:
        raise SwitchWorldError(f"Rerun verification failed: {verified.stderr[-500:]}")
    entities = (
        "switchworld/provenance",
        "switchworld/baseline",
        "switchworld/adapted",
        "switchworld/target",
    )
    for entity in entities:
        inspected = subprocess.run(
            # Inspect one required entity at a time so each captured CLI result
            # stays bounded to the entity being validated.  The CLI filter is
            # still an independent RRD parser and proves every decoded-media
            # entity exists.
            [
                sys.executable,
                "-m",
                "rerun",
                "rrd",
                "print",
                "--entity",
                entity,
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if inspected.returncode or entity not in inspected.stdout:
            raise SwitchWorldError(
                f"Rerun inspection did not find decoded media entity {entity}"
            )
    return {
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
        "frame_count": count,
        "rerun_verify": "passed",
        "rerun_inspection": "passed",
    }


def visualize(
    *,
    prepared_uri: str,
    baseline_uri: str,
    adapted_uri: str,
    metrics_uri: str,
    output_uri: str,
    run_id: str,
) -> dict[str, Any]:
    """Emit two decoded paired MP4s and a verified Rerun recording.

    Args:
        prepared_uri: S3 prefix emitted by :func:`prepare_case`.
        baseline_uri: Exact baseline MP4 emitted by :func:`run_baseline`.
        adapted_uri: Exact adapter MP4 emitted by :func:`run_adapter`.
        metrics_uri: Exact real-frame metrics JSON emitted by :func:`measure`.
        output_uri: Fresh S3 prefix for paired MP4 and RRD artifacts.
        run_id: NPA workflow run ID used as the RRD recording identity.

    Returns:
        Decoded-media hashes and Rerun verification evidence.

    Raises:
        SwitchWorldError: Pairing or Rerun verification fails.
    """

    storage = StorageClient.from_environment()
    with tempfile.TemporaryDirectory(prefix="npa-switchworld-viz-") as temporary:
        root = Path(temporary)
        files = _download_prepared(storage, prepared_uri, root)
        _download(storage, baseline_uri, root / "baseline.mp4")
        _download(storage, adapted_uri, root / "adapted.mp4")
        _download(storage, metrics_uri, root / "metrics.json")
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        baseline_pair = _pair_videos(
            root / "baseline.mp4",
            root / "adapted.mp4",
            root / "baseline_vs_adapted.mp4",
        )
        target_pair = _pair_videos(
            files["target.mp4"], root / "adapted.mp4", root / "target_vs_adapted.mp4"
        )
        rrd = _build_rrd(
            root / "baseline.mp4",
            root / "adapted.mp4",
            files["target.mp4"],
            metrics,
            root / "switchworld.rrd",
            run_id,
        )
        evidence = {
            "schema": "npa.switchworld.visualization.v1",
            "baseline_vs_adapted": baseline_pair,
            "target_vs_adapted": target_pair,
            "rrd": rrd,
        }
        _write_json(root / "visualization.json", evidence)
        _upload_tree(
            storage,
            root,
            output_uri,
            (
                "baseline_vs_adapted.mp4",
                "target_vs_adapted.mp4",
                "switchworld.rrd",
                "visualization.json",
            ),
        )
        return evidence


def _parser() -> argparse.ArgumentParser:
    """Build the narrow stage-command CLI used by the workflow specification."""

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    for option in (
        "reference",
        "target",
        "controls",
        "latent",
        "condition",
        "contexts",
        "output",
    ):
        prepare.add_argument(f"--{option}-uri", required=True)
    baseline = actions.add_parser("baseline")
    adapter = actions.add_parser("adapter")
    for item in (baseline, adapter):
        item.add_argument("--prepared-uri", required=True)
        item.add_argument("--output-uri", required=True)
        item.add_argument("--seed", type=int, required=True)
    measure_parser = actions.add_parser("measure")
    for option in ("prepared", "baseline", "adapted", "output"):
        measure_parser.add_argument(f"--{option}-uri", required=True)
    visualize_parser = actions.add_parser("visualize")
    for option in ("prepared", "baseline", "adapted", "metrics", "output"):
        visualize_parser.add_argument(f"--{option}-uri", required=True)
    visualize_parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch a real SwitchWorld workflow stage and print its factual JSON result."""

    args = _parser().parse_args(argv)
    if args.action == "prepare":
        result = prepare_case(
            reference_uri=args.reference_uri,
            target_uri=args.target_uri,
            controls_uri=args.controls_uri,
            latent_uri=args.latent_uri,
            condition_uri=args.condition_uri,
            contexts_uri=args.contexts_uri,
            output_uri=args.output_uri,
        )
    elif args.action == "baseline":
        result = run_baseline(
            prepared_uri=args.prepared_uri, output_uri=args.output_uri, seed=args.seed
        )
    elif args.action == "adapter":
        result = run_adapter(
            prepared_uri=args.prepared_uri, output_uri=args.output_uri, seed=args.seed
        )
    elif args.action == "measure":
        result = measure(
            prepared_uri=args.prepared_uri,
            baseline_uri=args.baseline_uri,
            adapted_uri=args.adapted_uri,
            output_uri=args.output_uri,
        )
    else:
        result = visualize(
            prepared_uri=args.prepared_uri,
            baseline_uri=args.baseline_uri,
            adapted_uri=args.adapted_uri,
            metrics_uri=args.metrics_uri,
            output_uri=args.output_uri,
            run_id=args.run_id,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
