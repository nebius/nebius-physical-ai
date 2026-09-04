"""Real Cosmos-Transfer2.5 inference runner.

Shared by the sim2real augment stage, the Cosmos synthetic fan-out workflow, and
the ``npa workbench cosmos2 transfer`` CLI so they run the actual world-transfer
model (video-to-video) instead of writing descriptor stubs.

The transfer runtime lives in the ``npa-cosmos2-transfer`` image at
``/opt/cosmos/cosmos-transfer2.5`` (Python 3.10 + torch cu128 + flash-attn in its
own ``.venv``). This module shells out to that venv's ``examples/inference.py`` so
it stays import-safe on the default interpreter (no torch/cuda import here).

Callers that run outside the transfer image (unit tests, CPU hosts) should guard
on :func:`cosmos_transfer_available` and fall back to their descriptor path.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from npa.workbench.cosmos.control_contract import (
    COSMOS_TRANSFER_CHECKPOINTS as CONTROL_MODALITY_MODELS,
    ControlContractError,
    validate_control_request,
)

DEFAULT_REPO = "/opt/cosmos/cosmos-transfer2.5"
# No upstream media is bundled in the redistributable image. Callers must supply
# either an input clip (the preferred path) or an explicit operator-owned spec.
DEFAULT_SPEC = ""

# Control modalities the pinned Cosmos Transfer 2.5 accepts (upstream CONTROL_KEYS).
# Depth is intentionally precomputed-only: NPA neither downloads nor executes
# Video Depth Anything weights.
INPUT_CONTROLS = tuple(CONTROL_MODALITY_MODELS)
INPUT_AUTO_CONTROLS = ("edge", "vis", "seg")
DEFAULT_INPUT_CONTROL = "edge"
# ``seg`` is the only modality whose on-the-fly generator is text-driven: upstream
# feeds ``control_prompt`` to GroundingDINO to pick the objects SAM2 then tracks,
# and defaults it to the first 128 words of the appearance prompt when unset.
CONTROL_PROMPT_MODALITIES = ("seg",)
# Each modality is a separate ControlNet checkpoint, so a seg/depth run downloads
# weights an edge run never touches. Named here for the operator-facing error.
DISABLE_CONTENT_GUARDRAILS_ENV = "NPA_COSMOS_DISABLE_CONTENT_GUARDRAILS"
GUARDRAIL_REPO = "nvidia/Cosmos-Guardrail1"
GUARDRAIL_REVISION = "d6d4bfa899a71454a700907664f3e88f503950cf"
GUARDRAIL_NLTK_MATERIALIZED_DIR = "npa-guardrail-nltk-data"
GUARDRAIL_NLTK_READY_MARKER = ".npa-materialization.json"
GUARDRAIL_NLTK_MANIFEST_SCHEMA = "npa.cosmos.guardrail_nltk.v1"
# Live job 339 reported SUCCEEDED while the spec promised ``manifest.json`` and
# the then-reference-only tool wrote ``index.json`` with a different schema.
# Keep these two artifact contracts named and distinct: the real publisher now
# writes the canonical transfer manifest, while reference augmentation retains
# its frame index. ``test_spec_declared_outputs`` binds workflow declarations to
# the appropriate helper so this cannot regress into another false success.
TRANSFER_MANIFEST_FILENAME = "manifest.json"
TRANSFER_MANIFEST_SCHEMA = "npa.cosmos2.transfer.v1"
TRANSFER_MANIFEST_MODE = "cosmos_transfer2.5_gpu"
TRANSFER_MANIFEST_STATUS = "executed"
# Scheduler-managed augment: each wave attempt publishes only below its opaque
# ``_attempts/<attempt-id>/`` prefix. The leader conditionally replaces the
# canonical manifest (after joining shards for a gang); consumers follow only it.
SHARD_MANIFEST_PREFIX = "manifest-rank-"
SHARD_MANIFEST_SCHEMA = "npa.cosmos2.transfer_shard.v1"
PUBLICATION_CLAIM_STATUS = "publishing"
PUBLICATION_GENERATION_FIELD = "publication_generation"
ATTEMPT_PREFIX = "_attempts"
SCHEDULER_PUBLICATION_IDENTITY_FIELDS = frozenset(
    {
        "attempt_id",
        PUBLICATION_GENERATION_FIELD,
        "logical_publication",
        "logical_wave_id",
        "membership_digest",
        "scheduler_fence_sequence",
        "scheduler_fence_attempt",
        "scheduler_launch_id",
    }
)
AUGMENTED_FRAMES_INDEX = "index.json"
AUGMENTED_FRAMES_SCHEMA = "npa.sim2real.augmented_frames.v1"
REFERENCE_AUGMENT_MODE = "reference_augment"
REFERENCE_AUGMENT_STATUS = "executed_reference"
# Neutral photoreal prompt used when the caller conditions on an input clip but
# supplies no appearance prompt of its own.
_DEFAULT_INPUT_PROMPT = (
    "photorealistic, natural lighting, high detail, sharp focus, realistic textures"
)


class FrameExtractionError(RuntimeError):
    """Raised when the frame-extraction subprocess cannot decode a video."""


class ControlModalityError(ValueError):
    """Raised when a caller asks for a control modality the model does not have."""


def resolve_control_modality(control: str) -> str:
    """Return the validated control modality for ``control``.

    Fails closed on an unknown modality. NPA used to silently rewrite anything
    outside edge/vis to ``edge``, which meant an operator who asked for ``seg``
    got an edge-conditioned render and no signal that the request was dropped.
    """

    try:
        checkpoint, _weight = validate_control_request(
            modality=control or DEFAULT_INPUT_CONTROL,
            weight=1.0,
            control_asset="precomputed" if str(control or "").strip().lower() == "depth" else "",
        )
    except ControlContractError as exc:
        raise ControlModalityError(str(exc)) from exc
    return checkpoint.modality


def resolve_control_weight(control_weight: float | str) -> float:
    """Return ``control_weight`` after enforcing upstream's range.

    Upstream types it ``Field(ge=0.0, le=1.0)``, so an out-of-range weight is a
    pydantic error raised inside the container — after the accelerator is held and
    the ControlNet weights are loaded. Reject it while that is still cheap.
    """

    try:
        _checkpoint, weight = validate_control_request(
            modality="edge", weight=control_weight
        )
    except ControlContractError as exc:
        raise ControlModalityError(str(exc)) from exc
    return weight


class ProtectedChromaError(RuntimeError):
    """Raised when configured protected-region color preservation cannot complete."""


def _parse_protected_regions(value: str) -> list[tuple[float, float, float, float]]:
    """Parse normalized protected rectangles for source-chroma restoration."""

    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProtectedChromaError("protected regions must be valid JSON") from exc
    if not isinstance(raw, list) or not raw:
        raise ProtectedChromaError("source-chroma mode requires protected regions")
    regions: list[tuple[float, float, float, float]] = []
    for index, item in enumerate(raw):
        bounds: Any = item.get("bounds") if isinstance(item, dict) else item
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            raise ProtectedChromaError(
                f"protected region {index} must have four normalized bounds"
            )
        try:
            x0, y0, x1, y1 = (float(part) for part in bounds)
        except (TypeError, ValueError) as exc:
            raise ProtectedChromaError(
                f"protected region {index} has non-numeric bounds"
            ) from exc
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ProtectedChromaError(
                f"protected region {index} bounds must satisfy "
                "0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1"
            )
        regions.append((x0, y0, x1, y1))
    return regions


def preserve_source_chroma(
    transfer: dict[str, Any],
    *,
    source_video: str,
    regions_json: str = "",
    masks_dir: str = "",
    segmentation: dict[str, Any] | None = None,
    feather_pixels: int = 12,
    luma_max_delta: int = 32,
) -> dict[str, Any]:
    """Restore source chroma in protected regions while retaining generated light.

    Cosmos still generates every frame. This optional deterministic post-process
    restores source Cb/Cr per pixel inside feathered normalized rectangles or
    frame-aligned SAM2 masks;
    generated luma is retained within a bounded per-pixel distance from source,
    so mild illumination/exposure augmentation remains visible while extreme
    darkening or brightening fails to alter protected identity colors. Exact
    frame-count and geometry alignment are required to avoid color ghosts across
    moving boundaries.
    Frame-count or decode mismatches fail closed instead of publishing partially
    corrected output.
    """

    if bool(regions_json) == bool(masks_dir):
        raise ProtectedChromaError(
            "protected chroma requires exactly one of regions_json or masks_dir"
        )
    regions = _parse_protected_regions(regions_json) if regions_json else []
    mask_root = Path(masks_dir) if masks_dir else None
    if mask_root is not None and not mask_root.is_dir():
        raise ProtectedChromaError("protected SAM2 mask directory is missing")
    if feather_pixels < 1:
        raise ProtectedChromaError("protected chroma feather must be positive")
    if not 0 <= luma_max_delta <= 255:
        raise ProtectedChromaError("protected luma max delta must be within 0..255")
    source = Path(source_video)
    augmented = Path(str(transfer.get("video_path") or ""))
    if not source.is_file() or not augmented.is_file():
        raise ProtectedChromaError(
            "protected chroma needs readable source and augmented videos"
        )
    output = augmented.with_name(f"{augmented.stem}-source-chroma.mp4")
    script = r'''
import av, json, numpy as np, sys
from pathlib import Path
source_path, augmented_path, frames_dir_text, regions_text, masks_dir_text, feather_text, luma_delta_text = sys.argv[1:]
frames_dir = Path(frames_dir_text)
regions = json.loads(regions_text) if regions_text else []
regions = [r.get("bounds") if isinstance(r, dict) else r for r in regions]
masks_dir = Path(masks_dir_text) if masks_dir_text else None
feather = int(feather_text)
luma_delta = int(luma_delta_text)
src_container = av.open(source_path)
aug_container = av.open(augmented_path)
aug_stream = aug_container.streams.video[0]
rate = aug_stream.average_rate or aug_stream.base_rate or 30
width, height = int(aug_stream.width), int(aug_stream.height)
masks = []
for bounds in regions:
    x0 = max(0, min(width - 1, int(round(float(bounds[0]) * width))))
    y0 = max(0, min(height - 1, int(round(float(bounds[1]) * height))))
    x1 = max(x0 + 1, min(width, int(round(float(bounds[2]) * width))))
    y1 = max(y0 + 1, min(height, int(round(float(bounds[3]) * height))))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.minimum(
        np.minimum(xx - x0, x1 - 1 - xx),
        np.minimum(yy - y0, y1 - 1 - yy),
    ).astype(np.float32)
    alpha = np.clip((distance + 1.0) / float(feather), 0.0, 1.0)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y0:y1, x0:x1] = alpha
    masks.append(mask)
rect_alpha = np.maximum.reduce(masks) if masks else None
src_frames = iter(src_container.decode(video=0))
aug_frames = iter(aug_container.decode(video=0))
count = 0
for aug_frame in aug_frames:
    try:
        src_frame = next(src_frames)
    except StopIteration as exc:
        raise RuntimeError("source has fewer frames than augmented clip") from exc
    src = src_frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
    aug = aug_frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
    srcf, augf = src.astype(np.float32), aug.astype(np.float32)
    src_cb = 128.0 - 0.168736 * srcf[..., 0] - 0.331264 * srcf[..., 1] + 0.5 * srcf[..., 2]
    src_cr = 128.0 + 0.5 * srcf[..., 0] - 0.418688 * srcf[..., 1] - 0.081312 * srcf[..., 2]
    src_y = 0.299 * srcf[..., 0] + 0.587 * srcf[..., 1] + 0.114 * srcf[..., 2]
    y = 0.299 * augf[..., 0] + 0.587 * augf[..., 1] + 0.114 * augf[..., 2]
    aug_cb = 128.0 - 0.168736 * augf[..., 0] - 0.331264 * augf[..., 1] + 0.5 * augf[..., 2]
    aug_cr = 128.0 + 0.5 * augf[..., 0] - 0.418688 * augf[..., 1] - 0.081312 * augf[..., 2]
    if masks_dir is not None:
        from PIL import Image, ImageFilter
        mask_path = masks_dir / f"mask-{count:06d}.png"
        if not mask_path.is_file():
            raise RuntimeError(f"missing frame-aligned protected mask {mask_path.name}")
        with Image.open(mask_path) as opened_mask:
            mask_image = opened_mask.convert("L").resize((width, height), Image.Resampling.NEAREST)
        binary_mask = np.asarray(mask_image, dtype=np.float32) / 255.0
        if feather > 1:
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=max(0.5, feather / 2.0)))
            # Feather inward only. Multiplying by the original binary mask keeps
            # source chroma from bleeding into unprotected augmentation pixels.
            alpha = (np.asarray(mask_image, dtype=np.float32) / 255.0) * binary_mask
        else:
            alpha = binary_mask
    else:
        if rect_alpha is None:
            raise RuntimeError("protected region mask is missing")
        alpha = rect_alpha
    bounded_y = np.clip(y, src_y - float(luma_delta), src_y + float(luma_delta))
    y = y * (1.0 - alpha) + bounded_y * alpha
    cb = aug_cb * (1.0 - alpha) + src_cb * alpha
    cr = aug_cr * (1.0 - alpha) + src_cr * alpha
    rgb = np.stack((
        y + 1.402 * (cr - 128.0),
        y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0),
        y + 1.772 * (cb - 128.0),
    ), axis=-1)
    frame = av.VideoFrame.from_ndarray(np.clip(rgb, 0, 255).astype(np.uint8), format="rgb24")
    frame.to_image().save(frames_dir / f"frame-{count:06d}.png")
    count += 1
try:
    next(src_frames)
except StopIteration:
    pass
else:
    raise RuntimeError("source has more frames than augmented clip")
aug_container.close(); src_container.close()
if count == 0:
    raise RuntimeError("no frames decoded")
if masks_dir is not None:
    mask_files = sorted(masks_dir.glob("mask-*.png"))
    if len(mask_files) != count:
        raise RuntimeError("protected SAM2 mask count differs from video frame count")
print(json.dumps({"frames": count, "fps": float(rate)}))
'''
    try:
        with tempfile.TemporaryDirectory(
            prefix="npa-protected-chroma-", dir=str(augmented.parent)
        ) as frames_dir:
            completed = subprocess.run(
                [
                    str(_venv_python(cosmos_transfer_repo())),
                    "-c",
                    script,
                    str(source),
                    str(augmented),
                    frames_dir,
                    regions_json,
                    str(mask_root or ""),
                    str(feather_pixels),
                    str(luma_max_delta),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            try:
                decoded = json.loads(str(completed.stdout).strip().splitlines()[-1])
                frame_count = int(decoded["frames"])
                fps = float(decoded["fps"])
            except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ProtectedChromaError(
                    "protected source-chroma decoder returned invalid metadata"
                ) from exc
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(Path(frames_dir) / "frame-%06d.png"),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
    except ProtectedChromaError:
        raise
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = str(getattr(exc, "stderr", "") or exc).strip()
        raise ProtectedChromaError(
            f"protected source-chroma restoration failed: {detail}"[:300]
        ) from exc
    if not output.is_file() or output.stat().st_size <= 0:
        raise ProtectedChromaError("protected source-chroma restoration wrote no video")
    result = dict(transfer)
    result["video_path"] = str(output)
    result["video_bytes"] = output.stat().st_size
    result["protected_chroma"] = {
        "mode": "source-chroma",
        "method": (
            "sam2-mask-feathered-per-pixel-source-chroma"
            if mask_root is not None
            else "feathered-per-pixel-source-chroma"
        ),
        "region_count": len(regions),
        "feather_pixels": feather_pixels,
        "luma_max_delta": luma_max_delta,
        "frame_count": frame_count,
    }
    if mask_root is not None:
        result["protected_chroma"]["segmentation"] = segmentation or {
            "engine": "meta-sam2-upstream"
        }
    return result


def _spec_for_input_video(
    repo: Path,
    *,
    input_video: str,
    prompt: str,
    control: str,
    control_weight: float,
    guidance: float,
    name: str,
    seed: int | None = None,
    control_asset: str = "",
    control_prompt: str = "",
    mask_asset: str = "",
    mask_prompt: str = "",
) -> tuple[str, str]:
    """Write a Cosmos Transfer 2.5 controlnet spec that CONDITIONS ON ``input_video``.

    ``video_path`` is the caller's real input clip; edge/vis/seg may be computed
    from it, while depth requires ``control_asset`` from an operator-owned
    weight-free method. The output preserves structure/motion while ``prompt``
    drives a new appearance -- i.e. a genuine augmentation of the caller's footage.

    ``control_prompt`` names the objects on-the-fly ``seg`` should segment.
    ``mask_asset`` / ``mask_prompt`` write upstream's region mask, a binary
    spatiotemporal video restricting the control to the white pixels; giving both
    is rejected because upstream accepts only one. Returns
    ``(spec_path_relative_to_repo, control_modality)``.
    """
    import json as _json

    try:
        checkpoint, normalized_weight = validate_control_request(
            modality=control,
            weight=control_weight,
            control_asset=control_asset,
            control_prompt=control_prompt,
            mask_asset=mask_asset,
            mask_prompt=mask_prompt,
        )
    except ControlContractError as exc:
        raise ControlModalityError(str(exc)) from exc
    modality = checkpoint.modality
    control_config: dict[str, Any] = {
        "control_weight": normalized_weight
    }
    if control_asset:
        control_config["control_path"] = str(Path(control_asset).resolve())
    if control_prompt:
        control_config["control_prompt"] = control_prompt
    if mask_asset:
        control_config["mask_path"] = str(Path(mask_asset).resolve())
    if mask_prompt:
        control_config["mask_prompt"] = mask_prompt
    spec = {
        "name": str(name or "npa_input"),
        "prompt": str(prompt or "").strip() or _DEFAULT_INPUT_PROMPT,
        # Absolute path so it resolves regardless of where the spec file lives.
        "video_path": str(Path(input_video).resolve()),
        "guidance": guidance,
        modality: control_config,
    }
    if seed is not None:
        spec["seed"] = int(seed)
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name or "input"))
    spec_path = repo / f"_npa_input_spec_{safe}.json"
    spec_path.write_text(_json.dumps(spec, indent=2), encoding="utf-8")
    return str(spec_path.relative_to(repo)), modality


def _classify_output_videos(
    out_dir: Path,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Split an inference output directory into generated / control / mask videos.

    Upstream writes ``<name>.mp4`` for the generated clip and, per supported
    modality, ``<name>_control_<key>.mp4`` plus ``<name>_mask_<key>.mp4`` when a
    region mask was generated from a prompt. Both exact sidecar shapes must be
    excluded from the generated set: a full-frame binary mask compresses well but
    not always to less than the render, so picking the largest file could publish
    a mask as the augmentation. Unknown sidecar keys are also quarantined when
    their generated sibling exists, which fails closed if upstream adds a modality
    before NPA learns how to publish its evidence. Ordinary run names may
    themselves contain words such as ``control`` or ``_mask_`` and remain
    generated output.
    """

    import re as _re

    control: dict[str, str] = {}
    masks: dict[str, str] = {}
    generated: list[str] = []
    paths = [
        Path(path)
        for path in sorted(glob.glob(str(out_dir / "**" / "*.mp4"), recursive=True))
    ]
    path_set = set(paths)
    sidecar_pattern = _re.compile(
        r"^(?P<base>.+)_(?P<kind>control|mask)_"
        r"(?P<key>[a-z0-9][a-z0-9_-]*)\.mp4$"
    )
    for candidate in paths:
        match = sidecar_pattern.match(candidate.name)
        if match is None:
            generated.append(str(candidate))
            continue
        key = match.group("key")
        has_generated_sibling = (
            candidate.with_name(f"{match.group('base')}.mp4") in path_set
        )
        if not has_generated_sibling and key not in INPUT_CONTROLS:
            # A standalone render may legitimately contain these words. Only a
            # known modality suffix is intrinsically evidence-shaped; an unknown
            # suffix needs the upstream sibling relationship to disambiguate it.
            generated.append(str(candidate))
            continue
        if has_generated_sibling and key in INPUT_CONTROLS:
            target = control if match.group("kind") == "control" else masks
            target.setdefault(key, str(candidate))
        # Unknown keys with a generated sibling, and known sidecars missing their
        # sibling, are intentionally omitted until a coherent evidence contract
        # can be established.
    return generated, control, masks


def cosmos_transfer_repo() -> Path:
    return Path(os.environ.get("COSMOS_TRANSFER_REPO", DEFAULT_REPO))


def _venv_python(repo: Path) -> Path:
    return repo / ".venv" / "bin" / "python"


def _venv_has_torch(py: Path) -> bool:
    # Probe defensively: a mirrored/hardened transfer image can make the venv
    # python unreadable (stat -> PermissionError) or non-executable. Treat any
    # OSError as "runtime unavailable" so callers fall back to the descriptor
    # path instead of crashing the augment stage.
    try:
        if not py.exists():
            return False
    except OSError:
        return False
    try:
        proc = subprocess.run(
            [str(py), "-c", "import torch, flash_attn"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return proc.returncode == 0


def cosmos_transfer_available() -> bool:
    """True when the real Cosmos-Transfer2.5 runtime is present and runnable.

    The redistributable image bakes the locked inference venv. Runtime dependency
    self-healing would make the executed dependency set differ from the audited
    image, so a missing venv is unavailable rather than a cue to download packages.
    """

    repo = cosmos_transfer_repo()
    if not (repo / "examples" / "inference.py").is_file():
        return False
    return _venv_has_torch(_venv_python(repo))


def ensure_env(repo: Path) -> Path:
    """Return the audited inference venv; never mutate or download at run time."""

    py = _venv_python(repo)
    if _venv_has_torch(py):
        return py
    raise RuntimeError(
        "cosmos-transfer2.5 audited inference venv is missing or unusable; "
        "rebuild the pinned npa-cosmos2-transfer image"
    )


def _require_runtime_hf_token() -> None:
    """Refuse gated-model inference before any anonymous/partial download starts."""

    if not os.environ.get("HF_TOKEN", "").strip():
        raise RuntimeError(
            "HF_TOKEN is required at run time for gated Cosmos Transfer weights; "
            "no model download was attempted"
        )


class GuardrailNLTKDataError(RuntimeError):
    """Typed failure to prepare the pinned Cosmos guardrail NLTK payload."""

    def __init__(self, category: str, message: str) -> None:
        self.category = category
        super().__init__(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guardrail_nltk_inventory(root: Path) -> dict[str, dict[str, Any]]:
    """Return a content inventory, rejecting links and non-regular payloads."""

    files: dict[str, dict[str, Any]] = {}
    for entry in sorted(root.rglob("*")):
        relative = entry.relative_to(root)
        if relative.as_posix() == GUARDRAIL_NLTK_READY_MARKER:
            continue
        if entry.is_symlink():
            raise GuardrailNLTKDataError(
                "content_invalid",
                f"Cosmos guardrail NLTK materialization contains a link: {relative}; "
                "remove the revision-scoped cache and retry the pinned fetch",
            )
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise GuardrailNLTKDataError(
                "content_invalid",
                f"Cosmos guardrail NLTK materialization contains a non-regular file: "
                f"{relative}; remove the revision-scoped cache and retry",
            )
        files[relative.as_posix()] = {
            "sha256": _sha256_file(entry),
            "size": entry.stat().st_size,
        }
    if not files:
        raise GuardrailNLTKDataError(
            "content_invalid",
            "Pinned Cosmos guardrail NLTK materialization is empty; verify the exact "
            "upstream revision and subtree before retrying",
        )
    return files


def _guardrail_nltk_tree_sha256(files: dict[str, dict[str, Any]]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_guardrail_nltk_ready_marker(root: Path) -> None:
    files = _guardrail_nltk_inventory(root)
    payload = {
        "schema": GUARDRAIL_NLTK_MANIFEST_SCHEMA,
        "repo_id": GUARDRAIL_REPO,
        "revision": GUARDRAIL_REVISION,
        "file_count": len(files),
        "files": files,
        "tree_sha256": _guardrail_nltk_tree_sha256(files),
    }
    marker = root / GUARDRAIL_NLTK_READY_MARKER
    marker.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.chmod(marker, 0o444)


def _verify_guardrail_nltk_materialization(destination: Path) -> int:
    """Verify exact revision identity and every copied byte before cache reuse."""

    marker = destination / GUARDRAIL_NLTK_READY_MARKER
    if destination.is_symlink() or not destination.is_dir() or marker.is_symlink():
        raise GuardrailNLTKDataError(
            "cache_invalid",
            "Cosmos guardrail NLTK cache is not a verified regular-file directory; "
            "remove only its revision-scoped path and retry",
        )
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GuardrailNLTKDataError(
            "cache_invalid",
            "Cosmos guardrail NLTK cache has no readable completion manifest; "
            "remove only its revision-scoped path and retry",
        ) from exc
    expected_identity = (
        payload.get("schema") == GUARDRAIL_NLTK_MANIFEST_SCHEMA
        and payload.get("repo_id") == GUARDRAIL_REPO
        and payload.get("revision") == GUARDRAIL_REVISION
    )
    if not expected_identity or not isinstance(payload.get("files"), dict):
        raise GuardrailNLTKDataError(
            "cache_invalid",
            "Cosmos guardrail NLTK cache identity does not match the exact pinned "
            "repository revision; do not reuse it",
        )
    files = _guardrail_nltk_inventory(destination)
    if (
        payload["files"] != files
        or payload.get("file_count") != len(files)
        or payload.get("tree_sha256") != _guardrail_nltk_tree_sha256(files)
    ):
        raise GuardrailNLTKDataError(
            "cache_invalid",
            "Cosmos guardrail NLTK cache content does not match its verified manifest; "
            "remove only its revision-scoped path and retry the pinned fetch",
        )
    return len(files)


def _guardrail_nltk_download_error(exc: Exception) -> GuardrailNLTKDataError:
    name = type(exc).__name__.lower()
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status == 429 or "ratelimit" in name or "rate_limit" in name:
        return GuardrailNLTKDataError(
            "rate_limited",
            "Hugging Face rate-limited the pinned Cosmos guardrail NLTK fetch; "
            "reuse a verified cache or retry later",
        )
    if "revisionnotfound" in name or "revision_not_found" in name:
        return GuardrailNLTKDataError(
            "revision_unavailable",
            f"Pinned Cosmos guardrail revision {GUARDRAIL_REVISION} is unavailable; "
            "do not substitute another revision without a reviewed source update",
        )
    if status in {401, 403} or "gatedrepo" in name or "repositorynotfound" in name:
        return GuardrailNLTKDataError(
            "access_denied",
            "Hugging Face denied the pinned Cosmos guardrail NLTK payload fetch; "
            "verify the operator token has exact upstream repository access",
        )
    return GuardrailNLTKDataError(
        "network_unavailable",
        "Unable to fetch the pinned Cosmos guardrail NLTK payload from Hugging Face; "
        "restore network access or prewarm a verified revision-scoped cache",
    )


def prepare_guardrail_nltk_data(*, hf_home: str | None = None) -> int:
    """Download and safely materialize the pinned guardrail tokenizer data.

    Hugging Face snapshots represent files as symlinks into their local blob
    store. NLTK 3.10.3 deliberately refuses to follow those links when opening
    tokenizer data. Download only the pinned guardrail subtree, prove every file
    resolves inside the same cache, and copy it into a revision-scoped regular-file
    tree outside the snapshot. Upstream may refresh its snapshot after this step;
    the separate ``NLTK_DATA`` tree therefore remains safe and stable. This
    preserves the path-security fix without disabling Cosmos guardrails or
    trusting an unpinned snapshot.
    """

    home = Path(hf_home or os.environ.get("HF_HOME", "/opt/cosmos-data/hf_cache"))
    destination = home / GUARDRAIL_NLTK_MATERIALIZED_DIR / GUARDRAIL_REVISION
    if destination.exists() or destination.is_symlink():
        return _verify_guardrail_nltk_materialization(destination)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise GuardrailNLTKDataError(
            "runtime_invalid",
            "The audited Cosmos Transfer runtime is missing huggingface_hub; "
            "rebuild the pinned image before fetching guardrail data",
        ) from exc

    hub = (home / "hub").resolve()
    try:
        downloaded = snapshot_download(
            repo_id=GUARDRAIL_REPO,
            revision=GUARDRAIL_REVISION,
            allow_patterns=["blocklist/nltk_data/**"],
            cache_dir=hub,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as exc:
        raise _guardrail_nltk_download_error(exc) from exc
    snapshot = Path(downloaded).resolve()
    nltk_data = (snapshot / "blocklist" / "nltk_data").resolve()
    if (
        not nltk_data.is_dir()
        or not snapshot.is_relative_to(hub)
        or not nltk_data.is_relative_to(snapshot)
    ):
        raise GuardrailNLTKDataError(
            "content_invalid",
            "Pinned Cosmos guardrail NLTK subtree is missing or outside the exact "
            "Hugging Face cache snapshot",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{GUARDRAIL_REVISION}.", dir=destination.parent)
    )
    materialized = 0
    try:
        for entry in sorted(nltk_data.rglob("*")):
            relative = entry.relative_to(nltk_data)
            output = staging / relative
            if entry.is_dir():
                if entry.is_symlink():
                    raise GuardrailNLTKDataError(
                        "content_invalid",
                        f"Unsafe Cosmos guardrail cache directory link: {relative}",
                    )
                output.mkdir(parents=True, exist_ok=True)
                continue
            target = entry.resolve(strict=True)
            if not target.is_file() or not target.is_relative_to(hub):
                raise RuntimeError(f"unsafe Cosmos guardrail cache file: {entry}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with target.open("rb") as source, output.open("wb") as copied:
                shutil.copyfileobj(source, copied)
            os.chmod(output, 0o444)
            materialized += 1
        if materialized == 0:
            raise GuardrailNLTKDataError(
                "content_invalid", "Pinned Cosmos guardrail NLTK subtree is empty"
            )
        _write_guardrail_nltk_ready_marker(staging)
        try:
            os.replace(staging, destination)
        except OSError:
            if not destination.exists():
                raise
            winner_count = _verify_guardrail_nltk_materialization(destination)
            shutil.rmtree(staging)
            return winner_count
    except GuardrailNLTKDataError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise GuardrailNLTKDataError(
            "materialization_failed",
            "Could not atomically materialize the pinned Cosmos guardrail NLTK "
            "payload; no partial cache was published, so retry after checking the "
            "revision-scoped cache filesystem",
        ) from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _verify_guardrail_nltk_materialization(destination)


def _guardrail_nltk_data_path(hf_home: str) -> Path:
    """Return the regular-file NLTK tree created for the pinned guardrail."""

    return (
        Path(hf_home)
        / GUARDRAIL_NLTK_MATERIALIZED_DIR
        / GUARDRAIL_REVISION
    )


def _spec_with_prompt(repo: Path, spec: str, prompt: str, *, tag: str = "") -> str:
    """Write a copy of ``spec`` with its text prompt overridden; return its path.

    Cosmos controlnet specs carry the text prompt that steers appearance. Patching
    it lets the sampled appearance combo actually condition the diffusion (same
    control video / motion, new look) instead of being a decorative label. The
    copy sits next to the original so relative control-asset paths still resolve.
    ``tag`` makes the patched filename unique per variant so concurrent multiply
    fan-out (one inference per GPU) never clobbers a sibling's spec.
    Best-effort: on any failure we fall back to the original spec.
    """
    import json as _json

    try:
        spec_path = repo / spec
        data = _json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return spec
        data["prompt"] = prompt
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(tag or ""))
        prefix = f"_npa_prompted_{safe}_" if safe else "_npa_prompted_"
        patched = spec_path.with_name(prefix + spec_path.name)
        patched.write_text(_json.dumps(data, indent=2), encoding="utf-8")
        return str(patched.relative_to(repo))
    except Exception:  # noqa: BLE001 - prompt override is best-effort
        return spec


def run_cosmos_transfer(
    *,
    run_id: str = "",
    spec: str | None = None,
    prompt: str | None = None,
    out_subdir: str | None = None,
    hf_home: str | None = None,
    input_video: str | None = None,
    control: str = DEFAULT_INPUT_CONTROL,
    control_weight: float = 1.0,
    control_asset: str = "",
    control_prompt: str = "",
    mask_asset: str = "",
    mask_prompt: str = "",
    guidance: float = 3.0,
    seed: int | None = None,
    cuda_visible_devices: str | None = None,
    variant_tag: str = "",
    disable_content_guardrails: bool | None = None,
) -> dict[str, Any]:
    """Run a real Cosmos-Transfer2.5 inference; return the generated video + metadata.

    ``spec`` is a controlnet spec path relative to the transfer repo (or the
    ``COSMOS_TRANSFER_SPEC`` environment override). No upstream fixture is baked.
    ``prompt`` (or ``COSMOS_TRANSFER_PROMPT``), when set, overrides the spec's text
    prompt so the sampled appearance actually conditions the augmentation.

    When ``input_video`` is provided the transfer is CONDITIONED ON THAT CLIP: a
    controlnet spec is built with ``video_path`` = the input and the ``control``
    selected modality (or precomputed depth control), so the output is a real augmentation of the
    caller's footage (new appearance from ``prompt``, same structure/motion).
    ``control_asset`` substitutes a precomputed control video, ``control_prompt``
    names what on-the-fly ``seg`` segments, and ``mask_asset``/``mask_prompt``
    restrict the control to a region.
    When ``input_video`` is absent, the caller must provide an operator-owned spec.

    The returned ``control_videos`` / ``mask_videos`` map each modality to the
    control and region-mask videos upstream wrote next to the generated clip.
    """

    repo = cosmos_transfer_repo()
    _require_runtime_hf_token()
    py = ensure_env(repo)
    tag = str(variant_tag or run_id or "input")
    conditioned_control = ""
    if input_video:
        spec, conditioned_control = _spec_for_input_video(
            repo,
            input_video=input_video,
            prompt=prompt or os.environ.get("COSMOS_TRANSFER_PROMPT", ""),
            control=control,
            control_weight=control_weight,
            guidance=guidance,
            name=tag,
            seed=seed,
            control_asset=control_asset,
            control_prompt=control_prompt,
            mask_asset=mask_asset,
            mask_prompt=mask_prompt,
        )
    else:
        spec = spec or os.environ.get("COSMOS_TRANSFER_SPEC", DEFAULT_SPEC)
        if not spec:
            raise ValueError(
                "Cosmos Transfer inference requires input_video or an explicit "
                "COSMOS_TRANSFER_SPEC; no upstream media is bundled"
            )
        prompt = prompt or os.environ.get("COSMOS_TRANSFER_PROMPT", "")
        if prompt:
            spec = _spec_with_prompt(repo, spec, prompt, tag=tag)
    out = out_subdir or f"outputs/{run_id or 'transfer'}"
    out_abs = repo / out
    if out_abs.exists():
        shutil.rmtree(out_abs)

    env = dict(os.environ)
    env["HF_HOME"] = hf_home or os.environ.get("HF_HOME", "/opt/cosmos-data/hf_cache")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_guardrail_nltk_data(hf_home=env["HF_HOME"])
    safe_nltk_data = str(_guardrail_nltk_data_path(env["HF_HOME"]))
    env["NLTK_DATA"] = os.pathsep.join(
        part for part in (safe_nltk_data, env.get("NLTK_DATA", "")) if part
    )
    if cuda_visible_devices is not None and str(cuda_visible_devices).strip() != "":
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices).strip()
    # Only the specs WE synthesized this call are ephemeral; never delete a
    # caller-supplied spec. Per-variant tags keep siblings
    # from clobbering each other, so removing exactly our file is fan-out safe.
    # Capture its content first so callers can still inspect the effective spec
    # after the file is gone (nothing depends on the ephemeral file persisting).
    temp_spec = repo / spec if Path(spec).name.startswith(("_npa_input_spec_", "_npa_prompted_")) else None
    spec_json: dict[str, Any] | None = None
    if temp_spec is not None:
        try:
            import json as _json

            spec_json = _json.loads(temp_spec.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            spec_json = None
    if disable_content_guardrails is None:
        disable_content_guardrails = os.environ.get(
            DISABLE_CONTENT_GUARDRAILS_ENV, ""
        ).strip().lower() in {"1", "true", "yes", "on"}
    argv = [str(py), "examples/inference.py", "-i", spec, "-o", out]
    if disable_content_guardrails:
        # Upstream exposes this explicit setup option for domains whose valid
        # generated pixels are outside the generic video guardrail's calibration
        # set. Keep the NPA default fail-closed; operators must opt out per run.
        argv.append("--disable-guardrails")
    try:
        # Upstream progress output includes the effective prompt and local input
        # path. Keep it in an unnamed, process-local file that is destroyed at
        # completion; the retained task log reports only aggregate NPA evidence.
        with tempfile.TemporaryFile() as vendor_log:
            subprocess.run(
                argv,
                cwd=repo,
                env=env,
                check=True,
                stdout=vendor_log,
                stderr=subprocess.STDOUT,
            )
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError(
            "Cosmos Transfer inference failed; inspect GPU/model access and retry"
        ) from None
    finally:
        if temp_spec is not None:
            try:
                temp_spec.unlink()
            except OSError:
                pass

    generated, control_videos, mask_videos = _classify_output_videos(out_abs)
    # Upstream already ran its generated-video guardrail before writing this
    # file. Do not reuse the container golden-eval's 100 KiB heuristic here:
    # a short valid transfer can produce a ~9 KiB video (live job 371). S3
    # publication below still fails closed unless PyAV can decode at least one
    # exact frame, which is the artifact contract consumers need.
    produced = sorted(
        (f for f in generated if os.path.getsize(f) > 0),
        key=os.path.getsize,
        reverse=True,
    )
    if not produced:
        raise RuntimeError(f"cosmos-transfer2.5 produced no output video in {out_abs}")
    return {
        "video_path": produced[0],
        "video_bytes": os.path.getsize(produced[0]),
        "control_path": next(iter(control_videos.values()), ""),
        "control_videos": control_videos,
        "mask_videos": mask_videos,
        "control_weight": float(control_weight),
        "control_prompt": control_prompt,
        "mask_prompt": mask_prompt,
        "control_asset": control_asset,
        "mask_asset": mask_asset,
        "inference_seed": seed,
        "out_dir": str(out_abs),
        "spec": spec,
        "spec_json": spec_json,
        "repo": str(repo),
        "input_conditioned": bool(input_video),
        "input_video": str(input_video or ""),
        "control": conditioned_control,
        "content_guardrails_enabled": not disable_content_guardrails,
    }


def extract_frames(video_path: str, dest_dir: Path, *, max_frames: int = 8) -> list[Path]:
    """Extract up to ``max_frames`` evenly-spaced PNG frames from ``video_path``.

    Runs in the transfer venv (which ships PyAV). A successful decode with no
    video frames returns ``[]``; subprocess and PyAV failures retain their stderr
    and original exception as :class:`FrameExtractionError`.
    """

    repo = cosmos_transfer_repo()
    py = _venv_python(repo)
    dest_dir.mkdir(parents=True, exist_ok=True)
    script = (
        "import av, sys\n"
        "from pathlib import Path\n"
        "vp, dest, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])\n"
        "with av.open(vp) as c:\n"
        "    frames = [f for f in c.decode(video=0)]\n"
        "step = max(1, len(frames) // n) if frames else 1\n"
        "sel = frames[::step][:n]\n"
        "for i, fr in enumerate(sel):\n"
        "    fr.to_image().save(str(dest / f'frame-{i:05d}.png'))\n"
        "print(len(sel))\n"
    )
    try:
        subprocess.run(
            [str(py), "-c", script, video_path, str(dest_dir), str(max_frames)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = str(exc.stderr or exc.stdout or exc).strip()
        raise FrameExtractionError(
            f"frame extraction failed for {video_path!r} with exit code "
            f"{exc.returncode}: {detail}"
        ) from exc
    except OSError as exc:
        raise FrameExtractionError(
            f"could not start frame extraction for {video_path!r}: {exc}"
        ) from exc
    return sorted(dest_dir.glob("frame-*.png"))


def publish_transfer_clip(
    transfer: dict[str, Any],
    output_uri: str,
    *,
    run_id: str = "",
    clip_name: str = "",
    variables: dict[str, Any] | None = None,
    variant_index: int = 0,
    max_frames: int = 8,
    frames_output_uri: str = "",
    control_output_uri: str = "",
    require_frames: bool = False,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Publish ONE real Cosmos-Transfer2.5 result as a per-clip dir under
    ``output_uri`` (the ``cosmos_augmented/`` prefix), returning the clip's
    descriptor (no run-level manifest is written here).

    Writes:

        <clip>/augmented_video.mp4
        <clip>/frame-00000.png ... (or ``frames_output_uri/frame-*.png``)
        <clip>/metadata.json      (variables + mode, for the Rerun label)

    When ``control_output_uri`` is set, the conditioning signal itself is
    published too, under a SIBLING prefix rather than inside ``<clip>/``:

        <control>/<clip>/control_<modality>.mp4        (e.g. the seg map)
        <control>/<clip>/control_<modality>/frame-*.png
        <control>/<clip>/mask_<modality>.mp4           (region mask, when used)
        <control>/<clip>/mask_<modality>/frame-*.png

    The sibling prefix is deliberate: the evaluator enumerates clip directories
    under ``cosmos_augmented/`` and falls back to the alphabetically first PNG in
    one, so a ``control/`` child would hand the attribute-verify VLM a
    segmentation map instead of the augmented frame it must grade.

    This is the unit of "multiply": the caller runs one inference per sampled
    appearance combo and publishes each as its own clip, then calls
    :func:`write_run_manifest` once to emit the run-level ``manifest.json``.
    """

    if not output_uri.startswith("s3://"):
        raise ValueError(f"output_uri must be an s3:// prefix, got: {output_uri!r}")
    from npa.clients.storage import StorageClient
    import tempfile as _tempfile

    client = storage_client or StorageClient.from_environment()
    base = output_uri if output_uri.endswith("/") else output_uri + "/"
    clip = clip_name or (f"aug-{run_id}" if run_id else "aug0")
    clip_base = f"{base}{clip}/"
    frames_base = (
        frames_output_uri.rstrip("/") + "/" if frames_output_uri else clip_base
    )
    video_uri = f"{clip_base}augmented_video.mp4"

    import json as _json

    # This publish path only runs after the REAL Cosmos Transfer 2.5 model
    # executed on GPU, so record the GPU mode (kept in sync with the provenance
    # classifier in data_factory_provenance.py). When the transfer was
    # conditioned on the caller's input clip, record that provenance so the run
    # view can show the augmentation is genuinely derived from real input.
    input_conditioned = bool(transfer.get("input_conditioned"))
    conditioned_input = Path(str(transfer.get("input_video") or "")).name
    conditioned_control = str(transfer.get("control") or "")
    content_guardrails_enabled = bool(
        transfer.get("content_guardrails_enabled", True)
    )
    protected_chroma = transfer.get("protected_chroma") or {"mode": "off"}
    refinement = transfer.get("refinement") or {}
    effective_control_weight = transfer.get("effective_control_weight")
    effective_guidance = transfer.get("effective_guidance")
    inference_seed = transfer.get("inference_seed")
    conditioning_clip_uri = str(transfer.get("conditioning_clip_uri") or "")

    control_uris: dict[str, str] = {}
    control_frames: dict[str, list[str]] = {}
    control_evidence: dict[str, str] = {
        "status": "pending" if control_output_uri else "not_requested"
    }
    frame_index: list[dict[str, str]] = []
    with _tempfile.TemporaryDirectory(prefix="npa-cosmos-pub-") as tmp:
        frames = extract_frames(transfer["video_path"], Path(tmp) / "frames", max_frames=max_frames)
        if require_frames and not frames:
            raise RuntimeError(
                "Cosmos Transfer completed but no frames could be extracted from "
                f"{transfer['video_path']!r}; refusing to publish a manifest whose "
                "augmented_frames_uri has no frame-NNNNN.png objects."
            )
        # Validate the required frame contract before publishing any object. A
        # zero-frame decode must not leave a plausible video-only success behind.
        client.upload_file(transfer["video_path"], video_uri)
        for i, frame_path in enumerate(frames):
            key = f"frame-{i:05d}.png"
            client.upload_file(str(frame_path), f"{frames_base}{key}")
            frame_index.append({"frame_id": f"frame-{i:05d}", "uri": f"{frames_base}{key}"})

        clip_meta: dict[str, Any] = {
            "schema": TRANSFER_MANIFEST_SCHEMA,
            "mode": TRANSFER_MANIFEST_MODE,
            "clip": clip,
            # Position of this variant in the sampled combo order. It is the same
            # number whichever node rendered the clip, so a merged multi-node
            # manifest can restore the single-node ordering.
            "variant_index": int(variant_index),
            "variables": variables or {},
            "prompt": str((variables or {}).get("prompt") or ""),
            "control_spec": transfer.get("spec", ""),
            "input_conditioned": input_conditioned,
            "conditioned_input": conditioned_input,
            "conditioning_clip_uri": conditioning_clip_uri,
            "control": conditioned_control,
            "control_weight": float(transfer.get("control_weight", 0.0) or 0.0),
            "control_prompt": str(transfer.get("control_prompt") or ""),
            "mask_prompt": str(transfer.get("mask_prompt") or ""),
            "control_uris": control_uris,
            "control_evidence": control_evidence,
            "content_guardrails_enabled": content_guardrails_enabled,
            "protected_chroma": protected_chroma,
            "refinement": refinement,
            "effective_control_weight": effective_control_weight,
            "effective_guidance": effective_guidance,
            "inference_seed": inference_seed,
        }
        cm = Path(tmp) / "metadata.json"

        def _upload_metadata() -> None:
            cm.write_text(
                _json.dumps(clip_meta, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            client.upload_file(str(cm), f"{clip_base}metadata.json")

        # Commit the completed render before optional evidence.  If a control
        # sidecar upload fails, consumers still get a coherent generated video,
        # frames, and metadata that does not claim the evidence exists.
        _upload_metadata()
        if control_output_uri:
            try:
                control_uris, control_frames = _publish_control_signal(
                    transfer,
                    control_output_uri,
                    clip=clip,
                    workdir=Path(tmp) / "control",
                    max_frames=max_frames,
                    client=client,
                )
            except Exception as exc:  # noqa: BLE001 - evidence is non-fatal
                control_uris = {}
                control_frames = {}
                control_evidence = {
                    "status": "failed",
                    # Never persist provider errors: they can contain signed URLs
                    # or credential-adjacent request details.
                    "error_type": type(exc).__name__,
                }
            else:
                control_evidence = {
                    "status": "published" if control_uris else "missing"
                }
            clip_meta["control_uris"] = control_uris
            clip_meta["control_evidence"] = control_evidence
            try:
                _upload_metadata()
            except Exception as exc:  # noqa: BLE001 - core metadata is durable
                # The first metadata object says `pending`, never that evidence
                # exists.  A separately retryable evidence/finalization failure
                # must not discard a completed generated variant.
                logging.getLogger(__name__).debug(
                    "control evidence metadata finalization failed: %s",
                    type(exc).__name__,
                )

    return {
        "clip": clip,
        "variant_index": int(variant_index),
        "clip_base": clip_base,
        "augmented_video_uri": video_uri,
        "frame_count": len(frame_index),
        "frames": frame_index,
        "frames_uri": frames_base,
        "control_spec": transfer.get("spec", ""),
        "video_bytes": int(transfer.get("video_bytes", 0) or 0),
        "input_conditioned": input_conditioned,
        "conditioned_input": conditioned_input,
        "conditioning_clip_uri": conditioning_clip_uri,
        "control": conditioned_control,
        "control_weight": float(transfer.get("control_weight", 0.0) or 0.0),
        "control_prompt": str(transfer.get("control_prompt") or ""),
        "mask_prompt": str(transfer.get("mask_prompt") or ""),
        "control_uris": control_uris,
        "control_frames": control_frames,
        "control_evidence": control_evidence,
        "content_guardrails_enabled": content_guardrails_enabled,
        "protected_chroma": protected_chroma,
        "refinement": refinement,
        "effective_control_weight": effective_control_weight,
        "effective_guidance": effective_guidance,
        "inference_seed": inference_seed,
        "variables": variables or {},
    }


def _publish_control_signal(
    transfer: dict[str, Any],
    control_output_uri: str,
    *,
    clip: str,
    workdir: Path,
    max_frames: int,
    client: Any,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Publish the control map and region mask that conditioned one variant.

    Returns ``({artifact_name: video_uri}, {artifact_name: [frame_uri]})`` where
    the artifact name is ``control_<modality>`` or ``mask_<modality>``. Frame
    extraction is best-effort: a control map that will not decode is worth
    reporting as a missing preview, not worth failing a completed render over.
    """

    if not control_output_uri.startswith("s3://"):
        raise ValueError(
            f"control_output_uri must be an s3:// prefix, got: {control_output_uri!r}"
        )
    base = control_output_uri if control_output_uri.endswith("/") else control_output_uri + "/"
    clip_base = f"{base}{clip}/"
    signals: list[tuple[str, str]] = [
        (f"control_{modality}", path)
        for modality, path in sorted((transfer.get("control_videos") or {}).items())
    ]
    signals += [
        (f"mask_{modality}", path)
        for modality, path in sorted((transfer.get("mask_videos") or {}).items())
    ]

    uris: dict[str, str] = {}
    frames: dict[str, list[str]] = {}
    for name, path in signals:
        if not path or not os.path.isfile(path):
            continue
        video_uri = f"{clip_base}{name}.mp4"
        client.upload_file(path, video_uri)
        uris[name] = video_uri
        try:
            extracted = extract_frames(path, workdir / name, max_frames=max_frames)
        except FrameExtractionError:
            continue
        frame_uris: list[str] = []
        for index, frame_path in enumerate(extracted):
            frame_uri = f"{clip_base}{name}/frame-{index:05d}.png"
            client.upload_file(str(frame_path), frame_uri)
            frame_uris.append(frame_uri)
        if frame_uris:
            frames[name] = frame_uris
    return uris, frames


def _upload_json(client: Any, document: dict[str, Any], uri: str) -> str:
    """Upload ``document`` as pretty-printed JSON to ``uri``."""

    import json as _json
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory(prefix="npa-cosmos-man-") as tmp:
        local = Path(tmp) / Path(uri).name
        local.write_text(
            _json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return client.upload_file(str(local), uri)


def publish_transfer_failure(
    failure: dict[str, Any],
    output_uri: str,
    *,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Persist one failed variant attempt without hiding successful siblings.

    The document deliberately stores a sanitized failure category/type rather
    than vendor stderr, which can contain the effective prompt or local input
    path. ``output_uri`` may already be scheduler-attempt scoped; callers pass
    the same prefix used for successful clip publication so every attempt stays
    additive and attributable.
    """

    if not output_uri.startswith("s3://"):
        raise ValueError(f"output_uri must be an s3:// prefix, got: {output_uri!r}")
    from npa.clients.storage import StorageClient

    index = int(failure.get("variant_index", -1))
    if index < 0:
        raise ValueError("variant failure requires a non-negative variant_index")
    document = dict(failure)
    document.update(
        {
            "schema": "npa.cosmos2.transfer.variant-failure/v1",
            "status": "failed",
            "variant_index": index,
            "promotion_eligible": False,
        }
    )
    failure_uri = (
        f"{output_uri.rstrip('/')}/_failures/variant-{index:05d}.json"
    )
    _upload_json(storage_client or StorageClient.from_environment(), document, failure_uri)
    document["failure_uri"] = failure_uri
    return document


def _json_bytes(document: dict[str, Any]) -> bytes:
    import json as _json

    return (_json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _validated_attempt_id(attempt_id: str) -> str:
    normalized = str(attempt_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", normalized):
        raise ValueError(
            "multi-node attempt_id must contain 1-128 letters, digits, '.', '_', or '-'"
        )
    return normalized


def attempt_output_uri_for(output_uri: str, attempt_id: str) -> str:
    """Return the private object prefix for one gang recovery generation."""

    return (
        f"{output_uri.rstrip('/')}/{ATTEMPT_PREFIX}/"
        f"{_validated_attempt_id(attempt_id)}"
    )


def validate_committed_run_manifest(
    document: Any, output_uri: str = ""
) -> list[dict[str, Any]]:
    """Validate the canonical consumer contract and return its variants.

    The canonical document is a security/correctness boundary: recovery leaves
    older generations below ``_attempts/``, so consumers must never infer a
    generation by listing. Every referenced generated video is constrained to
    this augment root and, for an attempt publication, its exact opaque prefix.
    """

    if not isinstance(document, dict):
        raise ValueError("canonical Cosmos augment manifest is not an object")
    from npa.workflows.paidf_cosmos3 import (
        MANIFEST_SCHEMA as COSMOS3_MANIFEST_SCHEMA,
        PaidfCosmos3Error,
        validate_committed_augment_manifest,
    )

    if document.get("schema") == COSMOS3_MANIFEST_SCHEMA:
        try:
            return validate_committed_augment_manifest(document, output_uri)
        except PaidfCosmos3Error as exc:
            raise ValueError(str(exc)) from exc
    if not (
        document.get("schema") == TRANSFER_MANIFEST_SCHEMA
        and document.get("mode") == TRANSFER_MANIFEST_MODE
        and document.get("status") == TRANSFER_MANIFEST_STATUS
    ):
        raise ValueError(
            "canonical Cosmos augment manifest has an invalid schema, mode, or status"
        )
    variants = document.get("variants")
    if not isinstance(variants, list) or int(document.get("variant_count", -1)) != len(
        variants
    ):
        raise ValueError("canonical Cosmos augment manifest has inconsistent variants")
    failed_variants = document.get("failed_variants", [])
    if not isinstance(failed_variants, list):
        raise ValueError(
            "canonical Cosmos augment manifest has inconsistent failed variants"
        )
    try:
        failed_variant_count = int(
            document.get("failed_variant_count", len(failed_variants))
        )
        attempted_variant_count = int(
            document.get(
                "attempted_variant_count", len(variants) + len(failed_variants)
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "canonical Cosmos augment manifest has inconsistent attempt counts"
        ) from exc
    if (
        failed_variant_count != len(failed_variants)
        or attempted_variant_count != len(variants) + len(failed_variants)
        or attempted_variant_count < 0
    ):
        raise ValueError(
            "canonical Cosmos augment manifest has inconsistent attempt counts"
        )
    attempt_id = str(document.get("attempt_id") or "").strip()
    try:
        node_count = int(document.get("node_count", 1) or 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("canonical Cosmos augment manifest has invalid node_count") from exc
    if node_count < 1:
        raise ValueError("canonical Cosmos augment manifest has invalid node_count")
    scheduler_owned = node_count > 1 or any(
        field in document for field in SCHEDULER_PUBLICATION_IDENTITY_FIELDS
    )
    if scheduler_owned and not attempt_id:
        raise ValueError(
            "scheduler-fenced Cosmos augment manifest has incomplete publication identity"
        )
    if attempt_id:
        _validated_attempt_id(attempt_id)
        try:
            fence = (
                int(document.get("scheduler_fence_sequence", 0)),
                int(document.get("scheduler_fence_attempt", 0)),
            )
            publication_generation = int(
                document.get(PUBLICATION_GENERATION_FIELD, 0)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "scheduler-fenced Cosmos augment manifest has invalid publication identity"
            ) from exc
        required_text = {
            "logical_wave_id": str(document.get("logical_wave_id") or "").strip(),
            "membership_digest": str(document.get("membership_digest") or "").strip(),
            "scheduler_launch_id": str(
                document.get("scheduler_launch_id") or ""
            ).strip(),
        }
        if (
            min(fence) < 1
            or publication_generation < 1
            or document.get("logical_publication") != "conditional"
            or not all(required_text.values())
        ):
            raise ValueError(
                "scheduler-fenced Cosmos augment manifest has incomplete publication identity"
            )
    root = output_uri.rstrip("/") + "/" if output_uri else ""
    if not root and variants:
        first_video = str(variants[0].get("augmented_video_uri") or "")
        if attempt_id:
            marker = f"/{ATTEMPT_PREFIX}/{attempt_id}/"
            if marker in first_video:
                root = first_video.split(marker, 1)[0] + "/"
        else:
            marker = "/cosmos_augmented/"
            if marker in first_video:
                root = first_video.split(marker, 1)[0] + marker
    if not root:
        raise ValueError("canonical Cosmos augment manifest output root is indeterminate")
    expected_prefix = (
        f"{root.rstrip('/')}/{ATTEMPT_PREFIX}/{attempt_id}/"
        if attempt_id
        else root
    )
    seen_clips: set[str] = set()
    seen_indices: set[int] = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError("canonical Cosmos augment manifest has an invalid variant")
        clip = str(variant.get("clip") or "").strip()
        video_uri = str(variant.get("augmented_video_uri") or "").strip()
        try:
            variant_index = int(variant.get("variant_index", index))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "canonical Cosmos augment manifest has an invalid variant index"
            ) from exc
        if not clip or clip in seen_clips or not video_uri.startswith(expected_prefix):
            raise ValueError(
                "canonical Cosmos augment manifest variant is duplicated or outside "
                "its declared publication prefix"
            )
        if variant_index in seen_indices:
            raise ValueError("canonical Cosmos augment manifest duplicates a variant index")
        seen_clips.add(clip)
        seen_indices.add(variant_index)
        control_uris = variant.get("control_uris") or {}
        if not isinstance(control_uris, dict):
            raise ValueError("canonical Cosmos augment manifest control_uris is invalid")
        if attempt_id and any(
            f"/{ATTEMPT_PREFIX}/{attempt_id}/" not in str(uri or "")
            for uri in control_uris.values()
        ):
            raise ValueError(
                "canonical Cosmos augment manifest references control evidence from "
                "another attempt"
            )
    for failure in failed_variants:
        if not isinstance(failure, dict):
            raise ValueError(
                "canonical Cosmos augment manifest has an invalid failed variant"
            )
        try:
            variant_index = int(failure.get("variant_index", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "canonical Cosmos augment manifest has an invalid failed variant index"
            ) from exc
        failure_uri = str(failure.get("failure_uri") or "").strip()
        if (
            variant_index < 0
            or variant_index in seen_indices
            or failure.get("status") != "failed"
            or failure.get("promotion_eligible") is not False
            or not failure_uri.startswith(expected_prefix)
            or "/_failures/" not in failure_uri
        ):
            raise ValueError(
                "canonical Cosmos augment manifest has invalid failed variant provenance"
            )
        seen_indices.add(variant_index)
    if seen_indices != set(range(attempted_variant_count)):
        raise ValueError("canonical Cosmos augment manifest has incomplete variant indices")
    return variants


def claim_run_publication(
    output_uri: str,
    *,
    run_id: str,
    logical_wave_id: str,
    node_count: int,
    membership_digest: str,
    scheduler_fence_sequence: int,
    scheduler_fence_attempt: int,
    scheduler_launch_id: str,
    storage_client: Any = None,
    nonce_factory: Any = None,
) -> tuple[str, str, int]:
    """Atomically claim the canonical manifest for a scheduler wave attempt.

    The durable NPA workflow scheduler supplies the ordered ``(sequence,
    attempt)`` fence before launch. Workload arrival order is never treated as
    authority: a lower/equal token cannot supersede the canonical document. A
    cryptographic nonce keeps the attempt prefix unguessably distinct. The
    returned ETag is the final-publication fence: only a merge whose claim is
    still current may replace the ``publishing`` document with ``executed``.

    Stock SkyPilot 0.12.2 does not expose a globally ordered recovery identity to
    workloads. An inner managed recovery therefore retains the same scheduler
    token and fails closed here. The NPA runtime may submit an explicit retry
    with a higher attempt token after the prior managed job is terminal.
    """

    if not output_uri.startswith("s3://"):
        raise ValueError(f"output_uri must be an s3:// prefix, got: {output_uri!r}")
    logical_wave = str(logical_wave_id or "").strip()
    if not logical_wave:
        raise ValueError("scheduler publication claim requires a logical wave id")
    members = str(membership_digest or "").strip()
    if not members:
        raise ValueError("scheduler publication claim requires a membership digest")
    expected_nodes = int(node_count)
    if expected_nodes < 1:
        raise ValueError("scheduler publication claim requires at least one node")
    fence = (int(scheduler_fence_sequence), int(scheduler_fence_attempt))
    if min(fence) < 1:
        raise ValueError("publication claim requires a positive scheduler fence")
    launch_id = str(scheduler_launch_id or "").strip()
    if not launch_id:
        raise ValueError("publication claim requires a scheduler launch id")

    from npa.clients.storage import (
        StorageClient,
        StorageError,
        StoragePreconditionFailed,
    )

    client = storage_client or StorageClient.from_environment()
    canonical_uri = transfer_manifest_uri_for(output_uri)
    current = client.read_bytes_with_etag(canonical_uri)
    prior_generation = 0
    prior_fence = (0, 0)
    expected_etag = ""
    if current is not None:
        raw, expected_etag = current
        try:
            import json as _json

            document = _json.loads(raw.decode("utf-8"))
            if not isinstance(document, dict):
                raise TypeError("manifest is not an object")
            if not (
                document.get("schema") == TRANSFER_MANIFEST_SCHEMA
                and document.get("mode") == TRANSFER_MANIFEST_MODE
                and document.get("status")
                in {TRANSFER_MANIFEST_STATUS, PUBLICATION_CLAIM_STATUS}
                and str(document.get("run_id") or "") == str(run_id or "")
            ):
                raise ValueError("manifest identity is contradictory")
            prior_generation = int(document.get(PUBLICATION_GENERATION_FIELD, 0) or 0)
            prior_node_count = int(document.get("node_count", 1) or 1)
            scheduler_owned = prior_node_count > 1 or any(
                field in document
                for field in SCHEDULER_PUBLICATION_IDENTITY_FIELDS
            )
            if scheduler_owned:
                _validated_attempt_id(str(document.get("attempt_id") or ""))
                prior_fence = (
                    int(document.get("scheduler_fence_sequence", 0)),
                    int(document.get("scheduler_fence_attempt", 0)),
                )
                required = (
                    prior_generation >= 1,
                    min(prior_fence) >= 1,
                    bool(str(document.get("logical_wave_id") or "").strip()),
                    bool(str(document.get("membership_digest") or "").strip()),
                    bool(str(document.get("scheduler_launch_id") or "").strip()),
                    document.get("status") != TRANSFER_MANIFEST_STATUS
                    or document.get("logical_publication") == "conditional",
                )
                if not all(required):
                    raise ValueError(
                        "prior scheduler publication has incomplete identity"
                    )
        except (UnicodeDecodeError, TypeError, ValueError) as exc:
            raise StorageError(
                f"canonical transfer manifest is unreadable at {canonical_uri}; "
                "refusing to replace an unexamined publication fence"
            ) from exc
        if prior_generation < 0:
            raise StorageError(
                f"canonical transfer manifest has an invalid publication generation "
                f"at {canonical_uri}"
            )
        if fence <= prior_fence:
            raise RuntimeError(
                "scheduler publication claim is stale or duplicates an existing "
                f"scheduler fence (requested={fence}, current={prior_fence}); "
                "transparent SkyPilot recovery is fenced until the NPA runtime "
                "issues a higher retry token"
            )
    generation = prior_generation + 1
    make_nonce = nonce_factory or (lambda: secrets.token_hex(32))
    nonce = str(make_nonce() or "").strip()
    if len(nonce) < 16:
        raise ValueError("publication nonce must contain at least 16 characters")
    attempt_material = (
        f"{logical_wave}\0{fence[0]}\0{fence[1]}\0{members}\0{launch_id}\0{nonce}"
    ).encode("utf-8")
    attempt_id = hashlib.sha256(attempt_material).hexdigest()
    claim = {
        "schema": TRANSFER_MANIFEST_SCHEMA,
        "mode": TRANSFER_MANIFEST_MODE,
        "status": PUBLICATION_CLAIM_STATUS,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "logical_wave_id": logical_wave,
        PUBLICATION_GENERATION_FIELD: generation,
        "node_count": expected_nodes,
        "membership_digest": members,
        "scheduler_fence_sequence": fence[0],
        "scheduler_fence_attempt": fence[1],
        "scheduler_launch_id": launch_id,
        "variant_count": 0,
        "variants": [],
    }
    try:
        claim_etag = client.put_bytes_conditional(
            _json_bytes(claim),
            canonical_uri,
            if_match=expected_etag,
            if_none_match=not expected_etag,
            content_type="application/json",
        )
    except StoragePreconditionFailed as exc:
        raise RuntimeError(
            "scheduler publication claim was superseded before GPU work; "
            "refusing to race another recovery generation"
        ) from exc
    return attempt_id, claim_etag, generation


def build_run_manifest(
    clips: list[dict[str, Any]],
    *,
    run_id: str = "",
    variant_parallelism: int = 1,
    node_count: int = 1,
    shards: list[dict[str, Any]] | None = None,
    attempt_id: str = "",
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the run-level transfer manifest for ``clips`` (no I/O).

    Shared by the single-node publisher (:func:`write_run_manifest`) and the
    multi-node shard merge (:func:`merge_shard_manifests`) so both emit the same
    document for the same set of variants.
    """

    first = clips[0] if clips else {}
    variant_failures = list(failures or [])
    frames = [f for c in clips for f in c.get("frames", [])]
    manifest = {
        "schema": TRANSFER_MANIFEST_SCHEMA,
        "mode": TRANSFER_MANIFEST_MODE,
        "status": TRANSFER_MANIFEST_STATUS,
        "run_id": run_id,
        "clips": [c.get("clip", "") for c in clips],
        "variant_count": len(clips),
        "attempted_variant_count": len(clips) + len(variant_failures),
        "failed_variant_count": len(variant_failures),
        "failed_variants": variant_failures,
        # "multiply": one Cosmos Transfer 2.5 inference per sampled appearance
        # combo. >1 clips means the run genuinely amplified across scenarios.
        "multiply_mode": "multi-variant" if len(clips) > 1 else "single-variant",
        # Concurrent variant renders across the whole augment block: the sum of
        # each node's GPU fan-out, so it is the pod's GPU count on one node and
        # the gang's total on many.
        "variant_parallelism": max(1, int(variant_parallelism or 1)),
        "node_count": max(1, int(node_count or 1)),
        "augmented_video_uri": first.get("augmented_video_uri", ""),
        "augmented_videos": [c.get("augmented_video_uri", "") for c in clips],
        "frame_count": sum(int(c.get("frame_count", 0) or 0) for c in clips),
        "frames": frames,
        "augmented_frames_uri": first.get("frames_uri", ""),
        "control_spec": first.get("control_spec", ""),
        "video_bytes": sum(int(c.get("video_bytes", 0) or 0) for c in clips),
        "input_conditioned": bool(first.get("input_conditioned")),
        "conditioned_input": first.get("conditioned_input", ""),
        "conditioning_clip_uri": first.get("conditioning_clip_uri", ""),
        "control": first.get("control", ""),
        # What actually conditioned the render, so a consumer can tell a
        # seg-conditioned batch from an edge-conditioned one without re-reading
        # the spec: the modality's weight, the text that drove on-the-fly
        # segmentation, and the region mask that limited where it applied.
        "control_weight": float(first.get("control_weight", 0.0) or 0.0),
        "control_prompt": str(first.get("control_prompt") or ""),
        "mask_prompt": str(first.get("mask_prompt") or ""),
        "control_uris": first.get("control_uris", {}),
        "content_guardrails_enabled": bool(
            first.get("content_guardrails_enabled", True)
        ),
        "protected_chroma": first.get("protected_chroma", {"mode": "off"}),
        "refinement": first.get("refinement", {}),
        "effective_control_weight": first.get("effective_control_weight"),
        "effective_guidance": first.get("effective_guidance"),
        "inference_seed": first.get("inference_seed"),
        "variants": [
            {
                "clip": c.get("clip", ""),
                "variant_index": int(c.get("variant_index", index) or 0),
                "variables": c.get("variables", {}),
                "prompt": str((c.get("variables") or {}).get("prompt") or ""),
                "frame_count": int(c.get("frame_count", 0) or 0),
                "augmented_video_uri": c.get("augmented_video_uri", ""),
                "control_uris": c.get("control_uris", {}),
                "protected_chroma": c.get("protected_chroma", {"mode": "off"}),
                "refinement": c.get("refinement", {}),
                "effective_control_weight": c.get("effective_control_weight"),
                "effective_guidance": c.get("effective_guidance"),
                "inference_seed": c.get("inference_seed"),
            }
            for index, c in enumerate(clips)
        ],
    }
    if shards is not None:
        manifest["shards"] = shards
    if attempt_id:
        manifest["attempt_id"] = str(attempt_id)
    return manifest


def write_run_manifest(
    clips: list[dict[str, Any]],
    output_uri: str,
    *,
    run_id: str = "",
    storage_client: Any = None,
    variant_parallelism: int = 1,
    node_count: int = 1,
    shards: list[dict[str, Any]] | None = None,
    attempt_id: str = "",
    publication_claim_etag: str = "",
    publication_generation: int = 0,
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write the run-level ``cosmos_augmented/manifest.json`` listing every clip
    produced by the (possibly multi-variant) augment stage; return the manifest.

    ``clips`` are the descriptors returned by :func:`publish_transfer_clip`.
    ``variant_parallelism`` records how many GPUs the fan-out ran across (1 ==
    sequential) so provenance can surface the multi-GPU amplification.
    """

    if not output_uri.startswith("s3://"):
        raise ValueError(f"output_uri must be an s3:// prefix, got: {output_uri!r}")
    from npa.clients.storage import StorageClient

    expected = max(1, int(node_count or 1))
    conditional_parts = (
        bool(publication_claim_etag),
        int(publication_generation) > 0,
        bool(str(attempt_id or "").strip()),
    )
    if expected > 1 and not all(conditional_parts):
        raise ValueError(
            "multi-node shard merge requires the rank-0 publication claim fence"
        )
    if any(conditional_parts) and not all(conditional_parts):
        raise ValueError(
            "conditional run publication requires attempt_id, claim ETag, and "
            "positive generation together"
        )
    client = storage_client or StorageClient.from_environment()
    manifest = build_run_manifest(
        clips,
        run_id=run_id,
        variant_parallelism=variant_parallelism,
        node_count=node_count,
        shards=shards,
        attempt_id=attempt_id,
        failures=failures,
    )
    if publication_claim_etag:
        if not attempt_id or int(publication_generation) < 1:
            raise ValueError(
                "conditional run publication requires attempt_id and positive generation"
            )
        manifest[PUBLICATION_GENERATION_FIELD] = int(publication_generation)
        manifest["logical_publication"] = "conditional"
        from npa.clients.storage import StoragePreconditionFailed

        current = client.read_bytes_with_etag(transfer_manifest_uri_for(output_uri))
        try:
            import json as _json

            current_document = (
                _json.loads(current[0].decode("utf-8")) if current is not None else {}
            )
        except (UnicodeDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "multi-node publication fence is unreadable before finalization"
            ) from exc
        if not (
            current is not None
            and current[1] == publication_claim_etag
            and isinstance(current_document, dict)
            and current_document.get("status") == PUBLICATION_CLAIM_STATUS
            and str(current_document.get("attempt_id") or "") == str(attempt_id)
            and str(current_document.get("run_id") or "") == str(run_id or "")
            and int(current_document.get("node_count", 0) or 0)
            == int(node_count)
            and int(current_document.get(PUBLICATION_GENERATION_FIELD, 0) or 0)
            == int(publication_generation)
        ):
            raise RuntimeError(
                "multi-node augment publication fence is absent, contradictory, "
                "or already superseded"
            )
        for field in (
            "logical_wave_id",
            "membership_digest",
            "scheduler_fence_sequence",
            "scheduler_fence_attempt",
            "scheduler_launch_id",
        ):
            value = current_document.get(field)
            if value in (None, ""):
                raise RuntimeError(
                    f"multi-node publication claim is missing required {field}"
                )
            manifest[field] = value

        try:
            client.put_bytes_conditional(
                _json_bytes(manifest),
                transfer_manifest_uri_for(output_uri),
                if_match=str(publication_claim_etag),
                content_type="application/json",
            )
        except StoragePreconditionFailed as exc:
            raise RuntimeError(
                "multi-node augment publication was fenced by a newer recovery "
                f"attempt; refusing to publish stale attempt {attempt_id}"
            ) from exc
    else:
        _upload_json(client, manifest, transfer_manifest_uri_for(output_uri))
    return manifest


def shard_manifest_uri_for(output_uri: str, rank: int, *, attempt_id: str) -> str:
    """Return the shard-manifest URI one node of a gang-scheduled augment writes."""

    attempt_uri = attempt_output_uri_for(output_uri, attempt_id)
    return f"{attempt_uri}/{SHARD_MANIFEST_PREFIX}{int(rank)}.json"


def write_shard_manifest(
    clips: list[dict[str, Any]],
    output_uri: str,
    *,
    run_id: str = "",
    rank: int,
    node_count: int,
    variant_parallelism: int = 1,
    variant_total: int = 0,
    attempt_id: str,
    scheduler_fence_sequence: int,
    scheduler_fence_attempt: int,
    scheduler_launch_id: str,
    logical_wave_id: str,
    publication_generation: int,
    storage_client: Any = None,
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Publish ONE node's share of a multi-node augment as a shard manifest.

    Every node of the gang writes its own file, so the nodes never contend for a
    single key. ``clips`` carry their global ``variant_index``, which is what lets
    :func:`merge_shard_manifests` restore the sampled combo order.
    """

    if not output_uri.startswith("s3://"):
        raise ValueError(f"output_uri must be an s3:// prefix, got: {output_uri!r}")
    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    normalized_attempt_id = _validated_attempt_id(attempt_id)
    fence = (int(scheduler_fence_sequence), int(scheduler_fence_attempt))
    launch_id = str(scheduler_launch_id or "").strip()
    logical_wave = str(logical_wave_id or "").strip()
    generation = int(publication_generation)
    if min(fence) < 1 or not launch_id or not logical_wave or generation < 1:
        raise ValueError("multi-node shard requires the complete scheduler fence")
    attempt_prefix = attempt_output_uri_for(output_uri, normalized_attempt_id) + "/"
    invalid_uris = [
        str(clip.get("augmented_video_uri") or "")
        for clip in clips
        if not str(clip.get("augmented_video_uri") or "").startswith(attempt_prefix)
    ]
    if invalid_uris:
        raise ValueError(
            "multi-node shard descriptors must point inside their attempt-scoped "
            "output prefix"
        )
    variant_failures = list(failures or [])
    shard = {
        "schema": SHARD_MANIFEST_SCHEMA,
        "mode": TRANSFER_MANIFEST_MODE,
        "status": TRANSFER_MANIFEST_STATUS,
        "run_id": run_id,
        "attempt_id": normalized_attempt_id,
        "logical_wave_id": logical_wave,
        PUBLICATION_GENERATION_FIELD: generation,
        "scheduler_fence_sequence": fence[0],
        "scheduler_fence_attempt": fence[1],
        "scheduler_launch_id": launch_id,
        "rank": int(rank),
        "node_count": max(1, int(node_count or 1)),
        "variant_parallelism": max(1, int(variant_parallelism or 1)),
        "variant_total": max(0, int(variant_total or 0)),
        "variant_count": len(clips),
        "attempted_variant_count": len(clips) + len(variant_failures),
        "failed_variant_count": len(variant_failures),
        "failed_variants": variant_failures,
        "clips": [c.get("clip", "") for c in clips],
        "clip_descriptors": clips,
    }
    _upload_json(
        client,
        shard,
        shard_manifest_uri_for(output_uri, rank, attempt_id=normalized_attempt_id),
    )
    return shard


def merge_shard_manifests(
    output_uri: str,
    *,
    run_id: str = "",
    node_count: int,
    attempt_id: str,
    storage_client: Any = None,
    timeout_s: float | None = None,
    poll_interval_s: float = 15.0,
    progress_interval_s: float = 60.0,
    sleep: Any = None,
    monotonic: Any = None,
    progress: Any = None,
    publication_claim_etag: str = "",
    publication_generation: int = 0,
) -> dict[str, Any]:
    """Wait for every node's shard manifest, then write the run manifest.

    Called by rank 0 only. The gang's nodes run the same augment command
    concurrently, so this is the join: it fetches ``manifest-rank-<k>.json`` for
    every expected rank from the attempt-private prefix, orders the clips by their global variant index, and
    writes the same ``manifest.json`` a single-node run would have produced. A
    rank that never reports must not become a manifest that silently omits its
    variants.

    The wait is unbounded by default, because there is no defensible duration to
    cap it at: a sibling's remaining work is however long its diffusions take, and
    a deadline short enough to be useful would fail runs that were about to
    succeed. Missing ranks and elapsed time are emitted periodically. Set
    ``NPA_COSMOS_SHARD_JOIN_TIMEOUT_S`` (or pass ``timeout_s``) when an operator
    does want a deadline, and the failure then names the missing ranks.
    """

    import json as _json
    import math as _math
    import time as _time
    import sys as _sys

    if not output_uri.startswith("s3://"):
        raise ValueError(f"output_uri must be an s3:// prefix, got: {output_uri!r}")
    from npa.clients.storage import StorageClient

    expected = max(1, int(node_count or 1))
    normalized_attempt_id = _validated_attempt_id(attempt_id)
    if not publication_claim_etag or int(publication_generation) < 1:
        raise ValueError(
            "multi-node shard merge requires the rank-0 publication claim fence"
        )
    client = storage_client or StorageClient.from_environment()

    def _read_authoritative_claim() -> dict[str, Any]:
        """Read and validate the canonical fence for every join iteration."""

        claim_current = client.read_bytes_with_etag(
            transfer_manifest_uri_for(output_uri)
        )
        try:
            claim_document = (
                _json.loads(claim_current[0].decode("utf-8"))
                if claim_current is not None
                else {}
            )
        except (UnicodeDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError("multi-node shard-join fence is unreadable") from exc
        if not (
            claim_current is not None
            and claim_current[1] == publication_claim_etag
            and isinstance(claim_document, dict)
            and claim_document.get("schema") == TRANSFER_MANIFEST_SCHEMA
            and claim_document.get("mode") == TRANSFER_MANIFEST_MODE
            and claim_document.get("status") == PUBLICATION_CLAIM_STATUS
            and str(claim_document.get("run_id") or "") == str(run_id or "")
            and str(claim_document.get("attempt_id") or "")
            == normalized_attempt_id
            and int(claim_document.get("node_count", 0) or 0) == expected
            and int(claim_document.get(PUBLICATION_GENERATION_FIELD, 0) or 0)
            == int(publication_generation)
        ):
            raise RuntimeError(
                "multi-node shard join publication fence was superseded or is no "
                "longer authoritative"
            )
        return claim_document

    claim_document = _read_authoritative_claim()
    expected_shard_identity = {
        "logical_wave_id": str(claim_document.get("logical_wave_id") or ""),
        PUBLICATION_GENERATION_FIELD: int(publication_generation),
        "scheduler_fence_sequence": int(
            claim_document.get("scheduler_fence_sequence", 0) or 0
        ),
        "scheduler_fence_attempt": int(
            claim_document.get("scheduler_fence_attempt", 0) or 0
        ),
        "scheduler_launch_id": str(claim_document.get("scheduler_launch_id") or ""),
    }
    if (
        not expected_shard_identity["logical_wave_id"]
        or not expected_shard_identity["scheduler_launch_id"]
        or int(expected_shard_identity["scheduler_fence_sequence"]) < 1
        or int(expected_shard_identity["scheduler_fence_attempt"]) < 1
    ):
        raise RuntimeError("multi-node publication claim has an incomplete scheduler fence")
    waiter = sleep or _time.sleep
    clock = monotonic or _time.monotonic
    reporter = progress or (
        lambda message: print(message, file=_sys.stderr, flush=True)
    )
    limit = timeout_s
    if limit is None:
        env_limit = str(os.environ.get("NPA_COSMOS_SHARD_JOIN_TIMEOUT_S", "")).strip()
        try:
            limit = float(env_limit) if env_limit else None
        except ValueError as exc:
            raise ValueError(
                "NPA_COSMOS_SHARD_JOIN_TIMEOUT_S must be a non-negative number"
            ) from exc
    if limit is not None and (
        not _math.isfinite(float(limit)) or float(limit) < 0
    ):
        raise ValueError("shard join timeout must be finite and non-negative")
    started = clock()
    deadline = None if limit is None else started + float(limit)
    last_progress: float | None = None
    shards: dict[int, dict[str, Any]] = {}

    def _is_current_shard(document: Any, rank: int) -> bool:
        """Reject stale/partial objects until the current rank overwrites them."""

        if not isinstance(document, dict):
            return False
        descriptors = document.get("clip_descriptors")
        failures = document.get("failed_variants", [])
        attempt_prefix = (
            attempt_output_uri_for(output_uri, normalized_attempt_id) + "/"
        )
        return bool(
            document.get("schema") == SHARD_MANIFEST_SCHEMA
            and document.get("mode") == TRANSFER_MANIFEST_MODE
            and document.get("status") == TRANSFER_MANIFEST_STATUS
            and str(document.get("run_id") or "") == str(run_id or "")
            and str(document.get("attempt_id") or "") == normalized_attempt_id
            and int(document.get("rank", -1)) == rank
            and int(document.get("node_count", 0)) == expected
            and all(
                document.get(field) == value
                for field, value in expected_shard_identity.items()
            )
            and isinstance(descriptors, list)
            and isinstance(failures, list)
            and int(document.get("variant_count", -1)) == len(descriptors)
            and int(document.get("failed_variant_count", -1)) == len(failures)
            and int(document.get("attempted_variant_count", -1))
            == len(descriptors) + len(failures)
            and all(
                isinstance(item, dict)
                and str(item.get("augmented_video_uri") or "").startswith(
                    attempt_prefix
                )
                for item in descriptors
            )
            and all(
                isinstance(item, dict)
                and int(item.get("variant_index", -1)) >= 0
                and str(item.get("failure_uri") or "").startswith(attempt_prefix)
                for item in failures
            )
        )

    while True:
        # A scheduler-authorized recovery may supersede this attempt while an old
        # leader is waiting for a missing sibling. Re-read the canonical claim on
        # every poll so the old process exits even when the default join has no
        # deadline and can never reach the final compare-and-swap publication.
        current_claim = _read_authoritative_claim()
        if any(
            current_claim.get(field) != value
            for field, value in expected_shard_identity.items()
        ):
            raise RuntimeError(
                "multi-node shard join publication fence is inconsistent with its "
                "scheduler attempt identity"
            )
        for rank in range(expected):
            if rank in shards:
                continue
            # Fetch the exact key rather than listing the prefix: a bucket listing
            # can lag behind a sibling upload. ``None`` is the only missing-rank
            # signal; credentials, endpoint, and permission errors propagate with
            # their provider evidence instead of becoming an unbounded wait.
            current = client.read_bytes_with_etag(
                shard_manifest_uri_for(
                    output_uri, rank, attempt_id=normalized_attempt_id
                )
            )
            if current is None:
                continue
            try:
                candidate = _json.loads(current[0].decode("utf-8"))
                if _is_current_shard(candidate, rank):
                    shards[rank] = candidate
            except (UnicodeDecodeError, TypeError, ValueError):
                # A partial or malformed object is never accepted; retry the same
                # exact key in case a compatible store exposed an in-flight write.
                pass
        if len(shards) == expected:
            break
        now = clock()
        missing = [r for r in range(expected) if r not in shards]
        if (
            last_progress is None
            or now - last_progress >= max(0.0, float(progress_interval_s))
        ):
            reporter(
                "multi-node augment shard join waiting: "
                f"attempt={normalized_attempt_id} missing_ranks={missing} "
                f"received_ranks={sorted(shards)} "
                f"elapsed={max(0.0, now - started):.1f}s "
                f"timeout={'disabled' if limit is None else f'{float(limit):g}s'}"
            )
            last_progress = now
        if deadline is not None and now >= deadline:
            raise RuntimeError(
                "multi-node augment: no shard manifest from rank(s) "
                f"{missing} for attempt {normalized_attempt_id} after "
                f"{float(limit or 0):.0f}s at {output_uri}. Those "
                "nodes did not finish publishing their variants, so the run "
                "manifest would understate the fan-out."
            )
        waiter(max(0.1, float(poll_interval_s)))

    totals = {int(shard.get("variant_total", -1)) for shard in shards.values()}
    if len(totals) != 1 or next(iter(totals), -1) < 1:
        raise RuntimeError(
            "multi-node augment: shard manifests disagree on the total variant "
            f"count for run {run_id!r}: {sorted(totals)}"
        )
    variant_total = next(iter(totals))
    ordered = sorted(
        (clip for shard in shards.values() for clip in shard.get("clip_descriptors", [])),
        key=lambda c: int(c.get("variant_index", 0) or 0),
    )
    ordered_failures = sorted(
        (
            failure
            for shard in shards.values()
            for failure in shard.get("failed_variants", [])
        ),
        key=lambda failure: int(failure.get("variant_index", -1)),
    )
    indices = sorted(
        [int(clip.get("variant_index", -1)) for clip in ordered]
        + [int(failure.get("variant_index", -1)) for failure in ordered_failures]
    )
    if indices != list(range(variant_total)):
        raise RuntimeError(
            "multi-node augment: shard manifests do not cover every variant exactly "
            f"once for run {run_id!r}; expected 0..{variant_total - 1}, got {indices}"
        )
    if not ordered:
        raise RuntimeError(
            "multi-node augment: every variant failed independently; failure "
            "shards were preserved and no empty run manifest was published"
        )
    return write_run_manifest(
        ordered,
        output_uri,
        run_id=run_id,
        storage_client=client,
        variant_parallelism=sum(
            max(1, int(shard.get("variant_parallelism", 1) or 1))
            for shard in shards.values()
            if int(shard.get("attempted_variant_count", 0) or 0) > 0
        )
        or 1,
        node_count=expected,
        shards=[
            {
                "rank": int(shard.get("rank", rank) or 0),
                "variant_count": int(shard.get("variant_count", 0) or 0),
                "attempted_variant_count": int(
                    shard.get("attempted_variant_count", 0) or 0
                ),
                "failed_variant_count": int(
                    shard.get("failed_variant_count", 0) or 0
                ),
                "variant_parallelism": max(1, int(shard.get("variant_parallelism", 1) or 1)),
                "clips": list(shard.get("clips", [])),
                "attempt_id": str(shard.get("attempt_id") or ""),
            }
            for rank, shard in sorted(shards.items())
        ],
        attempt_id=normalized_attempt_id,
        publication_claim_etag=publication_claim_etag,
        publication_generation=publication_generation,
        failures=ordered_failures,
    )


def publish_transfer_to_s3(
    transfer: dict[str, Any],
    output_uri: str,
    *,
    run_id: str = "",
    variables: dict[str, Any] | None = None,
    clip_name: str = "",
    max_frames: int = 8,
    frames_output_uri: str = "",
    control_output_uri: str = "",
    require_frames: bool = False,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Upload a single real Cosmos-Transfer2.5 result to S3 in the per-clip layout
    that ``data_factory_stages.curate`` and ``data_factory_viz.build_run_rrd``
    consume, plus the run-level manifest. Single-variant convenience wrapper
    around :func:`publish_transfer_clip` + :func:`write_run_manifest`; multi-variant
    callers publish each clip themselves and write one combined manifest.
    """

    clip = publish_transfer_clip(
        transfer,
        output_uri,
        run_id=run_id,
        clip_name=clip_name,
        variables=variables,
        max_frames=max_frames,
        frames_output_uri=frames_output_uri,
        control_output_uri=control_output_uri,
        require_frames=require_frames,
        storage_client=storage_client,
    )
    return write_run_manifest([clip], output_uri, run_id=run_id, storage_client=storage_client)


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".webp"}
_PERTURBATIONS = ("lighting", "contrast", "color", "blur")


def transfer_manifest_uri_for(output_uri: str) -> str:
    """Return the durable manifest URI written by a real transfer publish."""

    return output_uri.rstrip("/") + "/" + TRANSFER_MANIFEST_FILENAME


def augmented_frames_index_uri_for(output_uri: str) -> str:
    """Return the index URI written by reference augmentation."""

    return output_uri.rstrip("/") + "/" + AUGMENTED_FRAMES_INDEX


def _apply_perturbation(image: Any, perturbation: str, *, seed: int) -> Any:
    """Apply one deterministic, real image transform (a perturbation ControlNet
    would drive in the full model; here a genuine PIL transform, not a no-op)."""

    import random

    from PIL import ImageEnhance, ImageFilter

    rng = random.Random(seed)
    if perturbation == "lighting":
        return ImageEnhance.Brightness(image).enhance(rng.uniform(0.55, 1.6))
    if perturbation == "contrast":
        return ImageEnhance.Contrast(image).enhance(rng.uniform(0.6, 1.7))
    if perturbation == "color":
        return ImageEnhance.Color(image).enhance(rng.uniform(0.3, 1.9))
    if perturbation == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 2.2)))
    return image


def _collect_source_images(src_dir: Path, max_inputs: int) -> list[Path]:
    return sorted(
        (p for p in src_dir.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )[:max_inputs]


def reference_augment_frames(
    input_uri: str,
    output_uri: str,
    *,
    run_id: str = "",
    variants_per_frame: int = 2,
    max_inputs: int = 8,
) -> dict[str, Any]:
    """Produce real augmented image frames without the heavy Cosmos model.

    Downloads the source frames from ``input_uri``, applies genuine per-frame PIL
    augmentations (lighting / contrast / color / blur), and writes/uploads the
    augmented PNGs to ``output_uri`` so downstream stages (e.g. VLM critique) get
    real image frames instead of a descriptor stub. Used when the
    Cosmos-Transfer2.5 runtime image is not present; ``--execute`` runs the real
    model instead.

    ``s3://`` URIs are read/written via :class:`StorageClient`; any other value is
    treated as a local directory (keeps the function unit-testable without S3).
    """

    import json
    import tempfile

    from PIL import Image

    def _is_s3(uri: str) -> bool:
        return uri.strip().startswith("s3://")

    with tempfile.TemporaryDirectory() as tmp:
        src_dir = Path(tmp) / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(tmp) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        storage = None
        if _is_s3(input_uri) or _is_s3(output_uri):
            from npa.clients.storage import StorageClient

            storage = StorageClient.from_environment()

        if _is_s3(input_uri):
            assert storage is not None
            storage.download_directory(input_uri, str(src_dir))
        else:
            local_src = Path(input_uri.replace("local://", "").replace("file://", ""))
            if local_src.is_dir():
                for item in local_src.rglob("*"):
                    if item.is_file():
                        dest = src_dir / item.relative_to(local_src)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest)
            elif local_src.is_file():
                # A single local image file is a valid source too.
                shutil.copy2(local_src, src_dir / local_src.name)

        sources = _collect_source_images(src_dir, max_inputs)
        if not sources:
            raise RuntimeError(
                f"cosmos2 transfer: no source images found under {input_uri!r}; "
                "expected at least one .png/.jpg frame to augment."
            )

        if _is_s3(output_uri):
            frames_uri = output_uri
            dest_dir = None
        else:
            dest_dir = Path(output_uri.replace("local://", "").replace("file://", ""))
            # Preserve an explicit local scheme in every returned frame URI so
            # ``frames[].uri`` and ``index_uri`` use the same address space.
            # Plain filesystem inputs remain plain paths for compatibility.
            frames_uri = output_uri.rstrip("/")

        index: list[dict[str, Any]] = []
        frame_no = 0
        for src in sources:
            base = Image.open(src).convert("RGB")
            for variant in range(max(1, variants_per_frame)):
                perturbation = _PERTURBATIONS[frame_no % len(_PERTURBATIONS)]
                augmented = _apply_perturbation(base, perturbation, seed=frame_no)
                name = f"frame-{frame_no:05d}.png"
                augmented.save(out_dir / name)
                index.append(
                    {
                        "frame_id": f"frame-{frame_no:05d}",
                        "perturbation": perturbation,
                        "source": src.name,
                        "uri": f"{frames_uri.rstrip('/')}/{name}",
                        "variant": variant,
                    }
                )
                frame_no += 1

        (out_dir / AUGMENTED_FRAMES_INDEX).write_text(
            json.dumps(
                {
                    "schema": AUGMENTED_FRAMES_SCHEMA,
                    "run_id": run_id,
                    "frame_count": frame_no,
                    "frames": index,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        if _is_s3(output_uri):
            assert storage is not None
            storage.upload_directory(str(out_dir), output_uri)
        else:
            assert dest_dir is not None
            dest_dir.mkdir(parents=True, exist_ok=True)
            for item in out_dir.iterdir():
                shutil.copy2(item, dest_dir / item.name)

    return {
        "augmented_frames_uri": frames_uri,
        "frames": index,
        "index_uri": augmented_frames_index_uri_for(output_uri),
        "frame_count": frame_no,
        "source_frame_count": len(sources),
    }


__all__ = [
    "ATTEMPT_PREFIX",
    "AUGMENTED_FRAMES_INDEX",
    "AUGMENTED_FRAMES_SCHEMA",
    "CONTROL_MODALITY_MODELS",
    "CONTROL_PROMPT_MODALITIES",
    "ControlModalityError",
    "FrameExtractionError",
    "INPUT_AUTO_CONTROLS",
    "INPUT_CONTROLS",
    "ProtectedChromaError",
    "REFERENCE_AUGMENT_MODE",
    "REFERENCE_AUGMENT_STATUS",
    "PUBLICATION_CLAIM_STATUS",
    "PUBLICATION_GENERATION_FIELD",
    "SHARD_MANIFEST_PREFIX",
    "SHARD_MANIFEST_SCHEMA",
    "TRANSFER_MANIFEST_FILENAME",
    "TRANSFER_MANIFEST_MODE",
    "TRANSFER_MANIFEST_SCHEMA",
    "preserve_source_chroma",
    "TRANSFER_MANIFEST_STATUS",
    "augmented_frames_index_uri_for",
    "attempt_output_uri_for",
    "build_run_manifest",
    "claim_run_publication",
    "cosmos_transfer_available",
    "cosmos_transfer_repo",
    "ensure_env",
    "extract_frames",
    "merge_shard_manifests",
    "publish_transfer_clip",
    "publish_transfer_failure",
    "publish_transfer_to_s3",
    "reference_augment_frames",
    "resolve_control_modality",
    "resolve_control_weight",
    "run_cosmos_transfer",
    "shard_manifest_uri_for",
    "transfer_manifest_uri_for",
    "write_run_manifest",
    "write_shard_manifest",
]
