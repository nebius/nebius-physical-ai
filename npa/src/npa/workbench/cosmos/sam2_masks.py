"""Real Meta SAM 2 video-mask generation for input-conditioned augmentation.

The module is imported only inside the Cosmos Transfer GPU image. That image
installs the official Meta source at an immutable commit; model
checkpoints are Apache-2.0 and fetched at runtime into the operator's existing
model cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlparse


MASK_SCHEMA = "npa.paidf.segmentation.v1"
SAM2_ENGINE = "meta-sam2-upstream"
SAM2_SOURCE_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_DISTRIBUTION = "SAM-2"
SAM2_DISTRIBUTION_VERSION = "1.0"
DEFAULT_SAM2_MODEL = "facebook/sam2.1-hiera-tiny"
DEFAULT_SAM2_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
SAM2_LICENSE = "Apache-2.0"
SAM2_LICENSE_URL = (
    "https://github.com/facebookresearch/sam2/blob/"
    f"{SAM2_SOURCE_REVISION}/LICENSE"
)


class Sam2MaskError(RuntimeError):
    """SAM2 could not produce the exact frame-aligned mask contract."""


@dataclass(frozen=True)
class Sam2MaskConfig:
    mode: str = "sam2-auto"
    model_id: str = DEFAULT_SAM2_MODEL
    model_revision: str = DEFAULT_SAM2_REVISION
    points_per_side: int = 16
    predicted_iou_threshold: float = 0.86
    stability_threshold: float = 0.92
    min_area_fraction: float = 0.002
    max_area_fraction: float = 0.65
    max_objects: int = 6

    def validate(self) -> None:
        if self.mode != "sam2-auto":
            raise Sam2MaskError("segmentation mode must be off or sam2-auto")
        if not self.model_id.strip() or not self.model_revision.strip():
            raise Sam2MaskError("SAM2 model id and immutable revision are required")
        if not self.model_id.startswith("facebook/sam2"):
            raise Sam2MaskError(
                "SAM2 model must be an official facebook/sam2 checkpoint"
            )
        if re.fullmatch(r"[0-9a-f]{40}", self.model_revision) is None:
            raise Sam2MaskError("SAM2 model revision must be an immutable 40-hex SHA")
        if not 4 <= self.points_per_side <= 64:
            raise Sam2MaskError("SAM2 points_per_side must be within 4..64")
        if not 0.0 <= self.predicted_iou_threshold <= 1.0:
            raise Sam2MaskError("SAM2 predicted-IoU threshold must be within 0..1")
        if not 0.0 <= self.stability_threshold <= 1.0:
            raise Sam2MaskError("SAM2 stability threshold must be within 0..1")
        if not 0.0 < self.min_area_fraction < self.max_area_fraction <= 1.0:
            raise Sam2MaskError(
                "SAM2 area fractions must satisfy 0 < min < max <= 1"
            )
        if not 1 <= self.max_objects <= 32:
            raise Sam2MaskError("SAM2 max_objects must be within 1..32")

    def public_contract(self) -> dict[str, Any]:
        """Return non-secret, reproducibility-relevant settings."""

        return {
            "mode": self.mode,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "points_per_side": self.points_per_side,
            "predicted_iou_threshold": self.predicted_iou_threshold,
            "stability_threshold": self.stability_threshold,
            "min_area_fraction": self.min_area_fraction,
            "max_area_fraction": self.max_area_fraction,
            "max_objects": self.max_objects,
        }


@dataclass(frozen=True)
class Sam2MaskResult:
    manifest: dict[str, Any]
    manifest_path: Path
    masks_dir: Path


def load_published_sam2_masks(
    output_uri: str,
    target_dir: str | Path,
    *,
    config: Sam2MaskConfig,
    run_id: str,
    storage_client: Any,
) -> Sam2MaskResult | None:
    """Reuse one immutable, exact-config mask contract from a prior retry."""

    config.validate()
    if not run_id.strip():
        raise Sam2MaskError("SAM2 mask reuse requires a non-empty run id")
    if not output_uri.startswith("s3://"):
        raise Sam2MaskError("SAM2 output URI must be an s3:// prefix")
    base = output_uri.rstrip("/") + "/"
    bucket, key = _split_s3(f"{base}manifest.json")
    try:
        response = storage_client.s3.get_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001
        if _missing_s3(exc):
            return None
        raise Sam2MaskError("could not inspect the published SAM2 contract") from exc
    body = response.get("Body")
    try:
        raw = body.read(1_000_001) if body is not None else b""
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if len(raw) > 1_000_000:
        raise Sam2MaskError("published SAM2 manifest exceeds the safe size limit")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Sam2MaskError("published SAM2 manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise Sam2MaskError("published SAM2 manifest must be a JSON object")
    expected = {
        "schema": MASK_SCHEMA,
        "status": "executed",
        "engine": SAM2_ENGINE,
        "component": "Meta Segment Anything Model 2",
        "component_version": SAM2_DISTRIBUTION_VERSION,
        "component_source": "https://github.com/facebookresearch/sam2",
        "component_revision": SAM2_SOURCE_REVISION,
        "license": {"spdx": SAM2_LICENSE, "url": SAM2_LICENSE_URL},
        "config": config.public_contract(),
        "manifest_uri": f"{base}manifest.json",
        "masks_uri": f"{base}masks/",
    }
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise Sam2MaskError(
            "published SAM2 contract differs from the requested immutable configuration"
        )
    try:
        frame_count = int(manifest["frame_count"])
        width = int(manifest["width"])
        height = int(manifest["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Sam2MaskError("published SAM2 manifest has invalid media bounds") from exc
    if not 1 <= frame_count <= 100_000 or not (
        1 <= width <= 16_384 and 1 <= height <= 16_384
    ):
        raise Sam2MaskError("published SAM2 manifest has unsafe media bounds")
    try:
        object_count = int(manifest["object_count"])
        coverage = manifest["mask_coverage"]
        coverage_values = [
            float(coverage[name]) for name in ("mean", "min", "max")
        ]
        runtime_seconds = float(manifest["runtime"]["seconds"])
        frames_per_second = float(manifest["runtime"]["frames_per_second"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Sam2MaskError("published SAM2 manifest has invalid evidence") from exc
    if (
        not 1 <= object_count <= config.max_objects
        or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coverage_values)
        or not 0.0 < coverage_values[0] < 1.0
        or coverage_values[2] <= 0.0
        or coverage_values[1] >= 1.0
        or not math.isfinite(runtime_seconds)
        or runtime_seconds < 0.0
        or not math.isfinite(frames_per_second)
        or frames_per_second <= 0.0
        or manifest["runtime"].get("device") != "cuda"
        or manifest.get("lineage")
        != {
            "source_kind": "frame-aligned-video",
            "source_frame_count": frame_count,
            "source_run_id": run_id,
            "mask_pattern": f"{base}masks/mask-%06d.png",
        }
    ):
        raise Sam2MaskError("published SAM2 manifest has invalid evidence")

    target = Path(target_dir)
    masks_dir = target / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for index in range(frame_count):
        local = masks_dir / f"mask-{index:06d}.png"
        mask_bucket, mask_key = _split_s3(f"{base}masks/{local.name}")
        try:
            storage_client.s3.download_file(mask_bucket, mask_key, str(local))
            _verify_binary_mask(local, width=width, height=height)
        except Sam2MaskError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise Sam2MaskError(
                "published SAM2 contract has a missing or invalid frame mask"
            ) from exc
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return Sam2MaskResult(manifest, manifest_path, masks_dir)


def publish_sam2_masks(
    result: Sam2MaskResult,
    output_uri: str,
    *,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Publish the versioned mask contract and return its public provenance."""

    if not output_uri.startswith("s3://"):
        raise Sam2MaskError("SAM2 output URI must be an s3:// prefix")
    if storage_client is None:
        from npa.clients.storage import StorageClient

        storage_client = StorageClient.from_environment()
    base = output_uri.rstrip("/") + "/"
    masks = sorted(result.masks_dir.glob("mask-*.png"))
    try:
        frame_count = int(result.manifest["frame_count"])
        width = int(result.manifest["width"])
        height = int(result.manifest["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Sam2MaskError("SAM2 manifest has invalid media bounds") from exc
    if len(masks) != frame_count:
        raise Sam2MaskError("SAM2 mask files do not match the manifest frame count")
    for mask in masks:
        _verify_binary_mask(mask, width=width, height=height)
    try:
        for mask in masks:
            storage_client.upload_file(str(mask), f"{base}masks/{mask.name}")
    except Exception as exc:  # noqa: BLE001
        raise Sam2MaskError("could not publish the SAM2 mask contract") from exc
    published = dict(result.manifest)
    published["manifest_uri"] = f"{base}manifest.json"
    published["masks_uri"] = f"{base}masks/"
    published["lineage"] = {
        **dict(published.get("lineage") or {}),
        "mask_pattern": f"{base}masks/mask-%06d.png",
    }
    result.manifest_path.write_text(
        json.dumps(published, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        storage_client.upload_file(
            str(result.manifest_path), published["manifest_uri"]
        )
    except Exception as exc:  # noqa: BLE001
        raise Sam2MaskError("could not publish the SAM2 mask manifest") from exc
    return published


def generate_sam2_video_masks(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    config: Sam2MaskConfig,
    run_id: str,
) -> Sam2MaskResult:
    """Run upstream SAM2 and emit one binary PNG mask per source frame."""

    config.validate()
    if not run_id.strip():
        raise Sam2MaskError("SAM2 mask generation requires a non-empty run id")
    source = Path(video_path)
    target = Path(output_dir)
    if not source.is_file():
        raise Sam2MaskError("SAM2 source video is missing")
    target.mkdir(parents=True, exist_ok=True)
    frames_dir = target / "frames"
    masks_dir = target / "masks"
    frames_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    started = time.perf_counter()
    frame_paths, width, height = _decode_video_to_jpegs(source, frames_dir)
    if not frame_paths:
        raise Sam2MaskError("SAM2 source video decoded zero frames")

    try:
        import numpy as np
        from PIL import Image
        import torch
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
    except ImportError as exc:
        raise Sam2MaskError(
            "the audited Cosmos Transfer image does not contain the upstream SAM2 runtime"
        ) from exc
    if not torch.cuda.is_available():
        raise Sam2MaskError("SAM2 segmentation requires a visible CUDA GPU")
    component_version = _package_version(SAM2_DISTRIBUTION)
    if component_version != SAM2_DISTRIBUTION_VERSION:
        raise Sam2MaskError(
            "the SAM2 runtime does not match the audited upstream package version"
        )

    config_name, checkpoint = _download_checkpoint(config)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    image_model = build_sam2(
        config_file=config_name,
        ckpt_path=checkpoint,
        device="cuda",
        mode="eval",
    )
    generator = SAM2AutomaticMaskGenerator(
        image_model,
        points_per_side=config.points_per_side,
        pred_iou_thresh=config.predicted_iou_threshold,
        stability_score_thresh=config.stability_threshold,
        crop_n_layers=0,
        output_mode="binary_mask",
    )
    first = np.asarray(Image.open(frame_paths[0]).convert("RGB"))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        proposals = generator.generate(first)
    boxes = _select_automatic_boxes(
        proposals,
        width=width,
        height=height,
        min_area_fraction=config.min_area_fraction,
        max_area_fraction=config.max_area_fraction,
        limit=config.max_objects,
    )
    # Automatic-mask proposals contain full-resolution CPU arrays. Only their
    # selected boxes cross into video propagation, so release them before a
    # second model instance is loaded on the GPU.
    del proposals, generator, image_model
    torch.cuda.empty_cache()
    if not boxes:
        raise Sam2MaskError(
            "SAM2 found no eligible protected regions; adjust the generic mask thresholds"
        )

    predictor = build_sam2_video_predictor(
        config_file=config_name,
        ckpt_path=checkpoint,
        device="cuda",
        mode="eval",
    )
    inference_state = predictor.init_state(video_path=str(frames_dir))
    context = torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), context:
        for object_id, box in enumerate(boxes, start=1):
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=object_id,
                box=np.asarray(box, dtype=np.float32),
            )
        frame_masks: dict[int, Any] = {}
        for frame_index, _object_ids, logits in predictor.propagate_in_video(
            inference_state
        ):
            combined = (logits > 0.0).any(dim=0).squeeze().cpu().numpy()
            frame_masks[int(frame_index)] = combined.astype(bool)

    if set(frame_masks) != set(range(len(frame_paths))):
        raise Sam2MaskError(
            "SAM2 did not return an exact mask for every source frame"
        )
    coverage: list[float] = []
    for index in range(len(frame_paths)):
        mask = frame_masks[index]
        if mask.shape != (height, width):
            mask_image = Image.fromarray(mask.astype("uint8") * 255)
            mask_image = mask_image.resize((width, height), resample=Image.Resampling.NEAREST)
            mask = np.asarray(mask_image) > 127
        coverage.append(float(mask.mean()))
        Image.fromarray(mask.astype("uint8") * 255).save(
            masks_dir / f"mask-{index:06d}.png"
        )
    if not coverage or not 0.0 < (sum(coverage) / len(coverage)) < 1.0:
        raise Sam2MaskError(
            "SAM2 produced an empty or all-frame-invalid mask contract"
        )
    del predictor, inference_state, frame_masks
    torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    manifest = {
        "schema": MASK_SCHEMA,
        "status": "executed",
        "engine": SAM2_ENGINE,
        "component": "Meta Segment Anything Model 2",
        "component_version": component_version,
        "component_source": "https://github.com/facebookresearch/sam2",
        "component_revision": SAM2_SOURCE_REVISION,
        "license": {"spdx": SAM2_LICENSE, "url": SAM2_LICENSE_URL},
        "config": config.public_contract(),
        "frame_count": len(frame_paths),
        "width": width,
        "height": height,
        "object_count": len(boxes),
        "mask_coverage": {
            "mean": sum(coverage) / len(coverage),
            "min": min(coverage),
            "max": max(coverage),
        },
        "runtime": {
            "device": "cuda",
            "seconds": elapsed,
            "frames_per_second": len(frame_paths) / elapsed if elapsed > 0 else None,
        },
        "lineage": {
            "source_kind": "frame-aligned-video",
            "source_frame_count": len(frame_paths),
            "source_run_id": run_id,
            "mask_pattern": "masks/mask-%06d.png",
        },
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return Sam2MaskResult(manifest, manifest_path, masks_dir)


def _decode_video_to_jpegs(
    source: Path, frames_dir: Path
) -> tuple[list[Path], int, int]:
    try:
        import av
    except ImportError as exc:
        raise Sam2MaskError("PyAV is required to decode SAM2 input video") from exc
    paths: list[Path] = []
    width = height = 0
    try:
        with av.open(str(source)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                image = frame.to_image().convert("RGB")
                width, height = image.size
                path = frames_dir / f"{index:06d}.jpg"
                image.save(path, quality=95, subsampling=0)
                paths.append(path)
    except Exception as exc:  # noqa: BLE001
        raise Sam2MaskError("SAM2 could not decode the source video") from exc
    return paths, width, height


def _download_checkpoint(config: Sam2MaskConfig) -> tuple[str, str]:
    try:
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
    except ImportError as exc:
        raise Sam2MaskError("SAM2 checkpoint download support is unavailable") from exc
    try:
        config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[config.model_id]
    except KeyError as exc:
        raise Sam2MaskError(
            "unsupported SAM2 model id; use an upstream facebook/sam2 or sam2.1 checkpoint"
        ) from exc
    try:
        checkpoint = hf_hub_download(
            repo_id=config.model_id,
            filename=checkpoint_name,
            revision=config.model_revision,
        )
    except Exception as exc:  # noqa: BLE001
        raise Sam2MaskError(
            "could not fetch the pinned upstream SAM2 checkpoint"
        ) from exc
    return config_name, checkpoint


def _select_automatic_boxes(
    proposals: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    min_area_fraction: float,
    max_area_fraction: float,
    limit: int,
) -> list[tuple[float, float, float, float]]:
    frame_area = float(width * height)
    ranked: list[tuple[float, tuple[float, float, float, float]]] = []
    for proposal in proposals:
        try:
            area_fraction = float(proposal["area"]) / frame_area
            x, y, box_width, box_height = (float(v) for v in proposal["bbox"])
            predicted_iou = float(proposal.get("predicted_iou", 0.0))
            stability = float(proposal.get("stability_score", 0.0))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if not min_area_fraction <= area_fraction <= max_area_fraction:
            continue
        if box_width <= 0 or box_height <= 0:
            continue
        # Prefer stable, accurately predicted foreground masks while avoiding
        # either tiny speckles or whole-frame/background masks.
        balance = math.sqrt(area_fraction * (1.0 - area_fraction))
        ranked.append(
            (
                predicted_iou * stability * balance,
                (x, y, x + box_width, y + box_height),
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [box for _score, box in ranked[:limit]]


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _verify_binary_mask(path: Path, *, width: int, height: int) -> None:
    """Require an intact, exact-size PNG containing only background/foreground."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise Sam2MaskError("Pillow is required to verify SAM2 masks") from exc
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (width, height):
                raise Sam2MaskError(
                    "SAM2 masks do not match the manifest PNG geometry"
                )
            values = image.convert("L").getcolors(maxcolors=257)
    except Sam2MaskError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise Sam2MaskError("SAM2 mask is not a valid PNG") from exc
    if values is None or any(value not in {0, 255} for _count, value in values):
        raise Sam2MaskError("SAM2 masks must be binary PNGs")


def _split_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise Sam2MaskError("SAM2 contract URI must identify an S3 object")
    return parsed.netloc, parsed.path.lstrip("/")


def _missing_s3(exc: Exception) -> bool:
    response = getattr(exc, "response", {})
    code = str((response.get("Error") or {}).get("Code") or "")
    return code in {"404", "NoSuchKey", "NotFound"}
