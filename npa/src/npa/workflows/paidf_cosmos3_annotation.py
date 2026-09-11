"""Caption every accepted Cosmos 3 variant and bind captions to evaluated bytes."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
from typing import Any

from npa.workflows.paidf_cosmos3 import (
    PaidfCosmos3Error,
    _download_source,
    _extract_frames,
    _read_json,
    _validated_quality_status,
    _write_json,
    validate_committed_augment_manifest,
)
from npa.workflows.paidf_cosmos3_media import video_sha256


def _accepted_variants(root: str, storage: Any) -> list[dict[str, Any]]:
    manifest = _read_json(root + "cosmos_augmented/manifest.json", storage=storage)
    variants = validate_committed_augment_manifest(manifest, root + "cosmos_augmented/")
    report = _read_json(root + "grade/cosmos_evaluator.json", storage=storage)
    disposition = _read_json(root + "grade/quality_disposition.json", storage=storage)
    if (_validated_quality_status(disposition) != "accepted"
            or not isinstance(report, dict) or report.get("status") != "completed"
            or report.get("passed") is not True):
        raise PaidfCosmos3Error("annotation requires complete accepted evaluation")
    clips = {item["clip_id"]: item for item in report["clips"]}
    if len(clips) != len(variants) or report.get("alignment_mode") != "required":
        raise PaidfCosmos3Error("annotation requires an aligned evaluation of every variant")
    for variant in variants:
        clip = clips.get(variant["clip"], {})
        alignment = variant.get("temporal_alignment")
        if (not isinstance(alignment, dict) or clip.get("passed") is not True
                or clip.get("temporal_alignment") != alignment):
            raise PaidfCosmos3Error("accepted evaluation does not match the current generation")
    return variants


def _caption_variant(variant: dict[str, Any], output: str, model: str, max_images: int,
                     max_tokens: int, storage: Any, call: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clip = variant["clip"]
    with tempfile.TemporaryDirectory(prefix="npa-paidf-c3-caption-") as temporary:
        root = Path(temporary)
        video = _download_source(variant["augmented_video_uri"], root / "video.mp4", storage=storage)
        expected = variant["temporal_alignment"]["generated_sha256"]
        if video_sha256(video) != expected:
            raise PaidfCosmos3Error("caption input no longer matches the evaluated video")
        _extract_frames(video, root / "frames")
        result = call(input_path=str(root / "frames"), output_path=output, model=model,
                      max_images=max_images, max_tokens=max_tokens)
    items = [asdict(item) for item in result.captions]
    if result.status != "completed" or not items or any(not item["caption"].strip() for item in items):
        raise PaidfCosmos3Error("every accepted variant needs nonempty real captions")
    captions = [{**item, "image": f"{clip}/{item['image']}"} for item in items]
    return {"clip": clip, "image_count": len(items), "video_sha256": expected}, captions


def annotate_accepted(
    run_root_uri: str, output_uri: str, model: str, max_images: str | int = 8,
    max_tokens: str | int = 512, *, storage: Any = None, captioner: Any = None,
) -> dict[str, Any]:
    """Caption a separate frame sample from every accepted variant.

    Args:
        run_root_uri: Run prefix with generation and accepted quality evidence.
        output_uri: Destination prefix for the combined captions.json.
        model: Hosted vision model.
        max_images: Maximum caption frames per variant.
        max_tokens: Maximum generated tokens per frame.
        storage: Optional artifact client for isolated testing.
        captioner: Optional caption implementation for isolated testing.
    Returns:
        Captions with per-variant coverage and evaluated video hashes.
    Raises:
        PaidfCosmos3Error: Acceptance, coverage or nonempty captions are missing.
        TokenFactoryToolError: Real hosted captioning fails.
    """
    from npa.workbench.token_factory import caption_images

    variants = _accepted_variants(run_root_uri.rstrip("/") + "/", storage)
    call = captioner or caption_images
    records, captions = [], []
    for variant in variants:
        record, items = _caption_variant(variant, output_uri, model, int(max_images), int(max_tokens), storage, call)
        records.append(record)
        captions.extend(items)
    payload = {"schema": "npa.token_factory.captions.v1", "status": "completed", "model": model,
               "image_count": len(captions), "captions": captions, "variants": records}
    _write_json(payload, output_uri.rstrip("/") + "/captions.json", storage=storage)
    return payload


def _validate_caption_coverage(report: dict[str, Any], expected: dict[str, str]) -> None:
    records = report.get("variants", [])
    actual = {item["clip"]: item["video_sha256"] for item in records if item.get("image_count", 0) > 0}
    captions = report.get("captions", [])
    if (report.get("status") != "completed" or len(records) != len(expected) or actual != expected
            or not captions or any(not item.get("caption", "").strip() for item in captions)):
        raise PaidfCosmos3Error("finalization requires nonempty captions for every accepted variant")
    images = [item.get("image", "") for item in captions]
    counts = {clip: sum(image.startswith(clip + "/") for image in images) for clip in expected}
    if (len(set(images)) != len(images) or sum(counts.values()) != len(images)
            or report.get("image_count") != len(images)
            or any(counts[record["clip"]] != record["image_count"] for record in records)):
        raise PaidfCosmos3Error("caption records do not prove coverage of every accepted variant")


def validate_annotation(root: str, storage: Any) -> dict[str, Any]:
    """Require complete caption coverage of the current accepted generation.

    Args:
        root: Run prefix ending with a slash.
        storage: Artifact client or None for normal resolution.
    Returns:
        Validated caption report.
    Raises:
        PaidfCosmos3Error: Captions are missing, empty, stale or incomplete.
    """
    variants = _accepted_variants(root, storage)
    report = _read_json(root + "labeled_augmented/captions.json", storage=storage)
    expected = {item["clip"]: item["temporal_alignment"]["generated_sha256"] for item in variants}
    _validate_caption_coverage(report, expected)
    return report
