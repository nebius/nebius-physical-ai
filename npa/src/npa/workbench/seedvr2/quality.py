"""Publish a fail-closed, review-only quality boundary for SeedVR2 output."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable

from npa.clients.storage import StorageClient
from npa.workbench.cosmos_evaluator.hallucination import (
    DEFAULT_BLUR_KSIZE as MOTION_BLUR_KSIZE,
    DEFAULT_DIST_TOL_PX as MOTION_DIST_TOL_PX,
    DEFAULT_GRAD_THRESH as MOTION_GRAD_THRESH,
    DEFAULT_MORPH_K as MOTION_MORPH_K,
    DEFAULT_THRESHOLD as MOTION_THRESHOLD,
    ENGINE_PORT as MOTION_ENGINE,
    check_hallucination,
)
from npa.workbench.cosmos_evaluator.temporal_consistency import (
    DEFAULT_BLUR_KSIZE as TEMPORAL_BLUR_KSIZE,
    DEFAULT_NOISE_FLOOR as TEMPORAL_NOISE_FLOOR,
    DEFAULT_REGIONS as TEMPORAL_REGIONS,
    DEFAULT_THRESHOLD as TEMPORAL_THRESHOLD,
    ENGINE as TEMPORAL_ENGINE,
    check_temporal_consistency,
)
from npa.workbench.cosmos_evaluator.upstream import CosmosEvaluatorError

from .artifacts import (
    _authorized_result_uri,
    _load_result,
    _validate_artifact_request,
    _validate_candidate_decode,
    _validate_result_context,
)
from .runtime import (
    SeedVR2Error,
    _canonical_json,
    _create_work_directory,
    _decoded_frame_hashes,
    _ensure_artifacts_absent,
    _probe_video,
    _publish_verified,
    _sha256,
    _storage_errors,
    _utc_now,
)
from .schemas import QUALITY_BOUNDARY_SCHEMA, VideoArtifactRequest


def _six_decimal_float(value: Any) -> bool:
    return type(value) is float and math.isfinite(value) and value == round(value, 6)


def _rounded_interval(value: float) -> tuple[float, float]:
    half_unit = 0.5e-6
    return max(0.0, value - half_unit), value + half_unit


def _rounded_value_overlaps(value: float, low: float, high: float) -> bool:
    reported_low, reported_high = _rounded_interval(value)
    epsilon = 1e-12
    return reported_high + epsilon >= low and reported_low - epsilon <= high


def _float32_reduction_tolerance(*values: float) -> float:
    scale = max((abs(value) for value in values), default=0.0)
    if scale == 0.0:
        return 0.0
    _, exponent = math.frexp(scale)
    return 8.0 * math.ldexp(1.0, exponent - 24)


def _temporal_region_configuration() -> list[dict[str, Any]]:
    return [
        {"id": region_id, "bounds": list(bounds)}
        for region_id, bounds in TEMPORAL_REGIONS
    ]


def _require_exact_keys(
    payload: dict[str, Any], expected: set[str], *, label: str
) -> None:
    if set(payload) != expected:
        raise SeedVR2Error(
            f"quality-boundary {label} diagnostic evidence is incomplete"
        )


def _validate_temporal_diagnostic(
    payload: dict[str, Any],
    *,
    run_id: str,
    frame_count: int,
    frame_width: int,
    frame_height: int,
) -> None:
    _require_exact_keys(
        payload,
        {
            "aggregation",
            "blur_ksize",
            "clip_id",
            "engine",
            "frame_counts_match",
            "noise_floor",
            "passed",
            "regions",
            "score",
            "threshold",
            "total_frames",
        },
        label="temporal",
    )
    regions = payload["regions"]
    if (
        payload["clip_id"] != run_id
        or type(payload["passed"]) is not bool
        or type(payload["threshold"]) is not float
        or payload["threshold"] != TEMPORAL_THRESHOLD
        or not _six_decimal_float(payload["score"])
        or not 0.0 <= payload["score"] <= 1.0
        or type(payload["total_frames"]) is not int
        or payload["total_frames"] != frame_count
        or payload["frame_counts_match"] is not True
        or type(payload["noise_floor"]) is not float
        or payload["noise_floor"] != TEMPORAL_NOISE_FLOOR
        or type(payload["blur_ksize"]) is not int
        or payload["blur_ksize"] != TEMPORAL_BLUR_KSIZE
        or payload["engine"] != TEMPORAL_ENGINE
        or payload["aggregation"] != "minimum-region-score"
        or not isinstance(regions, list)
        or len(regions) != len(TEMPORAL_REGIONS)
    ):
        raise SeedVR2Error(
            "quality-boundary temporal diagnostic evidence is incomplete"
        )

    region_scores: list[float] = []
    region_passes: list[bool] = []
    for region, (region_id, bounds) in zip(regions, TEMPORAL_REGIONS, strict=True):
        if not isinstance(region, dict):
            raise SeedVR2Error(
                "quality-boundary temporal diagnostic evidence is incomplete"
            )
        _require_exact_keys(
            region,
            {
                "acceleration_ratio",
                "augmented_mean_acceleration",
                "bounds",
                "noise_floor",
                "passed",
                "region_id",
                "residual_mean_acceleration",
                "score",
                "source_mean_acceleration",
            },
            label="temporal",
        )
        nonnegative_fields = (
            "source_mean_acceleration",
            "augmented_mean_acceleration",
            "residual_mean_acceleration",
            "acceleration_ratio",
        )
        if (
            region["region_id"] != region_id
            or not isinstance(region["bounds"], list)
            or len(region["bounds"]) != 4
            or any(type(value) is not float for value in region["bounds"])
            or any(not math.isfinite(value) for value in region["bounds"])
            or region["bounds"] != list(bounds)
            or type(region["noise_floor"]) is not float
            or region["noise_floor"] != TEMPORAL_NOISE_FLOOR
            or type(region["passed"]) is not bool
            or not _six_decimal_float(region["score"])
            or not 0.0 <= region["score"] <= 1.0
            or any(
                not _six_decimal_float(region[name]) or region[name] < 0.0
                for name in nonnegative_fields
            )
        ):
            raise SeedVR2Error(
                "quality-boundary temporal diagnostic evidence is incomplete"
            )

        source_low, source_high = _rounded_interval(region["source_mean_acceleration"])
        augmented_low, augmented_high = _rounded_interval(
            region["augmented_mean_acceleration"]
        )
        residual_low, residual_high = _rounded_interval(
            region["residual_mean_acceleration"]
        )
        ratio_low = augmented_low / max(source_high, TEMPORAL_NOISE_FLOOR)
        ratio_high = augmented_high / max(source_low, TEMPORAL_NOISE_FLOOR)
        score_low = min(
            1.0,
            TEMPORAL_NOISE_FLOOR / max(residual_high, TEMPORAL_NOISE_FLOOR),
        )
        score_high = min(
            1.0,
            TEMPORAL_NOISE_FLOOR / max(residual_low, TEMPORAL_NOISE_FLOOR),
        )
        minimum_residual = max(
            0.0,
            source_low - augmented_high,
            augmented_low - source_high,
        )
        maximum_residual = source_high + augmented_high
        verdict_possible = (
            score_high >= TEMPORAL_THRESHOLD
            if region["passed"]
            else score_low < TEMPORAL_THRESHOLD
        )
        if (
            not _rounded_value_overlaps(
                region["acceleration_ratio"], ratio_low, ratio_high
            )
            or not _rounded_value_overlaps(region["score"], score_low, score_high)
            or residual_high < minimum_residual
            or residual_low > maximum_residual
            or region["source_mean_acceleration"] > 510.0
            or region["augmented_mean_acceleration"] > 510.0
            or region["residual_mean_acceleration"] > 1020.0
            or not verdict_possible
        ):
            raise SeedVR2Error(
                "quality-boundary temporal diagnostic evidence is inconsistent"
            )
        region_scores.append(region["score"])
        region_passes.append(region["passed"])

    _validate_temporal_region_aggregation(
        regions, frame_width=frame_width, frame_height=frame_height
    )
    if payload["score"] != min(region_scores) or payload["passed"] is not all(
        region_passes
    ):
        raise SeedVR2Error(
            "quality-boundary temporal diagnostic evidence is inconsistent"
        )


def _validate_temporal_region_aggregation(
    regions: list[dict[str, Any]], *, frame_width: int, frame_height: int
) -> None:
    if frame_width % 2 != 0 or frame_height % 2 != 0:
        return
    tiles = regions[1:]
    areas: list[int] = []
    for _, (x0, y0, x1, y1) in TEMPORAL_REGIONS[1:]:
        left, top = int(x0 * frame_width), int(y0 * frame_height)
        right = int(math.ceil(x1 * frame_width))
        bottom = int(math.ceil(y1 * frame_height))
        areas.append((right - left) * (bottom - top))
    total_area = sum(areas)
    if total_area != frame_width * frame_height:
        raise SeedVR2Error(
            "quality-boundary temporal diagnostic evidence is inconsistent"
        )
    for field in (
        "source_mean_acceleration",
        "augmented_mean_acceleration",
        "residual_mean_acceleration",
    ):
        weighted_low = (
            sum(
                area * _rounded_interval(tile[field])[0]
                for tile, area in zip(tiles, areas, strict=True)
            )
            / total_area
        )
        weighted_high = (
            sum(
                area * _rounded_interval(tile[field])[1]
                for tile, area in zip(tiles, areas, strict=True)
            )
            / total_area
        )
        tolerance = _float32_reduction_tolerance(
            regions[0][field], weighted_low, weighted_high
        )
        if not _rounded_value_overlaps(
            regions[0][field],
            max(0.0, weighted_low - tolerance),
            weighted_high + tolerance,
        ):
            raise SeedVR2Error(
                "quality-boundary temporal diagnostic evidence is inconsistent"
            )


def _validate_motion_diagnostic(
    payload: dict[str, Any],
    *,
    run_id: str,
    frame_count: int,
    frame_width: int,
    frame_height: int,
) -> None:
    _require_exact_keys(
        payload,
        {
            "clip_id",
            "engine",
            "passed",
            "score",
            "threshold",
            "total_augmented_dynamic_pixels",
            "total_frames",
            "total_hallucinated_dynamic_pixels",
        },
        label="motion-hallucination",
    )
    hallucinated = payload["total_hallucinated_dynamic_pixels"]
    augmented = payload["total_augmented_dynamic_pixels"]
    maximum_dynamic_pixels = (frame_count - 1) * frame_width * frame_height
    if (
        payload["clip_id"] != run_id
        or type(payload["passed"]) is not bool
        or type(payload["threshold"]) is not float
        or payload["threshold"] != MOTION_THRESHOLD
        or type(payload["score"]) is not float
        or not math.isfinite(payload["score"])
        or not 0.0 <= payload["score"] <= 1.0
        or type(payload["total_frames"]) is not int
        or payload["total_frames"] != frame_count
        or type(hallucinated) is not int
        or type(augmented) is not int
        or not 0 <= hallucinated <= augmented
        or augmented > maximum_dynamic_pixels
        or payload["engine"] != MOTION_ENGINE
    ):
        raise SeedVR2Error(
            "quality-boundary motion-hallucination diagnostic evidence is incomplete"
        )
    expected_score = (
        1.0 if augmented == 0 else max(0.0, 1.0 - (hallucinated / augmented))
    )
    if not math.isclose(payload["score"], expected_score, rel_tol=0.0, abs_tol=1e-12):
        raise SeedVR2Error(
            "quality-boundary motion-hallucination diagnostic evidence is inconsistent"
        )
    if payload["passed"] is not (payload["score"] >= MOTION_THRESHOLD):
        raise SeedVR2Error(
            "quality-boundary motion-hallucination diagnostic evidence is inconsistent"
        )


def _diagnostic_payload(result: Any, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(_canonical_json(result.to_dict()))
    except (AttributeError, TypeError, ValueError) as exc:
        raise SeedVR2Error(
            f"quality-boundary {label} diagnostic evidence is incomplete"
        ) from exc
    if not isinstance(payload, dict):
        raise SeedVR2Error(
            f"quality-boundary {label} diagnostic evidence is incomplete"
        )
    return payload


def _validated_media(
    *,
    result: dict[str, Any],
    storage: Any,
    directory: Path,
    source_uri: str,
    candidate_uri: str,
) -> tuple[Path, Path, list[str], list[str]]:
    """Download, hash, fully decode, and timing-check the exact source/output pair."""

    source = directory / "source.mp4"
    candidate = directory / "restored.mp4"
    storage.download_file(source_uri, str(source))
    storage.download_file(candidate_uri, str(candidate))
    if _sha256(source) != result["input"].get("sha256"):
        raise SeedVR2Error("quality-boundary source hash disagrees with result.json")
    if _sha256(candidate) != result["output"].get("sha256"):
        raise SeedVR2Error("quality-boundary candidate hash disagrees with result.json")
    source_media = _probe_video(source)
    candidate_media = _probe_video(candidate)
    if source_media != result["input"].get("media"):
        raise SeedVR2Error("quality-boundary source media disagrees with result.json")
    if candidate_media != result["output"].get("media"):
        raise SeedVR2Error(
            "quality-boundary candidate media disagrees with result.json"
        )
    _validate_candidate_decode(result, candidate)
    source_frames = _decoded_frame_hashes(source)
    candidate_frames = _decoded_frame_hashes(candidate)
    if (
        len(source_frames) != len(candidate_frames)
        or len(source_frames) != source_media["frames"]
        or source_media["frames"] != candidate_media["frames"]
        or source_media["fps"] != candidate_media["fps"]
    ):
        raise SeedVR2Error(
            "quality-boundary source and candidate timing evidence is incomplete"
        )
    return source, candidate, source_frames, candidate_frames


def _document(
    *,
    request: VideoArtifactRequest,
    result: dict[str, Any],
    result_path: Path,
    source_uri: str,
    candidate_uri: str,
    source: Path,
    candidate: Path,
    frame_count: int,
    temporal: dict[str, Any],
    motion_hallucination: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": QUALITY_BOUNDARY_SCHEMA,
        "status": "review_only",
        "run_id": request.run_id,
        "created_at": _utc_now(),
        "quality_qualified": False,
        "sensor_truth_qualified": False,
        "downstream_use": "review_only",
        "result": {
            "uri": request.input_path,
            "sha256": _sha256(result_path),
        },
        "source": {
            "uri": source_uri,
            "sha256": _sha256(source),
            "media": result["input"]["media"],
        },
        "candidate": {
            "uri": candidate_uri,
            "sha256": _sha256(candidate),
            "media": result["output"]["media"],
            "derived_sensor_truth": False,
        },
        "timing_evidence": {
            "fully_decoded": True,
            "source_frames": frame_count,
            "candidate_frames": frame_count,
            "frame_counts_match": True,
            "frame_rates_match": True,
        },
        "diagnostics": {
            "temporal_consistency": {
                "enforcement": "advisory",
                "calibration": "not_calibrated_for_seedvr_acceptance",
                "scope": "source-resolution temporal residual; not output-resolution quality",
                "configuration": {
                    "threshold": TEMPORAL_THRESHOLD,
                    "noise_floor": TEMPORAL_NOISE_FLOOR,
                    "blur_ksize": TEMPORAL_BLUR_KSIZE,
                    "regions": _temporal_region_configuration(),
                },
                "result": temporal,
            },
            "motion_hallucination": {
                "enforcement": "advisory",
                "calibration": "not_calibrated_for_seedvr_acceptance",
                "scope": "source-resolution added motion; not static invented detail",
                "configuration": {
                    "threshold": MOTION_THRESHOLD,
                    "grad_thresh": MOTION_GRAD_THRESH,
                    "blur_ksize": MOTION_BLUR_KSIZE,
                    "morph_k": MOTION_MORPH_K,
                    "dist_tol_px": MOTION_DIST_TOL_PX,
                    "prefer_upstream": False,
                },
                "result": motion_hallucination,
            },
        },
        "not_qualified_reasons": [
            "objective_restoration_quality_not_accepted",
            "visual_acceptance_not_calibrated",
            "diagnostics_are_advisory",
            "generated_detail_is_not_observed_sensor_truth",
        ],
    }


@_storage_errors()
def assess(
    request: VideoArtifactRequest,
    *,
    storage_factory: Callable[[], Any] = StorageClient.from_environment,
) -> dict[str, Any]:
    """Publish diagnostics while refusing to promote output beyond review-only."""

    _validate_artifact_request(request, input_suffix=".json", output_kind="json")
    directory = _create_work_directory(request.run_id)
    storage = storage_factory()
    result_path = directory / "result.json"
    storage.download_file(request.input_path, str(result_path))
    result = _load_result(result_path)
    source_uri = _authorized_result_uri(
        result,
        "input",
        "uri",
        operation="assess SeedVR2 source video",
        suffix=".mp4",
    )
    candidate_uri = _authorized_result_uri(
        result,
        "artifacts",
        "restored_video",
        operation="assess SeedVR2 restored video",
        suffix=".mp4",
    )
    _validate_result_context(result, request, storage, directory)
    source, candidate, source_frames, _candidate_frames = _validated_media(
        result=result,
        storage=storage,
        directory=directory,
        source_uri=source_uri,
        candidate_uri=candidate_uri,
    )
    try:
        temporal_result = check_temporal_consistency(
            clip_id=request.run_id,
            original_video=source,
            augmented_video=candidate,
            threshold=TEMPORAL_THRESHOLD,
            regions=_temporal_region_configuration(),
            noise_floor=TEMPORAL_NOISE_FLOOR,
            blur_ksize=TEMPORAL_BLUR_KSIZE,
        )
        motion_result = check_hallucination(
            clip_id=request.run_id,
            original_video=source,
            augmented_video=candidate,
            grad_thresh=MOTION_GRAD_THRESH,
            blur_ksize=MOTION_BLUR_KSIZE,
            morph_k=MOTION_MORPH_K,
            dist_tol_px=MOTION_DIST_TOL_PX,
            threshold=MOTION_THRESHOLD,
            prefer_upstream=False,
        )
    except CosmosEvaluatorError as exc:
        raise SeedVR2Error(
            "quality-boundary diagnostics could not produce complete evidence"
        ) from exc
    temporal = _diagnostic_payload(temporal_result, label="temporal")
    motion_hallucination = _diagnostic_payload(
        motion_result, label="motion-hallucination"
    )
    _validate_temporal_diagnostic(
        temporal,
        run_id=request.run_id,
        frame_count=len(source_frames),
        frame_width=result["input"]["media"]["width"],
        frame_height=result["input"]["media"]["height"],
    )
    _validate_motion_diagnostic(
        motion_hallucination,
        run_id=request.run_id,
        frame_count=len(source_frames),
        frame_width=result["input"]["media"]["width"],
        frame_height=result["input"]["media"]["height"],
    )
    document = _document(
        request=request,
        result=result,
        result_path=result_path,
        source_uri=source_uri,
        candidate_uri=candidate_uri,
        source=source,
        candidate=candidate,
        frame_count=len(source_frames),
        temporal=temporal,
        motion_hallucination=motion_hallucination,
    )
    target = directory / "quality-boundary.json"
    target.write_bytes(_canonical_json(document))
    _ensure_artifacts_absent(storage, [request.output_path])
    _publish_verified(storage, target, request.output_path, directory / "readback")
    shutil.rmtree(directory)
    return document


__all__ = ["assess"]
