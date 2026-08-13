"""Source-relative temporal consistency for input-conditioned video variants.

This is an NPA companion diagnostic, not an upstream NVIDIA Cosmos Evaluator
check. It compares the *signed* temporal acceleration of aligned source and
augmented frames after Gaussian pre-filtering. The residual is two-sided: added
frame-to-frame instability and collapsed source motion both increase it. Scoring
against a fixed codec-noise floor avoids making the same artifact easier to pass
merely because the source contains more motion.

The default regions are the full frame and a 2x2 grid. Taking the lowest region
score prevents a localized artifact from being hidden by a clean frame average.
Callers may instead provide normalized rectangular regions as JSON. The metric
is advisory in the PAIDF reference workflow until a deployment explicitly
calibrates its noise floor and opts into hard enforcement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from npa.workbench.cosmos_evaluator.hallucination import (
    _gaussian_blur,
    _iter_gray_frames,
    _probe_size,
)
from npa.workbench.cosmos_evaluator.upstream import CosmosEvaluatorError

DEFAULT_THRESHOLD = 0.8
DEFAULT_NOISE_FLOOR = 0.25
DEFAULT_BLUR_KSIZE = 7
ENGINE = "npa-source-relative-temporal-residual-v2"

DEFAULT_REGIONS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("full-frame", (0.0, 0.0, 1.0, 1.0)),
    ("tile-0", (0.0, 0.0, 0.5, 0.5)),
    ("tile-1", (0.5, 0.0, 1.0, 0.5)),
    ("tile-2", (0.0, 0.5, 0.5, 1.0)),
    ("tile-3", (0.5, 0.5, 1.0, 1.0)),
)


@dataclass(frozen=True)
class TemporalRegionResult:
    region_id: str
    bounds: tuple[float, float, float, float]
    source_mean_acceleration: float
    augmented_mean_acceleration: float
    residual_mean_acceleration: float
    noise_floor: float
    acceleration_ratio: float
    score: float
    passed: bool


@dataclass(frozen=True)
class TemporalConsistencyResult:
    clip_id: str
    passed: bool
    threshold: float
    score: float
    total_frames: int
    frame_counts_match: bool
    noise_floor: float = DEFAULT_NOISE_FLOOR
    blur_ksize: int = DEFAULT_BLUR_KSIZE
    engine: str = ENGINE
    aggregation: str = "minimum-region-score"
    regions: list[TemporalRegionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_regions(
    value: str | Sequence[Any] | None,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Parse normalized rectangles, or return the generic tiled default."""

    if value is None or value == "" or (isinstance(value, (list, tuple)) and not value):
        return list(DEFAULT_REGIONS)
    raw: Any = value
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CosmosEvaluatorError(
                "--temporal-regions-json must be valid JSON"
            ) from exc
    if not isinstance(raw, list) or not raw:
        raise CosmosEvaluatorError(
            "--temporal-regions-json must be a non-empty JSON list"
        )

    parsed: list[tuple[str, tuple[float, float, float, float]]] = []
    for index, item in enumerate(raw):
        region_id = f"region-{index}"
        bounds: Any = item
        if isinstance(item, dict):
            region_id = str(item.get("id") or region_id)
            bounds = item.get("bounds")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            raise CosmosEvaluatorError(
                f"temporal region {index} must have four normalized bounds"
            )
        try:
            x0, y0, x1, y1 = (float(part) for part in bounds)
        except (TypeError, ValueError) as exc:
            raise CosmosEvaluatorError(
                f"temporal region {index} has non-numeric bounds"
            ) from exc
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise CosmosEvaluatorError(
                f"temporal region {index} bounds must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1"
            )
        parsed.append((region_id, (x0, y0, x1, y1)))
    return parsed


def check_temporal_consistency(
    *,
    clip_id: str,
    original_video: str | Path,
    augmented_video: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    regions: str | Sequence[Any] | None = None,
    noise_floor: float = DEFAULT_NOISE_FLOOR,
    blur_ksize: int = DEFAULT_BLUR_KSIZE,
) -> TemporalConsistencyResult:
    """Compare filtered, signed temporal acceleration over normalized regions."""

    original = Path(original_video)
    augmented = Path(augmented_video)
    for label, path in (("original", original), ("augmented", augmented)):
        if not path.is_file():
            raise CosmosEvaluatorError(f"{label} video not found: {path}")
    if not 0.0 < threshold <= 1.0:
        raise CosmosEvaluatorError(
            "temporal threshold must be greater than 0.0 and at most 1.0"
        )
    if noise_floor <= 0.0:
        raise CosmosEvaluatorError("temporal noise floor must be greater than 0.0")
    if blur_ksize < 1 or blur_ksize % 2 == 0:
        raise CosmosEvaluatorError(
            "temporal blur kernel must be a positive odd integer"
        )

    normalized_regions = parse_regions(regions)
    height, width = _probe_size(original)
    source_frames = _iter_gray_frames(original, height, width)
    augmented_frames = _iter_gray_frames(augmented, height, width)
    source_sums = np.zeros(len(normalized_regions), dtype=np.float64)
    augmented_sums = np.zeros(len(normalized_regions), dtype=np.float64)
    residual_sums = np.zeros(len(normalized_regions), dtype=np.float64)
    acceleration_frames = 0
    total_frames = 0
    counts_match = True

    try:
        source_window = [
            _prefilter(frame, blur_ksize)
            for frame in (next(source_frames, None), next(source_frames, None))
            if frame is not None
        ]
        augmented_window = [
            _prefilter(frame, blur_ksize)
            for frame in (next(augmented_frames, None), next(augmented_frames, None))
            if frame is not None
        ]
        if len(source_window) < 2 or len(augmented_window) < 2:
            raise CosmosEvaluatorError(
                "temporal consistency needs at least three decodable frames per clip"
            )
        total_frames = 2
        while True:
            source_current = next(source_frames, None)
            augmented_current = next(augmented_frames, None)
            if source_current is None or augmented_current is None:
                counts_match = source_current is None and augmented_current is None
                break
            filtered_source = _prefilter(source_current, blur_ksize)
            filtered_augmented = _prefilter(augmented_current, blur_ksize)
            source_acceleration = (
                filtered_source.astype(np.float32)
                - 2.0 * source_window[1].astype(np.float32)
                + source_window[0].astype(np.float32)
            )
            augmented_acceleration = (
                filtered_augmented.astype(np.float32)
                - 2.0 * augmented_window[1].astype(np.float32)
                + augmented_window[0].astype(np.float32)
            )
            residual_acceleration = np.abs(augmented_acceleration - source_acceleration)
            for index, (_, bounds) in enumerate(normalized_regions):
                y_slice, x_slice = _region_slices(bounds, height=height, width=width)
                source_sums[index] += float(
                    np.abs(source_acceleration[y_slice, x_slice]).mean()
                )
                augmented_sums[index] += float(
                    np.abs(augmented_acceleration[y_slice, x_slice]).mean()
                )
                residual_sums[index] += float(
                    residual_acceleration[y_slice, x_slice].mean()
                )
            source_window = [source_window[1], filtered_source]
            augmented_window = [augmented_window[1], filtered_augmented]
            acceleration_frames += 1
            total_frames += 1
    finally:
        source_frames.close()
        augmented_frames.close()

    if acceleration_frames == 0:
        raise CosmosEvaluatorError(
            "temporal consistency needs at least three decodable frames per clip"
        )

    results: list[TemporalRegionResult] = []
    for index, (region_id, bounds) in enumerate(normalized_regions):
        source_mean = float(source_sums[index] / acceleration_frames)
        augmented_mean = float(augmented_sums[index] / acceleration_frames)
        residual_mean = float(residual_sums[index] / acceleration_frames)
        ratio = augmented_mean / max(source_mean, noise_floor)
        score = min(1.0, noise_floor / max(residual_mean, noise_floor))
        results.append(
            TemporalRegionResult(
                region_id=region_id,
                bounds=bounds,
                source_mean_acceleration=round(source_mean, 6),
                augmented_mean_acceleration=round(augmented_mean, 6),
                residual_mean_acceleration=round(residual_mean, 6),
                noise_floor=round(noise_floor, 6),
                acceleration_ratio=round(ratio, 6),
                score=round(score, 6),
                passed=score >= threshold,
            )
        )

    score = min(region.score for region in results)
    passed = (
        counts_match and total_frames >= 3 and all(region.passed for region in results)
    )
    return TemporalConsistencyResult(
        clip_id=clip_id,
        passed=passed,
        threshold=threshold,
        score=score,
        total_frames=total_frames,
        frame_counts_match=counts_match,
        noise_floor=noise_floor,
        blur_ksize=blur_ksize,
        regions=results,
    )


def _prefilter(frame: np.ndarray, blur_ksize: int) -> np.ndarray:
    if blur_ksize < 3:
        return frame
    return _gaussian_blur(frame, blur_ksize)


def _region_slices(
    bounds: tuple[float, float, float, float], *, height: int, width: int
) -> tuple[slice, slice]:
    x0, y0, x1, y1 = bounds
    left = min(width - 1, int(x0 * width))
    top = min(height - 1, int(y0 * height))
    right = max(left + 1, min(width, int(np.ceil(x1 * width))))
    bottom = max(top + 1, min(height, int(np.ceil(y1 * height))))
    return slice(top, bottom), slice(left, right)
