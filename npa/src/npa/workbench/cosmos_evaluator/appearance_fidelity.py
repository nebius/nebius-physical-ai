"""Source-relative appearance fidelity for input-conditioned video variants.

This NPA companion check measures photometric drift between aligned source and
augmented frames.  It is intentionally scene-agnostic: normalized regions may
be supplied by a deployment, while the default full-frame plus 2x2 grid catches
both scene-wide colour casts and localized material recolouring.

The check works in CIELAB after Gaussian pre-filtering.  It reports luminance,
chroma, region-relative chroma, and frame-to-frame chroma-shift drift separately
so intended bounded exposure or white-balance changes remain distinguishable
from unintended object/material changes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Sequence

import numpy as np

from npa.workbench.cosmos_evaluator.hallucination import _gaussian_blur, _probe_size
from npa.workbench.cosmos_evaluator.upstream import CosmosEvaluatorError

DEFAULT_THRESHOLD = 0.8
DEFAULT_LUMINANCE_TOLERANCE = 18.0
DEFAULT_GLOBAL_CHROMA_TOLERANCE = 8.0
DEFAULT_LOCAL_CHROMA_TOLERANCE = 6.0
DEFAULT_CHROMA_INSTABILITY_TOLERANCE = 4.0
DEFAULT_BLUR_KSIZE = 7
DEFAULT_MAX_DIMENSION = 256
ENGINE = "npa-source-relative-appearance-fidelity-v1"

DEFAULT_REGIONS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("full-frame", (0.0, 0.0, 1.0, 1.0)),
    ("tile-0", (0.0, 0.0, 0.5, 0.5)),
    ("tile-1", (0.5, 0.0, 1.0, 0.5)),
    ("tile-2", (0.0, 0.5, 0.5, 1.0)),
    ("tile-3", (0.5, 0.5, 1.0, 1.0)),
)


@dataclass(frozen=True)
class AppearanceRegionResult:
    region_id: str
    bounds: tuple[float, float, float, float]
    luminance_delta_p95: float
    chroma_delta_p95: float
    local_chroma_residual_p95: float
    chroma_instability_p95: float
    score: float
    passed: bool


@dataclass(frozen=True)
class AppearanceFidelityResult:
    clip_id: str
    passed: bool
    threshold: float
    score: float
    total_frames: int
    frame_counts_match: bool
    luminance_tolerance: float = DEFAULT_LUMINANCE_TOLERANCE
    global_chroma_tolerance: float = DEFAULT_GLOBAL_CHROMA_TOLERANCE
    local_chroma_tolerance: float = DEFAULT_LOCAL_CHROMA_TOLERANCE
    chroma_instability_tolerance: float = DEFAULT_CHROMA_INSTABILITY_TOLERANCE
    blur_ksize: int = DEFAULT_BLUR_KSIZE
    max_dimension: int = DEFAULT_MAX_DIMENSION
    engine: str = ENGINE
    aggregation: str = "minimum-region-score"
    regions: list[AppearanceRegionResult] = field(default_factory=list)

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
                "--appearance-regions-json must be valid JSON"
            ) from exc
    if not isinstance(raw, list) or not raw:
        raise CosmosEvaluatorError(
            "--appearance-regions-json must be a non-empty JSON list"
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
                f"appearance region {index} must have four normalized bounds"
            )
        try:
            x0, y0, x1, y1 = (float(part) for part in bounds)
        except (TypeError, ValueError) as exc:
            raise CosmosEvaluatorError(
                f"appearance region {index} has non-numeric bounds"
            ) from exc
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise CosmosEvaluatorError(
                f"appearance region {index} bounds must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1"
            )
        parsed.append((region_id, (x0, y0, x1, y1)))
    return parsed


def check_appearance_fidelity(
    *,
    clip_id: str,
    original_video: str | Path,
    augmented_video: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    regions: str | Sequence[Any] | None = None,
    luminance_tolerance: float = DEFAULT_LUMINANCE_TOLERANCE,
    global_chroma_tolerance: float = DEFAULT_GLOBAL_CHROMA_TOLERANCE,
    local_chroma_tolerance: float = DEFAULT_LOCAL_CHROMA_TOLERANCE,
    chroma_instability_tolerance: float = DEFAULT_CHROMA_INSTABILITY_TOLERANCE,
    blur_ksize: int = DEFAULT_BLUR_KSIZE,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> AppearanceFidelityResult:
    """Measure bounded photometric drift over generic normalized regions."""

    original = Path(original_video)
    augmented = Path(augmented_video)
    for label, path in (("original", original), ("augmented", augmented)):
        if not path.is_file():
            raise CosmosEvaluatorError(f"{label} video not found: {path}")
    if not 0.0 < threshold <= 1.0:
        raise CosmosEvaluatorError(
            "appearance threshold must be greater than 0.0 and at most 1.0"
        )
    tolerances = {
        "luminance tolerance": luminance_tolerance,
        "global chroma tolerance": global_chroma_tolerance,
        "local chroma tolerance": local_chroma_tolerance,
        "chroma instability tolerance": chroma_instability_tolerance,
    }
    for label, value in tolerances.items():
        if value <= 0.0:
            raise CosmosEvaluatorError(f"appearance {label} must be greater than 0.0")
    if blur_ksize < 1 or blur_ksize % 2 == 0:
        raise CosmosEvaluatorError(
            "appearance blur kernel must be a positive odd integer"
        )
    if max_dimension < 16:
        raise CosmosEvaluatorError("appearance max dimension must be at least 16")

    normalized_regions = parse_regions(regions)
    source_height, source_width = _probe_size(original)
    height, width = _scaled_geometry(
        height=source_height, width=source_width, max_dimension=max_dimension
    )
    source_frames = _iter_rgb_frames(original, height, width)
    augmented_frames = _iter_rgb_frames(augmented, height, width)
    luminance_values: list[list[float]] = [[] for _ in normalized_regions]
    chroma_values: list[list[float]] = [[] for _ in normalized_regions]
    local_values: list[list[float]] = [[] for _ in normalized_regions]
    instability_values: list[list[float]] = [[] for _ in normalized_regions]
    prior_region_chroma: list[np.ndarray | None] = [None] * len(normalized_regions)
    total_frames = 0
    counts_match = True

    try:
        while True:
            source_frame = next(source_frames, None)
            augmented_frame = next(augmented_frames, None)
            if source_frame is None or augmented_frame is None:
                counts_match = source_frame is None and augmented_frame is None
                break
            source_lab = _rgb_to_lab(_prefilter(source_frame, blur_ksize))
            augmented_lab = _rgb_to_lab(_prefilter(augmented_frame, blur_ksize))
            delta = augmented_lab - source_lab
            full_shift = np.median(delta.reshape(-1, 3), axis=0)
            for index, (_, bounds) in enumerate(normalized_regions):
                y_slice, x_slice = _region_slices(
                    bounds, height=height, width=width
                )
                region_shift = np.median(
                    delta[y_slice, x_slice].reshape(-1, 3), axis=0
                )
                region_chroma = region_shift[1:3]
                luminance_values[index].append(float(abs(region_shift[0])))
                chroma_values[index].append(float(np.linalg.norm(region_chroma)))
                local_values[index].append(
                    float(np.linalg.norm(region_chroma - full_shift[1:3]))
                )
                prior = prior_region_chroma[index]
                if prior is not None:
                    instability_values[index].append(
                        float(np.linalg.norm(region_chroma - prior))
                    )
                prior_region_chroma[index] = region_chroma
            total_frames += 1
    finally:
        source_frames.close()
        augmented_frames.close()

    if total_frames == 0:
        raise CosmosEvaluatorError(
            "appearance fidelity needs at least one decodable frame per clip"
        )

    results: list[AppearanceRegionResult] = []
    for index, (region_id, bounds) in enumerate(normalized_regions):
        luminance = _percentile(luminance_values[index], 95.0)
        chroma = _percentile(chroma_values[index], 95.0)
        local = _percentile(local_values[index], 95.0)
        instability = _percentile(instability_values[index], 95.0)
        chroma_tolerance = (
            global_chroma_tolerance
            if region_id == "full-frame"
            else global_chroma_tolerance + local_chroma_tolerance
        )
        scores = (
            _bounded_score(luminance, luminance_tolerance),
            _bounded_score(chroma, chroma_tolerance),
            _bounded_score(local, local_chroma_tolerance),
            _bounded_score(instability, chroma_instability_tolerance),
        )
        score = min(scores)
        results.append(
            AppearanceRegionResult(
                region_id=region_id,
                bounds=bounds,
                luminance_delta_p95=round(luminance, 6),
                chroma_delta_p95=round(chroma, 6),
                local_chroma_residual_p95=round(local, 6),
                chroma_instability_p95=round(instability, 6),
                score=round(score, 6),
                passed=score >= threshold,
            )
        )

    score = min(region.score for region in results)
    return AppearanceFidelityResult(
        clip_id=clip_id,
        passed=counts_match and all(region.passed for region in results),
        threshold=threshold,
        score=score,
        total_frames=total_frames,
        frame_counts_match=counts_match,
        luminance_tolerance=luminance_tolerance,
        global_chroma_tolerance=global_chroma_tolerance,
        local_chroma_tolerance=local_chroma_tolerance,
        chroma_instability_tolerance=chroma_instability_tolerance,
        blur_ksize=blur_ksize,
        max_dimension=max_dimension,
        regions=results,
    )


def _iter_rgb_frames(
    video: Path, height: int, width: int
) -> Generator[np.ndarray, None, None]:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise CosmosEvaluatorError("appearance fidelity requires ffmpeg on PATH")
    cmd = [
        exe,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        f"scale={width}:{height}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    frame_bytes = height * width * 3
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    try:
        assert proc.stdout is not None
        partial = False
        while True:
            buffer = proc.stdout.read(frame_bytes)
            if not buffer:
                break
            if len(buffer) < frame_bytes:
                partial = True
                break
            yield np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
        returncode = proc.wait(timeout=30)
        if returncode != 0 or partial:
            detail = ""
            if proc.stderr is not None:
                detail = (proc.stderr.read() or b"").decode(
                    "utf-8", "replace"
                ).strip()
            raise CosmosEvaluatorError(
                f"ffmpeg failed to decode {video} (exit {returncode}): "
                f"{detail or 'stream ended mid-frame'}"[:300]
            )
    finally:
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                proc.kill()


def _scaled_geometry(*, height: int, width: int, max_dimension: int) -> tuple[int, int]:
    scale = min(1.0, max_dimension / max(height, width))
    scaled_width = max(2, int(round(width * scale)))
    scaled_height = max(2, int(round(height * scale)))
    # libswscale accepts arbitrary sizes, but even dimensions avoid codec-specific
    # chroma edge cases and make local comparisons reproducible.
    scaled_width += scaled_width % 2
    scaled_height += scaled_height % 2
    return scaled_height, scaled_width


def _prefilter(frame: np.ndarray, blur_ksize: int) -> np.ndarray:
    if blur_ksize < 3:
        return frame
    channels = [_gaussian_blur(frame[:, :, index], blur_ksize) for index in range(3)]
    return np.stack(channels, axis=2)


def _rgb_to_lab(frame: np.ndarray) -> np.ndarray:
    """Convert uint8 sRGB to float CIELAB (D65) without an OpenCV dependency."""

    rgb = frame.astype(np.float32) / 255.0
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    xyz = linear @ np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        (
            116.0 * transformed[:, :, 1] - 16.0,
            500.0 * (transformed[:, :, 0] - transformed[:, :, 1]),
            200.0 * (transformed[:, :, 1] - transformed[:, :, 2]),
        ),
        axis=2,
    )


def _region_slices(
    bounds: tuple[float, float, float, float], *, height: int, width: int
) -> tuple[slice, slice]:
    x0, y0, x1, y1 = bounds
    left = min(width - 1, int(x0 * width))
    top = min(height - 1, int(y0 * height))
    right = max(left + 1, min(width, int(np.ceil(x1 * width))))
    bottom = max(top + 1, min(height, int(np.ceil(y1 * height))))
    return slice(top, bottom), slice(left, right)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _bounded_score(observed: float, tolerance: float) -> float:
    return min(1.0, tolerance / max(observed, tolerance))
