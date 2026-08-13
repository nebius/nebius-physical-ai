"""Cosmos Evaluator hallucination check over an original / augmented clip pair.

Upstream (``checks/hallucination`` in
https://github.com/nvidia-cosmos/cosmos-evaluator, Apache-2.0, Copyright (c)
2026 NVIDIA CORPORATION & AFFILIATES) scores hallucinated motion like this:

1. decode both clips in lockstep, converting each frame to grayscale;
2. per frame pair, build a *dynamic mask* — absolute difference against the
   previous frame, Gaussian-blurred, thresholded at ``grad_thresh``, then
   morphologically opened with a ``morph_k`` ellipse;
3. distance-transform the original clip's dynamic mask, and count augmented
   dynamic pixels farther than ``dist_tol_px`` from any original dynamic pixel —
   those are hallucinated; and
4. ``score = max(0, 1 - hallucinated / augmented_dynamic)``, passing at
   ``threshold`` (upstream default 0.682).

Two engines produce that score, and every result records which one ran:

- ``cosmos-evaluator-upstream`` — upstream's own ``HallucinationProcessor``,
  used whenever an upstream checkout is importable (see :mod:`.upstream`).
- ``cosmos-evaluator-npa-port`` — the in-repo implementation of the algorithm
  above, so the check still runs real pixel work in images that do not carry an
  upstream checkout. It uses OpenCV when available (bit-for-bit the same
  primitives upstream calls) and otherwise equivalent NumPy kernels.

Frame decoding always shells out to ``ffmpeg`` as a raw grayscale stream, which
is the same decoding contract upstream uses and needs no OpenCV.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generator

import numpy as np

from npa.workbench.cosmos_evaluator.upstream import (
    CosmosEvaluatorError,
    ensure_upstream_importable,
)

_log = logging.getLogger(__name__)

# Upstream defaults, from checks/cosmos_evaluator.yaml (metropolis.hallucination).
DEFAULT_GRAD_THRESH = 10.0
DEFAULT_BLUR_KSIZE = 7
DEFAULT_MORPH_K = 3
DEFAULT_DIST_TOL_PX = 7.0
DEFAULT_THRESHOLD = 0.682

ENGINE_UPSTREAM = "cosmos-evaluator-upstream"
ENGINE_PORT = "cosmos-evaluator-npa-port"


@dataclass(frozen=True)
class HallucinationResult:
    """Upstream's ``HallucinationResult`` fields plus the engine that ran."""

    clip_id: str
    passed: bool
    threshold: float
    score: float
    total_frames: int
    total_hallucinated_dynamic_pixels: int
    total_augmented_dynamic_pixels: int
    engine: str = ENGINE_PORT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_hallucination(
    *,
    clip_id: str,
    original_video: str | Path,
    augmented_video: str | Path,
    grad_thresh: float = DEFAULT_GRAD_THRESH,
    blur_ksize: int = DEFAULT_BLUR_KSIZE,
    morph_k: int = DEFAULT_MORPH_K,
    dist_tol_px: float = DEFAULT_DIST_TOL_PX,
    threshold: float = DEFAULT_THRESHOLD,
    max_frames: int | None = None,
    prefer_upstream: bool = True,
) -> HallucinationResult:
    """Score hallucinated motion in ``augmented_video`` against ``original_video``."""

    original = Path(original_video)
    augmented = Path(augmented_video)
    for label, path in (("original", original), ("augmented", augmented)):
        if not path.is_file():
            raise CosmosEvaluatorError(f"{label} video not found: {path}")
    if not 0.0 <= threshold <= 1.0:
        raise CosmosEvaluatorError("threshold must be between 0.0 and 1.0")

    if prefer_upstream:
        upstream = _run_upstream(
            clip_id=clip_id,
            original=original,
            augmented=augmented,
            params={
                "grad_thresh": grad_thresh,
                "blur_ksize": blur_ksize,
                "morph_k": morph_k,
                "dist_tol_px": dist_tol_px,
                "threshold": threshold,
                "max_frames": max_frames,
            },
        )
        if upstream is not None:
            return upstream

    return _run_port(
        clip_id=clip_id,
        original=original,
        augmented=augmented,
        grad_thresh=grad_thresh,
        blur_ksize=blur_ksize,
        morph_k=morph_k,
        dist_tol_px=dist_tol_px,
        threshold=threshold,
        max_frames=max_frames,
    )


def _run_upstream(
    *,
    clip_id: str,
    original: Path,
    augmented: Path,
    params: dict[str, Any],
) -> HallucinationResult | None:
    """Run upstream's processor, or return ``None`` when it is unavailable."""

    root = ensure_upstream_importable()
    if root is None:
        return None
    try:
        from checks.hallucination.processor import HallucinationProcessor  # type: ignore
    except Exception as exc:  # noqa: BLE001 - any import failure falls back to the port
        _log.info("cosmos-evaluator upstream checkout at %s is not importable: %s", root, exc)
        return None

    config_dir = str(root / "checks")
    try:
        processor = HallucinationProcessor(params=params, config_dir=config_dir, verbose="WARNING")
        result = processor.process(clip_id, str(original), str(augmented))
    except Exception as exc:  # noqa: BLE001 - upstream failure falls back to the port
        _log.warning("upstream hallucination check failed (%s); using the in-repo port", exc)
        return None

    payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    return HallucinationResult(
        clip_id=str(payload.get("clip_id") or clip_id),
        passed=bool(payload.get("passed")),
        threshold=float(payload.get("threshold", params["threshold"])),
        score=float(payload.get("score", 0.0)),
        total_frames=int(payload.get("total_frames", 0)),
        total_hallucinated_dynamic_pixels=int(payload.get("total_hallucinated_dynamic_pixels", 0)),
        total_augmented_dynamic_pixels=int(payload.get("total_augmented_dynamic_pixels", 0)),
        engine=ENGINE_UPSTREAM,
    )


def _run_port(
    *,
    clip_id: str,
    original: Path,
    augmented: Path,
    grad_thresh: float,
    blur_ksize: int,
    morph_k: int,
    dist_tol_px: float,
    threshold: float,
    max_frames: int | None,
) -> HallucinationResult:
    height, width = _probe_size(original)
    orig_frames = _iter_gray_frames(original, height, width)
    aug_frames = _iter_gray_frames(augmented, height, width)

    total_hallucinated = 0
    total_augmented_dynamic = 0
    frame_count = 0
    try:
        prev_o = next(orig_frames, None)
        prev_a = next(aug_frames, None)
        if prev_o is None or prev_a is None:
            raise CosmosEvaluatorError(
                f"could not decode a first frame from {original.name} / {augmented.name}"
            )
        frame_count = 1
        while True:
            curr_o = next(orig_frames, None)
            curr_a = next(aug_frames, None)
            if curr_o is None or curr_a is None:
                break
            mask_o = _dynamic_mask(prev_o, curr_o, grad_thresh, blur_ksize, morph_k)
            mask_a = _dynamic_mask(prev_a, curr_a, grad_thresh, blur_ksize, morph_k)
            hallucinated, augmented_dynamic = _hallucination_counts(mask_o, mask_a, dist_tol_px)
            total_hallucinated += hallucinated
            total_augmented_dynamic += augmented_dynamic
            prev_o, prev_a = curr_o, curr_a
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        orig_frames.close()
        aug_frames.close()

    if total_augmented_dynamic == 0:
        score = 1.0
    else:
        score = max(0.0, 1.0 - (float(total_hallucinated) / float(total_augmented_dynamic)))

    return HallucinationResult(
        clip_id=clip_id,
        passed=score >= threshold,
        threshold=threshold,
        score=float(score),
        total_frames=frame_count,
        total_hallucinated_dynamic_pixels=int(total_hallucinated),
        total_augmented_dynamic_pixels=int(total_augmented_dynamic),
        engine=ENGINE_PORT,
    )


# ---------------------------------------------------------------------------
# Frame decoding (ffmpeg raw grayscale stream)
# ---------------------------------------------------------------------------


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise CosmosEvaluatorError("the hallucination check requires ffmpeg on PATH")
    return exe


def _probe_size(video: Path) -> tuple[int, int]:
    """Return ``(height, width)`` of ``video`` via ffprobe."""

    exe = shutil.which("ffprobe")
    if not exe:
        raise CosmosEvaluatorError("the hallucination check requires ffprobe on PATH")
    proc = subprocess.run(
        [
            exe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    raw = (proc.stdout or "").strip().splitlines()
    if proc.returncode != 0 or not raw:
        raise CosmosEvaluatorError(f"ffprobe could not read {video.name}: {(proc.stderr or '').strip()[:200]}")
    try:
        width, height = (int(part) for part in raw[0].split("x")[:2])
    except ValueError as exc:
        raise CosmosEvaluatorError(f"unexpected ffprobe geometry for {video.name}: {raw[0]!r}") from exc
    if width <= 0 or height <= 0:
        raise CosmosEvaluatorError(f"{video.name} has a degenerate frame size {width}x{height}")
    return height, width


def _iter_gray_frames(video: Path, height: int, width: int) -> Generator[np.ndarray, None, None]:
    """Yield ``height x width`` uint8 grayscale frames, scaling to match.

    Scaling in ffmpeg mirrors upstream's ``ensure_same_size``: the augmented clip
    is resampled to the original's geometry before masks are compared.
    """

    cmd = [
        _ffmpeg(),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        f"scale={width}:{height}",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]
    frame_bytes = height * width
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    try:
        assert proc.stdout is not None
        partial = False
        while True:
            buffer = proc.stdout.read(frame_bytes)
            if not buffer:
                break
            if len(buffer) < frame_bytes:
                # Every raw gray frame is exactly height*width bytes, so a short read
                # is a cut-off stream rather than the end of one.
                partial = True
                break
            yield np.frombuffer(buffer, dtype=np.uint8).reshape(height, width)
        # Only reached when the caller consumed the whole stream, so a non-zero exit
        # is a real decode failure and not this generator being abandoned early.
        # Silently truncating here would drop frames from one side of the comparison
        # and quietly bias the score.
        returncode = proc.wait(timeout=30)
        if returncode != 0 or partial:
            detail = ""
            if proc.stderr is not None:
                detail = (proc.stderr.read() or b"").decode("utf-8", "replace").strip()
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


# ---------------------------------------------------------------------------
# Mask math: OpenCV when present (upstream's primitives), NumPy otherwise
# ---------------------------------------------------------------------------


def _cv2() -> Any | None:
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:  # noqa: BLE001 - OpenCV is optional
        return None


def _dynamic_mask(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    grad_thresh: float,
    blur_ksize: int,
    morph_k: int,
) -> np.ndarray:
    """Binary mask of pixels that moved between two grayscale frames."""

    diff = np.abs(curr_gray.astype(np.int16) - prev_gray.astype(np.int16)).astype(np.uint8)
    kernel_size = max(1, int(blur_ksize) | 1)
    morph_size = max(1, int(morph_k) | 1)
    cv2 = _cv2()
    if cv2 is not None:
        if kernel_size >= 3:
            diff = cv2.GaussianBlur(diff, (kernel_size, kernel_size), 0)
        _, mask = cv2.threshold(diff, grad_thresh, 255, cv2.THRESH_BINARY)
        if morph_size >= 3:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
        return mask
    if kernel_size >= 3:
        diff = _gaussian_blur(diff, kernel_size)
    mask = np.where(diff > grad_thresh, np.uint8(255), np.uint8(0))
    if morph_size >= 3:
        mask = _binary_open(mask, morph_size)
    return mask


def _gaussian_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    """Separable Gaussian blur with OpenCV's default sigma for ``ksize``."""

    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    radius = ksize // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    height, width = image.shape
    padded = np.pad(image.astype(np.float64), radius, mode="reflect")
    rows = np.zeros((height, padded.shape[1]), dtype=np.float64)
    for index, weight in enumerate(kernel):
        rows += weight * padded[index : index + height, :]
    cols = np.zeros((height, width), dtype=np.float64)
    for index, weight in enumerate(kernel):
        cols += weight * rows[:, index : index + width]
    return np.clip(np.rint(cols), 0, 255).astype(np.uint8)


def _ellipse_element(size: int) -> np.ndarray:
    """OpenCV's ``MORPH_ELLIPSE`` structuring element as a boolean array."""

    radius = size // 2
    coords = np.arange(size) - radius
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    scale = max(radius, 1)
    return (yy**2 + xx**2) <= scale * scale


def _binary_open(mask: np.ndarray, size: int) -> np.ndarray:
    """Erode then dilate ``mask`` with an elliptical element."""

    element = _ellipse_element(size)
    eroded = _rank_filter(mask > 0, element, erode=True)
    opened = _rank_filter(eroded, element, erode=False)
    return np.where(opened, np.uint8(255), np.uint8(0))


def _rank_filter(binary: np.ndarray, element: np.ndarray, *, erode: bool) -> np.ndarray:
    radius = element.shape[0] // 2
    padded = np.pad(binary, radius, mode="constant", constant_values=erode)
    out = np.full(binary.shape, erode, dtype=bool)
    height, width = binary.shape
    for dy in range(element.shape[0]):
        for dx in range(element.shape[1]):
            if not element[dy, dx]:
                continue
            window = padded[dy : dy + height, dx : dx + width]
            out = np.logical_and(out, window) if erode else np.logical_or(out, window)
    return out


def _hallucination_counts(
    orig_mask: np.ndarray,
    aug_mask: np.ndarray,
    dist_tol_px: float,
) -> tuple[int, int]:
    """Count augmented dynamic pixels far from any original dynamic pixel."""

    orig_dyn = orig_mask > 0
    aug_dyn = aug_mask > 0
    num_aug = int(np.count_nonzero(aug_dyn))
    if num_aug == 0:
        return 0, 0
    distance = _distance_transform(orig_dyn)
    hallucinated = aug_dyn & (distance > float(dist_tol_px))
    return int(np.count_nonzero(hallucinated)), num_aug


def _distance_transform(dynamic: np.ndarray) -> np.ndarray:
    """Distance from every pixel to the nearest ``True`` pixel in ``dynamic``."""

    cv2 = _cv2()
    if cv2 is not None:
        source = np.where(dynamic, 0, 255).astype(np.uint8)
        return cv2.distanceTransform(source, cv2.DIST_L2, 3)
    try:
        from scipy import ndimage  # type: ignore

        return np.asarray(ndimage.distance_transform_edt(~dynamic), dtype=np.float64)
    except Exception:  # noqa: BLE001 - SciPy is optional too
        return _squared_edt(dynamic) ** 0.5


def _squared_edt(dynamic: np.ndarray) -> np.ndarray:
    """Exact squared Euclidean distance transform, one axis at a time.

    Squared Euclidean distance is separable, so a 1-D transform applied down the
    columns and then across the rows yields the exact 2-D result.
    """

    if not dynamic.any():
        return np.full(dynamic.shape, np.inf, dtype=np.float64)
    # A finite stand-in for "unreachable" keeps the envelope arithmetic free of
    # inf - inf; it is larger than any in-image squared distance.
    unreachable = float(dynamic.shape[0] ** 2 + dynamic.shape[1] ** 2) * 4.0 + 1.0
    grid = np.where(dynamic, 0.0, unreachable)
    grid = np.apply_along_axis(_edt_1d, 0, grid)
    return np.apply_along_axis(_edt_1d, 1, grid)


def _edt_1d(values: np.ndarray) -> np.ndarray:
    """1-D squared distance transform: ``min_j (i - j)^2 + values[j]``.

    Lower envelope of the parabolas rooted at each sample (Felzenszwalb &
    Huttenlocher, "Distance Transforms of Sampled Functions").
    """

    n = values.shape[0]
    roots = np.zeros(n, dtype=np.int64)
    bounds = np.empty(n + 1, dtype=np.float64)
    bounds[0] = -np.inf
    bounds[1] = np.inf
    k = 0
    for q in range(1, n):
        crossing = _parabola_crossing(values, q, roots[k])
        while crossing <= bounds[k]:
            k -= 1
            crossing = _parabola_crossing(values, q, roots[k])
        k += 1
        roots[k] = q
        bounds[k] = crossing
        bounds[k + 1] = np.inf
    out = np.empty(n, dtype=np.float64)
    k = 0
    for q in range(n):
        while bounds[k + 1] < q:
            k += 1
        root = roots[k]
        out[q] = float((q - root) ** 2) + values[root]
    return out


def _parabola_crossing(values: np.ndarray, q: int, root: int) -> float:
    """Where the parabola rooted at ``q`` overtakes the one rooted at ``root``."""

    return ((values[q] + q * q) - (values[root] + root * root)) / (2.0 * q - 2.0 * root)
