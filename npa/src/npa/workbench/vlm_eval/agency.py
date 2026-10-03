"""Measure the narrow stylized-frame oracle used by VLM agency calibration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import re
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image


AGENCY_STRUCTURAL_KIND = "stylized_color_bbox_agency_v1"
AGENCY_STRUCTURAL_CLAIMS = ("object_elevates", "actor_causes_motion")
AGENCY_STRUCTURAL_FRAME_SELECTION = "sequence"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AgencyStructuralError(ValueError):
    """Report a malformed or unevaluable agency structural check.

    Args:
        message: Human-readable failure reason.

    Returns:
        None.

    Raises:
        None.
    """


class StructuralFrame(Protocol):
    """Describe the selected-frame attributes consumed by the oracle.

    Args:
        label: Ordered source-relative frame label.
        data: Exact normalized PNG bytes.

    Returns:
        None.

    Raises:
        None.
    """

    label: str
    data: bytes


@dataclass(frozen=True)
class AgencyStructuralCheck:
    """Configure one stylized color-mask structural assertion.

    Args:
        kind: Versioned structural-check implementation identifier.
        claim: Claim whose fail-closed structural verdict is requested.
        required_frame_selection: Exact selection mode the check accepts.
        required_max_frames: Exact selected-frame count the check accepts.
        required_labels: Ordered selected-frame labels.
        required_normalized_png_sha256: Ordered selected-frame byte hashes.
        title_rows_excluded: Leading image rows excluded from both masks.
        object_r_min_inclusive: Inclusive red-channel floor for object pixels.
        object_g_max_inclusive: Inclusive green-channel ceiling for object pixels.
        object_b_max_inclusive: Inclusive blue-channel ceiling for object pixels.
        actor_channel_min_inclusive: Inclusive per-channel floor for actor pixels.
        actor_rg_delta_max_inclusive: Inclusive red-green actor-channel delta.
        actor_gb_delta_max_inclusive: Inclusive green-blue actor-channel delta.
        contact_px_inclusive: Maximum horizontal gap treated as proximity.
        movement_px_inclusive: Minimum pixel displacement treated as movement.

    Returns:
        An immutable structural-check configuration.

    Raises:
        None.
    """

    kind: str
    claim: str
    required_frame_selection: str
    required_max_frames: int
    required_labels: tuple[str, ...]
    required_normalized_png_sha256: tuple[str, ...]
    title_rows_excluded: int
    object_r_min_inclusive: int
    object_g_max_inclusive: int
    object_b_max_inclusive: int
    actor_channel_min_inclusive: int
    actor_rg_delta_max_inclusive: int
    actor_gb_delta_max_inclusive: int
    contact_px_inclusive: int
    movement_px_inclusive: int


@dataclass(frozen=True)
class AgencyStructuralResult:
    """Retain the measurements and verdict for one structural assertion.

    Args:
        kind: Versioned structural-check implementation identifier.
        claim: Claim that was measured.
        verdict: Structural pass, fail, or inconclusive outcome.
        reason: Human-readable basis for the verdict.
        selected_frame_count: Number of measured selected frames.
        object_mask_count: Frames containing a resolved object mask.
        actor_mask_count: Frames containing a resolved actor mask.
        object_completeness: Fraction of frames with an object mask.
        actor_completeness: Fraction of frames with an actor mask.
        dual_mask_coverage: Fraction of frames with both masks.
        object_endpoint_completeness: Fraction of ordered endpoints with objects.
        object_pixel_counts: Object-mask pixel count for each frame.
        actor_pixel_counts: Actor-mask pixel count for each frame.
        object_boxes: Inclusive object bounding box for each frame.
        actor_boxes: Inclusive actor bounding box for each frame.
        horizontal_gaps: Actor-to-object horizontal gap for each frame.
        object_bottoms: Object bounding-box bottom for each frame.
        minimum_gap_px: Minimum resolved actor-to-object horizontal gap.
        object_motion_range_px: Range of resolved object-bottom coordinates.
        signed_first_to_last_vertical_rise_px: Ordered endpoint vertical rise.
        selected_labels: Ordered selected-frame labels.
        normalized_png_sha256: Ordered selected-frame byte hashes.
        decoded_rgb_sha256: Ordered decoded-RGB pixel hashes.
        config: Exact configuration used for the measurement.

    Returns:
        An immutable structural result.

    Raises:
        None.
    """

    kind: str
    claim: str
    verdict: str
    reason: str
    selected_frame_count: int
    object_mask_count: int
    actor_mask_count: int
    object_completeness: float
    actor_completeness: float
    dual_mask_coverage: float
    object_endpoint_completeness: float
    object_pixel_counts: tuple[int, ...]
    actor_pixel_counts: tuple[int, ...]
    object_boxes: tuple[tuple[int, int, int, int] | None, ...]
    actor_boxes: tuple[tuple[int, int, int, int] | None, ...]
    horizontal_gaps: tuple[int | None, ...]
    object_bottoms: tuple[int | None, ...]
    minimum_gap_px: int | None
    object_motion_range_px: int | None
    signed_first_to_last_vertical_rise_px: int | None
    selected_labels: tuple[str, ...]
    normalized_png_sha256: tuple[str, ...]
    decoded_rgb_sha256: tuple[str, ...]
    config: AgencyStructuralCheck


@dataclass(frozen=True)
class _FrameMeasurement:
    object_pixels: int
    actor_pixels: int
    object_box: tuple[int, int, int, int] | None
    actor_box: tuple[int, int, int, int] | None
    normalized_png_sha256: str
    decoded_rgb_sha256: str


_CHECK_FIELDS = {
    "kind",
    "claim",
    "required_frame_selection",
    "required_max_frames",
    "required_labels",
    "required_normalized_png_sha256",
    "title_rows_excluded",
    "object_r_min_inclusive",
    "object_g_max_inclusive",
    "object_b_max_inclusive",
    "actor_channel_min_inclusive",
    "actor_rg_delta_max_inclusive",
    "actor_gb_delta_max_inclusive",
    "contact_px_inclusive",
    "movement_px_inclusive",
}


def parse_agency_structural_check(value: Any) -> AgencyStructuralCheck:
    """Parse and strictly validate an optional benchmark structural check.

    Args:
        value: JSON-compatible structural-check object.

    Returns:
        A frozen structural-check configuration.

    Raises:
        AgencyStructuralError: If the object is malformed or unsupported.
    """

    payload = _check_payload(value)
    kind, claim = _check_identity(payload)
    labels, hashes, frame_count = _frame_contract(payload)
    return _build_check(
        payload,
        kind=kind,
        claim=claim,
        labels=labels,
        hashes=hashes,
        frame_count=frame_count,
    )


def evaluate_agency_structure(
    frames: Sequence[StructuralFrame],
    check: AgencyStructuralCheck,
    *,
    frame_selection: str,
    max_frames: int,
) -> AgencyStructuralResult:
    """Evaluate a configured claim against already-selected immutable frames.

    Args:
        frames: Exact normalized frames that will later be scored.
        check: Validated structural-check configuration.
        frame_selection: Selection mode requested by the benchmark.
        max_frames: Frame limit requested by the benchmark.

    Returns:
        Complete measurements and a pass, fail, or inconclusive verdict.

    Raises:
        AgencyStructuralError: If selection or frame bytes violate the contract.
    """

    _validate_selection(
        frames,
        check,
        frame_selection=frame_selection,
        max_frames=max_frames,
    )
    measurements = tuple(_measure_frame(frame, check) for frame in frames)
    normalized_hashes = _validate_normalized_hashes(measurements, check)
    return _build_result(frames, check, measurements, normalized_hashes)


def _build_result(
    frames: Sequence[StructuralFrame],
    check: AgencyStructuralCheck,
    measurements: tuple[_FrameMeasurement, ...],
    normalized_hashes: tuple[str, ...],
) -> AgencyStructuralResult:
    object_boxes = tuple(measurement.object_box for measurement in measurements)
    actor_boxes = tuple(measurement.actor_box for measurement in measurements)
    gaps = tuple(
        _horizontal_gap(actor, obj) for actor, obj in zip(actor_boxes, object_boxes)
    )
    bottoms = tuple(box[3] if box is not None else None for box in object_boxes)
    verdict, reason = _claim_verdict(check, object_boxes, actor_boxes, gaps, bottoms)

    return AgencyStructuralResult(
        kind=check.kind,
        claim=check.claim,
        verdict=verdict,
        reason=reason,
        selected_frame_count=len(frames),
        object_pixel_counts=tuple(item.object_pixels for item in measurements),
        actor_pixel_counts=tuple(item.actor_pixels for item in measurements),
        object_boxes=object_boxes,
        actor_boxes=actor_boxes,
        horizontal_gaps=gaps,
        object_bottoms=bottoms,
        minimum_gap_px=_minimum(gaps),
        object_motion_range_px=_range(bottoms),
        signed_first_to_last_vertical_rise_px=_signed_rise(bottoms),
        selected_labels=tuple(frame.label for frame in frames),
        normalized_png_sha256=normalized_hashes,
        decoded_rgb_sha256=tuple(
            measurement.decoded_rgb_sha256 for measurement in measurements
        ),
        config=check,
        **_mask_coverage(object_boxes, actor_boxes),
    )


def _mask_coverage(
    object_boxes: Sequence[tuple[int, int, int, int] | None],
    actor_boxes: Sequence[tuple[int, int, int, int] | None],
) -> dict[str, int | float]:
    total = len(object_boxes)
    object_count = sum(box is not None for box in object_boxes)
    actor_count = sum(box is not None for box in actor_boxes)
    dual_count = sum(
        obj is not None and actor is not None
        for obj, actor in zip(object_boxes, actor_boxes)
    )
    endpoint_count = sum(box is not None for box in (object_boxes[0], object_boxes[-1]))
    return {
        "object_mask_count": object_count,
        "actor_mask_count": actor_count,
        "object_completeness": object_count / total,
        "actor_completeness": actor_count / total,
        "dual_mask_coverage": dual_count / total,
        "object_endpoint_completeness": endpoint_count / 2,
    }


def _check_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AgencyStructuralError("structural_check must be an object")
    unknown = sorted(set(value) - _CHECK_FIELDS)
    missing = sorted(_CHECK_FIELDS - set(value))
    if unknown:
        raise AgencyStructuralError(
            f"structural_check contains unknown fields: {', '.join(unknown)}"
        )
    if missing:
        raise AgencyStructuralError(
            f"structural_check is missing fields: {', '.join(missing)}"
        )
    return value


def _check_identity(value: dict[str, Any]) -> tuple[str, str]:
    kind = _required_string(value, "kind")
    if kind != AGENCY_STRUCTURAL_KIND:
        raise AgencyStructuralError(f"unsupported structural_check kind: {kind}")
    claim = _required_string(value, "claim")
    if claim not in AGENCY_STRUCTURAL_CLAIMS:
        raise AgencyStructuralError(f"unsupported structural_check claim: {claim}")
    return kind, claim


def _frame_contract(
    value: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    labels = _string_tuple(value, "required_labels")
    hashes = _string_tuple(value, "required_normalized_png_sha256")
    frame_count = _integer(value, "required_max_frames", minimum=1)
    if len(labels) != frame_count or len(hashes) != frame_count:
        raise AgencyStructuralError(
            "structural_check required labels and hashes must match required_max_frames"
        )
    if len(set(labels)) != len(labels):
        raise AgencyStructuralError("structural_check required labels must be unique")
    if any(_SHA256_PATTERN.fullmatch(digest) is None for digest in hashes):
        raise AgencyStructuralError(
            "structural_check normalized PNG hashes must be lowercase SHA-256 values"
        )
    return labels, hashes, frame_count


def _build_check(
    value: dict[str, Any],
    *,
    kind: str,
    claim: str,
    labels: tuple[str, ...],
    hashes: tuple[str, ...],
    frame_count: int,
) -> AgencyStructuralCheck:
    return AgencyStructuralCheck(
        kind=kind,
        claim=claim,
        required_frame_selection=_required_frame_selection(value),
        required_max_frames=frame_count,
        required_labels=labels,
        required_normalized_png_sha256=hashes,
        title_rows_excluded=_integer(value, "title_rows_excluded", minimum=0),
        object_r_min_inclusive=_channel(value, "object_r_min_inclusive"),
        object_g_max_inclusive=_channel(value, "object_g_max_inclusive"),
        object_b_max_inclusive=_channel(value, "object_b_max_inclusive"),
        actor_channel_min_inclusive=_channel(value, "actor_channel_min_inclusive"),
        actor_rg_delta_max_inclusive=_channel(value, "actor_rg_delta_max_inclusive"),
        actor_gb_delta_max_inclusive=_channel(value, "actor_gb_delta_max_inclusive"),
        contact_px_inclusive=_integer(value, "contact_px_inclusive", minimum=0),
        movement_px_inclusive=_integer(value, "movement_px_inclusive", minimum=1),
    )


def _validate_normalized_hashes(
    measurements: tuple[_FrameMeasurement, ...],
    check: AgencyStructuralCheck,
) -> tuple[str, ...]:
    hashes = tuple(measurement.normalized_png_sha256 for measurement in measurements)
    if hashes != check.required_normalized_png_sha256:
        raise AgencyStructuralError(
            "selected normalized PNG hashes do not match structural_check"
        )
    return hashes


def _validate_selection(
    frames: Sequence[StructuralFrame],
    check: AgencyStructuralCheck,
    *,
    frame_selection: str,
    max_frames: int,
) -> None:
    if frame_selection != check.required_frame_selection:
        raise AgencyStructuralError(
            "structural_check requires frame_selection="
            f"{check.required_frame_selection}"
        )
    if max_frames != check.required_max_frames:
        raise AgencyStructuralError(
            f"structural_check requires max_frames={check.required_max_frames}"
        )
    if len(frames) != check.required_max_frames:
        raise AgencyStructuralError(
            f"structural_check requires {check.required_max_frames} selected frames"
        )
    labels = tuple(frame.label for frame in frames)
    if labels != check.required_labels:
        raise AgencyStructuralError(
            "selected frame labels or order do not match structural_check"
        )


def _measure_frame(
    frame: StructuralFrame, check: AgencyStructuralCheck
) -> _FrameMeasurement:
    rgb = _frame_rgb(frame, check)
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    object_mask = (
        (red >= check.object_r_min_inclusive)
        & (green <= check.object_g_max_inclusive)
        & (blue <= check.object_b_max_inclusive)
    )
    actor_mask = (
        (red >= check.actor_channel_min_inclusive)
        & (green >= check.actor_channel_min_inclusive)
        & (blue >= check.actor_channel_min_inclusive)
        & (np.abs(red - green) <= check.actor_rg_delta_max_inclusive)
        & (np.abs(green - blue) <= check.actor_gb_delta_max_inclusive)
    )
    object_mask[: check.title_rows_excluded, :] = False
    actor_mask[: check.title_rows_excluded, :] = False
    return _FrameMeasurement(
        object_pixels=int(object_mask.sum()),
        actor_pixels=int(actor_mask.sum()),
        object_box=_mask_box(object_mask),
        actor_box=_mask_box(actor_mask),
        normalized_png_sha256=hashlib.sha256(frame.data).hexdigest(),
        decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
    )


def _frame_rgb(frame: StructuralFrame, check: AgencyStructuralCheck) -> np.ndarray:
    try:
        with Image.open(BytesIO(frame.data)) as image:
            rgb = np.asarray(image.convert("RGB"))
    except (OSError, ValueError) as exc:
        raise AgencyStructuralError(
            f"selected frame {frame.label!r} is not a valid image"
        ) from exc
    if check.title_rows_excluded >= rgb.shape[0]:
        raise AgencyStructuralError(
            "structural_check title_rows_excluded removes the whole image"
        )

    return rgb


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, columns = np.where(mask)
    if len(columns) == 0:
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    )


def _horizontal_gap(
    actor_box: tuple[int, int, int, int] | None,
    object_box: tuple[int, int, int, int] | None,
) -> int | None:
    if actor_box is None or object_box is None:
        return None
    return object_box[0] - actor_box[2]


def _claim_verdict(
    check: AgencyStructuralCheck,
    object_boxes: Sequence[tuple[int, int, int, int] | None],
    actor_boxes: Sequence[tuple[int, int, int, int] | None],
    gaps: Sequence[int | None],
    bottoms: Sequence[int | None],
) -> tuple[str, str]:
    if check.claim == "object_elevates":
        if object_boxes[0] is None or object_boxes[-1] is None:
            return "inconclusive", "object mask is missing at an ordered endpoint"
        rise = _signed_rise(bottoms)
        if rise is not None and rise >= check.movement_px_inclusive:
            return "pass", f"object rises {rise}px from first to last frame"
        return "fail", f"object signed endpoint rise is {rise}px"

    complete = all(
        obj is not None and actor is not None
        for obj, actor in zip(object_boxes, actor_boxes)
    )
    if not complete:
        return "inconclusive", "actor and object masks are not complete in every frame"
    motion = _range(bottoms)
    if motion is None or motion < check.movement_px_inclusive:
        return "fail", f"object motion range is {motion}px"
    minimum_gap = _minimum(gaps)
    if minimum_gap is None or minimum_gap <= check.contact_px_inclusive:
        return (
            "inconclusive",
            "actor proximity is necessary but insufficient to establish grasp",
        )
    return "fail", f"minimum actor-object horizontal gap is {minimum_gap}px"


def _minimum(values: Sequence[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _range(values: Sequence[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) - min(present) if present else None


def _signed_rise(values: Sequence[int | None]) -> int | None:
    if not values or values[0] is None or values[-1] is None:
        return None
    return values[0] - values[-1]


def _required_string(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result.strip():
        raise AgencyStructuralError(f"structural_check {field} must be a string")
    return result.strip()


def _required_frame_selection(value: dict[str, Any]) -> str:
    frame_selection = _required_string(value, "required_frame_selection")
    if frame_selection != AGENCY_STRUCTURAL_FRAME_SELECTION:
        raise AgencyStructuralError(
            "structural_check required_frame_selection must be sequence"
        )
    return frame_selection


def _string_tuple(value: dict[str, Any], field: str) -> tuple[str, ...]:
    raw = value.get(field)
    if not isinstance(raw, list) or not raw:
        raise AgencyStructuralError(
            f"structural_check {field} must be a non-empty array"
        )
    if any(not isinstance(item, str) or not item.strip() for item in raw):
        raise AgencyStructuralError(
            f"structural_check {field} values must be non-empty strings"
        )
    return tuple(item.strip() for item in raw)


def _integer(
    value: dict[str, Any],
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int):
        raise AgencyStructuralError(f"structural_check {field} must be an integer")
    if result < minimum or (maximum is not None and result > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise AgencyStructuralError(
            f"structural_check {field} must be at least {minimum}{suffix}"
        )
    return result


def _channel(value: dict[str, Any], field: str) -> int:
    return _integer(value, field, minimum=0, maximum=255)
