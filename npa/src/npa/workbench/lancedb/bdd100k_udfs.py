"""BDD100K-specific CPU UDFs for LanceDB backfills.

The perceptual hash UDF uses the standard 8x8 difference hash algorithm with
Pillow only: decode image bytes, convert to grayscale, resize to 9x8, then
compare adjacent pixels row-major. The 64-bit hash is stored in signed int64
two's-complement form so it fits LanceDB/PyArrow `int64`.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from PIL import Image


@dataclass(frozen=True)
class UDFSpec:
    """Registry metadata for one BDD100K UDF."""

    name: str
    function: Callable[..., pa.Array]
    input_columns: tuple[str, ...]
    output_column: str
    output_type: pa.DataType
    dependencies: tuple[str, ...] = ()


def udf_has_person(batch: pa.RecordBatch) -> pa.Array:
    """Return whether each row has a person annotation."""
    return _contains_category(batch, "person")


def udf_has_rider(batch: pa.RecordBatch) -> pa.Array:
    """Return whether each row has a rider annotation."""
    return _contains_category(batch, "rider")


def udf_person_bbox_area_pct(batch: pa.RecordBatch) -> pa.Array:
    """Return summed person bbox area divided by image area for each row."""
    categories = _column(batch, "ann_categories").to_pylist()
    bboxes = _column(batch, "ann_bboxes").to_pylist()
    widths = _column(batch, "width").to_pylist()
    heights = _column(batch, "height").to_pylist()
    values: list[float] = []
    for row_categories, row_bboxes, width, height in zip(categories, bboxes, widths, heights, strict=True):
        image_area = float(width or 0) * float(height or 0)
        if image_area <= 0.0:
            values.append(0.0)
            continue
        area = 0.0
        for category, bbox in zip(row_categories or [], row_bboxes or [], strict=False):
            if category != "person" or not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox[:4])
            area += max(0.0, x2 - x1) * max(0.0, y2 - y1)
        values.append(float(area / image_area))
    return pa.array(values, type=pa.float32())


def udf_dhash(batch: pa.RecordBatch) -> pa.Array:
    """Return a signed int64 difference hash for each row's image bytes."""
    hashes = [_to_signed_int64(_dhash_bytes(bytes(raw))) for raw in _column(batch, "image_bytes").to_pylist()]
    return pa.array(hashes, type=pa.int64())


def udf_is_duplicate(
    batch: pa.RecordBatch,
    *,
    dhash_column: str = "dhash",
    hamming_threshold: int = 5,
) -> pa.Array:
    """Return deterministic near-duplicate flags for a batch sorted by image_id."""
    if hamming_threshold < 0:
        raise ValueError("hamming_threshold must be non-negative")
    hashes = _column(batch, dhash_column).to_pylist()
    seen: list[int] = []
    values: list[bool] = []
    for raw_hash in hashes:
        if raw_hash is None:
            values.append(False)
            continue
        current = _to_unsigned_int64(int(raw_hash))
        duplicate = any(_hamming_distance(current, previous) <= hamming_threshold for previous in seen)
        values.append(duplicate)
        seen.append(current)
    return pa.array(values, type=pa.bool_())


BDD100K_UDFS: dict[str, UDFSpec] = {
    "has_person": UDFSpec(
        name="has_person",
        function=udf_has_person,
        input_columns=("ann_categories",),
        output_column="has_person",
        output_type=pa.bool_(),
    ),
    "has_rider": UDFSpec(
        name="has_rider",
        function=udf_has_rider,
        input_columns=("ann_categories",),
        output_column="has_rider",
        output_type=pa.bool_(),
    ),
    "person_bbox_area_pct": UDFSpec(
        name="person_bbox_area_pct",
        function=udf_person_bbox_area_pct,
        input_columns=("ann_categories", "ann_bboxes", "width", "height"),
        output_column="person_bbox_area_pct",
        output_type=pa.float32(),
    ),
    "dhash": UDFSpec(
        name="dhash",
        function=udf_dhash,
        input_columns=("image_bytes",),
        output_column="dhash",
        output_type=pa.int64(),
    ),
    "is_duplicate": UDFSpec(
        name="is_duplicate",
        function=udf_is_duplicate,
        input_columns=("image_id", "dhash"),
        output_column="is_duplicate",
        output_type=pa.bool_(),
        dependencies=("dhash",),
    ),
}


def _contains_category(batch: pa.RecordBatch, category: str) -> pa.Array:
    values = [category in (row_categories or []) for row_categories in _column(batch, "ann_categories").to_pylist()]
    return pa.array(values, type=pa.bool_())


def _column(batch: pa.RecordBatch, name: str) -> pa.Array:
    try:
        return batch.column(name)
    except KeyError as exc:
        raise ValueError(f"batch is missing required column {name!r}") from exc


def _dhash_bytes(raw: bytes) -> int:
    with Image.open(io.BytesIO(raw)) as image:
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        grayscale = image.convert("L").resize((9, 8), resampling)
        pixels = list(grayscale.getdata())
    value = 0
    bit = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            if pixels[offset + col] > pixels[offset + col + 1]:
                value |= 1 << bit
            bit += 1
    return value


def _to_signed_int64(value: int) -> int:
    value &= (1 << 64) - 1
    if value >= (1 << 63):
        return value - (1 << 64)
    return value


def _to_unsigned_int64(value: int) -> int:
    return value & ((1 << 64) - 1)


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


__all__ = [
    "BDD100K_UDFS",
    "UDFSpec",
    "udf_dhash",
    "udf_has_person",
    "udf_has_rider",
    "udf_is_duplicate",
    "udf_person_bbox_area_pct",
]
