"""LanceDB-backed PyTorch dataset for BDD100K-shaped detection rows."""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any, Iterable

REQUIRED_COLUMNS = ("image_bytes", "ann_bboxes", "ann_categories")
LOGGER = logging.getLogger(__name__)


class DetectionDatasetError(RuntimeError):
    """Raised when Lance rows cannot be converted into detector samples."""


class LanceDetectionDataset:
    """PyTorch-style dataset backed by a LanceDB table or materialized view."""

    def __init__(
        self,
        *,
        lance_uri: str = "",
        view: str = "",
        filter_sql: str | None = None,
        rows: Iterable[dict[str, Any]] | None = None,
        limit: int | None = None,
        label_map: dict[str, int] | None = None,
    ) -> None:
        self.lance_uri = lance_uri
        self.view = view
        self.filter_sql = filter_sql
        self.label_map = label_map
        self.rows = list(rows) if rows is not None else _read_lance_rows(lance_uri, view, filter_sql, limit)
        if not self.rows:
            raise DetectionDatasetError("detection dataset is empty")
        _validate_rows(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return _row_to_sample(self.rows[index], label_map=self.label_map)


def collate_detection_batch(batch: list[tuple[Any, dict[str, Any]]]):
    """Collate detection samples as torchvision expects: lists of images and targets."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_dataloader(
    *,
    lance_uri: str,
    view: str,
    batch_size: int,
    shuffle: bool = True,
    filter_sql: str | None = None,
    limit: int | None = None,
    label_map: dict[str, int] | None = None,
):
    """Create a torch DataLoader over a Lance materialized view."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - container/runtime path.
        raise DetectionDatasetError("torch is required for DataLoader construction") from exc
    dataset = LanceDetectionDataset(
        lance_uri=lance_uri,
        view=view,
        filter_sql=filter_sql,
        limit=limit,
        label_map=label_map,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_detection_batch)


def _read_lance_rows(
    lance_uri: str,
    view: str,
    filter_sql: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if not lance_uri.strip() or not view.strip():
        raise DetectionDatasetError("lance_uri and view are required")
    try:
        import lancedb
    except ImportError as exc:
        raise DetectionDatasetError("lancedb is required to read detection views") from exc
    try:
        db = lancedb.connect(lance_uri)
        table = db.open_table(view)
        if filter_sql:
            query = table.search().where(filter_sql)
            if limit:
                query = query.limit(limit)
            arrow_table = query.to_arrow()
        elif limit:
            arrow_table = table.search().limit(limit).to_arrow()
        else:
            arrow_table = table.to_arrow()
    except Exception as exc:
        raise DetectionDatasetError(f"failed to read Lance view {view}: {exc}") from exc
    return list(arrow_table.to_pylist())


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    missing = [name for name in REQUIRED_COLUMNS if name not in rows[0]]
    if missing:
        raise DetectionDatasetError(f"detection view is missing required column(s): {', '.join(missing)}")


def _row_to_sample(row: dict[str, Any], *, label_map: dict[str, int] | None = None):
    try:
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as exc:
        raise DetectionDatasetError("torch, numpy, and pillow are required to decode detection rows") from exc

    image = Image.open(BytesIO(_coerce_image_bytes(row["image_bytes"]))).convert("RGB")
    image_array = np.asarray(image, dtype="float32")
    image_tensor = torch.as_tensor(image_array, dtype=torch.float32).permute(2, 0, 1) / 255.0
    if label_map is None:
        boxes = _coerce_boxes(row["ann_bboxes"])
        categories = _coerce_categories(row["ann_categories"])
        if len(boxes) != len(categories):
            raise DetectionDatasetError("ann_bboxes and ann_categories length mismatch")
    else:
        boxes, categories = _coerce_mapped_targets(row["ann_bboxes"], row["ann_categories"], label_map=label_map)
    target = {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.as_tensor(categories, dtype=torch.int64) if categories else torch.empty((0,), dtype=torch.int64),
    }
    return image_tensor, target


def _coerce_image_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=True)
        except Exception:
            return value.encode("utf-8")
    if hasattr(value, "as_py"):
        return _coerce_image_bytes(value.as_py())
    raise DetectionDatasetError(f"unsupported image_bytes type: {type(value).__name__}")


def _coerce_boxes(value: Any) -> list[list[float]]:
    raw = value.as_py() if hasattr(value, "as_py") else value
    boxes: list[list[float]] = []
    for box in raw or []:
        coords = box.as_py() if hasattr(box, "as_py") else box
        if len(coords) != 4:
            raise DetectionDatasetError("each bbox must have four coordinates")
        x1, y1, x2, y2 = [float(coord) for coord in coords]
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return boxes


def _coerce_categories(value: Any) -> list[int]:
    raw = value.as_py() if hasattr(value, "as_py") else value
    return [int(item.as_py() if hasattr(item, "as_py") else item) for item in raw or []]


def _coerce_mapped_targets(
    boxes_value: Any,
    categories_value: Any,
    *,
    label_map: dict[str, int],
) -> tuple[list[list[float]], list[int]]:
    raw_boxes = boxes_value.as_py() if hasattr(boxes_value, "as_py") else boxes_value
    raw_categories = categories_value.as_py() if hasattr(categories_value, "as_py") else categories_value
    boxes: list[list[float]] = []
    categories: list[int] = []
    unknown_labels: set[str] = set()
    unknown_count = 0
    for raw_category, raw_box in zip(raw_categories or [], raw_boxes or [], strict=False):
        label_id = _mapped_label_id(raw_category, label_map)
        if label_id < 0:
            unknown_labels.add(str(raw_category.as_py() if hasattr(raw_category, "as_py") else raw_category))
            unknown_count += 1
            continue
        box = _coerce_box_or_none(raw_box)
        if box is None:
            continue
        boxes.append(box)
        categories.append(label_id)
    if unknown_labels:
        LOGGER.warning(
            "filtered %s detection annotation(s) with unknown label(s): %s",
            unknown_count,
            ", ".join(sorted(unknown_labels)),
        )
    return boxes, categories


def _mapped_label_id(value: Any, label_map: dict[str, int]) -> int:
    raw = value.as_py() if hasattr(value, "as_py") else value
    if isinstance(raw, str):
        return int(label_map.get(raw, -1))
    return int(raw)


def _coerce_box_or_none(value: Any) -> list[float] | None:
    coords = value.as_py() if hasattr(value, "as_py") else value
    if len(coords) != 4:
        raise DetectionDatasetError("each bbox must have four coordinates")
    x1, y1, x2, y2 = [float(coord) for coord in coords]
    if x2 > x1 and y2 > y1:
        return [x1, y1, x2, y2]
    return None
