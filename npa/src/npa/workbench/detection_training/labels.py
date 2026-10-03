"""Keep dataset category IDs distinct from Faster R-CNN's background class."""

from __future__ import annotations


def detector_label_map(label_map: dict[str, int] | None) -> dict[str, int] | None:
    """Preserve positive IDs; shift a zero-based dataset map past background 0."""
    if label_map is None:
        return None
    offset = 1 if 0 in label_map.values() else 0
    return {name: category_id + offset for name, category_id in label_map.items()}


def category_id_map(label_map: dict[str, int] | None) -> dict[int, int] | None:
    """Numeric materialized-view categories follow the same mapping as strings."""
    mapped = detector_label_map(label_map)
    if label_map is None or mapped is None:
        return None
    return {category_id: mapped[name] for name, category_id in label_map.items()}
