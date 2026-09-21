"""Measure whether the Rerun viewer clips geometry inside its own panes.

The offscreen renders cannot answer this. They are drawn at the aspect the camera
assumed, so a camera fitted for the wrong pane shape looks perfectly framed in
them and clips in the viewer. This reads the viewer's own captured frame.

Background model: the viewer paints a fixed dark gradient that varies vertically
and barely at all horizontally, so it is sampled per row from the pane's darkest
pixels rather than from the row median. The median inverts once geometry fills
most of a row -- content becomes the "background" and the measurement silently
reverses, which is exactly the trap this replaces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

#: How far a pixel must sit from the row's background before it counts as drawn.
#:
#: 90 was too low and reported every pane as clipped on every edge, because the viewer
#: paints a floor grid across the whole pane and 90 admits it as content. The measurement
#: is not delicate once the grid is excluded: on a 1024x768 capture the distance
#: distribution is sharply bimodal, median 14 for background and grid against a 90th
#: percentile of 376 for geometry, and every threshold from 150 to 320 returns the same
#: extent to within 0.015 of pane width. Above roughly 350 it starts eating the geometry's
#: own darker pixels instead. 220 sits in the middle of that plateau.
CONTENT_DISTANCE = 220
#: Share of a column that must be content before that column counts as occupied,
#: so one stray antialiased pixel cannot claim the frame edge.
COLUMN_OCCUPANCY = 0.005
#: Rows/columns within this many pixels of a pane edge count as touching it.
EDGE_BAND = 8


def _pane_bounds(frame: np.ndarray) -> int:
    """Find the vertical divider between the scene pane and the tab column.

    The divider is dark down the *whole* frame, which is what separates it from a dark
    edge of the geometry. Scoring columns by their darkest pixel picked a geometry edge
    instead and put the divider 62 pixels early on a 1024-wide capture, so this scores
    them by the share of rows that are dark and takes the most consistent column.
    """

    width = frame.shape[1]
    dark = (frame.sum(axis=2) < 200).mean(axis=0)
    lo, hi = int(width * 0.55), int(width * 0.75)
    return lo + int(np.argmax(dark[lo:hi]))


def audit_pane(pane: np.ndarray) -> dict:
    """Content extent and edge contact for one pane."""

    # Per-row background from the darkest tenth of the row: the gradient is the
    # darkest thing present, and geometry in this viewer is always lighter.
    floor = np.quantile(pane.reshape(pane.shape[0], -1, 3), 0.10, axis=1)
    content = np.abs(pane - floor[:, None, :]).sum(axis=2) > CONTENT_DISTANCE
    height, width = content.shape
    columns = content.sum(axis=0) > height * COLUMN_OCCUPANCY
    rows = content.sum(axis=1) > width * COLUMN_OCCUPANCY
    if not columns.any() or not rows.any():
        return {"pane_aspect": round(width / height, 3), "content": "none"}
    cols = np.flatnonzero(columns)
    rws = np.flatnonzero(rows)
    touching = [
        name
        for name, hit in (
            ("left", columns[:EDGE_BAND].any()),
            ("right", columns[-EDGE_BAND:].any()),
            ("top", rows[:EDGE_BAND].any()),
            ("bottom", rows[-EDGE_BAND:].any()),
        )
        if hit
    ]
    return {
        "pane_aspect": round(width / height, 3),
        "width_span": round(float((cols[-1] - cols[0] + 1) / width), 3),
        "height_span": round(float((rws[-1] - rws[0] + 1) / height), 3),
        "touches_pane_edges": touching,
        "clipped": bool(touching),
    }


def _pane_body(frame: np.ndarray, split: int) -> slice:
    """Find the 3D pane's own rows, without assuming the capture's DPI scale.

    The chrome is a fixed *logical* height, so a hardcoded pixel trim is only correct
    at the scale it was tuned for -- 110 rows suited a 2x screenshot and eats into the
    pane of a 1x window grab. The viewer's 3D background is the one strongly green
    thing on screen, so the pane is the rows where green leads both other channels.
    """

    column = frame[:, 80 : max(split - 40, 120)]
    green = column[:, :, 1] - (column[:, :, 0] + column[:, :, 2]) / 2.0
    is_pane = np.median(green, axis=1) > 8.0
    rows = np.flatnonzero(is_pane)
    if rows.size == 0:
        return slice(110, frame.shape[0] - 90)
    return slice(int(rows[0]), int(rows[-1]) + 1)


def audit(path: Path) -> dict:
    frame = np.asarray(Image.open(path).convert("RGB")).astype(int)
    height, width, _ = frame.shape
    split = _pane_bounds(frame)
    body = _pane_body(frame, split)
    # Inset off the pane borders proportionally, for the same reason the row trim is
    # detected rather than fixed: a 60-pixel left inset suits a 2x screenshot and on a
    # 1x window grab it lands exactly where the geometry begins, which then reads as the
    # geometry touching the edge.
    inset = max(int(round(width * 0.008)), 2)
    # Refuse to answer rather than answer wrongly. The viewer's chrome cannot take most of
    # the frame, so a body this short means the green-background detection found a stripe
    # instead of the pane -- which is what it does on the earlier comparison pair, where it
    # reported pane aspects of 4.3 and 5.4 that no 2:1 column split in any ordinary window
    # can produce. Those frames were measured analytically instead.
    if (body.stop - body.start) < height * 0.40:
        return {
            "frame": path.name,
            "size": [width, height],
            "pane_detection": "failed",
            "detected_body_rows": [body.start, body.stop],
            "why": "the 3D background was not found across enough rows to be the pane; "
            "no clipping verdict is reported for this frame",
        }
    return {
        "frame": path.name,
        "size": [width, height],
        "divider_x": split,
        "scene_pane": audit_pane(frame[body, inset : split - inset]),
        "active_tab_pane": audit_pane(frame[body, split + inset : width - inset]),
    }


if __name__ == "__main__":
    print(json.dumps([audit(Path(p)) for p in sys.argv[1:]], indent=2))
