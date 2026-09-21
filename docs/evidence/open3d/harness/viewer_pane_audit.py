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
CONTENT_DISTANCE = 90
#: Share of a column that must be content before that column counts as occupied,
#: so one stray antialiased pixel cannot claim the frame edge.
COLUMN_OCCUPANCY = 0.005
#: Rows/columns within this many pixels of a pane edge count as touching it.
EDGE_BAND = 8


def _pane_bounds(frame: np.ndarray) -> int:
    """Find the vertical divider between the scene pane and the tab column."""

    width = frame.shape[1]
    bright = (frame.sum(axis=2) > 300).mean(axis=0)
    window = bright[int(width * 0.55) : int(width * 0.75)]
    return int(width * 0.55) + int(np.argmin(window))


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


def audit(path: Path) -> dict:
    frame = np.asarray(Image.open(path).convert("RGB")).astype(int)
    height, width, _ = frame.shape
    split = _pane_bounds(frame)
    # Trim the viewer's own chrome: title bar, tab strip, and the time control.
    body = slice(110, height - 90)
    return {
        "frame": path.name,
        "size": [width, height],
        "divider_x": split,
        "scene_pane": audit_pane(frame[body, 60 : split - 8]),
        "active_tab_pane": audit_pane(frame[body, split + 8 : width - 20]),
    }


if __name__ == "__main__":
    print(json.dumps([audit(Path(p)) for p in sys.argv[1:]], indent=2))
