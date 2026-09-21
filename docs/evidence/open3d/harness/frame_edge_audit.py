"""Measure clipping and legibility in a rendered PNG, so neither is a matter of opinion.

Two readings the review kept having to make by eye, and got wrong in both
directions on the same pack:

**Clipping.** Content touching the image border is geometry the frame cut off.
Counted per edge, so "clipped at the bottom" is a number rather than an
impression. A frame is clipped when any border row or column carries content.

**Legibility.** Content whose luminance sits within a few levels of the
background is present in the file and invisible to a reader. Reported as the
share of content pixels that are low-contrast against the background, plus the
median contrast, because a mean hides a washed-out floor behind a dark wall.

Reads PNG bytes only. No geometry, no Open3D, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

#: A pixel counts as content when any channel differs from the background by more
#: than this. Anti-aliasing against a flat background lands within a level or two.
CONTENT_TOLERANCE = 6
#: Content within this luminance distance of the background is technically drawn
#: and practically unreadable. 24 of 255 is roughly where a mid-grey on white
#: stops resolving on a normal display at normal size.
LOW_CONTRAST_LIMIT = 24


def audit(path: Path, *, border: int = 2) -> dict:
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    height, width, _ = rgb.shape

    # The background is the modal corner colour: all four corners of a framed
    # render are background, and taking the mode tolerates one corner holding an
    # annotation.
    corners = np.stack(
        [rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]
    )
    values, counts = np.unique(corners, axis=0, return_counts=True)
    background = values[int(np.argmax(counts))]

    content = (np.abs(rgb - background).max(axis=2) > CONTENT_TOLERANCE)

    edges = {
        "top": content[:border, :],
        "bottom": content[-border:, :],
        "left": content[:, :border],
        "right": content[:, -border:],
    }
    touching = {name: int(band.sum()) for name, band in edges.items()}

    luminance = (rgb * np.array([0.2126, 0.7152, 0.0722])).sum(axis=2)
    background_luminance = float(
        (background * np.array([0.2126, 0.7152, 0.0722])).sum()
    )
    contrast = np.abs(luminance - background_luminance)[content]

    # How much of the frame the content spans, which is the other half of framing:
    # a frame can be unclipped and still strand its subject in empty space.
    rows, cols = np.where(content)
    span = (
        {
            "horizontal": float((cols.max() - cols.min() + 1) / width),
            "vertical": float((rows.max() - rows.min() + 1) / height),
        }
        if content.any()
        else {"horizontal": 0.0, "vertical": 0.0}
    )

    return {
        "frame": path.name,
        "resolution": [int(width), int(height)],
        "background_rgb": [int(v) for v in background],
        "content_pixels": int(content.sum()),
        "content_fraction": float(content.mean()),
        "border_width_examined": border,
        "content_pixels_touching_edge": touching,
        "clipped_edges": sorted(name for name, n in touching.items() if n > 0),
        "is_clipped": any(n > 0 for n in touching.values()),
        "content_span_fraction": span,
        "legibility": {
            "median_contrast_vs_background": float(np.median(contrast))
            if contrast.size
            else 0.0,
            "low_contrast_content_fraction": float(
                (contrast < LOW_CONTRAST_LIMIT).mean()
            )
            if contrast.size
            else 0.0,
            "low_contrast_limit": LOW_CONTRAST_LIMIT,
        },
    }


def main() -> int:
    out = Path(sys.argv[1])
    frames = [Path(a) for a in sys.argv[2:]]
    rows = [audit(path) for path in sorted(frames)]
    report = {
        "what": (
            "Per-frame clipping and legibility measured from PNG bytes. A frame is "
            "clipped when content occupies a border row or column; content is "
            "low-contrast when its luminance sits within "
            f"{LOW_CONTRAST_LIMIT}/255 of the background."
        ),
        "content_tolerance": CONTENT_TOLERANCE,
        "frames": rows,
        "clipped_frames": sorted(r["frame"] for r in rows if r["is_clipped"]),
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    for row in rows:
        print(
            f"{'CLIPPED' if row['is_clipped'] else '   ok  '} "
            f"{','.join(row['clipped_edges']) or '-':<22} "
            f"span {row['content_span_fraction']['horizontal']:.2f}x"
            f"{row['content_span_fraction']['vertical']:.2f}  "
            f"washed {row['legibility']['low_contrast_content_fraction'] * 100:5.1f}%  "
            f"{row['frame']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
