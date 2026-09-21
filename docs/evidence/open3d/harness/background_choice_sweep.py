"""Choose the render background from the content's own luminance, not from taste.

The delivered frames draw a scan whose colours span a near-white floor to black
sensor gaps onto a 250 background. The floor then sits 60-80 luminance levels from
the page, which is the "washed out" reading: legible in principle, pale in
practice.

I assumed darkening the page would be a trade, buying the light floor's contrast
at the dark speckle's expense, and wrote this to measure the trade instead of
guessing it. The measurement found no trade -- see the recorded reading -- which is
the reason to run it rather than reason about it. The content luminances are fixed
by the geometry and its real colours, so for each candidate background this
computes what a reader actually gets and reports the background that maximises the
worst-served content.

Reads the delivered PNGs. Does not modify them, and does not recolour geometry --
the point is to change the page, never the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

WEIGHTS = np.array([0.2126, 0.7152, 0.0722])
#: Content this close to the background is drawn and effectively unreadable.
LOW_CONTRAST_LIMIT = 24


def content_luminance(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    values, counts = np.unique(corners, axis=0, return_counts=True)
    background = values[int(np.argmax(counts))]
    mask = np.abs(rgb - background).max(axis=2) > 6
    return (rgb[mask] * WEIGHTS).sum(axis=1)


def score(luminance: np.ndarray, background_level: float) -> dict:
    contrast = np.abs(luminance - background_level)
    return {
        "background_luminance": float(background_level),
        "median_contrast": float(np.median(contrast)),
        # The reader's binding constraint is the worst-served content, not the
        # average, so the 5th percentile is the number that decides this.
        "p05_contrast": float(np.percentile(contrast, 5)),
        "low_contrast_fraction": float((contrast < LOW_CONTRAST_LIMIT).mean()),
    }


def main() -> int:
    out = Path(sys.argv[1])
    frames = [Path(a) for a in sys.argv[2:]]
    luminance = np.concatenate([content_luminance(p) for p in frames])

    candidates = np.arange(0, 256, 2, dtype=float)
    rows = [score(luminance, level) for level in candidates]

    # Maximise the worst-served content, breaking ties toward the median.
    best = max(rows, key=lambda r: (r["p05_contrast"], r["median_contrast"]))
    shipped = score(luminance, float((np.array([250, 250, 250]) * WEIGHTS).sum()))

    report = {
        "what": (
            "Contrast a reader actually gets from the delivered content, as a "
            "function of background luminance. Content luminances are taken from "
            "the delivered frames and never altered; only the page changes."
        ),
        "frames_measured": [p.name for p in frames],
        "content_pixels": int(luminance.size),
        "content_luminance_percentiles": {
            str(q): float(np.percentile(luminance, q)) for q in (1, 5, 25, 50, 75, 95, 99)
        },
        "shipped_background": shipped,
        "best_background": best,
        "sweep": rows,
        "reading": (
            "I expected this to be a trade -- a light floor against dark sensor "
            "gaps, with no background serving both -- and the measurement says "
            "otherwise. The content is not bimodal: it is overwhelmingly light, "
            "with a median luminance near 174 and only about 1% below 34. So there "
            "is no trade to balance, and darkening the page improves both the "
            "worst-served content and the typical content at once. The light floor "
            "the review called washed out is most of the picture, which is exactly "
            "why putting it on a near-white page cost so much."
        ),
        "how_to_read_this_against_the_delivered_frames": (
            "This sweep holds the content fixed and varies only the page, which is "
            "the controlled comparison and the only one that isolates the page. "
            "Comparing the old delivered frames directly against the new ones does "
            "not isolate it: the framing fix changed which pixels are drawn at all, "
            "so the share of low-contrast content there moves for two reasons at "
            "once. It rises slightly in that confounded comparison while falling in "
            "this controlled one. Both readings are recorded; the controlled one is "
            "the basis for the choice of page."
        ),
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"shipped bg lum {shipped['background_luminance']:5.1f}: "
        f"p05 {shipped['p05_contrast']:5.1f}  median {shipped['median_contrast']:5.1f}  "
        f"washed {shipped['low_contrast_fraction'] * 100:5.2f}%"
    )
    print(
        f"best    bg lum {best['background_luminance']:5.1f}: "
        f"p05 {best['p05_contrast']:5.1f}  median {best['median_contrast']:5.1f}  "
        f"washed {best['low_contrast_fraction'] * 100:5.2f}%"
    )
    for q in (1, 5, 25, 50, 75, 95, 99):
        print(f"  content p{q:<2} luminance {np.percentile(luminance, q):6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
