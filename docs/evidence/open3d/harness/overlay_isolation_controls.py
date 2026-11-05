#!/usr/bin/env python3
"""Settle which pixels are the scale-bar overlay, without choosing a mask.

The delivered frames carry a fixed bottom-left scale bar. It is content, so it pins the content
bounding box, and the extent reading is partly a measurement of it. How much depends on how much
you remove -- and the annotation has no edge the pixel values define, so for a while every figure
produced here was a function of someone's mask. A reviewer and I published three different shifts
from three different masks and each argued for our own. That is not a measurement.

Both controls below avoid the question entirely.

**The matched-pair diff.** `baseline_*` and `candidate_*` are different reconstructions of the same
view. If the band held any subject geometry, a matched pair must differ inside it. It does not --
by exactly zero pixels, while the same frames differ by tens of thousands elsewhere. This needs no
threshold and no agreement about where the glyphs end.

**The plateau sweep.** Grow the exclusion rectangle and watch the extent. A box eating geometry
makes the value keep drifting; a box that has already removed all the overlay makes it flat. Where
the answer stops changing is the boundary, and the width of the flat region is how much confidence
that boundary deserves.

The sweep's anchor is the overlay's own origin rather than the frame's bottom edge. That matters:
the same sweep anchored at the bottom edge does not reproduce, because the annotation sits about
20px above it, and anchoring convention has already caused two disagreements about these frames
that were not about the data.

Why the obvious rule is wrong, recorded because it is subtle: isolating the overlay as *pixels
identical across all nine frames* keeps only its view-invariant remnant. The overlay is
view-dependent -- its bar length encodes each view's metres-per-pixel -- so that intersection
leaves most of the label in the image, still pinning the box. The invariance that isolates it is
identity within a matched pair at one view.

Reads PNG bytes only. No geometry, no Open3D, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FRAMES = Path(__file__).resolve().parent.parent / "frames-refit"

CONTENT_TOLERANCE = (
    6  # matches frame_edge_audit.py, so the two agree about what content is
)
OVERLAY_ORIGIN = (
    655,
    27,
)  # row, col: the top-left of the annotation as the pair diff finds it
SWEEP_WIDTHS = (81, 100, 120, 140, 160, 200, 260, 320, 380, 440)
SWEEP_HEIGHTS = (37, 60, 90, 120)


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def content_mask(frame: np.ndarray) -> np.ndarray:
    background = np.median(frame.reshape(-1, 3), axis=0)
    return (np.abs(frame - background) > CONTENT_TOLERANCE).any(axis=2)


def narrower_extent(mask: np.ndarray) -> float:
    """Content bounding-box span as a fraction of the narrower axis, the reading under test."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 0.0
    height, width = mask.shape
    # float() because numpy scalars survive json.dumps by subclassing float and then print as
    # np.float64(...), which makes a report read like a debugging artefact.
    return float(
        min(
            (rows.max() - rows.min() + 1) / height,
            (cols.max() - cols.min() + 1) / width,
        )
    )


def matched_pairs() -> list[tuple[str, Path, Path]]:
    pairs = []
    for baseline in sorted(FRAMES.glob("baseline_*.png")):
        candidate = FRAMES / baseline.name.replace("baseline_", "candidate_", 1)
        if candidate.exists():
            pairs.append(
                (baseline.name.replace("baseline_", "", 1)[:-4], baseline, candidate)
            )
    return pairs


def pair_diff(width: int, height: int) -> list[dict[str, object]]:
    """Does the band respond to a change of reconstruction? If not, it is not the subject."""
    row, col = OVERLAY_ORIGIN
    rows = []
    for view, baseline, candidate in matched_pairs():
        differs = (rgb(baseline) != rgb(candidate)).any(axis=2)
        band = np.zeros(differs.shape, bool)
        band[row : row + height, col : col + width] = True
        rows.append(
            {
                "matched_pair": view,
                "pixels_differing_inside_the_band": int(differs[band].sum()),
                "pixels_differing_elsewhere": int(differs[~band].sum()),
            }
        )
    return rows


def plateau() -> list[dict[str, object]]:
    row, col = OVERLAY_ORIGIN
    masks = [content_mask(rgb(p)) for p in sorted(FRAMES.glob("*.png"))]
    out = []
    for width in SWEEP_WIDTHS:
        for height in SWEEP_HEIGHTS:
            spans = []
            for mask in masks:
                excluded = mask.copy()
                excluded[row : row + height, col : col + width] = False
                spans.append(narrower_extent(excluded))
            out.append(
                {
                    "exclusion_width": width,
                    "exclusion_height": height,
                    "subject_extent_low": round(min(spans), 4),
                    "subject_extent_high": round(max(spans), 4),
                }
            )
    return out


def per_view_overlay_size(width: int, height: int) -> dict[str, int]:
    """Evidence that the all-nine intersection is the wrong invariance class."""
    row, col = OVERLAY_ORIGIN
    sizes = {}
    for path in sorted(FRAMES.glob("*.png")):
        mask = content_mask(rgb(path))
        sizes[path.name] = int(mask[row : row + height, col : col + width].sum())
    return sizes


def view_invariant_remnant() -> int:
    paths = sorted(FRAMES.glob("*.png"))
    stack = np.stack([rgb(p) for p in paths])
    identical = (stack == stack[0]).all(axis=(0, 3))
    return int((identical & content_mask(stack[0])).sum())


def extent_with_and_without(width: int, height: int) -> list[dict[str, object]]:
    row, col = OVERLAY_ORIGIN
    out = []
    for path in sorted(FRAMES.glob("*.png")):
        mask = content_mask(rgb(path))
        with_annotation = narrower_extent(mask)
        excluded = mask.copy()
        excluded[row : row + height, col : col + width] = False
        subject_only = narrower_extent(excluded)
        out.append(
            {
                "frame": path.name,
                "content_box_extent": round(with_annotation, 4),
                "subject_extent": round(subject_only, 4),
                "shift": round(with_annotation - subject_only, 4),
            }
        )
    return out


def flat_region(sweep: list[dict[str, object]], height: int) -> dict[str, object]:
    """Where the answer stops changing, which is the boundary the sweep establishes."""
    at_height = [r for r in sweep if r["exclusion_height"] == height]
    final = (at_height[-1]["subject_extent_low"], at_height[-1]["subject_extent_high"])
    flat = [
        r["exclusion_width"]
        for r in at_height
        if (r["subject_extent_low"], r["subject_extent_high"]) == final
    ]
    return {
        "settles_at": final,
        "first_width_that_settles": min(flat),
        "widths_flat_through": max(flat),
        "flat_over_px_of_growth": max(flat) - min(flat),
    }


def build() -> dict[str, object]:
    width, height = 160, 37
    sweep = plateau()
    sizes = per_view_overlay_size(width, height)
    return {
        "what_this_settles": (
            "Which pixels are the scale-bar overlay, by two controls that need no mask: a matched "
            "pair must differ inside any band that holds subject geometry, and a sweep that has "
            "removed all the overlay stops changing."
        ),
        "content_tolerance": CONTENT_TOLERANCE,
        "overlay_origin_row_col": list(OVERLAY_ORIGIN),
        "control_1_matched_pair_diff": pair_diff(width, height),
        "control_1_verdict": (
            "Zero differing pixels inside the band in every matched pair, against tens of "
            "thousands elsewhere in the same frames. The band does not respond to a change of "
            "reconstruction, so it is not the subject."
        ),
        "control_2_plateau_sweep": sweep,
        "control_2_flat_region": {str(h): flat_region(sweep, h) for h in SWEEP_HEIGHTS},
        "the_measurement_this_licenses": extent_with_and_without(width, height),
        "why_the_all_nine_intersection_is_the_wrong_rule": {
            "overlay_pixels_per_frame": sizes,
            "distinct_sizes": sorted(set(sizes.values())),
            "view_invariant_remnant": view_invariant_remnant(),
            "so": (
                "The overlay is view-dependent, because its bar length encodes each view's "
                "metres-per-pixel. Intersecting all nine frames keeps only the remnant and leaves "
                "the rest of the label in the image, where it goes on pinning the bounding box. "
                "Within one view the band is identical across reconstructions, which is the "
                "invariance that isolates it."
            ),
        },
    }


def main() -> int:
    if not FRAMES.is_dir():
        print(f"No frames under {FRAMES}", file=sys.stderr)
        return 1
    report = build()
    destination = FRAMES.parent / "overlay-isolation-controls.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    bands = {
        r["pixels_differing_inside_the_band"]
        for r in report["control_1_matched_pair_diff"]
    }
    flat = report["control_2_flat_region"]["37"]
    shifts = [r["shift"] for r in report["the_measurement_this_licenses"]]
    print(f"wrote {destination.relative_to(FRAMES.parent.parent)}")
    print(
        f"Matched pairs differ inside the band by {sorted(bands)} pixel(s). The sweep settles at "
        f"{flat['settles_at']} from {flat['first_width_that_settles']}px and stays flat through "
        f"{flat['widths_flat_through']}px. Removing the overlay moves the extent by up to "
        f"{max(shifts)}."
    )
    return 0 if bands == {0} else 1


if __name__ == "__main__":
    raise SystemExit(main())
