"""Measure where the gate's walls stand, instead of asserting that it has any.

``frame_property_gate.py`` claims its framing checks bracket a frame from both
sides: zoom in far enough and content reaches the border, zoom out far enough and
the subject strands, so there is no zoom that passes while ruining the framing.
That is a claim about the gate's shape, and a docstring is not evidence for it.

Two gates, swept separately, because they answer different questions
-------------------------------------------------------------------
The **framing** gates (border content, subject spread) are the ones whose shape
is in question, so they are swept on their own. The **scale-label** check is not
a framing property at all -- it hashes the label's exact pixels, so it rejects
every resample including a 1.01x one. Folding it into the sweep would collapse
every band to the single point 1.00 and measure nothing.

Keeping them apart is what makes the result readable, and it is also the finding:

* Framing alone leaves a **hole**. One frame fails at 1.1x and 1.2x and then
  **passes again at 1.3x**, because by then the scale bar has been cropped out of
  the frame entirely and the geometry happens to miss the border on that step. A
  heavily cropped frame that had discarded its scale reference scored as well
  framed.
* The label check closes it, by construction, for any crop or rescale whatsoever.

So the pair is sound while neither half is sufficient, and that is reported here
as two columns rather than asserted in prose.

It also measures the alternative the review rejected. ``content_fraction`` rises
monotonically with zoom, so a fill gate has a floor and **no ceiling**: the
cheapest way to pass one is to keep zooming, and the first thing to go is the
sparse geometry at the edges.

What this is not
----------------
A 2D rescale about the frame centre, not a re-render. It moves pixels without
re-projecting geometry, so it exercises the *gate*, not the renderer or the
camera fitting -- those are pinned separately in
``npa/tests/workbench/test_open3d.py``. Nearest-neighbour resampling throughout,
so no pixel value is invented: an interpolating filter would blend content into
background and shift the contrast reading, which would be measuring the filter.
"""

from __future__ import annotations

import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame_edge_audit import audit
from frame_property_gate import (
    LABEL_SHA256,
    failures,
    label_digest,
    subject_spread,
)

#: Zoom factors to sweep. Dense near 1.0, where the walls are expected to sit.
FACTORS = (
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
    1.30,
    1.40,
    1.50,
    1.75,
    2.00,
)


def background_of(rgb: np.ndarray) -> np.ndarray:
    """The modal corner colour, the same reading ``frame_edge_audit`` takes."""
    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    values, counts = np.unique(corners, axis=0, return_counts=True)
    return values[int(np.argmax(counts))]


def rezoom(path: Path, factor: float, out: Path) -> None:
    """Scale the frame about its centre, padding with its own background colour."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    background = background_of(np.asarray(image))

    scaled = image.resize(
        (max(1, round(width * factor)), max(1, round(height * factor))),
        Image.NEAREST,
    )
    canvas = Image.new("RGB", (width, height), tuple(int(v) for v in background))

    left = (width - scaled.width) // 2
    top = (height - scaled.height) // 2
    if factor >= 1.0:
        # Zooming in: take the centre crop, which is what a tighter camera sees.
        canvas.paste(scaled.crop((-left, -top, -left + width, -top + height)), (0, 0))
    else:
        canvas.paste(scaled, (left, top))
    canvas.save(out)


def _contiguous(flags: list[bool]) -> bool:
    indices = [i for i, ok in enumerate(flags) if ok]
    if not indices:
        return False
    return indices == list(range(indices[0], indices[-1] + 1))


def _holes(flags: list[bool], factors: tuple[float, ...]) -> list[float]:
    """Zooms that fail while sitting between two that pass -- a gap in the band."""
    indices = [i for i, ok in enumerate(flags) if ok]
    if not indices:
        return []
    return [factors[i] for i in range(indices[0], indices[-1] + 1) if not flags[i]]


def sweep(frames: list[Path], scratch: Path) -> dict:
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in sorted(frames):
        steps = []
        for factor in FACTORS:
            staged = scratch / f"{path.stem}__zoom{factor:.2f}.png"
            rezoom(path, factor, staged)
            measured = audit(staged)
            spread = subject_spread(staged)
            # Framing only: the label check is swept as its own column below.
            found = failures(measured, spread)
            label_ok = label_digest(staged) == LABEL_SHA256
            staged.unlink()
            steps.append(
                {
                    "factor": factor,
                    "framing_passes": not found,
                    "scale_label_intact": label_ok,
                    "passes_both": (not found) and label_ok,
                    "why_framing_failed": [f.split(":")[0] for f in found],
                    "subject_spread_narrower_axis": spread["narrower_axis"],
                    "border_content_pixels": sum(
                        measured["content_pixels_touching_edge"].values()
                    ),
                    "content_fraction_the_gate_ignores": measured["content_fraction"],
                }
            )

        framing = [s["framing_passes"] for s in steps]
        passing = [s["factor"] for s, ok in zip(steps, framing) if ok]
        fills = [s["content_fraction_the_gate_ignores"] for s in steps]
        holes = _holes(framing, FACTORS)
        rows.append(
            {
                "path": path.name,
                "framing_pass_band": (
                    {
                        "lowest_passing_zoom": min(passing),
                        "highest_passing_zoom": max(passing),
                    }
                    if passing
                    else None
                ),
                "framing_band_is_closed": bool(passing)
                and min(passing) > min(FACTORS)
                and max(passing) < max(FACTORS),
                "framing_band_is_contiguous": _contiguous(framing),
                "framing_zooms_that_fail_inside_the_band": holes,
                "wall_below_is": next(
                    (s["why_framing_failed"] for s in steps if not s["framing_passes"]),
                    [],
                ),
                "wall_above_is": next(
                    (
                        s["why_framing_failed"]
                        for s in reversed(steps)
                        if not s["framing_passes"]
                    ),
                    [],
                ),
                "zooms_where_the_scale_label_survived": [
                    s["factor"] for s in steps if s["scale_label_intact"]
                ],
                "zooms_passing_both_gates": [
                    s["factor"] for s in steps if s["passes_both"]
                ],
                "fill_rises_monotonically_with_zoom": all(
                    a <= b + 1e-12 for a, b in itertools.pairwise(fills)
                ),
                "steps": steps,
            }
        )

    closed = [r for r in rows if r["framing_band_is_closed"]]
    holed = [r for r in rows if not r["framing_band_is_contiguous"]]
    return {
        "what": (
            "Synthetic zoom sweep over each delivered frame. The framing gates are "
            "swept alone, because their shape is the claim under test; the "
            "scale-label check is reported as its own column because it rejects "
            "every resample and would otherwise collapse each band to one point."
        ),
        "method": (
            "Nearest-neighbour rescale about the frame centre, padded with the "
            "frame's own modal corner colour. Moves pixels; does not re-render "
            "geometry or re-fit a camera."
        ),
        "factors_swept": list(FACTORS),
        "framing_bands_closed": f"{len(closed)}/{len(rows)}",
        "framing_bands_with_a_hole": sorted(r["path"] for r in holed),
        "what_the_hole_was": (
            "candidate_scan_and_surface__topdown.png fails framing at 1.1x and "
            "1.2x and passes again at 1.3x. By 1.3x the scale bar has been "
            "cropped out of the frame, so it no longer contributes border "
            "content, and the geometry happens to miss the border on that step. "
            "Framing alone therefore scores a heavily cropped frame that has "
            "discarded its scale reference as well framed."
        ),
        "the_label_check_closes_it": (
            "The label's pixels are hashed, so no resample or crop survives. "
            "Across every frame the only zoom that keeps the label intact is "
            "1.00, which is the identity transform."
        ),
        "scale_label_survived_only_at": sorted(
            {f for r in rows for f in r["zooms_where_the_scale_label_survived"]}
        ),
        "fill_rises_monotonically_in_every_frame": all(
            r["fill_rises_monotonically_with_zoom"] for r in rows
        ),
        "why_that_last_line_matters": (
            "A fill gate therefore has a floor and no ceiling, so it is passed by "
            "zooming without limit, and what gets cropped first is the sparse "
            "incomplete geometry these views exist to show. That is the review's "
            "objection, measured rather than repeated."
        ),
        "frames": rows,
    }


def main() -> int:
    args = sys.argv[1:]
    out = None
    if args and args[0] == "--out":
        out, args = Path(args[1]), args[2:]
    frames = [Path(a) for a in args]
    if not frames:
        print("usage: gate_zoom_sweep.py [--out REPORT.json] FRAME.png ...")
        return 2

    with tempfile.TemporaryDirectory(prefix="open3d-gate-zoom-") as temporary:
        report = sweep(frames, Path(temporary))
    if out is not None:
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for row in report["frames"]:
        band = row["framing_pass_band"]
        shape = (
            f"{band['lowest_passing_zoom']:.2f}..{band['highest_passing_zoom']:.2f}"
            if band
            else "never passes"
        )
        print(
            f"{'closed' if row['framing_band_is_closed'] else ' OPEN '} "
            f"{'contig' if row['framing_band_is_contiguous'] else ' HOLE ':<6} "
            f"framing {shape:>12}  "
            f"below={','.join(row['wall_below_is']) or '-':<10} "
            f"above={','.join(row['wall_above_is']) or '-':<10} "
            f"both={','.join(f'{f:.2f}' for f in row['zooms_passing_both_gates']) or '-':<6} "
            f"{row['path']}"
        )

    print(
        f"\nFraming bands closed: {report['framing_bands_closed']}. "
        f"With a hole: {len(report['framing_bands_with_a_hole'])} "
        f"({', '.join(report['framing_bands_with_a_hole']) or 'none'}). "
        f"The scale label survived only at zoom "
        f"{report['scale_label_survived_only_at']}, so the pair admits the "
        f"identity transform and nothing else."
    )
    print(
        f"Fill rises monotonically with zoom in every frame: "
        f"{report['fill_rises_monotonically_in_every_frame']}."
    )
    # A measurement harness, not a gate: it reports the shape it finds, and the
    # assertions about that shape live in the test module. Exits nonzero only if
    # the property the gate pair depends on stopped holding -- that the label
    # check admits the identity transform and nothing else.
    if report["scale_label_survived_only_at"] != [1.0]:
        print(
            "\nUNEXPECTED: the scale label survived a zoom other than the "
            "identity, so the label check is not closing the framing hole."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
