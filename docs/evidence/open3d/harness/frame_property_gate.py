"""Fail the build when a delivered frame is clipped, strands its subject, or washes it out.

The measurements mostly existed. ``frame_edge_audit.py`` has read these nine PNGs
since the pack was built, and an independent reviewer recomputed every number
from the committed bytes and agreed to ``0.0e+00``. What was missing is that
nothing executed any of it: the repository's Open3D tests pin the camera fitting
*math*, so a frame re-rendered by hand, captured in a different window, or
cropped afterwards could ship with any framing at all and no test would read the
pixels.

The fixed annotation that breaks the obvious measure
----------------------------------------------------
Framing wants to be measured as the subject's extent -- its bounding box as a
share of the frame. On these frames that reading is not about the subject.

Every one of the nine carries a scale bar reading *"1.0 m (scan units)"* drawn
at a fixed bottom-left position, and it is content like any other. It pins the
content bounding box, which is why all nine report a left margin of 27px, a
bottom margin of 28px, and a clipping headroom of 1.056 -- **identical across
nine different views of different geometry**, which is the tell that the number
is describing the annotation and not the scan. Remove the annotation block and
the extent span falls from 0.550-0.713 to 0.264-0.510.

The consequence is a false negative, not a cosmetic error. A subject shrunk to a
speck in the middle of the frame still has the annotation anchoring one corner
of the bounding box, so an extent-based floor **passes a frame that is plainly
stranded**. That case is constructed and asserted in
``npa/tests/workbench/test_open3d_frame_properties.py``; it passes an extent
gate at 0.40 and fails this one.

So stranding is gated on an interquartile spread instead -- the middle half of
the content's row and column positions. An annotation that is 1.6-3.6% of
content pixels cannot move a quartile: measured across the nine frames, removing
the block shifts the spread by at most 0.0042 while shifting the extent by up to
0.335, an **80x difference in sensitivity**. Extent is still recorded, because it
is the intuitive reading and a reviewer will want to see it, and because the gap
between the two numbers is itself the evidence for this paragraph.

Why not gate fill
-----------------
The other obvious knob is ``content_fraction``, and gating it would be a
mistake. For a sparse point cloud, fill and framing diverge: these frames span
half their narrower axis while filling 21%, because a point cloud is mostly the
gaps between points. The review's objection is the decisive one: **the cheapest
way to pass a fill gate is to zoom in until the cloud clips**, and losing the
sparse, incomplete geometry at the edges is exactly the defect these views exist
to show. Fill is recorded and not gated.

Why these cannot be gamed together
----------------------------------
``border content`` and ``subject spread`` bracket framing from opposite sides:

* Zoom in and spread rises, but content eventually reaches the border and
  ``border content`` fails.
* Zoom out and the border clears, but the content contracts and ``spread``
  fails.

So the pass band is *closed*. ``gate_zoom_sweep.py`` measures where both walls
stand rather than asserting that they exist -- and it found a hole in exactly
that claim. One frame failed at 1.1x and 1.2x, then **passed again at 1.3x**,
because by then the scale bar had been cropped out of the frame entirely and the
geometry happened to miss the border on that step. A heavily cropped frame that
had thrown away its scale reference scored as well framed.

That is what the scale-label check closes. The label's pixels are hashed and
must match exactly, so no resample or crop survives it, and the pass band is
contiguous on all nine frames. It was worth finding: the gate's own sweep,
reported as contiguity rather than as a bounding interval, is what exposed it.

An honest word about the thresholds
-----------------------------------
These are **regression ratchets, not quality claims**, and they were set with
the current measurements in hand, so nobody should read them as a blind
pre-registration. ``border content`` is the exception and the strongest of them:
zero is not a tuned number. The others sit far from the observed values on
purpose, so they fire on a real defect and not on ordinary re-render variation.
A gate that trips on noise gets deleted, and then it protects nothing.

What makes them mean anything is not the margin, it is that each one is shown to
fire on the defect it names, by construction, in the test module above.

Reads PNG bytes only. No geometry, no Open3D, no network.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame_edge_audit import CONTENT_TOLERANCE, audit

#: Content in a border row or column is geometry, or an annotation, that the frame
#: cut off. Zero is not a tuned threshold -- there is no amount of cut-off subject
#: that is acceptable in a view whose purpose is to show incomplete geometry
#: honestly, and a cut-off scale bar is a broken scale reference.
MAX_BORDER_CONTENT_PIXELS = 0

#: Interquartile spread of content positions, as a share of the narrower axis.
#: Observed across the delivered frames: 0.106-0.191. This is the stranding gate,
#: and it is deliberately not the extent -- see this module's docstring.
MIN_SUBJECT_SPREAD = 0.06

#: Median luminance distance between content and background. Observed: 137-169 of
#: 255. The floor is the operator's original complaint made mechanical: a washed
#: out render is one whose typical content pixel is close to its background.
MIN_MEDIAN_CONTRAST = 60.0

#: Share of content allowed to sit within the audit's low-contrast limit of the
#: background. Observed: 0.0002-0.0784. The distributional half of the same
#: reading -- a frame can hold a healthy median and still hide a third of its
#: content in a washed-out floor, which a median by construction cannot see.
MAX_LOW_CONTRAST_CONTENT_FRACTION = 0.30

#: The scale label's pixel rectangle, and the sha256 of those exact bytes. The
#: rectangle is the largest one around the glyphs that is byte-identical in all
#: nine delivered frames; the bar beneath it is excluded because its length
#: encodes each view's metres-per-pixel and so legitimately differs.
#:
#: This is deliberately an exact check and therefore a brittle one. It will fail
#: if the renderer moves the label, restyles it, changes the font, or changes the
#: wording -- and that is the intent. A frame that has lost its scale reference
#: has lost the only thing that makes its geometry measurable, and a gate that
#: quietly stopped checking would be worse than one that asks a human to look.
#: Re-blessing it is one line, and it should be a conscious act.
LABEL_RECT = (648, 670, 20, 240)
LABEL_SHA256 = "1bc310158d2020172cce6bda0d1214668f1a6b3eadc9ea6fd4d55a4440ed99e3"

#: Recorded and deliberately not gated, with the reasoning in the docstring.
#: Named here so a future reader can see each omission is a decision.
NOT_GATED = (
    "content_fraction",
    "extent_span_narrower_axis",
    "clipping_headroom_zoom",
)


def subject_spread(path: Path) -> dict:
    """Interquartile spread of content positions, which a small annotation cannot move.

    The extent of a point cloud is set by its most distant pixel, so any fixed
    overlay in a corner becomes part of the reading. A quartile is set by the
    bulk, so it measures the subject whether or not the frame is annotated.
    """
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    height, width, _ = rgb.shape

    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]])
    values, counts = np.unique(corners, axis=0, return_counts=True)
    background = values[int(np.argmax(counts))]
    content = np.abs(rgb - background).max(axis=2) > CONTENT_TOLERANCE

    rows, cols = np.where(content)
    if not rows.size:
        return {"horizontal": 0.0, "vertical": 0.0, "narrower_axis": 0.0}

    horizontal = float((np.percentile(cols, 75) - np.percentile(cols, 25)) / width)
    vertical = float((np.percentile(rows, 75) - np.percentile(rows, 25)) / height)
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "narrower_axis": min(horizontal, vertical),
    }


def label_digest(path: Path) -> str:
    """sha256 of the scale label's pixels, which no resample or crop survives."""
    top, bottom, left, right = LABEL_RECT
    patch = np.asarray(Image.open(path).convert("RGB"))[top:bottom, left:right]
    return hashlib.sha256(patch.tobytes()).hexdigest()


def clipping_headroom(row: dict) -> float:
    """How far the frame could zoom in before content reaches a border band.

    Recorded, not gated. On the delivered frames this is pinned by the scale bar
    rather than the scan, which is precisely why it is not a gate.
    """
    width, height = row["resolution"]
    span = row["content_span_fraction"]
    # Reconstructing the half-extents from the span is exact only for centred
    # content, so this is reported as the symmetric-case estimate it is.
    dx = span["horizontal"] * width / 2
    dy = span["vertical"] * height / 2
    if dx <= 0 or dy <= 0:
        return 0.0
    return min((width / 2 - 2) / dx, (height / 2 - 2) / dy)


def failures(row: dict, spread: dict, label: str | None = None) -> list[str]:
    """Return one message per violated property, empty when the frame passes."""
    found = []

    if label is not None and label != LABEL_SHA256:
        found.append(
            f"scale label missing or altered: the pixels at rows "
            f"{LABEL_RECT[0]}-{LABEL_RECT[1]}, cols {LABEL_RECT[2]}-{LABEL_RECT[3]} "
            f"hash to {label[:12]}, not {LABEL_SHA256[:12]}. Either the frame no "
            f"longer carries its scale reference, or the renderer changed the "
            f"label and LABEL_SHA256 needs re-blessing by a human who checked it"
        )

    touching = row["content_pixels_touching_edge"]
    border = sum(touching.values())
    if border > MAX_BORDER_CONTENT_PIXELS:
        edges = ", ".join(sorted(name for name, n in touching.items() if n > 0))
        found.append(
            f"clipped: {border} content pixel(s) in the border band at {edges}; "
            f"the frame cut off something it was drawing"
        )

    if spread["narrower_axis"] < MIN_SUBJECT_SPREAD:
        found.append(
            f"stranded: the middle half of the content spans "
            f"{spread['narrower_axis']:.4f} of the narrower axis, under "
            f"{MIN_SUBJECT_SPREAD}; unclipped but too small to read"
        )

    legibility = row["legibility"]
    median = legibility["median_contrast_vs_background"]
    if median < MIN_MEDIAN_CONTRAST:
        found.append(
            f"washed out: median content contrast {median:.1f} against the "
            f"background, under {MIN_MEDIAN_CONTRAST}"
        )

    low = legibility["low_contrast_content_fraction"]
    if low > MAX_LOW_CONTRAST_CONTENT_FRACTION:
        found.append(
            f"washed out floor: {low:.3f} of content sits within "
            f"{legibility['low_contrast_limit']}/255 of the background, over "
            f"{MAX_LOW_CONTRAST_CONTENT_FRACTION}; drawn and unreadable"
        )

    return found


def check(frames: list[Path]) -> dict:
    rows = []
    for path in sorted(frames):
        row = audit(path)
        spread = subject_spread(path)
        label = label_digest(path)
        found = failures(row, spread, label)
        extent = min(
            row["content_span_fraction"]["horizontal"],
            row["content_span_fraction"]["vertical"],
        )
        rows.append(
            {
                "path": path.name,
                "passed": not found,
                "failures": found,
                "gated": {
                    "border_content_pixels": sum(
                        row["content_pixels_touching_edge"].values()
                    ),
                    "subject_spread_narrower_axis": spread["narrower_axis"],
                    "median_contrast_vs_background": row["legibility"][
                        "median_contrast_vs_background"
                    ],
                    "low_contrast_content_fraction": row["legibility"][
                        "low_contrast_content_fraction"
                    ],
                    # The expected digest is recorded once in ``thresholds``; repeating
                    # it per frame would add nine identical hashes that name no file
                    # and that no guard can bind. The per-frame fact is whether it
                    # matched, and the actual digest is only interesting when it did
                    # not.
                    "scale_label_intact": label == LABEL_SHA256,
                    **({} if label == LABEL_SHA256 else {"scale_label_sha256": label}),
                },
                "recorded_not_gated": {
                    "content_fraction": row["content_fraction"],
                    "extent_span_narrower_axis": extent,
                    "clipping_headroom_zoom": clipping_headroom(row),
                },
            }
        )

    return {
        "what": (
            "Framing and legibility gate over the delivered frames, recomputed from "
            "PNG bytes rather than read back from the record that measured them."
        ),
        "stranding_is_gated_on_spread_not_extent_because": (
            "every frame carries a fixed bottom-left scale bar, which pins the "
            "content bounding box and makes the extent reading partly a "
            "measurement of the annotation. Removing the annotation block shifts "
            "the extent by up to 0.335 and the spread by at most 0.0042."
        ),
        "thresholds": {
            "max_border_content_pixels": MAX_BORDER_CONTENT_PIXELS,
            "min_subject_spread": MIN_SUBJECT_SPREAD,
            "min_median_contrast": MIN_MEDIAN_CONTRAST,
            "max_low_contrast_content_fraction": MAX_LOW_CONTRAST_CONTENT_FRACTION,
            "scale_label_sha256": LABEL_SHA256,
            "scale_label_rect_top_bottom_left_right": list(LABEL_RECT),
            "not_gated": list(NOT_GATED),
        },
        "frames": rows,
        "failed_frames": sorted(r["path"] for r in rows if not r["passed"]),
        "passed": all(r["passed"] for r in rows),
    }


def main() -> int:
    args = sys.argv[1:]
    out = None
    if args and args[0] == "--out":
        out, args = Path(args[1]), args[2:]

    frames = [Path(a) for a in args]
    if not frames:
        print("usage: frame_property_gate.py [--out REPORT.json] FRAME.png ...")
        return 2

    report = check(frames)
    if out is not None:
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for row in report["frames"]:
        gated, free = row["gated"], row["recorded_not_gated"]
        print(
            f"{'  ok  ' if row['passed'] else ' FAIL '} "
            f"edge {gated['border_content_pixels']:>6}  "
            f"spread {gated['subject_spread_narrower_axis']:.4f}  "
            f"contrast {gated['median_contrast_vs_background']:6.1f}  "
            f"washed {gated['low_contrast_content_fraction'] * 100:5.1f}%  "
            f"(extent {free['extent_span_narrower_axis']:.3f}, not gated)  "
            f"{row['path']}"
        )
        for failure in row["failures"]:
            print(f"         {failure}")

    if report["passed"]:
        print(
            f"\n{len(report['frames'])} frame(s) pass. No content in any border "
            f"band, the middle half of every subject spans at least "
            f"{MIN_SUBJECT_SPREAD} of its narrower axis, and none is washed out."
        )
        return 0

    print(
        f"\n{len(report['failed_frames'])} frame(s) failed: "
        + ", ".join(report["failed_frames"])
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
