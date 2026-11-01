#!/usr/bin/env python3
"""Measure how much the mask argument could have mattered to the statistic that gates.

An evening went into where the scale-bar annotation ends, and the figure that was disputed --
the bounding-box extent -- is a *diagnostic* the gate records and does not act on. The gate
acts on the interquartile spread of subject pixel positions. A reviewer asked the obvious
question neither of us had: how sensitive is each statistic to the mask? Quartiles ignore a
corner cluster almost by construction, so the answer should be "barely", and if it is, the
dispute was never about the gate.

It is also the strongest available justification for choosing the spread over the extent, and
it was recorded only as an absolute movement rather than as the ratio that makes the case.

Measuring it turned up something sharper. The extent plateau -- the control that says the
answer stops changing once the exclusion box is wide enough -- goes flat at 160px and stays
flat to 440px, and that flatness is **necessary but not sufficient** for "this box contains
only annotation". The extent stops responding once the annotation no longer pins the bounding
box, and then keeps not responding while a wider box eats interior subject pixels that were
never on the boundary. The matched-pair diff does notice: it is exactly zero through 260px and
nonzero from 320px, because two reconstructions of one view differ wherever real geometry sits.

So the two controls disagree about the wide end, and the pair diff is right. One figure in the
capability record was measured at 440px, inside the region the pair diff proves is
contaminated, and is corrected here.

All boxes are anchored at OVERLAY_ORIGIN, the top-left of the annotation, because anchoring
convention has already caused two disagreements here that were not about the data.

Writes docs/evidence/open3d/statistic-mask-sensitivity.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
REPO = EVIDENCE.parents[2]
OUT = EVIDENCE / "statistic-mask-sensitivity.json"

# How far to grow the exclusion box. The interesting region is the one where the extent has
# already gone flat but the pair diff has not yet fired.
WIDTHS = (81, 120, 160, 200, 230, 260, 290, 320, 380, 440)

# One height for both the certification and the measurement. Mutation testing found that a
# certification run on a 1px sliver still produces the same 290/320 boundary, so the two have to
# be tied together. The reported heights below are a self-declaration and are not what makes
# that safe -- a mutation that shrank the band left the declaration intact and survived. The
# test infers the height from the contamination counts, which scale with it.
BAND_HEIGHT = 120

# The spread floor the gate enforces, and the two figures the margin analysis produced.
SPREAD_FLOOR = 0.06
WEAKEST_DELIVERED_SPREAD = 0.1062
LARGEST_OBSERVED_MOVEMENT = 1.50


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


controls = _load("overlay_isolation_controls")
FRAMES = controls.FRAMES
ORIGIN = controls.OVERLAY_ORIGIN


def spread(mask: np.ndarray) -> float:
    """The gate's statistic: interquartile spread of content positions, narrower axis.

    Deliberately recomputed here from the mask rather than imported from the gate, so that
    this measurement compares two statistics over one set of pixels instead of comparing two
    pipelines that each build their own.
    """
    rows, cols = np.nonzero(mask)
    height, width = mask.shape
    return float(
        min(
            (np.percentile(cols, 75) - np.percentile(cols, 25)) / width,
            (np.percentile(rows, 75) - np.percentile(rows, 25)) / height,
        )
    )


def overlay_only_width() -> dict[str, object]:
    """The widest box the pair diff certifies as containing no subject geometry."""
    readings = []
    for width in WIDTHS:
        inside = [
            row["pixels_differing_inside_the_band"]
            for row in controls.pair_diff(width, BAND_HEIGHT)
        ]
        readings.append(
            {
                "exclusion_width": width,
                "max_pixels_differing_inside_the_band": max(inside),
                "certified_overlay_only": max(inside) == 0,
            }
        )
    certified = [r["exclusion_width"] for r in readings if r["certified_overlay_only"]]
    first_contaminated = next(
        (r["exclusion_width"] for r in readings if not r["certified_overlay_only"]),
        None,
    )
    return {
        "certified_at_band_height": BAND_HEIGHT,
        "readings": readings,
        "widest_certified_overlay_only": max(certified) if certified else None,
        "first_width_proven_to_contain_subject_geometry": first_contaminated,
    }


def sensitivity(width: int, height: int = BAND_HEIGHT) -> dict[str, object]:
    """Both statistics, with and without the box, over the same pixels."""
    row, col = ORIGIN
    per_frame = []
    for path in sorted(FRAMES.glob("*.png")):
        mask = controls.content_mask(controls.rgb(path))
        cut = mask.copy()
        cut[row : row + height, col : col + width] = False
        extent_before, extent_after = (
            controls.narrower_extent(mask),
            controls.narrower_extent(cut),
        )
        spread_before, spread_after = spread(mask), spread(cut)
        per_frame.append(
            {
                "frame": path.name,
                "extent_before": round(extent_before, 4),
                "extent_after": round(extent_after, 4),
                "extent_relative_change": round(
                    abs(extent_before - extent_after) / extent_before, 4
                ),
                "spread_before": round(spread_before, 4),
                "spread_after": round(spread_after, 4),
                "spread_absolute_change": round(abs(spread_before - spread_after), 4),
                "spread_relative_change": round(
                    abs(spread_before - spread_after) / spread_before, 4
                ),
            }
        )
    worst_extent = max(float(f["extent_relative_change"]) for f in per_frame)
    worst_spread = max(float(f["spread_relative_change"]) for f in per_frame)
    return {
        "exclusion_width": width,
        "measured_at_band_height": height,
        "per_frame": per_frame,
        "worst_extent_relative_change": round(worst_extent, 4),
        "worst_spread_relative_change": round(worst_spread, 4),
        "worst_spread_absolute_change": round(
            max(float(f["spread_absolute_change"]) for f in per_frame), 4
        ),
        "how_many_times_more_sensitive_the_extent_is": round(
            worst_extent / worst_spread, 1
        ),
    }


def per_frame_ratio_range(width: int) -> dict[str, object]:
    """Where the headline ratio sits among the nine, so it cannot be a flattering pick.

    The published figure is the worst extent change over the worst spread change. Both happen
    to come from the same frame, so it is a within-frame worst case rather than a
    worst-numerator-over-best-denominator construction. A reviewer noted that and called it the
    smallest per-frame ratio of the nine; it is the second smallest. One frame's spread does not
    move at all under the mask, which has no finite ratio and is reported separately rather than
    dropped or counted as a large number.
    """
    rows = sensitivity(width)["per_frame"]
    assert isinstance(rows, list)
    finite, unmoved = [], []
    for row in rows:
        extent, spread_change = (
            float(row["extent_relative_change"]),
            float(row["spread_relative_change"]),
        )
        (finite if spread_change else unmoved).append(
            {"frame": row["frame"], "ratio": round(extent / spread_change, 1)}
            if spread_change
            else {"frame": row["frame"], "ratio": None}
        )
    ratios = [float(r["ratio"]) for r in finite]
    return {
        "smallest_per_frame_ratio": min(ratios),
        "largest_per_frame_ratio": max(ratios),
        "frames_whose_spread_does_not_move_at_all": [r["frame"] for r in unmoved],
        "per_frame": finite + unmoved,
        "where_the_headline_figure_sits": "The published 26.7x is the second smallest of the "
        "nine, not the smallest -- that is 24.5x. Either way it is near the bottom of a 24.5x "
        "to 66.2x range, so the conclusion does not depend on a flattering frame.",
    }


def masked_and_unmasked_margin(width: int) -> dict[str, object]:
    """The floor margin depends slightly on whether the annotation is masked. Both are stated.

    1.77x is the unmasked minimum and 1.74x the masked one. A reviewer measured both and was
    careful to say the gap is the mask rather than a disagreement between us. Recording only one
    would invite someone to find the other and read it as a contradiction.
    """
    rows = sensitivity(width)["per_frame"]
    assert isinstance(rows, list)
    unmasked = min(float(r["spread_before"]) for r in rows)
    masked = min(float(r["spread_after"]) for r in rows)
    return {
        "weakest_unmasked_spread": round(unmasked, 4),
        "weakest_unmasked_over_floor": round(unmasked / SPREAD_FLOOR, 2),
        "weakest_masked_spread": round(masked, 4),
        "weakest_masked_over_floor": round(masked / SPREAD_FLOOR, 2),
        "the_difference_is_the_mask_not_a_disagreement": "The gate reads the masked frame, so "
        "1.74x is the operative margin and 1.77x is the unmasked view of the same thing.",
    }


def axis_convention_sensitivity(
    width: int, height: int = BAND_HEIGHT
) -> dict[str, object]:
    """Which per-frame ratios depend on the axis convention, so a disagreement is diagnosable.

    A reviewer recomputed the nine frames and matched at the top of the range while differing by
    almost exactly 2x on one frame, and proposed a definition difference in the spread term. It
    is testable, so here it is tested under four conventions: the narrower axis this gate uses,
    horizontal only, vertical only, and the mean.

    The result splits their two disagreements apart. Which frame has no finite ratio *is* an
    axis question -- under the narrower axis it is `baseline_scan_and_surface__overview`, and
    `candidate_surface_only__overview` is the undefined one only under a vertical-only spread,
    which is the frame they named. But the 2x gap is not, because that frame reads the same
    under all four conventions. A definition difference cannot produce it.

    This exists because anchoring convention already caused two disagreements about these frames
    that were not about the data. Reporting convention-invariance turns the third into a lookup.
    """
    row, col = ORIGIN
    conventions = {
        "narrower_axis_as_the_gate_uses": lambda h, v: min(h, v),
        "horizontal_only": lambda h, _: h,
        "vertical_only": lambda _, v: v,
        "mean_of_both_axes": lambda h, v: (h + v) / 2,
    }
    per_frame = []
    for path in sorted(FRAMES.glob("*.png")):
        mask = controls.content_mask(controls.rgb(path))
        cut = mask.copy()
        cut[row : row + height, col : col + width] = False
        before, after = controls.narrower_extent(mask), controls.narrower_extent(cut)
        extent_change = abs(before - after) / before
        ratios: dict[str, float | None] = {}
        for name, pick in conventions.items():
            axes = []
            for current in (mask, cut):
                rows_, cols_ = np.nonzero(current)
                frame_height, frame_width = current.shape
                axes.append(
                    pick(
                        float(
                            (np.percentile(cols_, 75) - np.percentile(cols_, 25))
                            / frame_width
                        ),
                        float(
                            (np.percentile(rows_, 75) - np.percentile(rows_, 25))
                            / frame_height
                        ),
                    )
                )
            moved = abs(axes[0] - axes[1]) / axes[0] if axes[0] else 0.0
            ratios[name] = round(extent_change / moved, 1) if moved else None
        finite = [v for v in ratios.values() if v is not None]
        per_frame.append(
            {
                "frame": path.name,
                "ratio_by_convention": ratios,
                "convention_invariant": len(finite) == len(conventions)
                and max(finite) - min(finite) < 1.0,
            }
        )
    return {
        "per_frame": per_frame,
        "frames_whose_ratio_does_not_depend_on_the_convention": [
            f["frame"] for f in per_frame if f["convention_invariant"]
        ],
        "what_this_settles": "A reviewer's 48.5x for observed_scan_only__overview against this "
        "harness's 24.6x cannot be a spread-definition difference: that frame is "
        "convention-invariant, reading the same under all four. Their undefined frame does "
        "resolve as an axis question -- it is the undefined one under a vertical-only spread, "
        "where this gate uses the narrower axis.",
        "the_likeliest_remaining_explanation": "Row attribution rather than measurement. Their "
        "48.5x is within 0.2 of this harness's 48.3x for candidate_surface_only__overview, the "
        "frame immediately above observed_scan_only__overview in sorted order, and the frame "
        "they report as having no finite ratio is the one this harness reads at 48.3x while its "
        "own undefined frame is baseline_scan_and_surface__overview. Two labels moved by one row "
        "would produce exactly both disagreements. Offered as the hypothesis the numbers "
        "support, not as a verdict on their harness, which only they can run.",
        "what_it_does_not_settle": "Whether that hypothesis is right. It is not the axis "
        "convention and not the anchor, both of which are now tested, so the remaining "
        "candidates are row attribution, the content threshold, or the frame bytes.",
    }


def why_the_old_figure_was_wrong() -> dict[str, object]:
    """The record said 0.0042. That was measured where the box is not overlay-only."""
    certified = overlay_only_width()["widest_certified_overlay_only"]
    assert isinstance(certified, int)
    return {
        "figure_the_record_carried": 0.0042,
        "measured_at_exclusion_width": 440,
        "corrected_figure": sensitivity(certified)["worst_spread_absolute_change"],
        "measured_at_certified_width": certified,
        "why_the_old_one_was_too_high": "At 440px the box provably contains subject geometry -- "
        "the matched pairs differ by thousands of pixels inside it -- so the movement it "
        "produced was the spread responding to removed subject content, not to the annotation. "
        "Roughly double the true figure.",
        "how_it_got_past_me": "I took the extent plateau as licence to use the widest box in "
        "the sweep, because the extent was flat there. Flat extent means the annotation no "
        "longer pins the bounding box; it does not mean the box is empty of subject. The "
        "control I needed was the one I already had.",
    }


def fragility_from_observed_movement() -> dict[str, object]:
    """A statement about the floor's edge that needs no new threshold."""
    margin = WEAKEST_DELIVERED_SPREAD / SPREAD_FLOOR
    return {
        "weakest_delivered_frame_over_floor": round(margin, 2),
        "largest_movement_observed_between_matched_reconstructions": LARGEST_OBSERVED_MOVEMENT,
        "headroom_beyond_already_observed_movement": round(
            margin / LARGEST_OBSERVED_MOVEMENT, 2
        ),
        "stated_plainly": "A reconstruction that moved as much again as the worst movement "
        "already measured would put the weakest delivered frame within about 18% of the floor.",
        "why_this_is_not_an_argument_for_retuning": "It is derived from measured movement, not "
        "from the nine frames the threshold was chosen on, so it argues for validating the edge "
        "against frames outside these nine rather than for moving the floor on them.",
    }


def build() -> dict[str, object]:
    certified = overlay_only_width()
    width = certified["widest_certified_overlay_only"]
    assert isinstance(width, int)
    return {
        "the_question": "How much could the mask dispute have changed the number the gate acts "
        "on?",
        "the_answer": "Almost nothing, and that is the case for the statistic rather than a "
        "footnote to it.",
        "which_statistic_gates": {
            "gated": "interquartile spread of subject pixel positions, narrower axis",
            "recorded_but_not_gated": "bounding-box extent",
            "why_the_dispute_was_about_the_wrong_one": "Quartiles are insensitive to a compact "
            "cluster at one corner, which is exactly what a fixed annotation is. The extent is "
            "defined by the outermost pixel, so a single annotation glyph can set it.",
        },
        "sensitivity_at_the_certified_overlay_only_width": sensitivity(width),
        "which_box_is_actually_overlay_only": certified,
        "the_plateau_is_necessary_but_not_sufficient": {
            "what_the_extent_plateau_shows": "That the annotation stops pinning the bounding box "
            "at 160px of exclusion and the extent reading stops changing.",
            "what_it_does_not_show": "That the box contains only annotation. The extent stays "
            "flat while a wider box removes interior subject pixels that were never on the "
            "boundary.",
            "what_the_pair_diff_shows_instead": "Exactly zero differing pixels through 260px and "
            "thousands from 320px, because two reconstructions of one view differ wherever real "
            "geometry sits. The two controls disagree above 260px and the pair diff is right.",
            "consequence": "The pair diff is the load-bearing control. The plateau is corroboration.",
        },
        "a_figure_this_corrects": why_the_old_figure_was_wrong(),
        "where_the_headline_ratio_sits_among_the_nine": per_frame_ratio_range(width),
        "the_margin_masked_and_unmasked": masked_and_unmasked_margin(width),
        "whether_the_ratios_depend_on_the_axis_convention": axis_convention_sensitivity(
            width
        ),
        "the_floors_fragility_without_proposing_a_new_floor": fragility_from_observed_movement(),
        "what_this_does_not_establish": "That the floor's edge is correct. It establishes that "
        "the mask question could not have moved the gate, and that the statistic choice was the "
        "right one for a reason that is now measured rather than asserted.",
        "provenance": {
            "frames": str(FRAMES.relative_to(REPO)),
            "anchored_at_overlay_origin_row_col": list(ORIGIN),
            "why_the_anchor_is_stated": "Two disagreements about these frames were anchoring "
            "conventions rather than data.",
        },
    }


def main() -> int:
    report = build()
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    s = report["sensitivity_at_the_certified_overlay_only_width"]
    certified = report["which_box_is_actually_overlay_only"]
    assert isinstance(s, dict) and isinstance(certified, dict)
    print(
        f"Removing the annotation moves the extent by up to "
        f"{float(s['worst_extent_relative_change']) * 100:.1f}% and the gated spread by "
        f"{float(s['worst_spread_relative_change']) * 100:.2f}% "
        f"({s['how_many_times_more_sensitive_the_extent_is']}x)."
    )
    print(
        f"Overlay-only is certified to "
        f"{certified['widest_certified_overlay_only']}px; subject geometry is proven present "
        f"from {certified['first_width_proven_to_contain_subject_geometry']}px."
    )
    print(f"Wrote {OUT.relative_to(REPO)}")
    # The whole argument rests on the gated statistic being the insensitive one. If that ever
    # inverts, this report should not be quietly published alongside the old conclusion.
    return (
        0
        if float(s["worst_spread_relative_change"])
        < float(s["worst_extent_relative_change"])
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
