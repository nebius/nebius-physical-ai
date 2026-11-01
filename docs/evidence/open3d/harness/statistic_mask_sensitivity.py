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
never on the boundary. The matched-pair diff does notice: it is exactly zero through 290px and
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

# Background tolerances to sweep. The published condition is overlay_isolation_controls'
# CONTENT_TOLERANCE; the rest exist because a reviewer's unstated threshold turned out to be the
# whole of a four-exchange disagreement about one diagnostic ratio.
TOLERANCES = tuple(range(0, 32, 2))

#: The frame the headline ratio is quoted from, which is not the frame the mask dispute
#: happened on. Named here rather than derived, because the headline figure itself only
#: exists in prose -- and bound to the sweep in the output, where
#: `ratio_at_the_published_tolerance` has to still read as the headline number, so this
#: cannot quietly come to point at a different frame.
HEADLINE_FRAME = "candidate_scan_and_surface__topdown.png"

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
    axis = {
        "horizontal": lambda h, v: h,
        "vertical": lambda h, v: v,
        "narrower": lambda h, v: min(h, v),
    }

    def quartile_axes(current):
        rows_, cols_ = np.nonzero(current)
        frame_height, frame_width = current.shape
        return (
            float((np.percentile(cols_, 75) - np.percentile(cols_, 25)) / frame_width),
            float((np.percentile(rows_, 75) - np.percentile(rows_, 25)) / frame_height),
        )

    def extent_axes(current):
        rows_, cols_ = np.nonzero(current)
        frame_height, frame_width = current.shape
        return (
            float((cols_.max() - cols_.min() + 1) / frame_width),
            float((rows_.max() - rows_.min() + 1) / frame_height),
        )

    per_frame = []
    for path in sorted(FRAMES.glob("*.png")):
        mask = controls.content_mask(controls.rgb(path))
        cut = mask.copy()
        cut[row : row + height, col : col + width] = False
        spread_before, spread_after = quartile_axes(mask), quartile_axes(cut)
        extent_before, extent_after = extent_axes(mask), extent_axes(cut)
        ratios: dict[str, float | None] = {}
        for numerator_axis, take_extent in axis.items():
            moved_extent = abs(
                take_extent(*extent_before) - take_extent(*extent_after)
            ) / take_extent(*extent_before)
            for denominator_axis, take_spread in axis.items():
                base = take_spread(*spread_before)
                moved_spread = (
                    abs(base - take_spread(*spread_after)) / base if base else 0.0
                )
                ratios[f"extent_{numerator_axis}_over_spread_{denominator_axis}"] = (
                    round(moved_extent / moved_spread, 1) if moved_spread else None
                )
        per_frame.append({"frame": path.name, "ratio_by_axis_pairing": ratios})

    return {
        "per_frame": per_frame,
        "how_to_quote_a_ratio_about_these_frames": "Name the numerator axis and the denominator "
        "axis. Naming only one is what let this run for three exchanges: a reviewer pinned the "
        "extent to the horizontal axis and varied only the spread axis, while this harness moved "
        "both together, so each of us was invariant within our own family and neither family was "
        "stated. The operative figure is the one the gate reads, "
        "extent_narrower_over_spread_narrower. The others are different quantities, not "
        "competing measurements of one.",
        "a_claim_of_mine_that_was_scoped_too_broadly": "I reported a frame as "
        "'convention-invariant' on the strength of four conventions that all moved extent and "
        "spread together. It is invariant within that family and not across the reviewer's, "
        "where the numerator is pinned. The label claimed more than the control tested, which is "
        "the same shape of error as a report that declares its own parameters.",
        "a_hypothesis_of_mine_that_was_refuted": "I suggested their figure came from a row "
        "misattribution, because it matched this harness's reading of the adjacent frame to "
        "within 0.2. They hashed all nine frames against the ones in this PR and found 9 of 9 "
        "byte-identical under the same names, which kills it. The coincidence was real and the "
        "inference from it was not.",
        "what_this_does_not_claim": "That one pairing is correct and another wrong. Both are "
        "well-defined quantities over the same pixels. The gate reads one of them, and that is "
        "the only reason this harness privileges it.",
        "the_pairing_matrix_isolates_the_remaining_difference_to_one_term": {
            "what_agrees": "The reviewer's extent-horizontal over spread-vertical column "
            "reproduces here exactly on both frames they tabulated, 24.6x and 18.9x.",
            "what_does_not": "Their extent-horizontal over spread-horizontal column does not: "
            "they read 48.5x and undefined where this harness reads 24.6x and 48.3x.",
            "what_follows": "Since the numerator is shared between their two columns and the "
            "vertical spread agrees, neither the extent term nor the vertical spread can be the "
            "difference. It is the horizontal quartile spread alone. On one frame this harness's "
            "horizontal spread moves 1.97x what theirs does; on the other theirs does not move "
            "at all while this one does.",
            "why_this_is_the_useful_form": "Three exchanges produced two refuted hypotheses, "
            "mine about row attribution and theirs about the axis definition. A pairing matrix "
            "narrows it to a single term without either of us guessing, and it is a term either "
            "harness can print.",
        },
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


def tolerance_hinge(width: int, height: int = BAND_HEIGHT) -> dict[str, object]:
    """Why the disputed ratio could sustain four exchanges: its denominator is one pixel wide.

    A reviewer swept the background tolerance on their own copy and found the disputed frame
    oscillating between roughly 24.6 and 48.8 as the threshold moved, which reproduces both of
    the values they had published. Their diagnosis was a one-pixel hinge: the denominator is the
    *relative movement* of an interquartile spread, that movement is one to two pixels in a
    126-pixel range, and one pixel crossing the threshold halves the denominator and doubles the
    ratio.

    It reproduces here in all five bands. It is also broader than the frame it was found on --
    five of the nine frames swing by 2x to 3x across the sweep, and two of them pass through
    undefined -- so this is a property of the quantity, not an anomaly of one frame. That makes a
    stability note too weak a response: a ratio with a one-pixel denominator cannot honestly be
    quoted as a bare number, so each one is published here with the interval it occupies.

    The gate is a separate question, and asserting it is unaffected is not the same as measuring
    it. The sweep therefore carries the gated statistic too.
    """
    row, col = ORIGIN
    frames = {path.name: controls.rgb(path) for path in sorted(FRAMES.glob("*.png"))}

    def at(frame: np.ndarray, tolerance: int) -> dict[str, object]:
        background = np.median(frame.reshape(-1, 3), axis=0)
        mask = (np.abs(frame - background) > tolerance).any(axis=2)
        cut = mask.copy()
        cut[row : row + height, col : col + width] = False
        rows_, cols_ = np.nonzero(mask)
        kept_rows, kept_cols = np.nonzero(cut)
        frame_height, frame_width = mask.shape

        def narrower_spread(r: np.ndarray, c: np.ndarray) -> float:
            return min(
                (np.percentile(c, 75) - np.percentile(c, 25)) / frame_width,
                (np.percentile(r, 75) - np.percentile(r, 25)) / frame_height,
            )

        def narrower_extent(r: np.ndarray, c: np.ndarray) -> float:
            return min(
                (c.max() - c.min() + 1) / frame_width,
                (r.max() - r.min() + 1) / frame_height,
            )

        before = narrower_spread(rows_, cols_)
        after = narrower_spread(kept_rows, kept_cols)
        moved_spread = abs(before - after) / before if before else 0.0
        moved_extent = abs(
            narrower_extent(rows_, cols_) - narrower_extent(kept_rows, kept_cols)
        ) / narrower_extent(rows_, cols_)
        return {
            "ratio": round(moved_extent / moved_spread, 2) if moved_spread else None,
            "gated_spread": round(after, 4),
            "horizontal_quartile_pixels_moved": float(
                round(
                    (np.percentile(cols_, 75) - np.percentile(cols_, 25))
                    - (np.percentile(kept_cols, 75) - np.percentile(kept_cols, 25)),
                    1,
                )
            ),
        }

    sweep = {
        name: {tolerance: at(frame, tolerance) for tolerance in TOLERANCES}
        for name, frame in frames.items()
    }
    published = controls.CONTENT_TOLERANCE

    per_frame = []
    for name, readings in sorted(sweep.items()):
        finite = [r["ratio"] for r in readings.values() if r["ratio"] is not None]
        undefined_at = [t for t, r in readings.items() if r["ratio"] is None]
        per_frame.append(
            {
                "frame": name,
                "ratio_at_the_published_tolerance": readings[published]["ratio"],
                "ratio_lowest_over_the_sweep": min(finite),
                "ratio_highest_over_the_sweep": max(finite),
                "swing": round(max(finite) / min(finite), 2),
                "undefined_at_tolerances": undefined_at,
            }
        )
    knife_edge = [entry for entry in per_frame if entry["swing"] >= 2.0]

    gate_verdict_changes = [
        name
        for name, readings in sweep.items()
        if not all(r["gated_spread"] >= SPREAD_FLOOR for r in readings.values())
    ]
    worst_gated = min(
        r["gated_spread"] for readings in sweep.values() for r in readings.values()
    )

    def bands_of(readings: dict[int, dict[str, object]]) -> list[dict[str, object]]:
        """Contiguous runs of one ratio across the sweep, which is where a hinge shows."""

        out: list[dict[str, object]] = []
        for tolerance in TOLERANCES:
            reading = readings[tolerance]
            if out and out[-1]["ratio"] == reading["ratio"]:
                tolerances = out[-1]["tolerances"]
                assert isinstance(tolerances, list)
                tolerances.append(tolerance)
                continue
            out.append(
                {
                    "tolerances": [tolerance],
                    "ratio": reading["ratio"],
                    "horizontal_quartile_pixels_moved": reading[
                        "horizontal_quartile_pixels_moved"
                    ],
                }
            )
        return out

    def plateau_of(readings: dict[int, dict[str, object]]) -> dict[str, object]:
        """How far the published reading holds on either side of the published condition.

        `stable_through` walks up only while every intermediate tolerance still agrees,
        so a value that recurs after an excursion does not extend the run.
        """
        here = readings[published]["ratio"]
        return {
            "ratio_at_the_published_tolerance": here,
            "stable_from": min(t for t in TOLERANCES if readings[t]["ratio"] == here),
            "stable_through": max(
                t
                for t in TOLERANCES
                if all(
                    readings[u]["ratio"] == here
                    for u in TOLERANCES
                    if published <= u <= t
                )
            ),
            "tolerances_in_the_run": sum(
                1 for t in TOLERANCES if readings[t]["ratio"] == here
            ),
            "of": len(TOLERANCES),
        }

    disputed = sweep["observed_scan_only__overview.png"]
    bands = bands_of(disputed)

    return {
        "the_mechanism": "The denominator is the relative movement of an interquartile spread "
        "under the overlay exclusion. On the disputed frame that movement is two pixels out of "
        "126 at the published tolerance and one pixel at a looser one. Halving a denominator "
        "that small doubles the ratio, so the quantity has a one-pixel hinge in it.",
        "the_published_tolerance": published,
        "the_disputed_frame_band_by_band": bands,
        "the_plateau_around_the_published_condition": {
            "stable_from": min(
                t
                for t in TOLERANCES
                if disputed[t]["ratio"] == disputed[published]["ratio"]
            ),
            "stable_through": max(
                t
                for t in TOLERANCES
                if all(
                    disputed[u]["ratio"] == disputed[published]["ratio"]
                    for u in TOLERANCES
                    if published <= u <= t
                )
            ),
            "why_it_matters": "The published reading is not balanced on the hinge; it sits on a "
            "plateau that runs from 0 through 8. That is why it is reproducible, and it is not an "
            "argument that the quantity is robust.",
        },
        "but_that_is_not_the_frame_the_headline_comes_from": {
            "raised_by": "review, P4",
            "the_asymmetry": (
                "The band table above is for observed_scan_only__overview.png, which is "
                "where the dispute happened. The headline ratio comes from "
                f"{HEADLINE_FRAME}, which is the widest-swinging of the nine. The "
                "argument for banding a frame is that an interval alone does not say "
                "whether a reading is reproducible, and by that argument the frame whose "
                "number travels is the one that most needs it."
            ),
            "the_headline_frame_band_by_band": bands_of(sweep[HEADLINE_FRAME]),
            "the_plateau_around_the_published_condition": plateau_of(
                sweep[HEADLINE_FRAME]
            ),
            "the_disputed_frame_for_comparison": plateau_of(disputed),
            "the_answer": (
                "It sits on a plateau, and a longer one than the disputed frame's: the "
                "run containing the published tolerance covers 11 of the 16 swept "
                "tolerances against the disputed frame's 6. So the reading the headline "
                "quotes is not balanced on the hinge."
            ),
            "swing_and_local_stability_are_different_properties": (
                "This is the part worth keeping. The concern was that the "
                "widest-swinging frame might be the fragile one, and swing is the ratio "
                "of the extremes across the whole sweep, which says nothing about how "
                "far the published condition sits from a transition. Here the frame with "
                "the widest swing has the longer plateau. Both frames are two steps from "
                "their nearest transition, so neither is more locally fragile than the "
                "other, and the frame that travels furthest across the sweep is not the "
                "one whose published reading is least reproducible. An interval still "
                "does not establish reproducibility -- that was the original point -- but "
                "neither does a wide interval refute it."
            ),
            "where_the_published_condition_sits_in_each_run": {
                HEADLINE_FRAME: (
                    "published tolerance 6, run 4 through 24: two steps above the lower "
                    "edge and eighteen below the upper"
                ),
                "observed_scan_only__overview.png": (
                    "published tolerance 6, run 0 through 8: the lower edge is the floor "
                    "of the sweep, and the upper is two steps away"
                ),
            },
        },
        "this_is_broader_than_the_frame_it_was_found_on": {
            "frames_swinging_2x_or_more": len(knife_edge),
            "of": len(per_frame),
            "widest_swing": max(entry["swing"] for entry in per_frame),
            "frames_passing_through_undefined": sorted(
                entry["frame"]
                for entry in per_frame
                if entry["undefined_at_tolerances"]
            ),
            "what_that_changes": "The reviewer found the hinge on one frame and asked for a "
            "stability note. The sweep makes it a property of the ratio rather than a quirk of "
            "that frame, which is worse than the note would have implied, so every entry is "
            "published with its interval instead.",
        },
        "per_frame_interval": per_frame,
        "the_gate_is_unaffected_and_this_is_the_measurement_not_the_assertion": {
            "frames_whose_verdict_changes_anywhere_in_the_sweep": gate_verdict_changes,
            "worst_delivered_spread_over_the_whole_sweep": round(worst_gated, 4),
            "worst_as_multiple_of_floor": round(worst_gated / SPREAD_FLOOR, 2),
            "gated_spread_by_frame_and_tolerance": {
                name: {
                    str(tolerance): readings[tolerance]["gated_spread"]
                    for tolerance in TOLERANCES
                }
                for name, readings in sorted(sweep.items())
            },
            "why_the_table_is_here_and_not_just_the_summary": "Mutation testing on this very "
            "section found that inflating the summary margin threefold changed the report and "
            "broke no test, because the test was reading the claim rather than the readings. "
            "That is the same defect a reviewer found in another gate of mine, where the report "
            "asserted its own validity. The table lets the test recompute both figures.",
            "why_the_gated_statistic_has_no_hinge": "The gate reads the spread itself, not the "
            "relative movement of the spread. An absolute quartile position shifts by a pixel "
            "when a pixel crosses the threshold; a ratio whose denominator is that shift moves "
            "by a factor of two. The fragile quantity is the diagnostic, and it is fragile "
            "because it is a ratio of two small differences.",
        },
        "the_dispute_is_now_explained_rather_than_adjudicated": {
            "what_the_reviewer_published": "48.5 for the disputed frame and undefined for "
            "candidate_surface_only__overview.",
            "what_reproduces_here": "At tolerances 24 through 30 this harness reads 48.38 on the "
            "disputed frame, and candidate_surface_only__overview is undefined from 22 upward. "
            "Their pair is reproducible on these frames at a looser threshold than the one this "
            "harness publishes.",
            "so_the_cause_was": "An unstated background tolerance acting on a quantity with a "
            "one-pixel hinge. Not a misattributed row, not an axis convention, and not the "
            "pairing asymmetry that replaced it.",
            "who_found_it": "The reviewer, by sweeping their own harness after their first two "
            "explanations failed, and against their own published figure. Confirmed here "
            "independently on these frames; all five bands match their table.",
            "what_it_cost": "Four exchanges, three dead hypotheses -- one of mine about rows, "
            "two of theirs about axes and pairing -- for a quantity neither of us had swept.",
        },
        "what_this_does_not_claim": "That the diagnostic should be removed or that its value at "
        "the published tolerance is wrong. It is well defined at a stated condition and it is "
        "reproducible there. The claim is narrower: it must be quoted with that condition and "
        "with its interval, because a reader recomputing it at a different threshold will "
        "otherwise conclude one of the two of us made an arithmetic error.",
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
            "what_the_pair_diff_shows_instead": "Exactly zero differing pixels through 290px and "
            "360 at 320px rising to thousands by 440px, because two reconstructions of one view "
            "differ wherever real geometry sits. The two controls disagree above 290px and the "
            "pair diff is right. An earlier version of this sentence said 260px, which "
            "understated the certified range by one step of the sweep while the table beside it "
            "and widest_certified_overlay_only both read 290.",
            "consequence": "The pair diff is the load-bearing control. The plateau is corroboration.",
        },
        "a_figure_this_corrects": why_the_old_figure_was_wrong(),
        "where_the_headline_ratio_sits_among_the_nine": per_frame_ratio_range(width),
        "the_margin_masked_and_unmasked": masked_and_unmasked_margin(width),
        "whether_the_ratios_depend_on_the_axis_convention": axis_convention_sensitivity(
            width
        ),
        "why_that_ratio_could_sustain_four_exchanges": tolerance_hinge(width),
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
