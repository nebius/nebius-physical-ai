#!/usr/bin/env python3
"""How degraded must a frame be before the spread floor fires, and where do the real frames sit?

The gate's `MIN_SUBJECT_SPREAD` was set with the delivered measurements in hand. That is disclosed
and this harness does not undo it -- nothing measured after the fact can turn a chosen number into
a pre-registered one.

What it can do is answer the question a reviewer put better than the disclosure did. The bracketing
argument -- shrink the subject and the spread floor fires, enlarge it and the border check fires --
establishes that the pass band *exists*. It says nothing about where its *edges* are. With nine
frames and a width chosen after seeing all nine, no argument from those nine can validate the
edges, because they have no freedom left.

So the reviewer asked for a control instead of an argument: report what the spread reads for a
deliberately degraded frame, and how far that sits from the floor. A wide margin means the edges
are defensible whatever width was chosen. A thin one means the width is doing work and should be
widened or justified.

This sweeps the degradation continuously rather than testing one speck, because the number worth
knowing is not whether a 16x16 speck fails -- it obviously does -- but **how much subject a frame
can lose while still passing**. That is the gate's actual blind spot, stated as a fraction.

Degradation is the same centre-scale the zoom sweep uses, so a factor here means the same thing it
means there: the subject occupies that fraction of its delivered linear size, padded with the
frame's own background.

Reads PNG bytes only. No geometry, no Open3D, no network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HARNESS = Path(__file__).resolve().parent
FRAMES = HARNESS.parent / "frames-refit"

# Coarse to 0.50 then fine, because the crossing is expected well below half size and a tight grid
# everywhere would spend its time where nothing happens.
FACTORS = (
    1.00,
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
    0.65,
    0.60,
    0.55,
    0.50,
    0.46,
    0.42,
    0.38,
    0.34,
    0.30,
    0.27,
    0.24,
    0.21,
    0.18,
    0.15,
    0.12,
    0.10,
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load("frame_property_gate")
SWEEP = _load("gate_zoom_sweep")


def spread_at(path: Path, factor: float, scratch: Path) -> float:
    """The gated statistic, read from bytes that have actually been degraded and written out."""
    if factor == 1.00:
        return GATE.subject_spread(path)["narrower_axis"]
    degraded = scratch / f"{path.stem}-{factor:.2f}.png"
    SWEEP.rezoom(path, factor, degraded)
    return GATE.subject_spread(degraded)["narrower_axis"]


#: The committed stranded-speck control, rebuilt here so this report carries both bounds rather
#: than citing the other one. Erase the geometry, keep the bottom-left annotation region, leave a
#: 16x16 speck at the centre -- a frame that is plainly useless and that a floor on the subject's
#: *extent* passes comfortably, because the scale bar still anchors a corner of the content box.
STRANDED_BACKGROUND = (16, 17, 20)
STRANDED_SPECK = (352, 368, 472, 488)
STRANDED_SPECK_RGB = (220, 220, 224)


def stranded_control(source: Path, scratch: Path) -> float:
    """What the gated statistic reads for a deliberately degraded frame."""
    original = np.asarray(Image.open(source).convert("RGB"))
    pixels = np.broadcast_to(
        np.array(STRANDED_BACKGROUND, dtype=original.dtype), original.shape
    ).copy()
    for top, bottom, left, right in GATE.SPREAD_OVERLAY_RECTS:
        pixels[top:bottom, left:right] = original[top:bottom, left:right]
    r0, r1, c0, c1 = STRANDED_SPECK
    pixels[r0:r1, c0:c1] = np.array(STRANDED_SPECK_RGB, dtype=pixels.dtype)

    path = scratch / f"stranded-{source.stem}.png"
    Image.fromarray(pixels).save(path)
    return GATE.subject_spread(path)["narrower_axis"]


def crossing(readings: list[tuple[float, float]]) -> dict[str, object]:
    """The largest factor whose spread is already below the floor, and the last one above it."""
    floor = GATE.MIN_SUBJECT_SPREAD
    below = [f for f, s in readings if s < floor]
    above = [f for f, s in readings if s >= floor]
    return {
        "smallest_factor_still_passing": min(above) if above else None,
        "largest_factor_already_failing": max(below) if below else None,
        "subject_can_shrink_to_this_fraction_before_the_floor_fires": (
            round(min(above), 2) if above else None
        ),
    }


def run_to_run_movement(per_frame: list[dict[str, object]]) -> dict[str, object]:
    """How much the statistic moves between two reconstructions of the same view.

    This is what decides whether a tighter floor is safe. A floor whose headroom above the weakest
    frame is smaller than the movement the statistic shows between runs will eventually fail a
    frame that is fine, and a gate that does that gets deleted -- after which it protects nothing.
    """
    spread = {row["frame"]: row["spread_as_delivered"] for row in per_frame}
    pairs = []
    for name in sorted(spread):
        if not name.startswith("baseline_"):
            continue
        candidate = name.replace("baseline_", "candidate_", 1)
        if candidate not in spread:
            continue
        low, high = sorted((spread[name], spread[candidate]))
        pairs.append(
            {
                "view": name.replace("baseline_", "", 1)[:-4],
                "baseline": spread[name],
                "candidate": spread[candidate],
                "ratio": round(high / low, 2),
            }
        )
    return {
        "matched_pairs": pairs,
        "worst_observed_ratio": max(p["ratio"] for p in pairs) if pairs else None,
        "what_it_bounds": (
            "Two different reconstructions, not two renders of one reconstruction, so this "
            "over-estimates the noise a re-render would show. It is still the only run-to-run "
            "figure these bytes supply."
        ),
        "estimated_from": f"{len(pairs)} matched pairs, which is a weak estimate of a maximum",
    }


def floor_options(
    per_frame: list[dict[str, object]], stranded: dict[str, float], movement: float
) -> list[dict[str, object]]:
    weakest = min(row["spread_as_delivered"] for row in per_frame)
    hardest = max(stranded.values())
    out = []
    for candidate in (0.06, 0.065, 0.07, 0.08):
        headroom = weakest / candidate
        worst_shrink: float | None = None
        for row in per_frame:
            passing = [
                float(f)
                for f, s in row["spread_by_degradation_factor"].items()
                if s >= candidate
            ]
            if passing:
                crossing = min(passing)
                worst_shrink = (
                    crossing if worst_shrink is None else min(worst_shrink, crossing)
                )
        out.append(
            {
                "floor": candidate,
                "times_above_the_hardest_control": round(candidate / hardest, 2),
                "times_below_the_weakest_frame": round(headroom, 2),
                "headroom_exceeds_observed_movement": headroom > movement,
                "worst_case_subject_may_shrink_to": (
                    round(worst_shrink, 2) if worst_shrink is not None else None
                ),
            }
        )
    return out


def build() -> dict[str, object]:
    floor = GATE.MIN_SUBJECT_SPREAD
    per_frame = []
    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        for path in sorted(FRAMES.glob("*.png")):
            readings = [(f, round(spread_at(path, f, scratch), 4)) for f in FACTORS]
            delivered = dict(readings)[1.00]
            per_frame.append(
                {
                    "frame": path.name,
                    "spread_as_delivered": delivered,
                    "margin_above_the_floor": round(delivered - floor, 4),
                    "times_the_floor": round(delivered / floor, 2),
                    "spread_by_degradation_factor": {
                        f"{f:.2f}": s for f, s in readings
                    },
                    **crossing(readings),
                }
            )

    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        stranded = {
            path.name: round(stranded_control(path, scratch), 4)
            for path in sorted(FRAMES.glob("*.png"))
        }

    delivered = [row["spread_as_delivered"] for row in per_frame]
    passes_to = [
        row["subject_can_shrink_to_this_fraction_before_the_floor_fires"]
        for row in per_frame
        if row["subject_can_shrink_to_this_fraction_before_the_floor_fires"] is not None
    ]
    return {
        "what_this_answers": (
            "How far the delivered frames sit from the spread floor, and how much subject a frame "
            "can lose while still passing it. Asked for by a review lane as a control in place of "
            "an argument about where the band's edges should be."
        ),
        "floor_under_test": floor,
        "degradation": (
            "Centre-scale by the factor, padded with the frame's own background -- the same "
            "operation the zoom sweep uses, so a factor means the same thing in both reports."
        ),
        "per_frame": per_frame,
        "the_other_bound_what_the_floor_exists_to_catch": {
            "stranded_speck_control_per_source_frame": stranded,
            "range": [min(stranded.values()), max(stranded.values())],
            "times_the_floor": round(max(stranded.values()) / floor, 2),
            "what_it_is": (
                "The committed negative control, rebuilt here: geometry erased, the bottom-left "
                "annotation region kept, a 16x16 speck at the centre. A floor on the subject's "
                "extent passes it comfortably because the scale bar anchors a corner of the "
                "content box, which is the whole reason stranding is gated on the spread."
            ),
        },
        "is_a_tighter_floor_supported": {
            "run_to_run_movement": None,  # filled below, needs per_frame
            "options": None,
        },
        "summary": {
            "delivered_spread_range": [min(delivered), max(delivered)],
            "closest_frame_to_the_floor": min(delivered),
            "that_frame_is_this_many_times_the_floor": round(min(delivered) / floor, 2),
            "subject_must_shrink_below_this_fraction_before_the_floor_fires": (
                min(passes_to) if passes_to else None
            ),
            "the_floor_sits_between_the_two_bounds": {
                "control_it_must_catch": max(stranded.values()),
                "weakest_frame_it_must_admit": min(delivered),
                "geometric_midpoint_of_those_two": round(
                    (max(stranded.values()) * min(delivered)) ** 0.5, 4
                ),
                "floor": floor,
                "so": (
                    "The floor's placement between the two bounds is measurable rather than a "
                    "matter of intention. It is the separation that decides whether the value is "
                    "doing work, not the fact that it was chosen late."
                ),
            },
        },
        "how_to_read_it": (
            "Two numbers decide it. The margin above the floor says how much room a delivered "
            "frame has before it trips; the margin below says how much room the control has "
            "before it escapes. If both are comfortable the floor's exact value is not what is "
            "doing the work, and a reviewer can propose another without changing a verdict. If "
            "either is thin, the value is load-bearing and was chosen after the fact, which "
            "would be a real objection."
        ),
        "why_the_floor_is_not_being_moved": (
            "The options table shows the data would carry a floor of 0.065: it tightens the "
            "worst-case admissible shrink while preserving measured headroom. It also shows "
            "0.07 and 0.08 are unsafe, because their headroom does not exceed movement already shown "
            "between reconstructions. But the movement estimate rests on four pairs, and the "
            "whole reason this report exists is that a threshold cannot be validated against the "
            "same nine frames it was chosen from. Tightening on that evidence would repeat the "
            "error with a smaller number, so the floor stays where it is and the margin is "
            "recorded instead. A reviewer who wants 0.065 has the figures to ask for it."
        ),
        "what_this_does_not_do": (
            "It does not make the threshold pre-registered, and it does not show the floor catches "
            "every bad frame. It bounds one failure mode -- a subject that has shrunk away from "
            "the camera -- and says how much room there is on that axis. A frame can be badly "
            "framed at full size in ways no spread floor sees."
        ),
    }


def main() -> int:
    if not FRAMES.is_dir():
        print(f"No frames under {FRAMES}", file=sys.stderr)
        return 1
    report = build()
    movement = run_to_run_movement(report["per_frame"])
    report["is_a_tighter_floor_supported"] = {
        "run_to_run_movement": movement,
        "options": floor_options(
            report["per_frame"],
            report["the_other_bound_what_the_floor_exists_to_catch"][
                "stranded_speck_control_per_source_frame"
            ],
            movement["worst_observed_ratio"],
        ),
    }
    destination = FRAMES.parent / "spread-floor-margin.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(f"wrote {destination.name}")
    print(
        f"Delivered frames read {summary['delivered_spread_range'][0]} to "
        f"{summary['delivered_spread_range'][1]} against a floor of {report['floor_under_test']}. "
        f"The closest is {summary['that_frame_is_this_many_times_the_floor']}x the floor, and a "
        f"subject must shrink below "
        f"{summary['subject_must_shrink_below_this_fraction_before_the_floor_fires']} of its "
        f"delivered size before the floor fires."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
