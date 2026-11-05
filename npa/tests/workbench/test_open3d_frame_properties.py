"""The delivered Open3D frames must be gated on their pixels, not on their filenames.

`npa/tests/workbench/test_open3d.py` pins the camera-fitting *math*, and a
companion suite binds each delivered frame to its bytes. Neither reads a pixel of
the delivered PNGs, so a frame re-rendered by hand, captured in a different
window, or cropped afterwards could ship with any framing at all and nothing
would go red. These tests close that, and then break each property the gate
claims to catch, to show the claim is true rather than merely stated.

Every control is built in `tmp_path` from the committed bytes. Nothing here
writes to `docs/evidence/`.

Two of these tests exist because writing this gate the obvious way produced a
gate that did not work:

* `test_a_stranded_subject_passes_the_naive_extent_reading_and_fails_this_gate`
  is the reason stranding is gated on an interquartile spread rather than on the
  subject's extent. Every frame carries a fixed scale bar, which anchors a corner
  of the content bounding box, so a frame whose geometry has collapsed to a speck
  still reads as widely spanning.
* `test_a_crop_that_slips_past_the_framing_checks_is_caught_by_the_label` is the
  hole the zoom sweep found in the framing pair, and the reason the scale label
  is hashed at all.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
FRAMES = EVIDENCE / "frames-refit"
HARNESS = EVIDENCE / "harness"

#: A floor on the subject's *extent*, which is the gate this one deliberately is
#: not. Used only to demonstrate the false negative it would have let through.
NAIVE_EXTENT_FLOOR = 0.40


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


gate = _load("frame_property_gate")
sweep = _load("gate_zoom_sweep")
edge = _load("frame_edge_audit")


def a_frame() -> Path:
    return FRAMES / "baseline_scan_and_surface__overview.png"


def background() -> np.ndarray:
    return np.array([16, 17, 20])


def content_mask(pixels: np.ndarray) -> np.ndarray:
    return (
        np.abs(pixels.astype(np.int16) - background()).max(axis=2)
        > edge.CONTENT_TOLERANCE
    )


def write(tmp_path: Path, name: str, pixels: np.ndarray) -> Path:
    out = tmp_path / f"{name}.png"
    Image.fromarray(pixels).save(out)
    return out


def keep_label(pixels: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Restore the label rectangle, so a control isolates one property at a time."""
    top, bottom, left, right = gate.LABEL_RECT
    pixels[top:bottom, left:right] = original[top:bottom, left:right]
    return pixels


def overlay_only(original: np.ndarray) -> np.ndarray:
    """Return the certified overlay on the frame background, with no subject pixels."""
    pixels = np.broadcast_to(background(), original.shape).copy().astype(original.dtype)
    for top, bottom, left, right in gate.SPREAD_OVERLAY_RECTS:
        pixels[top:bottom, left:right] = original[top:bottom, left:right]
    return pixels


def verdict(path: Path) -> list[str]:
    row = edge.audit(path)
    spread = gate.subject_spread(path)
    return [
        message.split(":")[0]
        for message in gate.failures(row, spread, gate.label_digest(path))
    ]


def extent_of(path: Path) -> float:
    span = edge.audit(path)["content_span_fraction"]
    return min(span["horizontal"], span["vertical"])


@pytest.fixture
def original() -> np.ndarray:
    return np.asarray(Image.open(a_frame()).convert("RGB"))


def test_every_delivered_frame_passes_the_gate() -> None:
    frames = sorted(FRAMES.glob("*.png"))
    assert len(frames) == 9, f"expected the nine delivered frames, found {len(frames)}"

    report = gate.check(frames)
    assert report["passed"], report["failed_frames"]


def test_the_gate_measures_the_bytes_rather_than_trusting_the_record() -> None:
    """A record says what was measured; this re-measures and must agree with it."""
    import json

    recorded = {
        row["frame"]: row
        for row in json.loads((EVIDENCE / "frame-edge-audit-after.json").read_text())[
            "frames"
        ]
    }
    for path in sorted(FRAMES.glob("*.png")):
        fresh = edge.audit(path)
        assert fresh["content_fraction"] == pytest.approx(
            recorded[path.name]["content_fraction"], abs=0.0
        ), f"{path.name} no longer measures what the record says it measured"


def test_content_in_the_border_band_is_caught(
    tmp_path: Path, original: np.ndarray
) -> None:
    pixels = original.copy()
    pixels[0:3, 400:560] = np.array([220, 220, 224], dtype=pixels.dtype)

    assert verdict(write(tmp_path, "clipped", pixels)) == ["clipped"]


def test_a_stranded_subject_passes_the_naive_extent_reading_and_fails_this_gate(
    tmp_path: Path, original: np.ndarray
) -> None:
    """The whole reason stranding is gated on spread and not on extent.

    The geometry is erased and replaced with a 16x16 speck at the centre. The
    fixed scale bar still anchors the bottom-left of the content bounding box, so
    the extent reading stays comfortably above a floor that looks reasonable --
    while the frame is plainly useless.
    """
    pixels = overlay_only(original)
    pixels[352:368, 472:488] = np.array([220, 220, 224], dtype=pixels.dtype)

    path = write(tmp_path, "stranded", pixels)

    assert extent_of(path) >= NAIVE_EXTENT_FLOOR, (
        "the control is only interesting if the naive extent gate would have "
        "passed this frame"
    )
    assert gate.subject_spread(path)["narrower_axis"] < gate.MIN_SUBJECT_SPREAD
    assert verdict(path) == ["stranded"]


@pytest.mark.parametrize("top,left", [(340, 460), (80, 850)])
def test_the_overlay_cannot_rescue_a_small_subject(
    tmp_path: Path, original: np.ndarray, top: int, left: int
) -> None:
    """The spread gate must measure the subject, not an overlay-plus-subject mixture.

    Forty pixels is deliberately close to the 0.06 floor on a 720px frame. With
    the scale overlay left in the quartile population this control can clear the
    floor; after the certified overlay band is excluded it fails regardless of
    whether the subject is centred or stranded near a corner.
    """
    pixels = overlay_only(original)
    pixels[top : top + 40, left : left + 40] = np.array(
        [220, 220, 224], dtype=pixels.dtype
    )

    path = write(tmp_path, f"overlay_and_small_subject_{top}_{left}", pixels)
    assert gate.subject_spread(path)["narrower_axis"] < gate.MIN_SUBJECT_SPREAD
    assert verdict(path) == ["stranded"]


def test_washed_out_content_is_caught(tmp_path: Path, original: np.ndarray) -> None:
    pixels = original.copy()
    mask = content_mask(pixels)
    pixels[mask] = (
        background() + (pixels[mask].astype(np.int16) - background()) * 0.12
    ).astype(pixels.dtype)

    assert "washed out" in verdict(
        write(tmp_path, "washed", keep_label(pixels, original))
    )


def test_a_washed_out_floor_is_caught_while_the_median_still_looks_healthy(
    tmp_path: Path, original: np.ndarray
) -> None:
    """Shows the second contrast check is not redundant with the first.

    Two in five content pixels are pushed to within a few levels of the
    background -- drawn, and invisible. A median by construction cannot see that,
    and here it stays well above its own floor while the distribution gate fires.
    """
    pixels = original.copy()
    indices = np.argwhere(content_mask(pixels))
    chosen = indices[
        np.random.default_rng(7).choice(
            len(indices), size=int(len(indices) * 0.40), replace=False
        )
    ]
    # +10 keeps these above the content tolerance of 6, so they still count as
    # content, while sitting inside the low-contrast limit of 24.
    pixels[chosen[:, 0], chosen[:, 1]] = (background() + 10).astype(pixels.dtype)

    path = write(tmp_path, "washed_floor", keep_label(pixels, original))
    row = edge.audit(path)

    assert row["legibility"]["median_contrast_vs_background"] > gate.MIN_MEDIAN_CONTRAST
    assert verdict(path) == ["washed out floor"]


def test_a_frame_that_lost_its_scale_label_is_caught(
    tmp_path: Path, original: np.ndarray
) -> None:
    pixels = original.copy()
    top, bottom, left, right = gate.LABEL_RECT
    pixels[top:bottom, left:right] = background().astype(pixels.dtype)

    assert verdict(write(tmp_path, "no_label", pixels)) == [
        "scale label missing or altered"
    ]


def test_the_label_check_also_fires_on_a_harmless_restyle_so_a_red_may_be_cosmetic(
    tmp_path: Path, original: np.ndarray
) -> None:
    """Names the intended failure mode, so the next reader does not loosen the gate.

    An exact pixel hash cannot distinguish *the frame lost its scale reference*
    from *someone changed the label's colour*. Both come out as the same red. A
    reviewer who meets this failure after a deliberate restyle should re-bless
    ``LABEL_SHA256`` having checked the label by eye -- not widen the check, which
    is how a repository ends up with an assertion that no longer asserts.

    The restyle here is as benign as it gets: the glyphs keep their exact shape
    and shift by four levels of grey.
    """
    pixels = original.copy()
    top, bottom, left, right = gate.LABEL_RECT
    patch = pixels[top:bottom, left:right]
    glyphs = np.abs(patch.astype(np.int16) - background()).max(axis=2) > 32
    patch[glyphs] = (patch[glyphs].astype(np.int16) - 4).astype(pixels.dtype)

    path = write(tmp_path, "restyled_label", pixels)

    assert verdict(path) == ["scale label missing or altered"]
    # The frame is otherwise untouched: nothing else the gate measures moved.
    assert edge.audit(path)["is_clipped"] is False
    assert gate.subject_spread(path)["narrower_axis"] > gate.MIN_SUBJECT_SPREAD


def test_a_crop_that_slips_past_the_framing_checks_is_caught_by_the_label(
    tmp_path: Path,
) -> None:
    """The hole the zoom sweep found, and the reason the label is hashed.

    At 1.3x this frame has had its scale bar cropped away entirely, so the bar no
    longer contributes border content and the geometry happens to miss the border
    on that step. The framing checks pass a frame that has thrown away its scale
    reference and cropped its subject.
    """
    source = FRAMES / "candidate_scan_and_surface__topdown.png"
    cropped = tmp_path / "zoom_1_30.png"
    sweep.rezoom(source, 1.30, cropped)

    row = edge.audit(cropped)
    spread = gate.subject_spread(cropped)

    assert gate.failures(row, spread) == [], (
        "this control is only meaningful while the framing checks alone still "
        "pass the 1.3x crop"
    )
    assert verdict(cropped) == ["scale label missing or altered"]


def test_the_gate_stays_quiet_on_a_change_that_is_not_a_defect(
    tmp_path: Path, original: np.ndarray
) -> None:
    """A gate that rejects everything protects nothing.

    Recolouring the geometry to a different high-contrast colour changes almost
    every content pixel and breaks no property the gate is about.
    """
    pixels = original.copy()
    pixels[content_mask(pixels)] = np.array([90, 200, 255], dtype=pixels.dtype)

    assert verdict(write(tmp_path, "recoloured", keep_label(pixels, original))) == []


def test_both_axes_of_the_spread_are_quartile_measures(tmp_path: Path) -> None:
    """Pins each axis separately, because ``min()`` hides a half-reverted measure.

    Mutation testing caught this: switching only the column axis back to an
    extent left every other test passing, since the narrower-axis ``min()`` just
    returned the still-correct row axis. A gate that is quartile-based on one
    axis and extent-based on the other is broken in a way no amount of frame
    testing would have shown.

    The synthetic frame is a dense central block plus two lone far-corner pixels,
    so extent and quartile spread disagree by an order of magnitude on both axes.
    """
    pixels = np.full((720, 960, 3), background(), dtype=np.uint8)
    bright = np.array([220, 220, 224], dtype=np.uint8)
    pixels[340:380, 460:500] = bright
    pixels[10, 10] = bright
    pixels[700, 940] = bright

    path = write(tmp_path, "two_outliers", pixels)
    spread = gate.subject_spread(path)

    assert extent_of(path) > 0.9, "the control needs extent and spread to disagree"
    assert spread["horizontal"] < 0.05, (
        f"the column axis reads {spread['horizontal']:.3f}, which is an extent, "
        f"not a quartile spread"
    )
    assert spread["vertical"] < 0.05, (
        f"the row axis reads {spread['vertical']:.3f}, which is an extent, not a "
        f"quartile spread"
    )


def narrower_readings(flags: np.ndarray) -> tuple[float, float]:
    """The extent and the quartile spread of a mask, on its narrower axis."""
    rows, cols = np.where(flags)
    height, width = flags.shape
    return (
        min(
            (cols.max() - cols.min() + 1) / width,
            (rows.max() - rows.min() + 1) / height,
        ),
        min(
            (np.percentile(cols, 75) - np.percentile(cols, 25)) / width,
            (np.percentile(rows, 75) - np.percentile(rows, 25)) / height,
        ),
    )


def provable_annotation() -> np.ndarray:
    """Pixels that are identical in all nine frames and are not background.

    Nine projections of different geometry from different cameras cannot agree
    pixel for pixel, so whatever is identical across all nine was drawn by the
    renderer rather than seen by the camera. That makes this a mask which
    provably contains no geometry.

    It is a *lower bound* on the annotation: the bar's length encodes each
    view's metres-per-pixel, so the part of it that varies is excluded, as are
    glyph edges anti-aliased against differing geometry behind them.
    """
    stack = np.stack(
        [
            np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
            for path in sorted(FRAMES.glob("*.png"))
        ]
    )
    identical = (stack == stack[0]).all(axis=(0, 3))
    return identical & content_mask(stack[0])


def test_the_spread_is_insensitive_to_the_annotation_under_every_mask_tried() -> None:
    """Half of the gate's central design decision, and the half that is robust.

    Removing the annotation barely moves the quartile spread, whether the mask is
    the provable one or a generous rectangle around the whole block. The extent is
    the opposite: how much it moves depends on how much you remove, which is the
    subject of the companion test.
    """
    annotation = provable_annotation()
    provable_spread_shifts, rectangle_spread_shifts = [], []

    for path in sorted(FRAMES.glob("*.png")):
        pixels = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
        mask = content_mask(pixels)
        _, spread = narrower_readings(mask)

        _, provable = narrower_readings(mask & ~annotation)
        rectangle = mask.copy()
        rectangle[645:700, 18:435] = False
        _, generous = narrower_readings(rectangle)

        provable_spread_shifts.append(abs(spread - provable))
        rectangle_spread_shifts.append(abs(spread - generous))

    assert max(provable_spread_shifts) < 0.01, (
        f"the spread moved by {max(provable_spread_shifts):.4f} when a mask "
        f"containing no geometry was removed"
    )
    assert max(rectangle_spread_shifts) < 0.01, (
        f"the spread moved by {max(rectangle_spread_shifts):.4f} under a "
        f"generous mask, so its robustness does not survive mask choice"
    )


def test_how_much_the_extent_moves_depends_on_where_you_draw_the_annotation() -> None:
    """Why exploratory rectangles are not the production mask.

    A generous rectangle that removes real geometry and an all-frame pixel
    intersection produce different extent shifts. This historical sensitivity
    check does not choose either mask: the production gate uses the separately
    matched-pair-certified overlay region.
    """
    annotation = provable_annotation()
    provable_shifts, rectangle_shifts = [], []

    for path in sorted(FRAMES.glob("*.png")):
        pixels = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
        mask = content_mask(pixels)
        extent, _ = narrower_readings(mask)

        provable, _ = narrower_readings(mask & ~annotation)
        rectangle = mask.copy()
        rectangle[645:700, 18:435] = False
        generous, _ = narrower_readings(rectangle)

        provable_shifts.append(abs(extent - provable))
        rectangle_shifts.append(abs(extent - generous))

    assert max(rectangle_shifts) > 3 * max(provable_shifts), (
        "the extent reading is supposed to be the mask-sensitive one; if the two "
        "masks now agree, the contamination argument needs restating"
    )


def test_the_extent_reading_rewards_worse_placement_and_the_spread_does_not(
    tmp_path: Path, original: np.ndarray
) -> None:
    """The decisive argument, and the one that needs no mask at all.

    A reviewer measured this independently and it is stronger than the shift
    figure I first leaned on. With the annotation anchoring one corner, moving a
    16x16 speck *further from centre* makes the extent reading go *up*. So an
    extent floor is weakest exactly where the framing is worst, which is the
    opposite of what a gate must do. The quartile spread is flat across all three
    placements and fires on every one.
    """
    placements = {
        "dead centre": (352, 472),
        "upper right": (120, 800),
        "far corner": (40, 900),
    }
    extents, spreads = [], []

    for name, (top, left) in placements.items():
        pixels = overlay_only(original)
        pixels[top : top + 16, left : left + 16] = np.array(
            [220, 220, 224], dtype=pixels.dtype
        )

        path = write(tmp_path, f"speck_{name.replace(' ', '_')}", pixels)
        extents.append(extent_of(path))
        spreads.append(gate.subject_spread(path)["narrower_axis"])

        assert extent_of(path) >= NAIVE_EXTENT_FLOOR, (
            f"the {name} speck scores {extent_of(path):.3f} on extent, so this "
            f"control no longer shows the floor being passed"
        )
        assert verdict(path) == ["stranded"]

    assert extents == sorted(extents), (
        f"extent is supposed to reward worse placement here, but read {extents}"
    )
    assert max(spreads) - min(spreads) < 0.01, (
        f"the spread is supposed to be placement-invariant, but ranged {spreads}"
    )


def test_a_fill_gate_would_have_no_ceiling(tmp_path: Path) -> None:
    """Why ``content_fraction`` is recorded and not gated.

    Fill rises without limit as a frame zooms in, so a fill floor is passed by
    cropping -- and what crops away first is the sparse, incomplete geometry
    these views exist to show.
    """
    source = a_frame()
    fills = []
    for factor in (0.5, 0.8, 1.0, 1.3, 1.8):
        staged = tmp_path / f"fill_{factor}.png"
        sweep.rezoom(source, factor, staged)
        fills.append(edge.audit(staged)["content_fraction"])

    assert fills == sorted(fills)
    assert fills[-1] > fills[0] * 2


def test_the_matched_pair_diff_isolates_the_overlay_without_a_mask() -> None:
    """The control that settled which pixels are annotation, after three masks disagreed.

    Baseline and candidate are different reconstructions of the same view. If the band held any
    subject geometry, a matched pair must differ inside it. Pinned because this is the evidence a
    published figure now rests on, and because an earlier figure here was corrected twice before
    anyone ran a control instead of arguing about mask boundaries.
    """
    controls = _load("overlay_isolation_controls")
    rows = controls.pair_diff(160, 37)

    assert len(rows) == 4, rows
    for row in rows:
        assert row["pixels_differing_inside_the_band"] == 0, row
        assert row["pixels_differing_elsewhere"] > 80_000, row


def test_the_plateau_sweep_settles_and_stays_settled() -> None:
    """A box still eating geometry keeps moving the answer; one that has stopped is the boundary."""
    controls = _load("overlay_isolation_controls")
    sweep = controls.plateau()

    for height in controls.SWEEP_HEIGHTS:
        flat = controls.flat_region(sweep, height)
        assert flat["settles_at"] == (0.2635, 0.5104), (height, flat)
        assert flat["first_width_that_settles"] <= 160, (height, flat)
        # Flat over a wide growth is the whole argument: a narrow flat region would not establish
        # that there is no subject content left in those rows.
        assert flat["flat_over_px_of_growth"] >= 200, (height, flat)


def test_the_all_nine_intersection_keeps_only_part_of_the_overlay() -> None:
    """Why the obvious isolation rule is wrong, which is what made the first correction wrong.

    The overlay's bar length encodes each view's metres-per-pixel, so it is view-dependent.
    Intersecting all nine frames keeps only the view-invariant remnant and leaves the rest of the
    label in the image, where it goes on pinning the content bounding box.
    """
    controls = _load("overlay_isolation_controls")
    sizes = controls.per_view_overlay_size(160, 37)
    remnant = controls.view_invariant_remnant()

    assert len(set(sizes.values())) > 1, sizes
    assert remnant < min(sizes.values()), (remnant, sizes)


def test_the_spread_floor_has_measured_margin_on_both_sides() -> None:
    """Answers 'is 0.06 too generous' with two measurements instead of an assertion.

    A review lane pointed out that the bracketing argument establishes the pass band exists and
    says nothing about its edges, and that no argument from these nine frames can validate them.
    So this pins the separation rather than the choice: the control the floor must catch, and the
    weakest frame it must admit, both stay on their own side of it.
    """
    margin = _load("spread_floor_margin")
    floor = gate.MIN_SUBJECT_SPREAD

    delivered = [
        gate.subject_spread(path)["narrower_axis"]
        for path in sorted(FRAMES.glob("*.png"))
    ]
    assert min(delivered) > floor, min(delivered)

    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        built = [
            (path, margin.stranded_control(path, Path(scratch)))
            for path in sorted(FRAMES.glob("*.png"))
        ]
        stranded = [spread for _, spread in built]
        # The control must be the collapsed-speck case, not merely an empty frame: the naive
        # extent reading has to still pass, because that false negative is the reason stranding
        # is gated on the spread at all. Mutation testing added this -- removing the speck left a
        # frame that also reads below the floor, so a test asserting only that could not tell the
        # two apart, and the weaker control would have proved a weaker claim.
        naive = [
            extent_of(Path(scratch) / f"stranded-{path.stem}.png") for path, _ in built
        ]
    assert max(stranded) < floor, max(stranded)
    assert min(naive) >= NAIVE_EXTENT_FLOOR, min(naive)

    # Once the overlay is excluded from the gated population, the negative control
    # has the larger margin. Pin that asymmetry so a future unmasked implementation
    # cannot silently restore the overlay's contribution.
    assert floor / max(stranded) > min(delivered) / floor


def test_the_gated_statistic_is_the_one_the_mask_cannot_move() -> None:
    """The mask dispute was about a diagnostic; the gate acts on a near-mask-free statistic.

    This is the strongest justification for choosing the spread over the extent, and it was
    recorded only as an absolute movement rather than as the ratio that makes the case. A
    reviewer asked for the ratio after neither of us had thought to measure it.
    """
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    sensitivity = report["sensitivity_at_the_certified_overlay_only_width"]

    assert sensitivity["worst_extent_relative_change"] > 0.40, sensitivity
    assert sensitivity["worst_spread_relative_change"] < 0.03, sensitivity
    # An order of magnitude is the claim. Pinning the exact 26.7 would pin an incidental.
    assert sensitivity["how_many_times_more_sensitive_the_extent_is"] >= 10.0, (
        sensitivity
    )


def test_a_flat_extent_plateau_does_not_prove_the_box_holds_no_subject() -> None:
    """The two controls disagree above 290px, and the pair diff is the one that is right.

    The extent goes flat at 160px and stays flat to 440px, which is what made me treat the
    widest box as safe. Flatness only means the annotation no longer pins the bounding box; a
    wider box can then remove interior subject pixels without the extent noticing. The pair
    diff does notice, because two reconstructions differ wherever real geometry sits.
    """
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    certified = report["which_box_is_actually_overlay_only"]
    readings = {r["exclusion_width"]: r for r in certified["readings"]}

    assert certified["widest_certified_overlay_only"] == 290, certified
    assert certified["first_width_proven_to_contain_subject_geometry"] == 320, certified
    # The gate's own exclusion and the plateau onset both sit inside the certified region,
    # which is what makes the published subject extent safe despite the correction.
    assert readings[160]["max_pixels_differing_inside_the_band"] == 0
    # And the contamination grows with width rather than appearing once, which is what a box
    # eating into geometry looks like as opposed to a single stray glyph.
    assert (
        readings[320]["max_pixels_differing_inside_the_band"]
        < readings[380]["max_pixels_differing_inside_the_band"]
        < readings[440]["max_pixels_differing_inside_the_band"]
    ), readings


def test_the_corrected_spread_sensitivity_replaced_a_contaminated_measurement() -> None:
    """0.0042 was measured at 440px, inside the region proven to contain subject geometry."""
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    correction = report["a_figure_this_corrects"]

    assert correction["figure_the_record_carried"] == 0.0042
    assert correction["corrected_figure"] == 0.0021
    assert correction["measured_at_certified_width"] == 290
    # The record must carry the corrected figure, not the one this report retires.
    record = json.loads(
        (EVIDENCE / "capability-record-frame-property-gate.json").read_text()
    )
    blob = json.dumps(record)
    assert "0.0021" in blob
    assert "0.0042" not in blob.replace("0.00421", "")


def test_the_floors_fragility_is_stated_from_observed_movement_not_from_the_nine_frames() -> (
    None
):
    """A margin statement that needs no new threshold, so it cannot be a disguised retune."""
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    fragility = report["the_floors_fragility_without_proposing_a_new_floor"]

    assert fragility["weakest_delivered_frame_over_floor"] == 1.74
    assert (
        fragility["largest_movement_observed_between_matched_reconstructions"] == 1.50
    )
    # 1.74 / 1.50: the masked weakest frame has about 16% beyond observed movement.
    assert fragility["headroom_beyond_already_observed_movement"] == 1.16


def test_the_certification_covered_the_annotations_full_height_not_a_strip() -> None:
    """Certification on a sliver must not license a measurement over the full band.

    Mutation testing found this: shrinking the pair-diff band to one pixel tall still produces
    the same 290/320 boundary, so every other assertion here passed while the certified region
    and the measured region had come apart.

    My first fix was to have the report state the height it used and assert the two agreed,
    which is a self-declaration a test reading that report cannot check -- the mutation changed
    the band and left the declaration intact, and it survived again. The height has to be
    inferred from behaviour instead. Contamination counts at a width known to cut into geometry
    scale with how much of the annotation's height the band covers: 124 pixels at 1px tall,
    1,248 at 10px, and 4,174 at 37px and above, where it saturates because the annotation is
    about 37 rows deep.
    """
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    readings = {
        r["exclusion_width"]: r
        for r in report["which_box_is_actually_overlay_only"]["readings"]
    }

    assert readings[440]["max_pixels_differing_inside_the_band"] >= 3000, readings[440]
    assert readings[380]["max_pixels_differing_inside_the_band"] >= 2000, readings[380]


def test_the_headline_ratio_is_not_a_flattering_pick_among_the_nine() -> None:
    """The published ratio must sit near the bottom of the per-frame range, not the top.

    A reviewer verified 26.7x and called it the smallest per-frame ratio of the nine. It is the
    second smallest; 24.5x is smaller. Worth pinning because the claim being made is that the
    conclusion does not depend on which frame you look at.
    """
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    spread = report["where_the_headline_ratio_sits_among_the_nine"]
    headline = report["sensitivity_at_the_certified_overlay_only_width"][
        "how_many_times_more_sensitive_the_extent_is"
    ]

    assert spread["smallest_per_frame_ratio"] >= 10.0, spread
    # The headline must be no kinder than the typical frame, which is the whole point.
    assert (
        headline
        <= (spread["smallest_per_frame_ratio"] + spread["largest_per_frame_ratio"]) / 2
    ), (headline, spread)
    # One frame's spread does not move at all. Counting it as a huge ratio would inflate the
    # range, and dropping it silently would hide a frame the mask cannot touch.
    assert spread["frames_whose_spread_does_not_move_at_all"], spread


def test_both_the_masked_and_unmasked_margins_are_recorded() -> None:
    """1.74x and 1.77x are the same quantity read two ways, and only one is operative."""
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    margin = report["the_margin_masked_and_unmasked"]

    assert margin["weakest_unmasked_over_floor"] == 1.77
    assert margin["weakest_masked_over_floor"] == 1.74
    # The gate reads the masked frame, so the masked figure must be the smaller of the two or
    # the record is describing the kinder number as the operative one.
    assert margin["weakest_masked_over_floor"] <= margin["weakest_unmasked_over_floor"]


def test_the_certified_width_in_prose_matches_the_table_beside_it() -> None:
    """A sentence said 260px where the table and the field both said 290px.

    Review caught it. It understated my own result by one step of the sweep and changed no
    conclusion, which is exactly why nothing else would have caught it. Pinned as an
    agreement between the prose and the measurement rather than as a literal, so a re-run
    cannot leave the sentence behind.
    """
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    box = report["which_box_is_actually_overlay_only"]
    certified = box["widest_certified_overlay_only"]

    zero = [
        reading["exclusion_width"]
        for reading in box["readings"]
        if reading["max_pixels_differing_inside_the_band"] == 0
    ]
    assert max(zero) == certified, (zero, certified)

    prose = report["the_plateau_is_necessary_but_not_sufficient"][
        "what_the_pair_diff_shows_instead"
    ]
    assert f"through {certified}px" in prose, prose
    assert f"above {certified}px" in prose, prose


def test_every_ratio_names_both_of_its_axes() -> None:
    """Quoting one axis is what let a cross-measurement disagreement run for three exchanges.

    A reviewer pinned the extent to the horizontal axis and varied only the spread axis; this
    harness moved both together. Each of us was invariant within our own family, neither family
    was stated, and two hypotheses died before the pairing was named. I had also reported a
    frame as "convention-invariant" when it was only invariant across conventions that move both
    axes at once, which is a label claiming more than the control tested.
    """
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    pairings = report["whether_the_ratios_depend_on_the_axis_convention"]["per_frame"]

    assert len(pairings) == 9, len(pairings)
    for frame in pairings:
        keys = frame["ratio_by_axis_pairing"].keys()
        # Three axis choices on each side, so nine pairings, and every key names both.
        assert len(keys) == 9, sorted(keys)
        for key in keys:
            assert key.startswith("extent_") and "_over_spread_" in key, key
    assert (
        "extent_narrower_over_spread_narrower" in pairings[0]["ratio_by_axis_pairing"]
    )


def test_the_pairing_matrix_isolates_the_disagreement_to_the_horizontal_spread() -> (
    None
):
    """The reviewer's vertical-spread column reproduces here; their horizontal one does not.

    The numerator is shared between their two columns, so agreeing on one and differing on the
    other localises it to a single term without either side guessing. Pinned because this is the
    handoff, and it is a term either harness can print.
    """
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    by_frame = {
        f["frame"]: f["ratio_by_axis_pairing"]
        for f in report["whether_the_ratios_depend_on_the_axis_convention"]["per_frame"]
    }
    observed = by_frame["observed_scan_only__overview.png"]
    candidate = by_frame["candidate_surface_only__overview.png"]

    # Reproduces their extent-horizontal over spread-vertical column exactly, on both frames.
    assert observed["extent_horizontal_over_spread_vertical"] == 24.6, observed
    assert candidate["extent_horizontal_over_spread_vertical"] == 18.9, candidate
    # Differs from their horizontal-spread column on both: they read 48.5 and undefined.
    assert observed["extent_horizontal_over_spread_horizontal"] == 24.6, observed
    assert candidate["extent_horizontal_over_spread_horizontal"] == 48.3, candidate


def test_the_disputed_ratio_is_published_with_its_interval_not_as_a_bare_number() -> (
    None
):
    """A ratio whose denominator is one pixel wide cannot honestly be quoted as one number.

    A reviewer swept the background tolerance and found this diagnostic oscillating between
    roughly 24.6 and 48.8, which reproduced both figures they had published. The denominator is
    the relative movement of an interquartile spread; that movement is one to two pixels in 126,
    so one pixel crossing the threshold doubles the ratio. They asked for a stability note. The
    sweep showed it was not one frame, so every entry carries its interval instead.
    """
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    hinge = report["why_that_ratio_could_sustain_four_exchanges"]

    for entry in hinge["per_frame_interval"]:
        assert "ratio_lowest_over_the_sweep" in entry, entry
        assert "ratio_highest_over_the_sweep" in entry, entry
        assert (
            entry["ratio_highest_over_the_sweep"]
            >= entry["ratio_lowest_over_the_sweep"]
        )
        # An interval that excludes the published reading would be a different quantity's.
        published = entry["ratio_at_the_published_tolerance"]
        if published is not None:
            assert (
                entry["ratio_lowest_over_the_sweep"]
                <= published
                <= entry["ratio_highest_over_the_sweep"]
            ), entry

    breadth = hinge["this_is_broader_than_the_frame_it_was_found_on"]
    assert breadth["frames_swinging_2x_or_more"] >= 5, breadth
    assert breadth["widest_swing"] >= 2.9, breadth


def test_the_published_reading_sits_on_a_plateau_rather_than_on_the_hinge() -> None:
    """Reproducibility of the published figure is a property of where it sits, not of the ratio.

    The value is stable from tolerance 0 through 8 and jumps at 10. That is why two harnesses can
    agree at the stated condition while disagreeing by 2x a few steps away.
    """
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    hinge = report["why_that_ratio_could_sustain_four_exchanges"]
    plateau = hinge["the_plateau_around_the_published_condition"]
    published = hinge["the_published_tolerance"]

    assert plateau["stable_from"] <= published <= plateau["stable_through"], plateau

    bands = hinge["the_disputed_frame_band_by_band"]
    containing = [b for b in bands if published in b["tolerances"]]
    assert len(containing) == 1, bands
    assert sorted(containing[0]["tolerances"]) == list(
        range(plateau["stable_from"], plateau["stable_through"] + 1, 2)
    ), (containing, plateau)
    # The mechanism, not just the symptom: the bands differ by one pixel of quartile movement.
    moved = {b["horizontal_quartile_pixels_moved"] for b in bands}
    assert len(moved) > 1, bands
    assert max(moved) - min(moved) == 1.0, moved


def test_the_diagnostic_spread_survives_every_tolerance_in_its_own_sweep() -> None:
    """The diagnostic's spread is stable; this is not a production-gate measurement.

    This harness uses a median background and varies its tolerance. The production
    gate uses a modal corner background and a fixed tolerance, so its behavior is
    covered by direct gate tests rather than inferred from this table.
    """
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    measured = report["why_that_ratio_could_sustain_four_exchanges"][
        "diagnostic_spread_sensitivity_not_the_production_gate"
    ]

    floor = json.loads(
        (HARNESS / "statistic_mask_sensitivity.py")
        .read_text()
        .split("SPREAD_FLOOR = ")[1]
        .split("\n")[0]
    )
    table = measured["gated_spread_by_frame_and_tolerance"]
    assert len(table) == 9, sorted(table)

    # Recomputed from the readings, because asserting the summary let a 3x inflation of the
    # margin pass mutation testing: the test was reading the claim, not the measurement.
    readings = [value for row in table.values() for value in row.values()]
    assert min(readings) == measured["worst_delivered_spread_over_the_whole_sweep"], (
        min(readings),
        measured["worst_delivered_spread_over_the_whole_sweep"],
    )
    assert measured["worst_as_multiple_of_floor"] == pytest.approx(
        min(readings) / floor, abs=0.01
    ), measured
    assert min(readings) >= floor, min(readings)

    failing = sorted(
        name for name, row in table.items() if any(v < floor for v in row.values())
    )
    assert (
        failing
        == measured[
            "frames_crossing_the_nominal_floor_anywhere_in_the_diagnostic_sweep"
        ]
    )
    assert failing == [], failing


def test_the_dispute_record_names_the_tolerance_that_reproduces_the_other_figure() -> (
    None
):
    """An explanation that cannot be rerun is an assertion. This one names its condition."""
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    explained = report["why_that_ratio_could_sustain_four_exchanges"][
        "the_dispute_is_now_explained_rather_than_adjudicated"
    ]

    assert "24 through 30" in explained["what_reproduces_here"], explained
    assert "48.38" in explained["what_reproduces_here"], explained
    hinge = report["why_that_ratio_could_sustain_four_exchanges"]
    reproducing = [
        b for b in hinge["the_disputed_frame_band_by_band"] if b["ratio"] == 48.38
    ]
    assert len(reproducing) == 1, hinge["the_disputed_frame_band_by_band"]
    assert reproducing[0]["tolerances"] == [24, 26, 28, 30], reproducing


def _headline_section() -> dict:
    report = json.loads((EVIDENCE / "statistic-mask-sensitivity.json").read_text())
    hinge = report["why_that_ratio_could_sustain_four_exchanges"]
    return report, hinge, hinge["but_that_is_not_the_frame_the_headline_comes_from"]


def test_the_frame_the_headline_comes_from_is_banded_too() -> None:
    """Review's P4: the frame I banded was not the frame the number is quoted from.

    The band table covered `observed_scan_only__overview.png`, where the mask dispute
    happened. The headline ratio comes from a different frame, and by the argument that
    justified banding anything -- an interval does not say whether a reading is
    reproducible, the band structure does -- the frame whose number travels is the one
    that most needs it. From the published data a reviewer could not tell whether 26.7
    sat on a plateau or beside a hinge.

    It sits on a plateau. The run containing the published tolerance covers 11 of the 16
    swept tolerances, against 5 for the disputed frame.

    That 11 does not identify the frame: pointing `HEADLINE_FRAME` at
    `candidate_surface_only__overview.png` produces a run of 11 as well, which this test
    accepts. The identity is bound by the next test instead, and the two are only sound
    together.
    """
    report, hinge, section = _headline_section()
    published = hinge["the_published_tolerance"]
    plateau = section["the_plateau_around_the_published_condition"]

    assert plateau["stable_from"] <= published <= plateau["stable_through"]

    # Both counts are read off the band table rather than asserted as literals. The
    # first version of this test wrote `== 11` and `== 6`, and the 6 was wrong: the
    # harness had counted every tolerance sharing the ratio instead of the run, and
    # 24.58 returns at tolerance 18 after an excursion. A literal cannot notice that it
    # disagrees with the table printed beside it, which is how it reached a reviewer.
    for label, band_table, pl in (
        (
            "headline",
            section["the_headline_frame_band_by_band"],
            plateau,
        ),
        (
            "disputed",
            hinge["the_disputed_frame_band_by_band"],
            section["the_disputed_frame_for_comparison"],
        ),
    ):
        band = next(b for b in band_table if published in b["tolerances"])
        tolerances = band["tolerances"]
        # Contiguous, or "plateau" is the wrong word for it.
        assert tolerances == list(range(min(tolerances), max(tolerances) + 1, 2)), band
        assert pl["tolerances_in_the_run"] == len(tolerances), (label, band, pl)
        assert pl["stable_from"] == min(tolerances), (label, band, pl)
        assert pl["stable_through"] == max(tolerances), (label, band, pl)
        # Recurrences belong outside the run by construction, and the disputed frame has
        # one. Naming them is what makes the count auditable rather than merely correct.
        assert all(
            t not in tolerances for t in pl["the_same_ratio_recurs_outside_the_run"]
        ), (label, pl)

    assert plateau["tolerances_in_the_run"] == 11, plateau
    assert section["the_disputed_frame_for_comparison"][
        "the_same_ratio_recurs_outside_the_run"
    ] == [18]


def test_the_named_headline_frame_is_the_frame_the_headline_ratio_belongs_to() -> None:
    """`HEADLINE_FRAME` is a name, so it is bound to the number rather than trusted.

    The headline figure only exists in prose in this record, so the constant cannot be
    derived from it. What can be checked is that the frame it names still carries that
    ratio in the nine-frame table -- so pointing it at a different frame stops agreeing
    with the table beside it.
    """
    report, _, section = _headline_section()
    plateau = section["the_plateau_around_the_published_condition"]
    quoted = plateau["ratio_at_the_published_tolerance"]

    rows = report["where_the_headline_ratio_sits_among_the_nine"]["per_frame"]
    # Within the coarser record's own precision, rather than by equality after rounding.
    # The table publishes one decimal and the sweep two, so the same quantity reads 26.7
    # and 26.65. Re-rounding does not bridge them -- the underlying value is ~26.651, and
    # `round(26.65, 1)` is 26.6 in binary floating point -- and a half-tenth tolerance
    # lands exactly on the boundary, where the float difference is 0.050000000000000711.
    matching = [
        row
        for row in rows
        if row["ratio"] is not None and abs(row["ratio"] - quoted) < 0.1
    ]
    assert len(matching) == 1, (
        "the named headline frame's ratio does not appear exactly once in the "
        "nine-frame table, so the name and the table no longer agree",
        quoted,
        matching,
    )
    assert (
        matching[0]["frame"]
        in section["where_the_published_condition_sits_in_each_run"]
    )


def test_the_record_does_not_claim_a_wide_swing_makes_a_reading_fragile() -> None:
    """The finding, which is the opposite of what the question expected.

    The frame with the widest swing across the sweep has the *longer* plateau at the
    published condition, and both frames are two steps from their nearest transition. So
    swing and local stability are different properties, and a reading cannot be called
    fragile because its frame travels a long way somewhere else in the sweep.

    This is pinned because it is the kind of conclusion that invites being tidied into
    the simpler, wrong sentence -- that the widest-swinging frame is the least reliable.
    """
    _, _, section = _headline_section()
    # The note has to exist, but its wording is not what makes the claim true. Asserting
    # an English phrase appears inside it would be a grep over prose -- the defect review
    # caught on #1013 -- so what is checked is the measured relation the note describes.
    assert section["swing_and_local_stability_are_different_properties"]

    headline = section["the_plateau_around_the_published_condition"]
    disputed = section["the_disputed_frame_for_comparison"]
    assert headline["tolerances_in_the_run"] > disputed["tolerances_in_the_run"], (
        "the headline frame no longer has the longer plateau, so this note is stale",
        headline,
        disputed,
    )
    assert set(section["where_the_published_condition_sits_in_each_run"]) == {
        "candidate_scan_and_surface__topdown.png",
        "observed_scan_only__overview.png",
    }


#: The one hash in this record that names no file: it covers a rectangle of pixels
#: inside a rendered frame, so there is nothing on disk to compare it against. Listed by
#: name so the loop below can refuse any *other* unbound hash instead of skipping
#: quietly past it.
NOT_A_FILE_HASH = {"what_is_gated.scale_label_sha256"}


def _sha256_fields(node: object, prefix: str = "") -> dict[str, str]:
    found: dict[str, str] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("sha256") and isinstance(value, str):
                found[prefix + key] = value
            else:
                found.update(_sha256_fields(value, f"{prefix}{key}."))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.update(_sha256_fields(value, f"{prefix}[{index}]."))
    return found


def test_every_file_hash_in_the_capability_record_matches_the_file_it_names() -> None:
    """The record's hashes went stale twice, because nothing compared them to anything.

    Both times a harness edit invalidated a pin and the suite stayed green; both times
    the repair was manual, after the fact. A reviewer demonstrated the gap rather than
    asserting it -- they corrupted the harness and then its output, and this file
    reported 36 passed each time.

    The record is already parsed a few tests above for its prose, so this is a loop over
    an object that was there all along. It would have failed at push time on both
    occasions.
    """
    record = json.loads(
        (EVIDENCE / "capability-record-frame-property-gate.json").read_text()
    )
    entries = record["provenance"]["harness_that_produced_this"]
    assert entries, record["provenance"]

    for entry in entries:
        named = EVIDENCE / entry["path"]
        assert named.is_file(), f"{entry['path']} is pinned but absent"
        committed_at = entry.get("bytes_committed_at")
        if committed_at:
            historical = subprocess.run(
                [
                    "git",
                    "-C",
                    str(EVIDENCE.parents[2]),
                    "show",
                    f"{committed_at}:docs/evidence/open3d/{entry['path']}",
                ],
                check=True,
                capture_output=True,
            ).stdout
            actual = hashlib.sha256(historical).hexdigest()
        else:
            actual = hashlib.sha256(named.read_bytes()).hexdigest()
        assert actual == entry["sha256"], (
            f"{entry['path']} is pinned at {entry['sha256'][:12]} and is actually "
            f"{actual[:12]} at {committed_at or 'the current tree'}, so the record "
            "describes bytes that do not exist at its declared identity"
        )


def test_no_hash_in_the_record_escapes_that_check_unclassified() -> None:
    """A hash added later is the way this gap reopens, so a new one has to be classified.

    Checking the seven entries that exist today would leave the next pin added anywhere
    else in the record enforced by nothing -- which is the original defect, just deferred.
    Every `*sha256` field must therefore either name a file under the entries the test
    above walks, or be listed as deliberately not a file hash.
    """
    record = json.loads(
        (EVIDENCE / "capability-record-frame-property-gate.json").read_text()
    )
    enforced = {
        f"provenance.harness_that_produced_this.[{index}].sha256"
        for index, _ in enumerate(record["provenance"]["harness_that_produced_this"])
    }
    assert _sha256_fields(record).keys() - enforced == NOT_A_FILE_HASH, (
        "a hash in this record is neither checked against a file nor declared as not "
        "being a file hash",
        sorted(_sha256_fields(record).keys() - enforced - NOT_A_FILE_HASH),
    )
