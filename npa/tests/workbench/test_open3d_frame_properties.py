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
    pixels = original.copy()
    keep = np.zeros(pixels.shape[:2], bool)
    keep[640:700, 0:440] = True
    pixels[~keep] = background().astype(pixels.dtype)
    pixels[352:368, 472:488] = np.array([220, 220, 224], dtype=pixels.dtype)

    path = write(tmp_path, "stranded", pixels)

    assert extent_of(path) >= NAIVE_EXTENT_FLOOR, (
        "the control is only interesting if the naive extent gate would have "
        "passed this frame"
    )
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


def test_the_fixed_annotation_contaminates_the_extent_but_not_the_spread() -> None:
    """The measurement behind the gate's central design decision.

    Removing the annotation block moves the extent reading by a lot and the
    spread reading by almost nothing. That ratio is why one is gated and the
    other is only recorded.
    """
    extent_shifts, spread_shifts = [], []

    for path in sorted(FRAMES.glob("*.png")):
        pixels = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
        mask = content_mask(pixels)
        without = mask.copy()
        without[645:700, 18:435] = False

        def narrower(flags: np.ndarray) -> tuple[float, float]:
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

        with_extent, with_spread = narrower(mask)
        without_extent, without_spread = narrower(without)
        extent_shifts.append(abs(with_extent - without_extent))
        spread_shifts.append(abs(with_spread - without_spread))

    assert max(spread_shifts) < 0.01, (
        "the spread reading is supposed to be insensitive to a small fixed "
        f"annotation, but it moved by {max(spread_shifts):.4f}"
    )
    assert max(extent_shifts) > 0.20, (
        "if the extent reading were no longer contaminated, this gate could be "
        "simplified back to measuring extents"
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
