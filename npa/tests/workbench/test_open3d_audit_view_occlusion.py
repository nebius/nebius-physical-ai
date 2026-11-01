"""The audit view's subject has to be visible, not merely inside the frame.

`camera_framing_probe.py` already answers "is the geometry in frame". The view called
"Removed: unsupported surface" needs a different question answered, because it draws the
retained surface in the same pane: a vertex can sit at reach 0.3, comfortably inside the
frame and reported clean by that probe, and be entirely behind opaque geometry.

Measured by ray casting from the camera that ships, 35.2% of the removed surface's area
was visible on a partial scan and 38.2% on a resampled complete mesh. Hiding one entity
by default takes those to 78.5% and 81.8%.

What makes it a view defect rather than a framing defect is that no camera fixes it. Over
24 azimuths at the product's own elevation the removed surface never exceeded 49% visible
even ignoring its own self-occlusion, and the shipped camera was already within three
points of the best of them. The occluder is what is drawn, not where it is drawn from.

These tests run offline against the committed record and the production tables. The
measurement itself needs Open3D and two of its published datasets, so it lives in
`harness/audit_view_occlusion.py` and the figures it produced live in the record.
"""

from __future__ import annotations

import math
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "docs/evidence/open3d/audit-view-occlusion.json"
HARNESS = REPO / "docs/evidence/open3d/harness/audit_view_occlusion.py"

#: Frozen in `program/claims/open3d-audit-view-occlusion/CLAIM.md` before the first
#: measurement. A majority of the thing the view exists to show. The shipped camera
#: failed it at 35.2%, and the gate was not moved to accommodate that.
MINIMUM_VISIBLE_AREA = 0.50


@pytest.fixture
def record() -> dict:
    return json.loads(RECORD.read_text())


def test_every_scene_clears_the_threshold_frozen_before_measuring(record) -> None:
    """The gate, applied to the view as it ships."""

    scenes = [
        row for row in record["scenes"] if row.get("verdict") != "no_removed_surface"
    ]
    assert scenes, "no measured scenes"
    for row in scenes:
        assert row["visible_area_fraction"] >= MINIMUM_VISIBLE_AREA, row["scene"]


def test_the_measurement_is_of_the_view_as_configured_not_an_assumption(record) -> None:
    """The occluder set has to come from the product, or it measures the harness.

    A harness that hardcodes which entities are shown keeps reporting the fixed number
    after someone re-enables the entity that caused the defect. So the row records which
    entities the view shows and which it hides, and both have to agree with the table
    the blueprint builds itself from.
    """
    from npa.workbench.open3d.runner import REMOVED_VIEW, VIEW_GEOMETRY

    view = VIEW_GEOMETRY[REMOVED_VIEW]
    for row in record["scenes"]:
        if row.get("verdict") == "no_removed_surface":
            continue
        assert row["view_contents"] == list(view.contents), row["scene"]
        assert row["view_hidden_by_default"] == list(view.hidden_by_default), row[
            "scene"
        ]
        assert "/world/mesh_unsupported" in row["entities_shown"], row["scene"]
        assert "/world/mesh" not in row["entities_shown"], row["scene"]


def test_the_blueprint_hides_what_the_table_says_to_hide() -> None:
    """One table drives contents, camera geometry and visibility.

    The visibility override used to be a literal in the blueprint call for one view.
    Left there, re-covering a view's subject would look like an edit to what the view
    carries. This asserts the blueprint reads the table, for every view that has an
    entry -- including the per-fragment clouds, whose override moved into it.
    """
    rrb = pytest.importorskip("rerun.blueprint")
    rr = pytest.importorskip("rerun")

    from npa.workbench.open3d.runner import VIEW_GEOMETRY, _blueprint, _view_cameras

    cameras = _view_cameras(
        {
            "fused": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "mesh": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            "unsupported": [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
        },
        [0.0, 0.0, 1.0],
    )
    blueprint = _blueprint(rr, rrb, cameras)

    expected = {
        name: set(view.hidden_by_default)
        for name, view in VIEW_GEOMETRY.items()
        if view.hidden_by_default
    }
    assert expected, "nothing is hidden, so this would pass vacuously"

    found = dict(_view_overrides(blueprint))
    for name, entities in expected.items():
        assert found.get(name) == entities, (name, found.get(name), entities)

    # And nothing is hidden that the table did not ask to hide.
    for name, entities in found.items():
        assert entities == expected.get(name, set()), name


def _view_overrides(node, found=None):
    """Every named view in the blueprint tree, with the entities its overrides hide.

    Walks the structure rather than stringifying it. `str()` of the root container
    renders one line naming the container object and nothing inside it, so a test that
    searched that text found nothing, and would have passed for any blueprint at all --
    including one with the override removed, which is the edit it exists to catch.
    """
    found = [] if found is None else found
    # Rerun stores the `overrides=` argument under `visualizer_overrides`, not under
    # `overrides`. Read from the wrong attribute this returned nothing for every view.
    overrides = getattr(node, "visualizer_overrides", None)
    name = getattr(node, "name", None)
    if overrides and name:
        found.append(
            (
                str(name),
                {
                    str(path)
                    for path, behaviour in dict(overrides).items()
                    if _hides(behaviour)
                },
            )
        )
    for attribute in ("root_container", "blueprint", "contents"):
        child = getattr(node, attribute, None)
        if child is None:
            continue
        for item in child if isinstance(child, (list, tuple)) else [child]:
            if hasattr(item, "visualizer_overrides") or hasattr(item, "contents"):
                _view_overrides(item, found)
    return found


def _hides(behaviour) -> bool:
    """Read the flag's actual value, not any object's repr.

    Rerun wraps it in a component batch whose `str()` is the default object repr with
    an address in it and no value at all, so a text search over that returned False for
    an entity that was genuinely hidden. The value is in the arrow array.
    """
    visible = getattr(behaviour, "visible", None)
    if visible is None:
        return False
    return False in visible.as_arrow_array().to_pylist()


def test_the_retained_surface_is_what_was_hiding_the_subject(record) -> None:
    """Attribution, not just a before and after.

    Each row carries the visibility with the retained surface shown and without it,
    from the same camera, so the improvement is attributable to that one entity rather
    than to anything else that changed in the same commit.
    """
    for row in record["scenes"]:
        if row.get("verdict") == "no_removed_surface":
            continue
        shown = row["visible_with_retained_surface_shown"]
        hidden = row["visible_with_only_its_own_self_occlusion"]
        assert hidden > shown, row["scene"]
        assert row["area_the_retained_surface_hides"] == pytest.approx(hidden - shown)
        # The shipped configuration has to be one of the two, not a third thing.
        assert row["visible_area_fraction"] == pytest.approx(hidden)


def test_a_zero_cannot_be_a_framing_defect_in_disguise(record) -> None:
    """Occlusion and clipping are different failures with different owners.

    A sample behind the eye is a framing defect and `camera_framing_probe.py` owns it.
    Folded into an occlusion number it would read as hidden geometry, so it is counted
    separately and has to be absent for the headline to mean what it says.
    """
    for row in record["scenes"]:
        if row.get("verdict") == "no_removed_surface":
            continue
        assert row["samples_behind_eye"] == 0, row["scene"]


def test_the_instrument_reports_both_extremes(record) -> None:
    """Controls A3 and A4 from the claim, so a number is not a constant.

    Geometry fully behind an occluder must read approximately nothing visible, and
    geometry fully in front of the same occluder from the same eye must read
    approximately everything. One half alone leaves the instrument unfalsifiable.
    """
    controls = record["controls"]
    assert controls["fully_behind_an_occluder"][
        "visible_area_fraction"
    ] == pytest.approx(0.0, abs=1e-6)
    assert controls["fully_in_front_of_it"]["visible_area_fraction"] == pytest.approx(
        1.0, abs=1e-6
    )
    # Both controls must be in frame, or the zero is a clipping result.
    for control in controls.values():
        assert control["samples_behind_eye"] == 0


def test_no_camera_angle_would_have_fixed_this(record) -> None:
    """The reason the fix is the view's contents and not its camera.

    If some azimuth had cleared the gate, changing what the view draws would have been
    the wrong remedy, and the right one would have been a camera. The sweep says
    otherwise, and the claim's stated remedy follows from this row rather than from a
    preference.
    """
    sweep = record["why_the_camera_is_not_the_fix"]
    assert sweep["azimuths_measured"] >= 24
    assert sweep["best_visible_area_fraction"] < MINIMUM_VISIBLE_AREA
    # The shipped camera was already near the best available, so the loss was never
    # attributable to a poorly chosen angle.
    assert (
        sweep["best_visible_area_fraction"] - sweep["shipped_visible_area_fraction"]
    ) < 0.05


def product_offset_direction(np, camera_fit, up: list[float]) -> list[float]:
    """The direction the product places an eye in, from the product's own code.

    Asked rather than reimplemented. The offset is built from `up` and
    `CAMERA_ELEVATION_DEGREES` alone, so fitting any point set returns it -- a unit
    tetrahedron and a larger offset box agree to nine decimals. Reimplementing the
    formula here would pass just as well and would stop tracking the product the moment
    it changed.
    """
    fit = camera_fit(
        np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], float
        ),
        up,
        1.0,
    )
    offset = [fit["eye"][i] - fit["look_target"][i] for i in range(3)]
    span = math.sqrt(sum(value * value for value in offset))
    return [value / span for value in offset]


def test_the_harness_uses_the_product_camera_rather_than_its_own(record) -> None:
    """A measurement of a camera the harness fitted itself proves nothing about the
    one that ships, so the recorded camera has to be the production fit.

    This test was named for that property and did not enforce it. It asserted
    `camera_elevation_degrees > 0.0`, which is satisfied by any positive number --
    including 35.0, the synthetic control eye that already exists in this very harness.
    A reviewer changed the record's 28.0 to 35.0 and the suite still passed 11 of 11.
    `camera_eye` was checked with `len(...) == 3`, which admits any three floats.

    The three other assertions were over harness *source text*, and source text cannot
    substantiate a claim about what the harness actually did.

    What it checks now, in increasing strength:

    * the recorded angle equals the product's own constant, imported rather than
      copied -- which is what the earlier `> 0.0` was reaching for;
    * the eye's *whole direction* from its look target must equal the direction the
      product fits. That direction is obtained by calling the product's own `_camera()`
      on throwaway points, not by reimplementing its formula here: the offset depends
      only on `up` and the constant, never on the geometry, so any point set returns
      it. Two deliberately different ones were checked and agree to nine decimals.

    An earlier version of this test bound only the *elevation* recovered from the eye,
    and argued that two scenes agreeing on it could not both be hand-written. Review
    falsified that directly: they replaced both eyes with invented ones chosen to sit
    at exactly 28 degrees, roughly a thousand units out on a different azimuth, and
    the suite passed 11 of 11. Two forgeries agree perfectly whenever the forger knows
    the constant, and the constant is in the source. Binding the full direction closes
    that: those same two eyes are 0.62 away from the product's direction, because their
    azimuth is free and the product's is not.

    What is still unbound is the *distance* along that direction. Solving it needs the
    point geometry -- it is fitted so the widest actual sample sits just inside the
    frame -- which is exactly why this test binds the direction instead. An eye placed
    at the right direction and the wrong distance would pass here. Said plainly rather
    than implied, because the previous version of this docstring claimed more than it
    could support.

    `camera_look_target` and `camera_up` were added to the record for this. They come
    from the same `_view_cameras()` call as the eye, and the recomputation that added
    them reproduced the recorded eye to 0.0 in both scenes, so no measured value moved.
    """

    import numpy as np

    from npa.workbench.open3d.runner import CAMERA_ELEVATION_DEGREES, _camera

    directions = {}
    for row in record["scenes"]:
        if row.get("verdict") == "no_removed_surface":
            continue
        scene = row["scene"]
        assert row["camera_elevation_degrees"] == CAMERA_ELEVATION_DEGREES, (
            f"{scene} records a camera at {row['camera_elevation_degrees']} degrees, "
            f"but the product ships {CAMERA_ELEVATION_DEGREES}"
        )

        eye = row["camera_eye"]
        target = row["camera_look_target"]
        up = row["camera_up"]
        assert len(eye) == len(target) == len(up) == 3

        offset = [eye[i] - target[i] for i in range(3)]
        span = math.sqrt(sum(value * value for value in offset))
        up_span = math.sqrt(sum(value * value for value in up))
        assert span > 0.0 and up_span > 0.0, (scene, offset, up)
        offset = [value / span for value in offset]
        up = [value / up_span for value in up]

        recovered = math.degrees(math.asin(sum(offset[i] * up[i] for i in range(3))))
        assert abs(recovered - CAMERA_ELEVATION_DEGREES) < 1e-6, (
            f"{scene}'s eye sits {recovered:.6f} degrees above its look target, not "
            f"the {CAMERA_ELEVATION_DEGREES} the product fits and the record claims"
        )

        expected = product_offset_direction(np, _camera, row["camera_up"])
        drift = max(abs(offset[i] - expected[i]) for i in range(3))
        assert drift < 1e-9, (
            f"{scene}'s eye sits {drift:.3e} away from the direction the product "
            f"fits: recorded {[round(v, 6) for v in offset]} against the product's "
            f"{[round(v, 6) for v in expected]}. The elevation alone can be right "
            f"while the azimuth is invented."
        )
        directions[scene] = offset

    assert len(directions) >= 2, f"need two scenes to compare fits, got {directions}"


def test_the_record_states_its_own_stability_rather_than_implying_exactness(
    record,
) -> None:
    """A figure republished to the last digit looks falsified on the next run.

    Seeding the resampled scene was necessary and is not sufficient:
    `orient_normals_consistent_tangent_plane` is multithreaded and its tie-breaking is
    not seeded, so triangles within floating-point reach of the crop limit change which
    side they land on. Two full runs agreed to 2e-5 on every published fraction while
    moving a handful of triangles between kept and removed.

    The gate has four orders of magnitude of margin over that, which is the only reason
    it does not matter -- so the record has to say both things, and this checks it does.
    """
    reproducibility = record["reproducibility"]
    assert reproducibility["sampling_seed"]
    assert "not bit" in " ".join(reproducibility).lower() or any(
        "bit_exact" in key for key in reproducibility
    )
    assert "2e-5" in reproducibility["and_still_not_bit_exact"]

    # And the margin that makes it immaterial has to be real, not asserted.
    for row in record["scenes"]:
        if row.get("verdict") == "no_removed_surface":
            continue
        assert row["visible_area_fraction"] - MINIMUM_VISIBLE_AREA > 0.2, row["scene"]


def numbers_in(node) -> set[float]:
    """Every numeric leaf of the record, booleans excluded."""

    if isinstance(node, dict):
        return {value for item in node.values() for value in numbers_in(item)}
    if isinstance(node, list):
        return {value for item in node for value in numbers_in(item)}
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return set()
    return {float(node)}


#: Percentages quoted about this measurement that the record does not hold as a fraction.
#: Pinned by equality rather than counted, so a figure that *stops* resolving shows up
#: here by name instead of being absorbed into a total. Empty is the honest state: every
#: percentage in scope came from this record.
QUOTED_BUT_NOT_IN_THE_RECORD: set[str] = set()


def quoting_prose() -> list[tuple[str, str]]:
    """The places this measurement is restated for a human, and nowhere else.

    Scoped to the audit view's own table entry rather than all of `runner.py`. Run over
    the whole file this test flagged `11.1%` and `7.4%`, which are real and are bound to
    `pane-fit-window-aspect-sweep.json` by that measurement's own tests. Exempting them
    here would have meant pinning two figures this record cannot check -- a test that
    tolerates drift it cannot see is worse than one with a narrower scope that says so.

    Review measured what the scope leaves exposed: an occlusion percentage written into
    the composite view's comment instead of this one would not be bound. The obvious
    repair is to widen the scan to the whole file and resolve each figure against *any*
    committed record, and that is worse than the exposure. The 40 records under
    `docs/evidence/open3d` hold 1,787 numeric leaves between them, and **50.3% of
    arbitrary `NN.N%` values find a coincidental match** among them -- `11.1%` itself
    resolves against three records that have nothing to do with window aspect. A guard
    that passes half of all drifts while looking authoritative is worse than a narrow
    one that states its edge, so the scope stays and the next reader gets the number
    rather than having to re-derive it.
    """
    runner = (REPO / "npa/src/npa/workbench/open3d/runner.py").read_text().splitlines()
    start = next(i for i, line in enumerate(runner) if "REMOVED_VIEW: _View(" in line)
    end = next(i for i, line in enumerate(runner[start:], start) if line == "    ),")
    block = "\n".join(runner[start : end + 1])

    # A refactor that moves or renames the entry must not silently empty the scan and
    # leave this passing over nothing.
    assert "hidden_by_default" in block or "/world/mesh" in block, block
    return [("runner.py REMOVED_VIEW entry", block), ("this file", __doc__ or "")]


def test_every_percentage_quoted_in_code_or_prose_is_one_the_record_holds(
    record,
) -> None:
    """The defect this exists to stop, which it did not stop the first time.

    This measurement was stated three times -- the record, a comment in `runner.py`, and
    the docstring above -- and the two restatements were both wrong and did not agree
    with each other: 81.8 in the record against 82.5 and 82.1. Review caught it within
    minutes of the commit and pointed out that I already ship this mechanism twice over,
    on #1007 and #1009, and that neither was pointed here.

    So it is pointed here. Every `NN.N%` in those files has to round-trip to a fraction
    the record actually holds, at the precision it is quoted to. The gate threshold is
    excluded by construction: 50% is an acceptance criterion rather than a measurement,
    and it is already bound to `MINIMUM_VISIBLE_AREA` by the tests above.
    """
    held = numbers_in(record)
    unresolved = set()
    resolved = 0

    for where, text in quoting_prose():
        for literal in set(re.findall(r"(?<![\w.])(\d{1,3}\.\d)%", text)):
            value = float(literal) / 100.0
            decimals = len(literal.split(".")[1]) + 2
            if any(
                round(number, decimals) == round(value, decimals) for number in held
            ):
                resolved += 1
            else:
                unresolved.add(f"{literal} ({where})")

    assert resolved >= 8, (
        f"only {resolved} percentages bound; the scan stopped matching"
    )
    assert unresolved == QUOTED_BUT_NOT_IN_THE_RECORD, {
        "no_longer_backed_by_the_record": sorted(
            unresolved - QUOTED_BUT_NOT_IN_THE_RECORD
        ),
        "newly_backed_so_remove_from_the_exception_set": sorted(
            QUOTED_BUT_NOT_IN_THE_RECORD - unresolved
        ),
    }


def test_widening_the_scan_to_every_record_would_be_a_coin_flip(record) -> None:
    """Why `quoting_prose` is narrow, asserted rather than argued.

    Review noted that a figure written outside the scoped block would not be bound, and
    the natural repair is to check every percentage in `runner.py` against every
    committed record. This measures that repair instead of adopting it: across the
    evidence directory a coincidental match is roughly as likely as not, so the widened
    guard would pass about half of all drifts while reading as comprehensive.

    Kept as a test rather than only a comment so that if the record set ever becomes
    sparse enough for the wide check to mean something, this fails and the scope
    decision gets revisited on evidence instead of staying narrow out of habit.
    """
    leaves: set[float] = set()
    for path in sorted((REPO / "docs/evidence/open3d").glob("*.json")):
        leaves |= numbers_in(json.loads(path.read_text()))

    candidates = [tenths / 1000.0 for tenths in range(1000)]
    collisions = sum(
        1
        for value in candidates
        if any(round(leaf, 3) == round(value, 3) for leaf in leaves)
    )
    rate = collisions / len(candidates)

    # Well above the point where "backed by some record" stops being informative. If
    # this drops below a third, widening is worth reconsidering.
    assert rate > 0.33, (
        f"only {rate:.1%} of arbitrary percentages collide with the {len(leaves)} "
        "numeric leaves on record, so a whole-file check against all records may now "
        "be strong enough to replace the scope"
    )

    # The concrete instance behind the rate: a figure this record does not own, matching
    # records that do not own it either.
    pane_fit = 0.111
    unrelated = [
        path.name
        for path in sorted((REPO / "docs/evidence/open3d").glob("*.json"))
        if "pane-fit" not in path.name
        and any(
            round(leaf, 3) == pane_fit
            for leaf in numbers_in(json.loads(path.read_text()))
        )
    ]
    assert unrelated, "11.1% no longer collides; re-measure before trusting this"
