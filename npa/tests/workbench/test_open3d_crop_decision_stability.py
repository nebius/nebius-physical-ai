"""The support crop's decision, as opposed to its average.

`capability-record-support-crop.json` states its reproducibility as "numeric to ~4e-06
relative on surface area". The same record publishes a per-cell downstream figure, and
after #1015 measured the Poisson layers underneath differing by 1-2% of their points
between repeats, the open question was whether the per-element answer a user reads moves
with them. It does not, and the margin is measured rather than asserted.

Measured at two capture densities, because a margin established on a sparse cloud says
nothing about a real capture unless the wobble is shown not to grow with the geometry.

Every figure in that record is a between-run comparison and is expected to move, so
these assert verdicts, the control, and ratios -- never a count.
"""

from __future__ import annotations

import hashlib
import pathlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
RECORD = EVIDENCE / "crop-decision-stability.json"


def _record() -> dict[str, Any]:
    return json.loads(RECORD.read_text())


def _case_keys() -> list[str]:
    return sorted(_record()["cases"])


def _case(key: str) -> dict[str, Any]:
    return _record()["cases"][key]


@pytest.mark.parametrize("key", _case_keys())
def test_the_control_fires_before_any_stability_is_claimed(key: str) -> None:
    """Without this the record is unreadable, and the mistake has precedent here.

    A decision that reproduces proves nothing if nothing upstream moved. A thread
    experiment on this same record reached a clean-looking refutation that way -- the
    lever never engaged, so the measurement had no subject. So the first thing asserted
    is that the input was held fixed and the reconstruction nonetheless varied.
    """
    control = _case(key)["did_the_reconstruction_vary_at_all"]

    assert control["distinct_fused_clouds"] == 1, (
        "the fused cloud moved between repeats, so nothing below can be attributed to "
        "the reconstruction rather than to registration"
    )
    assert control["distinct_normal_sets"] == 1, control
    assert control["poisson_varied_between_repeats"], (
        "the reconstruction came back identical, so the stable decision below is not "
        "evidence of robustness -- it is evidence that nothing was asked of it"
    )
    assert control["distinct_poisson_meshes"] > 1, control


@pytest.mark.parametrize("key", _case_keys())
def test_the_variation_is_real_geometry_and_not_a_reordering(key: str) -> None:
    """Different bytes and different points are not the same claim.

    A reordered vertex list has different bytes and identical content, and a threshold
    applied per vertex is indifferent to it. Only genuinely moved positions could carry
    a vertex across the cut, so only that reading makes the stability result meaningful.
    """
    how = _case(key)["did_the_reconstruction_vary_at_all"][
        "and_whether_that_is_only_reordering"
    ]
    assert how["verdict"] == "genuinely different vertex positions", how
    assert how["largest_set_difference"] > 0, how
    for one in how["per_repeat"]:
        assert one["vertices_in_repeat_1_only"] > 0, one


@pytest.mark.parametrize("key", _case_keys())
def test_the_decision_a_user_reads_is_the_same_decision_every_repeat(key: str) -> None:
    """The result, and the reason this record exists.

    Asserted as agreement rather than as a cell count: the counts are between-run
    comparisons on a reconstruction that is not reproducible, and pinning one would
    fail for exactly the thing being documented.
    """
    agreement = _case(key)["the_per_cell_decision_a_user_reads"]
    assert agreement, "no repeats were compared"
    for one in agreement:
        assert one["cells_whose_verdict_flipped"] == 0, (
            f"{key} repeat {one['repeat']}: {one['cells_whose_verdict_flipped']} cells "
            "changed their mind about being unsupported, which contradicts the "
            "record's own conclusion"
        )
        assert one["jaccard"] == 1.0, one


@pytest.mark.parametrize("key", _case_keys())
def test_the_observed_separation_is_published_next_to_the_cut(key: str) -> None:
    """A result that holds by orders of magnitude, stated as the ratio.

    "The decision is stable" is only useful with the distance it is stable by, so the
    record carries both the observed movement and its size relative to the cut. What
    that ratio is *not* is checked separately, below: it is not a clearance.
    """
    far = _case(key)["did_the_reconstruction_vary_at_all"]["and_how_far_it_moved"]

    worst = far["worst_displacement_m"]
    limit = far["the_threshold_it_would_have_to_cross_m"]
    assert worst > 0.0, "nothing moved, so there is nothing to compare to the cut"
    assert limit > 0.0, far
    assert far["cut_over_the_worst_separation_here"] == pytest.approx(
        limit / worst, rel=1e-9
    )
    assert far["cut_over_the_worst_separation_here"] > 100.0, (
        f"{key}: the largest observed separation {worst} is within two orders of "
        f"magnitude of the {limit} cut"
    )


@pytest.mark.parametrize("key", _case_keys())
def test_the_margin_is_quoted_from_the_tail_and_not_from_the_middle(key: str) -> None:
    """The margin's input is a tail statistic, which an independent lane demonstrated.

    Their sweep hit one per-pair worst three orders of magnitude above its neighbours
    and then failed to reproduce it three times on a byte-identical file. A margin
    computed from a handful of repeats therefore describes the middle of a distribution
    and not the case it has to survive, so the headline figure has to be the worst value
    known anywhere -- including an observation this run did not reproduce.
    """
    far = _case(key)["did_the_reconstruction_vary_at_all"]["and_how_far_it_moved"]
    distribution = far["the_distribution_of_the_per_pair_worst"]
    elsewhere = far["the_tail_observed_elsewhere"]

    assert distribution["pairs_compared"] >= 20, (
        f"{distribution['pairs_compared']} pairs cannot describe the tail of a "
        "statistic whose known worst case is three orders of magnitude from its median"
    )
    for field in ("min_m", "median_m", "p90_m", "max_m"):
        assert distribution[field] > 0.0, (field, distribution)

    # Retained rather than dropped: it is the observation that bounds the claim.
    assert elsewhere["reproduced_in_three_further_attempts"] is False
    assert elsewhere["cells_that_flipped_there"] == 0
    assert elsewhere["cut_over_displacement"] == pytest.approx(
        elsewhere["cut_m"] / elsewhere["displacement_m"], rel=1e-9
    )
    assert elsewhere["cut_over_displacement_is"] == "cut_m / displacement_m"

    quoted = far["cut_over_the_worst_separation_known_anywhere"]
    limit = far["the_threshold_it_would_have_to_cross_m"]
    worst_known = max(far["worst_displacement_m"], elsewhere["displacement_m"])
    assert far["the_worst_displacement_known_anywhere_m"] == pytest.approx(
        worst_known, rel=1e-9
    )
    assert quoted == pytest.approx(limit / worst_known, rel=1e-9), (
        "the published ratio is not this case's cut over the worst displacement known "
        "anywhere. Taking the smaller of two *ratios* instead carried the 5 cm case's "
        "cut into a case with a 2 cm one and published 22.9x where 9.1x is correct"
    )
    assert quoted > 1.0, (
        f"at the worst displacement on record the cut is only {quoted}x it, so the "
        "movement is of the same order as the threshold"
    )


def test_the_unreproduced_tail_is_bounded_rather_than_dismissed() -> None:
    """Zero occurrences is a rate bound, and a rate bound is not an absence.

    Both halves have to be in the record. Reporting only the bound turns an observed
    event into a non-event; reporting only the event discards what 46 pairs establish
    about how rare it is.
    """
    bound = _record()["does_the_tail_reproduce_here"]

    assert bound["did_it_recur"] is False
    assert bound["pairs_compared_here"] >= 40, bound
    assert 0.0 < bound["upper_bound_on_its_per_pair_rate"] < 0.2, bound
    assert bound["upper_bound_on_its_per_pair_rate"] == pytest.approx(
        1.0 - 0.05 ** (1.0 / bound["pairs_compared_here"]), rel=1e-9
    ), "the bound does not follow from the pair count it claims to be computed from"
    assert bound["how_many_times_larger_that_event_is"] > 1.0, bound
    assert "a bound is not an absence" in bound["what_this_does_and_does_not_settle"]


@pytest.mark.parametrize("key", _case_keys())
def test_the_published_figure_is_checked_the_way_a_user_runs_it(key: str) -> None:
    """One arm attributes, the other is the user's situation. Both are required.

    Holding the fused cloud fixed is what makes the result attributable to Poisson, and
    it is not what anyone runs. The first version of this record quoted the figure as
    exactly reproducible on the strength of that arm alone, and a later invocation
    returned a different number.
    """
    downstream = _case(key)["the_published_downstream_figure"]
    fixed = downstream["phantom_cells_removed_by_the_crop"]
    rebuilt = downstream["with_the_fused_cloud_rebuilt_instead"]

    assert rebuilt["invocations"] >= 3, rebuilt
    assert rebuilt["distinct_fused_clouds"] == rebuilt["invocations"], (
        "the rebuilt arm did not actually rebuild: two invocations produced the same "
        "fused cloud, so registration was not varying and this arm adds nothing"
    )

    # The attribution sentence has to follow from what the two arms measured.
    together = downstream["what_the_two_arms_together_show"]
    moving = rebuilt["phantom_cells_removed"]
    if fixed["distinct"] == 1 and moving["spread"] > 0:
        assert "not exactly reproducible end to end" in together, together
        assert "belongs to the" in together and "registration" in together, together
    elif fixed["distinct"] > 1:
        assert "not exactly reproducible even with the fused cloud held fixed" in (
            together
        ), together


def test_both_density_ratios_run_in_the_same_direction() -> None:
    """Two sibling ratios silently disagreed about which case was the numerator.

    `vertices_grew_by_factor` was the denser over the sparser; the displacement one was
    max over min of its own values, which happened to be the sparser over the denser.
    The pair was then read as "the denser mesh moved its vertices further" when the
    measurement says it moved them less far.

    The previous guard asserted `displacement < vertices`, which holds either way
    round, so it could not see the inversion. This recomputes both from the per-case
    figures instead of trusting either field.
    """
    survived = _record()["does_the_margin_survive_a_denser_capture"]
    vertices = survived["vertices_per_case"]
    displacement = survived["worst_displacement_per_case_m"]

    sparser, denser = sorted(vertices, key=vertices.get)
    assert survived["vertices_denser_over_sparser"] == pytest.approx(
        vertices[denser] / vertices[sparser], rel=1e-9
    )
    assert survived["displacement_denser_over_sparser"] == pytest.approx(
        displacement[denser] / displacement[sparser], rel=1e-9
    ), (
        "the displacement ratio is not the denser case over the sparser one, so it "
        "runs in the opposite direction to the vertex ratio beside it and the two "
        "cannot be read together"
    )
    # Naming both cases is not enough -- the numerator has to come first, or the
    # disclosure reads correct while describing the inverted ratio.
    disclosed = survived["both_ratios_are"]
    assert disclosed.index(denser) < disclosed.index(sparser), (
        f"the disclosure puts {sparser} before {denser}, so it describes the sparser "
        f"case as the numerator while the figures use the denser one: {disclosed}"
    )


def _harness():
    """The harness module itself, so its generators can be called with stated inputs.

    Reading a generated sentence out of the record only ever exercises the one input the
    record happens to hold. Review's B981 lived in a band the record does not visit.
    """
    path = EVIDENCE / _record()["provenance"]["harness"]
    spec = importlib.util.spec_from_file_location("crop_decision_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A phrase that must appear for each band, stated here rather than recomputed from the
# generator's comparison. The previous guards branched on `displacement < vertices` and
# then asserted the string that branch produces, so they ratified the boundary instead of
# checking it -- and the boundary was wrong across a wide interval.
_EXPECTED_READING = {
    "shrinks": "fell as the capture got denser",
    "fixed": "held roughly constant",
    "proportional": "scaled with the capture",
    "faster_than_the_capture": "grew faster than the capture",
    "partial": "neither reading is supported",
    "indeterminate": "too close together",
}


# (vertex ratio, displacement ratio) -> band, chosen by hand. The 4.2-against-4.2937 row
# is the case that exposed the two-way branch: it is proportional scaling to within two
# percent and was being published as a fixed-magnitude artefact. The 2.1 row is the one I
# checked by hand and accepted the wrong answer for.
_STATED_BANDS = (
    (4.2937, 0.5484, "shrinks"),
    (4.2937, 1.0, "fixed"),
    (4.2937, 1.05, "fixed"),
    (4.2937, 2.1, "partial"),
    (4.2937, 3.5, "partial"),
    (4.2937, 4.2, "proportional"),
    (4.2937, 4.2937, "proportional"),
    (4.2937, 6.0, "faster_than_the_capture"),
    (1.02, 1.0, "indeterminate"),
)


def test_each_pair_of_ratios_lands_in_the_band_stated_for_it() -> None:
    """The bands, against a table written out by hand.

    Review found the two-way branch handed the whole interval between 1.0 and the vertex
    ratio to "fixed-magnitude artefact", so a displacement ratio of 4.2 against a vertex
    ratio of 4.2937 -- proportional to within two percent -- published as fixed. Both
    guards missed it because both recomputed the generator's own comparison and then
    asserted the string it produces.
    """
    harness = _harness()
    for vertices, displacement, expected in _STATED_BANDS:
        got = harness._how_displacement_scales(vertices, displacement)
        assert got == expected, (vertices, displacement, got, expected)

        said = harness._what_density_establishes(vertices, displacement, 1.0e-05)
        assert _EXPECTED_READING[expected] in said, (vertices, displacement, said)
        # And no other band's phrase, so a sentence cannot satisfy two readings at once.
        for other, phrase in _EXPECTED_READING.items():
            if other != expected:
                assert phrase not in said, (vertices, displacement, other, said)


def test_no_band_is_wide_enough_to_swallow_the_interval_between_its_neighbours() -> (
    None
):
    """The defect was a band's width, so width is what this measures.

    Walking the displacement ratio across the whole range at a fixed vertex ratio, no
    single band may cover more than half of it. The two-way version gave one band
    everything from 1.0 to 4.29 and the other everything above, which is exactly what
    this rejects.
    """
    harness = _harness()
    vertices = 4.2937
    seen: dict[str, int] = {}
    steps = 400
    for index in range(steps):
        displacement = 0.1 + index * (2.0 * vertices - 0.1) / (steps - 1)
        band = harness._how_displacement_scales(vertices, displacement)
        seen[band] = seen.get(band, 0) + 1

    assert len(seen) >= 4, f"only {len(seen)} bands are reachable: {seen}"
    widest, count = max(seen.items(), key=lambda pair: pair[1])
    assert count <= steps // 2, (
        f"the {widest!r} band covers {count} of {steps} sampled ratios, so it is "
        "absorbing the interval between its neighbours rather than naming a case"
    )


def test_the_directional_claim_is_derived_from_the_direction() -> None:
    """A directional sentence written beside the numbers instead of from them.

    The prose guard on this record only covered the flip-count sentences, so a claim
    about which way the displacement moved sat outside every check. Both readings were
    reachable when this ran, and which one is correct is the entire contribution of the
    density case.
    """
    survived = _record()["does_the_margin_survive_a_denser_capture"]
    ratio = survived["displacement_denser_over_sparser"]
    said = survived["what_that_establishes"]

    if ratio < 1.0:
        assert f"{1.0 / ratio:.2f}x less far" in said, (ratio, said)
    elif ratio > 1.0:
        assert f"{ratio:.2f}x further" in said, (ratio, said)

    # The reading has to follow from the band the generator reports, and the band has to
    # be one of the named ones. Which band each pair of ratios belongs in is checked
    # against a stated table below rather than against the generator's own comparison.
    band = survived["how_the_displacement_scales"]
    assert band in _EXPECTED_READING, band
    assert _EXPECTED_READING[band] in said, (band, said)


def test_the_margin_is_set_by_the_cut_and_not_by_the_capture() -> None:
    """The generalisation, and the first thing a reader with a real capture would ask.

    A margin measured on one sparse cloud is worth nothing if the displacement scales
    with the geometry, because then it is a property of this input rather than of the
    crop. The denser case exists to answer that, and it is asserted as the comparison
    between the two growth factors rather than as either figure alone.
    """
    survived = _record()["does_the_margin_survive_a_denser_capture"]

    vertices = survived["vertices_denser_over_sparser"]
    displacement = survived["displacement_denser_over_sparser"]
    assert vertices > 2.0, (
        f"the two cases differ by only {vertices}x in vertex count, which is too little "
        "to tell whether the displacement tracks the geometry"
    )
    assert displacement < vertices, (
        f"the displacement grew {displacement}x against {vertices}x more vertices, so "
        "it tracks the capture and the margin reported here does not transfer to a "
        "denser scan"
    )

    # Not `in {"shrinks", "fixed"}`. That asserted the answer, so the run that
    # disagreed failed this file instead of being published by it, and on three runs
    # of these same two settings the band has been shrinks, shrinks and partial. What
    # must hold is that the reading the record prints follows from the ratio it
    # measured, and that the comfortable readings are the only ones allowed to carry
    # the "set by the cut" claim.
    harness = _harness()
    band = survived["how_the_displacement_scales"]
    assert band == harness._how_displacement_scales(vertices, displacement), (
        "the band the record publishes is not the band its own two ratios select"
    )
    reading = harness._SCALING_READINGS[band]
    assert reading in survived["what_that_establishes"]
    if band not in {"shrinks", "fixed"}:
        assert "neither reading is supported" in reading, (
            f"the band is {band}, which does not support the margin being a property "
            "of the crop, and the record does not say so"
        )
    assert "two operating points" in survived["what_that_establishes"], (
        "the conclusion reads as a statement about capture density, which this "
        "comparison cannot make"
    )


def test_the_aggregate_cannot_stand_in_for_the_per_element_answer() -> None:
    """Both are reported, because reporting only the first is the defect in scope.

    The support-crop record describes its reproducibility with an aggregate, which is
    true and does not answer the question a user asks. This record has to carry both or
    it repeats the omission it was written to close.
    """
    for key in _case_keys():
        case = _case(key)
        assert case["the_aggregate_the_record_reports"]
        assert case["the_per_cell_decision_a_user_reads"]
        for field in ("removed_vertex_count", "removed_fraction"):
            assert field in case["the_aggregate_the_record_reports"], (key, field)

        # The figure the support-crop record publishes per-cell, carried through.
        downstream = case["the_published_downstream_figure"]
        assert downstream["phantom_cells_removed_by_the_crop"]["distinct"] == 1, (
            f"{key}: the published downstream cell count moved between repeats"
        )


def test_the_record_binds_the_harness_that_produced_it() -> None:
    provenance = _record()["provenance"]
    assert "this evidence directory" in provenance["path_is_relative_to"]
    harness = EVIDENCE / provenance["harness"]
    assert harness.exists(), harness
    assert (
        hashlib.sha256(harness.read_bytes()).hexdigest() == provenance["harness_sha256"]
    ), "the harness changed without the record being regenerated"


def test_the_two_arms_cannot_drift_onto_different_grids() -> None:
    """A real defect, caught late, worth a guard rather than a comment.

    The second arm open-coded the consumer's cell grid and truncated where the first
    rounds. The two arms were then measuring different grids, the figure moved by about
    fifty cells, and it read as a finding about registration until the arithmetic was
    checked. One builder now serves both, so the guard is that there is only one.
    """
    harness = EVIDENCE / _record()["provenance"]["harness"]
    source = harness.read_text()

    assert source.count("np.meshgrid") == 1, (
        "the cell grid is constructed in more than one place, which is exactly how the "
        "two arms came to measure different grids"
    )
    assert source.count("def _grid(") == 1
    assert source.count("= _grid(cloud)") == 2, (
        "both arms should obtain the grid from the shared builder; one of them is "
        "building its own again"
    )


def test_the_shared_grid_rounds_rather_than_truncates() -> None:
    """The behaviour the structural guard above cannot see.

    That guard counts `np.meshgrid` in the source, which pins the specific defect that
    happened and would pass a second grid built some other way. This one calls the
    builder and checks the decision that actually differed between the two copies:
    rounding the cell count up from 3.7 cells rather than truncating to 3.

    Needs no Open3D -- the builder only asks a cloud for its bounding box.
    """
    harness = EVIDENCE / _record()["provenance"]["harness"]
    spec = importlib.util.spec_from_file_location("crop_decision_probe", harness)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cell = module.CELL
    extent = (cell * 3.7, cell, cell)

    class _Box:
        def get_min_bound(self):
            return [0.0, 0.0, 0.0]

        def get_extent(self):
            return list(extent)

    class _Cloud:
        def get_axis_aligned_bounding_box(self):
            return _Box()

    centres, radius = module._grid(_Cloud())

    assert len(centres) == 4, (
        f"{len(centres)} cells along a 3.7-cell extent: the builder is truncating, "
        "which is exactly the difference that made the two arms measure different "
        "grids and move the published figure by about fifty cells"
    )
    assert radius == pytest.approx(module.SUPPORT_RADIUS_FACTOR * cell)
    assert centres[0][0] == pytest.approx(cell / 2), (
        "cells are not centred in their own extent"
    )


@pytest.mark.parametrize("key", _case_keys())
def test_the_prose_matches_the_measurement_it_describes(key: str) -> None:
    """Both conclusions were reachable when this ran, so the text is generated.

    The first draft hard-coded the instability it expected to find and read as nonsense
    against a zero flip count. The harness now writes the sentence from the outcome, and
    this checks the two have not come apart.
    """
    case = _case(key)
    flips = sum(
        one["cells_whose_verdict_flipped"]
        for one in case["the_per_cell_decision_a_user_reads"]
    )
    rely, benign = case["what_a_user_may_rely_on"], case["whether_that_is_benign"]

    if flips == 0:
        assert "not only the aggregate" in rely, rely
        # This asserted "by a measured margin rather than by luck", which is the claim
        # the record withdrew: nothing here measures distance to the decision boundary.
        # What the no-flip branch owes a reader is the observation it rests on.
        assert "no cell changed its verdict" in benign, benign
        assert "measured margin" not in benign, benign
    else:
        assert "only to a Jaccard of" in rely, rely


def test_the_ratio_that_selects_the_band_is_not_sold_as_reproducible() -> None:
    """Two runs of this harness gave 0.5484 and 0.8889 for the same quantity.

    It is a ratio of two maxima of 23-pair distributions, so it moves between runs by
    more than its own distance from a band edge -- 0.8889 sits 0.011 from the shrinks
    boundary. The conclusion survived because the two bands either side of that edge
    carry the same reading, which is a fact about this comparison and not luck a reader
    should have to notice for themselves.
    """
    survived = _record()["does_the_margin_survive_a_denser_capture"]
    reproducible = survived["is_this_ratio_itself_reproducible"]

    observed = reproducible["the_displacement_ratios_this_record_retains"]
    assert len(observed) >= 2, (
        "only one run is disclosed, so the record cannot show the ratio moving and a "
        "reader has no reason to distrust the single value quoted"
    )
    assert reproducible["runs_this_is_drawn_from"] == len(observed)
    assert survived["displacement_denser_over_sparser"] in observed, (
        "this run's own ratio is missing from the list of ratios observed"
    )

    spread = max(observed) / min(observed)
    assert spread > 1.1, (
        f"the disclosed ratios span only {spread}x, which does not support the "
        "warning the record attaches to them"
    )
    assert "Do not quote it" in reproducible["the_ratio_is_not"]

    # And the conclusion has to be the thing that reproduced, stated as such.
    harness = _harness()
    bands = {
        harness._how_displacement_scales(survived["vertices_denser_over_sparser"], one)
        for one in observed
    }
    # Whether the retained ratios agree is a fact about the runs, not a requirement
    # on them: the third retained ratio lands in a different band from the first two.
    # What the record must do is tell the reader which case it is in.
    if bands <= {"shrinks", "fixed"}:
        assert "support the same conclusion" in reproducible["the_band_is"]
    else:
        assert (
            "establishes nothing until that is resolved"
            in (reproducible["the_band_is"])
        ), (
            f"the retained ratios reach bands {bands} with different readings and the "
            "record still presents a conclusion"
        )
        assert "different bands" in reproducible["the_band_is"]


@pytest.mark.parametrize(
    ("limit", "worst_here", "elsewhere_displacement", "expected"),
    [
        pytest.param(0.05, 1e-06, 2.187e-03, 0.05 / 2.187e-03, id="the 5 cm cut"),
        pytest.param(
            0.02, 1e-06, 2.187e-03, 0.02 / 2.187e-03, id="a cut that is not 5 cm"
        ),
        pytest.param(
            0.02, 5e-03, 2.187e-03, 0.02 / 5e-03, id="here is worse than elsewhere"
        ),
        pytest.param(
            0.1, 2.187e-03, 2.187e-03, 0.1 / 2.187e-03, id="equal displacements"
        ),
    ],
)
def test_the_margin_is_this_cases_cut_over_the_worst_displacement_anywhere(
    limit: float, worst_here: float, elsewhere_displacement: float, expected: float
) -> None:
    """A stated table, so the composition is checked rather than mirrored.

    The row that matters is the second: with a cut that is not 5 cm, composing on the
    ratio and composing on the displacement give different answers (22.86 against
    9.14), and the committed record published the first against a 0.02 m cut named in
    the same sentence. Every other row agrees under both compositions, which is why
    the mirrored assertion passed for a month.
    """

    on_the_displacement = limit / max(worst_here, elsewhere_displacement)
    assert on_the_displacement == pytest.approx(expected, rel=1e-9)

    on_the_ratio = min(limit / worst_here, 0.05 / elsewhere_displacement)
    if limit != 0.05 and worst_here < elsewhere_displacement:
        assert on_the_ratio != pytest.approx(expected, rel=1e-3), (
            "this row is supposed to be the one that distinguishes the two "
            "compositions, and it no longer does"
        )


def test_every_case_publishes_the_margin_against_its_own_cut() -> None:
    """The committed record, case by case, including the one with the 2 cm cut."""

    record = _record()
    elsewhere = None
    for key, case in record["cases"].items():
        far = case["did_the_reconstruction_vary_at_all"]["and_how_far_it_moved"]
        elsewhere = far["the_tail_observed_elsewhere"]
        limit = far["the_threshold_it_would_have_to_cross_m"]
        worst_known = max(far["worst_displacement_m"], elsewhere["displacement_m"])
        quoted = far["cut_over_the_worst_separation_known_anywhere"]
        assert quoted == pytest.approx(limit / worst_known, rel=1e-9), key
        # And the prose, which quoted a figure the arithmetic beside it contradicted:
        # 23 x 2.187e-03 = 0.0503 m, not the 0.02 m cut named in the same sentence.
        assert f"{quoted:.1f}x" in far["what_that_means"], (
            f"{key}: the sentence quotes a margin the record does not compute"
        )
    assert elsewhere is not None and elsewhere["cut_m"] == 0.05, (
        "the tail observation's own cut; if it changes, the reason the ratio was not "
        "portable changes with it"
    )


def test_the_harness_mirrors_the_products_parameters_rather_than_claiming_to() -> None:
    """The harness says "every parameter and every call is the same". Nothing held it.

    The record's harness-sha binding proves the harness did not change, which is the
    wrong direction for this risk: a later change to `ReconstructRequest` defaults or
    to `runner._reconstruct` silently falsifies the record's central claim with no
    failing test. Both sides are readable without Open3D.
    """

    import re

    from npa.workbench.open3d.schemas import ReconstructRequest

    harness = (EVIDENCE / _record()["provenance"]["harness"]).read_text()

    def literal(name: str) -> str:
        found = re.search(rf"^{name} = (.+)$", harness, re.M)
        assert found, f"{name} is no longer a module-level literal in the harness"
        return found.group(1).strip()

    defaults = ReconstructRequest.model_fields
    assert literal("POISSON_DEPTH") == str(defaults["poisson_depth"].default)
    assert literal("DENSITY_QUANTILE") == str(defaults["density_quantile"].default)
    assert literal("SUPPORT_DISTANCE_FACTOR") == str(
        defaults["support_distance_factor"].default
    )

    runner = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src"
        / "npa"
        / "workbench"
        / "open3d"
        / "runner.py"
    ).read_text()
    for call in (
        "radius=voxel * 2.0, max_nn=30",
        "orient_normals_consistent_tangent_plane(k=30)",
    ):
        assert call in " ".join(runner.split()), (
            f"the product no longer calls {call!r}, so the harness's mirror of the "
            "reconstruct stage is stale and the record's claim is false"
        )
    flat = " ".join(harness.split())
    assert "radius=VOXEL * 2.0, max_nn=30" in flat
    assert "orient_normals_consistent_tangent_plane(k=30)" in flat

    consumer = (EVIDENCE / "harness" / "consumer_occupancy.py").read_text()
    assert literal("CELL") in consumer, (
        "the consumer's cell size moved, so the downstream figures here are a near "
        "relative of the published quantity rather than the same one"
    )
    assert literal("SUPPORT_RADIUS_FACTOR") in consumer


def test_every_provenance_block_carries_a_real_digest() -> None:
    """Two of the three used to ship the literal placeholder "recomputed below".

    Invisible to both guards: the evidence audit only inspects values matching
    64 hex characters, so the placeholder was neither verified nor counted as unbound,
    and the binding test read the top-level block alone.
    """

    import hashlib

    record = _record()
    blocks = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "harness_sha256" in node:
                blocks.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(record)
    assert len(blocks) >= 3, (
        f"expected a provenance block per case plus one, got {len(blocks)}"
    )
    for block in blocks:
        harness = EVIDENCE / block["harness"]
        assert harness.exists(), harness
        assert (
            block["harness_sha256"] == hashlib.sha256(harness.read_bytes()).hexdigest()
        ), (
            "a provenance block does not bind the harness bytes -- either a stale "
            "digest or the placeholder shipped again"
        )


def test_the_density_comparison_states_its_confound() -> None:
    """The section is titled "does the margin survive a denser capture". It is one.

    Both cases fuse the same three upstream fragments; the difference is
    `voxel_down_sample` at two settings, and that one knob also moves the normal
    radius, the crop threshold and the comparison cell size. A reader takes "a capture
    four times denser" as a second scan with density isolated, and the two-point ratio
    cannot attribute anything to density.
    """

    section = _record()["does_the_margin_survive_a_denser_capture"]
    confound = section["what_differs_between_the_two_cases"]
    assert "same capture" in confound
    assert "voxel_down_sample" in confound
    assert "does not isolate capture density" in confound
    assert "two operating points" in confound
    established = section["what_that_establishes"]
    assert "two operating points" in established, (
        "the generated conclusion still reads as a statement about density"
    )


def test_the_retained_ratio_history_does_not_claim_to_be_every_run() -> None:
    """ "Every displacement ratio this harness has produced" over a frozen literal.

    The harness never appends to it, so each run replaced the current value and
    discarded the previous one unless a human edited the constant. One more run with a
    third distinct value and the sentence lists 2 of 3 while reading as exhaustive --
    the same shape as the defect the update section says it is fixing.
    """

    ratio = _record()["does_the_margin_survive_a_denser_capture"][
        "is_this_ratio_itself_reproducible"
    ]
    assert (
        "Every displacement ratio this harness has produced"
        not in (ratio["the_ratio_is_not"])
    )
    assert "this record retains" in ratio["the_ratio_is_not"]
    assert "not a log" in ratio["the_ratio_is_not"]
    assert ratio["the_history_is_not_append_only"]
    retained = ratio["the_displacement_ratios_this_record_retains"]
    assert ratio["runs_this_is_drawn_from"] == len(retained)


def test_the_displacement_is_described_as_the_bound_it_is() -> None:
    """A one-sided 1-NN distance was reported as "the largest distance any vertex moved".

    Without vertex correspondence that is a lower bound, and it was computed in one
    direction, which misses a vertex present in one repeat and absent from the other.
    It is symmetric now, and the sentences say which quantity it is.
    """

    for key, case in _record()["cases"].items():
        far = case["did_the_reconstruction_vary_at_all"]["and_how_far_it_moved"]
        assert "nearest-neighbour distance" in far["what_that_means"], key
        for one in far["per_repeat"]:
            assert "symmetric nearest-neighbour" in one["measured"], key
        benign = case["whether_that_is_benign"]
        assert "largest distance any vertex moved" not in benign, key
        assert "lower bound" in benign, key


def test_the_harness_phantom_figure_is_not_sold_as_the_published_one() -> None:
    """Two readings sat in one case object.

    "The published figure is exactly reproducible under the crop" beside "the figures
    are not comparable to the record's published values" -- and the harness builds its
    own fused cloud, so the 733 it gets is its own quantity that happens to equal the
    published one.
    """

    for key, case in _record()["cases"].items():
        both = case["the_published_downstream_figure"][
            "what_the_two_arms_together_show"
        ]
        if "exactly reproducible under the crop" not in both:
            continue
        assert "This harness's own phantom-cells-removed figure" in both, key
        assert "is not the published number re-measured" in both, key
        assert "not comparable" in case["what_this_does_not_establish"], key


@pytest.mark.parametrize("key", _case_keys())
def test_the_cut_to_separation_ratio_is_not_published_as_a_decision_margin(
    key: str,
) -> None:
    """The ratio is two lengths compared. It is not a factor of safety.

    The crop keeps a vertex when its distance to the cloud is at most the cut, so what
    decides a verdict is |distance - cut|. A vertex sitting at 0.049999 m against the
    0.05 m cut is a micron from flipping, and the 2.187e-03 m observation flips it --
    while cut/movement for that pair is 22.86. So the ratio cannot say the decision
    clears anything by that factor, and this record computes no per-vertex clearance
    at all. What supports the verdict is the finite no-flip count.
    """

    far = _case(key)["did_the_reconstruction_vary_at_all"]["and_how_far_it_moved"]

    assert far["how_that_ratio_is_computed"] == (
        "the_threshold_it_would_have_to_cross_m divided by "
        "the_worst_displacement_known_anywhere_m"
    )

    disclaimer = far["what_the_cut_to_separation_ratio_is_not"]
    counterexample = disclaimer["it_is_not_a_decision_margin"]
    assert "0.049999" in counterexample, (
        "the record asserts the ratio is not a margin without showing why; the "
        "arithmetic is the part a reader can check"
    )
    assert "|distance - cut|" in counterexample
    assert "does not say a decision clears its boundary" in counterexample
    assert "not a safety factor" in disclaimer["what_it_is"]
    assert "no cell changed its verdict" in disclaimer["what_does_support_the_verdict"]
    assert "correspondence" in disclaimer["what_would_be_needed_for_a_real_margin"]

    # And the generated prose, which is where the claim actually reached a reader.
    for sentence in (far["what_that_means"], _case(key)["whether_that_is_benign"]):
        flat = " ".join(sentence.split())
        assert "clears" not in flat, (
            f"{key}: a sentence still says the decision clears its threshold: {flat}"
        )
        assert "no cell changed its verdict" in flat, (
            f"{key}: the sentence drops the observation that does support the verdict"
        )
    assert "not as clearance" in " ".join(_case(key)["whether_that_is_benign"].split())


def test_the_counterexample_the_record_states_is_arithmetically_true() -> None:
    """Check the record's own worked example rather than trusting its wording."""

    far = _case("voxel_0.05")["did_the_reconstruction_vary_at_all"][
        "and_how_far_it_moved"
    ]
    elsewhere = far["the_tail_observed_elsewhere"]

    cut = elsewhere["cut_m"]
    movement = elsewhere["displacement_m"]
    distance = 0.049999

    # The ratio the record reports for this pair, and the clearance that actually
    # decides the verdict.
    assert cut / movement == pytest.approx(elsewhere["cut_over_displacement"], rel=1e-9)
    assert cut / movement > 20.0
    assert abs(distance - cut) < movement, (
        "the worked example no longer flips, so the record's counterexample does not "
        "demonstrate what it claims"
    )
    assert distance <= cut < distance + movement, (
        "the vertex has to start inside the cut and end outside it for the flip to be "
        "the point"
    )


def test_the_readme_paragraph_matches_the_record_it_summarises() -> None:
    """The prose drifted from the record across three regenerations before this.

    It said four distinct meshes where the record says 24, a 4-cell rebuild span for
    both cases where they differ, "a factor of 23" for a case whose cut gives 9.1, a
    capture four times denser where both cases are one capture downsampled twice, and
    a ratio of 0.8889 the record no longer retained. Each figure here is read back out
    of the record, so the next regeneration fails this rather than publishing stale
    prose.
    """

    readme = " ".join((EVIDENCE / "README.md").read_text().split())
    record = _record()

    scene = record["cases"]["voxel_0.05"]
    fine = record["cases"]["voxel_0.02"]
    varied = scene["did_the_reconstruction_vary_at_all"]

    assert f"{varied['distinct_poisson_meshes']} distinct meshes" in readme
    assert (
        f"{len(scene['the_per_cell_decision_a_user_reads'])} pair comparisons" in readme
    )

    for case in (scene, fine):
        far = case["did_the_reconstruction_vary_at_all"]["and_how_far_it_moved"]
        ratio = far["cut_over_the_worst_separation_known_anywhere"]
        assert f"{ratio:.1f}x" in readme, (
            f"the README does not quote this case's ratio {ratio:.1f}x; a ratio from "
            "another cut is exactly the defect this section was corrected for"
        )

    for case, label in ((scene, "scene"), (fine, "fine")):
        span = case["the_published_downstream_figure"][
            "with_the_fused_cloud_rebuilt_instead"
        ]["phantom_cells_removed"]["spread"]
        assert f"{span:.0f}-cell span" in readme or f"{span:.0f} cell" in readme, (
            f"the {label} case's rebuild span {span:.0f} is not in the README"
        )

    retained = record["does_the_margin_survive_a_denser_capture"][
        "is_this_ratio_itself_reproducible"
    ]["the_displacement_ratios_this_record_retains"]
    for one in retained:
        assert f"{one:.4f}" in readme, (
            f"the README omits the retained ratio {one:.4f}, which is how the "
            "inconvenient run went missing from the prose last time"
        )

    # And the claim the record withdraws must not survive in the summary.
    assert "withdraws the conclusion" in readme
    assert "not a safety margin" in readme
    assert "the margin is set by the cut, which a caller chooses" not in readme


@pytest.mark.parametrize("key", _case_keys())
def test_the_benign_verdict_claims_the_observation_and_not_a_margin(key: str) -> None:
    """The sentence a reader of this record reaches first.

    It opened with "Yes, and by a measured margin rather than by luck", which this
    record does not measure: the margin would be |distance - cut| per vertex and
    nothing here computes it. Adding the correct explanation further down the same
    string did not make the opening true, and the opening is the part that gets
    quoted. It also closed by offering the no-flip count as something that transfers
    to another scan, which it cannot -- it was counted on this capture, at these
    repeats, with this cut.
    """

    benign = " ".join(_case(key)["whether_that_is_benign"].split())
    pairs = len(_case(key)["the_per_cell_decision_a_user_reads"])

    assert benign.startswith(
        f"Yes, on every comparison made here: across {pairs} pairs no cell changed "
        "its verdict."
    ), f"{key}: the verdict no longer opens with the observation that supports it"
    assert "measured margin" not in benign, (
        f"{key}: the verdict claims a measured margin again"
    )
    assert "not a measurement of how much room the decision had" in benign

    assert f"The {pairs}-pair no-flip count does not transfer" in benign, (
        f"{key}: the no-flip count is offered as transferable again"
    )
    assert "What transfers to another scan is the method" in benign
    assert "a different scan has to produce its own" in benign


def test_the_displacement_docstring_does_not_teach_the_inference_it_refutes() -> None:
    """The docstring taught "safe by a factor nobody needs to think about".

    Comments are out of scope as a rule, and this one is the explanation a future
    author reads before changing the function whose output says the opposite. The
    historical claim is described rather than restated so this check is unambiguous.
    """

    harness = (EVIDENCE / _record()["provenance"]["harness"]).read_text()
    docstring = harness.split("def _how_far_the_movers_move")[1].split('"""')[1]
    flat = " ".join(docstring.split())

    assert "safe by a factor" not in flat
    assert "nobody needs to think" not in flat
    assert "the cut is not the clearance" in flat
    assert "comparison of two lengths" in flat
    assert "lower bound" in flat, (
        "the docstring no longer says what the measurement actually is"
    )
