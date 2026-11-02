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
def test_the_margin_is_measured_and_not_the_verdict_itself(key: str) -> None:
    """A result that holds by orders of magnitude, stated as the ratio.

    "The decision is stable" is only useful with the distance it is stable by. A
    displacement of the same order as the cut would make this luck; the record carries
    the ratio so a reader can re-check it against their own scan instead of trusting
    this verdict on a different geometry.
    """
    far = _case(key)["did_the_reconstruction_vary_at_all"]["and_how_far_it_moved"]

    worst = far["worst_displacement_m"]
    limit = far["the_threshold_it_would_have_to_cross_m"]
    assert worst > 0.0, "nothing moved, so there is no margin to report"
    assert limit > 0.0, far
    assert far["margin_factor"] == pytest.approx(limit / worst, rel=1e-9)
    assert far["margin_factor"] > 100.0, (
        f"{key}: the largest vertex movement {worst} is within two orders of magnitude "
        f"of the {limit} cut, so a stable decision here would be luck rather than margin"
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
    assert elsewhere["margin_factor"] == pytest.approx(
        elsewhere["cut_m"] / elsewhere["displacement_m"], rel=1e-9
    )

    quoted = far["margin_at_the_worst_displacement_known_anywhere"]
    assert quoted == pytest.approx(
        min(far["margin_factor"], elsewhere["margin_factor"]), rel=1e-9
    ), (
        "the headline margin is not the worst of what was measured here and what was "
        "measured elsewhere, so it describes the comfortable case"
    )
    assert quoted > 1.0, (
        f"at the worst displacement on record the margin is {quoted}, so a vertex "
        "could reach the cut and the decision is not stable by construction"
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


def test_the_margin_is_set_by_the_cut_and_not_by_the_capture() -> None:
    """The generalisation, and the first thing a reader with a real capture would ask.

    A margin measured on one sparse cloud is worth nothing if the displacement scales
    with the geometry, because then it is a property of this input rather than of the
    crop. The denser case exists to answer that, and it is asserted as the comparison
    between the two growth factors rather than as either figure alone.
    """
    survived = _record()["does_the_margin_survive_a_denser_capture"]

    vertices = survived["vertices_grew_by_factor"]
    displacement = survived["displacement_grew_by_factor"]
    assert vertices > 2.0, (
        f"the two cases differ by only {vertices}x in vertex count, which is too little "
        "to tell whether the displacement tracks the geometry"
    )
    assert displacement < vertices, (
        f"the displacement grew {displacement}x against {vertices}x more vertices, so "
        "it tracks the capture and the margin reported here does not transfer to a "
        "denser scan"
    )
    assert "fixed-magnitude" in survived["what_that_establishes"]


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
        assert "by a measured margin rather than by luck" in benign, benign
    else:
        assert "only to a Jaccard of" in rely, rely
