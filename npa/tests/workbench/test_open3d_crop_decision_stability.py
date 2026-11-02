"""The support crop's decision, as opposed to its average.

`capability-record-support-crop.json` states its reproducibility as "numeric to ~4e-06
relative on surface area". The same record publishes a per-cell downstream figure, and
after #1015 measured the Poisson layers underneath differing by 1-2% of their points
between repeats, the open question was whether the per-element answer a user reads moves
with them. It does not, on this input, and the margin is measured rather than asserted.

Every figure in that record is a between-run comparison and is expected to move, so
these assert verdicts, the control, and one ratio -- never a count.
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


def test_the_control_fires_before_any_stability_is_claimed() -> None:
    """Without this the record is unreadable, and the mistake has precedent here.

    A decision that reproduces proves nothing if nothing upstream moved. A thread
    experiment on this same record reached a clean-looking refutation that way -- the
    lever never engaged, so the measurement had no subject. So the first thing asserted
    is that the input was held fixed and the reconstruction nonetheless varied.
    """
    control = _record()["did_the_reconstruction_vary_at_all"]

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


def test_the_variation_is_real_geometry_and_not_a_reordering() -> None:
    """Different bytes and different points are not the same claim.

    A reordered vertex list has different bytes and identical content, and a threshold
    applied per vertex is indifferent to it. Only genuinely moved positions could carry
    a vertex across the cut, so only that reading makes the stability result meaningful.
    """
    how = _record()["did_the_reconstruction_vary_at_all"][
        "and_whether_that_is_only_reordering"
    ]
    assert how["verdict"] == "genuinely different vertex positions", how
    assert how["largest_set_difference"] > 0, how
    for one in how["per_repeat"]:
        assert one["vertices_in_repeat_1_only"] > 0, one


def test_the_decision_a_user_reads_is_the_same_decision_every_repeat() -> None:
    """The result, and the reason this record exists.

    Asserted as agreement rather than as a cell count: the counts are between-run
    comparisons on a reconstruction that is not reproducible, and pinning one would
    fail for exactly the thing being documented.
    """
    agreement = _record()["the_per_cell_decision_a_user_reads"]
    assert agreement, "no repeats were compared"
    for one in agreement:
        assert one["cells_whose_verdict_flipped"] == 0, (
            f"repeat {one['repeat']}: {one['cells_whose_verdict_flipped']} cells "
            "changed their mind about being unsupported, which contradicts the "
            "record's own conclusion"
        )
        assert one["jaccard"] == 1.0, one


def test_the_margin_is_measured_and_not_the_verdict_itself() -> None:
    """A result that holds by four orders of magnitude, stated as the ratio.

    "The decision is stable" is only useful with the distance it is stable by. A
    displacement of the same order as the cut would make this luck; the record carries
    the ratio so a reader can re-check it against their own scan instead of trusting
    this verdict on a different geometry.
    """
    far = _record()["did_the_reconstruction_vary_at_all"]["and_how_far_it_moved"]

    worst = far["worst_displacement_m"]
    limit = far["the_threshold_it_would_have_to_cross_m"]
    assert worst > 0.0, "nothing moved, so there is no margin to report"
    assert limit > 0.0, far
    assert far["margin_factor"] == pytest.approx(limit / worst, rel=1e-9)
    assert far["margin_factor"] > 100.0, (
        f"the largest vertex movement {worst} is within two orders of magnitude of the "
        f"{limit} cut, so a stable decision here would be luck rather than margin"
    )


def test_the_aggregate_cannot_stand_in_for_the_per_element_answer() -> None:
    """Both are reported, because reporting only the first is the defect in scope.

    The support-crop record describes its reproducibility with an aggregate, which is
    true and does not answer the question a user asks. This record has to carry both or
    it repeats the omission it was written to close.
    """
    record = _record()
    assert record["the_aggregate_the_record_reports"]
    assert record["the_per_cell_decision_a_user_reads"]
    for key in ("removed_vertex_count", "removed_fraction"):
        assert key in record["the_aggregate_the_record_reports"], key

    # The downstream figure the support-crop record publishes per-cell, carried through.
    downstream = record["the_published_downstream_figure"]
    assert downstream["phantom_cells_removed_by_the_crop"]["distinct"] == 1, (
        "the published downstream cell count moved between repeats"
    )


def test_the_record_binds_the_harness_that_produced_it() -> None:
    provenance = _record()["provenance"]
    assert "this evidence directory" in provenance["path_is_relative_to"]
    harness = EVIDENCE / provenance["harness"]
    assert harness.exists(), harness
    assert (
        hashlib.sha256(harness.read_bytes()).hexdigest() == provenance["harness_sha256"]
    ), "the harness changed without the record being regenerated"


def test_the_prose_matches_the_measurement_it_describes() -> None:
    """Both conclusions were reachable when this ran, so the text is generated.

    The first draft hard-coded the instability it expected to find and read as nonsense
    against a zero flip count. The harness now writes the sentence from the outcome, and
    this checks the two have not come apart.
    """
    record = _record()
    flips = sum(
        one["cells_whose_verdict_flipped"]
        for one in record["the_per_cell_decision_a_user_reads"]
    )
    rely, benign = record["what_a_user_may_rely_on"], record["whether_that_is_benign"]

    if flips == 0:
        assert "not only the aggregate" in rely, rely
        assert "by a measured margin rather than by luck" in benign, benign
    else:
        assert "only to a Jaccard of" in rely, rely
