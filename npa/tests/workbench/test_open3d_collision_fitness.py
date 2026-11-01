"""The collision-fitness verdict, and the control that is the only reason to believe it.

These read committed evidence rather than re-running Open3D: the meshes are retained
privately and the harness needs the real bundle. What is testable here is that the
verdict was stated rather than abstained on, that it is backed by an instrument whose
closed-mesh control passes, and that the record keeps the two corrections that came
out of building it.
"""

from __future__ import annotations

import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
VERDICT = EVIDENCE / "collision-fitness-verdict.json"
RECORD = EVIDENCE / "capability-record-collision-fitness.json"


def _verdict() -> dict:
    return json.loads(VERDICT.read_text())


def _record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_verdict_is_stated_not_abstained_on() -> None:
    """The lane brief asks for a verdict; every prior record said "no claim is made".

    Abstention was honest while nothing had been measured. Pinned so the chain cannot
    quietly return to abstaining now that the measurement exists.
    """
    report = _verdict()
    assert report["verdict"] == "not fit for collision use", report["verdict"]
    assert "no claim" not in report["verdict"].lower()
    # The verdict has to carry the reason, not just the label.
    assert "not closed" in report["why"]


def test_the_instrument_passes_a_closed_mesh_control() -> None:
    """Without this the verdict is worthless, so it is the first thing checked.

    A parity test that calls every mesh undefined has measured ray direction, not
    geometry. The control is a sphere Open3D constructs, which is closed by
    construction and must come back with zero undefined cells.
    """
    validation = _verdict()["instrument_validation"]
    assert validation["control_is_watertight"] is True, validation
    assert validation["undefined_cells"] == 0, validation
    assert validation["instrument_is_valid"] is True, validation


def test_the_control_agrees_with_an_independent_open3d_query() -> None:
    """Parity is cross-checked against compute_signed_distance and the analytic volume.

    compute_signed_distance is only defined for closed meshes, so it cannot be the
    primary instrument on an open surface -- but on the control it is a fair second
    opinion, and a third is the sphere's volume in cells.
    """
    validation = _verdict()["instrument_validation"]
    assert validation["parity_agrees_with_signed_distance"] is True, validation
    assert (
        validation["inside_cells_by_parity"]
        == validation["inside_cells_by_signed_distance"]
    )
    # Within a few percent of the analytic estimate; a cell grid will not match exactly.
    analytic = validation["analytic_cell_estimate"]
    assert abs(validation["inside_cells_by_parity"] - analytic) / analytic < 0.05, (
        validation
    )


def test_the_first_broken_instrument_is_recorded_with_its_number() -> None:
    """It reported 521 undefined cells on a closed sphere. That stays in the record.

    The failure mode is reusable: axis-aligned rays against an axis-aligned probe grid
    graze shared edges, so parity flips for reasons unrelated to holes.
    """
    validation = _verdict()["instrument_validation"]
    assert validation["what_the_first_version_reported_here"] == 521, validation
    assert "axis-aligned" in validation["why_it_was_wrong"].lower()

    recorded = _record()["my_first_instrument_was_broken_and_a_control_caught_it"]
    assert "521" in recorded["how_it_surfaced"], recorded


def test_neither_variant_is_presented_as_a_closed_control() -> None:
    """I assumed the baseline was closed, from an existing docstring. It is not.

    Both variants are open, so the synthetic sphere is the only control in the report.
    This pins the correction, because presenting the baseline as closed would make the
    candidate's result look like a trade it never was.
    """
    report = _verdict()
    for name in ("baseline", "candidate"):
        topology = report["variants"][name]["topology"]
        assert topology["is_watertight"] is False, (name, topology)
        assert topology["boundary_edges"] > 0, (name, topology)

    correction = report[
        "i_expected_the_baseline_to_be_the_closed_control_and_it_is_not"
    ]
    assert "Both variants are open" in correction, correction


def test_the_existing_justification_that_named_the_weaker_reason_is_recorded() -> None:
    """consumer_occupancy.py says "one of the two meshes is deliberately not closed".

    That invites the reading that the other is closed. Neither is. The decision it was
    justifying was right, for a stronger reason than the one written down, and the
    record says so rather than silently correcting it.
    """
    note = _verdict()["an_existing_justification_in_this_chain_named_the_weaker_reason"]
    assert "neither is" in note.lower(), note
    assert "right" in note.lower(), note


def test_the_counterintuitive_ordering_is_explained() -> None:
    """Baseline has 8x fewer boundary edges and a worse undefined fraction.

    A reviewer will spot that immediately, so the record explains it: once a surface is
    open at all, parity tracks crossable surface area rather than hole count. Pinned
    together with the direction, so the explanation cannot drift from the numbers.
    """
    report = _verdict()
    baseline = report["variants"]["baseline"]
    candidate = report["variants"]["candidate"]

    assert (
        baseline["topology"]["boundary_edges"] < candidate["topology"]["boundary_edges"]
    )
    assert (
        baseline["inside_outside_is_well_defined"][
            "undefined_fraction_of_probed_volume"
        ]
        > candidate["inside_outside_is_well_defined"][
            "undefined_fraction_of_probed_volume"
        ]
    )
    assert baseline["topology"]["triangles"] > candidate["topology"]["triangles"]

    note = report["why_the_mesh_with_eight_times_fewer_boundary_edges_scores_worse"]
    assert "read as a hole metric" in note, note
    assert "not as a quality score" in note, note


def test_the_verdict_is_corroborated_by_a_second_instrument() -> None:
    """Parity says the query is undefined; the generous reading says it is fabricated.

    Resolving every undefined cell in the mesh's favour still leaves most of the
    inferred solid unobserved, which is the occupancy consumer's finding reached a
    different way. A verdict resting on one instrument would be weaker than this.
    """
    generous = _verdict()["variants"]["candidate"]["if_a_planner_trusted_it_anyway"]
    assert generous["unobserved_share_of_inferred_solid"] > 0.5, generous
    assert generous["unobserved_inferred_solid_volume_m3"] > 0.0, generous

    # And the direction matches the occupancy record: candidate fabricates less than baseline.
    baseline = _verdict()["variants"]["baseline"]["if_a_planner_trusted_it_anyway"]
    assert (
        generous["unobserved_inferred_solid_volume_m3"]
        < baseline["unobserved_inferred_solid_volume_m3"]
    )


def test_the_narrower_fit_output_is_named_with_its_figures() -> None:
    """ "Not fit" on its own is not the deliverable the brief asks for.

    The brief wants an honest narrower useful output, so the verdict points at
    occupancy-band mapping and carries that record's measured phantom rates, and says
    plainly that it is a different question rather than a workaround.
    """
    fit = _verdict()["what_is_fit_instead"]
    assert "downstream-consumer.json" in fit, fit
    assert "0.3245" in fit and "0.0708" in fit, fit
    assert "not a workaround" in fit, fit


def test_the_record_binds_its_harness_and_report_bytes() -> None:
    """Same hash-binding contract the rest of this evidence directory uses."""
    import hashlib

    bound = _record()["provenance"]["harness_that_produced_this"]
    assert len(bound) == 2, bound
    for entry in bound:
        target = EVIDENCE / entry["path"]
        assert target.exists(), entry
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        assert digest == entry["sha256"], (entry["path"], digest, entry["sha256"])
