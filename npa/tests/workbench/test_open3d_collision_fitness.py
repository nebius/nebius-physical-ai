"""The collision-fitness verdict, and the control that is the only reason to believe it.

These read committed evidence rather than re-running Open3D: the meshes are retained
privately and the harness needs the real bundle. What is testable here is that the
verdict was stated rather than abstained on, that it is backed by an instrument whose
closed-mesh control passes, and that the record keeps the two corrections that came
out of building it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
VERDICT = EVIDENCE / "collision-fitness-verdict.json"
RECORD = EVIDENCE / "capability-record-collision-fitness.json"
HARNESS = EVIDENCE / "harness" / "collision_fitness_verdict.py"


def _harness(monkeypatch: pytest.MonkeyPatch):
    """Load the harness with open3d stubbed.

    The gate is a decision about a dict and needs no geometry, but the module imports
    open3d at top level because its measurement functions do. Stubbing keeps these tests
    runnable in CI, which has no Open3D, while still executing the real gate code rather
    than a copy of it.
    """
    monkeypatch.setitem(sys.modules, "open3d", types.ModuleType("open3d"))
    spec = importlib.util.spec_from_file_location("collision_fitness_verdict", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_the_control_gates_on_open3d_agreement_and_independent_analytic_volume() -> (
    None
):
    """Parity is cross-checked against signed distance and gated on analytic volume.

    compute_signed_distance is only defined for closed meshes, so it cannot be the
    primary instrument on an open surface. It also uses ray hit counting, so the
    independent check is the sphere's closed-form volume in cells.
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
    assert validation["analytic_relative_error"] < validation["analytic_tolerance"]


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


def test_the_interior_distance_diagnostic_is_not_called_corroboration() -> None:
    """A surface sensor cannot observe ordinary solid interior.

    Distance from inferred interior cells to observed surface samples therefore cannot
    establish fabrication or independently corroborate the topology verdict.
    """
    generous = _verdict()["variants"]["candidate"]["if_a_planner_trusted_it_anyway"]
    assert "ordinary interior" in generous["measurement_scope"], generous
    assert "not a fabricated" in generous["measurement_scope"], generous
    report = _verdict()
    assert "interior_distance_diagnostic_not_corroboration" in report
    assert "and_if_the_undefined_cells_were_resolved_generously" not in report


def test_historical_input_identity_is_honestly_unbound_and_future_runs_hash_it() -> (
    None
):
    provenance = _verdict()["input_provenance"]
    assert "not bound" in provenance["status"], provenance
    assert "cannot prove" in provenance["consequence"], provenance
    source = HARNESS.read_text()
    assert "hashlib.sha256(path.read_bytes()).hexdigest()" in source
    assert '"open3d_version": o3d.__version__' in source


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


def test_a_failed_control_writes_no_verdict_to_the_published_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review's finding: the report was written three lines before the control was read.

    So a quarantined run left a file at the published name stating a fitness verdict from
    an instrument just proven invalid -- and the consumers of these records are tests that
    read the JSON, which would have passed on it. This executes the real main() with a
    forced-invalid control and asserts the published path stays absent.
    """
    harness = _harness(monkeypatch)
    published = tmp_path / "verdict.json"
    harness.OUT = published
    harness.build = lambda: {
        "verdict": harness.verdict_for({"instrument_is_valid": False}, fit=True),
        "instrument_validation": {"instrument_is_valid": False, "undefined_cells": 521},
    }

    assert harness.main() == 1

    assert not published.exists(), "a failed control must not occupy the published name"
    quarantined = tmp_path / f"verdict{harness.QUARANTINE}.json"
    assert quarantined.exists(), sorted(p.name for p in tmp_path.iterdir())
    assert json.loads(quarantined.read_text())["verdict"] == harness.WITHHELD


def test_the_verdict_field_itself_depends_on_the_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a sibling field two levels down that a reader has to go and find.

    The earlier version derived the verdict from the mesh alone, so the control could fail
    without the verdict changing at all. Both fitness outcomes are checked, because a gate
    that only withholds the unfavourable answer is not a gate.
    """
    harness = _harness(monkeypatch)
    valid = {"instrument_is_valid": True}
    invalid = {"instrument_is_valid": False}

    assert harness.verdict_for(valid, fit=False) == "not fit for collision use"
    assert harness.verdict_for(valid, fit=True) == "fit"
    assert harness.verdict_for(invalid, fit=False) == harness.WITHHELD
    assert harness.verdict_for(invalid, fit=True) == harness.WITHHELD


def test_the_published_artifact_states_a_verdict_only_because_its_control_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The invariant tying the two together on the committed artifact.

    Reads as: this file is allowed to carry a fitness verdict. If the control ever fails,
    this test says the verdict must have become the withheld string.
    """
    harness = _harness(monkeypatch)
    report = _verdict()
    valid = report["instrument_validation"]["instrument_is_valid"]

    assert valid is True, report["instrument_validation"]
    assert report["verdict"] != harness.WITHHELD
    assert harness.publication_path(report, Path("x/verdict.json")) == Path(
        "x/verdict.json"
    )


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


def test_the_record_binds_the_tests_that_pin_it() -> None:
    """This file, by the hash of its own bytes.

    A reviewer found the record named this file without binding it, so everything
    `what_they_pin` claims could stop being true without the record noticing. The
    binding is self-referential on purpose: editing these tests fails this assertion
    until the record is updated, which is the only thing that makes the claim
    maintained rather than a snapshot of one afternoon.

    The path is resolved from the repository root, not from the evidence directory
    like the provenance entries above. That difference cost the reviewer a wrong
    MISSING verdict on their first pass, so the record states its base explicitly and
    this test reads it rather than assuming.
    """
    import hashlib

    bound = _record()["regression_tests"]
    assert "repository root" in bound["path_is_relative_to"], bound
    target = EVIDENCE.parents[2] / bound["file"]
    assert target.exists(), (bound["file"], str(target))
    assert target.resolve() == Path(__file__).resolve(), (
        "the record binds a different file than the one making this assertion"
    )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest == bound["sha256"], (
        f"{bound['file']} is now {digest}, the record says {bound['sha256']}; "
        "update the record's sha256 together with whatever these tests now pin"
    )
