"""Two capability records told a reader which knob to turn. Neither knob works.

Both statements are limitations, which is the honest place for them, and neither record
is at fault: the reconstruction's own non-determinism was measured later. What these
tests hold is that the corrections landed, that they say what was measured, and that the
new record's conclusions follow from its own rows rather than from whichever run wrote
the file -- which is the mistake the record itself documents making.

No Open3D here. The record is read as data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs" / "evidence" / "open3d"
RECORD = EVIDENCE / "declared-reproducibility-causes.json"
HARNESS = EVIDENCE / "harness" / "declared_reproducibility_causes.py"

#: The phrasings that sent a reader to the wrong knob. Each must be gone from the record
#: it was in, or the correction did not happen.
FALSIFIED_ATTRIBUTIONS = {
    "capability-record-distance-bands.json": (
        "because the 300k-point sampling is not seeded"
    ),
    "capability-record-support-crop.json": (
        "Floating-point results vary with container thread count"
    ),
}


def record() -> dict:
    return json.loads(RECORD.read_text())


def claim(name: str) -> dict:
    return record()["claims_examined"][name]


def conditions() -> dict:
    return record()["conditions"]


@pytest.mark.parametrize(
    ("filename", "falsified"), sorted(FALSIFIED_ATTRIBUTIONS.items())
)
def test_the_corrected_record_no_longer_states_the_wrong_cause(
    filename: str, falsified: str
) -> None:
    assert falsified not in (EVIDENCE / filename).read_text(), (
        f"{filename} still attributes its run-to-run variation to a cause this "
        "measurement ruled out"
    )


@pytest.mark.parametrize("filename", sorted(FALSIFIED_ATTRIBUTIONS))
def test_each_corrected_record_points_at_the_measurement(filename: str) -> None:
    """A corrected limitation has to be followable, or the reader is only told less."""

    limitation = json.loads((EVIDENCE / filename).read_text())["limitations"][3]
    assert "declared-reproducibility-causes.json" in limitation
    assert "poisson-figure-reproducibility.json" in limitation


@pytest.mark.parametrize("name", ["distance_bands", "support_crop"])
def test_each_examined_claim_quotes_the_sentence_it_replaced(name: str) -> None:
    """The quote must be the pre-correction text, and must no longer be in the file.

    If it were still there the correction never landed; if it were the new text the
    record would be arguing with itself.
    """

    examined = claim(name)
    cited = REPO / examined["source"]
    assert cited.exists()
    assert examined["text"] not in cited.read_text()
    assert examined["field"] == "limitations[3]"
    assert (
        "no longer in the file it cites"
        in (record()["the_corrections_applied"]["how_to_read_the_quotes_above"])
    )


def test_the_corrections_block_names_both_files_it_edited() -> None:
    changed = record()["the_corrections_applied"]["what_changed"]
    named = {Path(entry["file"]).name for entry in changed}
    assert named == set(FALSIFIED_ATTRIBUTIONS)
    assert all(entry["field"] == "limitations[3]" for entry in changed)


def test_the_seeded_condition_really_held_the_input_still() -> None:
    """The whole argument rests on this. If the input moved, nothing is shown."""

    for label in ("sampling_seeded", "sampling_seeded_and_one_thread"):
        block = conditions()[label]
        assert block["sampling_seeded"] is True
        assert block["input_cloud_is_the_same_bytes_every_repeat"] is True
        assert block["input_cloud_digests_observed"] == 1
        assert block["downsampled_is_the_same_bytes_every_repeat"] is True


def test_the_unseeded_condition_is_the_one_the_records_ran() -> None:
    block = conditions()["as_both_records_ran_it"]
    assert block["sampling_seeded"] is False
    assert block["input_cloud_is_the_same_bytes_every_repeat"] is False
    assert block["input_cloud_digests_observed"] == block["repeats"], (
        "every unseeded repeat should have produced its own cloud, or the condition is "
        "not what it claims"
    )


def test_the_thread_knob_engaged_where_it_was_claimed() -> None:
    """An unmeasured knob is how this lane got a void result once before."""

    limited = conditions()["sampling_seeded_and_one_thread"]
    assert limited["the_thread_knob_engaged"] is True
    assert limited["max_threads_observed_during"] == 1
    assert limited["max_threads_before"] > 1
    # And the condition it is compared against must not have been limited.
    assert conditions()["sampling_seeded"]["the_thread_knob_engaged"] is None
    assert conditions()["sampling_seeded"]["max_threads_observed_during"] > 1


@pytest.mark.parametrize(
    "label",
    ["as_both_records_ran_it", "sampling_seeded", "sampling_seeded_and_one_thread"],
)
@pytest.mark.parametrize("figure", ["unsupported_area_fraction", "surface_area"])
def test_every_recorded_spread_agrees_with_its_own_values(
    label: str, figure: str
) -> None:
    block = conditions()[label][figure]
    values = block["values"]
    assert len(values) == conditions()[label]["repeats"]
    assert block["distinct"] == len(set(values))
    assert block["identical"] is (len(set(values)) == 1)
    assert block["absolute_spread"] == pytest.approx(max(values) - min(values))
    mean = sum(values) / len(values)
    assert block["relative_spread"] == pytest.approx(block["absolute_spread"] / mean)


def test_neither_figure_settled_in_any_condition() -> None:
    """The finding is that no condition made these reproduce. If one did, say so."""

    for label, block in conditions().items():
        for figure in ("unsupported_area_fraction", "surface_area"):
            assert block[figure]["identical"] is False, (
                f"{label} reproduced {figure} exactly, which would change the finding "
                "and must not pass silently"
            )


def test_seeding_is_recorded_as_reducing_the_spread_not_removing_it() -> None:
    """Both halves. Removing it would falsify the correction; not reducing it would
    falsify the record's agreement with the original attribution."""

    unseeded = conditions()["as_both_records_ran_it"]["unsupported_area_fraction"]
    seeded = conditions()["sampling_seeded"]["unsupported_area_fraction"]
    assert seeded["absolute_spread"] < unseeded["absolute_spread"]
    assert seeded["identical"] is False


def test_the_comparisons_are_made_against_every_run_observed() -> None:
    """A spread over four repeats moved enough between runs to falsify a sentence
    derived from one of them. The ranges must cover every run, and say how many."""

    observed = record()["observed_across_independent_runs_of_this_harness"]
    current = conditions()
    pairs = [
        (
            "unseeded_unsupported_area_fraction_absolute_spread",
            current["as_both_records_ran_it"]["unsupported_area_fraction"][
                "absolute_spread"
            ],
        ),
        (
            "fixed_input_surface_area_relative_spread",
            current["sampling_seeded"]["surface_area"]["relative_spread"],
        ),
        (
            "one_thread_surface_area_relative_spread",
            current["sampling_seeded_and_one_thread"]["surface_area"][
                "relative_spread"
            ],
        ),
    ]
    for key, this_run in pairs:
        block = observed[key]
        assert block["runs"] == len(block["values"]) >= 2, (
            f"{key} must carry more than one run, or the range is one measurement "
            "wearing a range's clothes"
        )
        assert this_run in block["values"], (
            f"{key} must include the run that wrote this file"
        )
        assert block["min"] == min(block["values"])
        assert block["max"] == max(block["values"])


def test_the_range_is_quoted_in_the_prose_that_rests_on_it() -> None:
    observed = record()["observed_across_independent_runs_of_this_harness"]
    unseeded = observed["unseeded_unsupported_area_fraction_absolute_spread"]
    prose = claim("distance_bands")["what_holds"]
    assert f"{unseeded['min']:.3e}" in prose
    assert f"{unseeded['max']:.3e}" in prose
    assert f"{unseeded['runs']} independent runs" in prose
    assert "straddling it rather than bounding it" in prose


def test_the_straddle_is_a_straddle() -> None:
    """The prose says the unseeded spread straddles the record's gap rather than
    bounding it. That has to be what the observations show."""

    observed = record()["observed_across_independent_runs_of_this_harness"]
    unseeded = observed["unseeded_unsupported_area_fraction_absolute_spread"]
    gap = observed["the_recorded_gap_being_compared_against"]
    assert unseeded["min"] < gap < unseeded["max"], (
        "if the gap sits outside the observed range, 'straddling' is the wrong word "
        "and the sentence has to change with the measurement"
    )


def test_the_declared_surface_area_figure_is_not_claimed_as_a_bound() -> None:
    observed = record()["observed_across_independent_runs_of_this_harness"]
    fixed = observed["fixed_input_surface_area_relative_spread"]
    declared = observed["the_declared_surface_area_figure"]
    assert fixed["max"] < declared, (
        "every fixed-input run must sit inside the declared figure, or the record's "
        "'what_holds' is wrong and the figure itself needs revisiting"
    )
    assert "a further run could exceed it" in claim("support_crop")["what_holds"]


def test_the_thread_comparison_states_which_way_it_came_out() -> None:
    observed = record()["observed_across_independent_runs_of_this_harness"]
    one = observed["one_thread_surface_area_relative_spread"]
    default = observed["fixed_input_surface_area_relative_spread"]
    prose = claim("support_crop")["what_does_not_hold"]
    expected = "larger in every run" if one["min"] > default["max"] else "overlapping"
    assert expected in prose
    assert "it is not what makes this irreproducible" in prose


def test_the_record_disclaims_what_four_repeats_cannot_support() -> None:
    claims = record()["what_this_does_not_claim"]
    assert any("bound" in c for c in claims)
    assert any("thread count has no effect" in c for c in claims)
    assert any("figures are wrong" in c for c in claims)


def test_the_residue_is_attributed_to_the_record_that_measured_it() -> None:
    where = record()["where_the_residue_comes_from"]
    assert "poisson-figure-reproducibility.json" in where


def test_a_citation_a_reviewer_cannot_open_says_so() -> None:
    """The cited record is on an unmerged branch, so on main the link goes nowhere.

    This lane has already had a provenance pin re-based for naming a commit no reviewer
    could resolve. The disclosure has to stay until the file is actually here, and go
    when it arrives, so neither state is quietly wrong.
    """

    disclosure = record().get("the_record_it_cites_is_not_on_main_yet")
    if (EVIDENCE / "poisson-figure-reproducibility.json").exists():
        assert disclosure is None, (
            "the cited record is present, so the disclosure that it is not is now "
            "false and should be removed"
        )
        return
    assert disclosure, (
        "the cited record is absent from this tree, so the record must say where it is"
    )
    assert "1021" in disclosure
    assert "cannot open it" in disclosure


def test_the_harness_names_the_construction_it_used() -> None:
    generator = record()["generator"]
    assert "ArmadilloMesh" in generator["scene"]
    assert generator["seed"]
    assert HARNESS.read_text().count("set_max_threads") >= 2
