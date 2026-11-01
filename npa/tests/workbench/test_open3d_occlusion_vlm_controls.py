"""Guards for the occlusion control pair and the judge rows taken against it.

The pair is a fourth defect class for VLM geometry judging. What has to stay true is not
the judges' answers, which are theirs, but that the control is a control: one camera, one
set of geometry bytes, labels derived from measurement, and a frozen criterion reported
as it came out rather than as it was hoped.

Offline against the committed records. The frames and the provider calls that produced
these rows need Open3D, its published Eagle scan, and authorized credentials, so they
live in `docs/evidence/open3d/harness/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def repository_root(start: Path | None = None) -> Path:
    for candidate in (start or Path(__file__).resolve()).parents:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("not inside a repository checkout")


REPO = repository_root()
CONTROLS = REPO / "docs/evidence/open3d/vlm-controls-v4"

#: The control-strength criterion frozen in the claim before anything was measured.
#: It failed at 0.5216. It is still written here as 0.50, because a gate edited to fit
#: its result is not a gate, and a test that reads the result back out of the record it
#: is checking proves nothing.
FROZEN_A2_THRESHOLD = 0.50

#: v3's four judges, reused so these rows are comparable with its D1-D3 rows.
EXPECTED_MODELS = {
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "moonshotai/Kimi-K2.6",
    "moonshotai/Kimi-K3",
    "zai-org/GLM-5.3-Flash",
}


@pytest.fixture(scope="module")
def controls() -> dict:
    return json.loads((CONTROLS / "controls-manifest.json").read_text())


@pytest.fixture(scope="module")
def cases() -> dict:
    return json.loads((CONTROLS / "cases.json").read_text())


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads((CONTROLS / "judge-summary.json").read_text())


@pytest.fixture(scope="module")
def attempts() -> dict:
    return json.loads((CONTROLS / "judge-attempts.json").read_text())


def test_the_pair_differs_in_one_thing_and_it_is_not_the_camera(controls) -> None:
    """A pair whose frames differ in viewpoint cannot attribute an answer to the defect.

    This is the whole validity argument for the control, so it is checked rather than
    asserted in prose: one camera, and the same layer bytes under both frames.
    """
    assert controls["shared_layers_are_identical_bytes"] is True
    camera = controls["camera"]
    for key in ("eye", "look_target", "up"):
        assert len(camera[key]) == 3, camera

    frames = controls["frames"]
    assert set(frames) == {"D4_occluded", "D4_unoccluded"}
    # Same subject geometry behind both, so the footprint denominators are comparable.
    assert (
        frames["D4_occluded"]["subject_pixels_drawn"]
        == frames["D4_unoccluded"]["subject_pixels_drawn"]
    )
    assert frames["D4_occluded"]["sha256"] != frames["D4_unoccluded"]["sha256"]


def test_the_frozen_control_strength_criterion_is_reported_as_failed(
    controls, summary
) -> None:
    """A2 required the occluded footprint below 50% and it measured 52.16%.

    The record has to say so. This is the test most likely to be quietly deleted later,
    because what it pins is an admission: the criterion missed, the threshold stayed, and
    the reason it was the wrong criterion is recorded without becoming an excuse to move
    it. If the measurement ever does clear 50%, this fails and the record gets rewritten
    deliberately instead of drifting.
    """
    measured = controls["measurement"]["occluded_over_unoccluded"]
    assert measured > FROZEN_A2_THRESHOLD, measured

    a2 = summary["acceptance_as_frozen"]["A2_defect_visible_in_pixels_below_50_percent"]
    assert a2["result"] == "FAILED"
    assert a2["measured"] == pytest.approx(measured)

    # The stated requirement has to still name the frozen number. Written as a search
    # for the word "not" in the explanation -- which is what this line was first -- it
    # would have been a grep over prose, the exact defect review caught on #1013 one
    # phase earlier. Comparing the record against a constant this test owns is a check;
    # looking for an English word in a paragraph is not.
    assert f"{FROZEN_A2_THRESHOLD:.0%}" in a2["required"], a2["required"]


def test_the_prediction_was_registered_before_the_calls_and_is_reported_against_them(
    summary, attempts
) -> None:
    """A3 predicted at least 2 of 4 would accept the occluded frame. Four did.

    Checked against the attempts rather than against the sentence, so the claim cannot
    drift from the rows that support it.
    """
    accepted = {
        a["requested_model"]
        for a in attempts["attempts"]
        if a["case_id"] == "D4_occluded"
        and a.get("parsed", {}).get("verdict") == "accept"
    }
    a3 = summary["acceptance_as_frozen"][
        "A3_prediction_at_least_2_of_4_accept_the_occluded_frame"
    ]
    assert set(a3["models_that_accepted_it"]) == accepted
    assert a3["measured"] == f"{len(accepted)} of 4"
    assert len(accepted) >= 2, accepted


def test_the_labels_come_from_measurements_and_not_from_looking(
    cases, controls
) -> None:
    """Every expected value has to trace to a number in a record, by construction.

    The occluded frame's labels follow its measured footprint fraction and its pale-pixel
    count; the unoccluded frame's follow the absence of an occluder. Written by hand they
    would be an opinion about a picture, which is the thing a control exists not to be.
    """
    provenance = cases["label_provenance"]
    measured = controls["measurement"]["occluded_over_unoccluded"]
    assert f"{measured:.4f}" in provenance, provenance
    assert "audit-view-occlusion.json" in provenance

    by_id = {case["id"]: case for case in cases["cases"]}
    assert by_id["D4_occluded"]["expected_verdict"] == "reject"
    assert by_id["D4_unoccluded"]["expected_verdict"] == "accept"
    # Correctly framed and still not reviewable is the entire point of the class.
    assert by_id["D4_occluded"]["expected_framing"] == "reviewable"
    assert by_id["D4_occluded"]["expected_occlusion"] != "not_occluded"


def test_the_boundary_label_discloses_the_instrument_that_disagrees(cases) -> None:
    """`partly_hidden` is a judgement on a boundary, so the record states both readings.

    52% of the footprint survives and 35% of the area does. A control that published the
    area figure and the footprint label would be quietly choosing whichever number made
    the defect look worse.
    """
    disclosed = cases["boundary_disclosed"]
    assert "52" in disclosed and "35" in disclosed, disclosed
    assert "failed" in disclosed.lower()


def test_no_judge_is_qualified_by_this_and_the_record_says_so(
    summary, attempts
) -> None:
    """Qualification is the VLM lane's verdict and v3's negative result stands.

    The failure mode this guards is the easy one: a geometry PR quoting a hosted model's
    `accept` as visual acceptance for itself.
    """
    assert "not_a_qualification" in attempts
    a4 = summary["acceptance_as_frozen"]["A4_not_a_qualification"]
    assert "not_applicable" in a4
    assert "stands" in a4


def test_every_attempt_is_a_real_recorded_call_with_one_try(attempts) -> None:
    """The provenance the contract requires, checked on the rows rather than promised."""

    rows = attempts["attempts"]
    assert len(rows) == 8, len(rows)
    assert {row["requested_model"] for row in rows} == EXPECTED_MODELS

    for row in rows:
        assert row["status"] in {"parsed", "inconclusive"}, row["status"]
        assert row["frame_sha256"] and row["submitted_image_sha256"], row
        # Proves which bytes went out, not merely that a file of that name existed.
        assert row["submitted_image_sha256"] != row["frame_sha256"], row
        assert row["metrics"]["retries"] == 0, row

        if row["status"] == "parsed":
            assert row["returned_model"], row
            assert row["raw_response_sha256"], row
            assert row["usage"] != "unavailable", row

    ids = [row["provider_request_id"] for row in rows if row["status"] == "parsed"]
    assert len(set(ids)) == len(ids), "a repeated request id is one call reported twice"


def test_a_model_that_answered_the_probe_identically_is_not_counted_as_seeing(
    summary,
) -> None:
    """The probe's job is to separate reading the pixels from answering from priors.

    Two of the four said a pale surface was present in both frames, including the one
    where it had been hidden. That is not a wrong answer worth scoring; it is an answer
    that carries no information about whether the image was used. Treating those two
    false accepts as equivalent to the other two would overstate the finding, so the
    record distinguishes them.
    """
    probe = summary["image_dependency_probe"]
    assert set(probe) == EXPECTED_MODELS

    tracked = {model for model, row in probe.items() if row["changed_correctly"]}
    assert tracked, "no model's answer tracked the pixels; the probe found nothing"
    for model, row in probe.items():
        if model in tracked:
            assert row["occluded"] == "yes" and row["unoccluded"] == "no", row
        else:
            assert "no evidence" in row["reading"], row

    assert "weaker evidence" in summary["probe_limitation"]
    assert len(tracked) < len(EXPECTED_MODELS), (
        "all four tracked the pixels, so the limitation text is now wrong"
    )


def test_the_point_radius_is_the_products_own_and_the_reason_is_recorded(
    controls,
) -> None:
    """The first render used a 3x3 point stamp and the pair looked identical.

    Nine times the area of the product's sub-pixel radius, the 796k-point cloud buried
    the retained surface, and drawing or omitting that surface moved 854 pixels of
    691,200. I nearly published that as a finding about the defect; it was a finding
    about my rasterizer. The harness now derives the radius from `runner.py`'s logged
    `voxel * 0.25` and says why in the source.
    """
    harness = (
        REPO / "docs/evidence/open3d/harness/audit_view_vlm_controls.py"
    ).read_text()
    assert "FUSED_POINT_RADIUS_PX = 0" in harness
    assert "voxel * 0.25" in harness and "0.41" in harness

    # The product's radius, so a change there is visible here rather than silent.
    runner = (REPO / "npa/src/npa/workbench/open3d/runner.py").read_text()
    assert "radii=voxel * 0.25" in runner

    assert controls["frames"]["D4_occluded"]["subject_pixels_drawn"] > 1000


def test_the_two_instruments_are_not_presented_as_measuring_the_same_thing(
    controls,
) -> None:
    """Footprint and surface area have different denominators, and the record says so.

    Agreement between two instruments is only worth something if the reader is told what
    each one measures. These two disagree by 17 points on the same defect for a real
    reason, and a record that printed both without that sentence would read as an error.
    """
    measurement = controls["measurement"]
    assert "footprint" in measurement["denominator"]
    assert "area" in measurement["denominator"]
    assert "audit_view_occlusion.py" in measurement["independent_of"]

    harness = (
        REPO / "docs/evidence/open3d/harness/audit_view_vlm_controls.py"
    ).read_text()
    # The self-occlusion asymmetry is the reason the denominators differ at all.
    assert "back faces" in harness


def test_the_finding_is_tied_to_v3s_result_rather_than_announced_alone(
    summary,
) -> None:
    """This is an extension of a negative result, not a fresh discovery.

    v3 found every parsed call accepted its clipped frame. The same four models accept an
    occluded one. Stated without that lineage it would read as novel, and a reader would
    have to rediscover that the VLM lane got there first in three other defect classes.
    """
    finding = summary["the_finding"]
    assert "v3" in finding
    assert "clipped" in finding
    assert summary["limits"].startswith("Two frames")
    assert summary["confusion"]["false_accepts"] == 4
    assert summary["confusion"]["false_rejects"] == 0
