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

import ast
import json
import re
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


def test_the_headline_ratio_is_the_one_the_committed_frames_give(controls) -> None:
    """The published number had no reader tying it to the frames beside it.

    A reviewer read this record as internally inconsistent: that the headline
    0.521556785774199 was 3109/5961 while the committed footprint was 5960, making it a
    value carried over from an earlier run. The committed footprint is 5961, that file
    has only ever been written once, and re-running the harness here reproduces 5961 and
    the published ratio to the digit. The record was consistent and the reading was of
    their own fresh run, which produced 5960 in their environment.

    But nothing here checked it, which is why the question could be open at all. So the
    arithmetic now has a reader: the headline must equal the per-frame fraction it is
    defined as, and that fraction must equal the visible-over-drawn counts published
    beside it. A value carried over from an earlier run fails this, which is the defect
    that was suspected and the one this lane keeps finding in its own records -- a number
    with no reader drifts.

    Needs no Open3D, so it runs wherever the suite runs.
    """
    frames, measurement = controls["frames"], controls["measurement"]
    for name, frame in frames.items():
        recomputed = frame["subject_pixels_visible"] / frame["subject_pixels_drawn"]
        assert frame["visible_pixel_fraction"] == recomputed, (
            f"{name}: published fraction {frame['visible_pixel_fraction']} is not "
            f"{frame['subject_pixels_visible']}/{frame['subject_pixels_drawn']} "
            f"= {recomputed}"
        )

    expected = (
        frames["D4_occluded"]["visible_pixel_fraction"]
        / frames["D4_unoccluded"]["visible_pixel_fraction"]
    )
    assert measurement["occluded_over_unoccluded"] == expected, (
        f"the headline {measurement['occluded_over_unoccluded']} is not the ratio the "
        f"committed frames give, {expected}"
    )


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

    # Keyed on whether *any* field moved, not on one field having an expected value.
    # The first version asked `pale_surface_present` to be yes-then-no, which a reviewer
    # disproved as a necessity: a fifth model varied another field, so it read both
    # images, and still answered pale identically. Presence is defensible in both frames.
    tracked = {model for model, row in probe.items() if row["read_the_images"]}
    assert tracked, "no judge varied any field; the probe found nothing"
    assert len(tracked) < len(EXPECTED_MODELS), (
        "every judge varied something, so the limitation this records no longer applies"
    )

    for model, row in probe.items():
        varied = row["fields_that_varied_across_the_pair"]
        # The label has to follow from the fields, not sit beside them.
        assert bool(varied) == row["read_the_images"], (model, row)
        for field in varied:
            assert (
                row["per_field"][field]["occluded"]
                != row["per_field"][field]["unoccluded"]
            ), (model, field, row["per_field"])
        for field, values in row["per_field"].items():
            if field not in varied:
                assert values["occluded"] == values["unoccluded"], (
                    model,
                    field,
                    values,
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


def test_the_mechanism_claim_counts_only_judges_with_evidence_of_reading(
    summary,
) -> None:
    """The headline said four; the evidence beside it supports two.

    `the_finding` originally added that all four "correctly reported the pale occluding
    surface as present, so they saw it", which does not survive the probe limitation
    recorded in the same file: two judges returned byte-identical answers for two
    different images, so nothing in their calls shows they saw anything. Review found the
    two statements contradicting each other.

    The accept count is still four -- that is what the confusion matrix measures. What
    narrows to two is the claim about *why* they accepted.
    """
    probe = summary["image_dependency_probe"]
    evidenced = {model for model, row in probe.items() if row["read_the_images"]}

    assert summary["confusion"]["false_accepts"] == len(probe), (
        "the accept count comes from the calls, not from the probe"
    )
    assert len(evidenced) == 2, evidenced
    assert "so they saw it" not in summary["the_finding"]

    narrowed = summary["how_many_judges_the_mechanism_claim_rests_on"]
    assert str(len(evidenced)) in narrowed.split(",")[0].lower() or "two" in (
        narrowed.split(",")[0].lower()
    ), narrowed


def test_the_weakest_judge_in_that_pair_is_named_with_its_one_field(
    summary,
) -> None:
    """Two judges is not two equal judges, and the difference is worth stating.

    DeepSeek V4.1 Flash varies exactly one field, and it is the contested one; its answer
    moves in the defensible direction, which is real evidence, but it is one field. Kimi
    K3 varies two including `subject_visibility` and is the only judge whose evidence of
    reading does not rest on the field the correction questions.

    Review's note says DeepSeek varied two. It varied one, and the difference is the
    whole point of recording this, so the count is asserted against the probe rather
    than repeated from prose.
    """
    probe = summary["image_dependency_probe"]
    varied = {
        model: row["fields_that_varied_across_the_pair"]
        for model, row in probe.items()
        if row["read_the_images"]
    }
    assert sorted(len(v) for v in varied.values()) == [1, 2], varied

    weakest = min(varied, key=lambda model: len(varied[model]))
    assert varied[weakest] == ["pale_surface_present"], varied

    # The record writes models by display name and the probe keys them by slug, so the
    # two are compared with separators normalised rather than by requiring one spelling.
    def spoken(name: str) -> str:
        return name.rsplit("/", 1)[-1].replace("-", " ").lower()

    assert spoken(weakest) in spoken(summary["where_the_discriminator_is_weakest"]), (
        weakest,
        summary["where_the_discriminator_is_weakest"][:120],
    )

    strongest = max(varied, key=lambda model: len(varied[model]))
    assert "subject_visibility" in varied[strongest], varied


def test_the_frozen_plan_hash_says_where_it_is_retained(summary) -> None:
    """A frozen-criteria hash that names no file is unusable to the reader it exists for.

    It matters most here precisely because A2 failed: the hash is how a reviewer confirms
    the criteria predate the result rather than taking that on trust. Review checked and
    found it matches no file at any of the three commits.

    It is the hash of the claim document in the program's private claim directory, which
    is deliberately not committed. So the record now says that, and reproduces the frozen
    criteria inline so the content is reviewable without the directory.
    """
    assert len(summary["frozen_plan_sha256"]) == 64
    retention = summary["frozen_plan_retention"]
    assert "not committed" in retention or "deliberately not" in retention

    # The point of the disclosure is that the criteria are readable here instead.
    frozen = summary["acceptance_as_frozen"]
    assert frozen, summary.keys()
    text = json.dumps(frozen)
    assert "A2" in text and "A5" in text, text[:200]


def test_the_record_says_which_part_of_the_number_reproduces(controls) -> None:
    """A fifteen-digit figure whose own spread is in the fourth decimal.

    Ten runs across two installs produced eight different pixel counts and no repeated
    render hash, so a reader who reruns this once will not match
    `occluded_over_unoccluded` and had no way to tell a second draw from a bug. The
    statement that fixes that is the harness's own constant, not prose added to the JSON
    afterwards, so a regeneration emits it and this test fails if the two drift apart.

    The band is checked against the published figure rather than taken on faith: a
    statement claiming a range that excludes the number beside it would be worse than no
    statement. It is also checked for not being vacuous, because containing the figure is
    too easy -- widening the claim to 0.1..0.9 in both files passed every other assertion
    here. Vacuity is defined by what the sentence claims rather than by a chosen
    tolerance: the band must clear the frozen criterion, since the sentence says every
    run did, and it must not be more than an order of magnitude wider than the movement
    it describes. A first attempt bounded the width at exactly 1e-3 and rejected a
    perfectly fair 0.521..0.522, because that subtraction comes to 1.0000000000000009e-3
    in binary -- a threshold placed where float representation decides the outcome.

    Needs no Open3D.
    """
    said = controls["reproducibility"]["what_reproduces"]

    harness = CONTROLS.parent / "harness" / "audit_view_vlm_controls.py"
    constant = next(
        ast.literal_eval(node.value)
        for node in ast.parse(harness.read_text()).body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "WHAT_REPRODUCES"
    )
    assert said == constant, (
        "the manifest's statement is no longer the one the harness would write"
    )

    measured = controls["measurement"]["occluded_over_unoccluded"]
    low, high = (
        float(one) for one in re.search(r"(0\.\d+)\.\.(0\.\d+)", said).groups()
    )
    assert low <= measured <= high + 1e-4, (
        f"the claimed band {low}..{high} does not contain the published {measured}"
    )
    assert low > FROZEN_A2_THRESHOLD, (
        f"the claimed band {low}..{high} reaches below the frozen {FROZEN_A2_THRESHOLD} "
        "criterion, so it no longer supports the sentence it is part of"
    )
    assert high - low <= 1e-2, (
        f"the claimed band {low}..{high} is an order of magnitude wider than the "
        "fourth-decimal movement this statement exists to describe"
    )
    assert measured > FROZEN_A2_THRESHOLD, measured
