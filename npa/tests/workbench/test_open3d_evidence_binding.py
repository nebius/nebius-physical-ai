"""The Open3D evidence guards must gate, not merely exist.

Two guards live under `docs/evidence/open3d/harness/`, and until this file neither was run by any
test or CI job. A gate nobody runs is documentation, so the committed set could drift past both of
them without a single check going red. These tests run both against the committed evidence, and
then break each thing the guards claim to catch to show that the claim is true.

The negative controls work on a copy. Each harness resolves its evidence root relative to its own
location, so copying the directory and invoking the copied harness audits the copy, and the
committed set is never mutated by a test run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
BIND = "harness/bind_delivered_frames.py"
AUDIT = "harness/audit_evidence_references.py"


def run(root: Path, harness: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / harness), *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    copy = tmp_path / "open3d"
    shutil.copytree(EVIDENCE, copy)
    return copy


def edit(root: Path, name: str, mutate) -> None:
    path = root / name
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data, indent=2) + "\n")


def test_the_committed_frames_are_bound_to_their_bytes() -> None:
    result = run(EVIDENCE, BIND, "--verify")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_committed_evidence_passes_its_reference_guard() -> None:
    result = run(EVIDENCE, AUDIT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_frame_whose_bytes_changed_is_caught(evidence: Path) -> None:
    frame = evidence / "frames-refit" / "observed_scan_only__overview.png"
    with frame.open("ab") as handle:
        handle.write(b"\x00")

    result = run(evidence, BIND, "--verify")
    assert result.returncode == 1, result.stdout
    assert "observed_scan_only__overview.png" in result.stdout
    # The point of binding: the measurements that describe this frame are now known to describe
    # bytes that are gone, instead of silently continuing to look correct.
    assert "bound to" in result.stdout and "hash to" in result.stdout


def test_two_frames_with_swapped_digests_are_both_caught(evidence: Path) -> None:
    def swap(data: dict) -> None:
        first, second = data["frames"][0], data["frames"][1]
        first["sha256"], second["sha256"] = second["sha256"], first["sha256"]

    edit(evidence, "delivered-frames-manifest.json", swap)

    result = run(evidence, BIND, "--verify")
    assert result.returncode == 1, result.stdout
    assert result.stdout.count("is bound to") == 2, result.stdout
    # A swap is the case a membership check passes and a binding check fails, which this evidence
    # set has been corrected for once already. Both halves must report.
    shared = run(evidence, AUDIT)
    assert shared.returncode == 1, shared.stdout


def test_a_measurement_that_drifts_from_the_record_that_took_it_is_caught(
    evidence: Path,
) -> None:
    def drift(data: dict) -> None:
        data["frames"][0]["content_fraction"] = 0.99

    edit(evidence, "delivered-frames-manifest.json", drift)

    result = run(evidence, BIND, "--verify")
    assert result.returncode == 1, result.stdout
    assert "content_fraction" in result.stdout


def test_a_required_measurement_cannot_be_deleted(evidence: Path) -> None:
    def delete_measurement(data: dict) -> None:
        data["frames"][0].pop("content_fraction")

    edit(evidence, "delivered-frames-manifest.json", delete_measurement)

    result = run(evidence, BIND, "--verify")
    assert result.returncode == 1, result.stdout
    assert "omits required measurement content_fraction" in result.stdout


def test_the_background_sweep_is_bound_to_the_frames_it_measured(
    evidence: Path,
) -> None:
    sweep = json.loads((evidence / "background-choice-sweep-refit.json").read_text())
    assert len(sweep["frame_bindings"]) == len(sweep["frames_measured"]) == 9

    sweep["frame_bindings"][0]["sha256"] = "0" * 64
    (evidence / "background-choice-sweep-refit.json").write_text(
        json.dumps(sweep, indent=2, sort_keys=True) + "\n"
    )

    result = run(evidence, AUDIT)
    assert result.returncode == 1, result.stdout
    assert "background-choice-sweep-refit.json" in result.stdout


def test_a_frame_delivered_without_a_binding_is_caught(evidence: Path) -> None:
    shutil.copy(
        evidence / "frames-refit" / "observed_scan_only__overview.png",
        evidence / "frames-refit" / "unbound_extra_view.png",
    )

    result = run(evidence, BIND, "--verify")
    assert result.returncode == 1, result.stdout
    assert "unbound_extra_view.png" in result.stdout
    assert "does not bind it" in result.stdout


def test_an_equality_claim_that_stops_holding_is_caught(evidence: Path) -> None:
    def break_pair(data: dict) -> None:
        data["image"]["byte_verified_not_label_trusted"]["runner_py_inside_image"] = (
            "0" * 64
        )

    edit(evidence, "capability-record-workflow-replay.json", break_pair)

    result = run(evidence, AUDIT)
    assert result.returncode == 1, result.stdout
    # This pair's only purpose is to be an equality check: it is what distinguishes bytes verified
    # inside the image from a build-arg label someone passed in.
    assert "runner_py_at_commit" in result.stdout
    assert "same bytes" in result.stdout


def test_a_record_contradicting_its_own_conclusion_is_caught(evidence: Path) -> None:
    def lie(data: dict) -> None:
        data["controls"]["committed_report_is_current"]["identical"] = False

    edit(evidence, "derived-report-controls.json", lie)

    result = run(evidence, AUDIT)
    assert result.returncode == 1, result.stdout
    assert "identical" in result.stdout


def test_two_records_disagreeing_about_one_input_are_caught(evidence: Path) -> None:
    def disagree(data: dict) -> None:
        data["inputs"]["fragments"]["cloud_bin_1"] = "f" * 64

    edit(evidence, "capability-record-support-crop.json", disagree)

    result = run(evidence, AUDIT)
    assert result.returncode == 1, result.stdout
    assert "cloud_bin_1" in result.stdout


def test_the_guard_reports_what_it_could_not_check(evidence: Path) -> None:
    """The counts are the honest part, so they have to stay in the output.

    An earlier version of this guard announced that every checkable hash matched while quietly
    skipping most of them. The summary now separates hashes bound to a file, hashes checked as
    equality claims, and hashes nothing here can check.
    """

    result = run(evidence, AUDIT)
    assert result.returncode == 0, result.stdout
    assert "bound to a named file" in result.stdout
    assert "equality claims" in result.stdout
    assert "cannot be bound to anything" in result.stdout


def test_a_reframed_frame_cannot_be_rebound_against_stale_measurements(
    evidence: Path,
) -> None:
    """The gap a review lane found: fresh digests paired with measurements of bytes that are gone.

    Copying measurements rather than recomputing them stops this file and the record that took them
    from drifting apart. It did not stop the record itself from describing different pixels. The
    sequence that used to pass: re-render a frame, re-run the binder, and get a manifest whose
    digests are all correct and whose measurements were taken from the old bytes. Nothing compared
    the two, because the audit record did not say which bytes it had read.
    """
    frame = evidence / "frames-refit" / "candidate_scan_and_surface__topdown.png"
    with frame.open("ab") as handle:
        handle.write(b"\x00")

    # Rebuilding is what anyone would do after re-rendering, and is the step that used to launder
    # the staleness into a healthy-looking manifest.
    result = run(evidence, BIND)
    assert result.returncode != 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "candidate_scan_and_surface__topdown.png" in combined
    assert "Refusing to copy a measurement taken from different bytes" in combined


def test_an_audit_record_that_does_not_say_which_bytes_it_read_is_refused(
    evidence: Path,
) -> None:
    """A measurement that cannot name its input binds to a frame by filename only.

    This is the pre-fix shape of the audit record, so it is also the regression control: if the
    digest is ever dropped from that harness again, binding must fail loudly rather than quietly
    going back to trusting the filename.
    """

    def drop_digests(data: dict) -> None:
        for row in data["frames"]:
            row.pop("sha256", None)

    edit(evidence, "frame-edge-audit-after.json", drop_digests)

    built = run(evidence, BIND)
    assert built.returncode != 0, built.stdout + built.stderr
    assert "without recording which bytes it read" in built.stdout + built.stderr

    checked = run(evidence, BIND, "--verify")
    assert checked.returncode == 1, checked.stdout
    assert "by name only" in checked.stdout


def test_verify_catches_an_audit_record_that_measured_different_bytes(
    evidence: Path,
) -> None:
    def retarget(data: dict) -> None:
        data["frames"][0]["sha256"] = "0" * 64

    edit(evidence, "frame-edge-audit-after.json", retarget)

    result = run(evidence, BIND, "--verify")
    assert result.returncode == 1, result.stdout
    assert "describe bytes that are gone" in result.stdout


def test_the_audit_harness_records_the_digest_of_the_bytes_it_read(
    evidence: Path, tmp_path: Path
) -> None:
    """Pins the harness contract the binder leans on, which mutation testing found unpinned.

    The tests above all feed the binder a committed audit record, so they keep passing if the
    harness stops emitting digests -- the stale record still has them, and the gap only appears the
    next time someone regenerates it. That is exactly the delayed, quiet failure this evidence set
    has been corrected for before, so the harness is pinned directly here.
    """
    import hashlib

    frame = evidence / "frames-refit" / "baseline_surface_only__overview.png"
    report = tmp_path / "audit.json"

    result = run(evidence, "harness/frame_edge_audit.py", str(report), str(frame))
    assert result.returncode == 0, result.stdout + result.stderr

    rows = json.loads(report.read_text())["frames"]
    assert len(rows) == 1, rows
    assert rows[0]["sha256"] == hashlib.sha256(frame.read_bytes()).hexdigest(), (
        "the audit harness must record which bytes it measured; the binder refuses to copy a "
        "measurement that cannot name its input"
    )
