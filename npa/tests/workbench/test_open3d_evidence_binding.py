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
