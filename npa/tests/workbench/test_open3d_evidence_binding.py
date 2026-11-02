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
    assert "declared as such" in result.stdout


DECLARATION = "hashes_that_name_no_file_here"


def test_every_hash_nothing_can_check_says_what_it_is() -> None:
    """The count that drifted, turned into a gate.

    Twelve of the thirty-seven came out of this category by being checked, which is the
    better outcome and the one to reach for first. The twenty-five that remain cannot be
    checked here at all -- a run's artifacts, a module mounted from a dirty tree -- and were
    a number on a passing run. A review lane asked for them to be reduced or justified; one
    record was justified in prose that nothing read, and the rest sat unchanged for a month
    inside a count that never moved.
    """

    result = run(EVIDENCE, AUDIT)
    assert result.returncode == 0, result.stdout
    assert "declared as such" in result.stdout


def test_an_undeclared_unverifiable_hash_is_caught(evidence: Path) -> None:
    """Removing the note is enough to fail, which is what makes it load-bearing."""

    edit(
        evidence,
        "capability-record-standard-runtime.json",
        lambda d: d["independent_verification_of_the_output"].pop(DECLARATION),
    )
    result = run(evidence, AUDIT)
    assert result.returncode == 1
    assert "rrd_sha256" in result.stdout
    assert "no enclosing object declares why" in result.stdout


def test_a_declaration_nothing_can_find_is_prose(evidence: Path) -> None:
    """The exact shape of the month-long drift.

    `replay-b18d-public-report.json` explained its twenty-two hashes all along, under a key
    of its own. The audit reads one key, so a near-miss name left every one of them
    undeclared while the record read as though the question had been answered.
    """

    def rename(data: dict) -> None:
        data["hashes_are_of_the_originals"] = data.pop(DECLARATION)

    edit(evidence, "replay-b18d-public-report.json", rename)
    result = run(evidence, AUDIT)
    assert result.returncode == 1
    assert result.stdout.count("declares why") == 22, result.stdout


def test_a_declaration_reaches_the_hashes_nested_beneath_it(evidence: Path) -> None:
    """One note at the root covers twenty-two hashes several levels down.

    Which is why it has to be the accurate claim rather than a convenient one: this record
    is a sanitized copy of a run's result documents, so every hash in it is of bytes the run
    wrote, with no branch where that stops holding.
    """

    before = run(evidence, AUDIT)
    assert before.returncode == 0, before.stdout

    def bury(data: dict) -> None:
        data["retained_originals"][DECLARATION] = data.pop(DECLARATION)

    edit(evidence, "replay-b18d-public-report.json", bury)
    result = run(evidence, AUDIT)
    assert result.returncode == 1
    # The nine under `retained_originals` stay covered; the thirteen under `derived_fields`
    # no longer are.
    assert result.stdout.count("declares why") == 13, result.stdout
    assert "retained_originals" not in result.stdout


def test_an_empty_declaration_does_not_count(evidence: Path) -> None:
    """Otherwise the cheapest way past the gate is a key with nothing in it."""

    for value in ("", "   ", None, 0, {"why": "because"}):
        edit(
            evidence,
            "capability-record-support-crop.json",
            lambda d, v=value: d["provenance"].update({DECLARATION: v}),
        )
        result = run(evidence, AUDIT)
        assert result.returncode == 1, (value, result.stdout)
        assert "evaluation_plan_hash" in result.stdout, value


def test_a_declaration_does_not_excuse_a_stale_bound_digest(evidence: Path) -> None:
    """Declaring never overrides binding.

    A note covers everything beneath it, so without this the cheapest way to hide a stale
    digest would be one note at a record root. A hash that names a file is hashed and
    compared whatever the note says: binding is the stronger statement and has to win.
    """

    edit(
        evidence,
        "delivered-frames-manifest.json",
        lambda d: d.update({DECLARATION: "everything here is held elsewhere"}),
    )
    frame = next((evidence / "frames-refit").glob("*.png"))
    frame.write_bytes(frame.read_bytes() + b"\0")
    result = run(evidence, AUDIT)
    assert result.returncode == 1
    assert frame.name in result.stdout
    assert "which is actually" in result.stdout


def test_a_declaration_does_not_excuse_an_equality_claim(evidence: Path) -> None:
    """Same rule for the twelve that are checked without a file.

    These were the hashes a note would most plausibly be reached for, since they name no
    file either. They are checked against each other, so a note must not take them out of
    the comparison -- only out of the count of what nothing could check.
    """

    def diverge(data: dict) -> None:
        data[DECLARATION] = "these name no file here"
        verified = data["image"]["byte_verified_not_label_trusted"]
        verified["runner_py_inside_image"] = "f" * 64

    edit(evidence, "capability-record-workflow-replay.json", diverge)
    result = run(evidence, AUDIT)
    assert result.returncode == 1
    assert "runner_py" in result.stdout


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


def test_naming_a_file_that_is_not_here_does_not_escape_the_declaration_gate(
    evidence: Path,
) -> None:
    """The bypass: a path key that resolves to nothing used to be an exemption.

    The gate reached a hash only if the hash named no file at all. Naming *some*
    string that resolves to no file routed it to the "held elsewhere" bucket instead,
    which demanded nothing -- the same hash, the same unverifiability, no justification
    required, and the cheapest green path was therefore to invent a filename rather
    than to declare or to bind.

    Done here to a currently declared hash, with its note removed, so the only thing
    under test is whether the path key confers the exemption.
    """

    def plant(data: dict) -> None:
        held = data["independent_verification_of_the_output"]
        held.pop(DECLARATION)
        held["artifact_hashes"]["rrd"] = "point_cloud.rrd"

    edit(evidence, "capability-record-standard-runtime.json", plant)
    result = run(evidence, AUDIT)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "point_cloud.rrd" in result.stdout
    assert "names one the directory does not hold" in result.stdout


def test_a_run_artifact_name_still_describes_itself(evidence: Path) -> None:
    """And the subset that is exempt stays exempt, so the gate did not just widen.

    `result.json` and the other run-artifact names say what they are by being those
    names; a note repeating it would be ceremony. Asserting that from the committed
    tree is vacuous now that every such hash also sits under a note, so the note is
    removed here: `independent-verification.json`'s `hashes` map holds 21 hashes the
    directory cannot read, 8 of which are run artifacts by name. Without the note the
    other 13 are findings and those 8 are not, which is the boundary. Widen the gate
    to cover them and this fails at 21.
    """

    edit(
        evidence,
        "independent-verification.json",
        lambda d: d["hashes"].pop(DECLARATION),
    )
    result = run(evidence, AUDIT)
    assert result.returncode == 1, result.stdout + result.stderr
    findings = [line for line in result.stdout.splitlines() if "declares why" in line]
    assert len(findings) == 13, result.stdout
    for name in ("result.json", "manifest.json", "status.json"):
        assert not any(
            one.rsplit(" -> ", 1)[-1].startswith(name) for one in findings
        ), f"{name} describes itself and should not need a note: {findings}"


def test_the_declarations_cover_every_hash_the_directory_cannot_check() -> None:
    """The accounting, against the committed tree rather than a count in prose.

    Both gated buckets are empty of undeclared hashes, which is the property the audit
    exits 1 on. Asserted here too so the number moving is visible as a diff in this
    file rather than only as an exit code in a log nobody keeps.
    """

    result = run(EVIDENCE, AUDIT, "--verbose")
    assert result.returncode == 0, result.stdout + result.stderr
    declared_lines = [
        line for line in result.stdout.splitlines() if "declared as such by" in line
    ]
    assert len(declared_lines) == 2, result.stdout
    assert "54 hash(es)" in declared_lines[0], declared_lines
    assert "25 hash(es)" in declared_lines[1], declared_lines


def test_the_control_record_is_bound_to_the_report_it_calls_current(
    evidence: Path,
) -> None:
    """`committed_sha256` was checked against its own twin and against nothing else.

    The pair is compared by `EQUALITY_TRANSFORMS` and against the `identical` boolean,
    both of which hold for any two equal strings. Regenerate the report, update the one
    capability record that carries its digest, and this record kept a stale-but-equal
    pair and stayed internally consistent -- the record whose entire function is "the
    committed report is exactly what the generator produces" no longer describing the
    committed report, with nothing to notice.

    Naming the file makes `bound_pairs` hash it, so both checks now apply.
    """

    def stale(data: dict) -> None:
        current = data["controls"]["committed_report_is_current"]
        stale_digest = "0" * 64
        current["committed_sha256"] = stale_digest
        current["regenerated_sha256"] = stale_digest

    edit(evidence, "derived-report-controls.json", stale)
    result = run(evidence, AUDIT)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "committed_sha256" in result.stdout
    assert "replay-b18d-public-report.json" in result.stdout


def test_the_equality_claim_in_that_control_still_applies(evidence: Path) -> None:
    """Binding it did not displace the check it already had.

    Both halves are now checked two ways -- against the file and against each other --
    and a note cannot suppress either. Breaking only the regenerated half leaves the
    bound half correct, so this fails on the equality claim rather than on the binding.
    """

    def diverge(data: dict) -> None:
        data["controls"]["committed_report_is_current"]["regenerated_sha256"] = "1" * 64

    edit(evidence, "derived-report-controls.json", diverge)
    result = run(evidence, AUDIT)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "regenerated_sha256" in result.stdout


def test_the_replay_record_declares_its_hashes_under_the_key_the_audit_reads() -> None:
    """The second near-miss key, which the review found still in place.

    `capability-record-workflow-replay.json` carried
    `artifact_hashes_are_of_the_originals` -- the same sentence as the real
    declaration, under a name nothing reads. It survived because those hashes sat in
    the ungated bucket, so no gate was asking. Closing that bypass made the key either
    load-bearing or misleading, and it is now the real key, on the object it describes
    rather than at the record root.
    """

    record = json.loads(
        (EVIDENCE / "capability-record-workflow-replay.json").read_text()
    )
    assert "artifact_hashes_are_of_the_originals" not in record
    assert DECLARATION in record["artifact_hashes"]
    assert DECLARATION not in record, (
        "scoped to the hashes it describes, not to the whole record: a root note also "
        "covers every hash added later, which is the limit this gate does not close"
    )


def test_the_control_generator_emits_the_binding_its_record_carries() -> None:
    """The generator cannot be run here, so the two sides are tied by source.

    `derive_public_run_report_control.py` needs the run's retained artifacts, which are
    deliberately not committed, so no test and no CI job executes it. Adding `committed`
    to the record by hand and not to the generator would mean the next real run drops
    the binding and the audit stops checking it -- a guard removed by a regeneration
    that looks like a success.

    Read as source rather than by import because the module reads those artifacts at
    import-adjacent module scope.
    """

    import ast

    source = (EVIDENCE / "harness" / "derive_public_run_report_control.py").read_text()
    tree = ast.parse(source)
    emitted: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "committed_report_is_current"
                and isinstance(value, ast.Dict)
            ):
                emitted = {
                    inner.value
                    for inner in value.keys
                    if isinstance(inner, ast.Constant) and isinstance(inner.value, str)
                }
    assert emitted, "could not find the control block in the generator"
    assert "committed" in emitted, (
        "the generator does not name the report, so a regenerated control record would "
        "carry a pair of digests bound to nothing"
    )

    record = json.loads((EVIDENCE / "derived-report-controls.json").read_text())
    assert set(record["controls"]["committed_report_is_current"]) == emitted
