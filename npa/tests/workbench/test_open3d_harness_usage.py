"""An evidence harness run without its inputs must say what is missing.

The review contract requires an independent reviewer to "recompute meaningful
measurements" from the evidence. A reviewer who runs one of these harnesses the obvious
way used to get `IndexError: list index out of range` from `sys.argv[1]`, which reads as
a broken harness rather than a missing argument. Four harnesses produced that identical
message, one produced `ValueError: not enough values to unpack`, and one produced
`KeyError: 'INJECT_MODE'`.

This is not hypothetical. The reviewer on this lane reported being unable to reproduce a
measurement and fell back to arithmetic on committed bytes, which is how an uncheckable
figure survived in a record until they caught it another way.

Two harnesses in the same directory already used argparse and already reported usage
correctly, so the fix was to match the neighbour rather than to invent a convention.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = (
    Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d" / "harness"
)

# Every harness that needs an argument, and the word its usage message must name. The
# expected exit status is argparse's 2 for a usage error, which is what the two harnesses
# that already did this produce.
NEEDS_ARGUMENTS = {
    "frame_edge_audit.py": "output",
    "background_choice_sweep.py": "output",
    "camera_framing_probe.py": "artifacts",
    "render_views.py": "plan",
    "preview_from_retained_plys.py": "fused",
    "derive_public_run_report.py": "output",
    "derive_public_run_report_control.py": "artifacts",
}

# `verify_run.py` is deliberately absent. It imports Open3D at module scope, so in an
# environment without Open3D the import fails before argparse can report usage -- and
# that is the right order, because a reviewer missing the module cannot run it at all and
# needs to be told so. Its argument handling is covered by the source-level property
# test below rather than by execution, which is the honest limit of what CI can check.
IMPORTS_OPEN3D_BEFORE_PARSING = "verify_run.py"

# These newer harnesses are byte-pinned by accepted evidence records on main. The
# property tests below must not require editing their generator bytes merely because
# they use an older, locally safe spelling of the same contract.
BARE_SUBSCRIPT_EXCLUSIONS = {
    "register_seed_determinism.py": "length-guarded --omp-child protocol; record-pinned",
}
HARDCODED_OUTPUT_EXCLUSIONS = {
    "pose_graph_optimization.py": "documented OPEN3D_OUT exact-file override; record-pinned",
    "density_crop_contribution.py": "documented OPEN3D_OUT exact-file override; record-pinned",
}
GEOMETRY_READ_EXCLUSIONS = {
    "crop_decision_stability.py": "reads just-produced run_multiway output; record-pinned",
}

# Exceptions that mean "this harness did not understand its own invocation" rather than
# "this input is missing". Any of these in the output is the defect this file exists for.
INTERNAL_FAILURES = (
    "IndexError",
    "KeyError",
    "ValueError",
    "Traceback (most recent call last)",
)


def run(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HARNESS / name), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(("name", "word"), sorted(NEEDS_ARGUMENTS.items()))
def test_a_harness_run_without_arguments_names_what_it_wants(
    name: str, word: str
) -> None:
    result = run(name)
    output = result.stdout + result.stderr

    assert result.returncode == 2, (name, result.returncode, output[-400:])
    assert "required" in output, (name, output[-400:])
    assert word in output, (name, word, output[-400:])
    for failure in INTERNAL_FAILURES:
        assert failure not in output, (name, failure, output[-400:])


def test_the_fault_injector_explains_its_environment_instead_of_raising_keyerror() -> (
    None
):
    """It is loaded with `python -m` before the stage it disturbs, so it has no argv.

    That makes argparse the wrong shape for it, and makes a bare KeyError the easiest
    thing to leave in place. The message has to name the variables and their values,
    because the mode is one of three words that appear nowhere else.
    """
    result = run("inject_truncation.py")
    output = result.stdout + result.stderr

    assert "INJECT_MODE is not set" in output, output[-400:]
    assert "INJECT_LOG" in output, output[-400:]
    for mode in ("raise", "short", "none"):
        assert mode in output, (mode, output[-400:])
    assert "KeyError" not in output, output[-400:]


def test_the_fault_injector_requires_its_log_before_loading_the_real_stage() -> None:
    environment = dict(os.environ)
    environment["INJECT_MODE"] = "none"
    environment.pop("INJECT_LOG", None)
    result = subprocess.run(
        [sys.executable, str(HARNESS / "inject_truncation.py")],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 1, output
    assert "INJECT_LOG is not set" in output, output
    assert "Traceback" not in output, output


def test_no_harness_reads_a_required_input_by_bare_subscript() -> None:
    """The property, rather than the seven instances of it.

    Writing this check is what found the seventh. I had classified `verify_run.py` as
    "needs Open3D" because that was its first error message, so its identical
    `sys.argv[1]` was invisible until the pattern was searched for directly rather than
    inferred from what each harness happened to fail on first.

    `sys.argv[1:]` is excluded: passing a slice on to another parser is not a bare
    subscript and cannot raise IndexError.
    """
    offenders = []
    for path in sorted(HARNESS.glob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            bare_argv = "sys.argv[" in line and "sys.argv[1:]" not in line
            excluded = path.name in BARE_SUBSCRIPT_EXCLUSIONS
            if (bare_argv or "os.environ[" in line) and not excluded:
                offenders.append(f"{path.name}:{number}: {stripped}")

    assert offenders == [], offenders


def test_the_open3d_harness_excluded_from_execution_still_parses_its_argument() -> None:
    """What CI cannot run, it can still read.

    `verify_run.py` cannot reach argparse in this environment, so the execution test
    above skips it on purpose. Dropping it silently would leave the one harness a
    reviewer is most likely to reach for -- it recomputes the delivered hashes --
    unchecked, so its argument handling is asserted from the source instead.
    """
    source = (HARNESS / IMPORTS_OPEN3D_BEFORE_PARSING).read_text()

    assert "import argparse" in source, IMPORTS_OPEN3D_BEFORE_PARSING
    assert "parser.add_argument(" in source, IMPORTS_OPEN3D_BEFORE_PARSING
    assert "parse_args()" in source, IMPORTS_OPEN3D_BEFORE_PARSING
    assert "sys.argv[1]" not in source, IMPORTS_OPEN3D_BEFORE_PARSING


# The three committed frames both runnable harnesses are pinned against, and what they
# measure on them. Values, not key names: a first version of this test asserted only that
# a key was present, and mutation testing showed a changed measurement passed it.
PINNED_FRAMES = (
    "baseline_scan_and_surface__opposite.png",
    "baseline_scan_and_surface__overview.png",
    "baseline_scan_and_surface__topdown.png",
)


def pinned_frame_paths() -> list[Path]:
    frames = [HARNESS.parent / "frames-refit" / name for name in PINNED_FRAMES]
    missing = [p.name for p in frames if not p.exists()]
    assert missing == [], missing
    return frames


def test_the_usage_fix_did_not_move_the_edge_audit_measurement(tmp_path: Path) -> None:
    """The change is to the message, so every number has to be where it was.

    These two harnesses read only committed PNG bytes, which is what makes them runnable
    in CI at all. The Open3D ones cannot be: that image is an unpublished candidate and
    their inputs lived on an ephemeral volume that no longer exists.
    """
    out = tmp_path / "edge-audit.json"
    result = run(
        "frame_edge_audit.py", str(out), *(str(f) for f in pinned_frame_paths())
    )
    assert result.returncode == 0, result.stderr[-400:]

    report = json.loads(out.read_text())
    assert report["clipped_frames"] == [], report["clipped_frames"]
    assert [f["frame"] for f in report["frames"]] == list(PINNED_FRAMES)
    assert [f["is_clipped"] for f in report["frames"]] == [False, False, False]
    assert report["content_tolerance"] == 6, report["content_tolerance"]


def test_the_usage_fix_did_not_move_the_background_sweep_measurement(
    tmp_path: Path,
) -> None:
    out = tmp_path / "background-sweep.json"
    result = run(
        "background_choice_sweep.py", str(out), *(str(f) for f in pinned_frame_paths())
    )
    assert result.returncode == 0, result.stderr[-400:]

    report = json.loads(out.read_text())
    assert report["content_pixels"] == 438598, report["content_pixels"]
    assert report["frames_measured"] == list(PINNED_FRAMES)
    best = report["best_background"]
    assert best["background_luminance"] == 0.0, best
    assert best["p05_contrast"] == pytest.approx(84.1444, abs=1e-4), best
    assert best["median_contrast"] == pytest.approx(173.2888, abs=1e-4), best


def test_no_harness_imports_a_name_the_runner_does_not_have() -> None:
    """Two harnesses could not import at all, which no amount of usage help would fix.

    Both wanted `FABRICATION_AREA_SHARE` from the runner. That name is nowhere in npa/src;
    the constant is `FAR_BAND_AREA_SHARE`. A harness that cannot import cannot recompute
    anything, so this is checked as a property rather than left to the two instances.
    """
    runner = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "npa"
        / "workbench"
        / "open3d"
        / "runner.py"
    ).read_text()

    missing = []
    for path in sorted(HARNESS.glob("*.py")):
        source = path.read_text()
        for block in re.findall(
            r"from npa\.workbench\.open3d\.runner import \(([^)]*)\)", source
        ):
            for name in (n.strip().rstrip(",") for n in block.split()):
                if name and f"{name}" not in runner:
                    missing.append(f"{path.name}: {name}")

    assert missing == [], missing


def test_the_renamed_constant_is_the_one_the_committed_evidence_recorded() -> None:
    """Confirming the rename by value, not by the name looking similar.

    `FAR_BAND_AREA_SHARE` is 0.1, and 0.1 is the `reading_boundary` in both reports the two
    harnesses produced. That is what makes it the same constant rather than a plausible one.
    """
    runner = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "npa"
        / "workbench"
        / "open3d"
        / "runner.py"
    ).read_text()
    declared = re.search(r"^FAR_BAND_AREA_SHARE = ([0-9.]+)", runner, re.M)
    assert declared, "FAR_BAND_AREA_SHARE is gone; this rename needs revisiting"
    value = float(declared.group(1))

    for name in ("band-validity-sweep.json", "floor-across-geometry-and-sampling.json"):
        report = json.loads((HARNESS.parent / name).read_text())
        boundaries = {
            found
            for found in re.findall(
                r'"reading_boundary": ([0-9.]+)', json.dumps(report)
            )
        }
        assert boundaries == {str(value)}, (name, boundaries, value)


def test_an_unwritable_output_is_refused_before_the_measurement_runs(
    tmp_path: Path,
) -> None:
    """The failure used to arrive after the work, which is the expensive way round.

    `band_validity_sweep` reaches its write about 75 seconds in. The check is at import
    time now, so the refusal costs a second or two and names the variable that fixes it.
    """
    harness_io = HARNESS / "harness_io.py"
    assert harness_io.exists(), harness_io

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(HARNESS)!r})\n"
        "from harness_io import output_path\n"
        'output_path("/definitely-not-here/report.json")\n'
    )
    environment = {k: v for k, v in os.environ.items() if k != "OPEN3D_OUT_DIR"}
    result = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    message = result.stdout + result.stderr
    assert "Cannot write" in message, message
    assert "OPEN3D_OUT_DIR" in message, message
    assert "Traceback" not in message, message


def test_the_override_redirects_the_report_and_keeps_its_filename(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(HARNESS)!r})\n"
        f"os.environ['OPEN3D_OUT_DIR'] = {str(tmp_path)!r}\n"
        "from harness_io import output_path\n"
        'print(output_path("/data/band-validity-sweep.json"))\n'
    )
    result = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / "band-validity-sweep.json")


def test_the_exact_output_override_wins_over_the_directory_override(
    tmp_path: Path,
) -> None:
    exact = tmp_path / "exact.json"
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(HARNESS)!r})\n"
        f"os.environ['OPEN3D_OUT'] = {str(exact)!r}\n"
        f"os.environ['OPEN3D_OUT_DIR'] = {str(tmp_path / 'ignored')!r}\n"
        "from harness_io import output_path\n"
        'print(output_path("/data/default.json"))\n'
    )
    result = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(exact)


def test_every_hardcoded_data_output_goes_through_the_checked_resolver() -> None:
    """So a new harness cannot reintroduce the lose-it-at-the-end shape."""
    unchecked = [
        f"{path.name}:{number}"
        for path in sorted(HARNESS.glob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if line.startswith("OUT = Path(")
        and "/data/" in line
        and path.name not in HARDCODED_OUTPUT_EXCLUSIONS
    ]

    assert unchecked == [], unchecked


def test_every_harness_compiles() -> None:
    """The cheapest possible check, added because I broke three of these and lint found it.

    Converting the geometry reads put a `from harness_io import ...` line above a
    `from __future__` import in three harnesses, which is a hard SyntaxError. `ast.parse`
    accepts it -- the rule is enforced by the compiler, not the parser -- so this compiles
    rather than parses.
    """
    broken = []
    for path in sorted(HARNESS.glob("*.py")):
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as error:
            broken.append(f"{path.name}: {error.msg}")

    assert broken == [], broken


def test_no_harness_reads_geometry_without_checking_what_it_got() -> None:
    """Open3D's readers do not raise on a missing file; they return an empty geometry.

    That is why `band_profile_across_scenes` died in a numpy reduction with `zero-size array
    to reduction operation maximum` and `consumer_occupancy` inside Open3D's C++ with
    `IndexError: _Map_base::at`. Neither names a file, and both are several frames away from
    the read that caused them.

    Reads of the built-in `o3d.data` datasets are exempt: those download on demand and the
    path comes from the library, not from a staged bundle.
    """
    offenders = []
    for path in sorted(HARNESS.glob("*.py")):
        lines = path.read_text().splitlines()
        for number, line in enumerate(lines, start=1):
            nearby = "\n".join(lines[max(0, number - 4) : number])
            if "o3d.data" in nearby:
                continue
            if re.search(r"o3d\.io\.read_(point_cloud|triangle_mesh)\(", line):
                if path.name in GEOMETRY_READ_EXCLUSIONS:
                    continue
                offenders.append(f"{path.name}:{number}")

    assert offenders == [], offenders


def test_a_missing_geometry_is_reported_with_its_path_and_no_traceback(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(HARNESS)!r})\n"
        "from harness_io import read_cloud\n"
        f"read_cloud({str(tmp_path / 'absent.ply')!r})\n"
    )
    result = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, timeout=120
    )

    message = result.stdout + result.stderr
    assert result.returncode == 1, message
    assert "absent.ply" in message, message
    assert "No point cloud at" in message, message
    assert "Traceback" not in message, message


def test_a_file_that_exists_but_holds_nothing_is_reported_as_a_parse_failure(
    tmp_path: Path,
) -> None:
    """Existence is not enough: a truncated or unparseable file also reads as empty.

    Unlike the missing-file case, this one has to reach Open3D to find out, so it only runs
    where the package is installed.
    """
    pytest.importorskip("open3d")

    empty = tmp_path / "truncated.ply"
    empty.write_text("ply\nformat ascii 1.0\n")

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(HARNESS)!r})\n"
        "from harness_io import read_cloud\n"
        f"read_cloud({str(empty)!r})\n"
    )
    result = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, timeout=120
    )

    message = result.stdout + result.stderr
    assert result.returncode == 1, message
    assert "is empty" in message, message
    assert "parse failure" in message, message
    assert "Traceback" not in message, message


def test_the_staged_volume_note_appears_only_for_paths_under_that_volume(
    tmp_path: Path,
) -> None:
    """One harness reads `/a/fused.ply`, where advice about /data would be wrong."""
    source = (HARNESS / "harness_io.py").read_text()

    assert 'resolved.parts[:2] == ("/", "data")' in source, (
        "the /data note is unconditional again; it is wrong for paths outside that volume"
    )


AUDIT = HARNESS / "audit_evidence_references.py"
EVIDENCE = HARNESS.parent
PREVIEW = EVIDENCE / "capability-record-provenance-bound-preview.json"
PINNED_SCRIPT = "harness/preview_from_retained_plys.py"


def run_audit(root: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run the audit, optionally against a purpose-built evidence root.

    The provenance behaviour needs a checkout with history. CI checks out without
    any, so tests pointed at the real catalogue either crash on `git show` or skip,
    and a test that skips in CI is exactly the defect a sibling branch spent today
    removing. So the cases below build the history they need.
    """
    environment = dict(os.environ)
    if root is not None:
        environment["NPA_EVIDENCE_ROOT"] = str(root)
    return subprocess.run(
        [sys.executable, str(AUDIT)],
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
    )


def git(repository: Path, *arguments: str) -> str:
    environment = dict(os.environ)
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        env=environment,
    ).stdout.strip()


@pytest.fixture
def evidence_repository(tmp_path: Path) -> tuple[Path, str, str]:
    """An evidence root whose generator is committed once and then edited.

    Returns the root, the commit holding the original bytes, and their hash. This is
    the shape the real catalogue is in after a generator improvement: an artifact
    committed here, and the bytes that produced it reachable only through history.
    """
    root = tmp_path / "evidence"
    (root / "harness").mkdir(parents=True)
    generator = root / "harness" / "make_it.py"
    generator.write_text("print('the version that produced the artifact')\n")
    frame = root / "artifact.txt"
    frame.write_text("pixels\n")

    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "the generating bytes")
    at_commit = git(root, "rev-parse", "HEAD")
    original = hashlib.sha256(generator.read_bytes()).hexdigest()

    # The edit. The artifact cannot be regenerated, which is why the hash must stay.
    generator.write_text(
        "print('improved afterwards, and not what made the artifact')\n"
    )

    record = {
        "generator": {
            "script": "harness/make_it.py",
            "script_sha256": original,
            "script_bytes_committed_at": at_commit,
        },
        "capture": {
            "frame": "artifact.txt",
            "frame_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
        },
    }
    (root / "capability-record-under-test.json").write_text(
        json.dumps(record, indent=2)
    )
    return root, at_commit, original


def test_a_provenance_hash_is_verified_against_the_commit_it_names(
    evidence_repository: tuple[Path, str, str],
) -> None:
    """The generator no longer matches the working tree, and the record is still right."""
    root, _, original = evidence_repository
    at_head = hashlib.sha256((root / "harness/make_it.py").read_bytes()).hexdigest()
    assert at_head != original, "the fixture failed to make the two differ"

    result = run_audit(root)

    assert result.returncode == 0, result.stdout
    assert "is a provenance claim" in result.stdout, result.stdout


def test_declaring_a_commit_redirects_the_check_and_does_not_exempt_it(
    evidence_repository: tuple[Path, str, str],
) -> None:
    """The dangerous reading of this field would be 'stop checking this hash'.

    So corrupt the hash while leaving the commit declared, and require a finding. If
    the field were an exemption, a record could silence any mismatch by naming a
    commit, which is a worse gate than the one it replaces.
    """
    root, _, _ = evidence_repository
    record_path = root / "capability-record-under-test.json"
    record = json.loads(record_path.read_text())
    record["generator"]["script_sha256"] = "0" * 64
    record_path.write_text(json.dumps(record, indent=2))

    result = run_audit(root)

    assert result.returncode == 1, result.stdout
    assert "000000000000" in result.stdout, result.stdout
    assert "which is actually" in result.stdout, result.stdout


def test_a_commit_this_checkout_lacks_is_reported_not_called_a_mismatch(
    evidence_repository: tuple[Path, str, str],
) -> None:
    """A shallow clone is a property of the checkout, not a defect in the evidence.

    CI is that checkout, which is how this case stopped being hypothetical.
    """
    root, _, _ = evidence_repository
    record_path = root / "capability-record-under-test.json"
    record = json.loads(record_path.read_text())
    record["generator"]["script_bytes_committed_at"] = "f" * 40
    record_path.write_text(json.dumps(record, indent=2))

    result = run_audit(root)

    assert result.returncode == 0, result.stdout
    assert "a commit this checkout does not have" in result.stdout, result.stdout
    # Under its own heading, not folded into the run-artifact count, whose label says
    # "mostly a run's own artifacts" and would misdescribe this to the one reader who
    # can act on it.
    assert "1 hash(es) claim bytes at a commit" in result.stdout, result.stdout


def test_a_hash_with_no_declared_commit_is_still_read_from_the_working_tree(
    evidence_repository: tuple[Path, str, str],
) -> None:
    """The ordinary case has to keep working, so break it on purpose.

    Without this, making provenance resolve elsewhere could quietly have moved every
    hash off the working tree, and the suite would have looked just as green.
    """
    root, _, _ = evidence_repository
    record_path = root / "capability-record-under-test.json"
    record = json.loads(record_path.read_text())
    record["capture"]["frame_sha256"] = "1" * 64
    record_path.write_text(json.dumps(record, indent=2))

    result = run_audit(root)

    assert result.returncode == 1, result.stdout
    assert "capture.frame_sha256" in result.stdout, result.stdout


def test_the_audit_is_clean_on_the_real_catalogue() -> None:
    """Exit 0 either way: with history the pin is verified, without it it is reported.

    This is the one assertion that holds in a shallow checkout and a full one, which
    is why it is the one made against the real tree.
    """
    result = run_audit()

    assert result.returncode == 0, result.stdout


def test_the_real_records_provenance_pin_is_not_the_file_at_head() -> None:
    """The distinction, asserted where it is cheap: no history needed to hash HEAD.

    If someone later clears an audit report by overwriting that pin with the edited
    script's hash, this fails. That is the move worth preventing, and preventing it
    does not require the commit to be present.
    """
    generator = json.loads(PREVIEW.read_text())["generator"]
    at_head = hashlib.sha256((EVIDENCE / PINNED_SCRIPT).read_bytes()).hexdigest()

    assert generator["script_bytes_committed_at"], "the record stopped saying where"
    assert generator["script_sha256"] != at_head, (
        "the provenance pin now equals the working tree, which means it was "
        "overwritten with bytes that post-date the artifact it describes"
    )


def test_the_real_records_named_commit_actually_resolves_where_that_is_decidable() -> (
    None
):
    """A typo in the commit field downgrades the claim instead of failing anything.

    Review found the gap by mutation: replace `script_bytes_committed_at` with an
    unresolvable sha and the audit correctly reports "could not be checked either way"
    -- while every test in this file still passes. So a one-character slip silently
    converts a verified provenance claim into an unverified one, and the honest
    `inconclusive` the audit prints is what hides it.

    My first version asked `git rev-parse --is-shallow-repository`, assuming an absent
    commit means a shallow clone. CI disproved it: the checkout reports `false` and still
    does not have the commit, so the gate enforced where it could not decide and failed
    the shard. Not-shallow and holds-this-object are different questions.

    The real defect was underneath. The record pinned a commit that lives only on two
    lane branches -- byte-identical bytes, but absent from main and from this branch, so
    no reviewer of this PR could resolve it and neither could CI. A provenance pin that
    only its author can check is not much of a pin. It is now pinned to the main commit
    carrying the same bytes, verified by sha256 rather than assumed.

    That repair is what lets this require ancestry rather than mere presence: the named
    commit must be in the history of the artifact's own branch, not merely somewhere in
    the object store.

    Shallowness is still asked, but for what it actually tells us. A missing object and
    a shallow clone produce the same git error, so absence alone decides nothing -- that
    was the first bug. In a *complete* checkout, though, absence is a real answer: the
    commit is not in anything this repository fetched, which a typo and a foreign-branch
    pin both look like, and both are defects for a claim that has to be verifiable here.
    Only shallow-and-absent is genuinely undecidable, and only that skips.
    """
    declared = records_declaring_a_committed_at_pin()
    # Enumerated rather than named. The harness honours `<name>_bytes_committed_at` on
    # any record, so naming the only record that uses it today would hand the next one
    # the lenient shallow path with nothing requiring its commit be reachable -- and that
    # failure is silent, because the hash simply stops being checked. Review found this.
    assert declared, (
        "no record declares a committed-at pin, so this test is guarding nothing; if "
        "the field was renamed, this is the test that should say so"
    )

    for record_name, field, committed_at in declared:
        _assert_the_pin_is_reachable(record_name, field, committed_at)


def records_declaring_a_committed_at_pin() -> list[tuple[str, str, str]]:
    """Every record that pins bytes at a commit, found by the field's own shape.

    The mechanism is keyed on the `_bytes_committed_at` suffix, so this looks for that
    rather than for a filename, and a record that starts using it is covered the day it
    does.
    """

    def pins(node: object, path: str = "") -> list[tuple[str, str]]:
        # Walked generically rather than at a fixed depth: one record's top level is a
        # list, and the field is identified by its name wherever it sits.
        out: list[tuple[str, str]] = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith("_bytes_committed_at") and isinstance(value, str):
                    out.append((f"{path}{key}", value))
                else:
                    out.extend(pins(value, f"{path}{key}."))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                out.extend(pins(value, f"{path}[{index}]."))
        return out

    found: list[tuple[str, str, str]] = []
    for record in sorted(EVIDENCE.glob("*.json")):
        for field, value in pins(json.loads(record.read_text())):
            found.append((record.name, field, value))
    return found


def _assert_the_pin_is_reachable(
    record_name: str, field: str, committed_at: str
) -> None:
    try:
        probe = subprocess.run(
            ["git", "merge-base", "--is-ancestor", committed_at, "HEAD"],
            cwd=EVIDENCE,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"git cannot answer the ancestry question here: {exc}")

    # 0 ancestor, 1 present but not an ancestor, anything else git declining to answer
    # because it has no such object.
    if probe.returncode not in (0, 1):
        shallow = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=EVIDENCE,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        ).stdout.strip()
        if shallow != "false":
            pytest.skip(
                f"shallow checkout and no {committed_at[:12]} for {record_name}: a "
                "truncated history and a commit that does not exist are the same error"
            )

    assert probe.returncode == 0, (
        f"{record_name} pins bytes at {committed_at[:12]} in {field}, which is not an "
        "ancestor of this branch in a checkout that holds complete history, so nobody "
        "reviewing this branch can resolve it. A typo looks exactly like this, and so "
        f"does a commit that exists only on some other branch. git said: "
        f"{probe.stderr.strip() or 'it is present but not an ancestor'}"
    )

    result = run_audit()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "verified against the commit the record names" in result.stdout, (
        "the provenance claim was not checked against its own commit in a checkout "
        "that has the history to do it, so the commit the record names does not "
        "resolve here:\n" + result.stdout
    )
