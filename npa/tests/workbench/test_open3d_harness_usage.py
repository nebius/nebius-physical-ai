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
            if bare_argv or "os.environ[" in line:
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


def test_every_hardcoded_data_output_goes_through_the_checked_resolver() -> None:
    """So a new harness cannot reintroduce the lose-it-at-the-end shape."""
    unchecked = [
        f"{path.name}:{number}"
        for path in sorted(HARNESS.glob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if line.startswith("OUT = Path(") and "/data/" in line
    ]

    assert unchecked == [], unchecked
