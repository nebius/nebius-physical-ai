"""Report which full-suite prerequisites this environment has, without gating anything.

`make test` self-skips tests whose optional prerequisite (ffmpeg, kubectl, tmux,
Docker, Node, a CPU checkpoint runtime, or the Linux-only `os.memfd_create`) is
absent, so a local pass can silently cover less than CI without saying so
(CONTRIBUTING.md "Testing Requirements"). This script makes that gap visible in
one command instead of leaving a contributor to piece it together from prose
spread across CONTRIBUTING.md, npa/README.md, and individual test modules.

It also reports temp-directory disk headroom and provenance: pytest's default
`$TMPDIR/pytest-of-<user>` root is shared by every process that user runs, not
scoped to one checkout, so concurrent work across git worktrees on one machine
competes for the same disk and can exhaust it with no per-lane attribution.

It never fails: this is a report, not a gate. `make check-env` is the gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


@dataclass(frozen=True)
class Prereq:
    """One optional, full-suite-parity prerequisite and how to satisfy it.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """

    name: str
    present: bool
    coverage_lost: str
    fix: str
    linux_only: bool = False


def _binary_present(binary: str) -> bool:
    return shutil.which(binary) is not None


def _module_present(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def collect_prereqs() -> list[Prereq]:
    """Probe this interpreter and PATH for every documented optional test prerequisite.

    Args:
        None.
    Returns:
        One `Prereq` per documented optional dependency, in report order.
    Raises:
        None.
    """
    ffmpeg_ready = _binary_present("ffmpeg") and _binary_present("ffprobe")
    return [
        Prereq(
            "adapter extra (pyarrow)",
            _module_present("pyarrow"),
            "dataset/parquet conversion tests fail to import instead of running",
            'pip install -e "npa[dev,adapter]"',
        ),
        Prereq(
            "ffmpeg + ffprobe",
            ffmpeg_ready,
            "real video decode/export tests self-skip (test_ltx2_video_check.py etc.)",
            "install ffmpeg (provides ffprobe); set NPA_REQUIRE_FFMPEG=1 to turn the "
            "skip into a hard failure once installed, matching CI",
        ),
        Prereq(
            "CPU checkpoint runtime (torch)",
            _module_present("torch"),
            "real checkpoint load/export security tests self-skip",
            'pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 '
            '&& pip install -e "npa[sonic]"',
        ),
        Prereq(
            "kubectl",
            _binary_present("kubectl"),
            "provisioning tests that shell out to kubectl self-skip (test_provisioning.py)",
            "install kubectl",
        ),
        Prereq(
            "docker",
            _binary_present("docker"),
            "a few image/container tests self-skip",
            "install Docker",
        ),
        Prereq(
            "tmux",
            _binary_present("tmux"),
            "a few isolated-session/agent tests self-skip",
            "install tmux",
        ),
        Prereq(
            "node",
            _binary_present("node"),
            "a few agent/browser tests self-skip",
            "install Node.js",
        ),
        Prereq(
            "os.memfd_create",
            hasattr(os, "memfd_create"),
            "sealed in-memory image-byte-scan tests self-skip; this is Linux-only, "
            "not fixable on macOS",
            "run the full suite on Linux for this coverage",
            linux_only=True,
        ),
    ]


def _status(prereq: Prereq) -> str:
    if prereq.present:
        return "OK"
    if prereq.linux_only and sys.platform == "darwin":
        return "N/A (macOS)"
    return "MISSING"


def format_report(prereqs: list[Prereq]) -> str:
    """Render one line per prerequisite plus a summary footer.

    Args:
        prereqs: Prerequisites to render, in the order they should print.
    Returns:
        A human-readable, newline-joined report ending in a summary line.
    Raises:
        None.
    """
    lines = [f"Full-suite parity report for {sys.executable}:"]
    missing = 0
    for prereq in prereqs:
        status = _status(prereq)
        line = f"  [{status:10}] {prereq.name}"
        if status == "MISSING":
            missing += 1
            line += f"\n               lost coverage: {prereq.coverage_lost}\n               fix: {prereq.fix}"
        lines.append(line)
    if missing:
        lines.append(
            f"\n{missing} prerequisite(s) missing: `make test` will pass with less "
            "coverage than CI, by self-skipping rather than failing. This is not "
            "a gate; run the `fix` commands above for full parity, or ignore them "
            "for portable CLI/lint work."
        )
    else:
        lines.append("\nAll optional full-suite prerequisites are present.")
    return "\n".join(lines)


_LOW_SPACE_WARNING_GIB = 10
_STALE_RUN_WARNING_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class RetainedRun:
    """One retained `pytest-of-<user>/pytest-<N>` directory from any process, any checkout.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """

    path: Path
    age_seconds: float
    is_current: bool


@dataclass(frozen=True)
class TempSpaceReport:
    """Disk headroom and provenance for the temp root this host's pytest runs share.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """

    tmp_root: Path
    free_gib: float
    total_gib: float
    retained_runs: list[RetainedRun] = field(default_factory=list)


def collect_temp_space_report(tmp_root: Path | None = None, now: float | None = None) -> TempSpaceReport:
    """Inspect the effective pytest temp root for free space and other processes' retained runs.

    Args:
        tmp_root: Root to inspect; defaults to `tempfile.gettempdir()`, matching
            what an unconfigured `pytest` invocation resolves.
        now: Reference time for computing directory ages; defaults to `time.time()`.
    Returns:
        Free/total space on that filesystem, plus every sibling `pytest-<N>`
        run directory found under `pytest-of-<user>` there (not just this
        process's own), with the currently active one flagged.
    Raises:
        None.
    """
    root = tmp_root or Path(tempfile.gettempdir())
    reference_time = time.time() if now is None else now
    usage = shutil.disk_usage(root)
    pytest_root = root / f"pytest-of-{os.environ.get('USER', 'unknown')}"
    current = (pytest_root / "pytest-current").resolve() if (pytest_root / "pytest-current").exists() else None

    retained = []
    if pytest_root.is_dir():
        for entry in sorted(pytest_root.iterdir()):
            if not entry.is_dir() or entry.is_symlink() or not entry.name.startswith("pytest-"):
                continue
            age = reference_time - entry.stat().st_mtime
            retained.append(RetainedRun(entry, age, entry.resolve() == current))

    return TempSpaceReport(
        tmp_root=root,
        free_gib=usage.free / (1024**3),
        total_gib=usage.total / (1024**3),
        retained_runs=retained,
    )


def format_temp_space_report(report: TempSpaceReport) -> str:
    """Render free space plus other processes' retained run directories, oldest first.

    Args:
        report: Result of `collect_temp_space_report`.
    Returns:
        A human-readable report. Never recommends deleting a directory this
        function cannot prove is not the active run of some other process.
    Raises:
        None.
    """
    lines = [
        f"\nTemp root: {report.tmp_root} ({report.free_gib:.1f} GiB free of "
        f"{report.total_gib:.1f} GiB)."
    ]
    if report.free_gib < _LOW_SPACE_WARNING_GIB:
        lines.append(
            f"  WARNING: below {_LOW_SPACE_WARNING_GIB} GiB free. This root is shared by "
            "every process this user runs, not scoped to one checkout — concurrent "
            "work in other worktrees competes for the same space. Point this run at "
            "a lane-owned directory instead of the shared default:\n"
            "    pytest --basetemp=<owned-dir>/pytest-tmp ...\n"
            "  and clean up only that owned directory yourself when your lane's "
            "work is done; never remove another pytest-of-<user>/pytest-N directory "
            "by hand without confirming it is not `pytest-current` for a run still "
            "in progress."
        )
    stale = [run for run in report.retained_runs if not run.is_current and run.age_seconds > _STALE_RUN_WARNING_SECONDS]
    if stale:
        pytest_root = report.tmp_root / f"pytest-of-{os.environ.get('USER', 'unknown')}"
        noun = "directory is" if len(stale) == 1 else "directories are"
        lines.append(
            f"  {len(stale)} retained run {noun} older than "
            f"{_STALE_RUN_WARNING_SECONDS // 3600}h under {pytest_root}:"
        )
        for run in stale:
            lines.append(f"    {run.path} (age {run.age_seconds / 3600:.1f}h)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(format_report(collect_prereqs()))
    print(format_temp_space_report(collect_temp_space_report()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
