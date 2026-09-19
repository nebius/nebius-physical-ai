"""Report known optional test prerequisites and temp-disk state, without gating anything.

Some `npa/tests` files probe for an optional tool or module and behave
differently depending on the result -- verified per-file below, not assumed:
some self-skip (pytest.mark.skipif / pytest.importorskip), one instead makes
pytest fail to *collect* the affected files at all (an unguarded top-level
`import pyarrow`, missing without the `adapter` extra). This script reports
which of those is actually present, and which real consequence is documented
for it, so a contributor does not have to rediscover it the hard way.

It also reports temp-directory disk headroom: pytest's default
`$TMPDIR/pytest-of-<user>` root is shared by every process that user runs, not
scoped to one checkout, so concurrent work across git worktrees on one machine
competes for the same disk. That section is observational only -- see its
docstring for why it never recommends deleting anything.

Every probe here is best-effort: a probe or scan that itself fails is reported
as an observation, not raised, so this command is safe to run in a
half-broken environment. `make check-env` is the actual gate.
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
    """One optional prerequisite, its verified real consequence, and its fix.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """

    name: str
    present: bool | None  # None: the probe itself could not run.
    probe_error: str | None
    blocks_collection: bool  # True: pytest fails to collect, distinct from a skip.
    consequence: str
    fix: str
    linux_only: bool = False


def _probe(fn) -> tuple[bool | None, str | None]:
    """Run one presence probe, turning any exception into an observation, not a crash.

    Args:
        fn: Zero-argument callable returning a bool.
    Returns:
        `(result, None)` on success, or `(None, "<error>")` if `fn` raised.
    Raises:
        None.
    """
    try:
        return fn(), None
    except Exception as exc:  # a probe must never take the whole report down with it
        return None, f"{type(exc).__name__}: {exc}"


def collect_prereqs() -> list[Prereq]:
    """Probe this interpreter and PATH for every prerequisite verified against `npa/tests`.

    Args:
        None.
    Returns:
        One `Prereq` per verified optional dependency, in report order.
    Raises:
        None.
    """
    pyarrow_present, pyarrow_error = _probe(
        lambda: importlib.util.find_spec("pyarrow") is not None
    )
    ffmpeg_present, ffmpeg_error = _probe(
        lambda: (
            shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
        )
    )
    torch_present, torch_error = _probe(
        lambda: importlib.util.find_spec("torch") is not None
    )
    tmux_present, tmux_error = _probe(lambda: shutil.which("tmux") is not None)
    node_present, node_error = _probe(lambda: shutil.which("node") is not None)
    memfd_present, memfd_error = _probe(lambda: hasattr(os, "memfd_create"))

    return [
        Prereq(
            "adapter extra (pyarrow)",
            pyarrow_present,
            pyarrow_error,
            True,
            "FAILS `make test`, not a skip: multiple files under npa/tests/ do an "
            "unguarded top-level `import pyarrow` (for example "
            "npa/tests/test_lerobot_shared_video_offsets.py), so pytest cannot even "
            "collect them without it (ModuleNotFoundError at collection, non-zero exit)",
            'pip install -e "npa[dev,adapter]"',
        ),
        Prereq(
            "ffmpeg + ffprobe",
            ffmpeg_present,
            ffmpeg_error,
            False,
            "self-skips via pytest.mark.skipif in "
            "npa/tests/workbench/test_ltx2_video_check.py; NPA_REQUIRE_FFMPEG=1 "
            "turns that same skip into a hard failure instead, matching CI",
            "install ffmpeg (provides ffprobe)",
        ),
        Prereq(
            "CPU checkpoint runtime (torch)",
            torch_present,
            torch_error,
            False,
            'self-skips via pytest.importorskip("torch") in '
            "npa/tests/workbench/test_checkpoint_security.py and similar "
            "security tests; presence here does not confirm it is the CPU "
            "torch==2.13.0 wheel CI pins, only that some torch import works",
            "pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 "
            '&& pip install -e "npa[sonic]"',
        ),
        Prereq(
            "tmux",
            tmux_present,
            tmux_error,
            False,
            "self-skips via pytest.mark.skipif in "
            "npa/tests/smoke/test_golden_eval_converge.py",
            "install tmux",
        ),
        Prereq(
            "node",
            node_present,
            node_error,
            False,
            "self-skips via pytest.mark.skipif in npa/tests/cli/test_agent.py and "
            "npa/tests/unit/test_executive_film_player.py",
            "install Node.js",
        ),
        Prereq(
            "os.memfd_create",
            memfd_present,
            memfd_error,
            False,
            "self-skips via pytest.mark.skipif in "
            "npa/tests/docker/test_image_byte_scan_sealed_inputs.py; this is a "
            "Linux syscall wrapper the stdlib simply does not provide on macOS, "
            "nothing to install",
            "run the full suite on Linux for this coverage",
            linux_only=True,
        ),
    ]


def _status(prereq: Prereq) -> str:
    if prereq.present is None:
        return "UNKNOWN"
    if prereq.present:
        return "OK"
    if prereq.linux_only and sys.platform == "darwin":
        return "N/A (macOS)"
    return "MISSING"


def format_report(prereqs: list[Prereq]) -> str:
    """Render one line per prerequisite plus a summary that never overstates what passed.

    Args:
        prereqs: Prerequisites to render, in the order they should print.
    Returns:
        A human-readable report. Presence of a module or binary is reported as
        exactly that -- not as proof the corresponding CI behavior is fully
        reproduced (see the torch entry: import success is not a version check).
    Raises:
        None.
    """
    lines = [f"Known optional test prerequisites for {sys.executable}:"]
    blockers, skips, unknown = [], [], []
    for prereq in prereqs:
        status = _status(prereq)
        line = f"  [{status:10}] {prereq.name}"
        if status == "UNKNOWN":
            unknown.append(prereq.name)
            line += f"\n               probe failed: {prereq.probe_error}"
        elif status == "MISSING":
            line += f"\n               consequence: {prereq.consequence}\n               fix: {prereq.fix}"
            (blockers if prereq.blocks_collection else skips).append(prereq.name)
        lines.append(line)

    if blockers:
        lines.append(
            f"\n{len(blockers)} missing prerequisite(s) make `make test` FAIL outright "
            f"(pytest collection errors, not a quiet skip): {', '.join(blockers)}."
        )
    if skips:
        lines.append(
            f"\n{len(skips)} missing prerequisite(s) self-skip instead of failing, so "
            f"`make test` can still exit 0 with less coverage than CI: {', '.join(skips)}."
        )
    if unknown:
        lines.append(
            f"\n{len(unknown)} prerequisite check(s) could not run: {', '.join(unknown)}."
        )
    if not (blockers or skips or unknown):
        lines.append(
            "\nEvery known optional prerequisite is present. That is not by itself "
            "proof of full CI parity -- only that these specific checks passed."
        )
    return "\n".join(lines)


_LOW_SPACE_WARNING_GIB = 10
# _pytest.pathlib.LOCK_TIMEOUT: pytest itself only treats a numbered run's
# .lock file as dead, and the directory as its own to clean up, after this
# many seconds. Younger than this, pytest's own logic still considers a
# directory potentially live.
_PYTEST_LOCK_TIMEOUT_SECONDS = 3 * 24 * 60 * 60


@dataclass(frozen=True)
class RetainedRun:
    """One `pytest-of-<user>/pytest-<N>` directory found under the temp root, any process.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """

    path: Path
    age_seconds: float | None
    is_current: bool
    lock_age_seconds: float | None


@dataclass(frozen=True)
class TempSpaceReport:
    """Best-effort disk headroom and retained-run observations for the shared temp root.

    Args:
        None.
    Returns:
        None.
    Raises:
        None.
    """

    tmp_root: Path
    free_gib: float | None
    total_gib: float | None
    disk_usage_error: str | None
    pytest_root: Path | None
    retained_runs: list[RetainedRun] = field(default_factory=list)
    scan_error: str | None = None


def _current_user() -> tuple[str | None, str | None]:
    try:
        import getpass

        return getpass.getuser(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _describe_run(
    entry: Path, current: Path | None, reference_time: float
) -> RetainedRun | None:
    try:
        if (
            not entry.is_dir()
            or entry.is_symlink()
            or not entry.name.startswith("pytest-")
        ):
            return None
        age = reference_time - entry.stat().st_mtime
        is_current = current is not None and entry.resolve() == current
        lock_path = entry / ".lock"
        lock_age = (
            reference_time - lock_path.stat().st_mtime if lock_path.is_file() else None
        )
        return RetainedRun(entry, age, is_current, lock_age)
    except OSError:
        return None


def collect_temp_space_report(
    tmp_root: Path | None = None, now: float | None = None
) -> TempSpaceReport:
    """Inspect the effective pytest temp root for free space and other processes' retained runs.

    Args:
        tmp_root: Root to inspect; defaults to `tempfile.gettempdir()`, matching
            what an unconfigured `pytest` invocation resolves.
        now: Reference time for computing directory ages; defaults to `time.time()`.
    Returns:
        Best-effort free/total space plus every sibling `pytest-<N>` run
        directory found there. Any failure (permissions, a directory removed
        mid-scan, no resolvable user name) is captured as an error field, not
        raised.
    Raises:
        None.
    """
    root = tmp_root or Path(tempfile.gettempdir())
    reference_time = time.time() if now is None else now

    free_gib = total_gib = None
    disk_usage_error = None
    try:
        usage = shutil.disk_usage(root)
        free_gib, total_gib = usage.free / (1024**3), usage.total / (1024**3)
    except OSError as exc:
        disk_usage_error = f"{type(exc).__name__}: {exc}"

    user, scan_error = _current_user()
    pytest_root = (root / f"pytest-of-{user}") if user else None
    retained: list[RetainedRun] = []
    if pytest_root is not None:
        try:
            current_link = pytest_root / "pytest-current"
            current = current_link.resolve() if current_link.exists() else None
            if pytest_root.is_dir():
                for entry in sorted(pytest_root.iterdir()):
                    described = _describe_run(entry, current, reference_time)
                    if described is not None:
                        retained.append(described)
        except OSError as exc:
            scan_error = f"{type(exc).__name__}: {exc}"

    return TempSpaceReport(
        root, free_gib, total_gib, disk_usage_error, pytest_root, retained, scan_error
    )


def _format_age(age_seconds: float | None) -> str:
    return "unknown age" if age_seconds is None else f"age {age_seconds / 3600:.1f}h"


def _lock_note(run: RetainedRun) -> str:
    if run.lock_age_seconds is None:
        return ""
    if run.lock_age_seconds < _PYTEST_LOCK_TIMEOUT_SECONDS:
        return f", .lock present ({run.lock_age_seconds / 3600:.1f}h old: pytest still treats this as live)"
    return ", .lock present but older than pytest's own retention window"


def format_temp_space_report(report: TempSpaceReport) -> str:
    """Render free space plus observed retained run directories. Deliberately makes no deletion call.

    Args:
        report: Result of `collect_temp_space_report`.
    Returns:
        A read-only report. `pytest-current` only names the most recently
        started run through this exact temp root; it is not proof any other
        directory is inactive -- a concurrent process may use a different
        `--basetemp` entirely, or hold a live `.lock` file here. Deciding
        whether an idle directory is safe to remove is a human judgment call
        this function deliberately does not make.
    Raises:
        None.
    """
    lines = [
        "\nTemp-directory observations (read-only; no deletion is ever recommended here):"
    ]
    if report.disk_usage_error:
        lines.append(
            f"  could not read disk usage for {report.tmp_root}: {report.disk_usage_error}"
        )
    else:
        lines.append(
            f"  {report.tmp_root}: {report.free_gib:.1f} GiB free of {report.total_gib:.1f} GiB."
        )
        if report.free_gib < _LOW_SPACE_WARNING_GIB:
            lines.append(
                f"  Below {_LOW_SPACE_WARNING_GIB} GiB free. This root is shared by every "
                "process this user runs, not scoped to one checkout -- concurrent work in "
                "other worktrees competes for the same space. A directory you own avoids "
                "that: pytest --basetemp=<owned-dir> ..."
            )

    if report.scan_error:
        lines.append(
            f"  could not enumerate retained pytest run directories: {report.scan_error}"
        )
        return "\n".join(lines)
    if not report.retained_runs:
        return "\n".join(lines)

    lines.append(
        f"  {len(report.retained_runs)} run director(y/ies) under {report.pytest_root}:"
    )
    for run in report.retained_runs:
        marker = " [pytest-current target]" if run.is_current else ""
        lines.append(
            f"    {run.path}{marker}: {_format_age(run.age_seconds)}{_lock_note(run)}"
        )
    lines.append(
        "  This is observational only. A directory not marked as the pytest-current "
        "target is NOT thereby proven inactive -- do not delete anything here based on "
        "this report; pytest manages its own retention."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        print(format_report(collect_prereqs()))
    except (
        Exception
    ) as exc:  # this command must never itself crash a contributor's shell
        print(
            f"prereq report unavailable: {type(exc).__name__}: {exc}", file=sys.stderr
        )
    try:
        print(format_temp_space_report(collect_temp_space_report()))
    except Exception as exc:
        print(
            f"temp-space report unavailable: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
