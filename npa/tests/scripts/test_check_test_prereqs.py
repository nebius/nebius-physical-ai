"""Prove the full-suite prerequisite reporter is accurate and never blocks."""

import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import check_test_prereqs as prereqs  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_test_prereqs.py"
_DAY = 24 * 60 * 60


def _disk_usage(total_gib: float, free_gib: float):
    gib = 1024**3
    return lambda _path: SimpleNamespace(
        total=int(total_gib * gib), used=int((total_gib - free_gib) * gib), free=int(free_gib * gib)
    )


def test_present_binary_reports_ok(monkeypatch) -> None:
    """A prerequisite whose probe returns True renders as OK, not MISSING.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
    Returns:
        None.
    Raises:
        AssertionError: A present prerequisite is misreported as missing.
    """
    monkeypatch.setattr(prereqs.shutil, "which", lambda _name: "/usr/bin/fake")
    monkeypatch.setattr(prereqs.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(prereqs.os, "memfd_create", lambda *_a, **_k: None, raising=False)
    report = prereqs.format_report(prereqs.collect_prereqs())
    assert "MISSING" not in report
    assert "All optional full-suite prerequisites are present." in report


def test_missing_binary_reports_coverage_lost_and_fix(monkeypatch) -> None:
    """A prerequisite whose probe returns False names the lost coverage and the fix.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
    Returns:
        None.
    Raises:
        AssertionError: The report omits the coverage-loss or fix guidance.
    """
    monkeypatch.setattr(prereqs.shutil, "which", lambda _name: None)
    monkeypatch.setattr(prereqs.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.delattr(prereqs.os, "memfd_create", raising=False)
    report = prereqs.format_report(prereqs.collect_prereqs())
    assert "MISSING" in report
    assert "lost coverage:" in report
    assert "fix:" in report
    assert "not a gate" in report


def test_memfd_missing_on_macos_reports_not_applicable(monkeypatch) -> None:
    """`os.memfd_create` absence on macOS is labeled unfixable platform gap, not MISSING.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
    Returns:
        None.
    Raises:
        AssertionError: A Linux-only gap is reported as an actionable MISSING item.
    """
    monkeypatch.setattr(prereqs.shutil, "which", lambda _name: "/usr/bin/fake")
    monkeypatch.setattr(prereqs.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.delattr(prereqs.os, "memfd_create", raising=False)
    monkeypatch.setattr(prereqs.sys, "platform", "darwin")
    report = prereqs.format_report(prereqs.collect_prereqs())
    memfd_line = next(line for line in report.splitlines() if "os.memfd_create" in line)
    assert "N/A (macOS)" in memfd_line
    assert "MISSING" not in memfd_line


def test_script_always_exits_zero_as_a_real_subprocess() -> None:
    """The reporter never fails the invoking shell, whatever this host has installed.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: The report exits non-zero on a real, unmodified environment.
    """
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "Full-suite parity report for" in result.stdout
    assert "Temp root:" in result.stdout


def test_temp_space_report_warns_when_free_space_is_low(monkeypatch, tmp_path: Path) -> None:
    """Free space under the warning threshold recommends `--basetemp` isolation.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
        tmp_path: Isolated fixture directory standing in for the temp root.
    Returns:
        None.
    Raises:
        AssertionError: Low free space is not flagged, or omits the fix.
    """
    monkeypatch.setattr(prereqs.shutil, "disk_usage", _disk_usage(total_gib=100, free_gib=5))
    report = prereqs.collect_temp_space_report(tmp_root=tmp_path)
    rendered = prereqs.format_temp_space_report(report)
    assert "WARNING" in rendered
    assert "--basetemp" in rendered


def test_temp_space_report_is_quiet_with_ample_space_and_no_retained_runs(
    monkeypatch, tmp_path: Path
) -> None:
    """Ample free space and no retained runs produce no warning or stale-run lines.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
        tmp_path: Isolated fixture directory standing in for the temp root.
    Returns:
        None.
    Raises:
        AssertionError: A healthy environment is reported as needing action.
    """
    monkeypatch.setattr(prereqs.shutil, "disk_usage", _disk_usage(total_gib=100, free_gib=90))
    report = prereqs.collect_temp_space_report(tmp_root=tmp_path)
    rendered = prereqs.format_temp_space_report(report)
    assert "WARNING" not in rendered
    assert "older than" not in rendered


def test_temp_space_report_flags_stale_runs_but_never_the_current_one(
    monkeypatch, tmp_path: Path
) -> None:
    """A day-old sibling run is flagged as stale; the `pytest-current` target is not, even if also old.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
        tmp_path: Isolated fixture directory standing in for the temp root.
    Returns:
        None.
    Raises:
        AssertionError: The report misattributes staleness, or recommends
            deleting the directory a concurrent run may still be using.
    """
    monkeypatch.setenv("USER", "fakeuser")
    monkeypatch.setattr(prereqs.shutil, "disk_usage", _disk_usage(total_gib=100, free_gib=90))
    pytest_root = tmp_path / "pytest-of-fakeuser"
    stale_run = pytest_root / "pytest-1"
    current_run = pytest_root / "pytest-2"
    stale_run.mkdir(parents=True)
    current_run.mkdir(parents=True)
    old_mtime = time.time() - 2 * _DAY
    os.utime(stale_run, (old_mtime, old_mtime))
    os.utime(current_run, (old_mtime, old_mtime))
    (pytest_root / "pytest-current").symlink_to(current_run)

    report = prereqs.collect_temp_space_report(tmp_root=tmp_path)
    by_name = {run.path.name: run for run in report.retained_runs}
    assert by_name["pytest-1"].is_current is False
    assert by_name["pytest-2"].is_current is True

    rendered = prereqs.format_temp_space_report(report)
    assert str(stale_run) in rendered
    assert str(current_run) not in rendered
