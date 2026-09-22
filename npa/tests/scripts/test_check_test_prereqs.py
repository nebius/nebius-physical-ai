"""Prove the test-prerequisite reporter is accurate, never blocks, and never crashes."""

import getpass
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
        total=int(total_gib * gib),
        used=int((total_gib - free_gib) * gib),
        free=int(free_gib * gib),
    )


def _present_everything(monkeypatch) -> None:
    monkeypatch.setattr(prereqs.shutil, "which", lambda _name: "/usr/bin/fake")
    monkeypatch.setattr(prereqs.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        prereqs.os, "memfd_create", lambda *_a, **_k: None, raising=False
    )


def _absent_everything(monkeypatch) -> None:
    monkeypatch.setattr(prereqs.shutil, "which", lambda _name: None)
    monkeypatch.setattr(prereqs.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.delattr(prereqs.os, "memfd_create", raising=False)


def test_present_prereqs_report_ok(monkeypatch) -> None:
    """Every probe returning True renders as OK, with no failure/skip summary lines.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
    Returns:
        None.
    Raises:
        AssertionError: A present prerequisite is misreported as missing.
    """
    _present_everything(monkeypatch)
    report = prereqs.format_report(prereqs.collect_prereqs())
    assert "MISSING" not in report
    assert "Every known optional prerequisite is present." in report


def test_missing_adapter_extra_is_reported_as_a_collection_blocker_not_a_skip(
    monkeypatch,
) -> None:
    """Missing pyarrow is reported as making `make test` FAIL, never phrased as a quiet skip.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
    Returns:
        None.
    Raises:
        AssertionError: The report claims reduced coverage where the real
            consequence, verified against npa/tests, is a non-zero exit from
            pytest collection errors.
    """
    _absent_everything(monkeypatch)
    report = prereqs.format_report(prereqs.collect_prereqs())
    assert "FAIL outright" in report
    assert "adapter extra (pyarrow)" in report.split("FAIL outright")[1].split(".")[0]


def test_missing_skip_only_prereq_never_claims_it_fails_the_run(monkeypatch) -> None:
    """A prerequisite that only causes a self-skip is never listed among the FAIL-outright set.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
    Returns:
        None.
    Raises:
        AssertionError: A skip-only prerequisite is folded into the collection-blocker summary.
    """
    _absent_everything(monkeypatch)
    report = prereqs.format_report(prereqs.collect_prereqs())
    fail_section = report.split("FAIL outright")[1].split(".")[0]
    assert "tmux" not in fail_section
    assert "self-skip instead of failing" in report


def test_probe_failure_is_reported_as_unknown_not_a_crash(monkeypatch) -> None:
    """A probe that raises is surfaced as UNKNOWN with the error, not propagated.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
    Returns:
        None.
    Raises:
        AssertionError: `collect_prereqs`/`format_report` raise instead of
            reporting the failure as an observation.
    """

    def _boom(_name):
        raise PermissionError("no access")

    monkeypatch.setattr(prereqs.shutil, "which", _boom)
    monkeypatch.setattr(prereqs.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        prereqs.os, "memfd_create", lambda *_a, **_k: None, raising=False
    )
    report = prereqs.format_report(prereqs.collect_prereqs())
    assert "UNKNOWN" in report
    assert "PermissionError: no access" in report


def test_memfd_missing_on_macos_reports_not_applicable(monkeypatch) -> None:
    """`os.memfd_create` absence on macOS is labeled an unfixable platform gap, not MISSING.

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
    assert "Known optional test prerequisites for" in result.stdout
    assert "Temp-directory observations" in result.stdout


def test_temp_space_report_warns_when_free_space_is_low(
    monkeypatch, tmp_path: Path
) -> None:
    """Free space under the warning threshold recommends `--basetemp` isolation.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
        tmp_path: Isolated fixture directory standing in for the temp root.
    Returns:
        None.
    Raises:
        AssertionError: Low free space is not flagged, or omits the fix.
    """
    monkeypatch.setattr(
        prereqs.shutil, "disk_usage", _disk_usage(total_gib=100, free_gib=5)
    )
    report = prereqs.collect_temp_space_report(tmp_root=tmp_path)
    rendered = prereqs.format_temp_space_report(report)
    assert "Below" in rendered
    assert "--basetemp" in rendered


def test_temp_space_report_is_quiet_with_ample_space_and_no_retained_runs(
    monkeypatch, tmp_path: Path
) -> None:
    """Ample free space and no retained runs produce no warning and no run listing.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
        tmp_path: Isolated fixture directory standing in for the temp root.
    Returns:
        None.
    Raises:
        AssertionError: A healthy environment is reported as needing action.
    """
    monkeypatch.setattr(
        prereqs.shutil, "disk_usage", _disk_usage(total_gib=100, free_gib=90)
    )
    monkeypatch.setattr(getpass, "getuser", lambda: "fakeuser-with-no-retained-dirs")
    report = prereqs.collect_temp_space_report(tmp_root=tmp_path)
    rendered = prereqs.format_temp_space_report(report)
    assert "Below" not in rendered
    assert "run director" not in rendered


def test_temp_space_report_never_implies_a_non_current_run_is_safe_to_delete(
    monkeypatch, tmp_path: Path
) -> None:
    """Both a very old sibling run and the current one are listed, with no deletion suggestion.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
        tmp_path: Isolated fixture directory standing in for the temp root.
    Returns:
        None.
    Raises:
        AssertionError: The report is silent about a retained run, or implies
            that a directory other than `pytest-current` is inactive/deletable.
    """
    monkeypatch.setattr(
        prereqs.shutil, "disk_usage", _disk_usage(total_gib=100, free_gib=90)
    )
    monkeypatch.setattr(getpass, "getuser", lambda: "fakeuser")
    pytest_root = tmp_path / "pytest-of-fakeuser"
    old_run = pytest_root / "pytest-1"
    current_run = pytest_root / "pytest-2"
    old_run.mkdir(parents=True)
    current_run.mkdir(parents=True)
    old_mtime = time.time() - 2 * _DAY
    prereqs.os.utime(old_run, (old_mtime, old_mtime))
    (pytest_root / "pytest-current").symlink_to(current_run)

    report = prereqs.collect_temp_space_report(tmp_root=tmp_path)
    by_name = {run.path.name: run for run in report.retained_runs}
    assert by_name["pytest-1"].is_current is False
    assert by_name["pytest-2"].is_current is True

    rendered = prereqs.format_temp_space_report(report)
    assert str(old_run) in rendered
    assert str(current_run) in rendered
    assert "safe to delete" not in rendered.lower()
    assert "thereby proven inactive" in rendered


def test_temp_space_report_notes_a_live_lock_file(monkeypatch, tmp_path: Path) -> None:
    """A `.lock` file younger than pytest's own retention window is called out as still live.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
        tmp_path: Isolated fixture directory standing in for the temp root.
    Returns:
        None.
    Raises:
        AssertionError: A fresh `.lock` file is not surfaced, understating how
            confidently this report can call a directory abandoned.
    """
    monkeypatch.setattr(
        prereqs.shutil, "disk_usage", _disk_usage(total_gib=100, free_gib=90)
    )
    monkeypatch.setattr(getpass, "getuser", lambda: "fakeuser")
    pytest_root = tmp_path / "pytest-of-fakeuser"
    locked_run = pytest_root / "pytest-1"
    locked_run.mkdir(parents=True)
    (locked_run / ".lock").write_text("")

    report = prereqs.collect_temp_space_report(tmp_root=tmp_path)
    rendered = prereqs.format_temp_space_report(report)
    assert ".lock present" in rendered
    assert "pytest still treats this as live" in rendered


def test_temp_space_report_survives_a_disk_usage_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """`shutil.disk_usage` raising is reported as an observation, not propagated.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
        tmp_path: Isolated fixture directory standing in for the temp root.
    Returns:
        None.
    Raises:
        AssertionError: The report raises instead of degrading gracefully.
    """

    def _boom(_path):
        raise OSError("disk unavailable")

    monkeypatch.setattr(prereqs.shutil, "disk_usage", _boom)
    report = prereqs.collect_temp_space_report(tmp_root=tmp_path)
    rendered = prereqs.format_temp_space_report(report)
    assert "could not read disk usage" in rendered
    assert "disk unavailable" in rendered


def test_temp_space_report_survives_an_unresolvable_user(
    monkeypatch, tmp_path: Path
) -> None:
    """`getpass.getuser` raising (no USER/LOGNAME, no pwd entry) degrades to an observation.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
        tmp_path: Isolated fixture directory standing in for the temp root.
    Returns:
        None.
    Raises:
        AssertionError: The report raises instead of reporting the scan as unavailable.
    """
    monkeypatch.setattr(
        prereqs.shutil, "disk_usage", _disk_usage(total_gib=100, free_gib=90)
    )

    def _boom():
        raise OSError("no such user")

    monkeypatch.setattr(getpass, "getuser", _boom)
    report = prereqs.collect_temp_space_report(tmp_root=tmp_path)
    rendered = prereqs.format_temp_space_report(report)
    assert "could not enumerate retained pytest run directories" in rendered
    assert "no such user" in rendered
