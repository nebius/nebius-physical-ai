"""Prove the checkout-drift guard fails loudly on foreign/missing installs and passes when correct.

Each case launches the real script as a subprocess with `-S` (skip `site`, so no
ambient virtualenv or editable install leaks in) and an explicit `PYTHONPATH`,
so the only thing determining behavior is exactly what a contributor's shell
would export — not this test runner's own environment.
"""

from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_dev_environment.py"


def _make_checkout(root: Path, name: str) -> Path:
    """Create a minimal fake checkout with an importable `npa` package.

    Args:
        root: Directory under which to create the checkout.
        name: Subdirectory name for this fake checkout.
    Returns:
        The fake checkout's root path (containing `npa/src/npa/__init__.py`).
    Raises:
        None.
    """
    checkout = root / name
    package = checkout / "npa" / "src" / "npa"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    return checkout


def _run(repo_root: Path, pythonpath: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--repo-root", str(repo_root)],
        env={"PYTHONPATH": pythonpath, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_matching_install_passes(tmp_path: Path) -> None:
    """A checkout whose own src is on PYTHONPATH exits 0 with no output.

    Args:
        tmp_path: Isolated fixture directory.
    Returns:
        None.
    Raises:
        AssertionError: The guard rejects a correctly configured checkout.
    """
    checkout = _make_checkout(tmp_path, "correct")
    result = _run(checkout, str(checkout / "npa" / "src"))
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_foreign_checkout_without_pythonpath_fails(tmp_path: Path) -> None:
    """A foreign checkout's src on PYTHONPATH is rejected with an actionable message.

    Args:
        tmp_path: Isolated fixture directory.
    Returns:
        None.
    Raises:
        AssertionError: The guard fails to detect the mismatch.
    """
    target = _make_checkout(tmp_path, "target")
    foreign = _make_checkout(tmp_path, "foreign")
    result = _run(target, str(foreign / "npa" / "src"))
    assert result.returncode == 1
    assert "shared" in result.stderr
    assert str(foreign / "npa" / "src") in result.stderr
    assert str(target / "npa" / "src") in result.stderr
    assert "export PYTHONPATH" in result.stderr


def test_correcting_pythonpath_wins_over_a_foreign_entry(tmp_path: Path) -> None:
    """The target checkout's own src earlier on PYTHONPATH overrides a foreign entry later on it.

    Args:
        tmp_path: Isolated fixture directory.
    Returns:
        None.
    Raises:
        AssertionError: A later foreign entry shadows the corrected one.
    """
    target = _make_checkout(tmp_path, "target")
    foreign = _make_checkout(tmp_path, "foreign")
    pythonpath = f"{target / 'npa' / 'src'}:{foreign / 'npa' / 'src'}"
    result = _run(target, pythonpath)
    assert result.returncode == 0, result.stderr


def test_missing_install_fails_with_install_instructions(tmp_path: Path) -> None:
    """No install at all (fresh clone) fails with a `pip install -e` instruction, not a traceback.

    Args:
        tmp_path: Isolated fixture directory.
    Returns:
        None.
    Raises:
        AssertionError: The guard crashes instead of reporting the missing install.
    """
    target = _make_checkout(tmp_path, "target")
    empty = tmp_path / "empty-pythonpath"
    empty.mkdir()
    result = _run(target, str(empty))
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "pip install -e" in result.stderr
    assert "import npa" in result.stderr
