"""Exercise merge prechecks against real isolated Git histories."""

from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ci_merge_precheck  # noqa: E402
import ci_requirements  # noqa: E402


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", ".")
    _git(root, "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """Create a repository with a valid dependency fingerprint.

    Args:
        tmp_path: Per-test temporary directory.
    Returns:
        Repository ready for divergent commits.
    Raises:
        subprocess.CalledProcessError: Git initialization fails.
    """
    _git(tmp_path, "init", "-qb", "main")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.com")
    (tmp_path / "npa/ci").mkdir(parents=True)
    (tmp_path / "npa/pyproject.toml").write_text(
        '[project]\nrequires-python = ">=3.10"\ndependencies = ["anyio>=4.14.2"]\n'
        "[project.optional-dependencies]\ndev = []\nadapter = []\nsonic = []\ngenesis-test = []\n"
    )
    (tmp_path / "npa/ci/constraints.in").write_text("# constraints\n")
    (tmp_path / "npa/ci/requirements.txt").write_text(
        ci_requirements._FINGERPRINT_PREFIX
        + ci_requirements._fingerprint(tmp_path)
        + "\nanyio==4.14.2\n"
    )
    _commit(tmp_path, "base")
    return tmp_path


@pytest.mark.parametrize("force_legacy", [False, True])
def test_clean_merge_preserves_checkout_index_and_untracked_work(
    repository: Path, monkeypatch, force_legacy: bool
) -> None:
    if force_legacy:
        monkeypatch.setattr(
            ci_merge_precheck, "_supports_merge_tree_write_tree", lambda root: False
        )
    base = _git(repository, "rev-parse", "HEAD")
    _git(repository, "checkout", "-qb", "candidate")
    (repository / "candidate.txt").write_text("candidate\n")
    head = _commit(repository, "candidate")
    _git(repository, "checkout", "-q", "main")
    (repository / "base.txt").write_text("base\n")
    current_base = _commit(repository, "new base")
    _git(repository, "checkout", "-q", "candidate")
    (repository / "candidate.txt").write_text("uncommitted\n")
    (repository / "private.txt").write_text("untracked\n")
    before = _git(repository, "status", "--porcelain")
    index = _git(repository, "write-tree")
    report = ci_merge_precheck.check_merge(repository, "main", "HEAD")
    assert report["base"] == current_base != base
    assert report["head"] == head
    assert _git(repository, "show", f"{report['tree']}:candidate.txt") == "candidate"
    assert _git(repository, "show", f"{report['tree']}:base.txt") == "base"
    assert _git(repository, "status", "--porcelain") == before
    assert _git(repository, "write-tree") == index
    assert _git(repository, "rev-parse", "HEAD") == head


@pytest.mark.parametrize("force_legacy", [False, True])
def test_conflicts_are_rejected_without_starting_a_merge(
    repository: Path, monkeypatch, force_legacy: bool
) -> None:
    if force_legacy:
        monkeypatch.setattr(
            ci_merge_precheck, "_supports_merge_tree_write_tree", lambda root: False
        )
    _git(repository, "checkout", "-qb", "candidate")
    path = repository / "npa/ci/constraints.in"
    path.write_text("candidate change\n")
    _commit(repository, "candidate")
    _git(repository, "checkout", "-q", "main")
    path.write_text("main change\n")
    _commit(repository, "main")
    with pytest.raises(ValueError, match=r"Merge conflicts: npa/ci/constraints.in"):
        ci_merge_precheck.check_merge(repository, "main", "candidate")
    assert not (repository / ".git/MERGE_HEAD").exists()
    assert _git(repository, "status", "--porcelain") == ""


def test_wrong_fingerprint_after_conflict_resolution_is_rejected(
    repository: Path,
) -> None:
    base = _git(repository, "rev-parse", "HEAD")
    path = repository / "npa/pyproject.toml"
    path.write_text(path.read_text().replace("anyio>=4.14.2", "anyio>=4.15.1"))
    _commit(repository, "keep stale fingerprint while changing inputs")
    with pytest.raises(ValueError, match="CI dependencies changed"):
        ci_merge_precheck.check_merge(repository, base, "HEAD")


@pytest.mark.parametrize("reference", ["missing-main", "--help"])
def test_missing_or_option_shaped_revisions_fail(
    repository: Path, reference: str
) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        ci_merge_precheck.check_merge(repository, reference, "HEAD")
