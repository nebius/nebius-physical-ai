"""Exercise recovery source binding against real hostile local Git repositories."""

import hashlib
import os
from pathlib import Path
import shlex
import subprocess

import pytest

from npa.cluster.absent_evidence import AbsenceRecoveryError, _source_binding


PATHS = (
    "npa/src/npa/cli/cluster/terraform_lifecycle.py",
    "npa/src/npa/cluster_backends/mk8s.py",
    "npa/src/npa/cluster_backends/mk8s_execution.py",
)


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "producer"
    root.mkdir()
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    def git(*args, **kwargs):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], env=env, stderr=subprocess.PIPE, **kwargs
        )

    git("init")
    hashes = {}
    for name in PATHS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original source: " + name + "\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    git("add", ".")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "Original producer source",
    )
    revision = git("rev-parse", "HEAD").decode().strip()
    return root, revision, hashes, git


def test_source_binding_reads_committed_bytes_and_accepts_real_worktrees(repository):
    root, revision, expected, git = repository
    (root / PATHS[0]).write_text("# uncommitted source\n")
    assert _source_binding({"producer_repository": str(root)}, revision) == expected
    worktree = root.parent / "worktree"
    git("worktree", "add", "--detach", str(worktree), revision)
    assert (worktree / ".git").is_file()
    assert _source_binding({"producer_repository": str(worktree)}, revision) == expected


def test_source_binding_does_not_follow_replacement_objects(repository):
    root, revision, expected, git = repository
    original = git("rev-parse", revision + ":" + PATHS[0]).decode().strip()
    replacement = (
        git("hash-object", "-w", "--stdin", input=b"forged\n").decode().strip()
    )
    git("replace", original, replacement)
    assert git("show", revision + ":" + PATHS[0]) == b"forged\n"
    assert _source_binding({"producer_repository": str(root)}, revision) == expected


def test_missing_source_cannot_execute_promisor_remote_helper(repository):
    root, revision, _expected, git = repository
    marker = root.parent / "helper-executed"
    helper = root.parent / "remote-helper"
    helper.write_text("#!/bin/sh\ntouch " + shlex.quote(str(marker)) + "\nexit 1\n")
    helper.chmod(0o700)
    for key, value in (
        ("core.repositoryformatversion", "1"),
        ("extensions.partialClone", "origin"),
        ("remote.origin.promisor", "true"),
        ("remote.origin.url", "ext::" + str(helper)),
        ("protocol.ext.allow", "always"),
    ):
        git("config", key, value)
    blob = git("rev-parse", revision + ":" + PATHS[0]).decode().strip()
    (root / ".git/objects" / blob[:2] / blob[2:]).unlink()
    with pytest.raises(subprocess.CalledProcessError):
        git("show", revision + ":" + PATHS[0])
    assert marker.is_file(), "Control must demonstrate the configured helper runs"
    marker.unlink()
    with pytest.raises(subprocess.CalledProcessError):
        _source_binding({"producer_repository": str(root)}, revision)
    assert not marker.exists()


def test_source_binding_scrubs_ambient_repository_and_config(repository, monkeypatch):
    root, revision, expected, _git = repository
    monkeypatch.setenv("GIT_DIR", str(root.parent / "foreign.git"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(root.parent / "foreign-objects"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.repositoryformatversion")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "invalid")
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")
    assert _source_binding({"producer_repository": str(root)}, revision) == expected


@pytest.mark.parametrize(
    "kind", ["relative", "linked-root", "linked-marker", "not-git"]
)
def test_source_binding_rejects_ambiguous_checkout_paths(repository, kind, monkeypatch):
    root, revision, _expected, _git = repository
    if kind == "relative":
        monkeypatch.chdir(root.parent)
        root = Path(root.name)
    elif kind == "linked-root":
        linked = root.parent / "alias"
        linked.symlink_to(root, target_is_directory=True)
        root = linked
    elif kind == "linked-marker":
        saved = root.parent / "saved.git"
        (root / ".git").rename(saved)
        (root / ".git").symlink_to(saved, target_is_directory=True)
    else:
        root = root.parent
    with pytest.raises(AbsenceRecoveryError, match="canonical absolute Git checkout"):
        _source_binding({"producer_repository": str(root)}, revision)
