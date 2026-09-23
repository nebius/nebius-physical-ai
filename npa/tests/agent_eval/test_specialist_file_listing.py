"""Verify scoped file discovery without disclosing ungranted workspace names."""

import json
import os
import uuid

import pytest

from npa.agent_backend.specialists.config import Profile
from npa.agent_backend.specialists.store import TaskStore
from npa.agent_backend.specialists.tools import WorkbenchTools


@pytest.fixture
def listing(tmp_path):
    workspace = tmp_path / "workspace"
    files = {
        "src/public/read.py": "readable",
        "src/public/edit.py": "editable",
        "src/public/secret.py": "private sibling",
        "src/hidden/secret.txt": "private subtree",
        "docs/guide.md": "guide",
    }
    for name, content in files.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    profile = Profile(
        name="reader",
        description="Inspect authorized source",
        model="synthetic",
        workspace=workspace,
        read_paths=["src/public/read.py", "docs/guide.md"],
        write_paths=["src/public/edit.py", "src/new.py"],
    )
    return WorkbenchTools(profile, TaskStore(tmp_path / "state"), "listing")


def _call(tools, path, name="list_files", **arguments):
    return tools.execute(
        {
            "id": str(uuid.uuid4()),
            "function": {
                "name": name,
                "arguments": json.dumps({"path": path, **arguments}),
            },
        }
    )


@pytest.mark.parametrize(
    "directory,expected",
    [
        (".", ["docs/guide.md", "src/public/edit.py", "src/public/read.py"]),
        ("src", ["src/public/edit.py", "src/public/read.py"]),
        ("src/public", ["src/public/edit.py", "src/public/read.py"]),
        ("docs", ["docs/guide.md"]),
    ],
)
def test_exact_file_grants_can_be_discovered_from_ancestors(
    listing, directory, expected
):
    assert _call(listing, directory) == {"ok": True, "paths": expected}


def test_discovery_does_not_expand_reads_or_writes(listing):
    assert _call(listing, ".")["ok"] is True
    assert _call(listing, "src/public/secret.py", "read_file")["ok"] is False
    result = _call(
        listing,
        "src/public/secret.py",
        "edit_file",
        expected_sha256="unused",
        old="private sibling",
        new="changed",
    )
    assert result["ok"] is False
    assert (listing.root / "src/public/secret.py").read_text() == "private sibling"


@pytest.mark.parametrize(
    "path", ["../outside", "/etc", ".git", "src/hidden", "src/public/secret.py"]
)
def test_unauthorized_queries_reveal_no_names(listing, path):
    result = _call(listing, path)
    assert result["ok"] is False
    assert "paths" not in result
    assert "private" not in json.dumps(result)


def test_directory_grants_keep_descendants_without_traversing_private_subtrees(
    listing, monkeypatch
):
    listing.profile.read_paths = ["src/public"]
    listing.profile.write_paths = []
    visited = []
    original = os.scandir

    def observe(path):
        visited.append(str(path))
        return original(path)

    monkeypatch.setattr(os, "scandir", observe)
    result = _call(listing, ".")
    assert result["paths"] == [
        "src/public/edit.py",
        "src/public/read.py",
        "src/public/secret.py",
    ]
    assert str(listing.root / "src/hidden") not in visited
    assert str(listing.root / "docs") not in visited


def test_links_and_git_metadata_are_hidden_even_with_directory_grant(listing, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "private.txt"
    secret.write_text("private")
    public = listing.root / "src/public"
    (public / "external").symlink_to(outside, target_is_directory=True)
    (public / "internal").symlink_to(listing.root / "docs", target_is_directory=True)
    (public / "link.txt").symlink_to(secret)
    (public / "hard.txt").hardlink_to(secret)
    (public / ".git").mkdir()
    (public / ".git/config").write_text("private metadata")
    listing.profile.read_paths = ["src/public"]
    listing.profile.write_paths = []
    assert _call(listing, ".")["paths"] == [
        "src/public/edit.py",
        "src/public/read.py",
        "src/public/secret.py",
    ]
    assert _call(listing, "src/public/external")["ok"] is False


def test_exact_grant_with_symlink_ancestor_cannot_escape(listing, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("private")
    (listing.root / "alias").symlink_to(outside, target_is_directory=True)
    listing.profile.read_paths = ["alias/private.txt"]
    listing.profile.write_paths = []
    assert _call(listing, ".") == {"ok": True, "paths": []}
    assert _call(listing, "alias")["ok"] is False


def test_missing_grants_and_empty_grants_do_not_disclose_workspace(listing):
    listing.profile.read_paths = ["src/not-created.py"]
    listing.profile.write_paths = []
    assert _call(listing, ".") == {"ok": True, "paths": []}
    listing.profile.read_paths = []
    assert _call(listing, ".")["ok"] is False


def test_git_file_and_exact_hardlink_grant_are_not_listed(listing, tmp_path):
    (listing.root / ".git").write_text("gitdir: private-metadata")
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (listing.root / "hard.txt").hardlink_to(outside)
    listing.profile.read_paths = ["hard.txt"]
    listing.profile.write_paths = []
    assert _call(listing, ".") == {"ok": True, "paths": []}
