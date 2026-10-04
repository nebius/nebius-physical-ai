"""Keep paged source reads scoped and bind their edits to the complete file."""

import hashlib
import json
import uuid

import pytest

from npa.agent_backend.specialists.config import Profile
from npa.agent_backend.specialists.store import TaskStore
from npa.agent_backend.specialists.tools import WorkbenchTools


@pytest.fixture
def reader(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    profile = Profile(
        name="reader",
        description="Inspect source",
        model="synthetic",
        workspace=workspace,
        read_paths=["src"],
        write_paths=["src"],
    )
    path = workspace / "src/value.txt"
    path.write_text("first\nβeta\nlast")
    return WorkbenchTools(profile, TaskStore(tmp_path / "state"), "read"), path


def _call(tools, name="read_file", **arguments):
    return tools.execute(
        {
            "id": str(uuid.uuid4()),
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
    )


def test_range_uses_full_file_hash_and_keeps_newlines(reader):
    tools, path = reader
    result = _call(tools, path="src/value.txt", start_line=2, end_line=2)
    assert result["content"] == "βeta\n"
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (result["start_line"], result["end_line"], result["total_lines"]) == (
        2,
        2,
        3,
    )
    edited = _call(
        tools,
        "edit_file",
        path="src/value.txt",
        expected_sha256=result["sha256"],
        old="βeta",
        new="second",
    )
    assert edited["ok"] is True
    assert path.read_text() == "first\nsecond\nlast"


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, "first\nβeta\nlast"),
        ({"start_line": 2}, "βeta\nlast"),
        ({"start_line": 3, "end_line": 99}, "last"),
    ],
)
def test_default_and_open_ended_reads(reader, options, expected):
    result = _call(reader[0], path="src/value.txt", **options)
    assert result["ok"] is True
    assert result["content"] == expected
    assert result["end_line"] == 3


@pytest.mark.parametrize(
    "options",
    [
        {"start_line": 0},
        {"end_line": 0},
        {"start_line": 3, "end_line": 2},
        {"start_line": 4},
        {"start_line": True},
        {"end_line": "2"},
        {"start_line": 1.5},
    ],
)
def test_invalid_ranges_fail_without_file_contents(reader, options):
    result = _call(reader[0], path="src/value.txt", **options)
    assert result["ok"] is False
    assert "content" not in result


def test_empty_and_missing_files(reader):
    tools, path = reader
    path.write_text("")
    result = _call(tools, path="src/value.txt")
    assert result["ok"] is True and result["content"] == ""
    assert result["total_lines"] == result["end_line"] == 0
    path.unlink()
    assert _call(tools, path="src/value.txt", start_line=2)["ok"] is False


def test_changes_outside_read_range_still_reject_stale_edit(reader):
    tools, path = reader
    result = _call(tools, path="src/value.txt", start_line=2, end_line=2)
    path.write_text("changed elsewhere\nβeta\nlast")
    edited = _call(
        tools,
        "edit_file",
        path="src/value.txt",
        expected_sha256=result["sha256"],
        old="βeta",
        new="second",
    )
    assert edited["ok"] is False
    assert path.read_text() == "changed elsewhere\nβeta\nlast"


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", ".git/config"])
def test_ranges_do_not_expand_path_grants(reader, path):
    assert _call(reader[0], path=path, start_line=1, end_line=1)["ok"] is False
