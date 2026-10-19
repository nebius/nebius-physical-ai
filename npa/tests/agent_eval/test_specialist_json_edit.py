"""Edit scoped JSON strings while preserving redacted values and verification binding."""

import hashlib
import json

import pytest

from npa.agent_backend.specialists.config import Profile, TeamConfig
from npa.agent_backend.specialists.json_edit import _replace_json_text, _replace_json_value
from npa.agent_backend.specialists.team import SpecialistTeam
from npa.agent_backend.specialists.tools import WorkbenchTools


@pytest.fixture
def editor(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "config").mkdir(parents=True)
    data = {
        "shards": [
            {"output": "s3://example-bucket/outputs/shard-a/"},
            {"output": "s3://example-bucket/outputs/shard-a/"},
        ],
        "unchanged": {"token": "synthetic-sensitive-value"},
    }
    path = workspace / "config/plan.json"
    path.write_text(json.dumps(data, indent=2))
    profile = Profile(
        name="editor",
        description="Edit scoped configuration",
        model="synthetic/model",
        workspace=workspace,
        read_paths=["config"],
        write_paths=["config/plan.json"],
    )
    team = SpecialistTeam(
        TeamConfig(
            state_directory=tmp_path / "state",
            profiles=[profile],
            default_profile="editor",
        )
    )
    team.submit("Repair", specialist="editor", task_id="edit")
    return WorkbenchTools(profile, team.store, "edit"), path, data


def _call(tools, name, arguments, identifier):
    return tools.execute(
        {
            "id": identifier,
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
    )


def test_pointer_edit_preserves_hidden_prefix_and_unrelated_data(editor):
    tools, path, original = editor
    read = _call(tools, "read_file", {"path": "config/plan.json"}, "read")
    assert "s3://" not in read["content"]
    edited = _call(
        tools,
        "edit_file",
        {
            "path": "config/plan.json",
            "expected_sha256": read["sha256"],
            "json_pointer": "/shards/1/output",
            "old": "shard-a",
            "new": "shard-b",
        },
        "edit",
    )
    assert edited["ok"]
    expected = json.loads(json.dumps(original))
    expected["shards"][1]["output"] = "s3://example-bucket/outputs/shard-b/"
    assert json.loads(path.read_text()) == expected
    assert edited["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "s3://" not in json.dumps(edited)
    assert tools.store._originals("edit")[0]["path"] == "config/plan.json"
    assert tools.store._events("edit")[-1]["name"] == "edit_file"


def test_pointer_edit_reuses_receipt_and_rejects_stale_source(editor):
    tools, path, _ = editor
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    args = {
        "path": "config/plan.json",
        "expected_sha256": digest,
        "json_pointer": "/shards/1/output",
        "old": "shard-a",
        "new": "shard-b",
    }
    first = _call(tools, "edit_file", args, "one")
    assert _call(tools, "edit_file", args, "one") == first
    assert _call(tools, "edit_file", args, "two")["ok"] is False
    assert "changed" in _call(tools, "edit_file", args, "three")["error"]


def test_pointer_cannot_expand_path_grants(editor):
    tools, path, _ = editor
    result = _call(
        tools,
        "edit_file",
        {
            "path": "other.json",
            "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "json_pointer": "/shards/1/output",
            "old": "a",
            "new": "b",
        },
        "outside",
    )
    assert not result["ok"] and "scopes" in result["error"]


@pytest.mark.parametrize(
    "pointer",
    [
        "/shards/-/output",
        "/shards/01/output",
        "/shards/5/output",
        "/shards/~2/output",
        "shards/1/output",
    ],
)
def test_bad_pointer_never_changes_source(editor, pointer):
    tools, path, _ = editor
    original = path.read_bytes()
    result = _call(
        tools,
        "edit_file",
        {
            "path": "config/plan.json",
            "expected_sha256": hashlib.sha256(original).hexdigest(),
            "json_pointer": pointer,
            "old": "shard-a",
            "new": "shard-b",
        },
        "bad",
    )
    assert not result["ok"] and path.read_bytes() == original


@pytest.mark.parametrize(
    "content,pointer",
    [
        ('{"a":"x","a":"x"}', "/a"),
        ('{"a":NaN}', "/a"),
        ('{"a":3}', "/a"),
        ('{"a":"xx"}', "/a"),
    ],
)
def test_ambiguous_or_nonstring_values_are_rejected(content, pointer):
    with pytest.raises(ValueError):
        _replace_json_text(content, pointer, "x", "y")


def test_escaped_keys_arrays_and_root_json_string():
    result = _replace_json_text(
        '{"a/b":{"~key":["source"]}}\n', "/a~1b/~0key/0", "source", "target"
    )
    assert json.loads(result) == {"a/b": {"~key": ["target"]}}
    assert result.endswith("\n")
    assert (
        json.loads(_replace_json_text('"source"', "", "source", "target")) == "target"
    )


@pytest.mark.parametrize("value", [None, True, 3, 1.5, ["a", "b"], {"a": [None]}])
def test_native_json_assignment_supports_typed_values_and_root(value):
    assert json.loads(_replace_json_value('{"a/b":{"~key":[0]}}', "/a~1b/~0key/0", value)) == {"a/b": {"~key": [value]}}
    assert json.loads(_replace_json_value('{"old":1}', "", value)) == value


@pytest.mark.parametrize("content,pointer,value", [
    ('{"a":1,"a":2}', "/a", 3),
    ('{"a":NaN}', "/a", 3),
    ('{"a":1}', "/missing", 3),
    ('[0]', "/01", 3),
    ('[0]', "/-", 3),
    ('[0]', "/1", 3),
    ('{"a":1}', "/~2", 3),
    ('{"a":1}', "/a", float("inf")),
    ('{"a":1}', "/a", {"bad": float("nan")}),
])
def test_invalid_native_json_assignment_rejects_before_write(content, pointer, value):
    with pytest.raises(ValueError):
        _replace_json_value(content, pointer, value)


def test_native_assignment_preserves_grants_hash_binding_and_replay(editor):
    tools, path, original = editor
    read = _call(tools, "read_file", {"path": "config/plan.json"}, "read")
    assert "s3://" not in read["content"]
    args = {"path": "config/plan.json", "expected_sha256": read["sha256"],
            "json_pointer": "/shards/1/output", "value": "s3://example-bucket/outputs/shard-b/"}
    result = _call(tools, "replace_json", args, "replace")
    assert result["ok"] and result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    expected = json.loads(json.dumps(original))
    expected["shards"][1]["output"] = args["value"]
    assert json.loads(path.read_text()) == expected
    assert "s3://" not in json.dumps(result)
    assert _call(tools, "replace_json", args, "replace") == result
    assert not _call(tools, "replace_json", args, "stale")["ok"]
    assert not _call(tools, "replace_json", {**args, "path": "other.json"}, "outside")["ok"]
    assert tools.store._events("edit")[-3]["name"] == "replace_json"
    report_path = tools.store.directory / "artifacts/edit/changes.diff"
    assert report_path.exists() and tools.store._originals("edit")[0]["content"] == json.dumps(original, indent=2)


def test_native_assignment_requires_existing_file_and_required_value(editor):
    tools, path, _ = editor
    args = {"path": "config/plan.json", "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "json_pointer": ""}
    assert not _call(tools, "replace_json", args, "missing-value")["ok"]
    path.unlink()
    assert not _call(tools, "replace_json", {**args, "value": {}}, "missing-file")["ok"]
