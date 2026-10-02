"""Unit tests for the agent auth-secret helpers extracted from the
``npa.cli.agent`` god object into ``npa.cli.agent_auth`` (issue #491, PR #575).

Behavior is pinned identical to the pre-extraction implementation; the tests
import through the NEW module layout and separately assert the
``npa.cli.agent`` re-exports still resolve to the same objects.
"""

import os
import stat

import pytest

from npa.cli import agent_auth
from npa.cli.agent_auth import (
    _auth_secret_path,
    _cleanup_agent_local_files,
    _load_auth_secret,
    _write_auth_secret,
)


@pytest.fixture()
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NPA_CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_auth_secret_path_uses_config_dir(config_dir):
    path = _auth_secret_path("proj", "agent1")
    assert path == config_dir / "agents" / "proj" / "agent1" / "auth.env"


def test_auth_secret_path_falls_back_to_home(tmp_path, monkeypatch):
    monkeypatch.delenv("NPA_CONFIG_DIR", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    path = _auth_secret_path("proj", "agent1")
    assert path == tmp_path / ".npa" / "agents" / "proj" / "agent1" / "auth.env"


def test_write_then_load_roundtrip(config_dir):
    written = _write_auth_secret(
        project_alias="proj", name="agent1", user="alice", password="s3cret"
    )
    assert written.exists()
    # Credential file must not be world/group readable.
    mode = stat.S_IMODE(os.stat(written).st_mode)
    assert mode == 0o600
    assert _load_auth_secret(str(written)) == ("alice", "s3cret")


def test_load_auth_secret_missing_file(tmp_path):
    with pytest.raises(ValueError, match="auth secret not found"):
        _load_auth_secret(str(tmp_path / "nope" / "auth.env"))


def test_load_auth_secret_missing_keys(tmp_path):
    path = tmp_path / "auth.env"
    path.write_text("AGENT_USER=alice\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing AGENT_USER/AGENT_PASSWORD"):
        _load_auth_secret(str(path))


def test_load_auth_secret_ignores_malformed_lines(tmp_path):
    path = tmp_path / "auth.env"
    path.write_text(
        "garbage line\nAGENT_USER = alice \n# comment\nAGENT_PASSWORD=s3cret\n",
        encoding="utf-8",
    )
    assert _load_auth_secret(str(path)) == ("alice", "s3cret")


def test_cleanup_agent_local_files_removes_trees(config_dir, monkeypatch):
    # Keep the test hermetic: stub out the Terraform workdir lookup.
    import npa.deploy.provisioner as provisioner

    tf_dir = config_dir / "workbenches" / "proj" / "agent1"
    tf_dir.mkdir(parents=True)
    monkeypatch.setattr(provisioner, "working_dir_path", lambda alias, name: tf_dir)

    agent_dir = _auth_secret_path("proj", "agent1").parent
    agent_dir.mkdir(parents=True)
    (agent_dir / "auth.env").write_text("x", encoding="utf-8")

    _cleanup_agent_local_files("proj", "agent1")

    assert not agent_dir.exists()
    assert not tf_dir.exists()
    # Empty alias parents are pruned too.
    assert not (config_dir / "agents" / "proj").exists()
    assert not (config_dir / "workbenches" / "proj").exists()


def test_cleanup_agent_local_files_keeps_sibling_alias_parent(config_dir, monkeypatch):
    import npa.deploy.provisioner as provisioner

    tf_dir = config_dir / "workbenches" / "proj" / "agent1"
    tf_dir.mkdir(parents=True)
    monkeypatch.setattr(provisioner, "working_dir_path", lambda alias, name: tf_dir)

    agent_dir = _auth_secret_path("proj", "agent1").parent
    agent_dir.mkdir(parents=True)
    sibling = config_dir / "agents" / "proj" / "agent2"
    sibling.mkdir(parents=True)

    _cleanup_agent_local_files("proj", "agent1")

    assert not agent_dir.exists()
    # Sibling agent keeps the alias parent alive.
    assert (config_dir / "agents" / "proj").is_dir()


def test_reexports_resolve_to_new_module_objects():
    from npa.cli import agent as agent_module

    for name in (
        "_auth_secret_path",
        "_cleanup_agent_local_files",
        "_load_auth_secret",
        "_write_auth_secret",
    ):
        assert getattr(agent_module, name) is getattr(agent_auth, name), name
