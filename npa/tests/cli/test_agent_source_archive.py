"""Source deployment must not upload unrelated local or symlinked data."""

import os
import stat
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from npa.cli import agent_source_archive as subject
from npa.clients.config import ConfigError


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    names = ["npa/src/app.py", "deploy/cluster/main.tf", "workflows/example.yaml"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tracked source\n")

    def run(command, **kwargs):
        assert command[:6] == ["git", "--literal-pathspecs", "ls-files", "-z", "--cached", "--"]
        assert kwargs["cwd"] == tmp_path
        assert kwargs["check"] and kwargs["capture_output"]
        return SimpleNamespace(stdout=b"\0".join(os.fsencode(name) for name in names) + b"\0")

    monkeypatch.setattr(subject.subprocess, "run", run)
    monkeypatch.setattr(subject.tempfile, "tempdir", str(tmp_path))
    return tmp_path, names


def test_only_indexed_working_tree_bytes_are_archived(inventory):
    root, names = inventory
    (root / "npa/.env").write_text("PRIVATE_FIXTURE=not-a-secret\n")
    (root / "npa/customer-capture.json").write_text("private fixture\n")
    (root / names[0]).write_text("modified tracked source\n")
    (root / names[0]).chmod(0o4755)
    archive_path = Path(subject.create_agent_source_archive(root))
    try:
        assert stat.S_IMODE(archive_path.stat().st_mode) == 0o600
        with tarfile.open(archive_path) as archive:
            assert set(archive.getnames()) == set(names)
            assert archive.extractfile(names[0]).read() == b"modified tracked source\n"
            assert archive.getmember(names[0]).mode == 0o755
            assert all(member.uid == member.gid == 0 for member in archive.getmembers())
    finally:
        archive_path.unlink()


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "fifo", "missing"])
def test_non_regular_or_missing_source_fails_without_archive(inventory, kind):
    root, names = inventory
    path = root / names[0]
    path.unlink()
    if kind == "file_symlink":
        path.symlink_to(root / names[1])
    elif kind == "directory_symlink":
        path.parent.rmdir()
        path.parent.symlink_to(root / "deploy/cluster", target_is_directory=True)
    elif kind == "fifo":
        os.mkfifo(path)
    with pytest.raises(ConfigError, match="safely package"):
        subject.create_agent_source_archive(root)
    assert not list(root.glob("*.tar.gz"))


@pytest.mark.parametrize("name", ["/etc/passwd", "npa/../private", "npa//private", "private/file"])
def test_invalid_inventory_fails_before_archive(inventory, name):
    root, names = inventory
    names.append(name)
    with pytest.raises(ConfigError, match="Invalid path"):
        subject.create_agent_source_archive(root)
    assert not list(root.glob("*.tar.gz"))


def test_no_source_inventory_is_not_a_recursive_fallback(inventory, monkeypatch):
    root, _ = inventory

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(subject.subprocess, "run", fail)
    with pytest.raises(ConfigError, match="Git source inventory"):
        subject.create_agent_source_archive(root)
    assert not list(root.glob("*.tar.gz"))


def test_legacy_distill_deploy_uses_the_same_inventory(tmp_path, mocker, monkeypatch):
    monkeypatch.setenv("NPA_S3_BUCKET", "fixture-bucket")
    monkeypatch.setenv("NPA_PROJECT_ID", "project-fixture")
    from npa.workflows import distill_two_vm

    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"fixture archive")
    package = mocker.patch.object(
        distill_two_vm, "create_agent_source_archive", return_value=str(archive)
    )
    upload = mocker.patch.object(distill_two_vm, "_sftp_upload")
    ssh = mocker.MagicMock()
    ssh._config.user = "ubuntu"
    ssh.run.return_value = (0, "NPA_CLI_OK", "")
    distill_two_vm._setup_vm_in_directory(
        ssh, distill_two_vm.SIM_VM, "simulation", "/tmp/private-fixture"
    )
    package.assert_called_once_with(distill_two_vm._NPA_PACKAGE_ROOT.parent)
    upload.assert_any_call(ssh, str(archive), "/tmp/private-fixture/npa-src.tgz")
    assert not archive.exists()
    commands = [call.args[0] for call in ssh.run_or_raise.call_args_list]
    assert any("tar -xzf /tmp/private-fixture/npa-src.tgz -C /opt/npa/repo &&" in command for command in commands)
    assert "test -f /opt/npa/repo/npa/pyproject.toml" in commands
