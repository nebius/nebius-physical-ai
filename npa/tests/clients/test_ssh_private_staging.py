"""Exercise private upload permissions and failure cleanup over real SFTP."""
from __future__ import annotations

import io
import os
import socket
import stat
import threading
from pathlib import Path

import paramiko
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from npa.clients.config import SSHConfig
from npa.clients.ssh import SSHClient, SSHError


def _key():
    encoded = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH,
        serialization.NoEncryption(),
    ).decode()
    return paramiko.Ed25519Key.from_private_key(io.StringIO(encoded))


@pytest.fixture
def sftp_vm(tmp_path, monkeypatch):
    root = tmp_path / "remote"
    root.mkdir()
    (root / "tmp").mkdir(mode=0o1777)
    state = {"writes": [], "fail": "", "connections": [], "threads": []}
    host_key, user_key = _key(), _key()

    class Handle(paramiko.SFTPHandle):
        def write(self, offset, data):
            path = Path(self.filename)
            state["writes"].append({
                "parent_mode": stat.S_IMODE(path.parent.stat().st_mode),
                "file_mode": stat.S_IMODE(path.stat().st_mode),
            })
            if state["fail"] == "write":
                return paramiko.SFTP_FAILURE
            return super().write(offset, data)

    class Files(paramiko.SFTPServerInterface):
        def _path(self, path):
            # This fixture maps the protocol root into its own private sandbox.
            return root / path.lstrip("/")

        def _call(self, function, *args):
            try:
                function(*args)
                return paramiko.SFTP_OK
            except OSError as exc:
                return paramiko.SFTPServer.convert_errno(exc.errno)

        def mkdir(self, path, attr):
            return self._call(os.mkdir, self._path(path), attr.st_mode)

        def rmdir(self, path):
            return self._call(os.rmdir, self._path(path))

        def remove(self, path):
            return self._call(os.unlink, self._path(path))

        def lstat(self, path):
            try:
                return paramiko.SFTPAttributes.from_stat(self._path(path).lstat())
            except OSError as exc:
                return paramiko.SFTPServer.convert_errno(exc.errno)

        def chattr(self, path, attr):
            if state["fail"] == "chmod":
                return paramiko.SFTP_PERMISSION_DENIED
            return self._call(os.chmod, self._path(path), attr.st_mode)

        def open(self, path, flags, attr):
            try:
                local = self._path(path)
                fd = os.open(local, flags, attr.st_mode or 0o666)
                handle = Handle(flags)
                handle.filename = str(local)
                handle.writefile = os.fdopen(fd, "wb")
                return handle
            except OSError as exc:
                return paramiko.SFTPServer.convert_errno(exc.errno)

        def posix_rename(self, source, destination):
            if state["fail"] == "rename":
                return paramiko.SFTP_PERMISSION_DENIED
            return self._call(os.replace, self._path(source), self._path(destination))

    class Server(paramiko.ServerInterface):
        def check_auth_publickey(self, username, key):
            return paramiko.AUTH_SUCCESSFUL if key == user_key else paramiko.AUTH_FAILED

        def check_channel_request(self, kind, chanid):
            return paramiko.OPEN_SUCCEEDED if kind == "session" else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def connect(**_kwargs):
        local, remote = socket.socketpair()
        transport = paramiko.Transport(remote)
        transport.add_server_key(host_key)
        transport.set_subsystem_handler("sftp", paramiko.SFTPServer, Files)
        thread = threading.Thread(target=lambda: transport.start_server(server=Server()))
        thread.start()
        client_transport = paramiko.Transport(local)
        client_transport.connect(hostkey=host_key, username="fixture", pkey=user_key)
        client = paramiko.SSHClient()
        client._transport = client_transport
        state["connections"].extend([transport, client_transport])
        state["threads"].append(thread)
        return client

    ssh = SSHClient(SSHConfig(host="fixture", user="fixture", key_path="unused"))
    monkeypatch.setattr(ssh, "_connect", connect)
    yield ssh, root, state
    for transport in state["connections"]:
        transport.close()
    for thread in state["threads"]:
        thread.join()


def test_real_sftp_private_upload_is_atomic_and_private_during_every_write(sftp_vm):
    ssh, root, state = sftp_vm
    target = root / "tmp/config"
    target.write_text("original")
    assert ssh.upload_private_text("synthetic-value" * 100_000, "/tmp/config") == "/tmp/config"
    assert target.read_text() == "synthetic-value" * 100_000
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert state["writes"]
    assert all(event == {"parent_mode": 0o700, "file_mode": 0o600} for event in state["writes"])
    assert sorted(path.name for path in (root / "tmp").iterdir()) == ["config"]


@pytest.mark.parametrize("failure", ["chmod", "write", "rename"])
def test_real_sftp_upload_failure_preserves_existing_and_removes_staging(sftp_vm, failure):
    ssh, root, state = sftp_vm
    target = root / "tmp/config"
    target.write_text("original")
    state["fail"] = failure
    with pytest.raises(SSHError):
        ssh.upload_private_text("synthetic-value", "/tmp/config")
    assert target.read_text() == "original"
    assert sorted(path.name for path in (root / "tmp").iterdir()) == ["config"]


def test_real_sftp_replaces_symlink_without_writing_its_target(sftp_vm):
    ssh, root, _state = sftp_vm
    victim = root / "victim"
    victim.write_text("original")
    (root / "tmp/config").symlink_to(victim)
    ssh.upload_private_text("synthetic-value", "/tmp/config")
    assert victim.read_text() == "original"
    assert not (root / "tmp/config").is_symlink()


def test_real_sftp_refuses_attacker_precreated_staging_directory(sftp_vm, monkeypatch):
    ssh, root, state = sftp_vm
    from types import SimpleNamespace
    monkeypatch.setattr("npa.clients.ssh.uuid.uuid4", lambda: SimpleNamespace(hex="collision"))
    malicious = root / "tmp/.npa-stage-collision"
    malicious.mkdir(mode=0o777)
    victim = root / "victim"
    victim.write_text("original")
    (malicious / "payload").symlink_to(victim)
    with pytest.raises(SSHError):
        ssh.upload_private_text("synthetic-value", "/tmp/config")
    assert not state["writes"]
    assert victim.read_text() == "original"
    assert malicious.is_dir()


@pytest.mark.parametrize("failure", ["", "write"])
def test_real_sftp_token_file_is_private_and_removed_before_failed_exec(sftp_vm, failure):
    ssh, root, state = sftp_vm
    ssh._config.tokens = {"HF_TOKEN": "synthetic-value"}
    state["fail"] = failure
    # The real SSH server refuses command execution after accepting SFTP.
    with pytest.raises((paramiko.SSHException, OSError)):
        ssh.run("true")
    assert all(event == {"parent_mode": 0o700, "file_mode": 0o600} for event in state["writes"])
    assert list((root / "tmp").iterdir()) == []


@pytest.mark.parametrize("failure", [False, True])
def test_persistent_env_install_uses_real_atomic_coreutils_and_cleans_up(tmp_path, failure):
    """Run the actual destination-filesystem installation shell locally."""
    import getpass
    import shlex
    import subprocess
    import tempfile
    from contextlib import contextmanager
    from npa.deploy.configurator import write_remote_env_file

    commands = []
    staging = []

    class LocalSSH:
        @contextmanager
        def temporary_directory(self):
            with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
                staging.append(Path(directory))
                yield directory

        def upload_private_text(self, content, remote_path):
            assert stat.S_IMODE(Path(remote_path).parent.stat().st_mode) == 0o700
            with open(os.open(remote_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w") as output:
                output.write(content)

        def run_or_raise(self, command, **_kwargs):
            commands.append(command)
            # Retain actual mkdir/mktemp/install/mv/rm. Only privilege escalation
            # is unnecessary because this fixture owns its destination already.
            result = subprocess.run(
                ["bash", "-c", 'sudo() { "$@"; }; export -f sudo; ' + command],
                capture_output=True, text=True, check=False,
            )
            if result.returncode:
                raise SSHError("synthetic install failure")
            return result.returncode, result.stdout, result.stderr

    target = tmp_path / "destination" / "config"
    target.parent.mkdir()
    target.write_text("original")
    options = {"owner": "npa-no-such-user-fixture" if failure else getpass.getuser()}
    if failure:
        with pytest.raises(SSHError, match="install failure"):
            write_remote_env_file(LocalSSH(), str(target), {"HF_TOKEN": "synthetic-value"}, **options)
        assert target.read_text() == "original"
    else:
        write_remote_env_file(LocalSSH(), str(target), {"HF_TOKEN": "synthetic-value"}, **options)
        assert target.read_text() == "HF_TOKEN='synthetic-value'\n"
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert "mv -fT" in shlex.split(commands[-1])[-1]
    assert all("synthetic-value" not in command for command in commands)
    assert not any(path.exists() for path in staging)
    assert list(target.parent.iterdir()) == [target]


def test_concurrent_real_sftp_uploads_publish_one_complete_file(sftp_vm):
    from concurrent.futures import ThreadPoolExecutor
    ssh, root, state = sftp_vm
    candidates = ["first-value" * 100_000, "second-value" * 100_000]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(ssh.upload_private_text, value, "/tmp/config") for value in candidates]
        assert [future.result() for future in futures] == ["/tmp/config", "/tmp/config"]
    assert (root / "tmp/config").read_text() in candidates
    assert all(event == {"parent_mode": 0o700, "file_mode": 0o600} for event in state["writes"])
    assert sorted(path.name for path in (root / "tmp").iterdir()) == ["config"]


def test_legacy_distill_credentials_use_private_installer_and_preserve_env(monkeypatch, mocker):
    import importlib
    monkeypatch.setenv("NPA_PROJECT_ID", "project-fixture")
    monkeypatch.setenv("NPA_S3_BUCKET", "bucket-fixture")
    distill = importlib.import_module("npa.workflows.distill_two_vm")
    ssh = mocker.MagicMock()
    ssh._config.user = "fixture"
    ssh.run.return_value = (0, "KEEP='quoted value'\nAWS_SECRET_ACCESS_KEY=old\n", "")
    install = mocker.patch.object(distill, "write_remote_text_file")
    credential = "synthetic-value '$()' `literal`"
    distill._write_s3_env(ssh, {"nebius_secret_key": credential}, "fixture")
    content = install.call_args.args[2]
    assert "KEEP='quoted value'" in content
    assert "AWS_SECRET_ACCESS_KEY=old" not in content
    assert f"AWS_SECRET_ACCESS_KEY={credential}\n" in content
    assert install.call_args.kwargs == {"owner": "fixture", "mode": "0600"}
    assert all(credential not in call.args[0] for call in ssh.run.call_args_list)
    ssh.run_or_raise.assert_not_called()


def test_legacy_distill_rejects_multiline_docker_credential_before_install(monkeypatch, mocker):
    import importlib

    monkeypatch.setenv("NPA_PROJECT_ID", "project-fixture")
    monkeypatch.setenv("NPA_S3_BUCKET", "bucket-fixture")
    distill = importlib.import_module("npa.workflows.distill_two_vm")
    ssh = mocker.MagicMock()
    ssh.run.return_value = (0, "KEEP=literal\n", "")
    install = mocker.patch.object(distill, "write_remote_text_file")
    with pytest.raises(distill.TwoVMDistillError, match="newline or NUL"):
        distill._write_s3_env(ssh, {"nebius_secret_key": "synthetic\nINJECTED=1"}, "fixture")
    install.assert_not_called()
