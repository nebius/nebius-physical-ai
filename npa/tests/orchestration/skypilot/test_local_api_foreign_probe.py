from pathlib import Path

import pytest

from npa.orchestration.skypilot import workflow


def _foreign_process(root: Path, cwd: Path, *, port: str | None = None):
    process = root / "42"
    process.mkdir(parents=True)
    args = ["/other-managed/bin/python", "-m", "sky.server.server"]
    if port:
        args += ["--port", port]
    (process / "cmdline").write_bytes(b"\0".join(value.encode() for value in args))
    (process / "status").write_text("Uid:\t1234\t1234\t1234\t1234\nPPid:\t1\n")
    (process / "environ").write_bytes(b"HOME=/other-home\0")
    (process / "cwd").symlink_to(cwd)


def test_foreign_default_api_is_not_mistaken_for_no_daemon_or_stopped(tmp_path):
    root = tmp_path / "proc"
    _foreign_process(root, tmp_path)
    result = workflow._probe_local_api_daemon_cwd("/selected-managed/bin/sky", proc_root=root, uid=1234)
    assert not result.healthy
    assert result.outcome == "foreign_api_daemon"
    calls = []
    with pytest.raises(workflow.SkyPilotSubmitError, match="foreign_api_daemon"):
        workflow._ensure_local_api_daemon_cwd("/selected-managed/bin/sky", env={}, cwd=str(tmp_path),
            probe=lambda: result, runner=lambda *args, **kwargs: calls.append(args))
    assert calls == []


def test_owned_custom_port_does_not_interfere_with_other_default_api_probe(tmp_path):
    root = tmp_path / "proc"
    _foreign_process(root, tmp_path, port="49152")
    result = workflow._probe_local_api_daemon_cwd("/selected-managed/bin/sky", proc_root=root, uid=1234)
    assert result.healthy
    assert result.outcome == "absent"


def test_shared_network_namespace_is_checked_across_mount_namespaces(tmp_path):
    root = tmp_path / "proc"
    _foreign_process(root, tmp_path)
    for pid, mount in (("self", "mnt:[1]"), ("42", "mnt:[2]")):
        namespaces = root / pid / "ns"
        namespaces.mkdir(parents=True)
        (namespaces / "mnt").symlink_to(mount)
        (namespaces / "net").symlink_to("net:[1]")
    result = workflow._probe_local_api_daemon_cwd("/selected-managed/bin/sky", proc_root=root, uid=1234)
    assert result.outcome == "foreign_api_daemon"
