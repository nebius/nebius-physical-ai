"""Exercise worker identity, mount, credential and firewall guards without root access."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "workflows/workbench/cosmos3-wam-slurm"


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(RECIPE))
    spec = importlib.util.spec_from_file_location(
        "wam_worker", RECIPE / "slurm_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    etc = tmp_path / "etc"
    (etc / "slurm").mkdir(parents=True)
    (etc / "fstab").write_text("# existing mounts\n")

    def private_path(*parts):
        path = Path(*parts)
        return etc / path.relative_to("/etc") if path.is_relative_to("/etc") else path

    monkeypatch.setattr(module, "Path", private_path)
    monkeypatch.setattr(module, "_run", Mock())
    monkeypatch.setattr(
        module,
        "pwd",
        SimpleNamespace(
            getpwnam=Mock(side_effect=KeyError), getpwuid=Mock(side_effect=KeyError)
        ),
    )
    monkeypatch.setattr(
        module, "grp", SimpleNamespace(getgrgid=Mock(side_effect=KeyError))
    )
    return module


@pytest.fixture
def peers(tmp_path):
    return dict(
        slurm_uid=123,
        slurm_gid=124,
        user_name="synthetic-operator",
        user_uid=1000,
        user_home="/home/synthetic-operator",
        worker_name="synthetic-worker",
        controller_address="10.0.0.1",
        worker_address="10.0.0.2",
        shared_root=str(tmp_path / "shared"),
    )


@pytest.fixture
def worker_bundle(worker, peers, tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "peers.json").write_text(json.dumps(peers))
    monkeypatch.setattr(worker, "os", SimpleNamespace(geteuid=lambda: 0))
    monkeypatch.setattr(
        worker, "socket", SimpleNamespace(gethostname=lambda: peers["worker_name"])
    )
    monkeypatch.setattr(
        worker,
        "subprocess",
        SimpleNamespace(check_output=Mock(return_value="NVIDIA B200\n" * 8)),
    )
    worker.pwd.getpwuid.side_effect = None
    worker.pwd.getpwuid.return_value = SimpleNamespace(
        pw_name=peers["user_name"], pw_dir=peers["user_home"]
    )
    return bundle


def test_matching_fresh_worker_is_validated_without_mutation(
    worker, worker_bundle, peers
):
    assert worker._validate(worker_bundle) == peers
    worker._run.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    ["root", "configured", "hostname", "user", "home", "gpu_count", "gpu_type"],
)
def test_worker_bundle_rejects_mismatched_hosts(
    worker, worker_bundle, failure, tmp_path
):
    if failure == "root":
        worker.os.geteuid = lambda: 1000
    elif failure == "configured":
        (tmp_path / "etc/slurm/slurm.conf").touch()
    elif failure == "hostname":
        worker.socket.gethostname = lambda: "other-worker"
    elif failure in ("user", "home"):
        attribute = "pw_name" if failure == "user" else "pw_dir"
        setattr(worker.pwd.getpwuid.return_value, attribute, "different")
    else:
        worker.subprocess.check_output.return_value = (
            "NVIDIA B200\n" * 7 if failure == "gpu_count" else "NVIDIA H100\n" * 8
        )
    with pytest.raises((ValueError, RuntimeError)):
        worker._validate(worker_bundle)
    worker._run.assert_not_called()


def test_existing_matching_service_identity_is_reused(worker, peers):
    worker.pwd.getpwnam.side_effect = None
    worker.pwd.getpwnam.return_value = SimpleNamespace(pw_uid=123, pw_gid=124)
    worker._prepare_user(peers)
    worker._run.assert_not_called()
    worker.pwd.getpwuid.assert_not_called()


@pytest.mark.parametrize(("uid", "gid"), [(125, 124), (123, 125)])
def test_existing_service_identity_must_match_controller(worker, peers, uid, gid):
    worker.pwd.getpwnam.side_effect = None
    worker.pwd.getpwnam.return_value = SimpleNamespace(pw_uid=uid, pw_gid=gid)
    with pytest.raises(ValueError, match="service identity differs"):
        worker._prepare_user(peers)
    worker._run.assert_not_called()


@pytest.mark.parametrize("lookup", ["uid", "gid"])
def test_occupied_service_identity_is_not_reassigned(worker, peers, lookup):
    probe = worker.pwd.getpwuid if lookup == "uid" else worker.grp.getgrgid
    probe.side_effect = None
    probe.return_value = SimpleNamespace()
    with pytest.raises(ValueError, match="already assigned"):
        worker._prepare_user(peers)
    worker._run.assert_not_called()


def test_unused_controller_identity_creates_only_a_service_user(worker, peers):
    worker._prepare_user(peers)
    assert worker._run.call_args_list == [
        call("groupadd", "--system", "--gid", "124", "slurm"),
        call(
            "useradd",
            "--system",
            "--uid",
            "123",
            "--gid",
            "slurm",
            "--home-dir",
            "/var/lib/slurm",
            "--shell",
            "/usr/sbin/nologin",
            "slurm",
        ),
    ]


@pytest.mark.parametrize(
    "root", ["/", "relative", "/srv/with space", "/srv/with\nnewline"]
)
def test_mount_rejects_invalid_roots_before_mutation(worker, peers, root, tmp_path):
    peers["shared_root"] = root
    with pytest.raises(ValueError, match="invalid shared root"):
        worker._mount(peers)
    worker._run.assert_not_called()
    assert (tmp_path / "etc/fstab").read_text() == "# existing mounts\n"


def test_mount_refuses_to_hide_existing_data(worker, peers):
    root = Path(peers["shared_root"])
    root.mkdir()
    (root / "keep.txt").write_text("retained")
    with pytest.raises(ValueError, match="populated directory"):
        worker._mount(peers)
    worker._run.assert_not_called()
    assert (root / "keep.txt").read_text() == "retained"


def test_failed_mount_does_not_persist_a_broken_entry(worker, peers, tmp_path):
    worker._run.side_effect = subprocess.CalledProcessError(32, "mount")
    with pytest.raises(subprocess.CalledProcessError):
        worker._mount(peers)
    assert (tmp_path / "etc/fstab").read_text() == "# existing mounts\n"
    assert not (tmp_path / "etc/systemd").exists()


def test_mount_preserves_fstab_and_orders_worker_after_storage(worker, peers, tmp_path):
    worker._mount(peers)
    root = peers["shared_root"]
    source = f"{peers['controller_address']}:{root}"
    options = "nfsvers=4.2,hard,proto=tcp,_netdev"
    assert worker._run.call_args_list == [
        call("mount", "-t", "nfs4", "-o", options, source, root),
        call(
            "sudo",
            "-u",
            peers["user_name"],
            "test",
            "-f",
            str(Path(root) / "prepared.json"),
        ),
    ]
    assert (tmp_path / "etc/fstab").read_text() == (
        f"# existing mounts\n{source} {root} nfs4 {options} 0 0\n"
    )
    dropin = tmp_path / "etc/systemd/system/slurmd.service.d/shared-storage.conf"
    assert dropin.read_text() == f"[Unit]\nRequiresMountsFor={root}\n"


def test_private_munge_key_is_installed_before_worker_start(worker, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("slurm.conf", "gres.conf", "cgroup.conf"):
        (bundle / name).write_text("synthetic config\n")
    worker._configuration(bundle)
    for name in ("slurm.conf", "gres.conf", "cgroup.conf"):
        path = tmp_path / "etc/slurm" / name
        assert path.read_bytes() == (bundle / name).read_bytes()
        assert path.stat().st_mode & 0o777 == 0o644
    commands = worker._run.call_args_list
    install = call(
        "install",
        "-o",
        "munge",
        "-g",
        "munge",
        "-m",
        "400",
        str(bundle / "munge.key"),
        "/etc/munge/munge.key",
    )
    start = call("systemctl", "enable", "--now", "munge", "slurmd")
    assert commands.index(call("systemctl", "stop", "munge")) < commands.index(install)
    assert commands.index(install) < commands.index(start)


def test_worker_firewall_limits_peers_before_default_drop(worker, peers):
    worker._firewall(peers)
    commands = worker._run.call_args_list
    drop = call("iptables", "-A", "WAM_INPUT", "-j", "DROP")
    for role in ("controller", "worker"):
        allow = call(
            "iptables",
            "-A",
            "WAM_INPUT",
            "-s",
            peers[role + "_address"],
            "-j",
            "ACCEPT",
        )
        assert commands.index(allow) < commands.index(drop)
    assert commands[-2] == call("iptables", "-I", "INPUT", "1", "-j", "WAM_INPUT")
    assert commands[-1] == call("bash", str(RECIPE / "persist-firewall.sh"))


def _firewall_sandbox(tmp_path, monkeypatch):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "sudo").write_text('#!/bin/sh\nexec "$@"\n')
    recorder = f"""#!{sys.executable}
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
with open({str(tmp_path / "commands.jsonl")!r}, 'a') as stream:
    stream.write(json.dumps([name, *sys.argv[1:]]) + '\\n')
if name == 'ip6tables' and os.environ.get('WAM_TEST_IPV6_FAILURE'):
    sys.exit(2)
if name.endswith('-save'):
    print('# synthetic ' + name)
if name == 'ip6tables' and '-C' in sys.argv:
    sys.exit(1)
"""
    for command in ("ip6tables", "iptables-save", "ip6tables-save", "systemctl"):
        (binaries / command).write_text(recorder)
    for executable in binaries.iterdir():
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binaries}{os.pathsep}{os.environ['PATH']}")
    etc = tmp_path / "etc"
    (etc / "systemd/system").mkdir(parents=True)
    # Redirect only filesystem destinations; execute the real shell control flow.
    script = (RECIPE / "persist-firewall.sh").read_text().replace("/etc/", f"{etc}/")
    return script, etc


def test_firewall_installs_ipv6_default_deny_and_restores_both_families(
    tmp_path, monkeypatch
):
    script, etc = _firewall_sandbox(tmp_path, monkeypatch)
    subprocess.run(["bash", "-s"], input=script, text=True, check=True)
    commands = [
        json.loads(line)
        for line in (tmp_path / "commands.jsonl").read_text().splitlines()
    ]
    ipv6 = [command[1:] for command in commands if command[0] == "ip6tables"]
    rules = [command for command in ipv6 if command[0] == "-A"]
    assert rules == [
        ["-A", "WAM_INPUT", "-i", "lo", "-j", "ACCEPT"],
        [
            "-A",
            "WAM_INPUT",
            "-m",
            "conntrack",
            "--ctstate",
            "ESTABLISHED,RELATED",
            "-j",
            "ACCEPT",
        ],
        ["-A", "WAM_INPUT", "-p", "tcp", "--dport", "22", "-j", "ACCEPT"],
        ["-A", "WAM_INPUT", "-p", "ipv6-icmp", "-j", "ACCEPT"],
        ["-A", "WAM_INPUT", "-j", "DROP"],
    ]
    assert ipv6[-1] == ["-I", "INPUT", "1", "-j", "WAM_INPUT"]
    unit = (etc / "systemd/system/wam-slurm-firewall.service").read_text()
    for family in ("iptables", "ip6tables"):
        rules_path = etc / f"wam-slurm/{family}.rules"
        assert rules_path.read_text() == f"# synthetic {family}-save\n"
        assert rules_path.stat().st_mode & 0o777 == 0o600
        assert f"ExecStart=/usr/sbin/{family}-restore {rules_path}" in unit
    for service in (
        "slurmctld.service",
        "slurmd.service",
        "slurmdbd.service",
        "nfs-server.service",
        "rpcbind.service",
        "rpcbind.socket",
    ):
        dropin = etc / "systemd/system" / f"{service}.d/wam-firewall.conf"
        assert dropin.read_text() == (
            "[Unit]\nRequires=wam-slurm-firewall.service\n"
            "After=wam-slurm-firewall.service\n"
        )


def test_ipv6_filter_failure_cannot_report_success(tmp_path, monkeypatch):
    script, etc = _firewall_sandbox(tmp_path, monkeypatch)
    monkeypatch.setenv("WAM_TEST_IPV6_FAILURE", "1")
    result = subprocess.run(["bash", "-s"], input=script, text=True, check=False)
    assert result.returncode == 2
    assert not (etc / "wam-slurm").exists()
    assert not (etc / "systemd/system/wam-slurm-firewall.service").exists()
