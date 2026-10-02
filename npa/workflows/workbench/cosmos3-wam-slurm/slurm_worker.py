"""Install a private controller bundle on a fresh dedicated B200 Slurm worker."""

import argparse
import grp
import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import subprocess

from add_worker import _hosts, _private_address


def _run(*command):
    subprocess.run(command, check=True)


def _validate(bundle):
    if os.geteuid() != 0 or Path("/etc/slurm/slurm.conf").exists():
        raise RuntimeError("requires root on a fresh worker")
    peers = json.loads((bundle / "peers.json").read_text())
    if socket.gethostname().split(".")[0] != peers["worker_name"]:
        raise ValueError("worker hostname differs from the controller bundle")
    for role in ("controller", "worker"):
        _private_address(peers[role + "_address"])
    user = pwd.getpwuid(peers["user_uid"])
    if user.pw_name != peers["user_name"] or user.pw_dir != peers["user_home"]:
        raise ValueError(
            "worker user identity/home differs from the shared environment"
        )
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(names) != 8 or any("B200" not in name for name in names):
        raise ValueError("worker must expose eight B200 GPUs")
    return peers


def _firewall(peers):
    _run("iptables", "-N", "WAM_INPUT")
    _run("iptables", "-A", "WAM_INPUT", "-i", "lo", "-j", "ACCEPT")
    _run(
        "iptables",
        "-A",
        "WAM_INPUT",
        "-m",
        "conntrack",
        "--ctstate",
        "ESTABLISHED,RELATED",
        "-j",
        "ACCEPT",
    )
    _run("iptables", "-A", "WAM_INPUT", "-p", "tcp", "--dport", "22", "-j", "ACCEPT")
    for role in ("controller", "worker"):
        _run(
            "iptables",
            "-A",
            "WAM_INPUT",
            "-s",
            peers[role + "_address"],
            "-j",
            "ACCEPT",
        )
    _run("iptables", "-A", "WAM_INPUT", "-p", "icmp", "-j", "ACCEPT")
    _run("iptables", "-A", "WAM_INPUT", "-j", "DROP")
    _run("iptables", "-I", "INPUT", "1", "-j", "WAM_INPUT")
    _run("bash", str(Path(__file__).with_name("persist-firewall.sh")))


def _mount(peers):
    root = Path(peers["shared_root"])
    if (
        not root.is_absolute()
        or str(root) == "/"
        or any(character.isspace() for character in str(root))
    ):
        raise ValueError("invalid shared root")
    if root.exists() and any(root.iterdir()):
        raise ValueError("refusing to mount over an existing populated directory")
    root.mkdir(parents=True, exist_ok=True)
    remote = f"{peers['controller_address']}:{root}"
    options = "nfsvers=4.2,hard,proto=tcp,_netdev"
    _run("mount", "-t", "nfs4", "-o", options, remote, str(root))
    with Path("/etc/fstab").open("a") as stream:
        stream.write(f"{remote} {root} nfs4 {options} 0 0\n")
    dropin = Path("/etc/systemd/system/slurmd.service.d")
    dropin.mkdir(parents=True, exist_ok=True)
    (dropin / "shared-storage.conf").write_text(f"[Unit]\nRequiresMountsFor={root}\n")
    _run("sudo", "-u", peers["user_name"], "test", "-f", str(root / "prepared.json"))


def _configuration(bundle):
    for name in ("slurm.conf", "gres.conf", "cgroup.conf"):
        path = Path("/etc/slurm", name)
        shutil.copyfile(bundle / name, path)
        path.chmod(0o644)
    _run("systemctl", "stop", "munge")
    _run(
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
    _run(
        "install",
        "-d",
        "-o",
        "slurm",
        "-g",
        "slurm",
        "/var/spool/slurmd",
        "/var/log/slurm",
    )
    _run("systemctl", "daemon-reload")
    _run("systemctl", "disable", "slurmctld")
    _run("systemctl", "enable", "--now", "munge", "slurmd")


def _prepare_user(peers):
    try:
        user = pwd.getpwnam("slurm")
    except KeyError:
        user = None
    if user:
        if (user.pw_uid, user.pw_gid) != (peers["slurm_uid"], peers["slurm_gid"]):
            raise ValueError(
                "existing Slurm service identity differs from the controller"
            )
        return
    for lookup, identifier in (
        (pwd.getpwuid, peers["slurm_uid"]),
        (grp.getgrgid, peers["slurm_gid"]),
    ):
        try:
            lookup(identifier)
        except KeyError:
            continue
        raise ValueError("controller Slurm UID/GID is already assigned on this worker")
    _run("groupadd", "--system", "--gid", str(peers["slurm_gid"]), "slurm")
    _run(
        "useradd",
        "--system",
        "--uid",
        str(peers["slurm_uid"]),
        "--gid",
        "slurm",
        "--home-dir",
        "/var/lib/slurm",
        "--shell",
        "/usr/sbin/nologin",
        "slurm",
    )


def _main(args):
    peers = _validate(args.bundle)
    if args.prepare_user:
        _prepare_user(peers)
        return
    _firewall(peers)
    _hosts(peers)
    _mount(peers)
    _configuration(args.bundle)
    _run("sinfo", "-o", "%a %D %T %G")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prepare-user", action="store_true")
    os.umask(0o077)
    _main(parser.parse_args())
