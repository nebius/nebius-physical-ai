"""Join a second dedicated B200 worker to the native Slurm research cluster."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import socket
import subprocess


def _run(*command):
    subprocess.run(command, check=True)


def _private_address(value):
    address = ipaddress.ip_address(value)
    networks = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
    if not any(address in ipaddress.ip_network(network) for network in networks):
        raise ValueError("cluster peers must use private RFC1918 IPv4 addresses")
    return address


def _validate(args):
    if os.geteuid() != 0:
        raise RuntimeError("requires root on the dedicated controller")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]{0,62}", args.worker_name):
        raise ValueError("worker name must be a valid hostname")
    _private_address(args.worker_address)
    root = args.shared_root.resolve(strict=True)
    if any(character.isspace() for character in str(root)) or str(root) == "/":
        raise ValueError("shared root must be a dedicated path without whitespace")
    if not (root / "prepared.json").is_file():
        raise ValueError("shared root does not contain prepared WAM inputs")
    return root


def _configuration(args):
    config = Path("/etc/slurm/slurm.conf").read_text()
    node_lines = [line for line in config.splitlines() if line.startswith("NodeName=")]
    if len(node_lines) != 1 or " State=" not in node_lines[0]:
        raise ValueError("expected the recipe's single-node controller configuration")
    controller = socket.gethostname().split(".")[0]
    if args.worker_name == controller:
        raise ValueError("second worker must have a distinct hostname")
    fields = dict(item.split("=", 1) for item in node_lines[0].split())
    shape = " ".join(
        f"{key}={value}"
        for key, value in fields.items()
        if key not in ("NodeName", "NodeAddr", "State")
    )
    worker = f"NodeName={args.worker_name} NodeAddr={args.worker_address} {shape} State=UNKNOWN"
    config = config.replace(node_lines[0], node_lines[0] + "\n" + worker)
    config = config.replace(
        f"Nodes={controller} ", f"Nodes={controller},{args.worker_name} "
    )
    config = config.replace(
        "AccountingStorageHost=localhost", f"AccountingStorageHost={fields['NodeAddr']}"
    )
    return config, {
        "controller_name": controller,
        "controller_address": fields["NodeAddr"],
    }


def _hosts(peers):
    names = {peers["controller_name"], peers["worker_name"]}
    original = Path("/etc/hosts").read_text().splitlines()
    retained = [line for line in original if not names.intersection(line.split()[1:])]
    entries = [
        f"{peers[role + '_address']} {peers[role + '_name']}"
        for role in ("controller", "worker")
    ]
    Path("/etc/hosts").write_text("\n".join(retained + entries) + "\n")
    Path("/etc/cloud/cloud.cfg.d/99-wam-slurm.cfg").write_text(
        "manage_etc_hosts: false\n"
    )


def _export(root, peers):
    export = f"{root} {peers['worker_address']}(rw,sync,no_subtree_check,root_squash)\n"
    Path("/etc/exports.d/wam-slurm.exports").write_text(export)
    _run("systemctl", "enable", "--now", "nfs-server")
    _run("exportfs", "-ra")
    _run(
        "iptables",
        "-I",
        "WAM_INPUT",
        "1",
        "-s",
        peers["worker_address"],
        "-j",
        "ACCEPT",
    )
    _run("bash", str(Path(__file__).with_name("persist-firewall.sh")))


def _main(args):
    root = _validate(args)
    config, peers = _configuration(args)
    peers.update(
        worker_name=args.worker_name,
        worker_address=args.worker_address,
        shared_root=str(root),
        user_uid=root.stat().st_uid,
    )
    user = pwd.getpwuid(root.stat().st_uid)
    peers.update(user_name=user.pw_name, user_home=user.pw_dir)
    service = pwd.getpwnam("slurm")
    peers.update(slurm_uid=service.pw_uid, slurm_gid=service.pw_gid)
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    _hosts(peers)
    _export(root, peers)
    path = Path("/etc/slurm/slurm.conf")
    path.write_text(config)
    path.chmod(0o644)
    for name in ("slurm.conf", "gres.conf", "cgroup.conf"):
        shutil.copyfile(Path("/etc/slurm", name), args.output_dir / name)
    shutil.copyfile("/etc/munge/munge.key", args.output_dir / "munge.key")
    (args.output_dir / "peers.json").write_text(json.dumps(peers, indent=2) + "\n")
    for path in args.output_dir.iterdir():
        path.chmod(0o600)
    _run("scontrol", "reconfigure")
    print("Private worker bundle prepared; transfer it only over verified SSH")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-name", required=True)
    parser.add_argument("--worker-address", required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    os.umask(0o077)
    _main(parser.parse_args())
