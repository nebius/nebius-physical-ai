"""Configure a fresh dedicated controller with private accounting credentials."""

import os
from pathlib import Path
import re
import secrets
import socket
import subprocess


def _database():
    password = secrets.token_hex(32)
    query = (
        "CREATE DATABASE slurm_acct_db; "
        f"CREATE USER 'slurm'@'localhost' IDENTIFIED BY '{password}'; "
        "GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost'; FLUSH PRIVILEGES;"
    )
    subprocess.run(["mariadb"], input=query, text=True, check=True)
    config = f"""AuthType=auth/munge
DbdHost=localhost
SlurmUser=slurm
LogFile=/var/log/slurm/slurmdbd.log
PidFile=/run/slurmdbd.pid
StorageType=accounting_storage/mysql
StorageHost=localhost
StorageUser=slurm
StoragePass={password}
StorageLoc=slurm_acct_db
"""
    path = Path("/etc/slurm/slurmdbd.conf")
    path.write_text(config)
    path.chmod(0o600)
    subprocess.run(["chown", "slurm:slurm", str(path)], check=True)


def _node_settings():
    hardware = subprocess.check_output(["slurmd", "-C"], text=True).splitlines()[0]
    fields = dict(token.split("=", 1) for token in hardware.split() if "=" in token)
    shape = " ".join(
        f"{key}={fields[key]}"
        for key in (
            "CPUs",
            "Boards",
            "SocketsPerBoard",
            "CoresPerSocket",
            "ThreadsPerCore",
        )
    )
    return {
        "NODE": socket.gethostname().split(".")[0],
        "ADDRESS": subprocess.check_output(["hostname", "-I"], text=True).split()[0],
        "SHAPE": shape,
        "MEMORY": str(int(fields["RealMemory"]) - 16384),
    }


def _configure(cluster):
    settings = dict(_node_settings(), CLUSTER=cluster)
    config = Path(__file__).with_name("slurm.conf.in").read_text()
    for key, value in settings.items():
        config = config.replace("@" + key + "@", value)
    Path("/etc/slurm/slurm.conf").write_text(config)
    Path("/etc/slurm/gres.conf").write_text(
        "Name=gpu Type=b200 File=/dev/nvidia[0-7]\n"
    )
    Path("/etc/slurm/cgroup.conf").write_text(
        "CgroupPlugin=autodetect\nConstrainCores=yes\nConstrainDevices=yes\n"
        "ConstrainRAMSpace=yes\nEnableControllers=yes\n"
    )
    for name in ("slurm.conf", "gres.conf", "cgroup.conf"):
        Path("/etc/slurm", name).chmod(0o644)


def _main():
    if os.geteuid() != 0 or Path("/etc/slurm/slurm.conf").exists():
        raise RuntimeError("requires root on a fresh Slurm installation")
    cluster = os.environ["NPA_SLURM_CLUSTER_NAME"]
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,62}", cluster):
        raise ValueError(
            "cluster name must be lowercase letters, digits, underscores or dashes"
        )
    _database()
    _configure(cluster)


if __name__ == "__main__":
    os.umask(0o077)
    _main()
