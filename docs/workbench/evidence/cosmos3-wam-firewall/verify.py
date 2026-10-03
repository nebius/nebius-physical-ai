"""Check production firewall service dependencies in an isolated systemd container.

Run with the repository venv and --container pointing to a disposable container
whose PID 1 is systemd. No real firewall rules or Slurm/NFS daemons are exercised.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "npa/workflows/workbench/cosmos3-wam-slurm/persist-firewall.sh"
SERVICES = ["slurmctld", "slurmd", "slurmdbd", "nfs-server", "rpcbind"]


def _exec(container, command, *, check=True):
    return subprocess.run(
        ["docker", "exec", container, "sh", "-c", command],
        check=check,
        capture_output=True,
        text=True,
    )


def _install(container):
    subprocess.run(
        ["docker", "cp", str(SCRIPT), f"{container}:/root/firewall.sh"], check=True
    )
    _exec(
        container,
        """
mkdir -p /root/bin
printf '#!/bin/sh\nexec "$@"\n' > /root/bin/sudo
printf '#!/bin/sh\nexit 0\n' > /root/bin/ip6tables
cp /root/bin/ip6tables /root/bin/iptables-save
cp /root/bin/ip6tables /root/bin/ip6tables-save
chmod +x /root/bin/*
PATH=/root/bin:$PATH bash /root/firewall.sh
for family in iptables ip6tables; do
  printf '#!/bin/sh\n! test -e /run/fail-%s\n' "$family" > /usr/sbin/$family-restore
  chmod +x /usr/sbin/$family-restore
done
""",
    )
    _dummy_services(container)


def _dummy_services(container):
    for service in SERVICES:
        _exec(
            container,
            f"""
cat > /etc/systemd/system/{service}.service <<'UNIT'
[Service]
Type=simple
ExecStart=/bin/sh -c "touch /run/started-{service}; exec sleep infinity"
UNIT
""",
        )
    _exec(
        container,
        """
cat > /etc/systemd/system/rpcbind.socket <<'UNIT'
[Socket]
ListenStream=127.0.0.1:39088
UNIT
systemctl daemon-reload
""",
    )


def _start(container, unit):
    _exec(container, "systemctl reset-failed wam-slurm-firewall.service")
    result = _exec(container, f"systemctl start {unit}", check=False)
    restore = _exec(
        container, "systemctl show wam-slurm-firewall.service -p ExecMainStatus --value"
    )
    assert restore.stdout.strip() == ("1" if result.returncode else "0")
    return result.returncode


def _case(container, family):
    units = ["rpcbind.socket"] + [f"{name}.service" for name in SERVICES]
    _exec(
        container, "systemctl stop " + " ".join(units) + " wam-slurm-firewall.service"
    )
    _exec(container, "rm -f /run/started-* /run/fail-*; systemctl reset-failed")
    if family:
        _exec(container, f"touch /run/fail-{family}")
    codes = {unit: _start(container, unit) for unit in units}
    states = {
        unit: _exec(
            container, f"systemctl is-active {unit}", check=False
        ).stdout.strip()
        for unit in units
    }
    markers = _exec(
        container, "find /run -maxdepth 1 -name 'started-*' -printf '%f\\n'"
    ).stdout.splitlines()
    if family:
        assert all(codes.values()) and not markers, (codes, markers)
        assert all(state == "inactive" for state in states.values()), states
    else:
        assert not any(codes.values()) and len(markers) == len(SERVICES), (
            codes,
            markers,
        )
        assert all(state == "active" for state in states.values()), states
    return {
        "restore_failure": family,
        "start_exits": codes,
        "dependent_states": states,
        "daemon_markers": len(markers),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert _exec(args.container, "cat /proc/1/comm").stdout.strip() == "systemd"
    _install(args.container)
    cases = [
        _case(args.container, family) for family in ("iptables", "ip6tables", None)
    ]
    result = {
        "status": "passed",
        "scope": "real systemd activation; injected restore failures; dummy daemons",
        "production_script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "systemd": _exec(args.container, "systemctl --version").stdout.splitlines()[0],
        "cases": cases,
        "gpu_cluster_tested": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
