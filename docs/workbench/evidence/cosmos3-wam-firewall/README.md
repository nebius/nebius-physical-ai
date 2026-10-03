# Firewall dependency activation check

[verification.json](verification.json) records real systemd 252 activating the
production `persist-firewall.sh` unit and service drop-ins. Injected failures of
either IPv4 or IPv6 restoration prevent all five dependent services and the
rpcbind socket from starting. Successful restoration allows all six units to
activate. Restore backends and daemons are test doubles; the dependency engine
is real PID-1 systemd. This is not GPU-cluster reboot or packet-filter validation.
The hardening was added after the completed 8/16-GPU benchmark.

## Reproduce locally

Use a disposable privileged Docker container with its own cgroup namespace.
It contains no credentials, host mounts or cloud resources. Wait until
`systemctl is-system-running` reports `running` before executing the check.

```bash
docker run -d --privileged --cgroupns=private --tmpfs /run --tmpfs /run/lock \
  --name wam-review-systemd debian:bookworm-slim sh -c \
  'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y systemd systemd-sysv && exec /lib/systemd/systemd'
docker exec wam-review-systemd systemctl is-system-running
npa/.venv/bin/python docs/workbench/evidence/cosmos3-wam-firewall/verify.py \
  --container wam-review-systemd --output "$WAM_FIREWALL_REPORT"
docker rm -f wam-review-systemd
```

The verifier copies the production script, executes it with stub firewall
commands, and checks each dependent unit under both failures and success.
Output binds the exact production script by SHA-256 and records the systemd
version. The distribution tag may receive updates, so later systemd version
strings may differ; activation outcomes are the regression contract.
