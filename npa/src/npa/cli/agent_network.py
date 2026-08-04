"""Local network reachability checks for ``npa agent`` deploys.

Extracted from the ``npa.cli.agent`` monolith (kept under a size ratchet). The
agent deploy provisions a VM and then waits for its ``tcp/22`` from *this*
machine; when a corporate VPN, split tunnel or firewall blocks outbound SSH, the
wait ends in Terraform rolling the healthy VM back. A cheap outbound probe says so
before any of that happens.

Which host is probed matters. A split-tunnel policy commonly allows SSH to known
hosts (github.com) while dropping traffic to a fresh cloud IP, so a generic probe
that passes proves very little — it PASSes with that caveat spelled out. When the
operator already has an agent VM recorded in ``~/.npa``, its public IP is a real
Nebius endpoint and is probed instead, which is a genuine answer.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - type-checker visibility only
    from npa.workflows.sim2real_health import CheckResult

#: Fallback probe target when no Nebius VM IP is known. Overridable
#: (``host:port``) for networks that reach their own hosts but not this one, and
#: disableable with ``off`` where the probe is meaningless (unit tests, air-gapped
#: CI).
DEFAULT_SSH_EGRESS_PROBE = "github.com:22"
PROBE_ENV_VAR = "NPA_SSH_EGRESS_PROBE"
_DISABLED_VALUES = frozenset({"0", "off", "false", "no", "none", "skip"})
_TIMEOUT_SECONDS = 3.0

Connector = Callable[[tuple[str, int], float], object]

_AGENT_NETWORK_STATE_ADDRESSES = (
    "nebius_vpc_v1_network.workbench",
    # Compatibility with state written before the resource rename/moved block.
    "nebius_vpc_v1_network.lerobot",
)
_AGENT_INSTANCE_DESTROY_TARGETS = (
    "null_resource.wait_for_cloud_init",
    "nebius_compute_v1_instance.workbench",
)


def destroy_with_default_security_group_recovery(
    *,
    run_destroy: Callable[[], None],
    cleanup_ingress: Callable[[], None],
    tf_dir: str | Path | None,
    tf_vars: dict[str, str],
    cleanup_action: str,
    on_status: Callable[[str], None],
) -> None:
    """Run Terraform destroy with the provider-supported default-SG recovery."""

    from npa.clients.network import (
        NetworkCleanupError,
        is_default_security_group_delete_refusal,
        recover_default_security_group_delete,
    )
    from npa.deploy import provisioner
    from npa.deploy.provisioner import ProvisionerError

    try:
        run_destroy()
    except ProvisionerError as first_exc:
        network_id = ""
        if is_default_security_group_delete_refusal(str(first_exc)):
            try:
                managed = set(provisioner.state_list(tf_dir))
                for address in _AGENT_NETWORK_STATE_ADDRESSES:
                    if address in managed:
                        network_id = provisioner.state_resource_id(
                            address, tf_dir=tf_dir
                        )
                        break
            except ProvisionerError:
                network_id = ""
        try:
            recovered = recover_default_security_group_delete(
                error=str(first_exc),
                parent_network_id=network_id,
                parent_network_owned=bool(network_id),
                cleanup_action=cleanup_action,
                on_status=on_status,
            )
        except NetworkCleanupError as exc:
            raise ProvisionerError(str(exc)) from None
        if recovered:
            # Reconcile Terraform state after the parent deletion. Missing child
            # SG/network resources are idempotent provider reads.
            run_destroy()
            return
        cleanup_ingress()
        try:
            managed = set(provisioner.state_list(tf_dir))
            targets = [
                target
                for target in _AGENT_INSTANCE_DESTROY_TARGETS
                if target in managed
            ]
            if targets:
                provisioner.destroy(tf_dir=tf_dir, tf_vars=tf_vars, targets=targets)
            run_destroy()
        except ProvisionerError:
            raise first_exc from None


def _probe_setting() -> str:
    return str(os.environ.get(PROBE_ENV_VAR, "") or "").strip()


def _parse_target(raw: str, *, default_port: int = 22) -> tuple[str, int]:
    host, _, port = str(raw or "").strip().partition(":")
    try:
        return host.strip(), int(port or default_port)
    except ValueError:
        return host.strip(), default_port


def recorded_agent_ip() -> str:
    """Return a public IP from a saved agent record, or "".

    Reads ``~/.npa/config.yaml`` directly rather than through ``npa.cli.agent``,
    which imports this module.
    """
    try:
        from npa.clients.config import list_projects
    except Exception:  # noqa: BLE001 - the probe must not depend on config imports
        return ""
    try:
        projects = list_projects()
    except Exception:  # noqa: BLE001 - an unreadable config just means "no IP"
        return ""
    for project in (projects or {}).values():
        agents = (project or {}).get("agents") if isinstance(project, dict) else None
        for record in (agents or {}).values():
            ip = str((record or {}).get("public_ip", "") or "").strip()
            if ip and ip not in {"localhost", "127.0.0.1"} and not ip.startswith("127."):
                return ip
    return ""


def _agent_ssh_egress_result(connect: Connector | None = None) -> "CheckResult":
    """Outbound tcp/22 reachability check (WARN): the deploy waits for the VM's SSH.

    A heuristic, never a FAIL: a network that blocks the probe target but reaches
    Nebius is possible, and so is a host with no DNS at probe time. Both report
    PASS with a note so a healthy deploy is never blocked.
    """
    from npa.workflows.sim2real_health import CheckResult, PASS, WARN

    setting = _probe_setting()
    if setting.lower() in _DISABLED_VALUES:
        return CheckResult(
            name="ssh_egress",
            status=PASS,
            summary=f"Outbound SSH check skipped ({PROBE_ENV_VAR} disables it).",
        )
    nebius_ip = "" if setting else recorded_agent_ip()
    host, port = _parse_target(setting or nebius_ip or DEFAULT_SSH_EGRESS_PROBE)
    if not host:
        return CheckResult(
            name="ssh_egress",
            status=PASS,
            summary=f"Outbound SSH check skipped ({PROBE_ENV_VAR} has no host).",
        )
    is_nebius_target = bool(nebius_ip) and host == nebius_ip

    opener = connect or socket.create_connection
    try:
        connection = opener((host, port), _TIMEOUT_SECONDS)
    except (TimeoutError, socket.timeout, ConnectionRefusedError, OSError) as exc:
        if isinstance(exc, socket.gaierror):
            return CheckResult(
                name="ssh_egress",
                status=PASS,
                summary=f"Outbound SSH check skipped ({host} did not resolve).",
            )
        target = "your agent VM" if is_nebius_target else host
        return CheckResult(
            name="ssh_egress",
            status=WARN,
            summary=(
                f"This host could not open outbound tcp/{port} to {target} "
                f"within {_TIMEOUT_SECONDS:.0f}s."
            ),
            remedy=(
                "Agent deploy waits for the new VM's tcp/22 from this machine and "
                "rolls the VM back if it never answers, so a corporate VPN / split "
                "tunnel / firewall that blocks outbound SSH will fail the deploy. "
                "Deploy from a host with direct SSH egress, or set "
                f"{PROBE_ENV_VAR}=<host>:<port> to probe a host your network allows."
            ),
            details=(str(exc),),
        )
    close = getattr(connection, "close", None)
    if callable(close):
        close()
    if is_nebius_target:
        return CheckResult(
            name="ssh_egress",
            status=PASS,
            summary=f"Outbound tcp/{port} reaches your Nebius agent VM ({host}).",
        )
    return CheckResult(
        name="ssh_egress",
        status=PASS,
        summary=(
            f"Outbound tcp/{port} works from this host ({host}) — note that a split "
            "tunnel can still block a fresh Nebius public IP."
        ),
    )
