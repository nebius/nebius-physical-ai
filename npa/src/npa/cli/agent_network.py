"""Local network reachability checks for ``npa agent`` deploys.

Extracted from the ``npa.cli.agent`` monolith (kept under a size ratchet). The
agent deploy provisions a VM and then waits for its ``tcp/22`` from *this*
machine; when a corporate VPN, split tunnel or firewall blocks outbound SSH, the
wait burns five minutes and Terraform rolls the healthy VM back. A cheap outbound
probe says so before any of that happens.
"""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - type-checker visibility only
    from npa.workflows.sim2real_health import CheckResult

#: Probe target for outbound tcp/22. A public SSH endpoint is the only way to
#: learn whether this host can open an SSH connection at all before the agent VM
#: exists. Overridable (``host:port``) for networks that reach their own hosts but
#: not this one, and disableable with ``off`` where the probe is meaningless (unit
#: tests, air-gapped CI).
DEFAULT_SSH_EGRESS_PROBE = "github.com:22"
PROBE_ENV_VAR = "NPA_SSH_EGRESS_PROBE"
_DISABLED_VALUES = frozenset({"0", "off", "false", "no", "none", "skip"})
_TIMEOUT_SECONDS = 3.0


def _probe_setting() -> str:
    return str(os.environ.get(PROBE_ENV_VAR, "") or "").strip()


def _probe_target() -> tuple[str, int]:
    raw = _probe_setting() or DEFAULT_SSH_EGRESS_PROBE
    host, _, port = raw.partition(":")
    try:
        return host.strip() or "github.com", int(port or 22)
    except ValueError:
        return host.strip() or "github.com", 22


def _agent_ssh_egress_result(
    connect: Callable[[tuple[str, int], float], object] | None = None,
) -> "CheckResult":
    """Outbound tcp/22 reachability check (WARN): the deploy waits for the VM's SSH.

    A heuristic, never a FAIL: a network that blocks the probe target but reaches
    Nebius is possible, and so is a host with no DNS at probe time. Both report
    PASS with a note so a healthy deploy is never blocked.
    """
    from npa.workflows.sim2real_health import CheckResult, PASS, WARN

    if _probe_setting().lower() in _DISABLED_VALUES:
        return CheckResult(
            name="ssh_egress",
            status=PASS,
            summary=f"Outbound SSH check skipped ({PROBE_ENV_VAR} disables it).",
        )
    host, port = _probe_target()
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
        return CheckResult(
            name="ssh_egress",
            status=WARN,
            summary=(
                f"This host could not open outbound tcp/{port} to {host} "
                f"within {_TIMEOUT_SECONDS:.0f}s."
            ),
            remedy=(
                "Agent deploy waits for the new VM's tcp/22 from this machine and "
                "rolls the VM back if it never answers, so a corporate VPN / split "
                "tunnel / firewall that blocks outbound SSH will fail the deploy. "
                "Deploy from a host with direct SSH egress, or set "
                f"{PROBE_ENV_VAR}=<host>:<port> if only {host} is blocked."
            ),
            details=(str(exc),),
        )
    close = getattr(connection, "close", None)
    if callable(close):
        close()
    return CheckResult(
        name="ssh_egress",
        status=PASS,
        summary=f"Outbound tcp/{port} works from this host ({host}).",
    )
