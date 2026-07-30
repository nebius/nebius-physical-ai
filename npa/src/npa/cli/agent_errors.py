"""Human-facing diagnosis for ``npa agent`` deploy failures.

Extracted from the ``npa.cli.agent`` monolith (kept under a size ratchet) so the
deploy-failure diagnosis lives in one small, independently-tested place. It is
re-exported from ``npa.cli.agent`` for the deploy call site and tests.
"""

from __future__ import annotations


def _agent_deploy_failure_hint(detail: str) -> str:
    """Return a concise, actionable summary for a Terraform agent-deploy failure.

    A ``null_resource.wait_for_cloud_init`` failure otherwise surfaces as a wall
    of raw Terraform output (the whole local-exec provisioner body). Detect the
    two real causes — an unreachable SSH port and a failed cloud-init bootstrap —
    and return one short paragraph the operator can act on. Returns "" for any
    other failure so its raw error is preserved.
    """
    lowered = str(detail or "").lower()
    if "wait_for_cloud_init" not in lowered and "waiting for ssh" not in lowered:
        return ""
    if "cloud-init finished with status" in lowered or "status 'error'" in lowered:
        return (
            "The agent VM booted but its cloud-init bootstrap failed (cloud-init "
            "status: error), so the deploy rolled the VM back. Re-run with "
            "NPA_DEBUG=1 for the full log, or inspect "
            "/var/log/npa-agent-cloud-init.log on a retained VM to see which "
            "bootstrap step failed."
        )
    return (
        "The agent VM booted (RUNNING, with a public IP) but SSH never became "
        "reachable from this machine, so the deploy timed out and rolled the VM "
        "back. This is almost always local reachability, not the VM:\n"
        "  - can this host reach the VM's tcp/22? (corporate VPN / split-tunnel / "
        "firewall commonly block outbound SSH to fresh public IPs)\n"
        "  - does the security group allow tcp/22 from your address?\n"
        "  - does the private key match --ssh-public-key-path?\n"
        "Deploy from a host with direct network access to the VM, or re-run with "
        "NPA_DEBUG=1 for the full provisioner log."
    )
