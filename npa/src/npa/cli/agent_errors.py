"""Human-facing diagnosis for ``npa agent`` deploy failures.

Extracted from the ``npa.cli.agent`` monolith (kept under a size ratchet) so the
deploy-failure diagnosis lives in one small, independently-tested place. It is
re-exported from ``npa.cli.agent`` for the deploy call site and tests.
"""

from __future__ import annotations

import re

# Terraform reports a failed local-exec provisioner as
# ``Error running command '<entire script body>': exit status 1. Output: <output>``.
# The body contains every diagnostic string the script is able to print, so
# classifying a failure against the raw error would always match whichever
# branch is checked first. Drop the quoted body and keep only what the script
# actually printed at runtime.
_PROVISIONER_BODY_RE = re.compile(r"error running command '.*?': exit status", re.DOTALL)


def _provisioner_runtime_output(detail: str) -> str:
    """Return ``detail`` lowercased with the quoted provisioner script body removed."""
    return _PROVISIONER_BODY_RE.sub("error running command: exit status", str(detail or "").lower())


def _agent_deploy_failure_hint(detail: str) -> str:
    """Return a concise, actionable summary for a Terraform agent-deploy failure.

    A ``null_resource.wait_for_cloud_init`` failure otherwise surfaces as a wall
    of raw Terraform output (the whole local-exec provisioner body). Detect the
    two real causes — an unreachable SSH port and a failed cloud-init bootstrap —
    and return one short paragraph the operator can act on. Returns "" for any
    other failure so its raw error is preserved.
    """
    runtime = _provisioner_runtime_output(detail)
    if "wait_for_cloud_init" not in runtime and "waiting for ssh" not in runtime:
        return ""
    # ``cloud-init status: error`` is echoed by the status poll, which only runs
    # once SSH already worked, so it cannot be confused with a reachability
    # failure.
    if "cloud-init status: error" in runtime or "cloud-init finished with status" in runtime:
        return (
            "The agent VM booted but its cloud-init bootstrap failed (cloud-init "
            "status: error), so the deploy rolled the VM back. The failing "
            "bootstrap step is named in the `cloud-init status --long` output "
            "that streamed above; because the VM is gone, capture that output "
            "(or SSH in during a re-run and read "
            "/var/log/npa-agent-cloud-init.log) before retrying."
        )
    if "never authenticated" in runtime:
        # The wait distinguishes a closed port from a refused key; when tcp/22 did
        # open, reachability is not the problem.
        return (
            "The agent VM booted and its tcp/22 was reachable, but SSH never "
            "authenticated, so the deploy timed out and rolled the VM back. This is "
            "the key or the login user, not the network:\n"
            "  - does the private key next to --ssh-public-key-path match the public "
            "key the VM was created with?\n"
            "  - is --ssh-user right for this image (Nebius Ubuntu images use "
            "`ubuntu`)?"
        )
    return (
        "The agent VM booted (RUNNING, with a public IP) but its tcp/22 never "
        "opened from this machine, so the deploy timed out and rolled the VM back. "
        "This is almost always local reachability, not the VM:\n"
        "  - can this host reach the VM's tcp/22? (corporate VPN / split-tunnel / "
        "firewall commonly block outbound SSH to fresh public IPs, even when they "
        "allow SSH to known hosts such as github.com)\n"
        "  - does the security group allow tcp/22 from your address?\n"
        "Deploy from a host with direct network access to the VM; the full "
        "provisioner log streamed above."
    )
