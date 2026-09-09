"""Checkpoint completed agent service installation before credential staging."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import logging
import secrets
import shlex
import sys

from npa.clients.ssh import SSHClient, SSHError


_INSTALL_RECEIPT = "/opt/npa-agent/services-installed.sha256"
_LOG = logging.getLogger(__name__)


def _receipt_script(digest: str) -> str:
    return f"""
(
set -eu
stage="$(sudo mktemp /opt/npa-agent/.services-installed.XXXXXXXX)"
trap 'sudo rm -f -- "$stage"' EXIT
builtin printf '%s\\n' {shlex.quote(digest)} | sudo tee "$stage" >/dev/null
sudo mv -fT -- "$stage" {_INSTALL_RECEIPT}
)
"""


def _remove_installer(ssh: SSHClient, remote_path: str) -> None:
    primary_error = sys.exc_info()[1]
    try:
        ssh.run(f"rm -f -- {shlex.quote(remote_path)}")
    except SSHError:
        if primary_error is None:
            raise
        _LOG.warning(
            "Removing the private installer also failed over SSH; preserving the original error."
        )


def install_agent_services(
    ssh: SSHClient,
    *,
    setup_script: str,
    stage_source: Callable[[SSHClient], None],
    resuming: bool,
) -> bool:
    """Install services unless an exact completed installer can be resumed.

    Args:
        ssh: Independently authenticated SSH connection to the owned agent VM.
        setup_script: Fully rendered installer, including source identity and settings.
        stage_source: Stage the source archive before running a new installer.
        resuming: Allow reuse only during recovery of an interrupted bootstrap.

    Returns:
        True when a matching completed installation was reused.

    Raises:
        SSHError: The receipt cannot be read or installation/cleanup fails.
    """
    digest = hashlib.sha256(setup_script.encode("utf-8")).hexdigest()
    if resuming:
        code, stdout, _stderr = ssh.run(f"sudo cat {_INSTALL_RECEIPT} 2>/dev/null")
        if code == 0 and stdout.strip() == digest:
            return True
    remote_path = f"/tmp/npa-agent-bootstrap-{secrets.token_hex(6)}.sh"
    try:
        stage_source(ssh)
        ssh.upload_private_text(setup_script + _receipt_script(digest), remote_path)
        ssh.run_or_raise(
            f"chmod 700 {shlex.quote(remote_path)} && {shlex.quote(remote_path)}",
            label="run agent bootstrap",
        )
    finally:
        _remove_installer(ssh, remote_path)
    return False
