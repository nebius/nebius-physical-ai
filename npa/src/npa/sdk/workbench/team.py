"""Expose authenticated clients and independent operator setup through the Workbench SDK."""

from pathlib import Path

from npa.workbench.team.client import TeamClient
from npa.workbench.team.connection import connect, open_connection
from npa.workbench.team.submission_receipts import submit_saved, submit_workflow
from npa.workbench.team.models import SubmitRequest
from npa.workbench.team.accounts import Accounts
from npa.workbench.team.account_administration import issue_key_file, link_identity
from npa.workbench.team.setup_models import SetupRequest

__all__ = [
    "Accounts",
    "SubmitRequest",
    "TeamClient",
    "connect",
    "open_connection",
    "submit_saved",
    "submit_workflow",
    "issue_key_file",
    "link_identity",
    "SetupRequest",
    "setup_control_plane",
]


def setup_control_plane(request: SetupRequest, output_path: Path):
    """Run operator setup without loading its dependencies for ordinary API clients.

    Args:
        request: Operator-owned provider, cluster, server and TLS configuration.
        output_path: Private durable receipt reused for retries and Service recovery.
    Returns:
        Operator setup status and the private receipt path.
    Raises:
        TeamError, OSError: Setup selection, ownership, transport or I/O fails.
    """
    from npa.workbench.team.setup import setup_control_plane as setup

    return setup(request, output_path)
