"""Expose the authenticated team client through the standard Workbench SDK namespace."""

from npa.workbench.team.client import TeamClient
from npa.workbench.team.models import SubmitRequest
from npa.workbench.team.accounts import Accounts
from npa.workbench.team.account_administration import issue_key_file, link_identity
from npa.workbench.team.setup import setup_control_plane
from npa.workbench.team.setup_models import SetupRequest

__all__ = [
    "Accounts",
    "SubmitRequest",
    "TeamClient",
    "issue_key_file",
    "link_identity",
    "SetupRequest",
    "setup_control_plane",
]
