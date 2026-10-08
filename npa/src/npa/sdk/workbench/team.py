"""Expose the authenticated team client through the standard Workbench SDK namespace."""

from npa.workbench.team.client import TeamClient
from npa.workbench.team.models import SubmitRequest

__all__ = ["SubmitRequest", "TeamClient"]
