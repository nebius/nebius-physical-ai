"""Re-export shim for the shipped Foxglove backend helpers.

The implementation moved to ``npa.agent_backend.foxglove`` (shipped to the agent
VM as an importable module, like ``memory`` / ``retrieval`` / ``trace``). This
shim keeps the historical ``npa.cli.agent_foxglove`` import path working.
"""

from __future__ import annotations

from npa.agent_backend.foxglove import *  # noqa: F401,F403
from npa.agent_backend.foxglove import __all__ as _all

__all__ = list(_all)
