"""Expose the shared Isaac Lab-Arena evaluation contract.

``IsaacArenaRequest`` describes an evaluation, ``evaluate`` executes it, and
``capabilities`` reports the pinned upstream inventory and NPA support status.
Their defining modules carry the argument, return, and error documentation;
these re-exports preserve that single implementation for CLI and SDK clients.
Schema and upstream revision constants identify the corresponding documents.
"""

from .runtime import (
    ARTIFACT_SCHEMA,
    CAPABILITIES_SCHEMA,
    ISAAC_ARENA_REVISION,
    ISAAC_ARENA_VERSION,
    IsaacArenaError,
    IsaacArenaRequest,
    capabilities,
    evaluate,
)

__all__ = [
    "ARTIFACT_SCHEMA",
    "CAPABILITIES_SCHEMA",
    "ISAAC_ARENA_REVISION",
    "ISAAC_ARENA_VERSION",
    "IsaacArenaError",
    "IsaacArenaRequest",
    "capabilities",
    "evaluate",
]
