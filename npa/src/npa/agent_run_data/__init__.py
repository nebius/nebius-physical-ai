"""Agent run data collection - sanitized, outcome-linked trajectory emission.

Implements the `npa.agent.trajectory.v1` contract from
`skills/atomic/agent-run-data-collection/SKILL.md`. Every goal-level episode emits
one immutable, sanitized trajectory object to an append-only S3 dataset, with a
deterministic content-hash key and read-after-write verification. On S3 failure
the record is preserved in an owner-only local outbox and collection is reported
`pending`.
"""

from __future__ import annotations

from npa.agent_run_data.emitter import (
    AgentRunDataError,
    CollectionStatus,
    emit_trajectory,
    flush_outbox,
    goal_episode_boundary,
    resolve_dataset_config,
)

__all__ = [
    "AgentRunDataError",
    "CollectionStatus",
    "emit_trajectory",
    "flush_outbox",
    "goal_episode_boundary",
    "resolve_dataset_config",
]
