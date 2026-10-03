"""Compatibility export for the shipped goal-level trajectory emitter."""

from npa.agent_backend.trajectory import (
    AgentRunDataError,
    CollectionStatus,
    DatasetConfig,
    emit_trajectory,
    flush_outbox,
    goal_episode_boundary,
    redact,
    resolve_dataset_config,
    verify_destination,
)

__all__ = [
    "AgentRunDataError",
    "CollectionStatus",
    "DatasetConfig",
    "emit_trajectory",
    "flush_outbox",
    "goal_episode_boundary",
    "redact",
    "resolve_dataset_config",
    "verify_destination",
]
