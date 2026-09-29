"""Build TRAIN-only task-1 semantic labels without changing policy execution."""

from .collector import collect_episode_labels, deterministic_episode_split
from .interface import (
    PromotionDecision,
    PromotionMonitor,
    PromotionRequest,
    validate_evaluation_observation,
)
from .schema import BoundaryRequirement, SemanticLabel, TrainingTraceFrame

__all__ = (
    "BoundaryRequirement",
    "PromotionDecision",
    "PromotionMonitor",
    "PromotionRequest",
    "SemanticLabel",
    "TrainingTraceFrame",
    "collect_episode_labels",
    "deterministic_episode_split",
    "validate_evaluation_observation",
)
