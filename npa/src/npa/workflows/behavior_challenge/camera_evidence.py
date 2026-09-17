"""Prepare training-only camera evidence for π0.5 consistency regularization."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvidenceThresholds:
    """Configure visibility gates and the ambiguous evidence interval.

    Args:
        low: Upper boundary for irrelevant-camera regularization.
        high: Lower boundary for informative-camera regularization.
        focal: Minimum focal-object pixel fraction to include its receptacle.
        global_visibility: Minimum visibility in any camera to enable a frame.
        interaction_weight: Contribution of a visible focal object's receptacle.
    Returns:
        None.
    Raises:
        ValueError: A threshold is nonfinite or outside its valid range.
    """

    low: float = 0.2
    high: float = 0.8
    focal: float = 0.0001
    global_visibility: float = 0.001
    interaction_weight: float = 0.5

    def __post_init__(self):
        values = (self.low, self.high, self.focal, self.global_visibility, self.interaction_weight)
        if not np.isfinite(values).all() or not 0 < self.low < self.high < 1:
            raise ValueError("Evidence thresholds require 0 < low < high < 1")
        if not 0 <= self.focal < 1 or not 0 < self.global_visibility < 1:
            raise ValueError("Visibility thresholds must be pixel fractions")
        if self.interaction_weight < 0:
            raise ValueError("Interaction weight must be nonnegative")


def camera_weights(focal_area, interaction_area, thresholds=EvidenceThresholds()):
    """Compute EGR hinge weights from aligned, training-only visible pixel areas.

    Args:
        focal_area: Fractions shaped [batch, cameras] for manipulated objects.
        interaction_area: Matching fractions for receptacles or target surfaces.
        thresholds: Visibility gates and low/high evidence boundaries.
    Returns:
        Invariance and sufficiency weights, each shaped [batch, cameras].
    Raises:
        ValueError: Areas have incompatible shapes, nonfinite values or invalid ranges.
    """
    focal = _area_matrix(focal_area)
    interaction = _area_matrix(interaction_area)
    if focal.shape != interaction.shape:
        raise ValueError("Focal and interaction areas must have matching shapes")
    visible = focal > thresholds.focal
    score = focal + thresholds.interaction_weight * visible * interaction
    strongest = score.max(axis=-1, keepdims=True)
    active = strongest > thresholds.global_visibility
    evidence = score / np.maximum(strongest, np.finfo(np.float32).eps)
    invariance = np.maximum(0, (thresholds.low - evidence) / thresholds.low)
    sufficiency = np.maximum(0, (evidence - thresholds.high) / (1 - thresholds.high))
    return active * invariance, active * sufficiency


def _area_matrix(value):
    areas = np.asarray(value, dtype=np.float32)
    if areas.ndim != 2 or not all(areas.shape):
        raise ValueError("Visible areas require a nonempty [batch, cameras] matrix")
    if not np.isfinite(areas).all() or np.any((areas < 0) | (areas > 1)):
        raise ValueError("Visible areas must be finite fractions between zero and one")
    return areas


def sample_camera(weights, random):
    """Sample one camera and its importance multiplier for each training example.

    Args:
        weights: Nonnegative finite hinge weights shaped [batch, cameras].
        random: Explicit NumPy random generator, owned by the data pipeline.
    Returns:
        Camera indices and weight sums; zero-weight rows return index zero and zero.
    Raises:
        ValueError: Weights are empty, negative or nonfinite.
    """
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2 or not all(weights.shape):
        raise ValueError("Camera weights require a nonempty [batch, cameras] matrix")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Camera weights must be finite and nonnegative")
    total = weights.sum(axis=-1)
    if not np.isfinite(total).all():
        raise ValueError("Camera weight sums must be finite")
    draws = random.random(len(weights)) * total
    indices = (draws[:, None] >= np.cumsum(weights, axis=-1)).sum(axis=-1)
    indices = np.where(total > 0, np.minimum(indices, weights.shape[-1] - 1), 0)
    return indices.astype(np.int32), total


def consistency_loss(clean_velocity, corrupted_velocity, importance_weight, *, action_dimensions):
    """Return per-example importance-weighted flow consistency without detaching gradients.

    Use identical noise, flow time and ordinary augmentation for both predictions.
    Arrays stay in their caller's NumPy or JAX backend; no host conversion occurs.

    Args:
        clean_velocity: Flow velocities shaped [batch, horizon, padded action dimensions].
        corrupted_velocity: Matching velocities after the selected camera corruption.
        importance_weight: Per-example sums returned by ``sample_camera``.
        action_dimensions: Number of real action dimensions, excluding padding.
    Returns:
        A vector of differentiable per-example consistency losses.
    Raises:
        ValueError: Prediction shapes or action dimensions are incompatible.
    """
    if len(clean_velocity.shape) != 3 or clean_velocity.shape != corrupted_velocity.shape:
        raise ValueError("Flow predictions must have matching [batch, horizon, action] shapes")
    if not all(clean_velocity.shape) or importance_weight.shape != (clean_velocity.shape[0],):
        raise ValueError("Flow predictions and camera weights must have matching nonempty batches")
    if not isinstance(action_dimensions, int) or not 0 < action_dimensions <= clean_velocity.shape[-1]:
        raise ValueError("Real action dimensions must fit the flow prediction")
    difference = clean_velocity[..., :action_dimensions] - corrupted_velocity[..., :action_dimensions]
    return importance_weight * (difference * difference).mean(axis=(-2, -1))
