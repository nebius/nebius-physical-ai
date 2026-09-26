"""Build and validate the fixed conditional-replanning feature rows."""

from __future__ import annotations

from typing import Any

import numpy as np

ACTION_DIMENSION = 23
CHECK_OFFSET = 16
FEATURE_DIMENSION = 1233
HORIZON = 32


def require_array(
    value: Any, shape: tuple[int, ...], label: str, *, dtype: Any = np.float32
) -> np.ndarray:
    """Return one finite, contiguous array with an exact representation.

    Args:
        value: Candidate array-like value.
        shape: Required array shape.
        label: Human-readable value label.
        dtype: Required NumPy dtype.
    Returns:
        The validated C-contiguous array.
    Raises:
        ValueError: If shape, dtype, or finiteness differs.
    """
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.dtype(dtype):
        raise ValueError(f"{label} representation differs")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(array)


def build_features(
    continue_actions: np.ndarray,
    fresh_actions: np.ndarray,
    start_state: np.ndarray,
    fresh_state: np.ndarray,
    continue_mask: np.ndarray | None = None,
    fresh_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Serialize the preregistered 1,233-feature gate row.

    Args:
        continue_actions: Normalized actions at offsets 16 through 31.
        fresh_actions: Normalized actions from a fresh proposal.
        start_state: Transformed state at the original chunk start.
        fresh_state: Transformed state at offset 16.
        continue_mask: Optional validity mask for continued actions.
        fresh_mask: Optional validity mask for fresh actions.
    Returns:
        A contiguous float32 feature vector of length 1,233.
    Raises:
        ValueError: If any input differs from the fixed representation.
    """
    action_shape = (CHECK_OFFSET, ACTION_DIMENSION)
    cont = require_array(continue_actions, action_shape, "continue actions")
    fresh = require_array(fresh_actions, action_shape, "fresh actions")
    start = require_array(start_state, (HORIZON,), "start state")
    later = require_array(fresh_state, (HORIZON,), "fresh state")
    masks = _masks(continue_mask, fresh_mask)
    columns = (cont.ravel(), fresh.ravel(), (fresh - cont).ravel(), *masks)
    result = np.concatenate((*columns, start, later, later - start, [0.5]))
    result = np.ascontiguousarray(result, dtype=np.float32)
    if result.shape != (FEATURE_DIMENSION,):
        raise ValueError("conditional replanning feature dimension differs")
    return result


def proposal_errors(
    expert: np.ndarray, continued: np.ndarray, fresh: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the qualified float64 action-error summaries.

    Args:
        expert: Expert normalized actions shaped ``(rows, 16, 23)``.
        continued: Continued normalized actions with the same shape.
        fresh: Fresh normalized actions with the same shape.
    Returns:
        Continue error, fresh error, and proposal distance vectors.
    Raises:
        ValueError: If the arrays do not share the required representation.
    """
    shape = np.asarray(expert).shape
    if len(shape) != 3 or shape[1:] != (CHECK_OFFSET, ACTION_DIMENSION):
        raise ValueError("expert action representation differs")
    arrays = [
        require_array(item, shape, "normalized actions")
        for item in (expert, continued, fresh)
    ]
    target, cont, candidate = (item.astype(np.float64) for item in arrays)
    axes = (1, 2)
    return (
        np.mean(np.square(cont - target), axis=axes),
        np.mean(np.square(candidate - target), axis=axes),
        np.mean(np.square(candidate - cont), axis=axes),
    )


def _masks(
    continued: np.ndarray | None, fresh: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    result = []
    for label, value in (("continue", continued), ("fresh", fresh)):
        value = np.ones(CHECK_OFFSET, np.float32) if value is None else value
        mask = require_array(value, (CHECK_OFFSET,), f"{label} mask")
        if not np.isin(mask, (0.0, 1.0)).all():
            raise ValueError(f"{label} mask is not binary")
        result.append(mask)
    return result[0], result[1]
