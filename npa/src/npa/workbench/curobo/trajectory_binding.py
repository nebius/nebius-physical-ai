"""Bind optimized dynamics to every sample of the retained, retimed joint path."""

from __future__ import annotations

import numpy as np


# The pinned benchmark transition is BSPLINE_3, n_knots=16,
# interpolation_steps=4. Its boundary padding yields 20 cubic intervals.
_INTERVALS = 20
_OPTIMIZED_STEPS = 4
_FIELDS = ("position", "velocity", "acceleration", "jerk")
# Account for float32 kernel basis evaluation, derivative scaling and CPU
# polynomial reconstruction. This is a roundoff bound, NOT a torque/quality
# tolerance; it never changes the independent 1e-8 Nm replay gate.
_ROUNDOFF = 128 * np.finfo(np.float32).eps


def _aligned_arrays(optimized, retained):
    names = optimized["joint_names"]
    columns = [retained["joint_names"].index(name) for name in names]
    raw = [np.asarray(optimized[field], dtype=float) for field in _FIELDS]
    dense = [np.asarray(retained[field], dtype=float)[:, columns] for field in _FIELDS]
    return raw, dense


def _cubic_coefficients(raw, knot_dt):
    return (
        raw[0][:-1:_OPTIMIZED_STEPS],
        raw[1][:-1:_OPTIMIZED_STEPS] * knot_dt,
        raw[2][:-1:_OPTIMIZED_STEPS] * knot_dt**2 / 2,
        raw[3][:-1:_OPTIMIZED_STEPS] * knot_dt**3 / 6,
    )


def _require_cubic_samples(arrays, coefficients, steps, dt):
    samples = np.arange(len(arrays[0]))
    interval = np.minimum(samples // steps, _INTERVALS - 1)
    fraction = ((samples - interval * steps) / steps)[:, None]
    constant, linear, quadratic, cubic = [value[interval] for value in coefficients]
    expected = (
        constant + fraction * (linear + fraction * (quadratic + fraction * cubic)),
        linear + fraction * (2 * quadratic + 3 * fraction * cubic),
        2 * quadratic + 6 * fraction * cubic,
        6 * cubic,
    )
    scale = np.maximum(
        1, np.abs(constant) + np.abs(linear) + np.abs(quadratic) + np.abs(cubic)
    )
    for order, (values, predicted) in enumerate(zip(arrays, expected, strict=True)):
        difference = np.abs(values * (dt * steps) ** order - predicted)
        if not np.isfinite(difference).all() or np.any(difference > _ROUNDOFF * scale):
            raise ValueError("dynamics and retained trajectory interior paths differ")


def require_dynamics_path(optimized: dict, retained: dict) -> None:
    """Require exact shared samples or the pinned full cubic retiming relation.

    Args:
        optimized: Validated, named joint series consumed by inverse dynamics.
        retained: Validated joint series used for FK and visualization.

    Returns:
        None. Every active joint and every interior derivative is checked.

    Raises:
        ValueError: The series cannot represent the same pinned joint path.
    """
    raw, dense = _aligned_arrays(optimized, retained)
    raw_dt, dense_dt = optimized["dt"], retained["dt"]
    if raw_dt == dense_dt and all(
        np.array_equal(left, right) for left, right in zip(raw, dense, strict=True)
    ):
        return
    # Upstream calculate_traj_steps rounds to integer samples per knot with
    # float32 arithmetic, then the CUDA kernel retimes derivatives to that grid.
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            knot_dt = np.float32(raw_dt) * np.float32(_OPTIMIZED_STEPS)
            retained_dt = np.float32(dense_dt)
            if knot_dt <= 0 or retained_dt <= 0:
                raise ValueError("nonrepresentable float32 trajectory timing")
            steps = (knot_dt + retained_dt) / retained_dt
            if not np.isfinite(steps) or steps > np.iinfo(np.int32).max:
                raise ValueError("nonrepresentable pinned int32 sample count")
            dense_steps = int(steps)
    except (FloatingPointError, OverflowError, ValueError):
        raise ValueError("dynamics and retained trajectory timing is invalid") from None
    if (
        dense_steps < 1
        or len(raw[0]) != _INTERVALS * _OPTIMIZED_STEPS + 1
        or len(dense[0]) != _INTERVALS * dense_steps + 1
    ):
        raise ValueError("dynamics and retained trajectory retiming differs")
    coefficients = _cubic_coefficients(raw, float(knot_dt))
    _require_cubic_samples(raw, coefficients, _OPTIMIZED_STEPS, raw_dt)
    _require_cubic_samples(dense, coefficients, dense_steps, dense_dt)
