"""Tests for the learned-actor placement settle phase."""

from __future__ import annotations

import numpy as np
import pytest

from npa.workflows.sim2real.placement_policy import (
    SETTLE_HOLD_TRIGGER_DISTANCE_M,
    advance_settle_hold,
    settle_hold_trigger,
)


def test_settle_hold_requires_tighter_goal_lift_contact_and_closed_gripper() -> None:
    distance = np.array([0.039, 0.041, 0.039, 0.039, 0.039])
    lift = np.array([0.05, 0.05, 0.03, 0.05, 0.05])
    contact = np.array([True, True, True, False, True])
    closed = np.array([True, True, True, True, False])

    trigger = settle_hold_trigger(distance, lift, contact, closed)

    assert SETTLE_HOLD_TRIGGER_DISTANCE_M < 0.05
    assert trigger.tolist() == [True, False, False, False, False]


def test_settle_hold_is_monotonic_and_reports_only_new_latches() -> None:
    latched = np.array([False, True, False])
    latched, newly = advance_settle_hold(latched, np.array([True, True, False]))
    assert latched.tolist() == [True, True, False]
    assert newly.tolist() == [True, False, False]

    latched, newly = advance_settle_hold(latched, np.array([False, True, True]))
    assert latched.tolist() == [True, True, True]
    assert newly.tolist() == [False, False, True]


@pytest.mark.parametrize(
    ("trigger_distance_m", "minimal_lift_m"),
    [(0.0, 0.04), (0.04, 0.0)],
)
def test_settle_hold_rejects_nonpositive_physical_boundaries(
    trigger_distance_m: float, minimal_lift_m: float
) -> None:
    with pytest.raises(ValueError, match="positive"):
        settle_hold_trigger(
            0.03,
            0.05,
            True,
            True,
            trigger_distance_m=trigger_distance_m,
            minimal_lift_m=minimal_lift_m,
        )
