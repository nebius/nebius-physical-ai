"""Command faster aisle traversals and physical U-turns through the walking policy."""

import math

import numpy as np


class PatrolController:
    """Track aisle endpoints with acceleration ramps and stationary turn phases.

    Args: Qualified aisle, maximum forward speed, and straight/turnaround profile.
    Returns: A stateful velocity-command controller; never modifies body poses.
    Raises: None.
    """

    def __init__(self, route, speed, profile):
        self.route = route
        self.speed = speed
        self.profile = profile
        self.phase = "outbound"
        self.returning = False
        self.turns = 0
        self.forward = 0.0

    def command(self, bullet, robot, active):
        """Compute body-frame policy commands from measured position and heading.

        Args: Physics client, robot ID, and whether the settling phase has ended.
        Returns: Forward, lateral and yaw velocities at the 50 Hz policy clock.
        Raises: None.
        """
        progress, cross_track, yaw = self._tracking(bullet, robot)
        remaining = (
            progress - 0.7
            if self.returning
            else self.route["distance"] - 1.1 - progress
        )
        if (
            active
            and self.profile == "turnaround"
            and self.phase != "turning"
            and remaining < 0.25
        ):
            self.returning = not self.returning
            self.phase = "turning"
        heading = self.route["heading"] + (math.pi if self.returning else 0)
        if self.phase != "turning":
            heading -= np.clip(cross_track * (-1 if self.returning else 1), -0.4, 0.4)
        error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        if self.phase == "turning" and abs(error) < 0.22:
            self.turns += 1
            self.phase = "inbound" if self.returning else "outbound"
        target = min(self.speed, max(0.0, remaining * 1.4))
        if not active or self.phase == "turning":
            target = 0.0
        target *= max(0.0, math.cos(error)) ** 2
        self.forward += float(np.clip(target - self.forward, -0.024, 0.016))
        return np.array([self.forward, 0.0, np.clip(error * 2.0, -0.9, 0.9)])

    def _tracking(self, bullet, robot):
        position, orientation = bullet.getBasePositionAndOrientation(robot)
        yaw = bullet.getEulerFromQuaternion(orientation)[2]
        direction = np.array(
            [math.cos(self.route["heading"]), math.sin(self.route["heading"])]
        )
        delta = np.asarray(position[:2]) - self.route["start"][:2]
        return (
            float(delta @ direction),
            float(delta @ [-direction[1], direction[0]]),
            yaw,
        )
