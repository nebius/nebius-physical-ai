"""Measure navigation outcomes and peer isolation from real simulator transitions."""

from __future__ import annotations

import numpy as np

from npa.workflows.navigation.contract import finite_array


def snapshot(adapter, env, count: int) -> dict:
    """Validate world-frame poses, goals and physical contact measurements.

    Args:
        adapter: Trusted task module implementing measure(env).
        env: Native Isaac environment.
        count: Number of independent robots in the shared scene.
    Returns:
        Finite arrays for poses, goals and contact magnitudes.
    Raises:
        ValueError: Measurements are missing, negative or nonfinite.
    """
    raw = adapter.measure(env)
    shapes = {
        "position_m": (count, 3),
        "goal_m": (count, 2),
        "heading_rad": (count,),
        "obstacle_contact": (count,),
        "peer_contact": (count,),
        "physical_failure": (count,),
        "upright_cosine": (count,),
        "ground_clearance_m": (count,),
    }
    result = {key: finite_array(raw[key], shape, key) for key, shape in shapes.items()}
    if any((result[key] < 0).any() for key in ("obstacle_contact", "peer_contact")):
        raise ValueError("contact magnitudes cannot be negative")
    _physical_measurements(raw, result, count)
    return result


def _physical_measurements(raw, result, count):
    if not np.isin(result["physical_failure"], [0, 1]).all():
        raise ValueError("physical_failure must contain boolean indicators")
    if (np.abs(result["upright_cosine"]) > 1.00001).any():
        raise ValueError("upright_cosine must be a vertical alignment cosine")
    if (result["ground_clearance_m"] < 0).any():
        raise ValueError("ground_clearance_m cannot be negative")
    for key in ("minimum_ground_clearance_m", "ground_support_fraction"):
        if key in raw:
            result[key] = finite_array(raw[key], (count,), key)


def observations(value) -> dict:
    """Copy every policy and critic observation stream for isolation comparison.

    Args:
        value: Native observation mapping or tensor.
    Returns:
        Mapping of names to finite arrays.
    Raises:
        ValueError: An observation is empty or nonfinite.
    """
    if hasattr(value, "items"):
        result = {}
        for key, child in value.items():
            result.update(
                {f"{key}/{name}": array for name, array in observations(child).items()}
            )
        if not result:
            raise ValueError("empty observation mapping")
        return result
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=float)
    if array.ndim < 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("observations must be finite nonempty batched arrays")
    return {"value": array.copy()}


def verify_reset(adapter, env, cases, tolerance: float) -> dict:
    """Check that the task applied requested world poses and fixed goals.

    Args:
        adapter: Trusted task module with reset and measure functions.
        env: Native Isaac environment.
        cases: Deterministic case models, one per robot.
        tolerance: Maximum absolute position and goal error in metres.
    Returns:
        Measured reset snapshot.
    Raises:
        ValueError: Reset coordinates or goals differ from requested inputs.
    """
    adapter.reset(env, [case.model_dump() for case in cases])
    check = getattr(env.unwrapped, "npa_navigation_visibility_check", None)
    if check is not None:
        check()
    state = snapshot(adapter, env, len(cases))
    if state["physical_failure"].any():
        raise ValueError("adapter reset produced a physically invalid robot pose")
    for key, expected in (
        ("position_m", [c.position_m for c in cases]),
        ("goal_m", [c.goal_m for c in cases]),
    ):
        if not np.allclose(state[key], expected, atol=tolerance, rtol=0):
            raise ValueError(f"adapter reset did not apply {key}")
    headings = np.asarray([case.heading_rad for case in cases])
    error = (state["heading_rad"] - headings + np.pi) % (2 * np.pi) - np.pi
    if np.max(np.abs(error)) > tolerance:
        raise ValueError("adapter reset did not apply heading_rad")
    return state


def _probe_trace(adapter, env, wrapped, cases, actions, tolerance):
    import torch

    verify_reset(adapter, env, cases, tolerance)
    trace = [
        {
            "state": snapshot(adapter, env, len(cases)),
            "observations": observations(wrapped.get_observations()),
        }
    ]
    for action in actions:
        batch = torch.zeros((len(cases), len(action)), device=env.unwrapped.device)
        batch[0] = torch.tensor(action, device=batch.device)
        obs, _, done, _ = wrapped.step(batch)
        if bool(done.any()):
            raise ValueError("probe terminated/reset; choose nonterminal probe inputs")
        trace.append(
            {
                "state": snapshot(adapter, env, len(cases)),
                "observations": observations(obs),
            }
        )
    return trace


def _compare_traces(baseline, changed, tolerance):
    maximum = 0.0
    for left, right in zip(baseline, changed, strict=True):
        for group in ("state", "observations"):
            if left[group].keys() != right[group].keys():
                raise ValueError("observation/measurement streams changed during probe")
            for name, array in left[group].items():
                other = right[group][name]
                if array.shape != other.shape:
                    raise ValueError("probe measurement shapes changed")
                delta = float(np.max(np.abs(array[0] - other[0])))
                maximum = max(maximum, delta)
        if any(np.any(right["state"][key] > tolerance) for key in ("peer_contact",)):
            raise ValueError("robot-to-robot physics contact detected")
    if maximum > tolerance:
        raise ValueError("peer placement changed focal physics or perception")
    return maximum


def probe_isolation(adapter, env, wrapped, recipe) -> dict:
    """Run overlapping-peer and obstacle-contact controls in the same Isaac scene.

    Args:
        adapter: Trusted task module implementing reset and measurement.
        env: Native environment shared by all robots.
        wrapped: RSL-RL wrapper used by the learner.
        recipe: Validated recipe with native probe actions.
    Returns:
        Measured finite probe deltas and positive-control evidence.
    Raises:
        ValueError: Isolation, reset, motion or contact controls fail.
    """
    probe = recipe.probe
    parked = [probe.free] + [probe.parked] * (recipe.num_envs - 1)
    baseline = _probe_trace(
        adapter, env, wrapped, parked, probe.actions, probe.tolerance
    )
    motion = np.linalg.norm(
        baseline[-1]["state"]["position_m"][0] - probe.free.position_m
    )
    if motion <= probe.tolerance:
        raise ValueError("free-space probe did not move; isolation cannot be inferred")
    delta, contact = _probe_controls(adapter, env, wrapped, recipe, parked, baseline)
    return {
        "passed": True,
        "robots": recipe.num_envs,
        "maximum_peer_delta": delta,
        "free_motion_m": float(motion),
        "obstacle_contact": contact,
        "sensor_mode": recipe.sensor_mode,
        "camera_isolation_verified": False,
    }


def _probe_controls(adapter, env, wrapped, recipe, parked, baseline):
    probe = recipe.probe
    if any(
        np.any(row["state"][key] > probe.tolerance)
        for row in baseline
        for key in ("obstacle_contact", "peer_contact", "physical_failure")
    ):
        raise ValueError("free-space baseline has contacts or physical failure")
    overlap = _probe_trace(
        adapter,
        env,
        wrapped,
        [probe.free] * recipe.num_envs,
        probe.actions,
        probe.tolerance,
    )
    delta = _compare_traces(baseline, overlap, probe.tolerance)
    obstacle = _probe_trace(
        adapter,
        env,
        wrapped,
        [probe.obstacle] + parked[1:],
        probe.actions,
        probe.tolerance,
    )
    contact = max(float(row["state"]["obstacle_contact"][0]) for row in obstacle[1:])
    if contact <= probe.tolerance:
        raise ValueError("obstacle positive control produced no physical contact")
    return delta, contact


def episode_rows(cases, trajectory: list[dict], tolerance: float) -> list[dict]:
    """Compute goal, collision and path metrics without using task reward as success.

    Args:
        cases: Held-out cases.
        trajectory: Validated snapshots with an active mask before each step.
        tolerance: Success distance in metres.
    Returns:
        Per-case measured navigation outcomes.
    Raises:
        ValueError: Trajectory is empty or contains no rollout steps.
    """
    if len(trajectory) < 2:
        raise ValueError("evaluation has no rollout steps")
    rows = []
    for index, case in enumerate(cases):
        active = [trajectory[0]] + [s for s in trajectory[1:] if s["active"][index]]
        positions = np.asarray([s["position_m"][index] for s in active])
        distance = float(np.linalg.norm(positions[-1, :2] - case.goal_m))
        contacts = sum(s["obstacle_contact"][index] > 0 for s in active[1:])
        rows.append(
            _episode_row(case, active, positions, index, distance, contacts, tolerance)
        )
    return rows


def _episode_row(case, active, positions, index, distance, contacts, tolerance):
    peers = sum(s["peer_contact"][index] > 0 for s in active[1:])
    failures = sum(s["physical_failure"][index] > 0 for s in active)
    return {
        "case_id": case.id,
        "seed": case.seed,
        "steps": len(active) - 1,
        "goal_distance_m": distance,
        "collision_steps": int(contacts),
        "peer_collision_steps": int(peers),
        "physical_failure_steps": int(failures),
        "path_length_m": float(
            np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
        ),
        "success": bool(
            distance <= tolerance and not contacts and not peers and not failures
        ),
    }
