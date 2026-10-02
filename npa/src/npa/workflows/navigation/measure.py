"""Measure navigation outcomes and peer isolation from real simulator transitions."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from npa.workflows.navigation.contract import finite_array
from npa.workflows.navigation.probe_evidence import save_trace, trace_difference


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
        raise ValueError(
            "adapter reset did not apply heading_rad: "
            f"maximum wrapped error {np.max(np.abs(error)):.6g} rad"
        )
    return state


def _probe_trace(adapter, env, wrapped, cases, actions, tolerance, trace=None):
    import torch

    verify_reset(adapter, env, cases, tolerance)
    trace = [] if trace is None else trace
    trace.append(_trace_row(adapter, env, len(cases), wrapped.get_observations()))
    for action in actions:
        batch = torch.zeros((len(cases), len(action)), device=env.unwrapped.device)
        batch[0] = torch.tensor(action, device=batch.device)
        # Native managers replace metric tensors during step and mutate them
        # during later resets. Keep those tensors mutable outside this context.
        with torch.no_grad():
            obs, _, done, _ = wrapped.step(batch)
        if bool(done.any()):
            raise ValueError("probe terminated/reset; choose nonterminal probe inputs")
        trace.append(_trace_row(adapter, env, len(cases), obs))
    return trace


def _trace_row(adapter, env, count, obs):
    row = {"state": snapshot(adapter, env, count), "observations": observations(obs)}
    diagnostics = getattr(adapter, "probe_state", None)
    if diagnostics is not None:
        row["native"] = observations(diagnostics(env))
    return row


def _compare_traces(baseline, changed, tolerance, label="peer placement"):
    report = trace_difference(baseline, changed)
    if report["maximum_peer_contact"] > tolerance:
        raise ValueError("robot-to-robot physics contact detected")
    maximum = report["maximum"]
    if maximum["absolute_delta"] > tolerance:
        raise ValueError(
            f"{label} changed focal physics or perception: "
            f"{maximum['group']}/{maximum['stream']} step={maximum['step']} "
            f"component={maximum['component']} delta={maximum['absolute_delta']:.9g} "
            f"baseline={maximum['baseline']:.9g} changed={maximum['changed']:.9g}"
        )
    return maximum["absolute_delta"]


def probe_isolation(adapter, env, wrapped, recipe, output=None) -> dict:
    """Run overlapping-peer and obstacle-contact controls in the same Isaac scene.

    Args:
        adapter: Trusted task module implementing reset and measurement.
        env: Native environment shared by all robots.
        wrapped: RSL-RL wrapper used by the learner.
        recipe: Validated recipe with native probe actions.
        output: Optional native artifact directory retaining every measured trace.
    Returns:
        Measured finite probe deltas and positive-control evidence.
    Raises:
        ValueError: Isolation, reset, motion or contact controls fail.
    """
    probe = recipe.probe
    parked = [probe.free] + [probe.parked] * (recipe.num_envs - 1)
    baseline = _recorded_probe(adapter, env, wrapped, recipe, parked, output, "solo")
    motion = np.linalg.norm(
        baseline[-1]["state"]["position_m"][0] - probe.free.position_m
    )
    if motion <= probe.tolerance:
        raise ValueError("free-space probe did not move; isolation cannot be inferred")
    repeat, delta, contact = _probe_controls(
        adapter, env, wrapped, recipe, parked, baseline, output
    )
    return {
        "passed": True,
        "robots": recipe.num_envs,
        "maximum_peer_delta": delta,
        "maximum_repeat_delta": repeat,
        "free_motion_m": float(motion),
        "obstacle_contact": contact,
        "sensor_mode": recipe.sensor_mode,
        "camera_isolation_verified": False,
    }


def _probe_controls(adapter, env, wrapped, recipe, parked, baseline, output=None):
    probe = recipe.probe
    if any(
        np.any(row["state"][key] > probe.tolerance)
        for row in baseline
        for key in ("obstacle_contact", "peer_contact", "physical_failure")
    ):
        raise ValueError("free-space baseline has contacts or physical failure")
    repeat = _recorded_probe(adapter, env, wrapped, recipe, parked, output, "repeat")
    overlap = _recorded_probe(
        adapter, env, wrapped, recipe, [probe.free] * recipe.num_envs, output, "overlap"
    )
    _save_comparisons(output, baseline, repeat, overlap)
    obstacle = _recorded_probe(
        adapter, env, wrapped, recipe, [probe.obstacle] + parked[1:], output, "obstacle"
    )
    repeat_delta = _compare_traces(baseline, repeat, probe.tolerance, "repeated reset")
    delta = _compare_traces(baseline, overlap, probe.tolerance)
    contact = max(float(row["state"]["obstacle_contact"][0]) for row in obstacle[1:])
    if contact <= probe.tolerance:
        raise ValueError("obstacle positive control produced no physical contact")
    return repeat_delta, delta, contact


def assess_control_traces(recipe, traces, output):
    """Apply the existing physical control gates to independently recorded arms.

    Args:
        recipe: Unchanged sealed actions, reset cases and tolerances.
        traces: Complete solo, repeat, overlap and obstacle traces.
        output: Directory retaining the comparison report before rejection.
    Returns:
        Fresh-scene isolation results, without a warm-reset claim.
    Raises:
        ValueError: Any original motion, contact or trace-comparison gate fails.
    """
    if set(traces) != {"solo", "repeat", "overlap", "obstacle"}:
        raise ValueError("all four native controls are required")
    solo, repeat, overlap, obstacle = (
        traces[name] for name in ("solo", "repeat", "overlap", "obstacle")
    )
    _save_comparisons(output, solo, repeat, overlap)
    motion = np.linalg.norm(
        solo[-1]["state"]["position_m"][0] - recipe.probe.free.position_m
    )
    if motion <= recipe.probe.tolerance:
        raise ValueError("free-space probe did not move; isolation cannot be inferred")
    if any(
        np.any(row["state"][key] > recipe.probe.tolerance)
        for row in solo
        for key in ("obstacle_contact", "peer_contact", "physical_failure")
    ):
        raise ValueError("free-space baseline has contacts or physical failure")
    repeat_delta = _compare_traces(
        solo, repeat, recipe.probe.tolerance, "fresh-scene repeat"
    )
    peer_delta = _compare_traces(solo, overlap, recipe.probe.tolerance)
    contact = max(float(row["state"]["obstacle_contact"][0]) for row in obstacle[1:])
    if contact <= recipe.probe.tolerance:
        raise ValueError("obstacle positive control produced no physical contact")
    return {
        "passed": True,
        "robots": recipe.num_envs,
        "maximum_peer_delta": peer_delta,
        "maximum_repeat_delta": repeat_delta,
        "free_motion_m": float(motion),
        "obstacle_contact": contact,
        "sensor_mode": recipe.sensor_mode,
        "camera_isolation_verified": False,
        "control_protocol": "fresh_process_per_control.v1",
        "warm_reset_repeatability_verified": False,
    }


def _recorded_probe(
    adapter, env, wrapped, recipe, cases, output, name, retain_partial=False
):
    recorder = getattr(adapter, "record_probe_contacts", None)
    context = recorder(env, output, name) if recorder else nullcontext()
    trace = []
    try:
        with context:
            kwargs = {"trace": trace} if retain_partial else {}
            measured = _probe_trace(
                adapter,
                env,
                wrapped,
                cases,
                recipe.probe.actions,
                recipe.probe.tolerance,
                **kwargs,
            )
            trace = measured
    finally:
        if output is not None and trace:
            save_trace(output, name, trace)
    return trace


def _save_comparisons(output, baseline, repeat, overlap):
    if output is None:
        return
    from npa.workflows.navigation.artifacts import write_json

    write_json(
        output / "isolation-comparisons.json",
        {
            "schema": "npa.navigation.isolation-traces.v1",
            "focal_robot_index": 0,
            "observation_scope": "focal_robot_only",
            "repeatability": trace_difference(baseline, repeat),
            "peer_isolation": trace_difference(baseline, overlap),
        },
    )


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
