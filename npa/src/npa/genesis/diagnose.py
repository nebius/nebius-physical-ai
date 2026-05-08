"""Diagnose teacher policy failures in Genesis pick-and-place.

Loads a teacher checkpoint, runs N rollouts with per-timestep tracking
of privileged state, classifies each episode into a failure phase, and
returns a summary identifying the bottleneck with a suggested fix.

Failure phases (in task order):
    approach  — gripper never got close to the cube
    grasp     — reached the cube but never achieved finger contact
    lift      — contacted but cube never rose above the table
    place     — lifted cube but never reached the target zone
    timeout   — reached max_episode_steps (fallback)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class DiagnoseError(Exception):
    pass


# ── Failure phase thresholds ───────────────────────────────────────────

APPROACH_DIST_THRESHOLD = 0.08   # gripper must get within 8 cm of cube
LIFT_HEIGHT_THRESHOLD = 0.03     # cube must rise 3 cm above init z
PLACE_DIST_THRESHOLD = 0.08     # cube must get within 8 cm of target

# Keys that override diagnosis thresholds rather than EnvConfig fields.
_THRESHOLD_KEYS = {
    "approach_threshold": "approach_dist",
    "lift_threshold": "lift_height",
    "place_threshold": "place_dist",
}


# ── Per-episode tracking ───────────────────────────────────────────────

@dataclass
class EpisodeTrace:
    """Per-timestep metrics collected during a single rollout."""

    min_approach_dist: float = float("inf")   # closest gripper-to-cube distance
    max_contact_count: int = 0                 # peak simultaneous finger contacts (0/1/2)
    max_cube_height: float = 0.0              # highest cube z relative to init
    min_place_dist: float = float("inf")      # closest cube-to-target distance
    steps: int = 0
    success: bool = False

    def classify(
        self,
        approach_dist: float = APPROACH_DIST_THRESHOLD,
        lift_height: float = LIFT_HEIGHT_THRESHOLD,
        place_dist: float = PLACE_DIST_THRESHOLD,
    ) -> str:
        """Classify into a failure phase based on recorded metrics."""
        if self.success:
            return "success"
        if self.min_approach_dist > approach_dist:
            return "approach"
        if self.max_contact_count < 2:
            return "grasp"
        if self.max_cube_height < lift_height:
            return "lift"
        if self.min_place_dist > place_dist:
            return "place"
        return "timeout"


# ── Suggestion table ───────────────────────────────────────────────────

def _get_approach_suggestion(action_space: str) -> dict[str, Any]:
    """Build the approach bottleneck suggestion, varying by action space.

    When using joint-space actions, the policy often hits kinematic local
    minima — the arm can't reorient its wrist to reach forward.  Switching
    to Cartesian actions (delta xyz resolved via IK) eliminates this class
    of failure because the policy directly controls end-effector position.
    """
    if action_space == "joint":
        return {
            "description": (
                "Gripper never reached the cube. Joint-space actions cause "
                "kinematic local minima — the arm lowers toward the table but "
                "can't reorient its wrist to reach the cube. Switch to "
                "Cartesian action space (--action-space cartesian) so the "
                "policy controls end-effector position directly via IK. "
                "Also consider switching to linear distance reward "
                "(approach_scale=0) for uniform gradient at all distances."
            ),
            "fix": "switch_to_cartesian",
            "config_changes": {
                "approach_weight": 5.0,
                "approach_scale": 0,
                "action_space": "cartesian",
            },
            "human_hint": (
                "Switch to --action-space cartesian to avoid kinematic local "
                "minima. Also set approach_scale=0 for linear reward."
            ),
        }
    # Already in cartesian — only reward tuning will help
    return {
        "description": (
            "Gripper never reached the cube. The default exponential reward "
            "has near-zero gradient at starting distance (~0.58m). Switch "
            "to linear distance reward (approach_scale=0) for uniform "
            "gradient at all distances."
        ),
        "fix": "increase_approach_reward",
        "config_changes": {"approach_weight": 5.0, "approach_scale": 0},
        "human_hint": "Set approach_scale=0 for linear reward, increase approach_weight.",
    }


SUGGESTIONS: dict[str, dict[str, Any]] = {
    # "approach" is handled dynamically by _get_approach_suggestion()
    "approach": {
        "description": (
            "Gripper never reached the cube. The default exponential reward "
            "has near-zero gradient at starting distance (~0.58m). Switch "
            "to linear distance reward (approach_scale=0) for uniform "
            "gradient at all distances."
        ),
        "fix": "increase_approach_reward",
        "config_changes": {"approach_weight": 5.0, "approach_scale": 0},
        "human_hint": "Set approach_scale=0 for linear reward, increase approach_weight.",
    },
    "grasp": {
        "description": (
            "Gripper reached the cube but never achieved a two-finger grasp. "
            "The gripper action range may be too narrow, or the grasp reward "
            "is too low relative to approach."
        ),
        "fix": "increase_grasp_reward",
        "config_changes": {"grasp_weight": 5.0},
        "human_hint": "Increase grasp_weight or check gripper action range.",
    },
    "lift": {
        "description": (
            "Cube was grasped but never lifted off the table. Friction may "
            "be too low (cube slips) or the grasp reward doesn't incentivize "
            "holding."
        ),
        "fix": "increase_friction_and_grasp",
        "config_changes": {"friction_range": (0.6, 1.5), "grasp_weight": 5.0},
        "human_hint": "Raise friction_range lower bound and increase grasp_weight.",
    },
    "place": {
        "description": (
            "Cube was lifted but never reached the target zone. The episode "
            "may be too short, or the place reward weight is too low."
        ),
        "fix": "increase_place_reward_and_steps",
        "config_changes": {"place_weight": 6.0, "max_episode_steps": 750},
        "human_hint": "Increase place_weight and max_episode_steps.",
    },
    "timeout": {
        "description": (
            "Episodes timed out without a clear single failure mode. The "
            "policy may be oscillating or the episode length is too short."
        ),
        "fix": "increase_max_steps",
        "config_changes": {"max_episode_steps": 750},
        "human_hint": "Increase max_episode_steps.",
    },
}


# ── Trace update helper ────────────────────────────────────────────────


def _update_traces(
    priv_obs: dict,
    batch_traces: list[EpisodeTrace],
    env_done: list[bool],
    cube_init_z: float,
) -> None:
    """Update per-env episode traces from a privileged obs dict.

    Reads all metrics from the obs dict (ee_pos, object_pose,
    contact_flags, goal_position) rather than from the simulation
    directly, so both this function and _update_traces_from_obs use
    the same data source.
    """
    ee_pos = priv_obs["ee_pos"]              # (n_envs, 3)
    cube_pos = priv_obs["object_pose"][:, :3]  # (n_envs, 3)
    contacts = priv_obs["contact_flags"]     # (n_envs, 2)
    target_pos = priv_obs["goal_position"]   # (n_envs, 3)

    approach_dist = torch.norm(ee_pos - cube_pos, dim=-1).cpu().numpy()
    contact_count = (contacts > 0.5).sum(dim=-1).cpu().numpy()
    cube_z = cube_pos[:, 2].cpu().numpy()
    place_dist = torch.norm(cube_pos - target_pos, dim=-1).cpu().numpy()

    n_envs = len(batch_traces)
    for i in range(n_envs):
        if env_done[i]:
            continue
        t = batch_traces[i]
        t.min_approach_dist = min(t.min_approach_dist, float(approach_dist[i]))
        t.max_contact_count = max(t.max_contact_count, int(contact_count[i]))
        t.max_cube_height = max(t.max_cube_height, float(cube_z[i]) - cube_init_z)
        t.min_place_dist = min(t.min_place_dist, float(place_dist[i]))


def _update_traces_from_obs(
    obs: dict,
    batch_traces: list[EpisodeTrace],
    env_done: list[bool],
    dones,
    cube_init_z: float,
) -> None:
    """Update traces for just-finished envs using the obs dict from env.step().

    env.step() returns the terminal obs BEFORE auto-reset, so this is
    safe to call even though the simulation state has already been reset
    for done envs.  Only updates envs where dones[i] is True and
    env_done[i] is False (i.e. newly finished this step).
    """
    ee_pos = obs["ee_pos"]
    cube_pos = obs["object_pose"][:, :3]
    contacts = obs["contact_flags"]
    target_pos = obs["goal_position"]

    approach_dist = torch.norm(ee_pos - cube_pos, dim=-1).cpu().numpy()
    contact_count = (contacts > 0.5).sum(dim=-1).cpu().numpy()
    cube_z = cube_pos[:, 2].cpu().numpy()
    place_dist = torch.norm(cube_pos - target_pos, dim=-1).cpu().numpy()

    dones_np = dones.cpu().numpy() if hasattr(dones, "cpu") else dones

    n_envs = len(batch_traces)
    for i in range(n_envs):
        if env_done[i] or not dones_np[i]:
            continue
        t = batch_traces[i]
        t.min_approach_dist = min(t.min_approach_dist, float(approach_dist[i]))
        t.max_contact_count = max(t.max_contact_count, int(contact_count[i]))
        t.max_cube_height = max(t.max_cube_height, float(cube_z[i]) - cube_init_z)
        t.min_place_dist = min(t.min_place_dist, float(place_dist[i]))


# ── Main entry point ───────────────────────────────────────────────────

def diagnose_teacher(
    checkpoint_path: Path,
    n_envs: int = 1024,
    n_episodes: int = 0,
    seed: int = 42,
    env_overrides: dict[str, Any] | None = None,
    action_space: str = "cartesian",
) -> dict[str, Any]:
    """Run diagnostic rollouts and classify failures.

    Args:
        checkpoint_path: Path to teacher model.pt.
        n_envs: Number of parallel Genesis environments.
        n_episodes: Total episodes to evaluate. 0 = one batch (n_envs episodes).
        seed: Random seed.
        env_overrides: Optional dict of EnvConfig field overrides
            (e.g. from a previous tune round).
        action_space: "cartesian" or "joint". Used to tailor suggestions
            (e.g. recommend switching to cartesian when approach fails
            in joint mode).

    Returns:
        Diagnosis dict with per-phase counts, bottleneck, and suggestion.

    Raises:
        DiagnoseError: On failure.
    """
    import numpy as np

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise DiagnoseError(f"Checkpoint not found: {checkpoint_path}")

    # Auto-detect action space from checkpoint metadata when available.
    from npa.genesis.generate_demos import _read_checkpoint_action_space

    saved_space = _read_checkpoint_action_space(checkpoint_path)
    if saved_space is not None and saved_space != action_space:
        logger.info(
            "Checkpoint was trained with action_space=%s (caller passed %s). "
            "Using the checkpoint's value.",
            saved_space, action_space,
        )
        action_space = saved_space

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Separate diagnosis-threshold overrides from EnvConfig overrides.
    # Keys like approach_threshold, lift_threshold, place_threshold
    # control episode classification, not the simulation.
    threshold_overrides: dict[str, float] = {}
    filtered_env_overrides: dict[str, Any] = {}
    if env_overrides:
        for k, v in env_overrides.items():
            if k in _THRESHOLD_KEYS:
                threshold_overrides[_THRESHOLD_KEYS[k]] = float(v)
            else:
                filtered_env_overrides[k] = v

    # Create environment
    try:
        from npa.genesis.env_pick_place import EnvConfig, FrankaPickPlaceEnv

        cfg_kwargs: dict[str, Any] = {
            "n_envs": n_envs,
            "enable_cameras": False,
            "domain_randomize": True,
            "action_space": action_space,
        }
        if filtered_env_overrides:
            cfg_kwargs.update(filtered_env_overrides)
        env_cfg = EnvConfig(**cfg_kwargs)
        env = FrankaPickPlaceEnv(env_cfg)
    except Exception as exc:
        raise DiagnoseError(f"Failed to create Genesis environment: {exc}") from exc

    # Load teacher
    try:
        from npa.genesis.generate_demos import _load_teacher_policy

        actor_critic = _load_teacher_policy(checkpoint_path, env)
    except Exception as exc:
        raise DiagnoseError(f"Failed to load teacher: {exc}") from exc

    # Determine episode count
    single_batch = n_episodes <= 0
    target = n_envs if single_batch else n_episodes
    traces: list[EpisodeTrace] = []
    total_collected = 0
    batch_num = 0

    cube_init_z = env_cfg.cube_init_pos[2]

    while total_collected < target:
        batch_num += 1
        env.reset()

        # Per-env tracking for this batch
        batch_traces = [EpisodeTrace() for _ in range(n_envs)]
        env_done = [False] * n_envs

        for step in range(env_cfg.max_episode_steps):
            priv_obs = env.get_privileged_obs()
            with torch.no_grad():
                actions = actor_critic.act_inference(priv_obs["flat"])

            # Track metrics BEFORE stepping (pre-action state)
            _update_traces(priv_obs, batch_traces, env_done, cube_init_z)

            # env.step() returns obs computed BEFORE auto-reset
            # (env_pick_place.py L275), then auto-resets done envs
            # (L287).  The returned obs dict is the terminal state
            # for envs that just finished — we must use it directly
            # rather than re-reading from the simulation post-reset.
            terminal_obs, _, dones, info = env.step(actions)
            success_flags = info["success"].cpu().numpy()

            # Update traces for just-finished envs using the terminal
            # obs dict (captured before auto-reset).  For still-alive
            # envs, the next iteration's pre-step _update_traces call
            # will record their post-action state correctly.
            _update_traces_from_obs(
                terminal_obs, batch_traces, env_done, dones, cube_init_z,
            )

            for i in range(n_envs):
                if env_done[i]:
                    continue
                batch_traces[i].steps = step + 1
                if dones[i]:
                    env_done[i] = True
                    batch_traces[i].success = bool(success_flags[i])

            if all(env_done):
                break

        # Collect completed episodes
        for i in range(n_envs):
            if total_collected >= target:
                break
            traces.append(batch_traces[i])
            total_collected += 1

        if single_batch:
            break

    # Classify each episode
    phase_counts: dict[str, int] = {
        "success": 0, "approach": 0, "grasp": 0,
        "lift": 0, "place": 0, "timeout": 0,
    }
    for t in traces:
        phase = t.classify(**threshold_overrides)
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    n_total = len(traces)
    success_count = phase_counts["success"]
    success_rate = success_count / n_total if n_total > 0 else 0.0

    # Identify bottleneck: the most common failure phase (excluding success)
    failure_phases = {k: v for k, v in phase_counts.items() if k != "success" and v > 0}
    if failure_phases:
        bottleneck = max(failure_phases, key=failure_phases.get)  # type: ignore[arg-type]
    else:
        bottleneck = "none"

    # Use dynamic suggestion for approach (varies by action space)
    if bottleneck == "approach":
        suggestion = _get_approach_suggestion(action_space)
    else:
        suggestion = SUGGESTIONS.get(bottleneck, {})

    # Build result
    result: dict[str, Any] = {
        "n_episodes": n_total,
        "success_count": success_count,
        "success_rate": round(success_rate, 4),
        "phase_counts": phase_counts,
        "bottleneck": bottleneck,
    }
    if suggestion:
        result["suggestion"] = {
            "description": suggestion["description"],
            "fix": suggestion["fix"],
            "config_changes": _serialize_config_changes(suggestion["config_changes"]),
            "human_hint": suggestion["human_hint"],
        }

    logger.info(
        "Diagnosis complete: %d/%d succeeded (%.1f%%), bottleneck=%s",
        success_count, n_total, success_rate * 100, bottleneck,
    )

    return result


def _serialize_config_changes(changes: dict[str, Any]) -> dict[str, Any]:
    """Make config_changes JSON-serializable (tuples → lists)."""
    out: dict[str, Any] = {}
    for k, v in changes.items():
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


def save_diagnosis(result: dict[str, Any], output_path: Path) -> Path:
    """Write diagnosis result to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
    return output_path
