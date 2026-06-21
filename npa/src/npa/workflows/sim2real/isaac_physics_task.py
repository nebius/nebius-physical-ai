"""Inject GENERATED-env physics (friction, mass) into Isaac-Lab training.

The BYO Isaac trainer reads a generated train-env spec that carries concrete
``physics`` (friction, mass_scale). Today those numbers are read but dropped:
the stock ``Isaac-Lift-Cube-Franka-v0`` task has no object ``mass_props`` /
physics-material field to override and no friction/mass startup event, so they
cannot be applied via a plain hydra ``env.*`` override.

This module registers a task *variant* that adds Isaac-Lab startup
randomization events (``randomize_rigid_body_material`` /
``randomize_rigid_body_mass``) on the manipuland, parameterised from the
generated physics. The trainer imports this module IN-PROCESS (so the gym
registry sees the new id) and then runs the stock ``train.py`` against
``NPA_PHYSICS_TASK_ID``. With no physics env vars set, ``register`` is a no-op
and the loop stays on the stock task — the proven path is untouched.

The pure helpers (``clamp``, ``physics_params_from_env``) are unit-tested
here; ``register`` touches Isaac-Lab internals and is exercised by an
on-cluster probe (it imports gymnasium/isaaclab, unavailable off-GPU).
"""

from __future__ import annotations

import os
from typing import Any

NPA_PHYSICS_TASK_ID = "NPA-Lift-Cube-Franka-Physics-v0"
STOCK_TASK_ID = "Isaac-Lift-Cube-Franka-v0"

# Bounds keep a noisy/garbage generated value from producing a degenerate sim
# (frictionless or absurdly heavy object) that would make training meaningless.
FRICTION_MIN, FRICTION_MAX = 0.1, 2.0
MASS_SCALE_MIN, MASS_SCALE_MAX = 0.2, 3.0


def clamp(value: Any, lo: float, hi: float, default: float) -> float:
    """Parse ``value`` as float and clamp to [lo, hi]; ``default`` on garbage."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if x != x:  # NaN
        return default
    return max(lo, min(hi, x))


def physics_params_from_env(env: dict[str, str] | None = None) -> dict[str, float] | None:
    """Build clamped physics params from NPA_GEN_FRICTION / NPA_GEN_MASS_SCALE.

    Returns ``None`` when neither is set, so the caller falls back to the stock
    task (the generated env had no physics, or physics injection is disabled).
    """
    env = os.environ if env is None else env
    fr = env.get("NPA_GEN_FRICTION")
    ms = env.get("NPA_GEN_MASS_SCALE")
    if (fr is None or fr == "") and (ms is None or ms == ""):
        return None
    return {
        "friction": clamp(fr, FRICTION_MIN, FRICTION_MAX, 1.0),
        "mass_scale": clamp(ms, MASS_SCALE_MIN, MASS_SCALE_MAX, 1.0),
    }


def register(params: dict[str, float] | None = None) -> str | None:
    """Register the physics task variant in the gym registry; return its id.

    No-op (returns ``None``) when no physics params are available. Imports
    Isaac-Lab lazily so this module is importable off-GPU for unit tests.
    """
    params = params if params is not None else physics_params_from_env()
    if not params:
        return None

    import gymnasium as gym  # noqa: WPS433 (lazy: GPU-only dep)
    from isaaclab.envs import mdp  # noqa: WPS433
    from isaaclab.managers import EventTermCfg as EventTerm  # noqa: WPS433
    from isaaclab.managers import SceneEntityCfg  # noqa: WPS433
    from isaaclab_tasks.manager_based.manipulation.lift.config.franka import (  # noqa: WPS433
        joint_pos_env_cfg as franka_lift,
    )

    fr = params["friction"]
    ms = params["mass_scale"]

    class _PhysicsLiftEnvCfg(franka_lift.FrankaCubeLiftEnvCfg):  # type: ignore[name-defined]
        def __post_init__(self) -> None:  # noqa: WPS610
            super().__post_init__()
            # Static == dynamic friction (single deterministic value from the
            # generated env); restitution left at 0; one material bucket.
            self.events.npa_object_material = EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "static_friction_range": (fr, fr),
                    "dynamic_friction_range": (fr, fr),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 1,
                },
            )
            self.events.npa_object_mass = EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("object"),
                    "mass_distribution_params": (ms, ms),
                    "operation": "scale",
                },
            )

    stock_kwargs = gym.spec(STOCK_TASK_ID).kwargs
    gym.register(
        id=NPA_PHYSICS_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _PhysicsLiftEnvCfg,
            "rsl_rl_cfg_entry_point": stock_kwargs.get("rsl_rl_cfg_entry_point"),
        },
    )
    return NPA_PHYSICS_TASK_ID
