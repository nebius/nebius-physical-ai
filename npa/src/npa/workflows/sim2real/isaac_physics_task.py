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


def module_source() -> str:
    """Return this module's own source, for shipping into the Isaac container.

    The Isaac-Lab image has no ``npa`` package, so the BYO trainer writes this
    file into the job (via heredoc) and the train wrapper imports it post-boot.
    """
    from pathlib import Path

    return Path(__file__).read_text(encoding="utf-8")


# Post-boot train wrapper (runs INSIDE the Isaac-Lab container). The hard
# constraint (verified on-cluster): isaaclab.envs/mdp + gym.spec of the stock
# task pull USD ``pxr``, which only exists AFTER ``AppLauncher`` boots — so the
# variant registration MUST happen post-boot. This wrapper enforces the order:
#   (1) boot AppLauncher  (2) import isaaclab_tasks  (3) register the variant
#   (4) run the rsl_rl OnPolicyRunner like stock train.py.
# It reuses isaac_physics_task.register() (shipped alongside) as the single
# source of truth for the friction/mass event terms.
TRAIN_WRAPPER_SCRIPT = r'''
import os, sys, traceback
SYS_DIR = os.environ.get("NPA_PHYS_MODULE_DIR", "/tmp/npa_phys")
sys.path.insert(0, SYS_DIR)
NUM_ENVS = int(os.environ.get("PHYS_NUM_ENVS", "64"))
ITERS = int(os.environ.get("PHYS_ITERS", "2"))
SEED = int(os.environ.get("PHYS_SEED", "0"))
OUT = os.environ.get("PHYS_OUT_DIR", "/tmp/physrun")
os.makedirs(OUT, exist_ok=True)
# (1) boot the sim app FIRST — everything Isaac/pxr must come after this.
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import torch
import gymnasium as gym
# (2) register the stock tasks, then (3) the physics variant (post-boot import).
import isaaclab_tasks  # noqa: F401
import isaac_physics_task as physmod
params = physmod.physics_params_from_env()
print("PHYS_PARAMS", params, flush=True)
try:
    task = physmod.register(params)
except Exception:
    print("PHYS_REGISTER_FAILED", flush=True); traceback.print_exc(); os._exit(42)
task = task or physmod.STOCK_TASK_ID
print("PHYS_TASK", task, flush=True)
# (4) build env + rsl_rl runner and train, like stock train.py.
try:
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except Exception:
        from omni.isaac.lab_rl.rsl_rl import RslRlVecEnvWrapper  # older layout
    from rsl_rl.runners import OnPolicyRunner
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=NUM_ENVS)
    if SEED:
        try:
            env_cfg.seed = SEED
        except Exception as e:
            print("could not set env_cfg.seed:", repr(e), flush=True)
        torch.manual_seed(SEED)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    acfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    acfg["max_iterations"] = ITERS
    # Guarantee a checkpoint even for a tiny probe run.
    acfg["save_interval"] = max(1, min(int(acfg.get("save_interval", 50) or 50), ITERS))
    if SEED:
        acfg["seed"] = SEED
    print("PHYS_AGENT_CFG_KEYS", sorted(acfg.keys()), flush=True)
    env = gym.make(task, cfg=env_cfg)
    # Definitive check that the generated-physics events are LIVE (not a silent
    # stock fallback): the variant's event manager must carry our startup terms.
    try:
        ev_terms = list(getattr(env.unwrapped, "event_manager").active_terms.get("startup", []))
    except Exception:
        try:
            ev_terms = list(env.unwrapped.event_manager.active_terms)
        except Exception:
            ev_terms = []
    has_phys = any("npa_object_material" in str(t) for t in ev_terms) and \
               any("npa_object_mass" in str(t) for t in ev_terms)
    print("PHYS_EVENT_TERMS", ev_terms, "has_physics_events", has_phys, flush=True)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, acfg, log_dir=OUT, device="cuda:0")
    runner.learn(num_learning_iterations=ITERS, init_at_random_ep_len=True)
    # Defensive explicit save (learn saves at save_interval; ensure one exists).
    try:
        runner.save(os.path.join(OUT, "model_%d.pt" % ITERS))
    except Exception as e:
        print("explicit save failed (learn may have saved already):", repr(e), flush=True)
    import glob
    ckpts = sorted(glob.glob(os.path.join(OUT, "**", "model_*.pt"), recursive=True))
    # Final summary (survives any upstream log truncation): task + applied physics
    # + whether the injected events were present + checkpoint count.
    print("PHYS_SUMMARY task=%s friction=%s mass_scale=%s physics_events=%s ckpts=%d"
          % (task, (params or {}).get("friction"), (params or {}).get("mass_scale"),
             has_phys, len(ckpts)), flush=True)
    print("PHYS_CKPTS", ckpts, flush=True)
    print("PHYS_TRAIN_DONE" if (ckpts and has_phys) else "PHYS_TRAIN_NO_CKPT", flush=True)
except Exception:
    print("PHYS_TRAIN_FAILED", flush=True); traceback.print_exc(); os._exit(43)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
'''

