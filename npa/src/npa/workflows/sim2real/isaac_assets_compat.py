"""Compatibility remaps for upstream Omniverse asset-tree moves.

Isaac Lab 3.0.0b2.post1 pins the stock Franka Panda robot at
``{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd``. In the
Isaac 6.0 content tree NVIDIA later moved that asset (and its Props/Materials)
under ``FrankaEmika/Legacy/``, so the pinned URL now returns 404 and every
stock-Franka stage fails closed with ``USD file not found``. Remap the exact
stale path to its relocated, byte-identical Legacy location; any other robot
spec (BYO robots, explicit robot_spec_uri overrides) is left untouched.
"""

from __future__ import annotations

_STALE_SUFFIX = "/Robots/FrankaEmika/panda_instanceable.usd"
_LEGACY_SUFFIX = "/Robots/FrankaEmika/Legacy/panda_instanceable.usd"


def remap_moved_franka_usd(env_cfg) -> str:
    """Point a stale stock-Franka spawn at the relocated Legacy USD.

    Returns the effective robot USD path (possibly unchanged) for logging.
    """

    robot = getattr(getattr(env_cfg, "scene", None), "robot", None)
    spawn = getattr(robot, "spawn", None)
    usd_path = str(getattr(spawn, "usd_path", "") or "")
    if spawn is not None and usd_path.endswith(_STALE_SUFFIX) and "/Legacy/" not in usd_path:
        spawn.usd_path = usd_path[: -len(_STALE_SUFFIX)] + _LEGACY_SUFFIX
        return spawn.usd_path
    return usd_path


def migrate_rsl_rl_agent_cfg(agent_cfg):
    """Apply Isaac Lab's own rsl-rl cross-version migration to a registry cfg.

    Isaac Lab 3.0 cfg dataclasses still carry legacy fields (e.g. the model
    ``stochastic`` flag) that rsl-rl >= 5.0.0 no longer accepts; upstream's
    train scripts run ``handle_deprecated_rsl_rl_cfg`` before constructing
    ``OnPolicyRunner`` and so must every direct runner construction here.
    Absence of the utility (older Isaac Lab) means no migration is needed.
    """

    try:
        import importlib.metadata as _md

        from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

        migrated = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))
        return migrated if migrated is not None else agent_cfg
    except Exception as exc:  # noqa: BLE001 - fail open: older stacks need none
        print("RSL_RL_CFG_MIGRATION_SKIPPED", repr(exc), flush=True)
        return agent_cfg
