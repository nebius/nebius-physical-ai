"""Isaac Lab 3.0 compatibility remaps for upstream asset-tree moves."""

from __future__ import annotations

from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys

import pytest

from npa.workflows.sim2real.isaac_assets_compat import (
    migrate_rsl_rl_agent_cfg,
    remap_moved_franka_usd,
)

_STALE = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
)
_LEGACY = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/IsaacLab/Robots/FrankaEmika/Legacy/panda_instanceable.usd"
)


def _env_cfg(usd_path: str):
    spawn = SimpleNamespace(usd_path=usd_path)
    robot = SimpleNamespace(spawn=spawn)
    scene = SimpleNamespace(robot=robot)
    return SimpleNamespace(scene=scene)


def test_remaps_stale_stock_franka_usd_to_legacy_location() -> None:
    cfg = _env_cfg(_STALE)
    assert remap_moved_franka_usd(cfg) == _LEGACY
    assert cfg.scene.robot.spawn.usd_path == _LEGACY


def test_remap_is_idempotent_on_legacy_path() -> None:
    cfg = _env_cfg(_LEGACY)
    assert remap_moved_franka_usd(cfg) == _LEGACY
    assert cfg.scene.robot.spawn.usd_path == _LEGACY


def test_remap_leaves_byo_robot_specs_untouched() -> None:
    byo = "s3://customer-bucket/robots/custom_arm.usd"
    cfg = _env_cfg(byo)
    assert remap_moved_franka_usd(cfg) == byo
    assert cfg.scene.robot.spawn.usd_path == byo


def test_remap_handles_missing_robot_gracefully() -> None:
    cfg = SimpleNamespace(scene=SimpleNamespace())
    assert remap_moved_franka_usd(cfg) == ""


@pytest.mark.parametrize(
    "original,effective", [(_STALE, _LEGACY), ("custom.usd", "custom.usd")]
)
def test_shipped_smoke_uses_the_effective_robot_before_environment_creation(
    monkeypatch, original, effective
):
    path = (
        Path(__file__).resolve().parents[3]
        / "npa/docker/workbench/isaac-lab/smoke_functional.py"
    )
    spec = importlib.util.spec_from_file_location("isaac_native_smoke", path)
    smoke = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, smoke)
    spec.loader.exec_module(smoke)
    config = _env_cfg(original)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_tasks.utils",
        SimpleNamespace(parse_env_cfg=lambda *_args, **_kwargs: config),
    )
    monkeypatch.setitem(
        sys.modules,
        "gymnasium",
        SimpleNamespace(
            make=lambda task, *, cfg: (task, cfg.scene.robot.spawn.usd_path)
        ),
    )
    assert smoke._make_env("Isaac-Reach-Franka-v0", 64, "cuda:0") == (
        "Isaac-Reach-Franka-v0",
        effective,
    )


def test_cfg_migration_fails_open_without_isaaclab_rl() -> None:
    # Environments without isaaclab_rl (or with a pre-migration Isaac Lab) need
    # no migration; the cfg must pass through unchanged.
    sentinel = SimpleNamespace(policy="unchanged")
    assert migrate_rsl_rl_agent_cfg(sentinel) is sentinel
