"""Ensure native task configuration never imports implementation-side USD early."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from npa.workflows.navigation import reference


def _stub_configuration_types(monkeypatch):
    def term(**values):
        return SimpleNamespace(**values)

    modules = {
        "isaaclab.assets": SimpleNamespace(AssetBaseCfg=term),
        "isaaclab.managers": SimpleNamespace(
            EventTermCfg=term,
            ObservationTermCfg=term,
            RewardTermCfg=term,
            TerminationTermCfg=term,
        ),
        "isaaclab.sensors": SimpleNamespace(RayCasterCfg=term, patterns=None),
        "isaaclab.sim": SimpleNamespace(UsdFileCfg=term),
        "isaaclab.utils.configclass": SimpleNamespace(configclass=lambda cls: cls),
        "isaaclab.utils.string": SimpleNamespace(ResolvableString=str),
        "isaaclab_tasks.manager_based.navigation.config.anymal_c.navigation_env_cfg": SimpleNamespace(
            NavigationEnvCfg=type("NavigationEnvCfg", (), {})
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def task_config(monkeypatch):
    _stub_configuration_types(monkeypatch)
    # A real import of this module would traverse TerrainImporter -> pxr.
    name = "npa.workflows.navigation.reference_mdp"
    monkeypatch.setitem(sys.modules, name, None)
    path = Path(reference.__file__).with_name("reference_config.py")
    spec = importlib.util.spec_from_file_location("isolated_reference_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = SimpleNamespace(
        actions=SimpleNamespace(pre_trained_policy_action=SimpleNamespace()),
        events=SimpleNamespace(),
        commands=SimpleNamespace(pose_command=SimpleNamespace()),
        rewards=SimpleNamespace(
            termination_penalty=SimpleNamespace(),
            position_tracking=SimpleNamespace(weight=0.5),
            position_tracking_fine_grained=SimpleNamespace(weight=0.5),
        ),
        terminations=SimpleNamespace(),
    )
    module._task(cfg)
    return cfg, name


def test_task_configuration_defers_all_mdp_implementation_imports(task_config):
    cfg, name = task_config
    assert cfg.commands.pose_command.class_type == name + ":FixedGoalCommand"
    assert cfg.events.reset_base.func == name + ":reset_cases"
    assert cfg.rewards.progress.func == name + ":progress"
    assert cfg.rewards.arrival.func == name + ":arrival"
    assert cfg.terminations.base_contact.func == name + ":terminate"
    assert sys.modules[name] is None


def test_arrival_return_exceeds_loitering_including_timeout_bootstrap(task_config):
    cfg, _ = task_config
    # The pinned RewardManager multiplies every term by the native control dt.
    arrival_return = cfg.rewards.arrival.weight * 0.2
    maximum_occupancy_return = cfg.episode_length_s * (
        cfg.rewards.position_tracking.weight
        + cfg.rewards.position_tracking_fine_grained.weight
    )
    assert arrival_return == 40.0
    assert maximum_occupancy_return == 30.0
    assert arrival_return > maximum_occupancy_return
    # The native PPO timeout bootstrap is not bounded by the episode horizon.
    maximum_discounted_return = 0.2 * 1.0 / (1.0 - 0.99)
    assert maximum_discounted_return == pytest.approx(20.0)
    assert arrival_return > maximum_discounted_return
    for delayed_steps in (1, 5, 100):
        discount = 0.99**delayed_steps
        delayed_return = maximum_discounted_return * (1 - discount)
        delayed_return += discount * (arrival_return + 0.2)
        assert arrival_return > delayed_return


@pytest.mark.parametrize("large_presets", [False, True])
def test_physics_buffers_cover_measured_contact_workload(monkeypatch, large_presets):
    _stub_configuration_types(monkeypatch)
    physics = SimpleNamespace(
        gpu_found_lost_pairs_capacity=2**21,
        gpu_found_lost_aggregate_pairs_capacity=2**25,
        gpu_total_aggregate_pairs_capacity=2**21,
        gpu_collision_stack_size=2**30 if large_presets else 2**26,
        gpu_max_rigid_patch_count=2**20 if large_presets else 5 * 2**15,
    )
    preset = object()
    resolved = []

    def resolve(value):
        resolved.append(value)
        return physics

    monkeypatch.setitem(
        sys.modules,
        "isaaclab_tasks.utils.hydra",
        SimpleNamespace(resolve_presets=resolve),
    )
    path = Path(reference.__file__).with_name("reference_config.py")
    spec = importlib.util.spec_from_file_location("isolated_reference_physics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = SimpleNamespace(sim=SimpleNamespace(physics=preset), decimation=40)
    module._physics(cfg, 4000)
    assert resolved == [preset]
    assert cfg.sim.physics is physics
    assert physics.gpu_found_lost_pairs_capacity > 8_002_000
    assert physics.gpu_found_lost_aggregate_pairs_capacity > 168_010_000
    assert physics.gpu_total_aggregate_pairs_capacity > 8_002_000
    assert physics.gpu_collision_stack_size > 278_600_488
    assert physics.gpu_max_rigid_patch_count > 349_082
    if large_presets:
        assert physics.gpu_collision_stack_size == 2**30
        assert physics.gpu_max_rigid_patch_count == 2**20
    assert cfg.sim.render_interval == 40
