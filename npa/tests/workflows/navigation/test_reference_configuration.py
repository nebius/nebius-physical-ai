"""Ensure native task configuration never imports implementation-side USD early."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_task_configuration_defers_all_mdp_implementation_imports(monkeypatch):
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
        rewards=SimpleNamespace(termination_penalty=SimpleNamespace()),
        terminations=SimpleNamespace(),
    )
    module._task(cfg)
    assert cfg.commands.pose_command.class_type == name + ":FixedGoalCommand"
    assert cfg.events.reset_base.func == name + ":reset_cases"
    assert cfg.rewards.progress.func == name + ":progress"
    assert cfg.terminations.base_contact.func == name + ":terminate"
    assert sys.modules[name] is None


def test_physics_buffers_cover_observed_coincident_population(monkeypatch):
    _stub_configuration_types(monkeypatch)
    physics = SimpleNamespace(
        gpu_found_lost_pairs_capacity=2**21,
        gpu_found_lost_aggregate_pairs_capacity=2**25,
        gpu_total_aggregate_pairs_capacity=2**21,
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
    assert cfg.sim.render_interval == 40
