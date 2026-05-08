from __future__ import annotations

import importlib
import json
import sys
import types

import pytest


@pytest.fixture()
def genesis_modules(monkeypatch):
    fake_torch = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    module_names = [
        "npa.genesis.diagnose",
        "npa.genesis.env_pick_place",
        "npa.genesis.eval_student",
        "npa.genesis.generate_demos",
        "npa.genesis.train_teacher",
        "npa.genesis.tune",
    ]
    for name in module_names:
        sys.modules.pop(name, None)

    modules = {name.rsplit(".", 1)[-1]: importlib.import_module(name) for name in module_names}

    yield modules

    for name in module_names:
        sys.modules.pop(name, None)


def test_env_config_defaults_and_action_space_metadata(genesis_modules):
    env_pick_place = genesis_modules["env_pick_place"]

    cfg = env_pick_place.EnvConfig()

    assert cfg.n_envs == 4096
    assert cfg.action_space == "joint"
    assert cfg.camera_res == (480, 640)
    assert env_pick_place.FrankaPickPlaceEnv.N_PRIV_OBS == 22


def test_ppo_config_builds_independent_train_cfg(genesis_modules):
    train_teacher = genesis_modules["train_teacher"]

    ppo = train_teacher.PPOConfig()
    cfg = ppo.to_train_cfg()
    cfg["policy"]["actor_hidden_dims"].append(64)

    assert cfg["policy"]["class_name"] == "ActorCritic"
    assert cfg["algorithm"]["class_name"] == "PPO"
    assert ppo.actor_hidden_dims == [256, 256, 128]


@pytest.mark.parametrize(
    ("trace_kwargs", "expected"),
    [
        ({"success": True}, "success"),
        ({"min_approach_dist": 1.0}, "approach"),
        ({"min_approach_dist": 0.01, "max_contact_count": 1}, "grasp"),
        (
            {
                "min_approach_dist": 0.01,
                "max_contact_count": 2,
                "max_cube_height": 0.01,
            },
            "lift",
        ),
        (
            {
                "min_approach_dist": 0.01,
                "max_contact_count": 2,
                "max_cube_height": 0.1,
                "min_place_dist": 0.5,
            },
            "place",
        ),
        (
            {
                "min_approach_dist": 0.01,
                "max_contact_count": 2,
                "max_cube_height": 0.1,
                "min_place_dist": 0.01,
            },
            "timeout",
        ),
    ],
)
def test_episode_trace_classifies_failure_phases(
    genesis_modules,
    trace_kwargs,
    expected,
):
    diagnose = genesis_modules["diagnose"]

    trace = diagnose.EpisodeTrace(**trace_kwargs)

    assert trace.classify() == expected


def test_diagnose_suggestions_and_serialization_are_json_safe(genesis_modules, tmp_path):
    diagnose = genesis_modules["diagnose"]

    joint = diagnose._get_approach_suggestion("joint")
    cartesian = diagnose._get_approach_suggestion("cartesian")
    serialized = diagnose._serialize_config_changes(
        {"friction_range": (0.6, 1.5), "grasp_weight": 5.0}
    )
    output = diagnose.save_diagnosis({"phase_counts": {"success": 1}}, tmp_path / "d.json")

    assert joint["config_changes"]["action_space"] == "cartesian"
    assert "action_space" not in cartesian["config_changes"]
    assert serialized["friction_range"] == [0.6, 1.5]
    assert json.loads(output.read_text()) == {"phase_counts": {"success": 1}}


def test_checkpoint_action_space_reader_handles_valid_and_invalid_metadata(
    genesis_modules,
    tmp_path,
):
    generate_demos = genesis_modules["generate_demos"]
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("placeholder")

    assert generate_demos._read_checkpoint_action_space(checkpoint) is None

    (tmp_path / "arch_config.json").write_text('{"action_space": "cartesian"}')
    assert generate_demos._read_checkpoint_action_space(checkpoint) == "cartesian"

    (tmp_path / "arch_config.json").write_text("{bad json")
    assert generate_demos._read_checkpoint_action_space(checkpoint) is None


def test_resolve_pretrained_dir_accepts_lerobot_checkpoint_layouts(
    genesis_modules,
    tmp_path,
):
    eval_student = genesis_modules["eval_student"]
    exact = tmp_path / "exact"
    exact.mkdir()
    (exact / "config.json").write_text("{}")

    root = tmp_path / "root"
    latest = root / "checkpoints" / "000200" / "pretrained_model"
    older = root / "checkpoints" / "000100" / "pretrained_model"
    latest.mkdir(parents=True)
    older.mkdir(parents=True)
    (latest / "config.json").write_text("{}")
    (older / "config.json").write_text("{}")

    assert eval_student._resolve_pretrained_dir(exact) == exact
    assert eval_student._resolve_pretrained_dir(root) == latest

    with pytest.raises(eval_student.EvalError, match="Cannot find pretrained_model"):
        eval_student._resolve_pretrained_dir(tmp_path / "missing")


def test_tune_serializes_and_writes_env_overrides(genesis_modules, tmp_path):
    tune = genesis_modules["tune"]

    serialized = tune._serialize_overrides({"friction_range": (0.3, 1.2)})
    output = tmp_path / "round" / "env_overrides.json"
    tune._save_env_overrides(output, {"friction_range": (0.3, 1.2)})

    assert serialized == {"friction_range": [0.3, 1.2]}
    assert json.loads(output.read_text()) == {"friction_range": [0.3, 1.2]}
