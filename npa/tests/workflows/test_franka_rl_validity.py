"""Exercise measured-state rejection and mocked handoff ordering without claiming native physics proof."""

import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from npa.workflows import franka_rl_validity as validity


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def environment(torch):
    def proxy(value):
        return SimpleNamespace(torch=torch.tensor(value, dtype=torch.float32))

    class Scene(dict):
        env_origins = torch.zeros((2, 3))

    robot = SimpleNamespace(
        joint_pos=proxy([[0.0, 0.0, 0.4]] * 2),
        joint_vel=proxy([[0.0] * 3] * 2),
        joint_pos_limits=proxy([[[-1, 1], [-2, 2], [0, 0.8]]] * 2),
        joint_vel_limits=proxy([[2, 3, 1]] * 2),
        body_link_pos_w=proxy([[[0.0] * 3] * 2] * 2),
        body_link_quat_w=proxy([[[0.0, 0.0, 0.0, 1.0]] * 2] * 2),
        body_link_lin_vel_w=proxy([[[0.0] * 3] * 2] * 2),
        body_link_ang_vel_w=proxy([[[0.0] * 3] * 2] * 2),
        root_pos_w=proxy([[0.0] * 3] * 2),
        root_quat_w=proxy([[0.0, 0.0, 0.0, 1.0]] * 2),
    )
    obj = SimpleNamespace(
        root_pos_w=proxy([[0.5, 0.0, 0.06]] * 2),
        root_quat_w=proxy([[0.0, 0.0, 0.0, 1.0]] * 2),
        root_lin_vel_w=proxy([[0.0] * 3] * 2),
        root_ang_vel_w=proxy([[0.0] * 3] * 2),
    )
    arm = SimpleNamespace(latest_command_target=torch.tensor([[0.1, 0.2]] * 2))
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            npa_simulation_validity=validity.validity_contract(),
            sim=SimpleNamespace(dt=0.01),
            scene=SimpleNamespace(env_spacing=2.0),
        ),
        scene=Scene(
            robot=SimpleNamespace(
                data=robot, joint_names=["arm", "wrist", "passive_finger"]
            ),
            object=SimpleNamespace(data=obj),
        ),
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.tensor([7, 19]),
        common_step_counter=31,
        action_manager=SimpleNamespace(get_term=lambda name: arm),
        termination_manager=SimpleNamespace(
            get_term=Mock(return_value=torch.zeros(2, dtype=torch.bool))
        ),
    )
    env.unwrapped = env
    return env


def _joint_checks(env):
    data = env.scene["robot"].data
    return validity.joint_contract_violations(
        data.joint_pos.torch,
        data.joint_vel.torch,
        data.joint_pos_limits.torch,
        data.joint_vel_limits.torch,
        env.cfg.sim.dt,
    )


def test_contract_is_independent_and_legacy_configuration_does_not_load_isaac(
    monkeypatch,
):
    original = validity.validity_contract()
    changed = validity.validity_contract()
    changed["checks"].clear()
    assert validity.validity_contract() == original
    monkeypatch.setitem(sys.modules, "isaaclab.managers", None)
    config = SimpleNamespace(terminations={"existing": object()})
    existing = dict(config.terminations)
    validity.configure_validity(config, {})
    assert config.terminations == existing and not hasattr(
        config, "npa_simulation_validity"
    )


@pytest.mark.parametrize("dictionary", [True, False])
def test_configured_hook_is_lazy_and_preserves_existing_terminations(
    monkeypatch, dictionary
):
    factory = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(
        sys.modules, "isaaclab.managers", SimpleNamespace(TerminationTermCfg=factory)
    )
    existing = object()
    terms = {"time_out": existing} if dictionary else SimpleNamespace(time_out=existing)
    config = SimpleNamespace(terminations=terms)
    recipe = {"simulation_validity": validity.validity_contract()}
    validity.configure_validity(config, recipe)
    assert factory.call_count == 2
    assert (terms["time_out"] if dictionary else terms.time_out) is existing
    installed = terms if dictionary else vars(terms)
    assert set(installed) == {"time_out", "npa_validity", "npa_workspace_exit"}
    assert (
        installed["npa_validity"].func
        == "npa.workflows.franka_rl_validity:check_simulation_state"
    )
    assert (
        installed["npa_workspace_exit"].func
        == "npa.workflows.franka_rl_validity:outside_task_domain"
    )
    recipe["simulation_validity"]["checks"].clear()
    assert config.npa_simulation_validity == validity.validity_contract()


@pytest.mark.parametrize(
    "change",
    [
        {"rounding_epsilon_multiplier": 40},
        {"quaternion_norm_tolerance": 0.1},
        {"failure_action": "reset and continue"},
        {"checks": []},
        {"unsealed_threshold": 1000},
    ],
)
def test_tampered_contract_rejected_before_native_import(monkeypatch, change):
    monkeypatch.setitem(sys.modules, "isaaclab.managers", None)
    contract = {**validity.validity_contract(), **change}
    config = SimpleNamespace(terminations={})
    with pytest.raises(ValueError, match="Unsupported measured"):
        validity.configure_validity(config, {"simulation_validity": contract})
    assert config.terminations == {} and not hasattr(config, "npa_simulation_validity")


def test_legacy_state_is_not_read_or_certified(monkeypatch):
    env = SimpleNamespace(cfg=SimpleNamespace())
    channels = Mock(side_effect=AssertionError("legacy state must not be read"))
    monkeypatch.setattr(validity, "_state_channels", channels)
    validity.validate_simulation_state(env)
    assert validity.validity_evidence(env) == {
        "verified": False,
        "reason": "legacy run has no measured-state validity contract",
    }
    channels.assert_not_called()
    assert not hasattr(env, "npa_validity_checks")


@pytest.mark.parametrize("sign", [-1, 1])
def test_position_allowance_uses_one_physics_tick_not_control_decimation(
    environment, torch, sign
):
    env, data = environment, environment.scene["robot"].data
    # Two rad/s for 0.01 s permits 0.02 rad, not the 0.04-rad control-step distance.
    data.joint_pos.torch[0, 0] = sign * 1.019
    assert not _joint_checks(env)["joint_position"].any()
    data.joint_pos.torch[0, 0] = sign * 1.021
    assert _joint_checks(env)["joint_position"][0, 0]
    with pytest.raises(validity.SimulationValidityError) as caught:
        validity.validate_simulation_state(env)
    assert caught.value.evidence["position_allowance"] == pytest.approx(
        0.02 + 4 * torch.finfo(torch.float32).eps
    )


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("native_limit", [0.5, 2.0])
def test_velocity_accepts_four_eps_rounding_but_rejects_the_next_float(
    environment, torch, sign, native_limit
):
    data = environment.scene["robot"].data
    data.joint_vel_limits.torch[0, 0] = native_limit
    limit = data.joint_vel_limits.torch[0, 0]
    boundary = limit + 4 * torch.finfo(torch.float32).eps * max(1, native_limit)
    data.joint_vel.torch[0, 0] = sign * boundary
    assert not _joint_checks(environment)["joint_velocity"].any()
    data.joint_vel.torch[0, 0] = sign * torch.nextafter(
        boundary, torch.tensor(float("inf"))
    )
    with pytest.raises(validity.SimulationValidityError) as caught:
        validity.validate_simulation_state(environment)
    assert caught.value.evidence["channel"] == "joint_velocity"


def test_continuous_sentinel_bounds_accept_rotation_but_never_disable_velocity_checks(
    environment, torch
):
    data = environment.scene["robot"].data
    maximum = torch.finfo(torch.float32).max
    data.joint_pos_limits.torch[:, 0] = torch.tensor([-maximum, maximum])
    assert not torch.isfinite(
        data.joint_pos_limits.torch[:, 0, 1] - data.joint_pos_limits.torch[:, 0, 0]
    ).all()
    data.joint_pos.torch[:, 0] = torch.tensor([1e8, -1e8])
    validity.validate_simulation_state(environment)
    data.joint_vel.torch[1, 0] = 1000.0
    with pytest.raises(validity.SimulationValidityError) as caught:
        validity.validate_simulation_state(environment)
    assert caught.value.evidence["channel"] == "joint_velocity"
    assert caught.value.evidence["indices"] == [1, 0]


def test_huge_finite_passive_joint_is_rejected_despite_small_arm_commands(environment):
    environment.scene["robot"].data.joint_pos.torch[1, 2] = 1e8
    with pytest.raises(validity.SimulationValidityError) as caught:
        validity.check_simulation_state(environment)
    evidence = caught.value.evidence
    assert evidence["joint_name"] == "passive_finger"
    assert evidence["constraint"] == "native_limit_exceeded"
    assert evidence["measured"] == 1e8 and evidence["environment_index"] == 1
    assert evidence["commanded_arm_target"] == pytest.approx([0.1, 0.2])
    assert evidence["episode_step"] == 19 and evidence["control_step"] == 31
    assert evidence["physics_dt"] == 0.01 and evidence["control_dt"] == 0.02
    assert (
        evidence["phase"] == "post_physics_pre_reward"
        and evidence["substeps_monitored"] is False
    )
    assert (
        evidence["contract_sha256"]
        == hashlib.sha256(
            json.dumps(evidence["contract"], sort_keys=True).encode()
        ).hexdigest()
    )
    assert environment.npa_validity_failure is evidence
    json.dumps(evidence, allow_nan=False)


_CHANNELS = [
    "joint_position",
    "joint_velocity",
    "robot_body_position",
    "robot_body_quaternion",
    "robot_body_velocity",
    "robot_body_angular_velocity",
    "robot_root_position",
    "robot_root_quaternion",
    "object_position",
    "object_quaternion",
    "object_velocity",
    "object_angular_velocity",
]


@pytest.mark.parametrize("channel", _CHANNELS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_every_measured_state_channel_rejects_nonfinite_values(
    environment, channel, value
):
    tensor = validity._state_channels(environment)[channel]
    index = (1, *((0,) * (tensor.ndim - 1)))
    tensor[index] = value
    with pytest.raises(validity.SimulationValidityError) as caught:
        validity.validate_simulation_state(environment)
    evidence = caught.value.evidence
    assert evidence["constraint"] == "nonfinite" and evidence["channel"] == channel
    assert evidence["indices"] == list(index) and evidence["measured"] is None
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize(
    "channel", ["robot_body_quaternion", "robot_root_quaternion", "object_quaternion"]
)
def test_quaternion_normalization_is_checked_independently_of_finiteness(
    environment, channel
):
    quaternion = validity._state_channels(environment)[channel]
    quaternion[1, ..., 3] = 1.00005
    validity.validate_simulation_state(environment)
    quaternion[1, ..., 3] = 1.001
    with pytest.raises(validity.SimulationValidityError) as caught:
        validity.validate_simulation_state(environment)
    assert caught.value.evidence["constraint"] == "nonunit_quaternion"
    assert caught.value.evidence["measured"] == pytest.approx([0, 0, 0, 1.001])


def test_first_fault_is_deterministic_and_does_not_increment_valid_coverage(
    environment,
):
    validity.validate_simulation_state(environment)
    environment.scene["robot"].data.joint_vel.torch[1, 2] = float("nan")
    environment.scene["object"].data.root_pos_w.torch[0, 0] = float("inf")
    with pytest.raises(validity.SimulationValidityError) as caught:
        validity.validate_simulation_state(environment)
    assert caught.value.evidence["channel"] == "joint_velocity"
    assert caught.value.evidence["indices"] == [1, 2]
    assert environment.npa_validity_checks == 1
    with pytest.raises(ValueError, match="did not complete"):
        validity.validity_evidence(environment)


def test_valid_hook_does_not_terminate_or_change_task_state(environment, torch):
    before = {
        name: value.clone()
        for name, value in validity._state_channels(environment).items()
    }
    with pytest.raises(ValueError, match="did not complete"):
        validity.validity_evidence(environment)
    done = validity.check_simulation_state(environment)
    assert done.dtype == torch.bool and done.shape == (2,) and not done.any()
    assert all(
        torch.equal(value, validity._state_channels(environment)[name])
        for name, value in before.items()
    )
    receipt = validity.validity_evidence(environment)
    assert (
        receipt["verified"]
        and receipt["checked_batches"] == 1
        and receipt["environments_per_batch"] == 2
    )
    assert receipt["substeps_monitored"] is False
    receipt["contract"]["checks"].clear()
    assert environment.cfg.npa_simulation_validity == validity.validity_contract()


@pytest.mark.parametrize(
    "invalid",
    [
        "negative_dt",
        "nan_dt",
        "zero_velocity",
        "infinite_velocity",
        "reversed_limits",
        "nan_limits",
        "shape",
    ],
)
def test_invalid_native_limit_definitions_fail_closed(environment, invalid):
    data = environment.scene["robot"].data
    if invalid in ("negative_dt", "nan_dt"):
        environment.cfg.sim.dt = -0.01 if invalid == "negative_dt" else float("nan")
    elif invalid in ("zero_velocity", "infinite_velocity"):
        data.joint_vel_limits.torch[0, 0] = (
            0 if invalid == "zero_velocity" else float("inf")
        )
    elif invalid == "reversed_limits":
        data.joint_pos_limits.torch[0, 0] = data.joint_pos_limits.torch[0, 0].flip(0)
    elif invalid == "nan_limits":
        data.joint_pos_limits.torch[0, 0, 0] = float("nan")
    else:
        data.joint_vel_limits.torch = data.joint_vel_limits.torch[:, :1]
    with pytest.raises(ValueError, match="Native joint bounds"):
        validity.validate_simulation_state(environment)


@pytest.fixture
def wrapper_model(environment, monkeypatch, torch):
    events = []

    class NativeWrapperModel:
        """Model the upstream boundary sequence; this mock does not prove native execution order."""

        def __init__(self, env, clip_actions):
            self.unwrapped = env.unwrapped
            self.clip_actions = clip_actions
            self.corrupt_during_physics = self.corrupt_after_reset = False
            self.step_result = (
                {"policy": torch.zeros((2, 3))},
                torch.tensor([3.0, -2.0]),
                torch.tensor([True, False]),
                {"time_outs": torch.tensor([False, False])},
            )
            events.append("wrapper_constructed")

        def reset(self):
            events.append("reset")
            return "reset_observation", {}

        def step(self, actions):
            self.received_actions = actions
            events.append("physics")
            if self.corrupt_during_physics:
                self.unwrapped.scene["robot"].data.joint_pos.torch[0, 0] = 1000
            validity.check_simulation_state(self.unwrapped)
            events.extend(("reward", "automatic_reset"))
            if self.corrupt_after_reset:
                self.unwrapped.scene["object"].data.root_pos_w.torch[0, 0] = float(
                    "nan"
                )
            return self.step_result

    monkeypatch.setitem(
        sys.modules,
        "isaaclab_rl.rsl_rl",
        SimpleNamespace(RslRlVecEnvWrapper=NativeWrapperModel),
    )
    return events


def test_mocked_native_hook_blocks_reward_reset_and_learner_handoff(
    environment, wrapper_model
):
    wrapper = validity.build_validated_wrapper(environment, clip_actions=1.0)
    wrapper.corrupt_during_physics = True
    with pytest.raises(validity.SimulationValidityError) as caught:
        wrapper.step(None)
        wrapper_model.append("learner_consumed_transition")
    assert wrapper_model == ["wrapper_constructed", "physics"]
    assert caught.value.evidence["phase"] == "post_physics_pre_reward"


def test_checked_wrapper_rejects_invalid_automatic_reset_before_handoff(
    environment, wrapper_model
):
    wrapper = validity.build_validated_wrapper(environment, clip_actions=None)
    wrapper.corrupt_after_reset = True
    with pytest.raises(validity.SimulationValidityError) as caught:
        wrapper.step(None)
        wrapper_model.append("learner_consumed_observation")
    assert wrapper_model == [
        "wrapper_constructed",
        "physics",
        "reward",
        "automatic_reset",
    ]
    assert caught.value.evidence["phase"] == "post_step_before_policy"


def test_invalid_initial_state_prevents_learner_construction(
    environment, wrapper_model, monkeypatch
):
    from npa.workflows import (
        franka_rl_embodiments,
        franka_rl_environment,
        franka_rl_physics,
    )

    environment.scene["robot"].data.joint_vel.torch[0, 2] = 50
    learner = Mock()
    config = SimpleNamespace(clip_actions=None, to_dict=lambda: {})
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_tasks.utils",
        SimpleNamespace(load_cfg_from_registry=lambda *args: config),
    )
    monkeypatch.setattr(
        sys.modules["isaaclab_rl.rsl_rl"],
        "handle_deprecated_rsl_rl_cfg",
        lambda cfg, version: cfg,
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules, "rsl_rl.runners", SimpleNamespace(OnPolicyRunner=learner)
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "5.0.1")
    monkeypatch.setattr(
        franka_rl_embodiments,
        "embodiment_evidence",
        lambda env, recipe: {"profile": {}},
    )
    monkeypatch.setattr(franka_rl_physics, "stability_evidence", lambda env, recipe: {})
    recipe = {
        "task": "mock_task",
        "seed": 42,
        "steps_per_env": 24,
        "checkpoint_interval": 50,
        "iterations": 100,
    }
    with pytest.raises(validity.SimulationValidityError) as caught:
        franka_rl_environment.build_runner(environment, recipe)
    learner.assert_not_called()
    assert wrapper_model == ["wrapper_constructed"]
    assert caught.value.evidence["phase"] == "initial_before_learner"


def test_checked_explicit_reset_validates_before_returning_observations(
    environment, wrapper_model
):
    wrapper = validity.build_validated_wrapper(environment, clip_actions=0.5)
    assert wrapper.reset() == ("reset_observation", {})
    assert wrapper.clip_actions == 0.5
    environment.scene["robot"].data.joint_pos.torch[1, 1] = 100
    with pytest.raises(validity.SimulationValidityError) as caught:
        wrapper.reset()
    assert caught.value.evidence["phase"] == "reset_before_policy"


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("spacing", [2.0, 6.0])
def test_workspace_is_xyz_relative_to_origin_and_half_spacing_is_inclusive(
    environment, torch, axis, sign, spacing
):
    environment.cfg.scene.env_spacing = spacing
    environment.scene.env_origins = torch.tensor(
        [[10.0, -20.0, 30.0], [-100.0, 200.0, -300.0]]
    )
    position = environment.scene["object"].data.root_pos_w.torch
    position.copy_(environment.scene.env_origins)
    position[1, axis] += sign * spacing / 2
    assert not validity.outside_task_domain(environment).any()
    position[1, axis] += sign * 0.01
    assert validity.outside_task_domain(environment).tolist() == [False, True]
    # A finite task departure is an ordinary termination, not a physics-validity fault.
    assert not validity.check_simulation_state(environment).any()
    assert not hasattr(environment, "npa_validity_failure")


@pytest.mark.parametrize("spacing", [0.0, -1.0, float("nan"), float("inf")])
def test_workspace_requires_finite_positive_scene_spacing(environment, spacing):
    environment.cfg.scene.env_spacing = spacing
    with pytest.raises(ValueError, match="finite positive task envelope"):
        validity.outside_task_domain(environment)


@pytest.mark.parametrize("terminal_reward", [-8.0, 8.0])
def test_terminal_workspace_reward_is_zero_after_reset_without_changing_other_handoff_data(
    environment, wrapper_model, torch, terminal_reward
):
    wrapper = validity.build_validated_wrapper(environment, clip_actions=None)
    obs, rewards, dones, extras = wrapper.step_result
    rewards[0] = terminal_reward
    environment.termination_manager.get_term.return_value = torch.tensor([True, False])
    # The scene is already back inside the workspace: retained pre-reset flags must be used.
    assert not validity.outside_task_domain(environment).any()
    actions = torch.tensor([[0.2, -0.3], [0.1, 0.4]])
    before = actions.clone()
    actual_obs, actual_rewards, actual_dones, actual_extras = wrapper.step(actions)
    assert actual_obs is obs and actual_dones is dones and actual_extras is extras
    assert actual_rewards.tolist() == [0.0, -2.0]
    assert rewards.tolist() == [terminal_reward, -2.0]
    assert wrapper.received_actions is actions and torch.equal(actions, before)
    environment.termination_manager.get_term.assert_called_once_with(
        "npa_workspace_exit"
    )
    assert wrapper_model == [
        "wrapper_constructed",
        "physics",
        "reward",
        "automatic_reset",
    ]


def test_missing_guarded_workspace_term_cannot_silently_preserve_terminal_reward(
    environment, wrapper_model
):
    wrapper = validity.build_validated_wrapper(environment, clip_actions=None)
    environment.termination_manager.get_term.side_effect = RuntimeError(
        "required termination term is absent"
    )
    with pytest.raises(RuntimeError, match="required termination term"):
        wrapper.step(None)
        wrapper_model.append("learner_consumed_transition")
    assert "learner_consumed_transition" not in wrapper_model


def test_legacy_wrapper_keeps_reset_and_step_results_without_reading_state_or_workspace_terms(
    environment, wrapper_model, monkeypatch, torch
):
    del environment.cfg.npa_simulation_validity
    del environment.scene
    environment.termination_manager.get_term.side_effect = AssertionError(
        "legacy termination must not be read"
    )
    channels = Mock(side_effect=AssertionError("legacy state must not be read"))
    monkeypatch.setattr(validity, "_state_channels", channels)
    wrapper = validity.build_validated_wrapper(environment, clip_actions=None)
    assert wrapper.reset() == ("reset_observation", {})
    assert wrapper.step(None) is wrapper.step_result
    exits = validity.task_domain_exits(environment)
    assert exits.dtype == torch.bool and exits.shape == (2,) and not exits.any()
    channels.assert_not_called()
    environment.termination_manager.get_term.assert_not_called()
    assert not hasattr(environment, "npa_validity_checks")
