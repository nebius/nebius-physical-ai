"""Check outcome-driven training, unchanged evaluation, and physically bounded learned commands."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from npa.workflows import franka_rl
from npa.workflows.franka_rl_curriculum import TrainingCurriculum, training_ranges
from npa.workflows.franka_rl_learning import learning_profile, recipe_learning


def test_curriculum_advances_only_after_enough_successful_completed_episodes():
    curriculum = TrainingCurriculum(minimum_episodes=8, success_threshold=0.75)
    curriculum.observe(3, 4)
    assert curriculum.fraction == 0.25
    curriculum.observe(2, 4)
    assert curriculum.fraction == 0.25
    curriculum.observe(6, 8)
    assert curriculum.fraction == 0.4
    assert curriculum.history == [
        {"fraction": 0.25, "next_fraction": 0.25, "episodes": 8, "successes": 5,
         "success_rate": 0.625, "total_episodes": 8},
        {"fraction": 0.25, "next_fraction": 0.4, "episodes": 8, "successes": 6,
         "success_rate": 0.75, "total_episodes": 16},
    ]


def test_curriculum_saturates_at_the_exact_original_distribution():
    curriculum = TrainingCurriculum(1, 1.0)
    for _ in range(10):
        curriculum.observe(1, 1)
    assert curriculum.fraction == 1
    resets = {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)}
    goals = {"pos_x": (0.4, 0.6), "pos_y": (-0.25, 0.25), "pos_z": (0.25, 0.5)}
    assert training_ranges(1, resets, goals, (0.12, 0.18)) == (resets, goals)
    easier_reset, easier_goal = training_ranges(0.25, resets, goals, (0.12, 0.18))
    assert easier_reset["y"] == (-0.0625, 0.0625)
    assert 0.1 < easier_goal["pos_z"][0] < goals["pos_z"][0]
    assert goals["pos_z"] == (0.25, 0.5)


@pytest.mark.parametrize("successes,episodes", [(-1, 2), (3, 2), (0, -1), (True, 1)])
def test_curriculum_rejects_invalid_outcomes(successes, episodes):
    with pytest.raises(ValueError):
        TrainingCurriculum(8, 0.7).observe(successes, episodes)


def test_new_recipe_preserves_all_held_out_criteria_and_baseline_bindings():
    common = ["prepare", "--run-id", "learning-test", "--output-path", "s3://example/prepared/"]
    baseline = franka_rl._recipe(franka_rl._parser().parse_args(common))
    learned = franka_rl._recipe(franka_rl._parser().parse_args(common + ["--learning-recipe", "adaptive"]))
    assert learned.pop("learning") == learning_profile("adaptive")
    assert learned == baseline
    assert recipe_learning(baseline) is None
    corrupted = {"learning": learning_profile("adaptive")}
    corrupted["learning"]["curriculum"]["success_threshold"] = 0
    with pytest.raises(ValueError, match="Sealed learning"):
        recipe_learning(corrupted)


def _native_module(name, monkeypatch):
    path = Path(franka_rl.__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("learning_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def servo(monkeypatch):
    torch = pytest.importorskip("torch")

    class JointPositionAction:
        def __init__(self, cfg, env):
            self._asset, self._joint_ids = env.robot, [0, 1]
            self._raw_actions = torch.zeros(2, 2)
            self._processed_actions = torch.zeros(2, 2)

        @property
        def processed_actions(self):
            return self._processed_actions

        def process_actions(self, actions):
            self._raw_actions.copy_(actions)
            self._processed_actions.copy_(0.5 * actions)

        def reset(self, env_ids):
            self._raw_actions[env_ids] = 0

    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp.actions.joint_actions",
                        SimpleNamespace(JointPositionAction=JointPositionAction))
    module = _native_module("franka_rl_servo", monkeypatch)
    def proxy(value):
        return SimpleNamespace(torch=torch.tensor(value, dtype=torch.float32))
    data = SimpleNamespace(soft_joint_pos_limits=proxy([[[-3, 3], [-2, 2]]] * 2),
                           joint_vel_limits=proxy([[4, 1]] * 2), joint_pos=proxy([[0, 0]] * 2))
    env = SimpleNamespace(robot=SimpleNamespace(data=data), step_dt=0.02,
                          cfg=SimpleNamespace(npa_learning=learning_profile("adaptive")))
    return module.BoundedJointPositionAction(None, env), env, torch


def test_servo_bounds_physical_targets_and_slew_without_restricting_reach(servo):
    action, env, torch = servo
    previous = action.target.clone()
    for _ in range(200):
        action.process_actions(torch.tensor([[100.0, -100.0]] * 2))
        assert torch.all((action.target - previous).abs() <= torch.tensor([0.040001, 0.020001]))
        assert torch.all(action.target <= torch.tensor([3.0, 2.0]))
        assert torch.all(action.target >= torch.tensor([-3.0, -2.0]))
        previous = action.target.clone()
    assert torch.allclose(action.target, torch.tensor([[3.0, -2.0]] * 2))
    env.robot.data.joint_pos.torch[0] = torch.tensor([1.0, 0.5])
    action.reset([0])
    assert torch.equal(action.target[0], torch.tensor([1.0, 0.5]))
    assert torch.equal(action.target[1], torch.tensor([3.0, -2.0]))
    # The terminal command remains attributable even after automatic scene reset.
    assert torch.equal(action.latest_command_target, torch.tensor([[3.0, -2.0]] * 2))


def test_servo_refuses_nonfinite_policy_commands(servo):
    action, _, torch = servo
    with pytest.raises(ValueError, match="Nonfinite"):
        action.process_actions(torch.full((2, 2), float("nan")))


@pytest.fixture
def continuous_joint(servo):
    action, env, torch = servo
    maximum = torch.finfo(torch.float32).max
    hard = torch.tensor([[[-maximum, maximum], [-2.0, 2.0]]] * 2)
    midpoint = hard.mean(dim=-1)
    radius = (hard[..., 1] - hard[..., 0]) / 2
    env.robot.data.joint_pos_limits = SimpleNamespace(torch=hard)
    env.robot.data.soft_joint_pos_limits.torch = torch.stack((midpoint - radius, midpoint + radius), dim=-1)
    env.robot.data.joint_vel_limits.torch[:] = torch.tensor([0.628, 0.838])
    env.robot.cfg = SimpleNamespace(soft_joint_pos_limit_factor=1.0)
    return type(action), env, torch


@pytest.mark.parametrize("factor", [0.0, 0.95, 1.0])
def test_servo_recovers_native_continuous_joint_overflow(continuous_joint, factor):
    action_type, env, torch = continuous_joint
    original = env.robot.data.soft_joint_pos_limits.torch.clone()
    assert not torch.isfinite(original[:, 0]).any()
    env.robot.cfg.soft_joint_pos_limit_factor = factor

    action = action_type(None, env)

    assert torch.isfinite(action._limits).all()
    expected = env.robot.data.joint_pos_limits.torch[:, 0].double() * factor
    assert torch.equal(action._limits[:, 0], expected.float())
    assert torch.equal(action._limits[:, 1], original[:, 1])
    assert torch.equal(env.robot.data.soft_joint_pos_limits.torch, original)
    assert torch.equal(action._step_limit, torch.tensor([[0.628, 0.838]] * 2) * env.step_dt)


def test_continuous_joint_keeps_rotation_range_and_native_speed_cap(continuous_joint):
    action_type, env, torch = continuous_joint
    action = action_type(None, env)
    previous = action.target.clone()
    for _ in range(300):
        action.process_actions(torch.tensor([[100.0, -100.0]] * 2))
        assert torch.all((action.target - previous).abs() <= action._step_limit + 1e-6)
        previous = action.target.clone()
    assert (action.target[:, 0] > torch.pi).all()
    assert torch.equal(action.target[:, 1], torch.tensor([-2.0, -2.0]))


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), -0.1, 1.1])
def test_servo_recovery_rejects_invalid_native_soft_factor(continuous_joint, factor):
    action_type, env, _ = continuous_joint
    env.robot.cfg.soft_joint_pos_limit_factor = factor
    with pytest.raises(ValueError, match="Invalid native joint limits"):
        action_type(None, env)


@pytest.mark.parametrize("bounds", [(float("nan"), 1.0), (-float("inf"), 1.0), (2.0, 1.0)])
def test_servo_recovery_rejects_invalid_native_hard_limits(continuous_joint, bounds):
    action_type, env, torch = continuous_joint
    env.robot.data.joint_pos_limits.torch[:, 0] = torch.tensor(bounds)
    with pytest.raises(ValueError, match="Invalid native joint limits"):
        action_type(None, env)


@pytest.fixture
def reward(monkeypatch):
    torch = pytest.importorskip("torch")

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self._env = env

    monkeypatch.setitem(sys.modules, "isaaclab.managers", SimpleNamespace(ManagerTermBase=ManagerTermBase))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", SimpleNamespace(
        combine_frame_transforms=None, quat_apply_inverse=None, quat_inv=None, quat_mul=None))
    module = _native_module("franka_rl_learning_terms", monkeypatch)
    args = franka_rl._parser().parse_args(["prepare", "--run-id", "reward-test", "--learning-recipe",
                                         "adaptive", "--output-path", "s3://example/prepared/"])
    recipe = franka_rl._recipe(args)
    cfg = SimpleNamespace(params={"recipe": recipe, "training": True})
    env = SimpleNamespace(num_envs=1, device="cpu", reset_buf=torch.tensor([False]), extras={},
        action_manager=SimpleNamespace(action=torch.zeros(1, 8)), cfg=SimpleNamespace(
            events=SimpleNamespace(reset_object_position=SimpleNamespace(params={"pose_range": {
                "x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0, 0)}})),
            commands=SimpleNamespace(object_pose=SimpleNamespace(ranges=SimpleNamespace(
                pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5))))))
    instance = module.StableManipulationReward(cfg, env)
    values = [0.01, 0.01, 0.3, 0.01]
    monkeypatch.setattr(module, "_geometry", lambda env: tuple(torch.tensor([value]) for value in values))
    return instance, env, recipe, values, torch, module


def test_hold_reward_prefers_slow_continuous_holding_and_rejects_terminal_step(reward):
    instance, env, recipe, values, torch, _ = reward
    first = instance(env, recipe, True).item()
    for _ in range(18):
        instance(env, recipe, True)
    assert not instance.succeeded.item()
    env.reset_buf[:] = True
    instance(env, recipe, True)
    assert not instance.succeeded.item()
    env.reset_buf[:] = False
    for _ in range(20):
        stable = instance(env, recipe, True).item()
    assert instance.succeeded.item() and stable > first
    values[1] = 0.1
    moving = instance(env, recipe, True).item()
    assert moving < first and instance.streak.item() == 0
    instance.reset([0])
    assert not instance.succeeded.item()


def test_initial_settling_never_receives_a_lift_or_hold_bonus(reward):
    instance, env, recipe, values, torch, _ = reward
    values[:] = [0.01, 0.0, 0.06, 0.01]
    value = instance(env, recipe, True).item()
    reach_only = recipe["learning"]["reward"]["reach_weight"] * (1 - torch.tanh(torch.tensor(0.01 / 0.15)))
    assert value == pytest.approx(reach_only.item())
    assert not instance.succeeded.item()


def test_partial_goal_progress_does_not_grant_lift_hold_or_success(reward):
    instance, env, recipe, values, torch, _ = reward
    recipe["learning"] = learning_profile("adaptive-exploration")
    instance.settings = recipe["learning"]
    # Both positions remain below the unchanged lift threshold. Only distance
    # improves; no scripted action, closed-gripper condition, or success bonus.
    values[:] = [0.14, 0.0, 0.03, 0.01]
    low = instance(env, recipe, True).item()
    values[:] = [0.09, 0.0, 0.08, 0.01]
    for _ in range(25):
        progress = instance(env, recipe, True).item()
    assert progress > low
    settings = recipe["learning"]["reward"]
    expected = (settings["reach_weight"] * (1 - torch.tanh(torch.tensor(0.01 / 0.15)))
                + settings["goal_weight"] * (1 - torch.tanh(torch.tensor(0.09 / 0.1))))
    assert progress == pytest.approx(expected.item())
    assert instance.streak.item() == 0 and not instance.succeeded.item()
    assert instance.hold_fraction.item() == 0


def test_exploration_recipe_preserves_original_native_settings():
    from npa.workflows.franka_rl_learning import configure_learner, recipe_learning

    for name, std, entropy in (("adaptive", 0.5, 0.006), ("adaptive-exploration", 1.0, 0.02)):
        config = SimpleNamespace(actor=SimpleNamespace(distribution_cfg=SimpleNamespace()),
                                 critic=SimpleNamespace(), algorithm=SimpleNamespace(entropy_coef=0.006))
        recipe = {"learning": learning_profile(name)}
        assert recipe_learning(recipe) == recipe["learning"]
        configure_learner(config, recipe)
        assert config.actor.distribution_cfg.init_std == std
        assert config.algorithm.entropy_coef == entropy
    assert "goal_requires_lift" not in learning_profile("adaptive")["reward"]


def test_curriculum_ignores_initial_resets_and_stale_difficulty_episodes(reward):
    instance, env, recipe, values, torch, module = reward
    reset = env.cfg.events.reset_object_position
    env.event_manager = SimpleNamespace(get_term_cfg=lambda name: reset,
        set_term_cfg=lambda name, cfg: None)
    command = SimpleNamespace(cfg=env.cfg.commands.object_pose)
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    instance.curriculum = TrainingCurriculum(1, 1.0)
    assert module.adapt_training(env, [0]) == 0.25
    assert instance.completed_episodes == 0
    instance.reset([0])
    instance.succeeded[:] = True
    assert module.adapt_training(env, [0]) == 0.4
    # Another completion from the previous difficulty cannot promote again.
    assert module.adapt_training(env, [0]) == 0.4
    assert instance.curriculum.total_episodes == 1
    assert reset.params["pose_range"]["y"] == (-0.1, 0.1)
    instance.reset([0])
    assert env.extras["log"]["Learning/difficulty"] == 0.4
    instance.training = False
    with pytest.raises(ValueError, match="Evaluation must not adapt"):
        module.adapt_training(env, [0])


@pytest.mark.parametrize("training", [False, True])
def test_learning_configuration_keeps_full_reset_and_goal_ranges(monkeypatch, training):
    import builtins
    from copy import deepcopy
    from npa.workflows.franka_rl_learning import configure_learning

    monkeypatch.setitem(sys.modules, "isaaclab.managers", SimpleNamespace(
        CurriculumTermCfg=SimpleNamespace, ObservationTermCfg=SimpleNamespace, RewardTermCfg=SimpleNamespace))
    class ResolvableString(str):
        def __call__(self, *args, **kwargs):
            raise AssertionError("Native terms must not be resolved before simulator startup")

    monkeypatch.setitem(sys.modules, "isaaclab.utils.string", SimpleNamespace(ResolvableString=ResolvableString))
    original_import = builtins.__import__
    def before_simulator_import(name, *args, **kwargs):
        if name.startswith(("pxr", "npa.workflows.franka_rl_servo", "npa.workflows.franka_rl_learning_terms")):
            raise AssertionError("Runtime term imported before simulator startup")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", before_simulator_import)
    config = SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace()),
        observations=SimpleNamespace(policy=SimpleNamespace()),
        events={"reset_object": {"x": (-0.1, 0.1)}}, commands={"goal_z": (0.25, 0.5)})
    original = deepcopy((config.events, config.commands))
    configure_learning(config, {"learning": learning_profile("adaptive")}, training=training)
    assert (config.events, config.commands) == original
    assert bool(config.curriculum) == training
    assert config.rewards["manipulation"].params["training"] == training
    assert callable(config.actions.arm_action.class_type)
    assert isinstance(config.actions.arm_action.class_type, ResolvableString)
    assert config.observations.policy.manipulation.func.endswith(":manipulation_state")
    assert config.rewards["manipulation"].func.endswith(":StableManipulationReward")


def test_normalization_receipt_detects_changed_inference_statistics():
    torch = pytest.importorskip("torch")
    from npa.workflows.franka_rl_learning import normalization_evidence

    state = {"_mean": torch.zeros(1, 3), "_var": torch.ones(1, 3),
             "_std": torch.ones(1, 3), "count": torch.tensor(100)}
    policy = SimpleNamespace(obs_normalizer=SimpleNamespace(state_dict=lambda: state))
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: policy))
    recipe = {"learning": learning_profile("adaptive")}
    before = normalization_evidence(runner, recipe)
    assert before["sample_count"] == 100
    assert before == normalization_evidence(runner, recipe)
    state["_mean"][0, 1] = 0.01
    assert before != normalization_evidence(runner, recipe)
    state["_std"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="normalization buffers"):
        normalization_evidence(runner, recipe)
    assert normalization_evidence(None, {}) == {}
