"""Verify fresh Cartesian actors hold a measured pose without replacing checkpoints."""

from types import SimpleNamespace

import pytest

from npa.workflows.sim2real import isaac_franka_initialization as initialization
from npa.workflows.sim2real.isaac_franka_policy import SCENARIO_TASK_ID

torch = pytest.importorskip("torch")


@pytest.fixture
def fresh_actor():
    return SimpleNamespace(
        is_recurrent=True,
        mlp=torch.nn.Sequential(torch.nn.Linear(5, 16), torch.nn.Linear(16, 8)),
    )


@pytest.fixture
def native_ready_task(monkeypatch):
    pose = torch.tensor([[0.45, -0.03, 0.52, 0.0, 1.0, 0.0, 0.0]]).repeat(4, 1)
    scale = torch.tensor([[0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0]])
    task = SimpleNamespace(
        spec=SimpleNamespace(id=SCENARIO_TASK_ID),
        action_manager=SimpleNamespace(
            get_term=lambda name: SimpleNamespace(_scale=scale)
        ),
    )
    monkeypatch.setattr(initialization, "_root_tcp_pose", lambda actual: pose)
    return task, pose, scale


def test_initial_policy_commands_measured_ready_pose(fresh_actor, native_ready_task):
    task, pose, scale = native_ready_task
    audit = initialization.initialize_fresh_stock_franka_actor(
        fresh_actor, task, checkpoint_loaded=False
    )
    output = fresh_actor.mlp(torch.randn(4, 5))
    torch.testing.assert_close(output[:, :7] * scale, pose)
    assert output[:, -1].tolist() == [1.0] * 4
    assert audit["task_competence_established"] is False
    assert audit["inference_composition"] == "learned_actor_only"
    assert audit["maximum_ready_pose_position_spread_m"] == 0


def test_initialized_actor_remains_trainable(fresh_actor, native_ready_task):
    task, _, _ = native_ready_task
    initialization.initialize_fresh_stock_franka_actor(
        fresh_actor, task, checkpoint_loaded=False
    )
    before = fresh_actor.mlp[-1].weight.detach().clone()
    optimizer = torch.optim.Adam(fresh_actor.mlp.parameters(), lr=0.001)
    output = fresh_actor.mlp(torch.ones(4, 5))
    output.square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, fresh_actor.mlp[-1].weight)


def test_loaded_checkpoint_is_never_modified(fresh_actor, native_ready_task):
    task, _, _ = native_ready_task
    before = {
        name: value.clone() for name, value in fresh_actor.mlp.state_dict().items()
    }
    with pytest.raises(RuntimeError, match="loaded checkpoint"):
        initialization.initialize_fresh_stock_franka_actor(
            fresh_actor, task, checkpoint_loaded=True
        )
    for name, value in fresh_actor.mlp.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_reinitialization_cannot_overwrite_learning(fresh_actor, native_ready_task):
    task, _, _ = native_ready_task
    initialization.initialize_fresh_stock_franka_actor(
        fresh_actor, task, checkpoint_loaded=False
    )
    with pytest.raises(RuntimeError, match="already initialized"):
        initialization.initialize_fresh_stock_franka_actor(
            fresh_actor, task, checkpoint_loaded=False
        )


def test_quaternion_double_cover_keeps_one_ready_orientation(native_ready_task):
    _, pose, scale = native_ready_task
    pose[1:, 3:] *= -2
    action, spread = initialization._canonical_ready_action(pose, scale)
    torch.testing.assert_close(action[3:7], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert spread == 0


@pytest.mark.parametrize("kind", ["position", "orientation"])
def test_incompatible_vectorized_ready_poses_fail(native_ready_task, kind):
    _, pose, scale = native_ready_task
    if kind == "position":
        pose[1, 0] += 0.01
    else:
        pose[1, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="common robot-frame ready pose"):
        initialization._canonical_ready_action(pose, scale)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -0.1])
def test_invalid_action_scales_fail(native_ready_task, value):
    _, pose, scale = native_ready_task
    scale[0, 0] = value
    with pytest.raises(ValueError, match="action scales"):
        initialization._canonical_ready_action(pose, scale)


def test_per_environment_scaling_cannot_move_other_ready_poses(native_ready_task):
    _, pose, scale = native_ready_task
    scale = scale.repeat(4, 1)
    scale[1, 0] = 0.2
    with pytest.raises(ValueError, match="scaling differs"):
        initialization._canonical_ready_action(pose, scale)


def test_zero_quaternion_is_rejected(native_ready_task):
    _, pose, scale = native_ready_task
    pose[2, 3:7] = 0
    with pytest.raises(ValueError, match="orientation must be nonzero"):
        initialization._canonical_ready_action(pose, scale)


def test_unrelated_task_cannot_be_reinitialized(fresh_actor, native_ready_task):
    task, _, _ = native_ready_task
    task.spec.id = "unrelated-task"
    with pytest.raises(ValueError, match="stock scenario task"):
        initialization.initialize_fresh_stock_franka_actor(
            fresh_actor, task, checkpoint_loaded=False
        )
