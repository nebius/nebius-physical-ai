"""Check that goal completion beats occupancy without rewarding physical failures."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from npa.workflows.navigation import reference


@pytest.fixture
def arrival_task(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.envs.mdp.commands.pose_2d_command",
        SimpleNamespace(UniformPose2dCommand=object),
    )
    path = Path(reference.__file__).with_name("reference_mdp.py")
    spec = importlib.util.spec_from_file_location("isolated_reference_arrival", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    positions = torch.zeros((6, 3))
    positions[:, 0] = torch.tensor([0.49, 0.5, 0.51, 0.1, 0.1, 0.1])
    env = SimpleNamespace(
        num_envs=6,
        device="cpu",
        npa_probe=False,
        cfg=SimpleNamespace(npa_training=True),
        npa_goals=torch.zeros((6, 2)),
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=positions))
            )
        },
        npa_contacts=SimpleNamespace(
            obstacle=torch.tensor([0, 0, 0, 1, 0, 0]),
            peer=torch.tensor([0, 0, 0, 0, 1, 0]),
        ),
    )
    monkeypatch.setattr(
        module, "physical_failure", lambda _: torch.tensor([0, 0, 0, 0, 0, 1])
    )
    return module, env


def test_arrival_excludes_boundary_collision_peer_and_physical_failure(arrival_task):
    module, env = arrival_task
    assert module.arrival(env).tolist() == [1, 0, 0, 0, 0, 0]
    assert module.terminate(env).tolist() == [True, False, False, True, True, True]


def test_arrival_reward_remains_available_between_termination_and_reset(arrival_task):
    module, env = arrival_task
    # Native ManagerBasedRLEnv computes terminations, then rewards, then resets.
    assert module.terminate(env)[0].item() is True
    assert (module.arrival(env)[0] * 200.0 * 0.2).item() == 40.0
    env.scene["robot"].data.root_pos_w.torch[0, 0] = 4.0
    assert module.arrival(env)[0].item() == 0.0
    assert module.terminate(env)[0].item() is False


@pytest.mark.parametrize("probe,training", [(True, True), (False, False)])
def test_nonterminal_probe_and_evaluation_cannot_repeat_arrival_reward(
    arrival_task, probe, training
):
    module, env = arrival_task
    env.npa_probe, env.cfg.npa_training = probe, training
    assert module.arrival(env).tolist() == [0] * env.num_envs
    assert not module.terminate(env).any()
