"""Exercise real RSL-RL learning, checkpoint loading, normalization, and ONNX export."""

from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from npa.genesis.generate_demos import _load_teacher_policy  # noqa: E402
from npa.genesis.train_teacher import GenesisEnvWrapper, PPOConfig  # noqa: E402
from npa.workflows.sim2real.policy_export import export_policy_onnx  # noqa: E402


@dataclass
class _Config:
    max_episode_steps: int = 8


class _Dynamics:
    """Small deterministic tensor dynamics; this does not claim Genesis GPU physics."""

    n_envs = 4
    obs_dim = 6
    act_dim = 2
    device = "cpu"
    cfg = _Config()

    def __init__(self):
        self.state = torch.zeros(self.n_envs, self.obs_dim)
        self.steps = 0

    def get_privileged_obs(self):
        return {"flat": self.state.clone()}

    def reset(self):
        self.state.zero_()
        return self.get_privileged_obs()

    def step(self, actions):
        self.state[:, :2] += actions * 0.1
        self.steps += 1
        dones = torch.full((self.n_envs,), self.steps % 8 == 0, dtype=torch.bool)
        rewards = -self.state.square().sum(dim=1)
        return self.get_privileged_obs(), rewards, dones, {}


@pytest.mark.parametrize("activation", ["elu", "swish", "identity"])
@pytest.mark.parametrize("normalize", [False, True])
def test_real_ppo_train_save_load_and_export_preserve_deterministic_actions(
    tmp_path, normalize, activation
):
    from rsl_rl.runners import OnPolicyRunner

    torch.manual_seed(17)
    env = _Dynamics()
    config = PPOConfig(
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        num_steps_per_env=4,
        num_learning_epochs=1,
        num_mini_batches=2,
        empirical_normalization=normalize,
        activation=activation,
    )
    cfg = config.to_train_cfg()
    architecture = {
        "actor": cfg["actor"],
        "num_obs": env.obs_dim,
        "num_actions": env.act_dim,
    }
    (tmp_path / "arch_config.json").write_text(json.dumps(architecture))
    runner = OnPolicyRunner(
        GenesisEnvWrapper(env), cfg, str(tmp_path / "logs"), device="cpu"
    )
    runner.learn(2)
    checkpoint = tmp_path / "model.pt"
    runner.save(str(checkpoint))
    observations = torch.randn(env.n_envs, env.obs_dim)
    from tensordict import TensorDict

    expected = runner.get_inference_policy()(
        TensorDict({"policy": observations}, batch_size=[env.n_envs])
    )
    teacher = _load_teacher_policy(checkpoint, env)
    torch.testing.assert_close(teacher.act_inference(observations), expected)
    _assert_export(
        checkpoint, tmp_path / "export", observations, expected, normalize, activation
    )


def _assert_export(checkpoint, output, observations, expected, normalize, activation):
    runtime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    result = export_policy_onnx(checkpoint, out_dir=str(output), activation=activation)
    session = runtime.InferenceSession(
        result["onnx_path"], providers=["CPUExecutionProvider"]
    )
    observed = session.run(None, {"obs": observations.numpy()})[0]
    np.testing.assert_allclose(
        observed, expected.detach().numpy(), rtol=1e-5, atol=1e-6
    )
    contract = json.loads((output / "policy_contract.json").read_text())
    assert contract["network"]["framework"] == "rsl_rl.models.MLPModel"
    if normalize:
        assert contract["normalization"]["eps"] == 1e-2


def test_legacy_actor_checkpoint_remains_loadable_and_rejects_wrong_observation_shape(
    tmp_path,
):
    actor = torch.nn.Sequential(
        torch.nn.Linear(6, 8), torch.nn.ELU(), torch.nn.Linear(8, 2)
    )
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": {
                "actor." + key: value for key, value in actor.state_dict().items()
            }
        },
        checkpoint,
    )
    env = SimpleNamespace(obs_dim=6, act_dim=2, device="cpu")
    loaded = _load_teacher_policy(checkpoint, env)
    observations = torch.randn(4, 6)
    torch.testing.assert_close(loaded.act_inference(observations), actor(observations))
    with pytest.raises(ValueError, match="observation shape"):
        loaded.act_inference(torch.zeros(4, 5))


def test_new_checkpoint_rejects_partial_normalization_and_unsupported_networks(
    tmp_path,
):
    checkpoint = tmp_path / "invalid.pt"
    weights = {"mlp.0.weight": torch.ones(2, 6), "mlp.0.bias": torch.zeros(2)}
    env = SimpleNamespace(obs_dim=6, act_dim=2, device="cpu")
    for extra, message in (
        ({"obs_normalizer._mean": torch.zeros(6)}, "incomplete"),
        ({"cnn.weight": torch.ones(2)}, "plain RSL-RL"),
    ):
        torch.save({"actor_state_dict": {**weights, **extra}}, checkpoint)
        with pytest.raises(ValueError, match=message):
            _load_teacher_policy(checkpoint, env)
