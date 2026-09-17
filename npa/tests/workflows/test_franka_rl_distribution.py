"""Verify bounded exploration against Normal mathematics and the real native PPO API."""

from copy import deepcopy
import io
import math
from types import SimpleNamespace

import pytest

from npa.workflows.franka_rl_learning import configure_learner, distribution_evidence, learning_profile, recipe_learning


@pytest.fixture
def native():
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl.modules.distribution")
    from npa.workflows.franka_rl_distribution import BoundedGaussianDistribution

    return torch, BoundedGaussianDistribution


def _learner_config():
    return SimpleNamespace(actor=SimpleNamespace(distribution_cfg=SimpleNamespace()),
                           critic=SimpleNamespace(), algorithm=SimpleNamespace())


def test_bounded_profile_preserves_rewards_and_only_changes_exploration_distribution():
    original = learning_profile("adaptive-exploration")
    bounded = learning_profile("adaptive-bounded-exploration")
    definition = bounded["ppo"].pop("distribution")
    bounded["name"] = original["name"]
    assert bounded == original
    assert definition == {
        "class_name": "npa.workflows.franka_rl_distribution:BoundedGaussianDistribution",
        "parameterization": "bounded_sigmoid", "scale_semantics": "raw_action_standard_deviation",
        "min_std": 0.05, "max_std": 1.5,
    }
    altered = {"learning": learning_profile("adaptive-bounded-exploration")}
    altered["learning"]["ppo"]["distribution"]["max_std"] = 2.0
    with pytest.raises(ValueError, match="Sealed learning"):
        recipe_learning(altered)


def test_bounded_profile_configures_native_distribution_constructor_keywords():
    config = _learner_config()
    configure_learner(config, {"learning": learning_profile("adaptive-bounded-exploration")})
    assert config.actor.distribution_cfg == {
        "class_name": "npa.workflows.franka_rl_distribution:BoundedGaussianDistribution",
        "init_std": 1.0, "min_std": 0.05, "max_std": 1.5,
    }
    assert config.actor.obs_normalization and config.critic.obs_normalization
    assert config.algorithm.gamma == 0.99
    assert config.algorithm.entropy_coef == 0.02


@pytest.mark.parametrize("kwargs", [
    {"output_dim": 0}, {"output_dim": True}, {"min_std": 0}, {"min_std": -1},
    {"init_std": 0.05}, {"init_std": 1.5}, {"max_std": 0.5},
    {"init_std": math.nan}, {"min_std": math.inf}, {"max_std": math.inf},
])
def test_invalid_distribution_parameters_fail_before_sampling(native, kwargs):
    _, distribution = native
    with pytest.raises(ValueError):
        distribution(**{"output_dim": 3, **kwargs})


def test_every_probability_operation_uses_the_actual_bounded_normal(native):
    torch, distribution = native
    policy = distribution(3).double()
    mean = torch.tensor([[4.0, -3.0, 0.2], [-0.4, 0.0, 7.0]], dtype=torch.float64)
    policy.raw_std.data.copy_(torch.tensor([-2.0, 0.0, 2.0]))
    bounds = policy.std_bounds.tolist()
    scales = torch.tensor([bounds[0] + (bounds[1] - bounds[0]) / (1 + math.exp(-raw))
                           for raw in [-2.0, 0.0, 2.0]], dtype=torch.float64)
    reference = torch.distributions.Normal(mean, scales)
    policy.update(mean)
    with torch.random.fork_rng():
        torch.manual_seed(182)
        expected_sample = reference.sample()
        torch.manual_seed(182)
        torch.testing.assert_close(policy.sample(), expected_sample)
    torch.testing.assert_close(policy.params[0], mean)
    torch.testing.assert_close(policy.params[1], scales.expand_as(mean))
    torch.testing.assert_close(policy.log_prob(expected_sample), reference.log_prob(expected_sample).sum(-1))
    torch.testing.assert_close(policy.entropy, reference.entropy().sum(-1))
    assert torch.equal(policy.deterministic_output(mean), mean)
    expected_entropy = (scales.log() + 0.5 * math.log(2 * math.pi * math.e)).sum()
    torch.testing.assert_close(policy.entropy, expected_entropy.expand(mean.shape[0]))


def test_kl_matches_independent_diagonal_normal_formula(native):
    torch, distribution = native
    policy = distribution(2)
    old_mean, new_mean = torch.tensor([[1.0, -3.0]]), torch.tensor([[0.0, 2.0]])
    old_scale, new_scale = torch.tensor([[0.2, 1.2]]), torch.tensor([[1.4, 0.08]])
    expected = ((new_scale / old_scale).log()
                + (old_scale.square() + (old_mean - new_mean).square()) / (2 * new_scale.square()) - 0.5).sum(-1)
    actual = policy.kl_divergence((old_mean, old_scale), (new_mean, new_scale))
    torch.testing.assert_close(actual, expected)
    assert policy.kl_divergence((old_mean, old_scale), (old_mean, old_scale)).item() == 0


@pytest.mark.parametrize("direction", [-1, 1])
def test_entropy_optimization_cannot_cross_bounds_and_retains_scale_gradients(native, direction):
    torch, distribution = native
    policy = distribution(3)
    assert torch.equal(policy.learned_std, torch.ones(3))
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0)
    for _ in range(200):
        optimizer.zero_grad()
        policy.update(torch.zeros(4, 3))
        loss = direction * policy.entropy.mean()
        loss.backward()
        assert torch.isfinite(policy.raw_std.grad).all()
        assert (policy.raw_std.grad.abs() > 0).all()
        optimizer.step()
        assert (policy.learned_std >= 0.05).all() and (policy.learned_std <= 1.5).all()
    if direction == 1:
        assert (policy.learned_std < 1).all()
    else:
        assert (policy.learned_std > 1).all()
    policy.raw_std.data.copy_(torch.tensor([-1000.0, 0.0, 1000.0]))
    assert policy.learned_std[0] == policy.std_bounds[0]
    assert policy.learned_std[2] == policy.std_bounds[1]


def test_checkpoint_persists_bounds_raw_scale_and_exact_probability_semantics(native):
    torch, distribution = native
    original = distribution(3)
    original.raw_std.data.copy_(torch.tensor([-4.0, 0.5, 3.0]))
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    restored = distribution(3, min_std=0.1, max_std=2.0)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert set(restored.state_dict()) == {"raw_std", "std_bounds"}
    mean = torch.tensor([[4.0, -3.0, 2.0]])
    for policy in (original, restored):
        policy.update(mean)
    assert torch.equal(original.learned_std, restored.learned_std)
    assert torch.equal(original.log_prob(mean), restored.log_prob(mean))
    assert torch.equal(original.entropy, restored.entropy)


def test_evidence_reads_current_scale_and_rejects_changed_checkpoint_bounds(native):
    torch, distribution = native
    policy = distribution(3)
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: SimpleNamespace(distribution=policy)))
    recipe = {"learning": learning_profile("adaptive-bounded-exploration")}
    evidence = distribution_evidence(runner, recipe)
    assert evidence["learned_std"] == [1.0] * 3
    policy.raw_std.data.add_(0.5)
    assert distribution_evidence(runner, recipe)["learned_std"] != evidence["learned_std"]
    policy.std_bounds[1] = 3
    with pytest.raises(ValueError, match="violate the sealed recipe"):
        distribution_evidence(runner, recipe)
    policy.std_bounds[1] = 1.5
    policy.raw_std.data[0] = torch.nan
    with pytest.raises(ValueError, match="violate the sealed recipe"):
        distribution_evidence(runner, recipe)


def _native_ppo(torch):
    from rsl_rl.algorithms import PPO
    from tensordict import TensorDict

    settings = _learner_config()
    configure_learner(settings, {"learning": learning_profile("adaptive-bounded-exploration")})
    model = {"class_name": "MLPModel", "hidden_dims": [8], "activation": "elu", "obs_normalization": True}
    # Native PPO 5.0.1 does not checkpoint its separate adaptive learning-rate
    # scalar. A fixed rate isolates distribution/optimizer save-load equivalence.
    config = {"actor": {**model, "distribution_cfg": deepcopy(settings.actor.distribution_cfg)},
              "critic": dict(model), "algorithm": {"class_name": "PPO", "num_learning_epochs": 1,
                  "num_mini_batches": 1, "entropy_coef": settings.algorithm.entropy_coef,
                  "gamma": settings.algorithm.gamma, "schedule": "fixed"},
              "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
              "num_steps_per_env": 4, "multi_gpu": None}
    obs = TensorDict({"policy": torch.arange(20).reshape(4, 5).float() / 20}, batch_size=[4])
    algorithm = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=4, num_actions=3), config, "cpu")
    return algorithm, obs


def _ppo_update(torch, algorithm, obs):
    with torch.inference_mode():
        for _ in range(4):
            actions = algorithm.act(obs)
            algorithm.process_env_step(obs, -actions.square().sum(-1), torch.zeros(4), {})
        algorithm.compute_returns(obs)
    losses = algorithm.update()
    assert all(math.isfinite(value) for value in losses.values())


def test_native_ppo_update_save_load_and_deterministic_exports(native, tmp_path):
    torch, distribution = native
    algorithm, obs = _native_ppo(torch)
    assert isinstance(algorithm.actor.distribution, distribution)
    before = algorithm.actor.distribution.raw_std.detach().clone()
    _ppo_update(torch, algorithm, obs)
    assert not torch.equal(before, algorithm.actor.distribution.raw_std)
    checkpoint = io.BytesIO()
    torch.save(algorithm.save(), checkpoint)
    checkpoint.seek(0)
    restored, _ = _native_ppo(torch)
    assert restored.load(torch.load(checkpoint, weights_only=True), load_cfg=None, strict=True)
    algorithm.eval_mode()
    restored.eval_mode()
    expected = algorithm.actor(obs)
    torch.testing.assert_close(restored.actor(obs), expected, rtol=0, atol=0)
    assert restored.optimizer.state_dict()["state"]
    scripted = torch.jit.script(restored.actor.as_jit())
    scripted.save(str(tmp_path / "policy.pt"))
    torch.testing.assert_close(torch.jit.load(str(tmp_path / "policy.pt"))(obs["policy"]), expected)
    torch.testing.assert_close(restored.actor.as_onnx(verbose=False)(obs["policy"]), expected)
    with torch.random.fork_rng():
        torch.manual_seed(724)
        _ppo_update(torch, algorithm, obs)
        torch.manual_seed(724)
        _ppo_update(torch, restored, obs)
    for key, value in algorithm.actor.state_dict().items():
        torch.testing.assert_close(restored.actor.state_dict()[key], value, rtol=0, atol=0)


def test_native_onnx_artifact_keeps_the_unrestricted_deterministic_mean(native, tmp_path):
    torch, _ = native
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    algorithm, obs = _native_ppo(torch)
    algorithm.eval_mode()
    algorithm.actor.mlp[-1].bias.data.fill_(4.0)
    expected = algorithm.actor(obs).detach()
    assert (expected > 1.5).all()
    path = tmp_path / "policy.onnx"
    torch.onnx.export(algorithm.actor.as_onnx(verbose=False).eval(), (obs["policy"],), str(path),
                      input_names=["obs"], output_names=["actions"], opset_version=18)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(["actions"], {"obs": obs["policy"].numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(output), expected, rtol=1e-5, atol=1e-6)
