"""Verify retention against actual RSL-RL 5.0.1 tensors, optimizer state and RNG."""

from copy import deepcopy
from importlib.metadata import version
from types import SimpleNamespace

import pytest


@pytest.fixture
def native():
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    if version("rsl-rl-lib") != "5.0.1":
        pytest.skip("native anchor equivalence requires exact RSL-RL 5.0.1")
    from npa.workflows.navigation.anchor_state import state_digest

    return torch, state_digest


def _algorithm(torch, anchor=None, *, normalization=True, batches=2, schedule="fixed"):
    from rsl_rl.algorithms import PPO
    from tensordict import TensorDict

    model = dict(
        class_name="MLPModel",
        hidden_dims=[8],
        activation="elu",
        obs_normalization=normalization,
    )
    algorithm = dict(
        class_name="PPO",
        num_learning_epochs=2,
        num_mini_batches=batches,
        schedule=schedule,
    )
    if anchor is not None:
        algorithm.update(
            class_name="npa.workflows.navigation.anchor_ppo:BaselineAnchoredPPO",
            baseline_anchor=anchor,
        )
    config = dict(
        actor={
            **model,
            "distribution_cfg": dict(class_name="GaussianDistribution", init_std=0.5),
        },
        critic=dict(model),
        algorithm=algorithm,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        num_steps_per_env=4,
        multi_gpu=None,
    )
    obs = TensorDict(
        {"policy": torch.arange(20).reshape(4, 5).float() / 20}, batch_size=[4]
    )
    learner = PPO.construct_algorithm(
        obs, SimpleNamespace(num_envs=4, num_actions=3), config, "cpu"
    )
    return learner, obs


def _advance(torch, algorithm, obs):
    with torch.inference_mode():
        for _ in range(4):
            actions = algorithm.act(obs)
            algorithm.process_env_step(
                obs, -actions.square().sum(-1), torch.zeros(4), {}
            )
        algorithm.compute_returns(obs)
    return algorithm.update()


@pytest.fixture
def baseline(native, tmp_path, request):
    from npa.workflows.navigation.artifacts import file_sha256

    torch, _ = native
    torch.manual_seed(91)
    algorithm, obs = _algorithm(torch, normalization=getattr(request, "param", True))
    _advance(torch, algorithm, obs)
    payload = deepcopy(algorithm.save())
    payload.update(iter=37, infos=None)
    path = tmp_path / "original.pt"
    torch.save(payload, path)
    anchor = dict(
        coefficient=10.0, checkpoint=dict(file=path.name, sha256=file_sha256(path))
    )
    return payload, anchor, obs


def _before_binding(native, baseline, tmp_path, *, coefficient=10.0):
    from npa.workflows.navigation.contract import BaselineAnchor, InitialCheckpoint
    from npa.workflows.navigation.learning_evidence import _save_initial_checkpoint

    torch, _ = native
    payload, anchor, obs = baseline
    anchor = {**anchor, "coefficient": coefficient}
    normalization = "obs_normalizer.count" in payload["actor_state_dict"]
    algorithm, _ = _algorithm(torch, anchor, normalization=normalization)
    algorithm.load(deepcopy(payload), None, True)
    runner = SimpleNamespace(
        alg=algorithm, current_learning_iteration=37, cfg={"num_steps_per_env": 4}
    )
    recipe = SimpleNamespace(
        baseline_anchor=BaselineAnchor(**anchor),
        initial_checkpoint=InitialCheckpoint(**anchor["checkpoint"]),
        iterations=1,
        num_envs=4,
    )
    reference = tmp_path / "reference.pt"
    _save_initial_checkpoint(runner, reference)
    return algorithm, obs, runner, recipe


def _bound(native, baseline, tmp_path, *, coefficient=10.0):
    from npa.workflows.navigation.anchor_runtime import bind_after_snapshot

    result = _before_binding(native, baseline, tmp_path, coefficient=coefficient)
    _, _, runner, recipe = result
    bind_after_snapshot(runner, recipe, tmp_path / "reference.pt", tmp_path)
    return result


def test_binding_preserves_exact_native_snapshot_and_rng(native, baseline, tmp_path):
    from npa.workflows.navigation.anchor_runtime import bind_after_snapshot
    from npa.workflows.navigation.anchor_state import native_state
    from npa.workflows.navigation.contract import BaselineAnchor, InitialCheckpoint
    from npa.workflows.navigation.learning_evidence import _save_initial_checkpoint

    torch, digest = native
    payload, anchor, _ = baseline
    algorithm, _ = _algorithm(torch, anchor)
    algorithm.load(deepcopy(payload), None, True)
    runner = SimpleNamespace(alg=algorithm, current_learning_iteration=37)
    recipe = SimpleNamespace(
        baseline_anchor=BaselineAnchor(**anchor),
        initial_checkpoint=InitialCheckpoint(**anchor["checkpoint"]),
    )
    reference = tmp_path / "reference.pt"
    assert algorithm.anchor_teacher is None
    _save_initial_checkpoint(runner, reference)
    assert digest(torch.load(reference, weights_only=True)) == digest(payload)
    before, random = digest(algorithm.save()), torch.get_rng_state().clone()
    bind_after_snapshot(runner, recipe, reference, tmp_path)
    assert torch.equal(torch.get_rng_state(), random)
    assert digest(native_state(algorithm.save())) == before
    assert digest(algorithm.anchor_teacher.state_dict()) == digest(
        payload["actor_state_dict"]
    )
    optimizer = {
        id(parameter)
        for group in algorithm.optimizer.param_groups
        for parameter in group["params"]
    }
    for student, teacher in zip(
        algorithm.actor.parameters(), algorithm.anchor_teacher.parameters(), strict=True
    ):
        assert student.data_ptr() != teacher.data_ptr() and id(teacher) not in optimizer
        assert not teacher.requires_grad


def test_native_none_and_zero_coefficient_are_bitwise_equivalent(native, baseline):
    torch, digest = native
    payload, anchor, obs = baseline
    plain, _ = _algorithm(torch)
    disabled, _ = _algorithm(torch, {**anchor, "coefficient": 0.0})
    plain.load(deepcopy(payload), None, True)
    disabled.load(deepcopy(payload), None, True)
    torch.manual_seed(724)
    expected = _advance(torch, plain, obs)
    random = torch.get_rng_state().clone()
    torch.manual_seed(724)
    actual = _advance(torch, disabled, obs)
    assert torch.equal(torch.get_rng_state(), random)
    assert actual == expected and digest(disabled.save()) == digest(plain.save())
    assert disabled.anchor_teacher is None


@pytest.mark.parametrize("schedule", ["fixed", "adaptive"])
def test_refactored_update_matches_native_with_zero_auxiliary_loss(
    native, baseline, tmp_path, monkeypatch, schedule
):
    from npa.workflows.navigation.anchor_state import native_state

    torch, digest = native
    payload, _, obs = baseline
    plain, _ = _algorithm(torch, schedule=schedule)
    plain.load(deepcopy(payload), None, True)
    anchored, _, _, _ = _bound(native, baseline, tmp_path)
    anchored.schedule = schedule
    monkeypatch.setattr(
        anchored, "baseline_loss", lambda observations, params: params[0].sum() * 0
    )
    torch.manual_seed(724)
    expected = _advance(torch, plain, obs)
    random = torch.get_rng_state().clone()
    torch.manual_seed(724)
    actual = _advance(torch, anchored, obs)
    assert torch.equal(torch.get_rng_state(), random)
    assert {key: actual[key] for key in expected} == expected
    assert digest(native_state(anchored.save())) == digest(plain.save())
    assert anchored.learning_rate == plain.learning_rate


def test_kl_math_gradients_and_frozen_teacher(native, baseline, tmp_path):
    torch, digest = native
    algorithm, obs, _, _ = _bound(native, baseline, tmp_path)
    teacher = digest(algorithm.anchor_teacher.state_dict())
    algorithm.actor(obs, stochastic_output=True)
    assert (
        algorithm.baseline_loss(obs, algorithm.actor.output_distribution_params).item()
        == 0
    )
    with torch.no_grad():
        algorithm.actor.mlp[-1].bias.add_(0.15)
        algorithm.actor.distribution.std_param.add_(0.1)
    algorithm.actor(obs, stochastic_output=True)
    current = algorithm.actor.output_distribution_params
    penalty = algorithm.baseline_loss(obs, current)
    mean, std = algorithm.anchor_teacher.output_distribution_params
    expected = _gaussian_kl(torch, (mean, std), current)
    torch.testing.assert_close(penalty, expected)
    optimizer = torch.optim.SGD(algorithm.actor.parameters(), lr=1e-3)
    optimizer.zero_grad()
    penalty.backward()
    assert algorithm.actor.distribution.std_param.grad.abs().sum() > 0
    assert all(
        parameter.grad is None for parameter in algorithm.anchor_teacher.parameters()
    )
    optimizer.step()
    algorithm.actor(obs, stochastic_output=True)
    assert (
        algorithm.baseline_loss(obs, algorithm.actor.output_distribution_params)
        < penalty
    )
    assert digest(algorithm.anchor_teacher.state_dict()) == teacher


def _gaussian_kl(torch, previous, current):
    mean, std = previous
    components = (
        torch.log(current[1] / std)
        + (std.square() + (mean - current[0]).square()) / (2 * current[1].square())
        - 0.5
    )
    return components.sum(-1).mean()


def test_real_update_counts_all_samples_without_teacher_normalization_or_rng(
    native, baseline, tmp_path
):
    from npa.workflows.navigation.anchor_runtime import anchor_evidence

    torch, digest = native
    algorithm, obs, runner, recipe = _bound(native, baseline, tmp_path)
    teacher = digest(algorithm.anchor_teacher.state_dict())
    teacher_count = algorithm.anchor_teacher.obs_normalizer.count.clone()
    algorithm.train_mode()
    algorithm.actor(obs, stochastic_output=True)
    random = torch.get_rng_state().clone()
    algorithm.baseline_loss(obs, algorithm.actor.output_distribution_params)
    assert torch.equal(random, torch.get_rng_state())
    losses = _advance(torch, algorithm, obs * 2)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert digest(algorithm.anchor_teacher.state_dict()) == teacher
    assert torch.equal(teacher_count, algorithm.anchor_teacher.obs_normalizer.count)
    assert algorithm.actor.obs_normalizer.count > teacher_count
    evidence = anchor_evidence(runner, recipe, tmp_path)
    assert evidence["optimizer_steps_this_run"] == 4
    assert evidence["samples_this_run"] == 32
    assert "teacher_state_dict" not in evidence


@pytest.mark.parametrize("schedule", ["fixed", "adaptive"])
def test_resume_keeps_original_teacher_and_exact_next_update(
    native, baseline, tmp_path, schedule
):
    torch, digest = native
    original, anchor, obs = baseline
    algorithm, _, _, _ = _bound(native, baseline, tmp_path)
    algorithm.schedule = schedule
    _advance(torch, algorithm, obs)
    saved = deepcopy(algorithm.save())
    restored, _ = _algorithm(torch, anchor, schedule=schedule)
    restored.load(saved, None, True)
    assert restored.learning_rate == algorithm.learning_rate
    if schedule == "adaptive":
        assert algorithm.learning_rate != 0.001
    restored.bind_baseline(original, "c" * 64)
    assert digest(restored.anchor_teacher.state_dict()) == digest(
        original["actor_state_dict"]
    )
    torch.manual_seed(451)
    _advance(torch, algorithm, obs)
    torch.manual_seed(451)
    _advance(torch, restored, obs)
    assert digest(restored.save()) == digest(algorithm.save())


def test_inference_needs_only_candidate_checkpoint_and_no_teacher(
    native, baseline, tmp_path, raw_bundle, recipe_dict, monkeypatch
):
    from npa.workflows.field_failure.native_policy import _initialize
    from npa.workflows.navigation.anchor_config import configure_training_anchor
    from npa.workflows.navigation.contract import read_recipe

    torch, _ = native
    algorithm, obs, _, _ = _bound(native, baseline, tmp_path)
    _advance(torch, algorithm, obs)
    path = tmp_path / "candidate.pt"
    torch.save({**algorithm.save(), "iter": 41, "infos": None}, path)
    (tmp_path / "original.pt").unlink()
    protocol = {
        key: recipe_dict.get(key)
        for key in ("task", "adapter_module", "adapter_sha256", "source_bundle_sha256")
    }
    protocol.update(
        navigation_image=recipe_dict["image"], baseline_anchor_coefficient=10.0
    )
    _initialize(raw_bundle, protocol, path)
    recipe = read_recipe(raw_bundle)
    settings = {"algorithm": {"class_name": "PPO"}}
    configure_training_anchor(settings, recipe, training=False)
    assert settings == {"algorithm": {"class_name": "PPO"}}
    assert recipe.baseline_anchor is None
    inference, _ = _algorithm(torch)
    _restore_inference_on_cpu(torch, inference, recipe, raw_bundle, monkeypatch)
    algorithm.eval_mode()
    inference.eval_mode()
    torch.testing.assert_close(
        inference.actor(obs), algorithm.actor(obs), rtol=0, atol=0
    )
    assert not hasattr(inference, "anchor_teacher")


def _restore_inference_on_cpu(torch, algorithm, recipe, source, monkeypatch):
    from npa.workflows.navigation.initialization import initialize_runner

    decode = torch.load

    def on_cpu(path, *, map_location, weights_only):
        assert map_location == "cuda:0" and weights_only is True
        return decode(path, map_location="cpu", weights_only=True)

    monkeypatch.setattr(torch, "load", on_cpu)
    runner = SimpleNamespace(alg=algorithm, current_learning_iteration=0)
    result = initialize_runner(runner, recipe, source)
    assert runner.current_learning_iteration == 41
    assert result["checkpoint_sha256"] == recipe.initial_checkpoint.sha256


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "coefficient",
        "teacher",
        "source",
        "fields",
        "steps",
        "samples",
        "schedule",
    ],
)
def test_resume_rejects_tampered_or_missing_anchor_provenance(
    native, baseline, tmp_path, mutation
):
    from npa.workflows.navigation.anchor_state import ANCHOR_KEY

    torch, digest = native
    original, anchor, obs = baseline
    algorithm, _, _, _ = _bound(native, baseline, tmp_path)
    _advance(torch, algorithm, obs)
    saved = deepcopy(algorithm.save())
    if mutation == "missing":
        saved.pop(ANCHOR_KEY)
    elif mutation == "coefficient":
        saved[ANCHOR_KEY]["coefficient"] = 1.0
    elif mutation == "teacher":
        next(iter(saved[ANCHOR_KEY]["teacher_state_dict"].values())).add_(1)
        saved[ANCHOR_KEY]["teacher_state_sha256"] = digest(
            saved[ANCHOR_KEY]["teacher_state_dict"]
        )
    elif mutation == "source":
        saved[ANCHOR_KEY]["checkpoint_sha256"] = "d" * 64
    elif mutation == "steps":
        saved[ANCHOR_KEY]["optimizer_steps"] += 1
    elif mutation == "samples":
        saved[ANCHOR_KEY]["samples"] += 1
    elif mutation == "schedule":
        saved[ANCHOR_KEY]["num_learning_epochs"] += 1
    else:
        saved[ANCHOR_KEY].pop("samples")
    restored, _ = _algorithm(torch, anchor)
    with pytest.raises(ValueError):
        restored.load(saved, None, True)
        restored.bind_baseline(original, "c" * 64)


def test_incomplete_minibatch_population_and_unbound_update_reject(native, baseline):
    torch, _ = native
    payload, anchor, obs = baseline
    with pytest.raises(ValueError, match="complete native minibatch"):
        _algorithm(torch, anchor, batches=3)
    unbound, _ = _algorithm(torch, anchor)
    unbound.load(deepcopy(payload), None, True)
    with pytest.raises(ValueError, match="bound after the initial snapshot"):
        _advance(torch, unbound, obs)


@pytest.mark.parametrize("option", ["rnd_cfg", "symmetry_cfg", "multi_gpu_cfg"])
def test_unsupported_extensions_reject_before_native_initialization(
    native, baseline, option
):
    from npa.workflows.navigation.anchor_ppo import BaselineAnchoredPPO

    torch, _ = native
    _, anchor, _ = baseline
    plain, _ = _algorithm(torch)
    with pytest.raises(ValueError, match="single-device feedforward"):
        BaselineAnchoredPPO(
            plain.actor,
            plain.critic,
            plain.storage,
            baseline_anchor=anchor,
            **{option: {}},
        )


def test_native_version_and_source_guards_reject(native, monkeypatch):
    from npa.workflows.navigation import anchor_state

    monkeypatch.setattr(anchor_state, "version", lambda package: "5.5.1")
    with pytest.raises(ValueError, match="exact RSL-RL 5.0.1"):
        anchor_state.verify_native_source()
    monkeypatch.setattr(anchor_state, "version", lambda package: "5.0.1")
    monkeypatch.setattr(anchor_state, "PPO_SOURCE_SHA256", "a" * 64)
    with pytest.raises(ValueError, match="exact RSL-RL 5.0.1"):
        anchor_state.verify_native_source()


def test_null_resume_metadata_and_changed_teacher_reject(native, baseline, tmp_path):
    from npa.workflows.navigation.anchor_state import ANCHOR_KEY

    torch, _ = native
    payload, anchor, _ = baseline
    algorithm, _ = _algorithm(torch, anchor)
    with pytest.raises(ValueError, match="complete resume provenance"):
        algorithm.load({**deepcopy(payload), ANCHOR_KEY: None}, None, True)
    bound, _, _, _ = _bound(native, baseline, tmp_path)
    next(bound.anchor_teacher.parameters()).add_(1)
    with pytest.raises(ValueError, match="frozen baseline teacher state changed"):
        bound.save()


def test_nonfinite_or_zero_deviation_fails_before_update(native, baseline, tmp_path):
    torch, _ = native
    algorithm, obs, _, _ = _bound(native, baseline, tmp_path)
    algorithm.actor(obs, stochastic_output=True)
    mean, std = algorithm.actor.output_distribution_params
    with pytest.raises(ValueError, match="finite Gaussian"):
        algorithm.baseline_loss(obs, (mean * float("nan"), std))
    with pytest.raises(ValueError, match="positive Gaussian"):
        algorithm.baseline_loss(obs, (mean, torch.zeros_like(std)))


def test_state_digest_preserves_nested_mapping_structure(native):
    _, digest = native
    assert digest({"a": {"b": 1}, "c": 2}) != digest({"a": {"b": 1, "c": 2}})
    assert digest({"a": {}, "b": {}}) != digest({"a": {"b": {}}})
    assert digest({"a": [1, 2]}) != digest({"a": (1, 2)})


@pytest.mark.parametrize("baseline", [False], indirect=True)
def test_native_update_with_disabled_observation_normalization(
    native, baseline, tmp_path
):
    torch, digest = native
    algorithm, obs, _, _ = _bound(native, baseline, tmp_path)
    assert type(algorithm.actor.obs_normalizer) is torch.nn.Identity
    assert type(algorithm.anchor_teacher.obs_normalizer) is torch.nn.Identity
    teacher = digest(algorithm.anchor_teacher.state_dict())
    losses = _advance(torch, algorithm, obs)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert digest(algorithm.anchor_teacher.state_dict()) == teacher


@pytest.mark.parametrize("damage", ["snapshot", "optimizer", "rng"])
def test_binding_rejects_changed_initial_state_or_randomness(
    native, baseline, tmp_path, monkeypatch, damage
):
    from npa.workflows.navigation.anchor_runtime import bind_after_snapshot

    torch, _ = native
    algorithm, _, runner, recipe = _before_binding(native, baseline, tmp_path)
    reference = tmp_path / "reference.pt"
    if damage == "snapshot":
        saved = torch.load(reference, weights_only=True)
        saved["iter"] += 1
        torch.save(saved, reference)
    else:
        original = algorithm.bind_baseline

        def mutate_after_binding(*args):
            original(*args)
            if damage == "optimizer":
                algorithm.optimizer.param_groups[0]["lr"] *= 2
            else:
                torch.rand(1)

        monkeypatch.setattr(algorithm, "bind_baseline", mutate_after_binding)
    with pytest.raises(ValueError, match="snapshot differs|changed native"):
        bind_after_snapshot(runner, recipe, reference, tmp_path)
