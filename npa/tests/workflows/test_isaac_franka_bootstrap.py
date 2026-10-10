"""Prove training-only bootstrap scope, native action learning, and audit rejection."""

from types import SimpleNamespace
import copy

import pytest

from npa.workflows.sim2real import isaac_franka_bootstrap as bootstrap
from npa.workflows.sim2real.byo_isaac_trainer import (
    _validated_bootstrap_audit,
    _manifest_training_bootstrap,
    _bootstrap_schedule,
    build_isaac_job_manifest,
)
from npa.workflows.sim2real.isaac_job_payload import decode_compressed_bash_args


@pytest.mark.parametrize("eligible", [False, True])
def test_automatic_bootstrap_is_scoped_to_eligible_training(eligible):
    expected = "franka-ik-distillation-v1" if eligible else "none"
    assert bootstrap.resolve_training_bootstrap("auto", eligible=eligible) == expected
    assert bootstrap.resolve_training_bootstrap("none", eligible=eligible) == "none"


@pytest.mark.parametrize("setting", ["unknown", "franka-ik-distillation-v1"])
def test_invalid_or_ineligible_explicit_bootstrap_is_rejected(setting):
    with pytest.raises(ValueError):
        bootstrap.resolve_training_bootstrap(setting, eligible=False)


@pytest.mark.parametrize(
    "changed",
    [
        {"task": "Isaac-Reach-Franka-v0"},
        {"spec": {"robot_source": "custom_usd"}},
        {"resume_uri": "s3://test/checkpoint.pt"},
        {"scenarios": False},
        {"physics": {"friction": 0.5}},
    ],
)
def test_manifest_scope_excludes_other_tasks_robots_resume_and_legacy_physics(changed):
    arguments = {
        "task": "Isaac-Lift-Cube-Franka-v0",
        "spec": {"robot_source": "stock_franka"},
        "physics": None,
        "resume_uri": "",
        "scenarios": True,
        **changed,
    }
    assert _manifest_training_bootstrap("auto", **arguments) == "none"
    with pytest.raises(ValueError):
        _manifest_training_bootstrap("franka-ik-distillation-v1", **arguments)


def test_bootstrap_manifest_keeps_exact_goals_and_all_production_ppo_updates():
    manifest = build_isaac_job_manifest(
        job_name="test-bootstrap",
        run_id="test",
        image="registry/isaac:qualified",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=1024,
        iterations=2000,
        steps_per_env=24,
        s3_output_uri="s3://test/training/",
        s3_endpoint="",
        namespace="default",
        service_account="default",
        gpu_product="test-gpu",
        robot_spec={"robot_source": "stock_franka", "name": "franka"},
        scenarios_uri="s3://test/envs/train/envs.jsonl",
        scenarios_sha256="a" * 64,
        bootstrap_profile="franka-ik-distillation-v1",
    )
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    script = decode_compressed_bash_args(container["args"])
    assert "ROBOT_ITERS=2000" in script and "ROBOT_NUM_ENVS=1024" in script
    assert "ROBOT_STEPS_PER_ENV=24" in script
    assert "export ROBOT_BOOTSTRAP_PROFILE=franka-ik-distillation-v1" in script
    environment = {row["name"]: row.get("value") for row in container["env"]}
    assert environment["NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM"] == "0"


def test_bootstrap_schedule_respects_explicit_operator_settings(monkeypatch):
    monkeypatch.setenv("NPA_BYO_ISAAC_ENTROPY_COEF", "0.002")
    monkeypatch.setenv("NPA_BYO_ISAAC_PPO_LEARNING_RATE", "0.0002")
    monkeypatch.setenv("NPA_BYO_ISAAC_INIT_NOISE_STD", "0.1")
    settings = _bootstrap_schedule()
    assert (
        settings["entropy"],
        settings["learning_rate"],
        settings["initial_noise"],
    ) == ("0.002", "0.0002", "0.1")


def test_stock_entropy_opt_out_disables_bootstrap_annealing_and_noise_freeze(
    monkeypatch,
):
    monkeypatch.setenv("NPA_BYO_ISAAC_ENTROPY_COEF", "stock")
    settings = _bootstrap_schedule()
    assert all(
        settings[name] == ""
        for name in ("entropy", "final_entropy", "fraction", "noise")
    )


def _native_runner_fixture():
    task = SimpleNamespace(
        spec=SimpleNamespace(id="NPA-Lift-Cube-Franka-Scenarios-v0"),
        npa_scenario_rows=[{"env_id": "train-a"}],
    )
    env = SimpleNamespace(unwrapped=task, num_actions=8)
    actor = SimpleNamespace(is_recurrent=False)
    runner = SimpleNamespace(
        current_learning_iteration=0, alg=SimpleNamespace(actor=actor)
    )
    return env, runner


@pytest.mark.parametrize("split", ["validation", "gold_heldout", "gold", ""])
def test_bootstrap_refuses_validation_and_gold_before_native_execution(split):
    env, runner = _native_runner_fixture()
    with pytest.raises(ValueError, match="only the training split"):
        bootstrap._validate_bootstrap(env, runner, "franka-ik-distillation-v1", split)


@pytest.mark.parametrize("iteration", [1, 2000, 10000])
def test_bootstrap_cannot_replace_a_resumed_checkpoint(iteration):
    env, runner = _native_runner_fixture()
    runner.current_learning_iteration = iteration
    with pytest.raises(RuntimeError, match="resumed PPO checkpoint"):
        bootstrap._validate_bootstrap(env, runner, "franka-ik-distillation-v1", "train")


def test_bootstrap_rejects_interpolated_goals_before_demonstration_collection(
    monkeypatch,
):
    env, runner = _native_runner_fixture()
    monkeypatch.setenv("NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM", "1")
    with pytest.raises(RuntimeError, match="exact training goals"):
        bootstrap._validate_bootstrap(env, runner, "franka-ik-distillation-v1", "train")


def _audit():
    row = {
        "training_scenarios": 64,
        "teacher_stable_placements": 48,
        "examples": 1024,
        "optimizer_updates": 12,
        "first_loss": 0.8,
        "last_loss": 0.01,
    }
    return {
        "schema": "npa.sim2real.franka_bootstrap.v1",
        "profile": "franka-ik-distillation-v1",
        "scope": "sealed_training_only",
        "gold_used": False,
        "ppo_updates": 0,
        "training_goals_exact": True,
        "inference_composition": "learned_actor_only",
        "training_scenarios": 64,
        "training_scenarios_sha256": "a" * 64,
        "rounds": [{**row, "round": index} for index in range(2)],
    }


def _validate_audit(audit):
    return _validated_bootstrap_audit(
        audit, profile="franka-ik-distillation-v1", checksum="a" * 64, scenario_count=64
    )


def test_bootstrap_audit_binds_exact_training_identity_and_zero_ppo_updates():
    audit = _audit()
    assert _validate_audit(audit) == audit


@pytest.mark.parametrize(
    "field,value",
    [
        ("gold_used", True),
        ("gold_used", 0),
        ("ppo_updates", False),
        ("ppo_updates", 1),
        ("training_scenarios_sha256", "b" * 64),
        ("training_scenarios", 32),
        ("inference_composition", "teacher_controller"),
        ("scope", "gold_heldout"),
        ("training_goals_exact", False),
        ("training_goals_exact", 1),
    ],
)
def test_bootstrap_audit_rejects_unbound_or_nonliteral_training_evidence(field, value):
    audit = _audit()
    audit[field] = value
    with pytest.raises(RuntimeError):
        _validate_audit(audit)


@pytest.mark.parametrize(
    "field,value",
    [
        ("teacher_stable_placements", 31),
        ("teacher_stable_placements", 65),
        ("teacher_stable_placements", "48"),
        ("examples", False),
        ("optimizer_updates", 0),
        ("last_loss", float("nan")),
        ("last_loss", float("inf")),
        ("first_loss", -0.1),
        ("last_loss", "0.01"),
        ("round", 1),
    ],
)
def test_native_bootstrap_round_rejects_failed_or_invalid_learning(field, value):
    audit = copy.deepcopy(_audit())
    audit["rounds"][0][field] = value
    with pytest.raises(RuntimeError):
        _validate_audit(audit)


def test_measured_approach_and_gripper_closure_control_teacher_transition():
    torch = pytest.importorskip("torch")
    phase = torch.tensor([0, 0, 1, 2, 2, 3, 4])
    elapsed = torch.tensor([0.2, 0.2, 0.2, 0.3, 0.3, 1.0, 1.0])
    distance = torch.tensor([0.01, 0.05, 0.005, 0.005, 0.005, 0.01, 0.0])
    gaps = torch.tensor([0.08, 0.08, 0.08, 0.05, 0.08, 0.05, 0.05])
    result, _ = bootstrap._next_teacher_phase(phase, elapsed, distance, gaps)
    assert result.tolist() == [1, 0, 2, 3, 2, 4, 4]


def test_transport_targets_preserve_measured_cube_to_gripper_offset():
    torch = pytest.importorskip("torch")
    objects = torch.zeros(5, 3)
    goals = torch.ones(5, 3)
    lift_targets = torch.full((5, 3), 0.2)
    offsets = torch.full((5, 3), 0.003)
    target = bootstrap._teacher_target(
        torch.arange(5), objects, goals, lift_targets, offsets
    )
    assert torch.allclose(target[0], torch.tensor([0.0, 0.0, 0.1]))
    assert torch.equal(target[1:3], objects[1:3])
    assert torch.equal(target[3], lift_targets[3])
    assert torch.allclose(target[4], goals[4] + offsets[4])


def test_native_rsl_actor_learns_demonstration_actions_without_a_controller(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    TensorDict = pytest.importorskip("tensordict").TensorDict
    MLPModel = pytest.importorskip("rsl_rl.models").MLPModel
    torch.manual_seed(0)
    observations = TensorDict({"policy": torch.randn(256, 36)}, batch_size=[256])
    actor = MLPModel(
        observations,
        {"actor": ["policy"]},
        "actor",
        8,
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.05,
            "std_type": "scalar",
        },
    )
    labels = observations["policy"][:, :8] * 0.1
    before = actor(observations).detach().clone()
    noise = actor.distribution.std_param.detach().clone()
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    monkeypatch.setattr(bootstrap, "_FITTING_EPOCHS", 80)
    audit = bootstrap._fit_actor(actor, observations, labels, optimizer)
    after = actor(observations).detach()
    assert not torch.equal(before, after)
    assert torch.equal(noise, actor.distribution.std_param.detach())
    assert (after - labels).square().mean() < (before - labels).square().mean() / 10
    assert audit["optimizer_updates"] == 80
    assert audit["last_loss"] < audit["first_loss"]
