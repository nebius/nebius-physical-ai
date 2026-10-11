"""Prove training-only bootstrap scope, native action learning, and audit rejection."""

from types import SimpleNamespace
import copy
import json
import sys

import pytest

from npa.workflows.sim2real import isaac_franka_bootstrap as bootstrap
from npa.workflows.sim2real.isaac_franka_control import franka_control_contract
from npa.workflows.sim2real.isaac_franka_policy import stock_franka_policy_contract
from npa.workflows.sim2real.byo_isaac_trainer import (
    _validated_bootstrap_audit,
    _manifest_training_bootstrap,
    _bootstrap_schedule,
    build_isaac_job_manifest,
)
from npa.workflows.sim2real.isaac_job_payload import decode_compressed_bash_args


class TestNativeBootstrapCheckpoint:
    """Exercise the real RSL logger lifecycle without claiming native physics."""

    @pytest.fixture(autouse=True)
    def native_runtime(self):
        self.torch = pytest.importorskip("torch")
        pytest.importorskip("rsl_rl")
        self.tensor_dict = pytest.importorskip("tensordict").TensorDict

    def _environment(self):
        runtime = self

        class Environment:
            num_envs, num_actions, max_episode_length = 4, 8, 8
            device = "cpu"
            cfg = {}
            spec = SimpleNamespace(id="NPA-Lift-Cube-Franka-Scenarios-v0")
            npa_scenario_rows = [{"env_id": str(index)} for index in range(4)]

            def __init__(self):
                self.unwrapped = self
                self.episode_length_buf = runtime.torch.zeros(
                    4, dtype=runtime.torch.long
                )
                self.state = runtime.torch.randn(4, 39)

            def get_observations(self):
                return runtime.tensor_dict(
                    {"policy": self.state.clone()}, batch_size=[4]
                )

            def reset(self):
                return self.get_observations()

            def step(self, actions):
                self.state[:, :8] += actions * 0.01
                rewards = -self.state.square().mean(dim=1)
                dones = runtime.torch.zeros(4, dtype=runtime.torch.bool)
                return self.get_observations(), rewards, dones, {}

        return Environment()

    def _runner(self, output_dir):
        from npa.genesis.train_teacher import PPOConfig
        from rsl_rl.runners import OnPolicyRunner

        config = PPOConfig(
            actor_hidden_dims=[8],
            critic_hidden_dims=[8],
            num_steps_per_env=4,
            num_learning_epochs=1,
            num_mini_batches=2,
        )
        environment = self._environment()
        from npa.workflows.sim2real.isaac_franka_policy import (
            configure_stock_franka_actor,
            create_stock_franka_runner,
            SCENARIO_TASK_ID,
        )

        configuration = config.to_train_cfg()
        configure_stock_franka_actor(configuration, task=SCENARIO_TASK_ID)
        runner = create_stock_franka_runner(
            OnPolicyRunner,
            environment,
            configuration,
            task=SCENARIO_TASK_ID,
            log_dir=str(output_dir),
            device="cpu",
        )
        return environment, runner

    def _training_observations(self, monkeypatch, environment, tmp_path):
        from npa.workflows.sim2real import isaac_franka_control, isaac_scenario_task

        monkeypatch.setitem(sys.modules, "isaac_scenario_task", isaac_scenario_task)
        scenarios = tmp_path / "training.jsonl"
        scenarios.write_text(
            "".join(json.dumps(row) + "\n" for row in environment.npa_scenario_rows)
        )
        monkeypatch.setenv(isaac_scenario_task.SCENARIO_ENV, str(scenarios))
        monkeypatch.setenv("NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM", "0")
        monkeypatch.setattr(
            isaac_franka_control,
            "stock_franka_control_audit",
            lambda cfg: franka_control_contract(),
        )
        observations = self.tensor_dict(
            {
                "policy": self.torch.randn(64, 39),
                "__done": self.torch.zeros(64, 1, dtype=self.torch.bool),
                "__phase": (self.torch.arange(64) % 5).reshape(64, 1),
            },
            batch_size=[64],
        )
        observations["__done"].reshape(16, 4)[7::8] = True
        monkeypatch.setattr(
            bootstrap,
            "_training_round",
            lambda *args, **kwargs: (
                observations,
                observations["policy"][:, :8] * 0.1,
                self.torch.ones(4, dtype=self.torch.bool),
                {"arm_label_mse": 0.01, "gripper_label_mse": 0.01},
            ),
        )
        return observations

    def test_bootstrap_then_real_ppo_save_and_load(self, monkeypatch, tmp_path):
        from npa.workflows.sim2real import isaac_byo_robot_task

        self.torch.manual_seed(17)
        environment, runner = self._runner(tmp_path)
        observations = self._training_observations(monkeypatch, environment, tmp_path)
        actor_before = runner.alg.actor(observations).detach().clone()
        audit = bootstrap.bootstrap_franka_actor(
            environment,
            runner,
            profile="franka-cartesian-gru-distillation-v2",
            training_split="train",
            output_dir=str(tmp_path),
        )
        assert audit["ppo_updates"] == runner.current_learning_iteration == 0
        assert json.loads((tmp_path / "bootstrap-audit.json").read_text()) == audit
        runner.alg.actor.reset()
        assert not self.torch.equal(
            actor_before, runner.alg.actor(observations).detach()
        )
        runner.alg.actor.reset()
        phase = isaac_byo_robot_task.learn_ppo_phase(
            runner,
            start_iteration=0,
            num_learning_iterations=2,
            init_at_random_ep_len=False,
        )
        assert phase["completed_updates"] == 2
        self._assert_checkpoint_restores_actor(runner, observations, tmp_path)

    def _assert_checkpoint_restores_actor(self, runner, observations, tmp_path):
        checkpoint = tmp_path / "model.pt"
        runner.save(str(checkpoint))
        runner.alg.actor.reset()
        expected = runner.get_inference_policy()(observations).detach()
        _, restored = self._runner(tmp_path / "restored")
        from npa.workflows.sim2real.isaac_franka_policy import (
            SCENARIO_TASK_ID,
            validate_stock_franka_checkpoint,
        )

        periodic = list(tmp_path.glob("model_*.pt"))
        assert periodic, "native learn() did not produce periodic checkpoints"
        for path in periodic:
            payload = self.torch.load(path, map_location="cpu", weights_only=False)
            validate_stock_franka_checkpoint(payload["infos"], task=SCENARIO_TASK_ID)
        information = restored.load(str(checkpoint))
        validate_stock_franka_checkpoint(information, task=SCENARIO_TASK_ID)
        restored.alg.actor.reset()
        self.torch.testing.assert_close(
            restored.get_inference_policy()(observations), expected
        )

    def test_corrective_replay_repeats_before_ppo_when_actor_assessment_fails(
        self, monkeypatch, tmp_path
    ):
        self.torch.manual_seed(17)
        environment, runner = self._runner(tmp_path)
        self._training_observations(monkeypatch, environment, tmp_path)
        original_collection = bootstrap._training_round
        actor_counts = iter([0, 1, 4])
        fractions = []

        def collect(*args, learner_fraction):
            fractions.append(learner_fraction)
            observations, labels, successes, diagnostics = original_collection()
            if learner_fraction == 1.0:
                successes = self.torch.arange(4) < next(actor_counts)
            return observations, labels, successes, diagnostics

        monkeypatch.setattr(bootstrap, "_training_round", collect)
        audit = bootstrap.bootstrap_franka_actor(
            environment,
            runner,
            profile="franka-cartesian-gru-distillation-v2",
            training_split="train",
            output_dir=str(tmp_path),
        )
        assert fractions == [0.0, 0.5, 1.0, 0.75, 1.0, 0.75, 1.0]
        assert audit["ppo_updates"] == runner.current_learning_iteration == 0
        self._assert_corrective_audit(audit)

    def _assert_corrective_audit(self, audit):
        assert [
            row["strict_training_placements"]
            for row in audit["actor_handoff_assessments"]
        ] == [1, 4]
        fit = audit["additional_corrective_fits"][0]["fit"]
        assert fit["replay_batches"] == 4 and fit["replay_examples"] == 256
        assert fit["normalization_sample_count"] == 640
        assert (
            _validated_bootstrap_audit(
                audit,
                profile=audit["profile"],
                checksum=audit["training_scenarios_sha256"],
                scenario_count=4,
            )
            == audit
        )


@pytest.mark.parametrize("eligible", [False, True])
def test_automatic_bootstrap_is_scoped_to_eligible_training(eligible):
    expected = "franka-cartesian-gru-distillation-v2" if eligible else "none"
    assert bootstrap.resolve_training_bootstrap("auto", eligible=eligible) == expected
    assert bootstrap.resolve_training_bootstrap("none", eligible=eligible) == "none"


@pytest.mark.parametrize("setting", ["unknown", "franka-cartesian-gru-distillation-v2"])
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
        _manifest_training_bootstrap(
            "franka-cartesian-gru-distillation-v2", **arguments
        )


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
        bootstrap_profile="franka-cartesian-gru-distillation-v2",
    )
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    script = decode_compressed_bash_args(container["args"])
    assert "ROBOT_ITERS=2000" in script and "ROBOT_NUM_ENVS=1024" in script
    assert "ROBOT_STEPS_PER_ENV=24" in script
    assert (
        "export ROBOT_BOOTSTRAP_PROFILE=franka-cartesian-gru-distillation-v2" in script
    )
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
    actor = SimpleNamespace(is_recurrent=True)
    runner = SimpleNamespace(
        current_learning_iteration=0, alg=SimpleNamespace(actor=actor)
    )
    return env, runner


@pytest.mark.parametrize("split", ["validation", "gold_heldout", "gold", ""])
def test_bootstrap_refuses_validation_and_gold_before_native_execution(split):
    env, runner = _native_runner_fixture()
    with pytest.raises(ValueError, match="only the training split"):
        bootstrap._validate_bootstrap(
            env, runner, "franka-cartesian-gru-distillation-v2", split
        )


@pytest.mark.parametrize("iteration", [1, 2000, 10000])
def test_bootstrap_cannot_replace_a_resumed_checkpoint(iteration):
    env, runner = _native_runner_fixture()
    runner.current_learning_iteration = iteration
    with pytest.raises(RuntimeError, match="resumed PPO checkpoint"):
        bootstrap._validate_bootstrap(
            env, runner, "franka-cartesian-gru-distillation-v2", "train"
        )


def test_bootstrap_rejects_interpolated_goals_before_demonstration_collection(
    monkeypatch,
):
    env, runner = _native_runner_fixture()
    monkeypatch.setenv("NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM", "1")
    with pytest.raises(RuntimeError, match="exact training goals"):
        bootstrap._validate_bootstrap(
            env, runner, "franka-cartesian-gru-distillation-v2", "train"
        )


def _recurrent_fit(batches):
    return {
        "native_recurrent_model": "RNNModel-GRU",
        "recurrent_hidden_dim": 128,
        "real_reset_boundaries": True,
        "expert_phase_is_observation": False,
        "observation_normalization": True,
        "normalization_mean_nonzero": True,
        "fitting_epochs": 32,
        "replay_batches": batches,
        "replay_examples": 1024 * batches,
        "normalization_sample_count": 1024 * batches * (batches + 1) // 2,
        "optimizer_updates": 32 * batches,
        "replay_episodes": 64 * batches,
        "training_phase_example_counts": [1024 * batches, 0, 0, 0, 0],
        "first_loss": 0.8,
        "last_loss": 0.01,
    }


def _actor_collection(fraction):
    return {
        "scope": "sealed_training_only",
        "training_scenarios": 64,
        "strict_training_placements": 48,
        "learner_fraction": fraction,
        "inference_composition": "learned_actor_only"
        if fraction == 1.0
        else "training_demonstration_mixture",
        "gold_used": False,
        "ppo_updates": 0,
        "arm_label_mse": 0.01,
        "gripper_label_mse": 0.01,
    }


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
        "schema": "npa.sim2real.franka_bootstrap.v2",
        "profile": "franka-cartesian-gru-distillation-v2",
        "scope": "sealed_training_only",
        "gold_used": False,
        "ppo_updates": 0,
        "training_goals_exact": True,
        "robot_control": franka_control_contract(),
        "inference_composition": "learned_actor_only",
        "training_scenarios": 64,
        "training_scenarios_sha256": "a" * 64,
        "rounds": [
            {**row, "round": index, **_recurrent_fit(index + 1)} for index in range(2)
        ],
        "policy_model": stock_franka_policy_contract(),
        "actor_baseline": _actor_collection(1.0),
        "corrective_fit": {**_actor_collection(0.75), "fit": _recurrent_fit(3)},
        "actor_handoff": _actor_collection(1.0),
        "additional_corrective_fits": [],
        "actor_handoff_assessments": [_actor_collection(1.0)],
    }


def _validate_audit(audit):
    return _validated_bootstrap_audit(
        audit,
        profile="franka-cartesian-gru-distillation-v2",
        checksum="a" * 64,
        scenario_count=64,
    )


def test_bootstrap_audit_binds_exact_training_identity_and_zero_ppo_updates():
    audit = _audit()
    assert _validate_audit(audit) == audit


def _adaptive_audit():
    audit = _audit()
    audit["additional_corrective_fits"] = [
        {**_actor_collection(0.75), "fit": _recurrent_fit(4)}
    ]
    audit["actor_handoff_assessments"].insert(
        0, {**_actor_collection(1.0), "strict_training_placements": 31}
    )
    return audit


def test_adaptive_audit_preserves_failed_assessment_and_all_retained_fits():
    audit = _adaptive_audit()
    assert _validate_audit(audit) == audit
    audit["additional_corrective_fits"][0]["fit"] = _recurrent_fit(3)
    with pytest.raises(RuntimeError, match="exact replay"):
        _validate_audit(audit)


@pytest.mark.parametrize("history", [None, [], [_actor_collection(1.0)]])
def test_adaptive_audit_rejects_missing_actor_assessment_history(history):
    audit = _adaptive_audit()
    audit["actor_handoff_assessments"] = history
    with pytest.raises(RuntimeError, match="history"):
        _validate_audit(audit)


def test_adaptive_audit_cannot_continue_after_a_successful_actor_assessment():
    audit = _adaptive_audit()
    audit["actor_handoff_assessments"][0]["strict_training_placements"] = 32
    with pytest.raises(RuntimeError, match="replaced a successful handoff"):
        _validate_audit(audit)


def test_adaptive_audit_cannot_substitute_different_final_actor_evidence():
    audit = _adaptive_audit()
    audit["actor_handoff_assessments"][-1]["strict_training_placements"] = 47
    with pytest.raises(RuntimeError, match="differs from its final assessment"):
        _validate_audit(audit)


def test_additional_corrective_fit_cannot_consume_gold():
    audit = _adaptive_audit()
    audit["additional_corrective_fits"][0]["gold_used"] = True
    with pytest.raises(RuntimeError, match="training-only"):
        _validate_audit(audit)


@pytest.mark.parametrize(
    "field,value",
    [
        ("strict_training_placements", 31),
        ("strict_training_placements", 65),
        ("strict_training_placements", True),
        ("learner_fraction", 0.75),
        ("learner_fraction", 1),
        ("inference_composition", "training_demonstration_mixture"),
        ("scope", "gold_heldout"),
        ("gold_used", 0),
        ("ppo_updates", False),
        ("arm_label_mse", float("nan")),
        ("gripper_label_mse", -0.1),
    ],
)
def test_bootstrap_requires_literal_successful_actor_only_handoff(field, value):
    audit = _audit()
    audit["actor_handoff"][field] = value
    with pytest.raises(RuntimeError):
        _validate_audit(audit)


def test_teacher_success_cannot_replace_the_actor_handoff():
    audit = _audit()
    del audit["actor_handoff"]
    with pytest.raises(RuntimeError, match="actor collection"):
        _validate_audit(audit)


@pytest.mark.parametrize(
    "field,value",
    [
        ("normalization_sample_count", 0),
        ("normalization_sample_count", 6144.0),
        ("normalization_mean_nonzero", False),
        ("real_reset_boundaries", False),
        ("expert_phase_is_observation", True),
        ("expert_phase_is_observation", 0),
        ("replay_batches", 2),
        ("replay_examples", 2048),
        ("replay_episodes", False),
        ("training_phase_example_counts", [3071, 0, 0, 0, 0]),
    ],
)
def test_corrective_fit_requires_initialized_normalization_and_reset_sequences(
    field, value
):
    audit = _audit()
    audit["corrective_fit"]["fit"][field] = value
    with pytest.raises(RuntimeError):
        _validate_audit(audit)


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation_normalization", 1),
        ("expert_phase_is_observation", 0),
        ("action_dim", 8.0),
        ("native_ik_mode", "relative_pose"),
        ("native_action_scale", [1.0] * 7),
    ],
)
def test_bootstrap_policy_semantics_are_bound_even_when_dimensions_match(field, value):
    audit = _audit()
    audit["policy_model"][field] = value
    with pytest.raises(RuntimeError):
        _validate_audit(audit)


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
        ("robot_control", None),
    ],
)
def test_bootstrap_audit_rejects_unbound_or_nonliteral_training_evidence(field, value):
    audit = _audit()
    audit[field] = value
    with pytest.raises(RuntimeError):
        _validate_audit(audit)


@pytest.mark.parametrize(
    "field,value", [("robot_gravity_compensated", 1), ("decimation", 4.0)]
)
def test_bootstrap_control_audit_requires_literal_native_evidence(field, value):
    audit = _audit()
    audit["robot_control"][field] = value
    with pytest.raises(RuntimeError, match="literal"):
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


def test_failed_demonstrations_report_actual_coverage_before_actor_fitting(monkeypatch):
    import numpy as np

    successes = np.array([True, False, True, False, False])
    monkeypatch.setattr(
        bootstrap,
        "_training_round",
        lambda *args, **kwargs: (None, None, successes, {}),
    )

    def unexpected_fit(*args):
        pytest.fail("failed demonstrations reached supervised actor fitting")

    monkeypatch.setattr(bootstrap, "_fit_actor", unexpected_fit)
    with pytest.raises(RuntimeError, match="round 1, 2/5 training scenarios"):
        bootstrap._bootstrap_round(None, None, None, index=1)


def _placement_teacher_fixture(torch):
    objects = torch.tensor([[0.049, 0.0, 0.0], [0.05, 0.0, 0.0], [0.049, 0.0, 0.0]])
    velocities = torch.tensor([[0.029, 0.0, 0.0], [0.0, 0.0, 0.0], [0.03, 0.0, 0.0]])
    task = SimpleNamespace(
        scene={
            "object": SimpleNamespace(
                data=SimpleNamespace(
                    root_pos_w=objects,
                    root_lin_vel_w=velocities,
                )
            ),
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_pos_w=torch.zeros(3, 3),
                    root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
                )
            ),
            "ee_frame": SimpleNamespace(
                data=SimpleNamespace(
                    target_pos_w=objects.unsqueeze(1),
                )
            ),
        },
        command_manager=SimpleNamespace(
            get_command=lambda name: {
                "object_pose": torch.zeros(3, 7),
            }[name]
        ),
        npa_scenario_indices=torch.tensor([2, 4, 0]),
    )
    teacher = bootstrap._FrankaTeacher.__new__(bootstrap._FrankaTeacher)
    teacher.task = task
    teacher.stable_steps = torch.zeros(3, dtype=torch.long)
    teacher.successful_rows = torch.zeros(5, dtype=torch.bool)
    return teacher


def test_teacher_uses_shared_strict_placement_state_and_consecutive_dwell(monkeypatch):
    torch = pytest.importorskip("torch")
    from npa.workflows.sim2real import isaac_scenario_task

    monkeypatch.setitem(sys.modules, "isaac_scenario_task", isaac_scenario_task)
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.utils.math",
        SimpleNamespace(
            combine_frame_transforms=lambda position, rotation, target: (
                position + target,
                rotation,
            ),
        ),
    )
    teacher = _placement_teacher_fixture(torch)
    teacher._record_stable_placements()
    teacher._record_stable_placements()
    assert not teacher.successful_rows.any()
    teacher.task.scene["object"].data.root_lin_vel_w[0, 0] = 0.04
    teacher._record_stable_placements()
    assert teacher.stable_steps.tolist() == [0, 0, 0]
    teacher.task.scene["object"].data.root_lin_vel_w[0, 0] = 0.029
    for _ in range(3):
        teacher._record_stable_placements()
    assert teacher.successful_rows.tolist() == [False, False, True, False, False]


def test_native_rsl_actor_learns_demonstrations_with_fresh_normalization(monkeypatch):
    torch = pytest.importorskip("torch")
    TensorDict = pytest.importorskip("tensordict").TensorDict
    RNNModel = pytest.importorskip("rsl_rl.models").RNNModel
    from npa.workflows.sim2real.isaac_recurrent_imitation import (
        RecurrentDemonstrationReplay,
    )

    torch.manual_seed(0)
    observations = TensorDict(
        {
            "policy": torch.randn(256, 39) + 2.0,
            "__done": torch.zeros(256, 1, dtype=torch.bool),
            "__phase": (torch.arange(256) % 5).reshape(256, 1),
        },
        batch_size=[256],
    )
    observations["__done"].reshape(64, 4)[31::32] = True
    actor = RNNModel(
        observations[:4],
        {"actor": ["policy"]},
        "actor",
        8,
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=16,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.05,
            "std_type": "scalar",
        },
    )
    actor.eval()
    labels = observations["policy"][:, :8] * 0.1
    before = _native_sequence_outputs(actor, observations, num_envs=4)
    assert int(actor.obs_normalizer.count) == 0
    noise = actor.distribution.std_param.detach().clone()
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    replay = RecurrentDemonstrationReplay(4)
    audit = replay.fit(actor, observations, labels, optimizer, epochs=80)
    after = _native_sequence_outputs(actor, observations, num_envs=4)
    torch.testing.assert_close(
        actor.obs_normalizer.mean, observations["policy"].mean(dim=0)
    )
    assert audit["normalization_sample_count"] == 256
    assert audit["real_reset_boundaries"] is True and audit["replay_episodes"] == 8
    assert audit["expert_phase_is_observation"] is False
    assert actor.obs_groups == ["policy"]
    assert torch.equal(noise, actor.distribution.std_param.detach())
    assert (after - labels).square().mean() < (before - labels).square().mean() / 10
    assert audit["optimizer_updates"] == 80 and audit["last_loss"] < audit["first_loss"]


def _native_sequence_outputs(actor, observations, *, num_envs):
    import torch

    actor.reset()
    outputs = []
    with torch.no_grad():
        for offset in range(0, len(observations), num_envs):
            batch = observations[offset : offset + num_envs]
            outputs.append(actor(batch).clone())
            actor.reset(batch["__done"].squeeze(-1))
    actor.reset()
    return torch.cat(outputs)
