"""Tests for the BYO Isaac-Lab RSL-RL trainer (real RL for the sim2real loop).

The headline guarantee: the dry-run output satisfies the real
``VlmSignalUpdateResult.from_dict`` contract that ``_run_trainer_via_command``
enforces, and the live path builds a correct Isaac training Job manifest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from npa.workflows.sim2real import byo_isaac_trainer as byo
from npa.workflows.sim2real.isaac_byo_robot_task import (
    configure_convergence_action_noise,
)
from npa.workflows.sim2real.isaac_job_payload import decode_compressed_bash_args


def _manifest_script(manifest):
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    return decode_compressed_bash_args(container["args"])


def _write_signal(tmp_path):
    path = tmp_path / "signal.json"
    path.write_text(
        json.dumps(
            {
                "schema": "npa.sim2real.rl_signal.v1",
                "signals": [
                    {
                        "per_step": [
                            {"reward": 0.6, "advantage": 0.2},
                            {"reward": 0.4, "advantage": -0.1},
                        ]
                    },
                    {"per_step": [{"reward": 0.8, "advantage": 0.3}]},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


class _FakeNoiseParameter:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.requires_grad = True

    def detach(self):
        return self

    def fill_(self, value: float):
        self.values = [float(value) for _ in self.values]
        return self

    def requires_grad_(self, enabled: bool):
        self.requires_grad = enabled
        return self

    def reshape(self, *_shape):
        return self

    def tolist(self):
        return list(self.values)


def test_resume_convergence_noise_matches_deterministic_policy_contract() -> None:
    class Policy:
        state_dependent_std = False
        noise_std_type = "scalar"
        std = _FakeNoiseParameter([0.41] * 7)

    audit = configure_convergence_action_noise(Policy(), target_std=0.05)

    assert audit == {
        "noise_std_type": "scalar",
        "target_std": 0.05,
        "parameter_count": 7,
        "frozen": True,
    }
    assert Policy.std.values == pytest.approx([0.05] * 7)
    assert Policy.std.requires_grad is False


def test_resume_convergence_noise_supports_log_std_and_fails_closed() -> None:
    class LogPolicy:
        state_dependent_std = False
        noise_std_type = "log"
        log_std = _FakeNoiseParameter([0.0, 0.0])

    audit = configure_convergence_action_noise(LogPolicy(), target_std=0.1)
    assert audit["parameter_count"] == 2
    assert LogPolicy.log_std.values == pytest.approx([-2.302585093] * 2)

    class StateDependentPolicy:
        state_dependent_std = True
        noise_std_type = "scalar"

    with pytest.raises(RuntimeError, match="state-dependent"):
        configure_convergence_action_noise(StateDependentPolicy(), target_std=0.05)
    with pytest.raises(ValueError, match="finite"):
        configure_convergence_action_noise(LogPolicy(), target_std=float("nan"))


def test_read_signal_stats(tmp_path):
    stats = byo.read_signal_stats(str(_write_signal(tmp_path)))
    assert stats["step_count"] == 3
    assert stats["mean_reward"] == pytest.approx((0.6 + 0.4 + 0.8) / 3)
    assert stats["mean_advantage"] == pytest.approx((0.2 - 0.1 + 0.3) / 3)
    assert stats["mean_absolute_advantage"] == pytest.approx((0.2 + 0.1 + 0.3) / 3)
    assert stats["advantage_variance"] > 0.0


def test_read_signal_stats_missing_file_is_safe(tmp_path):
    stats = byo.read_signal_stats(str(tmp_path / "nope.json"))
    assert stats == {"mean_reward": 0.0, "mean_advantage": 0.0, "step_count": 0}


def test_build_update_result_satisfies_byo_contract(tmp_path):
    """The emitted dict must parse via the real VlmSignalUpdateResult.from_dict."""

    from npa.workbench.lerobot.policy_container import VlmSignalUpdateResult

    stats = byo.read_signal_stats(str(_write_signal(tmp_path)))
    result = byo.build_update_result(
        stats=stats,
        initial_reward_head=0.0,
        iterations=150,
        steps_per_env=24,
        checkpoint_uri="s3://bucket/run/model_latest.pt",
        status="success",
        duration_ms=1234.0,
    )
    # Required contract fields present + non-empty policy_output_after.
    assert result["reward_head_after"] != 0.0
    assert (
        isinstance(result["policy_output_after"], list)
        and result["policy_output_after"]
    )
    assert result["policy_delta_l2"] > 0.0  # a real trainer produced a checkpoint
    assert result["backend"] == "isaac_rsl_rl_ppo"
    assert result["checkpoint_path"].endswith("model_latest.pt")
    assert result["effective_learning_rate"] == 0.08
    assert result["learning_rate_scope"] == "vlm_signal_adapter_and_no_signal_control"
    parsed = VlmSignalUpdateResult.from_dict(result)
    assert parsed.checkpoint_path == "s3://bucket/run/model_latest.pt"
    assert parsed.backend == "isaac_rsl_rl_ppo"
    assert parsed.signal_statistics["nonzero_advantage_count"] == 3


def test_build_update_result_no_checkpoint_zero_delta(tmp_path):
    result = byo.build_update_result(
        stats={"mean_reward": 0.0, "mean_advantage": 0.0, "step_count": 0},
        initial_reward_head=0.0,
        iterations=10,
        checkpoint_uri="",
        status="failed",
        duration_ms=0.0,
    )
    assert result["policy_delta_l2"] == 0.0


def test_build_isaac_job_manifest_shape():
    manifest = byo.build_isaac_job_manifest(
        job_name="s2r-byo-isaac-train-run1",
        run_id="run1",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=1024,
        iterations=150,
        s3_output_uri="s3://bucket/sim2real-b/run1/byo-trainer/job/",
        s3_endpoint="https://s3.example",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    )
    spec = manifest["spec"]["template"]["spec"]
    container = spec["containers"][0]
    assert container["image"] == "reg/npa-isaac-lab:2.3.2.post1"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert spec["nodeSelector"]["nvidia.com/gpu.product"].startswith("NVIDIA-RTX-PRO")
    args = decode_compressed_bash_args(container["args"])
    assert max(map(len, container["args"])) < 128 * 1024
    assert "Isaac-Lift-Cube-Franka-v0" in args
    assert "--max_iterations 150" in args
    assert "--num_envs 1024" in args
    assert "agent.num_steps_per_env=24" in args
    assert byo.TRAIN_SCRIPT in args
    assert "npa.workflows.sim2real.runtime_attestation" in args
    assert "npa.workflows.sim2real.isaac_job_io upload-training" in args
    assert "pip install" not in args
    assert "<<" not in args
    env = {item["name"]: item["value"] for item in container["env"]}
    # Strict validation requires the object to remain stably placed at episode
    # end. Training therefore keeps running after the first three-step event so
    # the saturated dwell reward teaches post-success holding behavior.
    assert env["NPA_SIM2REAL_ENABLE_SUCCESS_TERMINATION"] == "0"
    assert env["NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM"] == "1"
    assert env["NPA_SIM2REAL_GOAL_CURRICULUM_FULL_STEP"] == "2160"

    opt_in = byo.build_isaac_job_manifest(
        job_name="s2r-byo-isaac-train-run1-opt-in",
        run_id="run1-opt-in",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=1024,
        iterations=150,
        s3_output_uri="s3://bucket/sim2real-b/run1-opt-in/byo-trainer/job/",
        s3_endpoint="https://s3.example",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        success_termination_enabled=True,
    )
    opt_in_env = {
        item["name"]: item["value"]
        for item in opt_in["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert opt_in_env["NPA_SIM2REAL_ENABLE_SUCCESS_TERMINATION"] == "1"


def test_dryrun_main_writes_contract_json(tmp_path, monkeypatch):
    from npa.workbench.lerobot.policy_container import VlmSignalUpdateResult

    out = tmp_path / "update.json"
    monkeypatch.setenv("NPA_BYO_ISAAC_DRYRUN", "1")
    monkeypatch.setenv("NPA_SIM2REAL_SIGNAL_JSON", str(_write_signal(tmp_path)))
    monkeypatch.setenv("NPA_SIM2REAL_OUTPUT_JSON", str(out))
    monkeypatch.setenv("NPA_BYO_ISAAC_ITERATIONS", "3")
    monkeypatch.setenv("NPA_SIM2REAL_LEARNING_RATE", "0.12")
    rc = byo.main()
    assert rc == 0
    payload = json.loads(out.read_text())
    parsed = VlmSignalUpdateResult.from_dict(payload)  # must not raise
    assert parsed.backend == "isaac_rsl_rl_ppo"
    assert parsed.steps == 3
    assert payload["effective_learning_rate"] == 0.12


def test_vlm_reward_overrides_targets_error_tag_term():
    # VLM says reaching is failing -> reaching_object weight boosted above default 1.0.
    stats = {
        "mean_reward": 0.2,
        "mean_advantage": 0.0,
        "step_count": 5,
        "error_tags": {"did_not_reach_object": 4, "minor": 1},
    }
    ov = byo.vlm_reward_overrides(stats)
    assert ov["env.rewards.reaching_object.weight"] > 1.0
    # untouched term stays at its default weight
    assert ov["env.rewards.lifting_object.weight"] == 15.0


def test_vlm_reward_overrides_low_reward_broadly_boosts_and_is_bounded():
    stats = {
        "mean_reward": -1.0,
        "mean_advantage": 0.0,
        "step_count": 3,
        "error_tags": {},
    }
    ov = byo.vlm_reward_overrides(stats)
    # broad boost applied (mult>1) but bounded to <= 2x default
    assert ov["env.rewards.lifting_object.weight"] > 15.0
    assert ov["env.rewards.lifting_object.weight"] <= 30.0
    assert ov["env.rewards.reaching_object.weight"] <= 2.0


def test_manifest_embeds_reward_overrides():
    ov = {
        "env.rewards.reaching_object.weight": 1.6,
        "env.rewards.lifting_object.weight": 15.0,
    }
    m = byo.build_isaac_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=512,
        iterations=30,
        s3_output_uri="s3://b/o/",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        reward_overrides=ov,
    )
    args = _manifest_script(m)
    assert "env.rewards.reaching_object.weight=1.6" in args


def test_manifest_embeds_exploration_overrides():
    # entropy_coef / init_noise_std become hydra overrides on the default train
    # command — the exploration fix that keeps the Lift policy from collapsing
    # into a reach-and-hover local optimum on an unlucky seed.
    m = byo.build_isaac_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=1024,
        iterations=600,
        s3_output_uri="s3://b/o/",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        entropy_coef="0.01",
        init_noise_std="1.2",
    )
    args = _manifest_script(m)
    assert "agent.algorithm.entropy_coef=0.01" in args
    assert "agent.policy.init_noise_std=1.2" in args


def test_scenario_wrapper_consumes_reward_and_native_ppo_contract() -> None:
    args = _manifest_script(
        byo.build_isaac_job_manifest(
            job_name="j",
            run_id="r",
            image="reg/npa-isaac-lab:2.3.2.post1",
            task="Isaac-Lift-Cube-Franka-v0",
            num_envs=1024,
            iterations=500,
            s3_output_uri="s3://b/o/",
            s3_endpoint="https://s3",
            namespace="default",
            service_account="agent-sa",
            gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
            reward_overrides={"env.rewards.lifting_object.weight": 21.0},
            entropy_coef="0.006",
            entropy_final_coef="0.0005",
            entropy_anneal_fraction="0.6",
            ppo_optimizer_learning_rate="0.001",
            init_noise_std="1.2",
            convergence_action_noise_std="0.05",
            validation_interval=100,
            object_usd="https://assets.example/cube.usd",
            scenarios_jsonl='{"scenario_config_digest":"digest"}\n',
            robot_spec={"robot_source": "stock_franka", "name": "franka"},
        )
    )
    assert "ROBOT_REWARD_OVERRIDES_JSON=" in args
    assert "lifting_object.weight" in args
    assert "ROBOT_PPO_LEARNING_RATE=0.001" in args
    assert "ROBOT_ENTROPY_COEF=0.006" in args
    assert "ROBOT_ENTROPY_FINAL_COEF=0.0005" in args
    assert "ROBOT_ENTROPY_ANNEAL_FRACTION=0.6" in args
    assert "ROBOT_INIT_NOISE_STD=1.2" in args
    assert "ROBOT_CONVERGENCE_ACTION_NOISE_STD=0.05" in args
    assert "ROBOT_VALIDATION_INTERVAL=100" in args
    assert "ROBOT_OBJECT_USD=https://assets.example/cube.usd" in args
    assert "/opt/npa/isaac-runtime/isaac_robot_train.py" in args
    # Those markers are produced by the wrapper baked into the immutable image,
    # not by source text injected into the live Job manifest.
    from npa.workflows.sim2real.isaac_byo_robot_task import TRAIN_WRAPPER_SCRIPT

    assert "ROBOT_REWARD_OVERRIDES_APPLIED" in TRAIN_WRAPPER_SCRIPT
    assert "ROBOT_PPO_SETTINGS_APPLIED" in TRAIN_WRAPPER_SCRIPT
    assert "ROBOT_ENTROPY_ANNEALED" in TRAIN_WRAPPER_SCRIPT
    assert "ROBOT_ACTION_NOISE_CONVERGENCE" in TRAIN_WRAPPER_SCRIPT
    assert "ROBOT_REWARD_OVERRIDES_APPLIED" not in args


def test_entropy_curriculum_validation_fails_closed() -> None:
    common = dict(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=1024,
        iterations=500,
        s3_output_uri="s3://b/o/",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    )
    with pytest.raises(ValueError, match="initial entropy"):
        byo.build_isaac_job_manifest(
            **common,
            entropy_final_coef="0.0005",
            entropy_anneal_fraction="0.6",
        )
    with pytest.raises(ValueError, match="between zero and entropy"):
        byo.build_isaac_job_manifest(
            **common,
            entropy_coef="0.006",
            entropy_final_coef="0.01",
            entropy_anneal_fraction="0.6",
        )
    with pytest.raises(ValueError, match="between zero and one"):
        byo.build_isaac_job_manifest(
            **common,
            entropy_coef="0.006",
            entropy_final_coef="0.0005",
            entropy_anneal_fraction="1.0",
        )
    with pytest.raises(ValueError, match="between zero and one"):
        byo.build_isaac_job_manifest(
            **common,
            entropy_coef="0.006",
            entropy_final_coef="0.0005",
            entropy_anneal_fraction="0.6",
            convergence_action_noise_std="1.1",
        )
    with pytest.raises(ValueError, match="two-phase"):
        byo.build_isaac_job_manifest(
            **common,
            convergence_action_noise_std="0.05",
        )


def test_manifest_downloads_sha_pinned_scenario_distribution_without_embedding() -> (
    None
):
    marker = "large-scenario-record-" + ("x" * 400_000)
    digest = hashlib.sha256(marker.encode()).hexdigest()
    manifest = byo.build_isaac_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=1024,
        iterations=500,
        s3_output_uri="s3://b/o/",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        robot_spec={"robot_source": "stock_franka", "name": "franka"},
        scenarios_jsonl=marker,
        scenarios_uri="s3://b/run/envs/train/envs.jsonl",
        scenarios_sha256=digest,
    )
    script = _manifest_script(manifest)

    assert marker not in script
    assert "s3://b/run/envs/train/envs.jsonl" in script
    assert digest in script
    assert "npa.workflows.sim2real.isaac_job_io download" in script
    assert f"--sha256 {digest}" in script
    assert "boto3" not in script
    assert "download_file" not in script
    assert (
        subprocess.run(
            ["bash", "-n"], input=script, text=True, capture_output=True, check=False
        ).returncode
        == 0
    )
    assert len(json.dumps(manifest).encode()) < 300_000


def test_manifest_rejects_large_embedded_scenario_distribution() -> None:
    with pytest.raises(ValueError, match="scenarios_uri"):
        byo.build_isaac_job_manifest(
            job_name="j",
            run_id="r",
            image="reg/npa-isaac-lab:2.3.2.post1",
            task="Isaac-Lift-Cube-Franka-v0",
            num_envs=1024,
            iterations=500,
            s3_output_uri="s3://b/o/",
            s3_endpoint="https://s3",
            namespace="default",
            service_account="agent-sa",
            gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
            robot_spec={"robot_source": "stock_franka", "name": "franka"},
            scenarios_jsonl="x" * 300_000,
        )


def test_manifest_omits_exploration_overrides_when_unset():
    # Unset -> default Franka train command stays byte-for-byte unchanged.
    m = byo.build_isaac_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=1024,
        iterations=600,
        s3_output_uri="s3://b/o/",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
    )
    args = _manifest_script(m)
    assert "agent.algorithm.entropy_coef" not in args
    assert "agent.policy.init_noise_std" not in args


def test_manifest_embeds_custom_object_usd():
    m = byo.build_isaac_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=512,
        iterations=30,
        s3_output_uri="s3://b/o/",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        object_usd="s3orhttp://assets/custom_sugar_box.usd",
        object_scale="(0.8, 0.8, 0.8)",
    )
    args = _manifest_script(m)
    assert (
        "env.scene.object.spawn.usd_path=s3orhttp://assets/custom_sugar_box.usd" in args
    )
    assert "env.scene.object.spawn.scale='(0.8, 0.8, 0.8)'" in args


def test_resolve_object_usd_defaults_to_rigid_ready_cube(monkeypatch):
    # Unset -> proven rigid-ready MultiColorCube on the public Omniverse CDN.
    monkeypatch.delenv("NPA_ISAAC_NUCLEUS_DIR", raising=False)
    usd = byo.resolve_object_usd("")
    assert usd.endswith(byo.DEFAULT_OBJECT_USD_REL)
    assert usd.startswith("https://omniverse-content-production")
    assert usd == byo.default_isaac_object_usd()


def test_resolve_object_usd_explicit_wins(monkeypatch):
    monkeypatch.delenv("NPA_ISAAC_NUCLEUS_DIR", raising=False)
    assert byo.resolve_object_usd("s3://b/custom.usd") == "s3://b/custom.usd"


def test_resolve_object_usd_stock_sentinel_opts_out():
    # Operator escape hatch: fall back to the built-in primitive cube.
    for sentinel in ("stock", "none", "PRIMITIVE", " Builtin "):
        assert byo.resolve_object_usd(sentinel) == ""


def test_default_isaac_object_usd_honors_nucleus_override(monkeypatch):
    monkeypatch.setenv("NPA_ISAAC_NUCLEUS_DIR", "https://mirror.internal/Isaac/")
    usd = byo.default_isaac_object_usd()
    assert usd == f"https://mirror.internal/Isaac/{byo.DEFAULT_OBJECT_USD_REL}"


def test_read_generated_train_env(tmp_path):
    envs = tmp_path / "envs.jsonl"
    envs.write_text(
        '{"env_id": "env-00006", "seed": 516456434, "physics": {"friction": 0.717, "mass_scale": 0.969}}\n'
        '{"env_id": "env-00007", "seed": 42, "physics": {}}\n',
        encoding="utf-8",
    )
    rec = byo.read_generated_train_env(str(tmp_path))
    assert rec["env_id"] == "env-00006"
    assert rec["seed"] == 516456434
    assert rec["physics"]["friction"] == 0.717


def test_read_generated_train_env_absent(tmp_path):
    assert byo.read_generated_train_env(str(tmp_path)) == {}
    assert byo.read_generated_train_env("") == {}


def test_manifest_embeds_generated_seed():
    m = byo.build_isaac_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=512,
        iterations=30,
        s3_output_uri="s3://b/o/",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        seed=516456434,
    )
    args = _manifest_script(m)
    # generated env seed drives randomization via train.py --seed (NOT a hydra
    # env.seed= override, which the Lift cfg rejects as a type error).
    assert "--seed 516456434" in args
    assert "env.seed=" not in args


def test_manifest_no_seed_arg_when_zero():
    m = byo.build_isaac_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=512,
        iterations=30,
        s3_output_uri="s3://b/o/",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        seed=0,
    )
    args = _manifest_script(m)
    assert "--seed" not in args


def test_manifest_physics_path_ships_wrapper_and_skips_stock_train():
    m = byo.build_isaac_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=64,
        iterations=2,
        s3_output_uri="s3://b/o/",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        seed=736958930,
        physics={"friction": 0.7, "mass_scale": 0.95},
    )
    args = _manifest_script(m)
    # Runs the wrapper baked into the exact runtime image and sets generated physics.
    assert "/opt/npa/isaac-runtime/isaac_physics_train.py" in args
    assert "NPA_GEN_FRICTION=0.7" in args and "NPA_GEN_MASS_SCALE=0.95" in args
    assert "PHYS_SEED=736958930" in args
    assert "/tmp/npa_phys/runner.py" not in args
    # physics path does NOT invoke stock train.py
    assert byo.TRAIN_SCRIPT not in args
    assert "npa.workflows.sim2real.isaac_job_io upload-training" in args


def test_manifest_default_path_unchanged_without_physics():
    m = byo.build_isaac_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=512,
        iterations=30,
        s3_output_uri="s3://b/o/",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        seed=42,
    )
    args = _manifest_script(m)
    # proven path: stock train.py, no physics wrapper
    assert byo.TRAIN_SCRIPT in args
    assert "isaac_physics_task.py" not in args
    assert "--seed 42" in args
    assert "--kit_args '--portable-root /tmp/npa-isaac-kit'" in args


def test_byo_wrapper_saves_resumed_absolute_iteration() -> None:
    script = _manifest_script(
        byo.build_isaac_job_manifest(
            job_name="j",
            run_id="r",
            image="reg/npa-isaac-lab:2.3.2.post1",
            task="Isaac-Lift-Cube-Franka-v0",
            num_envs=64,
            iterations=500,
            s3_output_uri="s3://b/o/",
            s3_endpoint="https://s3",
            namespace="default",
            service_account="agent-sa",
            gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
            robot_spec={"robot_source": "stock_franka", "name": "Franka"},
            scenarios_uri="s3://b/train/envs.jsonl",
            scenarios_sha256="a" * 64,
        )
    )
    assert "/opt/npa/isaac-runtime/isaac_robot_train.py" in script
    from npa.workflows.sim2real.isaac_byo_robot_task import TRAIN_WRAPPER_SCRIPT

    assert "current_learning_iteration" in TRAIN_WRAPPER_SCRIPT
    assert "final_iteration" in TRAIN_WRAPPER_SCRIPT
    assert "ROBOT_FINAL_CHECKPOINT" in TRAIN_WRAPPER_SCRIPT


def test_read_generated_train_env_s3_fallback(tmp_path, monkeypatch):
    # Local dir missing -> falls back to the S3 URI (orchestrator only syncs heldout).
    captured = {}

    class _FakeBody:
        def read(self_inner):
            return (
                b'{"env_id": "env-00006", "seed": 99, '
                b'"physics": {"friction": 0.71, "mass_scale": 0.93}}\n'
            )

    class _FakeS3:
        def get_object(self_inner, Bucket, Key):
            captured["bucket"] = Bucket
            captured["key"] = Key
            return {"Body": _FakeBody()}

    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: _FakeS3())
    rec = byo.read_generated_train_env(
        str(tmp_path / "nope"),
        envs_uri="s3://bucket/sim2real-b/run1/envs/train/envs.jsonl",
    )
    assert rec["env_id"] == "env-00006"
    assert rec["physics"]["friction"] == 0.71
    assert captured["bucket"] == "bucket"
    assert captured["key"] == "sim2real-b/run1/envs/train/envs.jsonl"


# --------------------------------------------------------------------------- #
# Outer-loop RESUME wiring (stage 11B "send back for more RL" must compound)
# --------------------------------------------------------------------------- #
def _resume_manifest(
    resume_uri="",
    physics=None,
    experiment_name=byo.DEFAULT_EXPERIMENT_NAME,
    resume_sha256="a" * 64,
):
    return byo.build_isaac_job_manifest(
        job_name="j",
        run_id="r",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=512,
        iterations=30,
        s3_output_uri="s3://b/o/",
        s3_endpoint="https://s3",
        namespace="default",
        service_account="agent-sa",
        gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        seed=42,
        resume_uri=resume_uri,
        resume_sha256=resume_sha256 if resume_uri else "",
        physics=physics,
        experiment_name=experiment_name,
    )


def test_manifest_resume_downloads_prior_checkpoint_and_passes_flags():
    uri = "s3://b/sim2real-b/run/byo-trainer/job/outer-01-iter-01/model_latest.pt"
    args = _manifest_script(_resume_manifest(resume_uri=uri))
    # downloads the prior checkpoint into the rsl_rl log dir train.py searches
    assert f"RESUME_FROM: {uri}" in args
    assert f"logs/rsl_rl/{byo.DEFAULT_EXPERIMENT_NAME}/{byo.RESUME_RUN_DIR}" in args
    assert "npa.workflows.sim2real.isaac_job_io download" in args
    assert f"--sha256 {'a' * 64}" in args
    assert "s3.download_file" not in args
    # tells train.py to resume that exact staged run
    assert "agent.resume=true" in args
    assert f"agent.load_run={byo.RESUME_RUN_DIR}" in args
    assert f"agent.load_checkpoint={byo.RESUME_CKPT_NAME}" in args
    env = {
        item["name"]: item["value"]
        for item in _resume_manifest(resume_uri=uri)["spec"]["template"]["spec"][
            "containers"
        ][0]["env"]
    }
    assert env["NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM"] == "0"


def test_manifest_resume_download_is_fail_closed_before_trainer_capture() -> None:
    uri = "s3://b/o/model_latest.pt"
    args = _manifest_script(_resume_manifest(resume_uri=uri))
    download = args.index("npa.workflows.sim2real.isaac_job_io download")
    trainer_capture = args.index("set +e")
    assert args.startswith("set -euo pipefail\n")
    assert download < trainer_capture


def test_byo_robot_staging_is_fail_closed_before_trainer_capture() -> None:
    args = _manifest_script(
        byo.build_isaac_job_manifest(
            job_name="j",
            run_id="r",
            image="reg/npa-isaac-lab:2.3.2.post1",
            task="Isaac-Lift-Cube-Franka-v0",
            num_envs=64,
            iterations=2,
            s3_output_uri="s3://b/o/",
            s3_endpoint="https://s3",
            namespace="default",
            service_account="agent-sa",
            gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
            robot_spec={"robot_source": "stock_franka", "name": "franka"},
            resume_uri="s3://b/o/model_latest.pt",
            resume_sha256="a" * 64,
            scenarios_uri="s3://b/o/train.jsonl",
            scenarios_sha256="a" * 64,
        )
    )
    resume = args.index("ROBOT_RESUME_FROM")
    scenario = args.index("--destination /tmp/npa_robot/scenarios.jsonl")
    trainer_capture = args.index("set +e")
    assert args.startswith("set -euo pipefail\n")
    assert scenario < trainer_capture
    assert resume < trainer_capture


def test_manifest_no_resume_keeps_default_path_unchanged():
    args = _manifest_script(_resume_manifest(resume_uri=""))
    assert "RESUME_FROM" not in args
    assert "agent.resume" not in args
    assert "s3.download_file" not in args  # only the upload tail uses boto3


def test_manifest_resume_requires_exact_checkpoint_digest() -> None:
    with pytest.raises(ValueError, match="resume_uri requires its exact SHA-256"):
        _resume_manifest(resume_uri="s3://b/o/model_latest.pt", resume_sha256="")


def test_manifest_resume_ignored_on_physics_path():
    # The physics variant trains a different task; resume must not be injected.
    uri = "s3://b/o/model_latest.pt"
    args = _manifest_script(
        _resume_manifest(resume_uri=uri, physics={"friction": 0.7, "mass_scale": 0.95})
    )
    assert "agent.resume" not in args
    assert "RESUME_FROM" not in args


def test_manifest_resume_honors_custom_experiment_name():
    args = _manifest_script(
        _resume_manifest(
            resume_uri="s3://b/o/model_latest.pt", experiment_name="my_robot_lift"
        )
    )
    assert f"logs/rsl_rl/my_robot_lift/{byo.RESUME_RUN_DIR}" in args


def test_sanitize_tag():
    assert byo._sanitize_tag("outer-02-iter-01") == "outer-02-iter-01"
    assert byo._sanitize_tag("a/b c") == "a-b-c"
    assert byo._sanitize_tag("--x--") == "x"
    assert byo._sanitize_tag("") == ""


def test_k8s_job_name_hashes_truncated_tail_to_avoid_run_collisions():
    shared = "sim2real-quality-gpu-20260804t172054z-6da8d6e3"
    first = byo.k8s_job_name("s2r-byo-isaac-roll", f"{shared}-first")
    second = byo.k8s_job_name("s2r-byo-isaac-roll", f"{shared}-second")

    assert len(first) <= 63
    assert len(second) <= 63
    assert first.startswith("s2r-byo-isaac-roll-")
    assert second.startswith("s2r-byo-isaac-roll-")
    assert first != second
    assert first == byo.k8s_job_name("s2r-byo-isaac-roll", f"{shared}-first")


def test_artifact_tag_from_output_dir_keeps_outer_iteration(tmp_path):
    output_dir = tmp_path / "actions" / "train" / "outer-02" / "iter-01"

    assert byo.artifact_tag_from_output_dir(output_dir) == "outer-02-iter-01"
    assert byo.artifact_tag_from_output_dir(tmp_path / "iter-01") == "iter-01"
    assert byo.artifact_tag("outer/02 iter 01") == "outer-02-iter-01"


def test_s3_object_sha256_streams_exact_checkpoint_bytes(monkeypatch) -> None:
    class Body:
        def __init__(self) -> None:
            self.chunks = iter((b"exact-", b"checkpoint", b""))

        def read(self, _size: int) -> bytes:
            return next(self.chunks)

        def close(self) -> None:
            return None

    class S3:
        def get_object(self, *, Bucket: str, Key: str):
            assert Bucket == "bucket"
            assert Key == "run/model.pt"
            return {"Body": Body()}

    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **k: S3())
    assert (
        byo.s3_object_sha256("s3://bucket/run/model.pt")
        == hashlib.sha256(b"exact-checkpoint").hexdigest()
    )


def test_run_isaac_training_job_tags_s3_path_per_iteration(monkeypatch):
    # NPA_SIM2REAL_TRAINER_TAG must make each iteration's checkpoint a DISTINCT S3
    # path (so the prior model survives for the next outer iteration to resume from
    # and outer iterations don't overwrite each other).
    captured = {}

    def fake_build(*args, **kwargs):
        captured["s3_output_uri"] = kwargs["s3_output_uri"]
        captured["resume_uri"] = kwargs.get("resume_uri", "")
        captured["resume_sha256"] = kwargs.get("resume_sha256", "")
        captured["scenarios_uri"] = kwargs.get("scenarios_uri", "")
        captured["scenarios_jsonl"] = kwargs.get("scenarios_jsonl", "")
        captured["scenarios_sha256"] = kwargs.get("scenarios_sha256", "")
        captured["entropy_coef"] = kwargs.get("entropy_coef", "")
        captured["entropy_final_coef"] = kwargs.get("entropy_final_coef", "")
        captured["entropy_anneal_fraction"] = kwargs.get("entropy_anneal_fraction", "")
        captured["ppo_optimizer_learning_rate"] = kwargs.get(
            "ppo_optimizer_learning_rate", ""
        )
        captured["convergence_action_noise_std"] = kwargs.get(
            "convergence_action_noise_std", ""
        )
        captured["success_termination_enabled"] = kwargs.get(
            "success_termination_enabled"
        )
        return {"manifest": True}

    monkeypatch.setattr(byo, "build_isaac_job_manifest", fake_build)
    monkeypatch.setattr(
        "npa.workflows.sim2real.k8s_client.KubernetesJobClient.from_environment",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "npa.workflows.sim2real.gpu_fallback.run_gpu_job_with_fallback",
        lambda **kwargs: {
            "job_name": kwargs["base_job_name"],
            "job_uid": "uid",
            "selected_product": "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
            "image_digests": ["reg/runtime@sha256:" + "a" * 64],
        },
    )
    monkeypatch.setattr(
        byo,
        "read_signal_stats",
        lambda *a, **k: {
            "mean_reward": 1.0,
            "step_count": 2,
            "nonzero_advantage_count": 2,
        },
    )
    scenario_rows = [
        {"difficulty": difficulty, "scenario_config_digest": f"cfg-{difficulty}"}
        for difficulty in ("easy", "medium", "hard")
    ]
    monkeypatch.setattr(
        byo,
        "read_generated_train_envs",
        lambda *a, **k: (scenario_rows, "{}\n"),
    )
    monkeypatch.setattr(
        byo,
        "_load_s3_json",
        lambda *a, **k: {
            "scenario_count": len(scenario_rows),
            "coverage_rate": 1.0,
        },
    )
    monkeypatch.setattr(
        byo,
        "_load_and_publish_ppo_telemetry",
        lambda *a, **k: {
            "telemetry_uri": "s3://bkt/ppo-telemetry.json",
            "raw_log_uri": "s3://bkt/train_full.log",
            "curves": [],
        },
    )
    monkeypatch.setattr(
        byo,
        "enumerate_periodic_checkpoints",
        lambda *a, **k: [
            {
                "training_iteration": 500,
                "checkpoint_uri": "s3://bkt/run/checkpoints/model_500.pt",
            }
        ],
    )
    monkeypatch.setenv("NPA_SIM2REAL_ISAAC_IMAGE", "reg/npa-isaac-lab:2.3.2.post1")
    monkeypatch.setenv("NPA_SIM2REAL_BUCKET", "bkt")
    monkeypatch.setenv("NPA_SIM2REAL_TRAINER_TAG", "outer-02-iter-01")
    monkeypatch.setenv(
        "NPA_SIM2REAL_TRAIN_ENVS_URI", "s3://bkt/myrun/envs/train/envs.jsonl"
    )
    monkeypatch.setenv("NPA_SIM2REAL_GPU_SCHEDULING_PROBE_SECONDS", "0")
    monkeypatch.setenv(
        "NPA_SIM2REAL_RESUME_CHECKPOINT_URI", "s3://bkt/prior/model_latest.pt"
    )
    monkeypatch.setenv("NPA_SIM2REAL_RESUME_CHECKPOINT_SHA256", "b" * 64)
    monkeypatch.delenv("NPA_BYO_ISAAC_PHYSICS", raising=False)

    result = byo.run_isaac_training_job("myrun", signal_json="ignored")
    # distinct, tagged path
    assert captured["s3_output_uri"].endswith("/outer-02-iter-01/")
    assert "byo-trainer" in captured["s3_output_uri"]
    # resume uri threaded through to the manifest builder
    assert captured["resume_uri"] == "s3://bkt/prior/model_latest.pt"
    assert captured["resume_sha256"] == "b" * 64
    assert captured["entropy_coef"] == byo.DEFAULT_RESUME_ENTROPY_COEF
    assert captured["entropy_final_coef"] == byo.DEFAULT_RESUME_ENTROPY_FINAL_COEF
    assert (
        captured["entropy_anneal_fraction"]
        == byo.DEFAULT_RESUME_ENTROPY_ANNEAL_FRACTION
    )
    assert (
        captured["ppo_optimizer_learning_rate"]
        == byo.DEFAULT_RESUME_PPO_OPTIMIZER_LEARNING_RATE
    )
    assert (
        captured["convergence_action_noise_std"]
        == byo.DEFAULT_RESUME_CONVERGENCE_ACTION_NOISE_STD
    )
    assert captured["success_termination_enabled"] is False
    assert captured["scenarios_uri"] == "s3://bkt/myrun/envs/train/envs.jsonl"
    assert captured["scenarios_jsonl"] == ""
    assert captured["scenarios_sha256"] == hashlib.sha256(b"{}\n").hexdigest()
    assert result["scenario_distribution"]["source_uri"] == (
        "s3://bkt/myrun/envs/train/envs.jsonl"
    )
    assert result["resume_checkpoint_uri"] == "s3://bkt/prior/model_latest.pt"
    assert result["resume_checkpoint_sha256"] == "b" * 64
    assert (
        result["scenario_distribution"]["source_sha256"]
        == hashlib.sha256(b"{}\n").hexdigest()
    )
    assert result["scenario_distribution"]["source_bytes"] == 3
    assert result["scenario_distribution"]["transport"] == "s3_sha256"
    # returned checkpoint points at the tagged path
    assert result["checkpoint_path"].endswith("/outer-02-iter-01/model_latest.pt")


def test_enumerate_periodic_checkpoints_uses_interval_and_keeps_final(
    monkeypatch,
) -> None:
    import boto3

    class Paginator:
        def paginate(self, **kwargs):
            assert kwargs["Prefix"].endswith("checkpoints/")
            return [
                {
                    "Contents": [
                        {"Key": f"run/checkpoints/model_{iteration}.pt"}
                        for iteration in (0, 25, 100, 200, 250)
                    ]
                }
            ]

    class Client:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return Paginator()

    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: Client())
    checkpoints = byo.enumerate_periodic_checkpoints(
        s3_output="s3://bucket/run/",
        endpoint="https://storage.example",
        validation_interval=100,
    )
    assert [item["training_iteration"] for item in checkpoints] == [100, 200, 250]
