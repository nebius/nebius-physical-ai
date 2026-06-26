"""Tests for the BYO Isaac-Lab RSL-RL trainer (real RL for the sim2real loop).

The headline guarantee: the dry-run output satisfies the real
``VlmSignalUpdateResult.from_dict`` contract that ``_run_trainer_via_command``
enforces, and the live path builds a correct Isaac training Job manifest.
"""

from __future__ import annotations

import json

import pytest

from npa.workflows.sim2real import byo_isaac_trainer as byo


def _write_signal(tmp_path):
    path = tmp_path / "signal.json"
    path.write_text(
        json.dumps(
            {
                "schema": "npa.sim2real.rl_signal.v1",
                "signals": [
                    {"per_step": [{"reward": 0.6, "advantage": 0.2},
                                  {"reward": 0.4, "advantage": -0.1}]},
                    {"per_step": [{"reward": 0.8, "advantage": 0.3}]},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_read_signal_stats(tmp_path):
    stats = byo.read_signal_stats(str(_write_signal(tmp_path)))
    assert stats["step_count"] == 3
    assert stats["mean_reward"] == pytest.approx((0.6 + 0.4 + 0.8) / 3)
    assert stats["mean_advantage"] == pytest.approx((0.2 - 0.1 + 0.3) / 3)


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
        checkpoint_uri="s3://bucket/run/model_latest.pt",
        status="success",
        duration_ms=1234.0,
    )
    # Required contract fields present + non-empty policy_output_after.
    assert result["reward_head_after"] != 0.0
    assert isinstance(result["policy_output_after"], list) and result["policy_output_after"]
    assert result["policy_delta_l2"] > 0.0  # a real trainer produced a checkpoint
    assert result["backend"] == "isaac_rsl_rl_ppo"
    assert result["checkpoint_path"].endswith("model_latest.pt")
    parsed = VlmSignalUpdateResult.from_dict(result)
    assert parsed.checkpoint_path == "s3://bucket/run/model_latest.pt"
    assert parsed.backend == "isaac_rsl_rl_ppo"


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
    args = container["args"][0]
    assert "Isaac-Lift-Cube-Franka-v0" in args
    assert "--max_iterations 150" in args
    assert "--num_envs 1024" in args
    assert byo.TRAIN_SCRIPT in args
    assert "model_latest.pt" in args  # uploads the trained checkpoint


def test_dryrun_main_writes_contract_json(tmp_path, monkeypatch):
    from npa.workbench.lerobot.policy_container import VlmSignalUpdateResult

    out = tmp_path / "update.json"
    monkeypatch.setenv("NPA_BYO_ISAAC_DRYRUN", "1")
    monkeypatch.setenv("NPA_SIM2REAL_SIGNAL_JSON", str(_write_signal(tmp_path)))
    monkeypatch.setenv("NPA_SIM2REAL_OUTPUT_JSON", str(out))
    monkeypatch.setenv("NPA_BYO_ISAAC_ITERATIONS", "3")
    rc = byo.main()
    assert rc == 0
    payload = json.loads(out.read_text())
    parsed = VlmSignalUpdateResult.from_dict(payload)  # must not raise
    assert parsed.backend == "isaac_rsl_rl_ppo"
    assert parsed.steps == 3


def test_vlm_reward_overrides_targets_error_tag_term():
    # VLM says reaching is failing -> reaching_object weight boosted above default 1.0.
    stats = {"mean_reward": 0.2, "mean_advantage": 0.0, "step_count": 5,
             "error_tags": {"did_not_reach_object": 4, "minor": 1}}
    ov = byo.vlm_reward_overrides(stats)
    assert ov["env.rewards.reaching_object.weight"] > 1.0
    # untouched term stays at its default weight
    assert ov["env.rewards.lifting_object.weight"] == 15.0


def test_vlm_reward_overrides_low_reward_broadly_boosts_and_is_bounded():
    stats = {"mean_reward": -1.0, "mean_advantage": 0.0, "step_count": 3, "error_tags": {}}
    ov = byo.vlm_reward_overrides(stats)
    # broad boost applied (mult>1) but bounded to <= 2x default
    assert ov["env.rewards.lifting_object.weight"] > 15.0
    assert ov["env.rewards.lifting_object.weight"] <= 30.0
    assert ov["env.rewards.reaching_object.weight"] <= 2.0


def test_manifest_embeds_reward_overrides():
    ov = {"env.rewards.reaching_object.weight": 1.6, "env.rewards.lifting_object.weight": 15.0}
    m = byo.build_isaac_job_manifest(
        job_name="j", run_id="r", image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0", num_envs=512, iterations=30,
        s3_output_uri="s3://b/o/", s3_endpoint="https://s3", namespace="default",
        service_account="agent-sa", gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        reward_overrides=ov)
    args = m["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "env.rewards.reaching_object.weight=1.6" in args


def test_manifest_embeds_custom_object_usd():
    m = byo.build_isaac_job_manifest(
        job_name="j", run_id="r", image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0", num_envs=512, iterations=30,
        s3_output_uri="s3://b/o/", s3_endpoint="https://s3", namespace="default",
        service_account="agent-sa", gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        object_usd="s3orhttp://assets/custom_sugar_box.usd", object_scale="(0.8, 0.8, 0.8)")
    args = m["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "env.scene.object.spawn.usd_path=s3orhttp://assets/custom_sugar_box.usd" in args
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
        job_name="j", run_id="r", image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0", num_envs=512, iterations=30,
        s3_output_uri="s3://b/o/", s3_endpoint="https://s3", namespace="default",
        service_account="agent-sa", gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        seed=516456434)
    args = m["spec"]["template"]["spec"]["containers"][0]["args"][0]
    # generated env seed drives randomization via train.py --seed (NOT a hydra
    # env.seed= override, which the Lift cfg rejects as a type error).
    assert "--seed 516456434" in args
    assert "env.seed=" not in args


def test_manifest_no_seed_arg_when_zero():
    m = byo.build_isaac_job_manifest(
        job_name="j", run_id="r", image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0", num_envs=512, iterations=30,
        s3_output_uri="s3://b/o/", s3_endpoint="https://s3", namespace="default",
        service_account="agent-sa", gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        seed=0)
    args = m["spec"]["template"]["spec"]["containers"][0]["args"][0]
    assert "--seed" not in args


def test_manifest_physics_path_ships_wrapper_and_skips_stock_train():
    m = byo.build_isaac_job_manifest(
        job_name="j", run_id="r", image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0", num_envs=64, iterations=2,
        s3_output_uri="s3://b/o/", s3_endpoint="https://s3", namespace="default",
        service_account="agent-sa", gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        seed=736958930, physics={"friction": 0.7, "mass_scale": 0.95})
    args = m["spec"]["template"]["spec"]["containers"][0]["args"][0]
    # ships the module + wrapper, sets the generated physics, runs the wrapper
    assert "isaac_physics_task.py" in args and "runner.py" in args
    assert "NPA_GEN_FRICTION=0.7" in args and "NPA_GEN_MASS_SCALE=0.95" in args
    assert "PHYS_SEED=736958930" in args
    assert "/tmp/npa_phys/runner.py" in args
    # physics path does NOT invoke stock train.py
    assert byo.TRAIN_SCRIPT not in args
    # still uploads model_latest.pt via the shared tail
    assert "model_latest.pt" in args


def test_manifest_default_path_unchanged_without_physics():
    m = byo.build_isaac_job_manifest(
        job_name="j", run_id="r", image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0", num_envs=512, iterations=30,
        s3_output_uri="s3://b/o/", s3_endpoint="https://s3", namespace="default",
        service_account="agent-sa", gpu_product="NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
        seed=42)
    args = m["spec"]["template"]["spec"]["containers"][0]["args"][0]
    # proven path: stock train.py, no physics wrapper
    assert byo.TRAIN_SCRIPT in args
    assert "isaac_physics_task.py" not in args
    assert "--seed 42" in args


def test_read_generated_train_env_s3_fallback(tmp_path, monkeypatch):
    # Local dir missing -> falls back to the S3 URI (orchestrator only syncs heldout).
    captured = {}

    class _FakeBody:
        def read(self_inner):
            return (b'{"env_id": "env-00006", "seed": 99, '
                    b'"physics": {"friction": 0.71, "mass_scale": 0.93}}\n')

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
