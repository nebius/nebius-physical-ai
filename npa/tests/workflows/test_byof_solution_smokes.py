from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
SKILL_PATH = ROOT / "skills" / "workflows" / "oss-solution-registry-onboard" / "SKILL.md"
CATALOG_PATH = ROOT / "docs" / "workbench" / "oss-solution-catalog.md"
SOLUTION_SPECS = sorted(
    path
    for path in WORKFLOW_DIR.glob("byof-*.yaml")
    if path.name != "byof.yaml"
)

# Primary capability contracts for onboarded and pending-live solution candidates
# (solution-specific ids; catalog status remains authoritative).
# Keep in sync with skills/workflows/oss-solution-registry-onboard/SKILL.md
# and docs/workbench/oss-solution-catalog.md.
SOLUTION_CAPABILITY_CONTRACTS = {
    "maniskill": {
        "capability_name": "gymnasium_pickcube_registration",
        "smoke_artifact_name": "maniskill_pickcube_step.json",
        "spec": "byof-maniskill.yaml",
        "must_exercise": [
            "gymnasium_pickcube_registration",
            "pickcube_cpu_step",
            "pickcube_parallel_envs",
        ],
    },
    "mujoco-playground": {
        "capability_name": "mjx_cartpole_step",
        "smoke_artifact_name": "mujoco_playground_cartpole_step.json",
        "spec": "byof-mujoco-playground.yaml",
        "must_exercise": [
            "mjx_cartpole_step",
            "mjx_cheetah_run_step",
            "train_jax_ppo_cartpole_smoke",  # attempted; may remain deferred
        ],
    },
    "robocasa": {
        "capability_name": "kitchen_task_registration",
        "smoke_artifact_name": "robocasa_kitchen_env_reset.json",
        "spec": "byof-robocasa.yaml",
        "must_exercise": [
            "kitchen_task_registration",
            "download_kitchen_assets_lw",
            "kitchen_egl_env_reset",
        ],
    },
    "openpi": {
        "capability_name": "policy_config_materialization",
        "smoke_artifact_name": "openpi_pi05_droid_config.json",
        "spec": "byof-openpi.yaml",
        "must_exercise": [
            "policy_config_materialization",
            "pi05_droid_checkpoint_download",
            "pi05_droid_checkpoint_infer",
        ],
    },
    "droid-policy-learning": {
        "capability_name": "rlds_config_generator_contract",
        "smoke_artifact_name": "droid_rlds_config_generator.json",
        "spec": "byof-droid-policy-learning.yaml",
        "must_exercise": [
            "rlds_config_generator_contract",
            "droid_100_download",
            "droid_100_config_gen",
        ],
    },
    "open-dreamer": {
        "capability_name": "dreamer4_tokenizer_train_two_gpu",
        "smoke_artifact_name": "open_dreamer_world_model_2gpu.json",
        "spec": "byof-open-dreamer.yaml",
        "must_exercise": [
            "jax_two_gpu_data_parallel_mesh",
            "minecraft_vpt_video_dataloader",  # real Minecraft/VPT gameplay
            "dreamer4_tokenizer_train_two_gpu",
            "dreamer4_latent_tokenization",  # tokenize_minecraft_dataset.py -> latents + stats
            "dreamer4_dynamics_train_two_gpu",
            "dreamer4_action_conditioned_dream_rollout",  # the headline dream
            "world_model_rerun_visualization",  # 3-stream Rerun .rrd artifact
        ],
    },
    "wan2.2": {
        "capability_name": "wan2.2_ti2v_5b_text_to_video",
        "smoke_artifact_name": "wan2_2_ti2v_5b_text_to_video.json",
        "spec": "byof-wan2.2.yaml",
        "must_exercise": [
            "wan2.2_ti2v_5b_text_to_video",
            "wan2.2_decoded_mp4_validation",
        ],
    },
    "wan2.2-multigpu": {
        "capability_name": "wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses",
        "smoke_artifact_name": "wan2_2_ti2v_5b_multigpu.json",
        "spec": "byof-wan2.2-multigpu.yaml",
        "documented": True,
        "must_exercise": [
            "wan2.2_ti2v_5b_text_to_video_multigpu_fsdp_ulysses",
            "wan2.2_distributed_rank_topology_validation",
            "wan2.2_decoded_mp4_validation",
        ],
    },
}


def _load_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), path
    config = payload.get("config")
    assert isinstance(config, dict), path
    return config


def test_byof_solution_specs_have_capability_smokes() -> None:
    assert SOLUTION_SPECS, "expected BYOF solution candidate specs"
    for path in SOLUTION_SPECS:
        config = _load_config(path)
        assert config.get("workload") == "solution-smoke", path.name
        assert str(config.get("solution_name") or "").strip(), path.name
        assert str(config.get("capability_name") or "").strip(), path.name
        artifact = str(config.get("smoke_artifact_name") or "").strip()
        smoke = str(config.get("smoke_command") or "")
        assert artifact.endswith(".json"), path.name
        assert "NPA_SMOKE_OUTPUT_DIR" in smoke, path.name
        assert artifact in smoke, path.name


def test_byof_solution_smokes_are_not_import_only() -> None:
    for path in SOLUTION_SPECS:
        smoke = str(_load_config(path).get("smoke_command") or "")
        assert ".write_text(" in smoke, path.name
        assert "json.dumps(" in smoke, path.name
        assert '"capability"' in smoke or "'capability'" in smoke, path.name
        assert '"solution"' in smoke or "'solution'" in smoke, path.name
        assert "capabilities_exercised" in smoke, path.name


def test_solution_capability_contracts_match_specs() -> None:
    by_solution = {
        str(_load_config(path).get("solution_name")): path for path in SOLUTION_SPECS
    }
    assert set(by_solution) == set(SOLUTION_CAPABILITY_CONTRACTS)
    for solution, expected in SOLUTION_CAPABILITY_CONTRACTS.items():
        path = by_solution[solution]
        config = _load_config(path)
        assert path.name == expected["spec"]
        assert config.get("capability_name") == expected["capability_name"]
        assert config.get("smoke_artifact_name") == expected["smoke_artifact_name"]
        smoke = str(config.get("smoke_command") or "")
        assert expected["capability_name"] in smoke
        assert expected["smoke_artifact_name"] in smoke
        for capability in expected["must_exercise"]:
            assert capability in smoke, (solution, capability)


def test_registry_skill_is_solution_specific_not_taxonomy() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "Do **not** force capabilities into a shared taxonomy" in text
    assert "Capability Testing Built Into Onboarding" in text
    assert "Capability Families (required taxonomy)" not in text
    for solution, expected in SOLUTION_CAPABILITY_CONTRACTS.items():
        if expected.get("documented", True) is False:
            continue
        assert expected["capability_name"] in text or expected["spec"] in text, solution
        assert f"byof-{solution}.yaml" in text or expected["spec"] in text


def test_oss_catalog_lists_solution_specific_capabilities() -> None:
    text = CATALOG_PATH.read_text(encoding="utf-8")
    assert "Native Capabilities Per Container" in text
    assert "shared taxonomy" in text.lower() or "solution-specific" in text.lower()
    for solution, expected in SOLUTION_CAPABILITY_CONTRACTS.items():
        if expected.get("documented", True) is False:
            continue
        assert expected["capability_name"] in text, solution
        assert expected["smoke_artifact_name"] in text, solution


def test_wan22_package_keeps_weights_runtime_only_and_claims_t2v_only() -> None:
    config = _load_config(WORKFLOW_DIR / "byof-wan2.2.yaml")
    build = str(config["build_command"])
    smoke = str(config["smoke_command"])

    assert config["repo_ref"] == "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
    assert config["base_image"] == "ubuntu:22.04"
    assert (
        config["resource_profile_yaml"]
        == "byof-solution-smoke-wan22-rtxpro-gpu"
    )
    assert config["wait_timeout"] == "0"
    assert "snapshot_download" not in build
    assert "Wan-AI/" not in build
    assert "snapshot_download" in smoke
    assert "921dbaf3f1674a56f47e83fb80a34bac8a8f203e" in smoke
    assert '"weights_baked": False' in smoke
    assert "wan2_2_runtime_inventory.json" in smoke
    assert "large_checkpoint_shaped_files" in smoke
    assert "python_packages" in smoke and "os_packages" in smoke
    assert "chmod -R a+rX /opt/byof/.venv /opt/byof" in build
    assert "flash_attn" not in build
    assert "from .attention import attention as flash_attention" in build
    assert "py_compile.compile('/opt/byof/wan/textimage2video.py', doraise=True)" in build
    assert "from wan.textimage2video import WanTI2V" not in build
    assert "torch.cuda.current_device" not in build
    assert "scaled_dot_product_attention" in smoke
    assert '"sm_120" not in torch_cuda_arch_list' in smoke
    assert 'devices[0]["compute_capability"] != [12, 0]' in smoke
    assert '"driver_versions": driver_versions' in smoke
    assert '"runtime_stack"' in smoke
    assert "Wan RTX PRO baseline must use native PyTorch SDPA" in smoke
    assert "wan_model.flash_attention is wan_attention.attention" in smoke
    assert '"sdpa_source_binding": sdpa_source_binding' in smoke
    assert "generator.generate(" in smoke and "save_video(" in smoke
    assert "cv2.VideoCapture" in smoke
    assert "frames are temporally uniform" in smoke
    assert "wan2.2_ti2v_5b_image_to_video (pending separate live" in smoke
    assert "Wan input mode and declared BYOF contract disagree" in smoke
    assert "wan2_2_ti2v_5b_image_to_video.json" in smoke
    assert '"bellboy_private_action_prediction"' in smoke
    assert "action_prediction" not in str(config["capability_name"])


def test_wan22_zero_wait_reaches_the_terminal_state_without_a_hidden_cap() -> None:
    repo_runner = (ROOT / "npa" / "scripts" / "run_byof_repo.py").read_text(
        encoding="utf-8"
    )
    verify_runner = (
        ROOT / "npa" / "scripts" / "run_byof_container_verify.py"
    ).read_text(encoding="utf-8")

    assert "str(min(args.wait_timeout, 3600))" not in repo_runner
    assert "str(args.wait_timeout)" in repo_runner
    assert "None if args.wait_timeout <= 0" in verify_runner
    assert "deadline is None or time.time() < deadline" in verify_runner


def test_wan22_multigpu_uses_the_pinned_official_distributed_path() -> None:
    config = _load_config(WORKFLOW_DIR / "byof-wan2.2-multigpu.yaml")
    smoke = str(config["smoke_command"])
    payload = yaml.safe_load(
        (WORKFLOW_DIR / "byof-wan2.2-multigpu.yaml").read_text(encoding="utf-8")
    )

    assert config["repo_ref"] == "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
    assert config["resource_profile_yaml"] == "byof-solution-smoke-wan22-b200-4gpu"
    assert config["wait_timeout"] == "0"
    assert "--nproc_per_node=4" in smoke
    assert "--dit_fsdp --t5_fsdp --ulysses_size 4" in smoke
    assert "runpy.run_path(\"/opt/byof/generate.py\"" in smoke
    assert "ShardingStrategy.FULL_SHARD" in smoke
    assert "ulysses_all_to_all_calls" in smoke
    assert "all_gather_object" in smoke
    assert "observer_final_barrier" in smoke
    assert "uuid_sha256" in smoke and '"uuid"' not in smoke.lower()
    assert '"sm_100"' in smoke and "[10, 0]" in smoke
    assert "ffprobe" in smoke and 'ffprobe != "h264"' in smoke
    assert "wan2_2_multigpu_topology.json" in smoke
    assert "snapshot_download" not in str(config["build_command"])
    assert '"weights_baked": False' in smoke
    assert payload["resources"]["gpu"]["accelerators"] == "B200:4"
    assert payload["resources"]["gpu"]["memory"] == "256Gi"
