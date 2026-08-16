from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"
SKILL_PATH = (
    ROOT / "skills" / "workflows" / "oss-solution-registry-onboard" / "SKILL.md"
)
CATALOG_PATH = ROOT / "docs" / "workbench" / "oss-solution-catalog.md"
WAN_INPUT_CONTRACT_PATH = (
    ROOT / "npa" / "docker" / "workbench" / "wan2-2" / "input_contract.py"
)
WAN_RUNTIME_REQUIREMENTS_PATH = (
    ROOT / "npa" / "docker" / "workbench" / "wan2-2" / "runtime-requirements.txt"
)
WAN_RUNTIME_SCRIPT_PATH = (
    ROOT / "npa" / "docker" / "workbench" / "wan2-2" / "wan_runtime.sh"
)
SOLUTION_SPECS = sorted(
    path for path in WORKFLOW_DIR.glob("byof-*.yaml") if path.name != "byof.yaml"
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
        "capability_name": "pi05_droid_jointpos_polaris_served_infer",
        "smoke_artifact_name": "openpi_pi05_droid_jointpos_polaris_inference.json",
        "spec": "byof-openpi.yaml",
        "must_exercise": [
            "pi05_droid_jointpos_polaris_checkpoint_download",
            "pi05_droid_jointpos_polaris_direct_infer",
            "pi05_droid_jointpos_polaris_served_infer",
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


def _load_wan_input_contract():
    spec = importlib.util.spec_from_file_location(
        "npa_wan_input_contract_test", WAN_INPUT_CONTRACT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_openpi_polaris_contract_is_runtime_only_and_position_targeted() -> None:
    config = _load_config(WORKFLOW_DIR / "byof-openpi.yaml")
    build = str(config["build_command"])
    smoke = str(config["smoke_command"])
    spec_text = (WORKFLOW_DIR / "byof-openpi.yaml").read_text(encoding="utf-8")

    assert config["repo_ref"] == "15a9616a00943ada6c20a0f158e3adb39df2ccac"
    assert config["resource_profile_yaml"] == "byof-solution-smoke-openpi-b200-gpu"
    assert config["capability_name"] == "pi05_droid_jointpos_polaris_served_infer"
    assert config["wait_timeout"] == -1
    assert (
        config["smoke_artifact_name"]
        == "openpi_pi05_droid_jointpos_polaris_inference.json"
    )
    assert "-arch=sm_100" in build
    assert "/opt/venv/bin/uv pip install" in build
    assert "--no-cache -e ." in build
    assert "pi05_droid_jointpos_polaris" not in build
    assert "openpi-assets/checkpoints" not in build
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES" not in spec_text
    assert '"weights_baked": False' in smoke
    assert "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris" in smoke
    assert 'download.maybe_download(CHECKPOINT_URI, token="anon")' in smoke
    assert "jax_backend.get_backend().platform_version" in smoke
    assert "WebsocketPolicyServer" in smoke
    assert "WebsocketClientPolicy" in smoke
    assert "policy.infer(observation)" in smoke
    assert "actions.shape[0] < 5" in smoke and "actions.shape[1] != 8" in smoke
    assert "np.isfinite(actions).all()" in smoke
    assert "joint_position_targets_dims_0_6_radians" in smoke
    assert "execute_about_5_targets_at_15_hz_then_requery" in smoke
    assert "deterministic_transport_smoke_only" in smoke
    assert "NPA_OPENPI_ACCEPT_GEMMA_TERMS" not in smoke
    assert "npa_build_metadata.json" in smoke
    assert 'build_metadata.get("build_command_executed") is not True' in smoke
    assert 'actions.dtype != np.dtype("float64")' in smoke
    assert "cuobjdump" in smoke
    assert "check=True" in smoke
    assert "len(nvidia_smi) != 1" in smoke
    assert "compute_capability != (10, 0)" in smoke
    assert "value.is_integer() and value >= 100" in smoke
    assert 're.fullmatch(r"(?:sm_)?(\\d{1,3})"' in smoke
    assert '"live_validated": False' in smoke

    python_smoke = smoke.split("/opt/venv/bin/python - <<'PY'\n", 1)[1].rsplit(
        "\nPY", 1
    )[0]
    tree = ast.parse(python_smoke)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "normalized_compute_capability"
    )
    namespace = {"re": re}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            "<openpi-smoke>",
            "exec",
        ),
        namespace,
    )
    normalize = namespace["normalized_compute_capability"]
    for representation in (
        10,
        100,
        10.0,
        100.0,
        "10",
        "100",
        "10.0",
        "sm_100",
        (10, 0),
    ):
        assert normalize(representation) == (10, 0), representation


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
        assert expected["capability_name"] in text or expected["spec"] in text, solution
        assert f"byof-{solution}.yaml" in text or expected["spec"] in text


def test_oss_catalog_lists_solution_specific_capabilities() -> None:
    text = CATALOG_PATH.read_text(encoding="utf-8")
    assert "Native Capabilities Per Container" in text
    assert "shared taxonomy" in text.lower() or "solution-specific" in text.lower()
    for solution, expected in SOLUTION_CAPABILITY_CONTRACTS.items():
        assert expected["capability_name"] in text, solution
        assert expected["smoke_artifact_name"] in text, solution


def test_wan22_package_keeps_weights_runtime_only_and_claims_t2v_only() -> None:
    config = _load_config(WORKFLOW_DIR / "byof-wan2.2.yaml")
    build = str(config["build_command"])
    smoke = str(config["smoke_command"])
    spec_text = (WORKFLOW_DIR / "byof-wan2.2.yaml").read_text(encoding="utf-8")

    assert config["repo_ref"] == "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
    assert config["base_profile"] == "prebuilt"
    assert config["base_image"] == "tool://wan2-2"
    assert config["pip_extra"] == "viz"
    assert config["resource_profile_yaml"] == "byof-solution-smoke-wan22-rtxpro-gpu"
    assert config["wait_timeout"] == "-1"
    assert build == ""
    assert "snapshot_download" not in build
    assert "Wan-AI/" not in build
    assert "snapshot_download" in smoke
    assert "921dbaf3f1674a56f47e83fb80a34bac8a8f203e" in smoke
    assert '"weights_baked": False' in smoke
    assert "wan2_2_runtime_inventory.json" in smoke
    assert "large_checkpoint_shaped_files" in smoke
    assert "python_packages" in smoke and "os_packages" in smoke
    assert "importlib.metadata.distribution(package_name)" in smoke
    assert "python_package_names.setdefault" in smoke
    assert "wan-runtime ensure" in smoke
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
    assert "resolve_wan_input_contract" in smoke
    assert "capability_name=wan2.2_ti2v_5b_image_to_video" in spec_text
    assert "smoke_artifact_name=wan2_2_ti2v_5b_image_to_video.json" in spec_text
    assert "wan2_2_ti2v_5b_image_to_video.json" in smoke
    assert '"wan2.2_action_prediction"' in smoke
    assert "action_prediction" not in str(config["capability_name"])
    assert config["rrd_uri"].endswith("wan2_2_ti2v_5b.rrd")
    assert config["rrd_manifest_uri"].endswith("wan2_2_ti2v_5b_rrd_manifest.json")


@pytest.mark.parametrize(
    "filename",
    ["byof-wan2.2.yaml", "byof-wan2.2-multigpu.yaml"],
)
def test_wan22_prompt_is_not_interpolated_as_shell_syntax(filename: str) -> None:
    from npa.orchestration.npa_workflow import build_plan, load_spec

    spec = load_spec(WORKFLOW_DIR / filename)
    hostile = '"; echo WAN_PROMPT_INJECTION; $(touch /tmp/never) #'
    spec.config["prompt"] = hostile
    rendered = "\n".join(build_plan(spec, run_id="prompt-safety").steps[0].argv)

    assert hostile not in rendered
    assert "WAN_PROMPT_INJECTION" not in rendered
    assert "{{config.prompt" not in rendered


def test_wan22_context_image_config_contract_fails_closed() -> None:
    contract = _load_wan_input_contract()
    config = _load_config(WORKFLOW_DIR / "byof-wan2.2.yaml")

    selected = contract.resolve_wan_input_contract(
        context_image_uri=str(config["context_image_uri"]),
        declared_capability=str(config["capability_name"]),
        declared_artifact=str(config["smoke_artifact_name"]),
    )
    assert selected is contract.TEXT_TO_VIDEO

    with pytest.raises(contract.WanInputContractError, match="must be an s3://"):
        contract.resolve_wan_input_contract(
            context_image_uri="https://example.invalid/context.png",
            declared_capability=contract.IMAGE_TO_VIDEO.capability_name,
            declared_artifact=contract.IMAGE_TO_VIDEO.artifact_name,
        )
    with pytest.raises(contract.WanInputContractError, match="contract disagree"):
        contract.resolve_wan_input_contract(
            context_image_uri="s3://project-bucket/inputs/context.png",
            declared_capability=str(config["capability_name"]),
            declared_artifact=str(config["smoke_artifact_name"]),
        )

    selected = contract.resolve_wan_input_contract(
        context_image_uri="s3://project-bucket/inputs/context.png",
        declared_capability=contract.IMAGE_TO_VIDEO.capability_name,
        declared_artifact=contract.IMAGE_TO_VIDEO.artifact_name,
    )
    assert selected is contract.IMAGE_TO_VIDEO


def test_wan22_multigpu_uses_the_pinned_official_distributed_path() -> None:
    config = _load_config(WORKFLOW_DIR / "byof-wan2.2-multigpu.yaml")
    smoke = str(config["smoke_command"])
    payload = yaml.safe_load(
        (WORKFLOW_DIR / "byof-wan2.2-multigpu.yaml").read_text(encoding="utf-8")
    )
    runtime_requirements = WAN_RUNTIME_REQUIREMENTS_PATH.read_text(encoding="utf-8")
    runtime_script = WAN_RUNTIME_SCRIPT_PATH.read_text(encoding="utf-8")

    assert config["repo_ref"] == "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
    assert config["resource_profile_yaml"] == "byof-solution-smoke-wan22-b200-4gpu"
    assert config["base_profile"] == "prebuilt"
    assert config["base_image"] == "tool://wan2-2"
    assert config["pip_extra"] == "viz"
    assert config["wait_timeout"] == "-1"
    assert "wan-runtime ensure" in smoke
    assert "nvidia-nccl-cu13==2.29.7" in runtime_requirements
    assert '"nvidia-nccl-cu13": "2.29.7"' in runtime_script
    assert 'item["loaded_nccl"]["version_code"] != 22907' in smoke
    assert 'item["nccl_build_api_version"] != [2, 29, 7]' in smoke
    assert "--ignore-installed --no-deps" in runtime_script
    assert "nvidia-nccl-cu12" not in runtime_requirements
    assert "--nproc_per_node=4" in smoke
    assert "--dit_fsdp --t5_fsdp --ulysses_size 4" in smoke
    assert 'runpy.run_path("/opt/byof/generate.py"' in smoke
    assert "ShardingStrategy.FULL_SHARD" in smoke
    assert "/opt/byof/.venv/bin/python -m torch.distributed.run" in smoke
    assert "/opt/byof/.venv/bin/torchrun" not in smoke
    assert 'export NCCL_CUMEM_ENABLE="0"' in smoke
    assert 'export NCCL_CUMEM_HOST_ENABLE="0"' in smoke
    assert 'export NCCL_NVLS_ENABLE="0"' in smoke
    assert 'export NCCL_SOCKET_IFNAME="=eth0"' in smoke
    assert 'export NCCL_SOCKET_FAMILY="AF_INET"' in smoke
    assert 'export NCCL_IB_DISABLE="1"' in smoke
    assert 'export TORCH_NCCL_USE_COMM_NONBLOCKING="1"' in smoke
    assert 'export NCCL_DEBUG="INFO"' in smoke
    assert 'export NCCL_DEBUG_SUBSYS="INIT,COLL,ENV"' in smoke
    assert "wan2_2_multigpu_nccl_rank_{rank}.log" in smoke
    assert "wan2_2_multigpu_nccl_summary.json" in smoke
    assert "process_group_destroyed" in smoke
    assert 'item["nccl_cumem_enable"] != "0"' in smoke
    assert 'item["nccl_cumem_host_enable"] != "0"' in smoke
    assert 'item["nccl_nvls_enable"] != "0"' in smoke
    assert 'item["nccl_socket_ifname"] != "=eth0"' in smoke
    assert 'item["nccl_socket_family"] != "AF_INET"' in smoke
    assert 'item["nccl_ib_disable"] != "1"' in smoke
    assert 'item["torch_nccl_use_comm_nonblocking"] != "1"' in smoke
    for progress_stage in (
        "wrapper_started",
        "upstream_modules_imported",
        "process_group_init_started",
        "process_group_initialized",
        "nccl_probe_started",
        "nccl_probe_completed",
        "process_group_destroyed",
    ):
        assert progress_stage in smoke
    assert 'sys.path.insert(0, "/opt/byof")' in smoke
    assert "ulysses.flash_attention = attention_module.attention" in smoke
    assert "attention_module.flash_attention = attention_module.attention" not in smoke
    assert "Ulysses is not bound to Wan native PyTorch SDPA" in smoke
    assert "sed -i" not in smoke
    assert "ulysses_all_to_all_calls" in smoke
    assert "all_gather_object" in smoke
    assert "observer_final_barrier" in smoke
    assert "uuid_sha256" in smoke and '"uuid"' not in smoke.lower()
    assert '"sm_100"' in smoke and "[10, 0]" in smoke
    assert "ffprobe" in smoke and 'ffprobe != "h264"' in smoke
    assert "wan2_2_multigpu_topology.json" in smoke
    assert '"nvidia-nccl-cu13"' in smoke
    assert "snapshot_download" not in str(config["build_command"])
    assert config["build_command"] == ""
    assert '"weights_baked": False' in smoke
    assert payload["resources"]["gpu"]["accelerators"] == "B200:4"
    assert payload["resources"]["gpu"]["memory"] == "256Gi"
    assert config["rrd_uri"].endswith("wan2_2_ti2v_5b_multigpu.rrd")
    assert config["rrd_manifest_uri"].endswith(
        "wan2_2_ti2v_5b_multigpu_rrd_manifest.json"
    )
