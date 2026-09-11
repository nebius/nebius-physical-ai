from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = ROOT / "workflows" / "testing" / "byof-libero.yaml"
READINESS_PATH = ROOT / "workflows" / "testing" / "byof-libero.readiness.json"
DOC_PATH = ROOT / "docs" / "workbench" / "byof-libero.md"
PROFILE_PATH = (
    ROOT
    / "npa"
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "byof-solution-smoke-libero-b200-gpu.yaml"
)
BYOF_RUNNER_PATH = ROOT / "npa" / "scripts" / "run_byof_repo.py"

SOURCE_REF = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
DATASET_REF = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
DATASET_SHA256 = "ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead"
BASE_IMAGE_SHA256 = "ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85"
LANGUAGE_MODEL_REF = "cd5ef92a9fb2f889e972770a36d4ed042daf221e"
LANGUAGE_MODEL_MODEL_SHA256 = (
    "d6992b8cd27d7a132eafce6a8210272329a371b1c762d453588795dd3835593e"
)
CAPABILITY = "libero_spatial_bc_rnn_train_reload_heldout"
PAYLOAD_SERVICE_ACCOUNT = "npa-byof-libero-payload"


def _workflow() -> dict[str, object]:
    payload = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _config() -> dict[str, object]:
    config = _workflow()["config"]
    assert isinstance(config, dict)
    return config


def _smoke_source() -> str:
    smoke = str(_config()["smoke_command"])
    source = smoke.split("/opt/venv/bin/python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    ast.parse(source)
    return source


def _expected_source_metadata_keys() -> set[str]:
    tree = ast.parse(_smoke_source())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not node.comparators:
            continue
        if not isinstance(node.left, ast.Name) or node.left.id != "source_metadata":
            continue
        expected = node.comparators[0]
        if isinstance(expected, ast.Dict):
            return {
                key.value
                for key in expected.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise AssertionError("source_metadata equality contract is missing")


def _generated_public_source_metadata_keys() -> set[str]:
    spec = importlib.util.spec_from_file_location("run_byof_repo_libero", BYOF_RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    dockerfile = module._dockerfile_text()
    line = next(line for line in dockerfile.splitlines() if '"source": "oss-byof"' in line)
    return set(re.findall(r'"([a-z_]+)"\s*:', line))


def test_libero_workflow_pins_reviewed_source_data_and_base_image() -> None:
    config = _config()
    build = str(config["build_command"])
    smoke = _smoke_source()

    assert config["repo_url"] == "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
    assert config["repo_ref"] == SOURCE_REF
    assert config["base_profile"] == "ubuntu"
    assert config["base_image"] == (
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:"
        + BASE_IMAGE_SHA256
    )
    assert config["source_prune_path"] == "libero/libero/assets"
    assert config["resource_profile_yaml"] == "byof-solution-smoke-libero-b200-gpu"
    assert config["capability_name"] == CAPABILITY
    assert config["smoke_artifact_name"] == "libero-smoke.json"
    assert config["wait_timeout"] == -1
    assert config["output_root"] == "s3://{{config.bucket}}/oss-solutions/libero"
    assert config["smoke_uri"] == (
        "{{config.output_root}}/{{run.id}}/libero-smoke.json"
    )

    assert DATASET_REF not in build
    assert DATASET_SHA256 not in build
    assert LANGUAGE_MODEL_REF not in build
    assert LANGUAGE_MODEL_MODEL_SHA256 not in build
    assert "huggingface.co/datasets" not in build
    assert "download_libero_datasets" not in build
    assert "rm -rf libero/libero/assets" not in build
    assert DATASET_REF in smoke
    assert DATASET_SHA256 in smoke
    assert "expected_size_bytes" in smoke
    assert '"license": "CC-BY-4.0"' in smoke
    assert '"license": "MIT"' in smoke
    assert "download_verified(dataset_path)" in smoke
    assert "partial.replace(path)" in smoke
    assert '"commit": SOURCE_REF' in smoke
    assert '"source_prune_path": SOURCE_PRUNE_PATH' in smoke
    assert '"git_objects_removed": True' in smoke
    assert "observed_source_revision != SOURCE_REF" in smoke
    assert "build_metadata.get(\"build_command_sha256\") != BUILD_COMMAND_SHA256" in smoke
    assert 'build_metadata.get("base_image_reference") != BASE_IMAGE_REFERENCE' in smoke
    assert 'build_metadata.get("base_image_digest") != BASE_IMAGE_DIGEST' in smoke
    assert 'build_metadata.get("base_image_digest_pinned") is not True' in smoke
    assert '"base_image_reference": build_metadata["base_image_reference"]' in smoke
    assert '"base_image_digest": build_metadata["base_image_digest"]' in smoke
    assert "generated_build_metadata_and_oci_config_labels" in smoke
    build_sha256 = hashlib.sha256(build.encode()).hexdigest()
    assert f'BUILD_COMMAND_SHA256 = "{build_sha256}"' in smoke


def test_libero_smoke_uses_real_upstream_bc_training_and_heldout_evaluation() -> None:
    smoke = _smoke_source()

    assert "from libero.lifelong.algos.base import Sequential" in smoke
    assert "from libero.lifelong.datasets import SequenceVLDataset, get_dataset" in smoke
    assert "policy=bc_rnn_policy" in smoke
    assert "algorithm.observe(batch)" in smoke
    assert "TRAIN_STEPS = 8" in smoke
    assert "observed_optimizer_steps != TRAIN_STEPS" in smoke
    assert "not training_losses_finite" in smoke
    assert "parameter_delta <= 0" in smoke
    assert "torch_save_model(algorithm.policy" in smoke
    assert "torch_load_model(" in smoke
    assert "load_state_dict(state_dict, strict=True)" in smoke
    assert "reloaded.policy.compute_loss(batch)" in smoke
    assert "reloaded.policy.forward(batch)" in smoke
    assert "distribution.mean" in smoke
    assert "torch.isfinite(actions).all()" in smoke

    assert "heldout_demo_ids" in smoke
    assert "train_demo_ids" in smoke
    assert "set(train_demo_ids) & set(heldout_demo_ids)" in smoke
    assert '"strategy": "deterministic_trajectory_disjoint_sha256_rank"' in smoke
    assert '"heldout_metrics"' in smoke
    assert '"evaluated_sample_count"' in smoke
    assert '"optimizer_steps"' in smoke
    assert '"reloaded_action"' in smoke
    assert '"finite": all_actions_finite' in smoke
    assert '"prediction_sha256": action_prediction_digest.hexdigest()' in smoke
    assert '"value_min": action_value_min' in smoke
    assert '"value_max": action_value_max' in smoke


def test_libero_smoke_uses_pinned_upstream_task_language_conditioning() -> None:
    smoke = _smoke_source()

    assert 'LANGUAGE_MODEL_REPOSITORY = "google-bert/bert-base-cased"' in smoke
    assert f'LANGUAGE_MODEL_REVISION = "{LANGUAGE_MODEL_REF}"' in smoke
    assert LANGUAGE_MODEL_MODEL_SHA256 in smoke
    assert 'LANGUAGE_MODEL_LICENSE = "Apache-2.0"' in smoke
    assert "AutoTokenizer.from_pretrained" in smoke
    assert "AutoModel.from_pretrained" in smoke
    assert "local_files_only=True" in smoke
    assert '"pooler_output"' in smoke
    assert "cfg.data.max_word_len" in smoke
    assert "upstream_LIBERO_bert_pooler_output" in smoke
    assert "TASK_EMBEDDING_SOURCE_SHA256" in smoke
    assert "observed_task_embedding_source_sha256" in smoke
    assert "embedding_bytes" not in smoke
    assert "np.resize" not in smoke
    assert "deterministic_single_task_768d_sha256" not in smoke


def test_libero_smoke_binds_exact_task_assets_and_real_sample_inventory() -> None:
    smoke = _smoke_source()

    assert "9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b" in smoke
    assert "c3a6a01fdc53ae1914fe24c8935088d723baee8f6ee3cd5f8d68e86aea3e2f1c" in smoke
    assert 'data_group.attrs.get("bddl_file_name"' in smoke
    assert 'data_group.attrs.get("problem_info"' in smoke
    assert "len(demo_ids) != 50 or sample_count != 5068" in smoke
    assert '"sample_count": sample_count' in smoke
    assert '"dataset_bddl_path": dataset_bddl' in smoke
    assert "libero_official_demo_sha256" in smoke
    assert "libero_upstream_bert_task_conditioning" in smoke
    assert "libero_trajectory_disjoint_heldout_split" in smoke


def test_libero_expected_source_metadata_matches_generated_public_schema() -> None:
    expected = {
        "source",
        "repo",
        "ref",
        "commit",
        "source_prune_path",
        "source_pruned",
        "git_objects_removed",
    }
    assert _expected_source_metadata_keys() == expected
    assert _generated_public_source_metadata_keys() == expected


def test_libero_smoke_requires_one_observed_b200_digest_and_never_renders() -> None:
    workflow = _workflow()
    resources = workflow["resources"]
    assert isinstance(resources, dict)
    gpu = resources["gpu"]
    assert isinstance(gpu, dict)
    assert gpu["accelerators"] == "B200:1"

    profile_text = PROFILE_PATH.read_text(encoding="utf-8")
    profile_docs = list(yaml.safe_load_all(profile_text))
    assert profile_docs[1]["resources"]["accelerators"] == "B200:1"
    payload_service_account = profile_docs[1]["resources"]["kubernetes"][
        "pod_config"
    ]["spec"]["serviceAccountName"]
    assert payload_service_account == PAYLOAD_SERVICE_ACCOUNT
    assert payload_service_account not in {"default", "skypilot-service-account"}
    assert "NVIDIA_VISIBLE_DEVICES" not in profile_docs[1]["envs"]
    assert profile_docs[1]["envs"]["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility"
    assert "missing required smoke artifact" in profile_text
    assert "SMOKE_EXIT_CODE=1" in profile_text
    assert "/usr/local/sbin/npa-skypilot-bootstrap-guard verify" in profile_text

    smoke = _smoke_source()
    assert "torch.cuda.device_count() != 1" in smoke
    assert '"B200" not in gpu_name.upper()' in smoke
    assert "compute_capability != (10, 0)" in smoke
    assert '"sm_100" not in torch_arches' in smoke
    assert "-arch=sm_100" in str(_config()["build_command"])
    assert "containerStatuses" in smoke
    assert "imageID" in smoke
    assert 'observed_digests = re.findall(r"sha256:[0-9a-f]{64}"' in smoke
    assert '"pod_observed_image_digest"' in smoke
    assert 'service_account_name != "npa-byof-libero-payload"' in smoke
    assert 'service_account_name in {"default", "skypilot-service-account"}' in smoke
    assert "bound_service_account_claims(token)" in smoke
    assert "NPA_LIBERO_EXPECTED_SERVICE_ACCOUNT_UID_SHA256" in smoke
    assert "NPA_LIBERO_EXPECTED_ROLE_BINDING_UID_SHA256" in smoke
    assert "NPA_LIBERO_EXPECTED_ALLOWED_NODE_SHA256" in smoke
    assert '"controller_service_account_separated": True' in smoke
    for name in (
        "NPA_LIBERO_EXPECTED_ALLOWED_NODE_SHA256",
        "NPA_LIBERO_EXPECTED_CLUSTER_IDENTITY_SHA256",
        "NPA_LIBERO_EXPECTED_NAMESPACE_SHA256",
        "NPA_LIBERO_EXPECTED_RBAC_SPEC_SHA256",
        "NPA_LIBERO_EXPECTED_ROLE_BINDING_UID_SHA256",
        "NPA_LIBERO_EXPECTED_ROLE_UID_SHA256",
        "NPA_LIBERO_EXPECTED_SERVICE_ACCOUNT_UID_SHA256",
    ):
        assert name in profile_docs[1]["envs"]
    assert "evaluate_one_task_success" not in smoke
    assert "OffScreenRenderEnv" not in smoke
    assert '"rendering_invoked": False' in smoke

    documentation = DOC_PATH.read_text(encoding="utf-8")
    assert PAYLOAD_SERVICE_ACCOUNT in documentation
    assert '`apiGroups: [""]`' in documentation
    assert '`resources: ["pods"]`' in documentation
    assert '`verbs: ["get"]`' in documentation
    assert "refuses a missing, `default`, or" in documentation


def test_libero_artifact_contract_is_fail_closed_and_scoped() -> None:
    smoke = _smoke_source()

    assert 'artifact = output_dir / "libero-smoke.json"' in smoke
    assert ".write_text(" in smoke
    assert "json.dumps(" in smoke
    for field in (
        '"solution"',
        '"capability"',
        '"capabilities_exercised"',
        '"source"',
        '"dataset"',
        '"sample_count"',
        '"split"',
        '"training"',
        '"heldout_metrics"',
        '"checkpoint"',
        '"reloaded_action"',
        '"runtime"',
        '"exit_status"',
    ):
        assert field in smoke
    assert '"status": "failed"' in smoke
    assert '"exit_status": 1' in smoke
    assert '"status": "passed"' in smoke
    assert '"exit_status": 0' in smoke
    assert '"cache_uploaded": False' in smoke
    assert '"render_assets_baked": False' in smoke
    assert '"git_objects_baked": False' in smoke
    for deferred in (
        "rendered_closed_loop_success_sweeps",
        "all_130_tasks",
        "lifelong_algorithm_comparison",
        "physical_robot_use",
    ):
        assert deferred in smoke


def test_libero_readiness_record_tracks_final_workflow_bytes() -> None:
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    workflow_sha256 = hashlib.sha256(WORKFLOW_PATH.read_bytes()).hexdigest()
    profile_sha256 = hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()

    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert readiness["workflow_sha256"] == workflow_sha256
    assert readiness["resource_profile_sha256"] == profile_sha256
    assert readiness["planning"]["validation"]["status"] == "verified"
    assert readiness["planning"]["task_fidelity"]["status"] == "verified"
    assert readiness["prerequisites"]["source_image"]["status"] in {
        "blocked",
        "unverified",
        "verified",
    }
    assert readiness["prerequisites"]["target_runtime"]["status"] in {
        "blocked",
        "unverified",
        "verified",
    }
    assert re.fullmatch(r"[0-9a-f]{64}", readiness["workflow_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", readiness["resource_profile_sha256"])
