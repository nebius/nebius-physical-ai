from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows" / "testing" / "byof-robomimic.yaml"
READINESS = ROOT / "workflows" / "testing" / "byof-robomimic.readiness.json"
DOC = ROOT / "docs" / "workbench" / "byof-robomimic.md"
PROFILE = (
    ROOT
    / "npa"
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "byof-solution-smoke-robomimic-b200-gpu.yaml"
)

SOURCE_REVISION = "d309eaecc18acf4152a830a895a6984b8ac71b05"
DATASET_REVISION = "74fa018461f479cd9fd15b924a16103012096203"
DATASET_SHA256 = "2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540"
BASE_DIGEST = "sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2"
BUILD_COMMAND_SHA256 = "40934ad75e79127e2b494adf73c59e0cd2164dda71dac2a142891cdda7a43bf6"
DEPENDENCY_LOCK_SHA256 = "910b762eb9fa6bb31bfb05d339845d8c68c81f0b3e85ab95ec3eefbca29cad71"


def _workflow_config() -> dict[str, object]:
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    config = payload["config"]
    assert isinstance(config, dict)
    return config


def _smoke_python() -> str:
    smoke = str(_workflow_config()["smoke_command"])
    return smoke.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def test_robomimic_workflow_validates_and_plans_real_byof_stage() -> None:
    spec = load_spec(WORKFLOW)
    plan = build_plan(spec, run_id="robomimic-plan-test")

    assert spec.name == "byof-robomimic"
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.tool_ref == "workbench.byof.repo"
    assert "byof-solution-smoke-robomimic-b200-gpu" in step.argv
    assert "robomimic-smoke.json" in step.argv
    assert "B200:1" in WORKFLOW.read_text(encoding="utf-8")


def test_robomimic_smoke_is_immutable_and_fails_closed() -> None:
    config = _workflow_config()
    build = str(config["build_command"])
    smoke = str(config["smoke_command"])

    assert config["repo_ref"] == SOURCE_REVISION
    assert config["base_image"].endswith(f"@{BASE_DIGEST}")
    assert config["resource_profile_yaml"] == "byof-solution-smoke-robomimic-b200-gpu"
    assert config["capability_name"] == "lift_ph_lowdim_checkpoint_reload_action"
    assert config["smoke_artifact_name"] == "robomimic-smoke.json"
    assert config["wait_timeout"] == -1
    assert "low_dim_v15.hdf5" not in build
    assert DATASET_REVISION not in build
    assert hashlib.sha256(build.encode()).hexdigest() == BUILD_COMMAND_SHA256
    assert "--only-binary=:all: --no-deps --require-hashes" in build
    assert "--no-deps --no-build-isolation -e ." in build
    assert "boto3==1.38.33 --hash=sha256:" in build
    assert "torch.__version__.split('+')[0] == '2.7.1'" in build
    lock_lines = [
        line[3:-3]
        for line in build.splitlines()
        if line.startswith("  '") and line.endswith("' \\")
    ]
    assert len(lock_lines) == 48
    lock_bytes = ("\n".join(lock_lines) + "\n").encode()
    assert hashlib.sha256(lock_bytes).hexdigest() == DEPENDENCY_LOCK_SHA256

    for required in (
        SOURCE_REVISION,
        DATASET_REVISION,
        DATASET_SHA256,
        '"split_train_val.py"',
        "np.random.seed(0)",
        "len(train_keys) + len(valid_keys) != len(demo_keys)",
        '"train.py"',
        "TRAIN_STEPS = 4",
        "VALIDATION_STEPS = 2",
        'config.train.hdf5_filter_key = "train"',
        'config.train.hdf5_validation_filter_key = "valid"',
        '["git", "-C", str(repo_root), "rev-parse", "HEAD"]',
        "source_head != SOURCE_REVISION",
        f'EXPECTED_BUILD_COMMAND_SHA256 = "{BUILD_COMMAND_SHA256}"',
        "build_metadata.get(\"build_command_sha256\") != EXPECTED_BUILD_COMMAND_SHA256",
        f'DEPENDENCY_LOCK_SHA256 = "{DEPENDENCY_LOCK_SHA256}"',
        "dependency_lock_hash != DEPENDENCY_LOCK_SHA256",
        "dependency_artifact_count != DEPENDENCY_ARTIFACT_COUNT",
        "policy_from_checkpoint(",
        "optimizer_step_count != TRAIN_STEPS",
        "action.shape != (7,)",
        '"B200" not in gpu_name.upper()',
        'architecture != "sm_100"',
        'os.environ.get("NPA_ROBOMIMIC_STRICT_B200_ATTESTED") != "1"',
        '"architecture": architecture',
        '"strict_reserved_capacity_attested": True',
        '"observed_head": source_head',
        '"trajectory_count": len(demo_keys)',
        '"sample_count": sum(sample_counts.values())',
        '"train_trajectory_count": len(train_keys)',
        '"validation_trajectory_count": len(valid_keys)',
        '"train_sample_count": sum(sample_counts[key] for key in train_keys)',
        '"validation_sample_count": sum(sample_counts[key] for key in valid_keys)',
        '"train_loss": train_loss',
        '"validation_loss": validation_loss',
        '"sha256": checkpoint_hash',
        '"finite": finite',
        '"within_allowed_range": within_range',
        '"accelerator_count": gpu_count',
        '"runtime_ref": runtime_image',
        '"image_id": pod_image_id',
        '"observation_source": "Kubernetes Pod status.containerStatuses[].imageID"',
        '"pod_image"',
    ):
        assert required in smoke

    assert "image_policy_sweeps" in smoke
    assert "simulator_rollouts" in smoke
    assert "full_algorithm_matrix" in smoke
    assert "robosuite" not in smoke
    ast.parse(_smoke_python())


def test_robomimic_profile_is_exactly_one_compute_only_b200() -> None:
    documents = list(yaml.safe_load_all(PROFILE.read_text(encoding="utf-8")))
    task = documents[1]
    assert task["resources"]["accelerators"] == "B200:1"
    assert (
        task["config"]["kubernetes"]["pod_config"]["spec"]["serviceAccountName"]
        == "npa-robomimic-observer"
    )
    assert task["envs"]["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility"
    assert task["envs"]["BYOF_IMAGE"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_STRICT_B200_ATTESTED"] == (
        "${NPA_E2E_MK8S_RESERVED_CAPACITY}"
    )
    assert "pip install --quiet boto3" not in PROFILE.read_text(encoding="utf-8")
    assert 'smoke_artifact["exit_status"] = smoke_exit_code' in PROFILE.read_text(
        encoding="utf-8"
    )
    text = PROFILE.read_text(encoding="utf-8").lower()
    for graphics_surface in ("vulkan", "egl"):
        assert graphics_surface not in text


def test_robomimic_readiness_binds_exact_workflow_bytes() -> None:
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    workflow_hash = hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()

    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert readiness["workflow_sha256"] == workflow_hash
    assert readiness["planning"]["validation"]["status"] in {
        "verified",
        "unverified",
    }
    assert readiness["planning"]["task_fidelity"]["status"] in {
        "verified",
        "unverified",
    }
    assert set(readiness["prerequisites"]) == {
        "output_storage",
        "worker_input",
        "credentials",
        "source_image",
        "target_runtime",
    }


def test_robomimic_delivery_boundaries_do_not_invent_runtime_consent() -> None:
    doc = DOC.read_text(encoding="utf-8")
    normalized_doc = " ".join(doc.split())
    workflow = WORKFLOW.read_text(encoding="utf-8")
    profile = PROFILE.read_text(encoding="utf-8")

    for boundary in (
        "| Source |",
        "| Baked runtime |",
        "| Weights |",
        "| Data and assets |",
        "| Runtime cache |",
        "| Outputs |",
    ):
        assert boundary in doc

    assert "changes delivery, not permission" in normalized_doc
    assert "No image has been built" in doc
    assert "pretrained weights" in doc
    assert "every layer and image history entry" in doc
    combined = "\n".join((doc, workflow, profile))
    for invented_proxy in (
        "NPA_ROBOMIMIC_ACCEPT",
        "NPA_ACCEPT_CUDA",
        "NPA_ACCEPT_CUDNN",
    ):
        assert invented_proxy not in combined
