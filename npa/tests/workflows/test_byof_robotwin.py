"""Contract tests for the RoboTwin 2.0 registry-candidate workflow."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.e2e import test_byof_onboarding_live_e2e as live_e2e  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows" / "testing" / "byof-robotwin.yaml"
READINESS = ROOT / "workflows" / "testing" / "byof-robotwin.readiness.json"
PROFILE = (
    ROOT
    / "npa"
    / "src"
    / "npa"
    / "workflows"
    / "byof"
    / "profiles"
    / "byof-solution-smoke-robotwin-rtxpro-gpu.yaml"
)
LIVE_E2E = ROOT / "npa" / "tests" / "e2e" / "test_byof_onboarding_live_e2e.py"
SOURCE_REVISION = "96c1feab536306b50c26af200044fcdf126e8904"
ASSET_REVISION = "785feb15aa4a4f532395ad2b1d2be5f28cb561ad"
CUROBO_REVISION = "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"


def _payload() -> dict[str, object]:
    value = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _config() -> dict[str, object]:
    value = _payload()["config"]
    assert isinstance(value, dict)
    return value


def _profile_task() -> dict[str, object]:
    documents = [
        document
        for document in yaml.safe_load_all(PROFILE.read_text(encoding="utf-8"))
        if document
    ]
    assert len(documents) == 2
    task = documents[1]
    assert isinstance(task, dict)
    return task


def _embedded_requirements(build: str) -> str:
    match = re.search(r"printf '%s'\s+(.*?)\s+\| base64 -d", build, re.DOTALL)
    assert match is not None
    chunks = re.findall(r"'([A-Za-z0-9+/=]+)'", match.group(1))
    assert chunks
    return base64.b64decode("".join(chunks)).decode()


def test_robotwin_workflow_validates_and_plans_the_byof_toolref() -> None:
    spec = load_spec(WORKFLOW)
    plan = build_plan(spec, run_id="robotwin-contract")

    assert spec.metadata["name"] == "byof-robotwin"
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_ref == "workbench.byof.repo"


def test_robotwin_pins_source_assets_build_inputs_and_private_runtime() -> None:
    config = _config()
    build = str(config["build_command"])
    requirements = _embedded_requirements(build)
    workflow_text = WORKFLOW.read_text(encoding="utf-8")

    assert config["repo_url"] == "https://github.com/RoboTwin-Platform/RoboTwin.git"
    assert config["repo_ref"] == SOURCE_REVISION
    assert config["base_profile"] == "ubuntu"
    assert config["base_image"] == (
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@"
        "sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719"
    )
    assert CUROBO_REVISION in build
    assert "torch==2.7.1+cu128" in requirements
    assert "torchvision==0.22.1+cu128" in requirements
    assert "nvidia-cudnn-cu12==9.7.1.26" in requirements
    assert hashlib.sha256(requirements.encode()).hexdigest() == (
        "5a2b78949b5be3e30089b930e2dfd2197a505f89a3b66cf26def2572306af5b8"
    )
    assert "TORCH_CUDA_ARCH_LIST=12.0" in build
    assert "python3-dev" in build
    assert "sapien==3.0.0b1" in requirements
    assert "PyTorch3D" not in build and "pytorch3d" not in build
    assert "TianxingChen/RoboTwin2.0" not in build
    assert ASSET_REVISION not in build
    assert "ghcr.io/nebius" not in workflow_text


def test_robotwin_smoke_is_a_real_successful_seed_search_replay_and_collection() -> None:
    config = _config()
    smoke = str(config["smoke_command"])

    assert config["workload"] == "solution-smoke"
    assert config["solution_name"] == "robotwin"
    assert config["capability_name"] == (
        "beat_block_hammer_successful_seed_replay_collection"
    )
    assert config["smoke_artifact_name"] == "robotwin-smoke.json"
    assert config["resource_profile_yaml"] == (
        "byof-solution-smoke-robotwin-rtxpro-gpu"
    )
    assert config["task"] == "beat_block_hammer"
    assert config["wait_timeout"] == -1

    for required in (
        ASSET_REVISION,
        "revision=ASSET_REVISION",
        'TASK = "beat_block_hammer"',
        'TASK_CONFIG = "demo_clean"',
        'task_config["episode_num"] = 1',
        '"scripts/update_embodiment_config_path.py"',
        '"official_embodiment_path_configuration"',
        '"scripts/collect_data.py", TASK, TASK_CONFIG',
        'episode_root / "seed.txt"',
        'episode_0000000.hdf5',
        'episode_0000000.mp4',
        'episode.attrs.get("source_format") != "RoboTwin"',
        'episode.attrs.get("source_path") != "native_collection"',
        'decoded_frames != action_count + 1',
        '"task_success": True',
        '"exit_status": 0',
        'artifact_path.write_text(json.dumps(',
    ):
        assert required in smoke

    python_smoke = smoke.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    ast.parse(python_smoke)


def test_robotwin_smoke_hard_fails_closed_on_gpu_vulkan_image_and_artifacts() -> None:
    smoke = str(_config()["smoke_command"])

    assert 'len(gpu_rows) != 1' in smoke
    assert '"RTX PRO 6000" not in gpu_name.upper()' in smoke
    assert 'compute_capability != "12.0"' in smoke
    assert 'torch_capability != (12, 0)' in smoke
    assert "NPA_BYOF_ROBOTWIN_RESERVATION_EVIDENCE_SHA256" in smoke
    assert "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_SHA256" in smoke
    assert "NPA_BYOF_ROBOTWIN_IMAGE_SCAN_ARCHIVES" in smoke
    assert '"built_image_asset_cache_output_absence"' in smoke
    assert '"npa_robotwin_image_byte_scan_v1"' in smoke
    assert '"policy": "STRICT"' in smoke
    assert 'vulkan.returncode != 0' in smoke
    assert '"RTX PRO 6000" not in vulkan_text.upper()' in smoke
    assert 'sapien.render.get_device_summary()' in smoke
    assert re.search(r"sha256:\[0-9a-f\]\{64\}", smoke)
    assert 'if not hdf5_path.is_file() or not video_path.is_file()' in smoke
    assert 'if payload["exit_status"] != 0' in smoke


def test_robotwin_profile_requests_one_rtx_pro_and_uploads_exact_evidence() -> None:
    task = _profile_task()
    resources = task["resources"]
    envs = task["envs"]
    run = str(task["run"])

    assert isinstance(resources, dict)
    assert resources["cloud"] == "kubernetes"
    assert resources["accelerators"] == (
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    )
    assert isinstance(envs, dict)
    assert envs["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert envs["VK_ICD_FILENAMES"] == "/usr/share/vulkan/icd.d/nvidia_icd.json"
    assert 'vulkaninfo --summary' in run
    assert 'test -f "${OUTPUT_DIR}/${BYOF_SMOKE_ARTIFACT_NAME}"' in run
    assert 'IfNoneMatch="*"' in run
    assert 's3.head_object' in run
    assert 'exit "${SMOKE_EXIT_CODE}"' in run


def test_robotwin_live_gate_requires_manager_context_and_license_decisions() -> None:
    live_test = LIVE_E2E.read_text(encoding="utf-8")

    for required in (
        "NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT",
        'payload.get("ownership_provenance")',
        'reservation.get("policy") == "STRICT"',
        'reservation.get("accelerator") == "RTXPRO-6000-BLACKWELL-SERVER-EDITION"',
        'reservation.get("count") == 1',
        '"nvidia_cuda_eula"',
        '"nvidia_cudnn_sla"',
        '"curobo_noncommercial_research_or_evaluation"',
        '"robotwin2_aggregate_asset_and_output_terms"',
        'ROBOTWIN_IMAGE_SCANNER',
        '"--skip-run"',
        '"--skip-build"',
        'env["NPA_BYOF_ROBOTWIN_RESERVATION_EVIDENCE_SHA256"] = runtime_sha256',
        'env["NPA_BYOF_ROBOTWIN_IMAGE_SCAN_SHA256"] = scan_report_sha256',
    ):
        assert required in live_test
    assert "NPA_BYOF_ROBOTWIN_STRICT_CAPACITY" not in live_test


def test_robotwin_live_gate_refuses_missing_owner_context_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT", raising=False)
    monkeypatch.setattr(
        live_e2e,
        "_activate_nebius_profile",
        lambda *_: pytest.fail("authorization refusal occurred too late"),
    )

    with pytest.raises(AssertionError, match="RUNTIME_CONTEXT is required"):
        live_e2e.test_live_robotwin_build_push_run_and_artifacts(None)


def test_robotwin_live_gate_refuses_incomplete_runtime_use_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    context = tmp_path / "runtime-context.json"
    context.write_text(
        json.dumps(
            {
                "solution": "robotwin",
                "ownership_provenance": "manager-issued",
                "reservation": {
                    "policy": "STRICT",
                    "accelerator": "RTXPRO-6000-BLACKWELL-SERVER-EDITION",
                    "count": 1,
                },
                "license_acceptance": {
                    "nvidia_cuda_eula": True,
                    "nvidia_cudnn_sla": True,
                    "curobo_noncommercial_research_or_evaluation": True,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(live_e2e, "REPO_ROOT", repo)
    monkeypatch.setenv("NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT", str(context))
    monkeypatch.setattr(
        live_e2e,
        "_activate_nebius_profile",
        lambda *_: pytest.fail("license refusal occurred too late"),
    )

    with pytest.raises(
        AssertionError,
        match="robotwin2_aggregate_asset_and_output_terms",
    ):
        live_e2e.test_live_robotwin_build_push_run_and_artifacts(None)


def test_robotwin_readiness_hash_matches_workflow() -> None:
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    assert readiness["schema_version"] == "workflow-readiness/v1"
    assert readiness["workflow_sha256"] == hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    assert set(readiness["planning"]) == {"validation", "task_fidelity"}
    assert set(readiness["prerequisites"]) == {
        "output_storage",
        "worker_input",
        "credentials",
        "source_image",
        "target_runtime",
    }
