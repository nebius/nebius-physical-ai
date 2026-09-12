"""Contract tests for the RoboTwin 2.0 registry-candidate workflow."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest
import yaml

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.robotwin_preflight import (
    RobotwinPreflightError,
)

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


def test_robotwin_workflow_validates_and_plans_the_byof_toolref() -> None:
    spec = load_spec(WORKFLOW)
    plan = build_plan(spec, run_id="robotwin-contract")

    assert spec.metadata["name"] == "byof-robotwin"
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_ref == "workbench.byof.repo"
    assert "--runtime-context-env" in plan.steps[0].argv
    assert "NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT" in plan.steps[0].argv
    assert spec.resources == {
        "launcher": {"cloud": "kubernetes", "cpus": "4+", "memory": "8+"}
    }
    assert "accelerators" not in spec.resources["launcher"]
    assert "image" not in spec.resources["launcher"]
    for private_field in (
        "nebius_profile",
        "kubeconfig",
        "kubernetes_context",
        "skypilot_config_path",
        "bootstrap_image",
    ):
        assert private_field not in " ".join(plan.steps[0].argv)


def test_robotwin_selects_only_the_quarantined_zero_payload_bootstrap() -> None:
    config = _config()
    build = str(config["build_command"])
    workflow_text = WORKFLOW.read_text(encoding="utf-8")

    assert config["repo_url"] == "https://github.com/RoboTwin-Platform/RoboTwin.git"
    assert config["repo_ref"] == SOURCE_REVISION
    assert config["base_profile"] == "prebuilt"
    assert config["base_image"] == "tool://robotwin"
    assert build == ""
    assert config["smoke_command"] == "/opt/npa/robotwin/robotwin-runtime run"
    assert "ghcr.io/nebius" not in workflow_text
    assert "nvidia/cuda" not in workflow_text
    assert CUROBO_REVISION not in workflow_text
    assert ASSET_REVISION not in workflow_text

    sys.path.insert(0, str(ROOT / "npa" / "scripts"))
    import run_byof_repo as runner

    assert runner.ROBOTWIN_BUILD_COMMAND_SHA256 == hashlib.sha256(
        build.encode()
    ).hexdigest()
    assert runner.ROBOTWIN_SMOKE_COMMAND_SHA256 == hashlib.sha256(
        str(config["smoke_command"]).encode()
    ).hexdigest()


def test_robotwin_runtime_lock_records_exact_deferred_boundaries() -> None:
    config = _config()
    lock = json.loads(
        (ROOT / "npa/docker/workbench/robotwin/runtime-lock.json").read_text()
    )

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

    assert lock["status"] == "incomplete"
    assert lock["weights"] == []
    assert lock["runtime_artifacts"] == []
    assert {item.get("version") for item in lock["sources"]} == {
        SOURCE_REVISION,
        "0.7.8",
    }
    curobo = next(item for item in lock["sources"] if item["name"] == "CuRobo")
    assert curobo["revision"] == CUROBO_REVISION
    assert curobo["use_restriction"] == "noncommercial-research-or-evaluation"
    assert {item["revision"] for item in lock["assets"]} == {ASSET_REVISION}
    assert all(item["sha256"] and item["size_bytes"] > 0 for item in lock["assets"])
    assert lock["packaging_shape"] == "whole-source-sdk-runtime-fetch"
    assert lock["operator_scope"] == {
        "statement": "noncommercial",
        "intended_activity": (
            "containerization-and-technical-workload-validation-and-evaluation"
        ),
        "record": "owner-only-bounded-manager-run-record",
        "lifetime": "expires-with-bounded-manager-run",
        "global_or_permanent": False,
        "curobo_compatibility": "noncommercial-research-or-evaluation-only",
        "service_and_output_use": "human-decision-required",
    }
    assert lock["access"]["status"] == "not-probed"
    assert lock["access"]["timing"] == "before-provisioning"
    assert lock["access"]["credential_owner"] == "customer"
    assert lock["access"]["credential_phase"] == "runtime-only-secret-value"
    assert lock["access"]["credential_persistence"] is False
    assert lock["access"]["token_is_terms_acceptance"] is False
    assert lock["cache"]["tier"] == "node-local-ephemeral"
    assert lock["cache"]["owner_access"] == "single-customer-single-workload"
    assert lock["cache"]["durable_reuse"] == (
        "disabled-until-rights-and-isolation-approved"
    )
    assert lock["cache"]["contains_credentials"] is False


def test_robotwin_profile_requests_one_rtx_and_runs_authorized_bootstrap() -> None:
    task = _profile_task()
    resources = task["resources"]
    envs = task["envs"]
    run = str(task["run"])

    assert isinstance(resources, dict)
    assert resources["cloud"] == "kubernetes"
    assert resources["accelerators"] == (
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    )
    pod_env = resources["kubernetes"]["pod_config"]["spec"]["containers"][0][
        "env"
    ]
    assert {item["name"] for item in pod_env} == {"POD_NAME", "POD_NAMESPACE"}
    assert {item["valueFrom"]["fieldRef"]["fieldPath"] for item in pod_env} == {
        "metadata.name",
        "metadata.namespace",
    }
    assert isinstance(envs, dict)
    for secret_name in (
        "NPA_INTERNAL_BYOF_ROBOTWIN_RUN_ID",
        "NPA_INTERNAL_BYOF_ROBOTWIN_OUTPUT_PREFIX",
        "NPA_INTERNAL_BYOF_ROBOTWIN_BUCKET",
        "NPA_INTERNAL_BYOF_ROBOTWIN_IMAGE",
        "NPA_INTERNAL_BYOF_ROBOTWIN_RUNTIME_AUTH_V1",
    ):
        assert f'${{{secret_name}:?}}' in run
    assert envs["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert envs["VK_ICD_FILENAMES"] == "/usr/share/vulkan/icd.d/nvidia_icd.json"
    assert '/bin/bash -lc "${BYOF_SMOKE_COMMAND}"' in run
    assert 'test -s "${NPA_SMOKE_OUTPUT_DIR}/${BYOF_SMOKE_ARTIFACT_NAME}"' in run
    assert "vulkaninfo" not in str(task["setup"])
    assert "boto3" not in run


def test_robotwin_has_exactly_one_accelerator_request_across_both_layers() -> None:
    outer = _payload()["resources"]
    inner = _profile_task()["resources"]
    accelerator_requests = [
        profile["accelerators"]
        for profile in outer.values()
        if "accelerators" in profile
    ]
    accelerator_requests.extend(
        [inner["accelerators"]] if "accelerators" in inner else []
    )

    assert accelerator_requests == [
        "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
    ]
    assert all("B200" not in request for request in accelerator_requests)


def test_robotwin_live_gate_requires_manager_context_and_license_decisions() -> None:
    live_test = LIVE_E2E.read_text(encoding="utf-8")

    assert inspect.signature(
        live_e2e.test_live_robotwin_build_push_run_and_artifacts
    ).parameters == {}
    for required in (
        "NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT",
        "load_runtime_authorization",
        '"workflow"',
        '"submit"',
        '"--secret-env"',
        'str(config["runtime_context_env"])',
        '"-m"',
        '"npa"',
    ):
        assert required in live_test
    for forbidden in (
        '"--registry"',
        '"--project"',
        '"--config-path"',
        'ROBOTWIN_IMAGE_SCANNER',
        'str(BYOF_RUNNER)',
    ):
        assert forbidden not in inspect.getsource(
            live_e2e.test_live_robotwin_build_push_run_and_artifacts
        )
    assert "NPA_BYOF_ROBOTWIN_STRICT_CAPACITY" not in live_test


def test_robotwin_live_gate_refuses_missing_owner_context_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT", raising=False)
    monkeypatch.setattr(
        live_e2e.subprocess,
        "run",
        lambda *_: pytest.fail("authorization refusal occurred too late"),
    )

    with pytest.raises(RobotwinPreflightError, match="context-missing"):
        live_e2e.test_live_robotwin_build_push_run_and_artifacts()


def test_robotwin_live_gate_refuses_incomplete_runtime_use_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text(
        "apiVersion: v1\nkind: Config\n"
        "contexts: [{name: private-context, context: {}}]\nusers: []\n",
        encoding="utf-8",
    )
    skypilot = tmp_path / "skypilot.yaml"
    skypilot.write_text("kubernetes: {}\n", encoding="utf-8")
    kubeconfig.chmod(0o600)
    skypilot.chmod(0o600)
    context = tmp_path / "runtime-context.json"
    context.write_text(
        json.dumps(
            {
                "solution": "robotwin",
                "ownership_provenance": "manager-issued",
                "workflow_sha256": "718bb6ae47c8e5e7e761303ebda9e962afa446a6b84030dade7c224cd255ece3",
                "source_revision": "96c1feab536306b50c26af200044fcdf126e8904",
                "curobo_revision": "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
                "asset_revision": "785feb15aa4a4f532395ad2b1d2be5f28cb561ad",
                "runtime_lock_sha256": "86d343677017e7e4934ed2cf9f42a9c924b07d88e205f03a79bcbbed817a772c",
                "bootstrap_image": "registry.example/private/robotwin/npa-robotwin@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "reservation": {
                    "policy": "STRICT",
                    "accelerator": "RTXPRO-6000-BLACKWELL-SERVER-EDITION",
                    "count": 1,
                },
                "license_acceptance": {
                    "nvidia_cuda_eula": True,
                    "nvidia_cudnn_sla": True,
                    "curobo_noncommercial_research_or_evaluation": True,
                    "robotwin2_aggregate_asset_and_output_terms": False,
                },
                "project": "private-project",
                "nebius_profile": "private-profile",
                "kubeconfig": str(kubeconfig),
                "kubernetes_context": "private-context",
                "skypilot_config_path": str(skypilot),
                "bucket": "private-bucket",
                "output_root": "s3://private-bucket/robotwin-output",
                "run_id": "robotwin-private-run",
            }
        ),
        encoding="utf-8",
    )
    context.chmod(0o600)
    monkeypatch.setattr(live_e2e, "REPO_ROOT", repo)
    monkeypatch.setenv("NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT", str(context))
    monkeypatch.setattr(
        live_e2e.subprocess,
        "run",
        lambda *_: pytest.fail("license refusal occurred too late"),
    )

    with pytest.raises(
        RobotwinPreflightError,
        match="license-decision-robotwin2_aggregate_asset_and_output_terms-missing",
    ):
        live_e2e.test_live_robotwin_build_push_run_and_artifacts()


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
