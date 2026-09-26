"""Live infra checks for generic BYOF solution onboarding (workflow + optional agent chat)."""

from __future__ import annotations
import base64
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import TextIO
from urllib.parse import urlparse
import pytest
from typer.testing import CliRunner
import yaml
from npa.cli.main import app
from npa.clients.config import resolve_container_registry
from npa.clients.project_credentials import s3_client_for_project
from npa.deploy.images import libero_qualified_image_manifest
from npa.execution_preflight import (
    ExecutionPreflightError,
    validate_gymnasium_task_configuration,
)
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.workflows.byof.live import (
    byof_ubuntu_validation_repo,
    byof_validation_repo,
    resolve_byof_kubernetes_target,
    resolve_byof_resource_yaml,
    resolve_skypilot_bin,
    skypilot_config_for_project,
)
from .agent_live_helpers import (
    CREATE_BYOF_WORKFLOW_PROMPT,
    ONBOARD_OSS_REPO_PROMPT,
    ONBOARD_SOLUTION_PROMPT,
    assert_grounded_onboard_solution_reply,
    load_agent_live_context,
)
from .npa_workflow_live_helpers import (
    assert_no_credential_leakage,
    live_bucket,
    live_credential_markers,
    parse_json_payload,
)
import secrets
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator
from urllib.parse import quote, urlsplit
from npa.deploy.images import is_public_registry
from npa.workflows.byof.live import (
    resolve_byof_profile_path,
)


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_INTEGRATION_E2E") != "1",
        reason="Set NPA_INTEGRATION_E2E=1 for live BYOF onboarding infra checks.",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
BYOF_SPEC = REPO_ROOT / "workflows" / "testing" / "byof.yaml"
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_repo.py"
GYMNASIUM_CONTAINER_VERIFY_RUNNER = (
    REPO_ROOT / "npa" / "scripts" / "run_byof_container_verify.py"
)
GYMNASIUM_ROBOTICS_SPEC = (
    REPO_ROOT / "workflows" / "testing" / "byof-gymnasium-robotics.yaml"
)
GYMNASIUM_KUBECTL_TIMEOUT_SECONDS = 30
GYMNASIUM_RECEIPT_TIMEOUT_SECONDS = 900
GYMNASIUM_CLEANUP_TIMEOUT_SECONDS = 180
GYMNASIUM_SKY_DOWN_TIMEOUT_SECONDS = 120
GYMNASIUM_RUNNER_TIMEOUT_SECONDS = 1800
GYMNASIUM_RUNNER_TERM_GRACE_SECONDS = 15
GYMNASIUM_CLEANUP_RESOURCES = (
    "roles",
    "rolebindings",
    "pods",
    "jobs",
    "secrets",
    "persistentvolumeclaims",
)
RUNNER = CliRunner()


ROBOMIMIC_SPEC = REPO_ROOT / "workflows" / "testing" / "byof-robomimic.yaml"

ROBOMIMIC_RUNTIME_LOCK = (
    REPO_ROOT
    / "npa"
    / "docker"
    / "workbench"
    / "robomimic"
    / "runtime-requirements.lock"
)

ROBOMIMIC_ENTITLEMENT_MAX_BYTES = 16 * 1024

ROBOMIMIC_SMOKE_PROOF_MAX_BYTES = 64 * 1024

_ROBOMIMIC_SMOKE_PROOF_ERROR = "robomimic smoke proof validation failed"

ROBOMIMIC_ENTITLEMENT_TERMS = [
    {
        "name": "NVIDIA CUDA Toolkit EULA",
        "url": "https://docs.nvidia.com/cuda/eula/index.html",
    },
    {
        "name": "NVIDIA Software License Agreement",
        "url": (
            "https://www.nvidia.com/en-us/agreements/enterprise-software/"
            "nvidia-software-license-agreement/"
        ),
    },
    {
        "name": "NVIDIA cuDNN Software License Agreement",
        "url": (
            "https://docs.nvidia.com/deeplearning/cudnn/backend/latest/"
            "reference/eula.html"
        ),
    },
]

ROBOMIMIC_SOURCE_REVISION = "d309eaecc18acf4152a830a895a6984b8ac71b05"

ROBOMIMIC_DATASET_REVISION = "74fa018461f479cd9fd15b924a16103012096203"

ROBOMIMIC_DATASET_SHA256 = (
    "2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540"
)

ROBOMIMIC_CAPABILITIES = {
    "lift_ph_lowdim_bc_train",
    "lift_ph_lowdim_heldout_validate",
    "lift_ph_lowdim_checkpoint_reload_action",
}

_ROBOMIMIC_ENTITLEMENT_REFUSAL_CATEGORIES = frozenset(
    {
        "binding-mismatch",
        "context-invalid",
        "contract-invalid",
        "fields-invalid",
        "identity-invalid",
        "record-invalid",
        "record-unsafe",
        "time-invalid",
    }
)


def _activate_nebius_profile() -> None:
    profile = os.environ.get("NPA_NEBIUS_PROFILE", "agent-sa").strip()
    if not profile:
        return
    subprocess.run(
        ["nebius", "profile", "activate", profile],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _parse_last_json_blob(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    idx = 0
    last_obj: dict[str, object] | None = None
    while idx < len(text):
        next_brace = text.find("{", idx)
        if next_brace < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, next_brace)
        except json.JSONDecodeError:
            idx = next_brace + 1
            continue
        if isinstance(obj, dict):
            last_obj = obj
        idx = max(end, next_brace + 1)
    if last_obj is None:
        raise ValueError(f"no JSON object found in command output:\n{text}")
    return last_obj


@pytest.fixture(scope="module")
def live_byof_built_image(e2e_project: str | None) -> str:
    preset_image = os.environ.get("NPA_BYOF_TEST_IMAGE", "").strip()
    if preset_image:
        return preset_image
    if os.environ.get("NPA_BYOF_LIVE_CONTAINER") != "1":
        pytest.skip("Set NPA_BYOF_LIVE_CONTAINER=1 for real BYOF container build/push.")
    _activate_nebius_profile()
    registry = resolve_container_registry(e2e_project)
    repo_url, repo_ref = byof_validation_repo()
    run_id = (
        os.environ.get("NPA_BYOF_CONTAINER_RUN_ID")
        or f"byof-container-live-{os.getpid()}"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(BYOF_RUNNER),
            "--registry",
            registry,
            "--repo-url",
            repo_url,
            "--repo-ref",
            repo_ref,
            "--project",
            e2e_project or "",
            "--run-id",
            run_id,
            "--base-profile",
            "isaac-lab",
            "--skip-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=int(os.environ.get("NPA_BYOF_CONTAINER_TIMEOUT", "3600")),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary.get("status") == "ok", summary
    build = summary.get("build", {})
    assert build.get("ok") is True
    assert build.get("pushed") is True
    image = str(summary["image"])
    assert registry in image
    return image


@pytest.fixture(scope="module")
def forbidden_markers() -> list[str]:
    return live_credential_markers()


def _materialize_byof_spec(tmp_path: Path, *, bucket: str) -> Path:
    text = BYOF_SPEC.read_text(encoding="utf-8")
    text = text.replace("bucket: example-bucket", f"bucket: {bucket}")
    path = tmp_path / "byof-live.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_live_isaac_byof_workflow_validate_and_plan(
    tmp_path: Path,
    e2e_project: str | None,
    forbidden_markers: list[str],
) -> None:
    bucket = live_bucket(e2e_project)
    path = _materialize_byof_spec(tmp_path, bucket=bucket)
    validate = RUNNER.invoke(
        app, ["workbench", "workflow", "validate-spec", str(path), "--json"]
    )
    payload = parse_json_payload(validate, forbidden_markers)
    assert payload["status"] == "valid"
    assert payload["name"] == "byof"
    assert "byof-run" in set(payload.get("states", []))

    plan = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(path),
            "--run-id",
            "byof-onboard-live",
            "--json",
        ],
    )
    plan_payload = parse_json_payload(plan, forbidden_markers)
    steps = plan_payload.get("steps", [])
    assert steps
    tool_refs = {
        step.get("tool_ref") or step.get("toolRef")
        for step in steps
        if isinstance(step, dict)
    }
    assert "workbench.byof.repo" in tool_refs


def test_live_isaac_byof_plan_builder_matches_cli(
    tmp_path: Path,
    e2e_project: str | None,
    forbidden_markers: list[str],
) -> None:
    bucket = live_bucket(e2e_project)
    path = _materialize_byof_spec(tmp_path, bucket=bucket)
    spec = load_spec(path)
    plan = build_plan(spec, run_id="byof-plan-builder")
    assert plan.steps
    assert_no_credential_leakage(
        json.dumps(plan.to_dict()), extra_forbidden=forbidden_markers
    )
    assert any(step.tool_ref == "workbench.byof.repo" for step in plan.steps)


def test_live_byof_registry_resolution(e2e_project: str | None) -> None:
    registry = resolve_container_registry(e2e_project)
    assert registry
    assert "/" in registry
    assert "example-bucket" not in registry
    assert "<your-registry>" not in registry


@pytest.mark.skipif(
    os.environ.get("NPA_AGENT_LIVE") != "1",
    reason="Set NPA_AGENT_LIVE=1 to exercise onboard_solution chat on the configured agent.",
)
def test_live_agent_onboard_solution_chat() -> None:
    ctx = load_agent_live_context()
    chat = ctx.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": ONBOARD_SOLUTION_PROMPT}]},
        timeout=30.0,
    )
    chat.raise_for_status()
    assert_grounded_onboard_solution_reply(chat.json())


@pytest.mark.skipif(
    os.environ.get("NPA_AGENT_LIVE") != "1",
    reason="Set NPA_AGENT_LIVE=1 to validate generic BYOF workflow draft on the configured agent.",
)
def test_live_agent_byof_workflow_draft_validate() -> None:
    ctx = load_agent_live_context()
    draft = ctx.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": CREATE_BYOF_WORKFLOW_PROMPT}]},
        timeout=30.0,
    )
    draft.raise_for_status()
    payload = draft.json()
    assert payload.get("ok") is True
    workflow_yaml = str(payload.get("workflow_yaml") or "")
    assert workflow_yaml
    assert "name: byof" in workflow_yaml or "byof-run" in workflow_yaml
    assert "<repo-url>" in workflow_yaml
    assert "<workload>" in workflow_yaml

    validate = ctx.post(
        "/api/workflows/validate", json={"yaml": workflow_yaml}, timeout=15.0
    )
    validate.raise_for_status()
    validate_payload = validate.json()
    assert validate_payload.get("ok") is True


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_CONTAINER") != "1",
    reason="Set NPA_BYOF_LIVE_CONTAINER=1 for real BYOF docker build/push/inspect.",
)
def test_live_byof_runner_container_build_push(live_byof_built_image: str) -> None:
    assert live_byof_built_image
    assert (
        "npa-byof" in live_byof_built_image or "npa-isaac-lab" in live_byof_built_image
    )


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_CONTAINER") != "1",
    reason="Set NPA_BYOF_LIVE_CONTAINER=1 for real BYOF docker build/push/inspect.",
)
def test_live_byof_container_has_validation_repo(live_byof_built_image: str) -> None:
    repo_url, repo_ref = byof_validation_repo()
    image = live_byof_built_image
    meta_proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "cat",
            image,
            "/opt/byof/npa_source_metadata.json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if meta_proc.returncode != 0:
        pytest.skip("validation repo layout differs from /opt/byof metadata path")
    metadata = json.loads(meta_proc.stdout)
    assert metadata["source"] == "oss-byof"
    assert metadata["repo"] == repo_url
    assert metadata["ref"] == repo_ref


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_GPU") != "1",
    reason="Set NPA_BYOF_LIVE_GPU=1 to run BYOF runner registry smoke on live infra (no build/push).",
)
def test_live_byof_runner_registry_smoke(e2e_project: str | None) -> None:
    registry = resolve_container_registry(e2e_project)
    proc = subprocess.run(
        [
            sys.executable,
            str(BYOF_RUNNER),
            "--registry",
            registry,
            "--project",
            e2e_project or "",
            "--skip-build",
            "--skip-run",
            "--run-id",
            "byof-live-registry-smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary.get("status") == "ok", summary
    assert summary["registry"] == registry
    assert registry in summary["image"]


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_GPU") != "1",
    reason="Set NPA_BYOF_LIVE_GPU=1 to submit a real Isaac BYOF SkyPilot smoke (build/push/run).",
)
def test_live_byof_runner_submit_smoke(
    e2e_project: str | None,
    live_byof_built_image: str,
) -> None:
    registry = resolve_container_registry(e2e_project)
    yaml_override = resolve_byof_resource_yaml(e2e_project, smoke=True)
    image = os.environ.get("NPA_BYOF_TEST_IMAGE", "").strip() or live_byof_built_image
    task = os.environ.get("NPA_BYOF_TASK", "Isaac-Cartpole-v0")
    cmd = [
        sys.executable,
        str(BYOF_RUNNER),
        "--registry",
        registry,
        "--project",
        e2e_project or "",
        "--image",
        image,
        "--yaml",
        yaml_override,
        "--task",
        task,
        "--iterations",
        "1",
        "--run-id",
        f"byof-live-submit-{os.getpid()}",
        "--skip-build",
    ]
    config_path = skypilot_config_for_project(e2e_project)
    if config_path:
        cmd.extend(["--config-path", config_path])
    env = dict(os.environ)
    target = resolve_byof_kubernetes_target(e2e_project)
    if target.kubeconfig:
        env["KUBECONFIG"] = target.kubeconfig
        env["NPA_BYOF_KUBECONFIG"] = target.kubeconfig
    if target.context:
        env["NPA_BYOF_K8S_CONTEXT"] = target.context
    skypilot_bin = resolve_skypilot_bin()
    if skypilot_bin:
        env["PATH"] = f"{Path(skypilot_bin).parent}:{env.get('PATH', '')}"
    env.setdefault("NPA_ISAAC_LAB_ACCEPT_PRECHECK_FAILURE", "1")
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=int(os.environ.get("NPA_BYOF_LIVE_TIMEOUT", "3600")),
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary.get("status") == "ok", summary
    run_summary = summary.get("run", {})
    assert isinstance(run_summary, dict)
    if run_summary.get("status") in {"submitted", "SUBMITTED"}:
        return
    final = run_summary.get("final", {})
    if isinstance(final, dict) and final.get("status"):
        assert final.get("status") in {
            "SUBMITTED",
            "SUCCEEDED",
            "RUNNING",
            "PENDING",
            "FAILED_PRECHECKS",
            "FAILED_SETUP",
            "FAILED",
        }
    submit = run_summary.get("submit", {})
    if isinstance(submit, dict) and submit.get("status"):
        assert submit.get("status") == "SUBMITTED", submit
    elif isinstance(final, dict) and final.get("status") == "FAILED_PRECHECKS":
        pass
    else:
        assert submit or final, run_summary


@pytest.fixture(scope="module")
def live_byof_ubuntu_built_image(e2e_project: str | None) -> str:
    if os.environ.get("NPA_BYOF_LIVE_UBUNTU") != "1":
        pytest.skip(
            "Set NPA_BYOF_LIVE_UBUNTU=1 for Ubuntu OSS BYOF container build/push."
        )
    _activate_nebius_profile()
    registry = resolve_container_registry(e2e_project)
    repo_url, repo_ref = byof_ubuntu_validation_repo()
    run_id = (
        os.environ.get("NPA_BYOF_UBUNTU_RUN_ID") or f"byof-ubuntu-live-{os.getpid()}"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(BYOF_RUNNER),
            "--registry",
            registry,
            "--repo-url",
            repo_url,
            "--repo-ref",
            repo_ref,
            "--project",
            e2e_project or "",
            "--run-id",
            run_id,
            "--base-profile",
            "ubuntu",
            "--skip-run",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=int(os.environ.get("NPA_BYOF_CONTAINER_TIMEOUT", "3600")),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary.get("status") == "ok", summary
    assert summary.get("base_profile") == "ubuntu"
    image = str(summary["image"])
    assert registry in image
    return image


@pytest.mark.skipif(
    os.environ.get("NPA_AGENT_LIVE") != "1",
    reason="Set NPA_AGENT_LIVE=1 to exercise OSS repo onboarding on the configured agent.",
)
def test_live_agent_oss_repo_onboard_solution_chat() -> None:
    ctx = load_agent_live_context()
    chat = ctx.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": ONBOARD_OSS_REPO_PROMPT}]},
        timeout=30.0,
    )
    chat.raise_for_status()
    assert_grounded_onboard_solution_reply(chat.json())


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_UBUNTU") != "1",
    reason="Set NPA_BYOF_LIVE_UBUNTU=1 for Ubuntu OSS BYOF container build/push.",
)
def test_live_byof_ubuntu_oss_container_build_push(
    live_byof_ubuntu_built_image: str,
) -> None:
    assert live_byof_ubuntu_built_image
    assert "npa-byof" in live_byof_ubuntu_built_image


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_UBUNTU") != "1",
    reason="Set NPA_BYOF_LIVE_UBUNTU=1 for Ubuntu OSS BYOF container metadata inspect.",
)
def test_live_byof_ubuntu_oss_container_metadata(
    live_byof_ubuntu_built_image: str,
) -> None:
    repo_url, repo_ref = byof_ubuntu_validation_repo()
    meta_proc = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "cat",
            live_byof_ubuntu_built_image,
            "/opt/byof/npa_source_metadata.json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert meta_proc.returncode == 0, meta_proc.stderr
    metadata = json.loads(meta_proc.stdout)
    assert metadata["repo"] == repo_url
    assert metadata["ref"] == repo_ref


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_UBUNTU") != "1"
    or os.environ.get("NPA_BYOF_LIVE_GPU") != "1",
    reason="Set NPA_BYOF_LIVE_UBUNTU=1 and NPA_BYOF_LIVE_GPU=1 for Ubuntu container-verify SkyPilot smoke.",
)
def test_live_byof_ubuntu_oss_container_verify_submit(
    e2e_project: str | None,
    live_byof_ubuntu_built_image: str,
) -> None:
    registry = resolve_container_registry(e2e_project)
    yaml_override = resolve_byof_resource_yaml(
        e2e_project, smoke=True, workload="container-verify"
    )
    cmd = [
        sys.executable,
        str(BYOF_RUNNER),
        "--registry",
        registry,
        "--project",
        e2e_project or "",
        "--image",
        live_byof_ubuntu_built_image,
        "--yaml",
        yaml_override,
        "--workload",
        "container-verify",
        "--run-id",
        f"byof-ubuntu-verify-{os.getpid()}",
        "--skip-build",
    ]
    config_path = skypilot_config_for_project(e2e_project)
    if config_path:
        cmd.extend(["--config-path", config_path])
    env = dict(os.environ)
    target = resolve_byof_kubernetes_target(e2e_project)
    if target.kubeconfig:
        env["KUBECONFIG"] = target.kubeconfig
    if target.context:
        env["NPA_BYOF_K8S_CONTEXT"] = target.context
    skypilot_bin = resolve_skypilot_bin()
    if skypilot_bin:
        env["PATH"] = f"{Path(skypilot_bin).parent}:{env.get('PATH', '')}"
    env.setdefault("NPA_ISAAC_LAB_ACCEPT_PRECHECK_FAILURE", "1")
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=int(os.environ.get("NPA_BYOF_LIVE_TIMEOUT", "3600")),
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary.get("status") == "ok", summary


def _plan_flag(argv: list[str], flag: str) -> str:
    index = argv.index(flag)
    assert index + 1 < len(argv), argv
    return argv[index + 1]


def _immutable_image_digest(image: str) -> str:
    normalized = image.removeprefix("docker:")
    assert normalized.count("@") == 1
    digest = normalized.rsplit("@", 1)[1]
    assert digest.startswith("sha256:") and len(digest) == 71
    assert all(character in "0123456789abcdef" for character in digest[7:])
    return digest


def _gymnasium_direct_runner_command(
    argv: list[str], *, image: str, run_id: str
) -> list[str]:
    command = [
        sys.executable,
        str(GYMNASIUM_CONTAINER_VERIFY_RUNNER),
        "--image",
        image,
        "--run-id",
        run_id,
    ]
    for flag in (
        "--yaml",
        "--output-root",
        "--smoke-command",
        "--solution-name",
        "--capability-name",
        "--smoke-artifact-name",
        "--wait-timeout",
        "--poll-interval",
    ):
        command.extend([flag, _plan_flag(argv, flag)])
    # The owner-side harness is the sole cleanup authority so its exact
    # sky-down result and namespace-baseline receipt remain observable.
    command.append("--no-cleanup")
    return command


def _assert_gymnasium_identity(artifact: dict[str, object]) -> None:
    assert artifact["solution"] == "gymnasium-robotics"
    assert (
        artifact["capability"]
        == "HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1"
    )
    exercised = set(artifact["capabilities_exercised"])
    assert {
        "registered_shadow_hand_environment",
        "mujoco_physics_steps",
        "continuous_touch_sensor_response",
        "mujoco_contacts",
        "egl_rgb_rendering",
        "rtx_pro_6000_blackwell_execution",
    } <= exercised

    source = artifact["source"]
    assert source["commit"] == "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
    assert source["package_version"] == "1.4.2"
    assert len(source["directly_loaded_xml_sha256"]) == 5
    assert len(source["directly_loaded_mesh_texture_sha256"]) == 14
    assert artifact["mujoco"]["python_version"] == "3.12.0"
    assert artifact["mujoco"]["native_version"] == "3.12.0"


def _assert_gymnasium_physics(artifact: dict[str, object]) -> None:
    environment = artifact["environment"]
    assert environment["reset_seed"] == 20260910
    assert environment["observation_shape"] == [153]
    assert environment["achieved_goal_shape"] == [7]
    assert environment["desired_goal_shape"] == [7]
    assert environment["synthetic_only_fixture"] is False
    physics = artifact["physics"]
    assert physics["environment_steps"] == 120
    assert physics["physics_substeps"] == 2400
    assert physics["finite_reward_count"] == 120
    assert physics["contact_count_sum"] > 0
    assert physics["steps_with_contacts"] > 0
    assert physics["step_calls_per_second"] > 0
    assert physics["physics_substeps_per_second"] > 0
    touch = artifact["touch_sensors"]
    assert touch["shape"] == [92]
    assert touch["nonzero_reading_count"] > 0
    assert touch["steps_with_nonzero_readings"] > 0
    assert touch["max_reading"] > 0
    transition = artifact["state_transition"]
    assert transition["max_object_position_delta_m"] > 1e-6
    assert transition["max_object_orientation_delta_rad"] > 0
    assert transition["max_full_qpos_delta_l2"] > 0


def _assert_gymnasium_rendering(artifact: dict[str, object]) -> None:
    rendering = artifact["rendering"]
    assert rendering["backend"] == "egl"
    assert rendering["rgb_frame_count"] >= 2
    assert rendering["rgb_frame_shapes"] == [[240, 320, 3]]
    assert len(rendering["distinct_rgb_frame_sha256"]) >= 2
    assert rendering["render_calls_per_second"] > 0
    assert any(
        "libEGL" in value["path"] and "nvidia" in value["path"].lower()
        for value in rendering["loaded_gl_egl_libraries"]
    )
    assert all(
        len(value["sha256"]) == 64 for value in rendering["loaded_gl_egl_libraries"]
    )


def _assert_gymnasium_runtime(
    artifact: dict[str, object], *, expected_digest: str
) -> None:
    runtime = artifact["runtime"]
    assert runtime["expected_image_digest"] == expected_digest
    assert runtime["pod_observed_image_digest"] == expected_digest
    assert runtime["gpu_count"] == 1
    assert runtime["exit_status"] == 0
    gpu = runtime["gpus"][0]
    assert "RTX PRO 6000" in gpu["name"] and "Blackwell" in gpu["name"]
    assert gpu["architecture"] == "Blackwell"
    assert gpu["compute_capability"] == "12.0"
    cache = runtime["cache_receipt"]
    assert cache["status"] == "ready"
    assert cache["source_commit"] == ("4d1ebecbc6436806cfbc0e42ebc36f594d05844e")
    assert cache["mujoco_version"] == "3.12.0"
    assert len(cache["manifest_sha256"]) == 64
    assert len(cache["receipt_sha256"]) == 64


def _assert_gymnasium_artifact_envelope(
    artifact: dict[str, object], *, payload: bytes
) -> None:
    embedded = artifact["artifact"]
    assert embedded["filename"] == "gymnasium-robotics-smoke.json"
    assert embedded["media_type"] == "application/json"
    assert embedded["size_bytes"] == len(payload)
    expected_normalized_hash = embedded["sha256"]
    embedded["sha256"] = "0" * 64
    normalized = (
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(normalized).hexdigest() == expected_normalized_hash


def _assert_gymnasium_robotics_artifact(
    payload: bytes, *, expected_digest: str
) -> dict[str, object]:
    artifact = json.loads(payload)
    _assert_gymnasium_identity(artifact)
    _assert_gymnasium_physics(artifact)
    _assert_gymnasium_rendering(artifact)
    _assert_gymnasium_runtime(artifact, expected_digest=expected_digest)
    _assert_gymnasium_artifact_envelope(artifact, payload=payload)
    return artifact


def _gymnasium_live_command(
    e2e_project: str | None,
) -> tuple[list[str], str, str]:
    run_id = f"gymnasium-robotics-{time.time_ns()}"
    spec = load_spec(GYMNASIUM_ROBOTICS_SPEC)
    spec.config["bucket"] = live_bucket(e2e_project)
    authorized_output_root = os.environ.get(
        "NPA_BYOF_GYMNASIUM_ROBOTICS_OUTPUT_ROOT", ""
    ).strip()
    assert authorized_output_root.startswith("s3://"), (
        "NPA_BYOF_GYMNASIUM_ROBOTICS_OUTPUT_ROOT must be the manager-authorized "
        "task-owned S3 prefix"
    )
    parsed_output_root = urlparse(authorized_output_root)
    assert parsed_output_root.netloc and parsed_output_root.path.strip("/"), (
        "the manager-authorized output root must include a bucket and prefix"
    )
    preset = os.environ.get("NPA_BYOF_GYMNASIUM_ROBOTICS_IMAGE", "").strip()
    assert preset, (
        "NPA_BYOF_GYMNASIUM_ROBOTICS_IMAGE must be the already scanned, "
        "task-private immutable image reference"
    )
    _immutable_image_digest(preset)
    spec.config["output_root"] = authorized_output_root.rstrip("/")
    spec.config["base_image"] = preset
    plan = build_plan(spec, run_id=run_id)
    assert len(plan.steps) == 1
    argv = list(plan.steps[0].argv)
    assert argv[:4] == ["npa", "workbench", "byof", "run"]
    assert _plan_flag(argv, "--repo-ref") == (
        "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
    )
    assert _plan_flag(argv, "--yaml") == (
        "byof-solution-smoke-gymnasium-robotics-rtxpro-gpu"
    )
    assert _plan_flag(argv, "--base-image") == preset
    cmd = _gymnasium_direct_runner_command(argv, image=preset, run_id=run_id)
    config_path = skypilot_config_for_project(e2e_project)
    if config_path:
        cmd.extend(["--config-path", config_path])
    return cmd, _plan_flag(argv, "--output-root"), run_id


def _gymnasium_live_env(e2e_project: str | None) -> dict[str, str]:
    env = dict(os.environ)
    target = resolve_byof_kubernetes_target(e2e_project)
    if target.kubeconfig:
        env["KUBECONFIG"] = target.kubeconfig
        env["NPA_BYOF_KUBECONFIG"] = target.kubeconfig
    if target.context:
        env["NPA_BYOF_K8S_CONTEXT"] = target.context
    sky_bin = resolve_skypilot_bin()
    if sky_bin:
        env["PATH"] = f"{Path(sky_bin).parent}:{env.get('PATH', '')}"
    return env


def _gymnasium_kubectl(
    env: dict[str, str],
    namespace: str,
    *args: str,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    kubeconfig = env.get("NPA_BYOF_KUBECONFIG", "").strip()
    context = env.get("NPA_BYOF_K8S_CONTEXT", "").strip()
    assert kubeconfig and context
    command = [
        "kubectl",
        "--kubeconfig",
        kubeconfig,
        "--context",
        context,
        "--namespace",
        namespace,
        *args,
    ]
    return subprocess.run(
        command,
        input=stdin,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=GYMNASIUM_KUBECTL_TIMEOUT_SECONDS,
    )


def _require_gymnasium_scheduling_contract(
    env: dict[str, str],
    *,
    namespace: str,
    config_path: str | None,
) -> None:
    assert config_path, "a task-private SkyPilot config is required"
    sky_config = Path(config_path)
    assert sky_config.stat().st_mode & 0o777 == 0o600, (
        "the task-private SkyPilot config must be mode 0600"
    )
    document = yaml.safe_load(sky_config.read_text(encoding="utf-8")) or {}
    kubernetes = document.get("kubernetes", {})
    assert isinstance(kubernetes, dict)
    allowed = kubernetes.get("allowed_nodes")
    assert isinstance(allowed, dict) and set(allowed) == {"names"}, (
        "SkyPilot kubernetes.allowed_nodes must use only the supported names mapping"
    )
    names = allowed.get("names")
    assert (
        isinstance(names, list)
        and len(names) == 1
        and isinstance(names[0], str)
        and names[0].strip()
    ), "SkyPilot must allow exactly one named node"

    kubeconfig = Path(env.get("NPA_BYOF_KUBECONFIG", "").strip())
    context = env.get("NPA_BYOF_K8S_CONTEXT", "").strip()
    assert context and kubeconfig.is_file(), (
        "a child-local kubeconfig and child-specific context are required"
    )
    assert kubeconfig.stat().st_mode & 0o777 == 0o600, (
        "the child-local kubeconfig must be mode 0600"
    )
    kube_document = yaml.safe_load(kubeconfig.read_text(encoding="utf-8")) or {}
    assert kube_document.get("current-context") == context
    contexts = kube_document.get("contexts", [])
    assert isinstance(contexts, list) and len(contexts) == 1
    assert contexts[0].get("name") == context
    assert contexts[0].get("context", {}).get("namespace") == namespace
    assert kubernetes.get("allowed_contexts") == [context], (
        "SkyPilot must allow only the selected child-specific context"
    )

    expected_name = env.get("NPA_BYOF_GYMNASIUM_ROBOTICS_NODE_NAME", "").strip()
    expected_uid = env.get("NPA_BYOF_GYMNASIUM_ROBOTICS_NODE_UID", "").strip()
    provider_group = env.get(
        "NPA_BYOF_GYMNASIUM_ROBOTICS_PROVIDER_NODE_GROUP_ID", ""
    ).strip()
    assert expected_name and expected_uid and provider_group, (
        "manager-issued node name, UID, and provider-group evidence are required"
    )
    assert names == [expected_name], (
        "the one allowed SkyPilot node must be the manager-assigned node"
    )

    result = _gymnasium_kubectl(
        env, namespace, "get", "node", expected_name, "--output", "json"
    )
    assert result.returncode == 0, "the one allowed node is not readable"
    node = json.loads(result.stdout)
    metadata = node.get("metadata", {})
    assert metadata.get("name") == expected_name
    assert metadata.get("uid") == expected_uid
    labels = metadata.get("labels", {})
    assert sum(value == provider_group for value in labels.values()) == 1, (
        "the allowed node must match exactly one provider-group label"
    )
    ready = [
        condition
        for condition in node.get("status", {}).get("conditions", [])
        if condition.get("type") == "Ready"
    ]
    assert len(ready) == 1 and ready[0].get("status") == "True"
    assert node.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu") == "1"
    products = [value for key, value in labels.items() if "gpu.product" in str(key)]
    assert len(products) == 1
    product = str(products[0]).lower().replace("_", "-")
    assert "rtx-pro-6000" in product and "blackwell" in product


def _gymnasium_evidence_dir(env: dict[str, str]) -> Path:
    configured = env.get("NPA_BYOF_GYMNASIUM_ROBOTICS_EVIDENCE_DIR", "").strip()
    assert configured, (
        "NPA_BYOF_GYMNASIUM_ROBOTICS_EVIDENCE_DIR must be an owner-private path"
    )
    evidence_dir = Path(configured).resolve()
    evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert evidence_dir.stat().st_mode & 0o077 == 0, (
        "owner-side evidence directory must not be group/world accessible"
    )
    return evidence_dir


def _new_gymnasium_private_file(path: Path) -> TextIO:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _require_gymnasium_owner_receipt_access(
    env: dict[str, str], *, namespace: str
) -> None:
    required = [("list", resource) for resource in GYMNASIUM_CLEANUP_RESOURCES]
    required.append(("create", "pods/exec"))
    for verb, resource in required:
        result = _gymnasium_kubectl(env, namespace, "auth", "can-i", verb, resource)
        assert result.returncode == 0 and result.stdout.strip() == "yes", (
            "the owner identity needs narrowly scoped "
            f"{verb} {resource} in the manager-authorized namespace; do not "
            "grant the workload service account Pod access"
        )


def _exact_gymnasium_run_pods(
    env: dict[str, str],
    *,
    namespace: str,
    run_id: str,
    include_terminating: bool = False,
) -> list[dict[str, object]]:
    result = _gymnasium_kubectl(
        env,
        namespace,
        "get",
        "pods",
        "--selector",
        "parent=skypilot",
        "--output",
        "json",
    )
    assert result.returncode == 0, (
        "owner-side Pod lookup failed; no workload RBAC change is permitted"
    )
    items = json.loads(result.stdout).get("items", [])
    pods = [
        item
        for item in items
        if item.get("metadata", {}).get("labels", {}).get("parent") == "skypilot"
        and item.get("metadata", {}).get("annotations", {}).get("skypilot-cluster-name")
        == run_id
        and (
            include_terminating or not item.get("metadata", {}).get("deletionTimestamp")
        )
    ]
    assert len(pods) <= 1, "expected at most one exact SkyPilot run Pod"
    return pods


def _gymnasium_gpu_quantity(item: dict[str, object], kind: str) -> int:
    containers = item.get("spec", {}).get("containers", [])
    return sum(
        int(container.get("resources", {}).get(kind, {}).get("nvidia.com/gpu", 0))
        for container in containers
    )


def _gymnasium_namespace_inventory(
    env: dict[str, str], *, namespace: str
) -> dict[str, list[dict[str, object]]]:
    inventory: dict[str, list[dict[str, object]]] = {}
    for resource in GYMNASIUM_CLEANUP_RESOURCES:
        result = _gymnasium_kubectl(env, namespace, "get", resource, "--output", "json")
        assert result.returncode == 0, f"cannot inventory namespace {resource}"
        items = json.loads(result.stdout).get("items", [])
        records = []
        for item in items:
            metadata = item.get("metadata", {})
            record = {"name": metadata.get("name"), "uid": metadata.get("uid")}
            if resource == "pods":
                record["gpu_requests"] = _gymnasium_gpu_quantity(item, "requests")
                record["gpu_limits"] = _gymnasium_gpu_quantity(item, "limits")
            records.append(record)
        inventory[resource] = sorted(records, key=lambda value: str(value["name"]))
    return inventory


def _require_gymnasium_empty_workload_baseline(
    inventory: dict[str, list[dict[str, object]]],
) -> None:
    for resource in (
        "roles",
        "rolebindings",
        "pods",
        "jobs",
        "persistentvolumeclaims",
    ):
        assert inventory[resource] == [], f"namespace baseline contains {resource}"
    assert sum(item.get("gpu_requests", 0) for item in inventory["pods"]) == 0
    assert sum(item.get("gpu_limits", 0) for item in inventory["pods"]) == 0


def _seal_gymnasium_namespace_baseline(
    env: dict[str, str], *, run_id: str, inventory: dict[str, object]
) -> None:
    path = _gymnasium_evidence_dir(env) / f"{run_id}-namespace-baseline.json"
    with _new_gymnasium_private_file(path) as stream:
        json.dump(inventory, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _wait_for_gymnasium_namespace_baseline(
    env: dict[str, str],
    *,
    namespace: str,
    baseline: dict[str, list[dict[str, object]]],
) -> bool:
    deadline = time.monotonic() + GYMNASIUM_CLEANUP_TIMEOUT_SECONDS
    while _gymnasium_namespace_inventory(env, namespace=namespace) != baseline:
        if time.monotonic() >= deadline:
            return False
        time.sleep(2)
    return True


def _gymnasium_sky_cluster_absent(
    env: dict[str, str], *, run_id: str, config_path: str | None
) -> bool:
    sky_bin = resolve_skypilot_bin()
    assert sky_bin, "SkyPilot executable is required for cluster absence proof"
    command = [sky_bin, "status"]
    if config_path:
        command.extend(["--config", config_path])
    command.extend(["--output", "json"])
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=GYMNASIUM_SKY_DOWN_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, "SkyPilot cluster status lookup failed"
    payload = json.loads(result.stdout)
    assert isinstance(payload, list), "unexpected SkyPilot cluster status schema"
    clusters = payload
    assert all(
        isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and bool(item["name"].strip())
        for item in clusters
    ), "unexpected SkyPilot cluster record schema"
    return not any(item.get("name") == run_id for item in clusters)


def _gymnasium_admitted_pod_identity(
    pod: dict[str, object], *, namespace: str, run_id: str
) -> dict[str, object]:
    assert pod.get("apiVersion") == "v1" and pod.get("kind") == "Pod", (
        "admitted workload must be a v1 Pod"
    )
    metadata = pod.get("metadata")
    assert isinstance(metadata, dict), "admitted Pod metadata must be an object"
    for key in ("name", "uid", "resourceVersion"):
        value = metadata.get(key)
        assert isinstance(value, str) and value and value == value.strip(), (
            "admitted Pod requires exact name, UID and resource version"
        )
    assert metadata.get("namespace") == namespace, "unexpected admitted Pod namespace"
    assert not metadata.get("deletionTimestamp"), "admitted Pod is terminating"
    assert metadata.get("labels", {}).get("parent") == "skypilot"
    assert metadata.get("annotations", {}).get("skypilot-cluster-name") == run_id
    return {
        key: metadata[key] for key in ("name", "uid", "namespace", "resourceVersion")
    }


def _gymnasium_admitted_volumes(spec: dict[str, object]) -> dict[str, object]:
    # The only approved mount is anonymous memory for local IPC. Anything else
    # needs its own reviewed configuration, never an inferred admission allowance.
    volumes = spec.get("volumes", [])
    mounts = spec["containers"][0].get("volumeMounts", [])
    assert isinstance(volumes, list) and isinstance(mounts, list), (
        "admitted Pod volume and mount populations must be lists"
    )
    if not volumes:
        assert mounts == [], "admitted Pod has an unapproved mount"
        return {"volumes": [], "mounts": []}
    assert volumes == [{"name": "dshm", "emptyDir": {"medium": "Memory"}}], (
        "admitted Pod has an unapproved volume population"
    )
    assert len(mounts) == 1 and isinstance(mounts[0], dict)
    mount = dict(mounts[0])
    if "readOnly" in mount:
        assert mount.pop("readOnly") is False, "unexpected shared-memory mount mode"
    if "mountPropagation" in mount:
        assert mount.pop("mountPropagation") == "None", "mount propagation forbidden"
    assert mount == {
        "name": "dshm",
        "mountPath": PurePosixPath("/dev", "shm").as_posix(),
    }, "admitted Pod has an unapproved mount population"
    return {"volumes": volumes, "mounts": mounts}


def _gymnasium_admitted_pod_policy(
    pod: dict[str, object], *, namespace: str, run_id: str
) -> dict[str, object]:
    identity = _gymnasium_admitted_pod_identity(pod, namespace=namespace, run_id=run_id)
    spec = pod.get("spec")
    assert isinstance(spec, dict), "admitted Pod spec must be an object"
    assert spec.get("serviceAccountName") in {"default", "skypilot-service-account"}, (
        "admitted Pod service account is missing or unapproved"
    )
    if "serviceAccount" in spec:
        assert spec["serviceAccount"] == spec["serviceAccountName"]
    for key in ("initContainers", "ephemeralContainers"):
        assert key not in spec or spec[key] == [], (
            "admitted Pod has injected containers"
        )
    documents = [{"config": {"kubernetes": {"pod_config": {"spec": spec}}}}]
    try:
        selected = validate_gymnasium_task_configuration(
            documents, solution_name="gymnasium-robotics"
        )
    except ExecutionPreflightError as exc:
        raise AssertionError(
            "admitted Pod violates Gymnasium isolation policy"
        ) from exc
    assert selected, "admitted Pod isolation policy was not applied"
    populations = _gymnasium_admitted_volumes(spec)
    container = spec["containers"][0]
    return {
        "policy": "gymnasium-admitted-pod.v1",
        "identity": identity,
        "service_account": spec["serviceAccountName"],
        "automount_service_account_token": spec["automountServiceAccountToken"],
        "container_names": [container["name"]],
        **populations,
        "pod_security_context": spec["securityContext"],
        "container_security_context": container["securityContext"],
        "spec_sha256": hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _gymnasium_pod_image_receipt(
    proc: subprocess.Popen[str],
    *,
    env: dict[str, str],
    namespace: str,
    run_id: str,
    image: str,
) -> dict[str, object]:
    expected_image = image.removeprefix("docker:")
    expected_digest = _immutable_image_digest(expected_image)
    expected_node_name = env.get("NPA_BYOF_GYMNASIUM_ROBOTICS_NODE_NAME", "").strip()
    assert expected_node_name, (
        "NPA_BYOF_GYMNASIUM_ROBOTICS_NODE_NAME must identify the assigned node"
    )
    deadline = time.monotonic() + GYMNASIUM_RECEIPT_TIMEOUT_SECONDS
    while proc.poll() is None:
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for the exact Pod image receipt")
        pods = _exact_gymnasium_run_pods(env, namespace=namespace, run_id=run_id)
        if not pods:
            time.sleep(2)
            continue
        pod = pods[0]
        metadata = pod.get("metadata", {})
        assert str(metadata.get("namespace", "")) == namespace, (
            "the exact run Pod must belong to the manager-authorized namespace"
        )
        phase = pod.get("status", {}).get("phase")
        assert phase in {"Pending", "Running"}
        spec = pod.get("spec", {})
        observed_node_name = str(spec.get("nodeName", "")).strip()
        if not observed_node_name and phase == "Pending":
            time.sleep(2)
            continue
        assert observed_node_name == expected_node_name, (
            "the exact run Pod must execute on the manager-assigned node"
        )
        containers = spec.get("containers", [])

        def gpu_quantity(container: dict[str, object], kind: str) -> int:
            return int(
                container.get("resources", {}).get(kind, {}).get("nvidia.com/gpu", 0)
            )

        matching_containers = [
            container
            for container in containers
            if str(container.get("image", "")).removeprefix("docker:") == expected_image
            and gpu_quantity(container, "requests") == 1
            and gpu_quantity(container, "limits") == 1
        ]
        assert len(matching_containers) == 1, (
            "the exact run Pod must contain one immutable one-GPU task container"
        )
        container = matching_containers[0]
        gpu_requests = gpu_quantity(container, "requests")
        gpu_limits = gpu_quantity(container, "limits")
        assert gpu_requests == gpu_limits == 1, (
            "the exact run Pod must request and limit one GPU"
        )
        other_containers = [item for item in containers if item is not container]
        assert all(
            gpu_quantity(item, kind) == 0
            for item in other_containers
            for kind in ("requests", "limits")
        ), "only the exact task container may request the one GPU"
        admitted_policy = _gymnasium_admitted_pod_policy(
            pod, namespace=namespace, run_id=run_id
        )
        statuses = {
            item.get("name"): item
            for item in pod.get("status", {}).get("containerStatuses", [])
        }
        status = statuses.get(container.get("name"), {})
        waiting = status.get("state", {}).get("waiting", {})
        assert waiting.get("reason") not in {
            "CreateContainerConfigError",
            "CreateContainerError",
            "ErrImagePull",
            "ImagePullBackOff",
            "InvalidImageName",
        }, f"exact run Pod cannot start: {waiting.get('reason', 'unknown')}"
        if not status.get("state", {}).get("running"):
            time.sleep(2)
            continue
        image_id = str(status.get("imageID", ""))
        observed_digests = set(re.findall(r"sha256:[0-9a-f]{64}", image_id.lower()))
        assert observed_digests == {expected_digest}, (
            "Kubernetes imageID differs from the scanned immutable image"
        )
        receipt: dict[str, object] = {
            "schema_version": "npa.byof.pod-image-receipt.v1",
            "source": "owner-side-kubernetes-status",
            "run_id": run_id,
            "pod_name": str(metadata.get("name", "")),
            "pod_namespace": str(metadata.get("namespace", "")),
            "pod_uid": str(metadata.get("uid", "")),
            "node_name": observed_node_name,
            "container_name": str(container.get("name", "")),
            "spec_image": str(container.get("image", "")).removeprefix("docker:"),
            "image_id": image_id,
            "expected_digest": expected_digest,
            "observed_digest": observed_digests.pop(),
            "observed_unix": round(time.time(), 3),
            "admitted_pod_policy": admitted_policy,
        }
        assert receipt["pod_name"] and receipt["pod_uid"]
        assert receipt["pod_namespace"] == namespace
        encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        in_pod_path = f"/workspace/byof-runs/{run_id}/npa_pod_image_receipt.json"
        writer = (
            "import os,pathlib,sys;"
            "path=pathlib.Path(sys.argv[1]);path.parent.mkdir(parents=True,exist_ok=True);"
            "tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(sys.stdin.buffer.read());"
            "os.chmod(tmp,0o600);tmp.replace(path)"
        )
        injected = _gymnasium_kubectl(
            env,
            namespace,
            "exec",
            "--stdin",
            str(receipt["pod_name"]),
            "--container",
            str(receipt["container_name"]),
            "--",
            "/usr/bin/python3",
            "-c",
            writer,
            in_pod_path,
            stdin=encoded,
        )
        assert injected.returncode == 0, (
            "owner-side receipt injection into the exact run Pod failed"
        )
        receipt_path = _gymnasium_evidence_dir(env) / (
            f"{run_id}-pod-image-receipt.json"
        )
        with _new_gymnasium_private_file(receipt_path) as stream:
            stream.write(encoded)
        return receipt
    raise AssertionError(
        "BYOF runner exited before an exact Pod image receipt was written"
    )


def _terminate_gymnasium_runner(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=GYMNASIUM_RUNNER_TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=GYMNASIUM_RUNNER_TERM_GRACE_SECONDS)


def _record_gymnasium_sky_down_logs(
    evidence_dir: Path,
    *,
    run_id: str,
    attempt: int,
    stdout: str | bytes,
    stderr: str | bytes,
) -> None:
    for suffix, content in (("stdout", stdout), ("stderr", stderr)):
        decoded = (
            content.decode(errors="replace") if isinstance(content, bytes) else content
        )
        path = evidence_dir / f"{run_id}-sky-down-{attempt}-{suffix}.log"
        with _new_gymnasium_private_file(path) as stream:
            stream.write(decoded)


def _issue_gymnasium_sky_down(
    env: dict[str, str], *, run_id: str, config_path: str | None, attempt: int
) -> None:
    sky_bin = resolve_skypilot_bin()
    assert sky_bin, "SkyPilot executable is required for exact-run cleanup"
    command = [sky_bin, "down"]
    if config_path:
        command.extend(["--config", config_path])
    command.extend(["--yes", run_id])
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=GYMNASIUM_SKY_DOWN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        _record_gymnasium_sky_down_logs(
            _gymnasium_evidence_dir(env),
            run_id=run_id,
            attempt=attempt,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )
        raise AssertionError("exact-run SkyPilot cleanup timed out") from exc
    _record_gymnasium_sky_down_logs(
        _gymnasium_evidence_dir(env),
        run_id=run_id,
        attempt=attempt,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    assert result.returncode == 0, "exact-run SkyPilot cleanup command failed"


def _cleanup_gymnasium_run(
    env: dict[str, str],
    *,
    namespace: str,
    run_id: str,
    config_path: str | None,
    namespace_baseline: dict[str, list[dict[str, object]]],
    issue_down: bool,
    cleanup_attempt: int = 1,
) -> None:
    if issue_down:
        _issue_gymnasium_sky_down(
            env,
            run_id=run_id,
            config_path=config_path,
            attempt=cleanup_attempt,
        )
    assert _gymnasium_sky_cluster_absent(env, run_id=run_id, config_path=config_path), (
        "the exact SkyPilot cluster still exists"
    )
    assert _wait_for_gymnasium_namespace_baseline(
        env, namespace=namespace, baseline=namespace_baseline
    ), "namespace did not return to its sealed pre-run inventory"


def _cleanup_gymnasium_run_after_success(
    env: dict[str, str],
    *,
    namespace: str,
    run_id: str,
    config_path: str | None,
    namespace_baseline: dict[str, list[dict[str, object]]],
) -> None:
    first_error: BaseException | None = None
    for attempt in (1, 2):
        try:
            _cleanup_gymnasium_run(
                env,
                namespace=namespace,
                run_id=run_id,
                config_path=config_path,
                namespace_baseline=namespace_baseline,
                issue_down=True,
                cleanup_attempt=attempt,
            )
        except BaseException as error:
            if first_error is None:
                first_error = error
            else:
                first_error.add_note(f"cleanup retry also failed: {error}")
            continue
        return
    assert first_error is not None
    raise first_error


def _gymnasium_expected_digest(summary: dict[str, object]) -> str:
    assert summary.get("mode") == "direct-launch"
    final = summary.get("final")
    assert isinstance(final, dict)
    assert final.get("status") == "SUCCEEDED"
    assert final.get("returncode") == 0
    image = os.environ.get("NPA_BYOF_GYMNASIUM_ROBOTICS_IMAGE", "").strip()
    return _immutable_image_digest(image)


def _require_gymnasium_hash_metadata(response: dict, digest: str) -> None:
    hashes = [
        value
        for name, value in response.get("Metadata", {}).items()
        if name.lower() == "sha256"
    ]
    assert hashes == [digest], "qualification object hash metadata mismatch"


def _gymnasium_remote_evidence(
    e2e_project: str | None, *, root_uri: str
) -> tuple[bytes, dict[str, object]]:
    parsed = urlparse(root_uri)
    assert parsed.scheme == "s3" and parsed.netloc
    s3 = s3_client_for_project(
        e2e_project,
        allow_host_creds=True,
        endpoint_url=os.environ.get("NPA_BYOF_S3_ENDPOINT", ""),
    )
    prefix = parsed.path.lstrip("/")
    expected_keys = {
        prefix + "gymnasium-robotics-smoke.json",
        prefix + "npa_byof_summary.json",
    }
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=parsed.netloc, Prefix=prefix
    )
    observed_keys = {item["Key"] for page in pages for item in page.get("Contents", [])}
    assert observed_keys == expected_keys, "unexpected qualification output object"

    def read_json(name: str) -> bytes:
        response = s3.get_object(Bucket=parsed.netloc, Key=prefix + name)
        payload = response["Body"].read()
        assert response.get("ContentLength") == len(payload)
        assert response.get("ContentType") == "application/json"
        _require_gymnasium_hash_metadata(response, hashlib.sha256(payload).hexdigest())
        json.loads(payload)
        return payload

    artifact = read_json("gymnasium-robotics-smoke.json")
    summary = json.loads(read_json("npa_byof_summary.json"))
    return artifact, summary


def _cleanup_gymnasium_failed_output(e2e_project: str | None, *, root_uri: str) -> None:
    parsed = urlparse(root_uri)
    assert parsed.scheme == "s3" and parsed.netloc
    prefix = parsed.path.lstrip("/")
    assert prefix and prefix.endswith("/"), "failed-run prefix must be exact"
    s3 = s3_client_for_project(
        e2e_project,
        allow_host_creds=True,
        endpoint_url=os.environ.get("NPA_BYOF_S3_ENDPOINT", ""),
    )
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=parsed.netloc, Prefix=prefix
    )
    keys = [item["Key"] for page in pages for item in page.get("Contents", [])]
    for key in keys:
        assert key.startswith(prefix)
        s3.delete_object(Bucket=parsed.netloc, Key=key)
    remaining = s3.list_objects_v2(Bucket=parsed.netloc, Prefix=prefix)
    assert not remaining.get("Contents"), "failed-run output prefix is not empty"


def _require_gymnasium_output_prefix_empty(
    e2e_project: str | None, *, root_uri: str
) -> None:
    parsed = urlparse(root_uri)
    assert parsed.scheme == "s3" and parsed.netloc
    prefix = parsed.path.lstrip("/")
    assert prefix and prefix.endswith("/"), "run output prefix must be exact"
    s3 = s3_client_for_project(
        e2e_project,
        allow_host_creds=True,
        endpoint_url=os.environ.get("NPA_BYOF_S3_ENDPOINT", ""),
    )
    response = s3.list_objects_v2(Bucket=parsed.netloc, Prefix=prefix, MaxKeys=1)
    assert not response.get("Contents"), "exact run output prefix is not empty"


@pytest.mark.public_inputs
@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_GYMNASIUM_ROBOTICS_LIVE_GPU") != "1",
    reason=(
        "Set NPA_BYOF_GYMNASIUM_ROBOTICS_LIVE_GPU=1 only with manager-owned "
        "STRICT RTX PRO 6000 capacity and child cloud authorization."
    ),
)
def test_live_gymnasium_robotics_exact_digest_capability(
    e2e_project: str | None,
) -> None:
    cmd, output_root, run_id = _gymnasium_live_command(e2e_project)
    run_root = output_root.rstrip("/") + f"/{run_id}/"
    _require_gymnasium_output_prefix_empty(e2e_project, root_uri=run_root)
    env = _gymnasium_live_env(e2e_project)
    namespace = os.environ.get("NPA_BYOF_GYMNASIUM_ROBOTICS_NAMESPACE", "").strip()
    assert namespace, (
        "NPA_BYOF_GYMNASIUM_ROBOTICS_NAMESPACE must be the manager-authorized "
        "task namespace"
    )
    evidence_dir = _gymnasium_evidence_dir(env)
    _require_gymnasium_owner_receipt_access(env, namespace=namespace)
    config_path = skypilot_config_for_project(e2e_project)
    _require_gymnasium_scheduling_contract(
        env, namespace=namespace, config_path=config_path
    )
    namespace_baseline = _gymnasium_namespace_inventory(env, namespace=namespace)
    _require_gymnasium_empty_workload_baseline(namespace_baseline)
    assert _gymnasium_sky_cluster_absent(env, run_id=run_id, config_path=config_path), (
        "the exact run already exists before submission"
    )
    _seal_gymnasium_namespace_baseline(env, run_id=run_id, inventory=namespace_baseline)
    stdout_path = evidence_dir / f"{run_id}-runner-stdout.log"
    stderr_path = evidence_dir / f"{run_id}-runner-stderr.log"
    proc: subprocess.Popen[str] | None = None
    cleanup_started = False
    try:
        with (
            _new_gymnasium_private_file(stdout_path) as stdout_stream,
            _new_gymnasium_private_file(stderr_path) as stderr_stream,
        ):
            proc = subprocess.Popen(
                cmd,
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
                cwd=str(REPO_ROOT),
                env=env,
                start_new_session=True,
            )
            receipt = _gymnasium_pod_image_receipt(
                proc,
                env=env,
                namespace=namespace,
                run_id=run_id,
                image=os.environ["NPA_BYOF_GYMNASIUM_ROBOTICS_IMAGE"],
            )
            returncode = proc.wait(timeout=GYMNASIUM_RUNNER_TIMEOUT_SECONDS)
            if returncode != 0:
                stdout_stream.flush()
                stderr_stream.flush()
                raise AssertionError(
                    f"BYOF runner exited {returncode}; inspect owner-private logs"
                )
        cleanup_started = True
        _cleanup_gymnasium_run_after_success(
            env,
            namespace=namespace,
            run_id=run_id,
            config_path=config_path,
            namespace_baseline=namespace_baseline,
        )
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        credential_markers = live_credential_markers()
        assert_no_credential_leakage(
            stdout + "\n" + stderr, extra_forbidden=credential_markers
        )
        summary = _parse_last_json_blob(stdout + "\n" + stderr)
        assert_no_credential_leakage(
            json.dumps(summary, sort_keys=True), extra_forbidden=credential_markers
        )
        expected_digest = _gymnasium_expected_digest(summary)
        assert receipt["expected_digest"] == expected_digest
        assert receipt["observed_digest"] == expected_digest
        artifact_bytes, remote_summary = _gymnasium_remote_evidence(
            e2e_project, root_uri=run_root
        )
        assert_no_credential_leakage(
            artifact_bytes.decode("utf-8"), extra_forbidden=credential_markers
        )
        assert_no_credential_leakage(
            json.dumps(remote_summary, sort_keys=True),
            extra_forbidden=credential_markers,
        )
        _assert_gymnasium_robotics_artifact(
            artifact_bytes, expected_digest=expected_digest
        )
        assert remote_summary["smoke_exit_code"] == 0
        assert remote_summary["smoke_artifact"]["size_bytes"] == len(artifact_bytes)
        assert (
            remote_summary["smoke_artifact"]["sha256"]
            == hashlib.sha256(artifact_bytes).hexdigest()
        )
        assert expected_digest in remote_summary["pod_observed_image_id"]
    except BaseException as primary_error:
        if proc is not None:
            try:
                _terminate_gymnasium_runner(proc)
            except BaseException as termination_error:
                primary_error.add_note(
                    f"BYOF runner termination also failed: {termination_error}"
                )
        if proc is not None and not cleanup_started:
            try:
                _cleanup_gymnasium_run_after_success(
                    env,
                    namespace=namespace,
                    run_id=run_id,
                    config_path=config_path,
                    namespace_baseline=namespace_baseline,
                )
            except BaseException as cleanup_error:
                primary_error.add_note(
                    f"exact-run cleanup also failed: {cleanup_error}"
                )
        try:
            _cleanup_gymnasium_failed_output(e2e_project, root_uri=run_root)
        except BaseException as output_cleanup_error:
            primary_error.add_note(
                f"failed-run output cleanup also failed: {output_cleanup_error}"
            )
        raise


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIBERO_LIVE_B200") != "1",
    reason="Set NPA_BYOF_LIBERO_LIVE_B200=1 to verify an operator-selected one-B200 qualification report.",
)
def test_libero_b200_qualification_report(e2e_project: str | None) -> None:
    """Own one customer-authorized run and independently read back every artifact."""

    qualification = libero_qualified_image_manifest()
    authorization_path = os.environ.get(
        "NPA_BYOF_LIBERO_CUSTOMER_AUTHORIZATION_FILE", ""
    ).strip()
    assert authorization_path, (
        "NPA_BYOF_LIBERO_CUSTOMER_AUTHORIZATION_FILE must select the "
        "owner-private customer/run authorization"
    )
    authorization = json.loads(Path(authorization_path).read_text(encoding="utf-8"))
    caller_path = os.environ.get(
        "NPA_BYOF_LIBERO_AUTHENTICATED_CALLER_FILE", ""
    ).strip()
    assert caller_path, (
        "NPA_BYOF_LIBERO_AUTHENTICATED_CALLER_FILE must select the owner-private "
        "short-lived authenticated-caller assertion"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", str(authorization["customer_identity_sha256"]))
    bucket = live_bucket(e2e_project)
    run_id = authorization["run_id"]
    profile = (
        REPO_ROOT
        / "npa"
        / "src"
        / "npa"
        / "workflows"
        / "byof"
        / "profiles"
        / "byof-solution-smoke-libero-b200-gpu.yaml"
    )
    cmd = [
        sys.executable,
        str(BYOF_RUNNER),
        "--repo-url",
        "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
        "--repo-ref",
        qualification["upstream_source_revision"],
        "--repo-auth",
        "none",
        "--project",
        e2e_project or "",
        "--registry",
        resolve_container_registry(e2e_project),
        "--base-profile",
        "prebuilt",
        "--base-image",
        "tool://libero",
        "--run-id",
        run_id,
        "--workload",
        "solution-smoke",
        "--smoke-command",
        "/opt/npa/libero/smoke.sh",
        "--solution-name",
        "libero",
        "--capability-name",
        "libero_spatial_bc_rnn_train_reload_heldout",
        "--smoke-artifact-name",
        "libero-smoke.json",
        "--libero-qualified-candidate-image",
        qualification["candidate_image"],
        "--libero-customer-runtime-authorization-file",
        authorization_path,
        "--libero-authenticated-caller-identity-file",
        caller_path,
        "--num-envs",
        "1",
        "--num-demos",
        "1",
        "--task",
        (
            "libero_spatial/"
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
            "and_place_it_on_the_plate"
        ),
        "--iterations",
        "1",
        "--yaml",
        str(profile),
        "--output-root",
        f"s3://{bucket}/oss-solutions/libero",
        "--wait-timeout",
        "-1",
        "--poll-interval",
        "60",
        "--cleanup",
        "--skip-build",
        "--skip-push",
    ]
    config_path = skypilot_config_for_project(e2e_project)
    if config_path:
        cmd.extend(["--config-path", config_path])
    env = dict(os.environ)
    target = resolve_byof_kubernetes_target(e2e_project)
    if target.kubeconfig:
        env["KUBECONFIG"] = target.kubeconfig
    if target.context:
        env["NPA_BYOF_K8S_CONTEXT"] = target.context
    skypilot_bin = resolve_skypilot_bin()
    if skypilot_bin:
        env["PATH"] = f"{Path(skypilot_bin).parent}:{env.get('PATH', '')}"
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=int(os.environ.get("NPA_BYOF_LIBERO_LIVE_TIMEOUT", "21600")),
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    runner_summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert runner_summary["status"] == "ok"
    assert runner_summary["image"] == qualification["candidate_image"]
    run_summary = runner_summary["run"]
    assert run_summary["run_id"] == run_id
    assert run_summary["final"]["status"] == "SUCCEEDED"
    assert run_summary["cleanup"]["ok"] is True
    assert run_summary["cleanup"]["verified"] is True
    assert run_summary["cleanup"]["remote_absence_verified"] is True
    assert set(run_summary["cleanup"]["resources_removed"]) >= {
        "libero-rolebinding",
        "libero-role",
        "libero-serviceaccount",
        "libero-namespace",
        "libero-payload-kubeconfig",
        "libero-isolated-skypilot-state",
    }
    assert run_summary["api_stop"] == {
        "ok": True,
        "preserved_for_cleanup_recovery": False,
    }
    binding = run_summary["libero_runtime_binding"]
    assert re.fullmatch(r"[0-9a-f]{64}", binding["namespace_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", binding["rbac_spec_sha256"])
    for key in (
        "controller_service_account_uid_sha256",
        "controller_role_uid_sha256",
        "controller_role_binding_uid_sha256",
        "controller_rbac_spec_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", binding[key])
    manager_live = run_summary["libero_manager_live_evidence"]
    assert manager_live["schema"] == "npa.libero.manager-live-evidence.v1"
    assert manager_live["observation_method"] == (
        "manager_kubernetes_pod_status_and_node_labels"
    )
    assert manager_live["gpu_family"] == "B200"
    assert manager_live["pod_gpu_count"] == 1
    assert manager_live["node_allocatable_gpu_count"] >= 1
    assert manager_live["pod_observed_image_digest"] == qualification["oci_digest"]
    for key in (
        "scheduler_job_id_sha256",
        "payload_pod_name_sha256",
        "payload_pod_uid_sha256",
        "skypilot_cluster_name_sha256",
        "namespace_sha256",
        "node_name_sha256",
        "node_uid_sha256",
        "service_account_uid_sha256",
        "node_gpu_products_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", manager_live[key])
    for key in (
        "scheduler_job_id_sha256",
        "payload_pod_name_sha256",
        "payload_pod_uid_sha256",
        "skypilot_cluster_name_sha256",
        "namespace_sha256",
        "service_account_uid_sha256",
    ):
        assert manager_live[key] == binding[key]

    s3 = s3_client_for_project(e2e_project, allow_host_creds=True)
    prefix = f"oss-solutions/libero/{run_id}/"

    def read_back(name: str) -> tuple[bytes, dict[str, object]]:
        response = s3.get_object(
            Bucket=bucket, Key=prefix + name, ChecksumMode="ENABLED"
        )
        try:
            payload = response["Body"].read()
        finally:
            response["Body"].close()
        return payload, response

    receipt_bytes, _ = read_back("npa_upload_receipt.json")
    upload_receipt = json.loads(receipt_bytes)
    assert upload_receipt["schema"] == "npa.libero.s3-upload-readback.v1"
    assert upload_receipt["run_id"] == run_id
    assert upload_receipt["status"] == "verified"
    assert upload_receipt["commit_marker"] == "npa_upload_receipt.json"
    expected_names = {
        "libero-bc-rnn-smoke.pth",
        "libero-smoke.json",
        "npa_byof_summary.json",
        "npa_runtime_bootstrap.json",
        "npa_runtime_metadata.json",
        "nvidia_smi.txt",
        "nvidia_smi_list.txt",
        "solution_smoke_stderr.log",
        "solution_smoke_stdout.log",
    }
    receipt_items = {item["name"]: item for item in upload_receipt["artifacts"]}
    assert set(receipt_items) == expected_names
    retrieved: dict[str, bytes] = {}
    for name, item in receipt_items.items():
        assert item["object_key"] == prefix + name
        payload, response = read_back(name)
        digest = hashlib.sha256(payload).hexdigest()
        assert len(payload) == item["size_bytes"]
        assert digest == item["sha256"]
        returned_checksum = response.get("ChecksumSHA256")
        assert returned_checksum == base64.b64encode(bytes.fromhex(digest)).decode(
            "ascii"
        )
        retrieved[name] = payload
    remote_summary = json.loads(retrieved["npa_byof_summary.json"])
    assert remote_summary["status"] == "success"
    assert remote_summary["run_id"] == run_id
    assert remote_summary["image"] == qualification["candidate_image"]
    report = json.loads(retrieved["libero-smoke.json"])

    assert report["schema"] == "npa.workbench.libero.bc-smoke.v1"
    assert report["status"] == "passed"
    assert report["exit_status"] == 0
    assert report["solution"] == "libero"
    assert report["capability"] == "libero_spatial_bc_rnn_train_reload_heldout"
    assert set(report["capabilities_exercised"]) == {
        "libero_official_demo_sha256",
        "libero_upstream_bert_task_conditioning",
        "libero_trajectory_disjoint_heldout_split",
        "libero_spatial_bc_rnn_train_reload_heldout",
    }

    source = report["source"]
    assert source["repository"] == "https://github.com/Lifelong-Robot-Learning/LIBERO"
    assert source["revision"] == "8f1084e3132a39270c3a13ebe37270a43ece2a01"
    assert source["observed_revision"] == source["revision"]
    assert source["license"] == "MIT"

    dataset = report["dataset"]
    assert dataset["repository"] == "yifengzhu-hf/LIBERO-datasets"
    assert dataset["revision"] == "f13aa24a3da8c43c7225569f28c562979fa0e35a"
    assert dataset["suite"] == "libero_spatial"
    assert dataset["task"] == (
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
    )
    assert dataset["license"] == "CC-BY-4.0"
    assert dataset["observed_sha256"] == (
        "ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead"
    )
    assert dataset["observed_size_bytes"] == 508779600
    assert dataset["demo_count"] == 50
    assert dataset["sample_count"] == report["sample_count"] == 5068

    assets = report["task_assets"]
    assert assets["bddl_sha256"] == (
        "9b59eb1287802868ad9bc78d58e6d36d4ba31134e679cfdbdf4b0feb660c959b"
    )
    assert assets["initial_states_sha256"] == (
        "c3a6a01fdc53ae1914fe24c8935088d723baee8f6ee3cd5f8d68e86aea3e2f1c"
    )
    assert assets["source_license"] == "MIT"

    language_model = report["task_language_model"]
    assert language_model["repository"] == "google-bert/bert-base-cased"
    assert language_model["revision"] == ("cd5ef92a9fb2f889e972770a36d4ed042daf221e")
    assert language_model["license"] == "Apache-2.0"
    assert language_model["delivery"] == "runtime_fetch"
    assert language_model["source_path"] == "libero/lifelong/utils.py"
    assert language_model["source_sha256"] == (
        "d1df48c6984a2938d60eebf70ba1c61cd2ea512e859fa0ed11abfc550beee3f1"
    )
    assert language_model["embedding_method"] == "upstream_LIBERO_bert_pooler_output"
    assert language_model["embedding_shape"] == [768]
    assert language_model["embedding_dtype"] == "float32"
    assert language_model["embedding_finite"] is True
    assert language_model["downloaded_this_run"] is True
    assert language_model["cache_uploaded"] is False
    assert language_model["files"]["pytorch_model.bin"] == {
        "expected_size_bytes": 435779157,
        "expected_sha256": (
            "d6992b8cd27d7a132eafce6a8210272329a371b1c762d453588795dd3835593e"
        ),
        "observed_sha256": (
            "d6992b8cd27d7a132eafce6a8210272329a371b1c762d453588795dd3835593e"
        ),
    }

    split = report["split"]
    assert split["strategy"] == "deterministic_trajectory_disjoint_sha256_rank"
    assert split["seed"] == 20260910
    assert split["disjoint"] is True
    assert split["train_demo_count"] == 40
    assert split["heldout_demo_count"] == 10
    assert split["train_sample_count"] == 4020
    assert split["heldout_sample_count"] == 1048
    assert split["train_demo_ids_sha256"] == (
        "9397e01984d1c213fc4bef667baca0492c0368dbbe55f20cb36a19e2f9f48fb4"
    )
    assert split["heldout_demo_ids_sha256"] == (
        "5e8cdf0d4435052e8304008542fccbd09196830ebf442a2a5db02fec3fa523d1"
    )

    training = report["training"]
    assert (
        training["algorithm"]
        == "upstream_libero_Sequential.observe_BCRNNPolicy.compute_loss"
    )
    assert training["optimizer"] == "torch.optim.AdamW"
    assert training["requested_optimizer_steps"] == 8
    assert training["optimizer_steps"] == 8
    assert training["all_losses_finite"] is True
    assert math.isfinite(float(training["first_loss"]))
    assert math.isfinite(float(training["final_loss"]))
    assert math.isfinite(float(training["parameter_max_abs_delta"]))
    assert float(training["parameter_max_abs_delta"]) > 0
    assert training["task_embedding"] == "upstream_LIBERO_bert_pooler_output"

    heldout = report["heldout_metrics"]
    assert heldout["partition"] == "heldout_trajectories_only"
    assert heldout["evaluated_sample_count"] == 1048
    assert math.isfinite(float(heldout["negative_log_likelihood"]))

    checkpoint = report["checkpoint"]
    assert checkpoint["strict_state_dict_load"] is True
    assert checkpoint["reloaded_with"] == "libero.lifelong.utils.torch_load_model"
    assert re.fullmatch(r"[0-9a-f]{64}", checkpoint["sha256"])
    assert (
        checkpoint["sha256"]
        == hashlib.sha256(retrieved["libero-bc-rnn-smoke.pth"]).hexdigest()
    )

    action = report["reloaded_action"]
    assert action["dtype"] == "float32"
    assert action["shape"][-1] == 7
    assert action["finite"] is True
    assert action["evaluated_sample_count"] == 1048
    assert re.fullmatch(r"[0-9a-f]{64}", action["prediction_sha256"])
    assert math.isfinite(float(action["value_min"]))
    assert math.isfinite(float(action["value_max"]))

    runtime = report["runtime"]
    assert "B200" in runtime["gpu_model"].upper()
    assert runtime["gpu_architecture"] == "sm_100"
    assert runtime["compute_capability"] == [10, 0]
    assert runtime["gpu_count"] == 1
    assert "sm_100" in runtime["torch_cuda_arch_list"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", runtime["pod_observed_image_digest"])
    assert runtime["observation_method"] == (
        "kubernetes_status_containerStatuses_imageID"
    )
    assert runtime["actual_service_account"] == "npa-byof-libero-payload"
    assert runtime["controller_service_account_separated"] is True
    for key in (
        "pod_name_sha256",
        "pod_uid_sha256",
        "namespace_sha256",
        "node_name_sha256",
        "cluster_identity_sha256",
        "service_account_uid_sha256",
        "role_uid_sha256",
        "role_binding_uid_sha256",
        "rbac_spec_sha256",
    ):
        assert re.fullmatch(r"[0-9a-f]{64}", runtime[key])
    assert (
        runtime["pod_observed_image_digest"]
        == manager_live["pod_observed_image_digest"]
    )
    assert runtime["pod_name_sha256"] == manager_live["payload_pod_name_sha256"]
    assert runtime["pod_uid_sha256"] == manager_live["payload_pod_uid_sha256"]
    assert runtime["namespace_sha256"] == manager_live["namespace_sha256"]
    assert runtime["node_name_sha256"] == manager_live["node_name_sha256"]
    assert (
        runtime["service_account_uid_sha256"]
        == manager_live["service_account_uid_sha256"]
    )
    smi_list = retrieved["nvidia_smi_list.txt"].decode("utf-8").splitlines()
    assert len(smi_list) == 1
    assert re.fullmatch(r"GPU 0: .*B200.*", smi_list[0], flags=re.IGNORECASE)
    assert b"B200" in retrieved["nvidia_smi.txt"].upper()

    build = report["build"]
    assert build["dataset_delivery"] == "runtime_fetch_to_run_scoped_cache"
    assert build["weights_delivery"] == "runtime_fetch_to_run_scoped_cache"
    assert build["render_assets_present_in_final_filesystem"] is False
    assert build["render_assets_removed_path"] == "libero/libero/assets"
    assert build["git_objects_present_in_final_filesystem"] is False
    assert build["independent_oci_layer_scan_required_before_live_use"] is True
    runtime_metadata = build["runtime_metadata"]
    assert runtime_metadata["schema"] == "npa.libero.runtime-cache.v1"
    assert runtime_metadata["source_revision"] == source["revision"]
    expected_runtime_manifest = json.loads(
        (
            REPO_ROOT
            / "npa"
            / "docker"
            / "workbench"
            / "libero"
            / "runtime-manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert expected_runtime_manifest["runtime_artifact_count"] == 135
    assert (
        runtime_metadata["runtime_artifact_count"]
        == expected_runtime_manifest["runtime_artifact_count"]
    )
    assert runtime_metadata["render_assets_present"] is False
    assert runtime_metadata["git_objects_present"] is False
    assert runtime_metadata["cache_uploaded"] is False
    assert re.fullmatch(
        r"[0-9a-f]{64}", build["accepted_canonical_build_metadata_sha256"]
    )
    assert build["base_image_digest"] == (
        "sha256:999137905e8718de681744822ccd965e1950e1baba089035060418e05e1d7496"
    )
    assert build["base_rootfs_material_digest"] == (
        "sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867"
    )
    assert build["base_image_digest_pinned"] is True
    assert build["base_image_provenance"] == (
        "independent_buildx_metadata_and_published_SLSA_material"
    )
    assert report["boundaries"]["cache_uploaded"] is False
    assert report["boundaries"]["rendering_invoked"] is False
    assert set(report["deferred"]) == {
        "rendered_closed_loop_success_sweeps",
        "all_130_tasks",
        "lifelong_algorithm_comparison",
        "physical_robot_use",
    }


class _RobomimicEntitlementRefusal(RuntimeError):
    """Carry only an approved, value-free entitlement refusal category."""

    def __init__(self, category: str) -> None:
        safe_category = (
            category
            if category in _ROBOMIMIC_ENTITLEMENT_REFUSAL_CATEGORIES
            else "record-invalid"
        )
        self.category = safe_category
        super().__init__(f"robomimic customer entitlement refused: {safe_category}")


def _require_robomimic_entitlement_context(condition: bool) -> None:
    """Reject invalid live selectors without retaining their values."""

    if not condition:
        raise _RobomimicEntitlementRefusal("context-invalid")


def _robomimic_storage_endpoint(value: str) -> str:
    """Accept only a canonical Nebius Object Storage HTTPS origin."""

    candidate = value.strip()
    malformed = False
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError:
        malformed = True
    if malformed:
        raise _RobomimicEntitlementRefusal("context-invalid")
    host = (parsed.hostname or "").lower()
    _require_robomimic_entitlement_context(parsed.scheme == "https")
    _require_robomimic_entitlement_context(
        parsed.username is None and parsed.password is None
    )
    _require_robomimic_entitlement_context(port in {None, 443})
    _require_robomimic_entitlement_context(
        parsed.path in {"", "/"} and not parsed.query and not parsed.fragment
    )
    _require_robomimic_entitlement_context(
        re.fullmatch(r"storage\.[a-z0-9-]+\.nebius\.cloud", host) is not None
    )
    canonical = f"https://{host}"
    _require_robomimic_entitlement_context(candidate.rstrip("/") == canonical)
    return canonical


def _robomimic_live_selectors(e2e_project: str | None) -> dict[str, str]:
    endpoint_candidates = {
        value.strip()
        for value in (
            os.environ.get("AWS_ENDPOINT_URL", ""),
            os.environ.get("NEBIUS_S3_ENDPOINT", ""),
        )
        if value.strip()
    }
    _require_robomimic_entitlement_context(len(endpoint_candidates) == 1)
    selectors = {
        "project": os.environ.get("NPA_E2E_PROJECT", "").strip(),
        "registry": os.environ.get("NPA_BYOF_ROBOMIMIC_REGISTRY", "").strip(),
        "registry_visibility": os.environ.get(
            "NPA_BYOF_ROBOMIMIC_REGISTRY_VISIBILITY", ""
        ).strip(),
        "image": os.environ.get("NPA_BYOF_ROBOMIMIC_IMAGE", "").strip(),
        "kubeconfig": os.environ.get("NPA_BYOF_KUBECONFIG", "").strip(),
        "context": os.environ.get("NPA_BYOF_K8S_CONTEXT", "").strip(),
        "namespace": os.environ.get("NPA_BYOF_K8S_NAMESPACE", "").strip(),
        "bucket": os.environ.get("NPA_E2E_S3_BUCKET", "").strip(),
        "runtime_pvc": os.environ.get("NPA_BYOF_ROBOMIMIC_RUNTIME_PVC", "").strip(),
        "runtime_inventory_sha256": os.environ.get(
            "NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256", ""
        ).strip(),
        "runtime_entitlement_file": os.environ.get(
            "NPA_BYOF_ROBOMIMIC_RUNTIME_ENTITLEMENT_FILE", ""
        ).strip(),
        "storage_endpoint": _robomimic_storage_endpoint(endpoint_candidates.pop()),
    }
    _require_robomimic_entitlement_context(
        bool(selectors["project"]) and e2e_project == selectors["project"]
    )
    _require_robomimic_entitlement_context(bool(selectors["registry"]))
    official_public = selectors["registry"] == "ghcr.io/nebius/nebius-physical-ai"
    _require_robomimic_entitlement_context(
        not is_public_registry(selectors["registry"]) or official_public
    )
    _require_robomimic_entitlement_context(
        selectors["registry_visibility"].lower()
        == ("public" if official_public else "private")
    )
    if official_public:
        _require_robomimic_entitlement_context(
            re.fullmatch(
                r"[0-9a-f]{40}",
                os.environ.get("NPA_BYOF_ROBOMIMIC_DEVELOPMENT_SHA", ""),
            )
            is not None
        )
    expected_image = re.escape(selectors["registry"].rstrip("/")) + (
        r"/npa-robomimic@sha256:[0-9a-f]{64}"
    )
    _require_robomimic_entitlement_context(
        re.fullmatch(expected_image, selectors["image"]) is not None
    )
    _require_robomimic_entitlement_context(
        bool(selectors["kubeconfig"]) and Path(selectors["kubeconfig"]).is_file()
    )
    _require_robomimic_entitlement_context(bool(selectors["context"]))
    _require_robomimic_entitlement_context(
        bool(selectors["namespace"]) and selectors["namespace"] != "default"
    )
    _require_robomimic_entitlement_context(bool(selectors["bucket"]))
    _require_robomimic_entitlement_context(bool(selectors["runtime_pvc"]))
    _require_robomimic_entitlement_context(
        re.fullmatch(r"[0-9a-f]{64}", selectors["runtime_inventory_sha256"]) is not None
    )
    _require_robomimic_entitlement_context(bool(selectors["runtime_entitlement_file"]))
    _require_robomimic_entitlement_context(
        os.environ.get("NPA_E2E_MK8S_RESERVED_CAPACITY") == "1"
    )
    _require_robomimic_entitlement_context(os.environ.get("NPA_BYOF_LIVE_GPU") == "1")
    _require_robomimic_entitlement_context(
        os.environ.get("NPA_BYOF_ROBOMIMIC_LIVE_B200") == "1"
    )
    return selectors


def _robomimic_private_record_bytes(path_value: str) -> bytes:
    """Read a symlink-free owner-private record without blocking on a FIFO."""

    path = Path(os.path.abspath(Path(path_value).expanduser()))
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise RuntimeError("robomimic customer entitlement path is invalid")
    directory_flags = (
        os.O_RDONLY
        | os.O_CLOEXEC
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )

    def open_parent() -> int:
        descriptor = -1
        open_failed = False
        try:
            descriptor = os.open(os.sep, directory_flags)
            for component in path.parent.parts[1:]:
                next_descriptor = os.open(component, directory_flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
        except OSError:
            if descriptor >= 0:
                os.close(descriptor)
            open_failed = True
        if open_failed:
            raise _RobomimicEntitlementRefusal("record-unsafe")
        details = os.fstat(descriptor)
        if details.st_uid != os.geteuid() or details.st_mode & 0o077:
            os.close(descriptor)
            raise _RobomimicEntitlementRefusal("record-unsafe")
        return descriptor

    parent_descriptor = open_parent()
    parent_identity = os.fstat(parent_descriptor)
    descriptor = -1
    read_failed = False
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or opened.st_mode & 0o077
            or opened.st_size > ROBOMIMIC_ENTITLEMENT_MAX_BYTES
        ):
            raise _RobomimicEntitlementRefusal("record-unsafe")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(ROBOMIMIC_ENTITLEMENT_MAX_BYTES + 1)
        closed = os.fstat(descriptor)
        if (
            len(raw) != opened.st_size
            or closed.st_size != opened.st_size
            or closed.st_dev != opened.st_dev
            or closed.st_ino != opened.st_ino
            or closed.st_nlink != 1
            or closed.st_uid != os.geteuid()
            or closed.st_mode & 0o077
        ):
            raise _RobomimicEntitlementRefusal("record-unsafe")
        current_parent = open_parent()
        try:
            current_identity = os.fstat(current_parent)
        finally:
            os.close(current_parent)
        if (current_identity.st_dev, current_identity.st_ino) != (
            parent_identity.st_dev,
            parent_identity.st_ino,
        ):
            raise _RobomimicEntitlementRefusal("record-unsafe")
        return raw
    except OSError:
        read_failed = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    if read_failed:
        raise _RobomimicEntitlementRefusal("record-unsafe")
    raise _RobomimicEntitlementRefusal("record-invalid")


def _preflight_robomimic_runtime_entitlement(
    *, selectors: dict[str, str], run_id: str, now: datetime | None = None
) -> dict[str, str]:
    """Validate the complete customer/run/runtime binding before side effects."""

    raw = _robomimic_private_record_bytes(selectors["runtime_entitlement_file"])
    record_invalid = False
    try:
        record = json.loads(raw)
        lock_raw = ROBOMIMIC_RUNTIME_LOCK.read_bytes()
        lock = json.loads(lock_raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        record_invalid = True
    if record_invalid:
        raise _RobomimicEntitlementRefusal("record-invalid")
    if not isinstance(record, dict) or not isinstance(lock, dict):
        raise _RobomimicEntitlementRefusal("record-invalid")
    contract = lock.get("customer_entitlement")
    if not isinstance(contract, dict) or contract != {
        "schema": "npa.robomimic.customer-runtime-entitlement.v1",
        "maximum_validity_seconds": 86_400,
        "responsibilities": [
            "runtime-use",
            "derivative-use",
            "service-use",
            "output-use",
            "no-redistribution-grant",
        ],
        "terms": ROBOMIMIC_ENTITLEMENT_TERMS,
    }:
        raise _RobomimicEntitlementRefusal("contract-invalid")
    lock_sha256 = hashlib.sha256(lock_raw).hexdigest()
    notice_identity = {
        "schema": contract.get("schema"),
        "runtime_id": lock.get("runtime_id"),
        "runtime_lock_sha256": lock_sha256,
        "terms": contract.get("terms"),
        "customer_responsibilities": contract.get("responsibilities"),
    }
    expected = {
        "schema": contract.get("schema"),
        "decision": "accepted",
        "customer_binding_sha256": hashlib.sha256(
            b"npa.robomimic.customer-binding.v1\0"
            + selectors["project"].encode("utf-8")
        ).hexdigest(),
        "run_id": run_id,
        "source_revision": lock.get("source_revision"),
        "runtime_id": lock.get("runtime_id"),
        "runtime_manifest_sha256": selectors["runtime_inventory_sha256"],
        "runtime_lock_sha256": lock_sha256,
        "terms": contract.get("terms"),
        "customer_responsibilities": contract.get("responsibilities"),
        "notice_sha256": hashlib.sha256(
            json.dumps(notice_identity, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    if set(record) != set(expected) | {"accepted_at", "expires_at"}:
        raise _RobomimicEntitlementRefusal("fields-invalid")
    if any(record.get(key) != value for key, value in expected.items()):
        raise _RobomimicEntitlementRefusal("binding-mismatch")
    time_invalid = False
    try:
        accepted_at = datetime.strptime(
            record["accepted_at"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        expires_at = datetime.strptime(
            record["expires_at"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        time_invalid = True
    if time_invalid:
        raise _RobomimicEntitlementRefusal("time-invalid")
    observed_now = now or datetime.now(timezone.utc)
    validity_seconds = int((expires_at - accepted_at).total_seconds())
    maximum_validity = contract.get("maximum_validity_seconds")
    if (
        not isinstance(maximum_validity, int)
        or accepted_at > observed_now
        or validity_seconds <= 0
        or validity_seconds > maximum_validity
        or observed_now >= expires_at
    ):
        raise _RobomimicEntitlementRefusal("time-invalid")
    return {
        "record_sha256": hashlib.sha256(raw).hexdigest(),
        "customer_binding_sha256": expected["customer_binding_sha256"],
    }


def _robomimic_runner_command(
    config: dict[str, object],
    registry: str,
    project: str,
    output_root: str,
    run_id: str,
    profile_yaml: Path,
    runtime_entitlement_file: str,
) -> list[str]:
    options = (
        ("--registry", registry),
        ("--project", project),
        ("--repo-url", str(config["repo_url"])),
        ("--repo-ref", str(config["repo_ref"])),
        ("--base-profile", str(config["base_profile"])),
        ("--base-image", str(config["base_image"])),
        ("--build-command", str(config["build_command"])),
        ("--workload", "solution-smoke"),
        ("--smoke-command", str(config["smoke_command"])),
        ("--solution-name", "robomimic"),
        ("--capability-name", str(config["capability_name"])),
        ("--smoke-artifact-name", "robomimic-smoke.json"),
        ("--yaml", str(profile_yaml)),
        ("--output-root", output_root),
        ("--wait-timeout", "-1"),
        ("--run-id", run_id),
        ("--robomimic-runtime-entitlement-file", runtime_entitlement_file),
    )
    return [
        sys.executable,
        str(BYOF_RUNNER),
        *(item for pair in options for item in pair),
    ]


def _materialize_robomimic_attested_profile(
    destination: Path,
    *,
    namespace: str,
    service_account: str,
    runtime_pvc: str,
    runtime_inventory_sha256: str,
    runtime_entitlement_file: str,
    runtime_entitlement_sha256: str,
    customer_binding_sha256: str,
) -> Path:
    """Bind STRICT placement and customer entitlement into a run-local profile."""

    assert os.environ.get("NPA_E2E_MK8S_RESERVED_CAPACITY") == "1"
    source = resolve_byof_profile_path("byof-solution-smoke-robomimic-b200-gpu")
    documents = list(yaml.safe_load_all(source.read_text(encoding="utf-8")))
    assert len(documents) == 2
    task = documents[1]
    assert task["envs"]["NPA_ROBOMIMIC_STRICT_B200_ATTESTED"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_EXPECTED_NAMESPACE"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_EXPECTED_SERVICE_ACCOUNT"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_RUNTIME_ENTITLEMENT_SHA256"] == ""
    assert task["envs"]["NPA_ROBOMIMIC_CUSTOMER_BINDING_SHA256"] == ""
    task["envs"]["NPA_ROBOMIMIC_STRICT_B200_ATTESTED"] = "1"
    task["envs"]["NPA_ROBOMIMIC_EXPECTED_NAMESPACE"] = namespace
    task["envs"]["NPA_ROBOMIMIC_EXPECTED_SERVICE_ACCOUNT"] = service_account
    task["envs"]["NPA_ROBOMIMIC_RUNTIME_INVENTORY_SHA256"] = runtime_inventory_sha256
    task["envs"]["NPA_ROBOMIMIC_RUNTIME_ENTITLEMENT_SHA256"] = (
        runtime_entitlement_sha256
    )
    task["envs"]["NPA_ROBOMIMIC_CUSTOMER_BINDING_SHA256"] = customer_binding_sha256
    task["file_mounts"]["/opt/npa-runtime-authorization/robomimic.json"] = (
        runtime_entitlement_file
    )
    task["config"]["kubernetes"]["pod_config"]["spec"]["serviceAccountName"] = (
        service_account
    )
    volumes = task["config"]["kubernetes"]["pod_config"]["spec"]["volumes"]
    runtime_volume = next(
        item for item in volumes if item["name"] == "robomimic-runtime"
    )
    assert (
        runtime_volume["persistentVolumeClaim"]["claimName"]
        == "npa-robomimic-runtime-placeholder"
    )
    runtime_volume["persistentVolumeClaim"]["claimName"] = runtime_pvc
    destination.write_text(
        yaml.safe_dump_all(documents, sort_keys=False), encoding="utf-8"
    )
    return destination


def _robomimic_observer_name(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    return f"npa-robomimic-{digest}"


def _robomimic_observer_manifests(
    *, run_id: str, namespace: str, service_account: str, owner_token: str
) -> list[dict[str, object]]:
    labels = {
        "app.kubernetes.io/managed-by": "npa-robomimic-live-gate",
    }
    annotations = {
        "npa.nebius.ai/run-id-sha256": hashlib.sha256(run_id.encode()).hexdigest(),
        "npa.nebius.ai/owner-token": owner_token,
    }

    def metadata() -> dict[str, object]:
        return {
            "name": service_account,
            "namespace": namespace,
            "labels": dict(labels),
            "annotations": dict(annotations),
        }

    return [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": metadata(),
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": metadata(),
            "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "RoleBinding",
            "metadata": metadata(),
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": service_account,
                    "namespace": namespace,
                }
            ],
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "Role",
                "name": service_account,
            },
        },
    ]


def _robomimic_kubectl(selectors: dict[str, str]) -> list[str]:
    return [
        "kubectl",
        "--kubeconfig",
        selectors["kubeconfig"],
        "--context",
        selectors["context"],
    ]


def _robomimic_can_i(
    kube: list[str],
    *,
    namespace: str,
    service_account: str,
    verb: str,
    resource: str,
    env: dict[str, str],
) -> bool:
    resource_type, separator, subresource = resource.partition("/")
    subresource_args = ["--subresource", subresource] if separator else []
    result = subprocess.run(
        [
            *kube,
            "auth",
            "can-i",
            verb,
            resource_type,
            *subresource_args,
            "--namespace",
            namespace,
            "--as",
            f"system:serviceaccount:{namespace}:{service_account}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    answer = result.stdout.strip().lower()
    if (result.returncode, answer) == (0, "yes"):
        return True
    if (result.returncode, answer) == (1, "no"):
        return False
    raise AssertionError(
        "kubectl auth can-i returned an uncertain robomimic result: "
        f"exit={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}"
    )


def _robomimic_resource_permissions(
    kube: list[str], *, namespace: str, service_account: str, env: dict[str, str]
) -> set[tuple[str, str, str, str]]:
    """Return every namespaced resource permission observed for the identity."""

    review = {
        "apiVersion": "authorization.k8s.io/v1",
        "kind": "SelfSubjectRulesReview",
        "spec": {"namespace": namespace},
    }
    result = subprocess.run(
        [
            *kube,
            "--as",
            f"system:serviceaccount:{namespace}:{service_account}",
            "create",
            "--raw",
            "/apis/authorization.k8s.io/v1/selfsubjectrulesreviews",
            "-f",
            "-",
        ],
        input=json.dumps(review),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError(
            "SelfSubjectRulesReview failed for the robomimic observer: "
            f"exit={result.returncode}, stderr={result.stderr.strip()!r}"
        )
    payload = json.loads(result.stdout)
    status = payload.get("status", {})
    if status.get("incomplete") or status.get("evaluationError"):
        raise AssertionError("robomimic observer permission review was incomplete")
    permissions: set[tuple[str, str, str, str]] = set()
    for rule in status.get("resourceRules", []):
        names = rule.get("resourceNames") or ["*"]
        permissions.update(
            (group, resource, name, verb)
            for group in rule.get("apiGroups", [])
            for resource in rule.get("resources", [])
            for name in names
            for verb in rule.get("verbs", [])
        )
    return permissions


def _assert_robomimic_observer_permissions(
    kube: list[str], *, namespace: str, service_account: str, env: dict[str, str]
) -> None:
    expected = {
        ("", "pods", "*", "get"),
        (
            "authorization.k8s.io",
            "selfsubjectaccessreviews",
            "*",
            "create",
        ),
        (
            "authorization.k8s.io",
            "selfsubjectrulesreviews",
            "*",
            "create",
        ),
        ("authentication.k8s.io", "selfsubjectreviews", "*", "create"),
    }
    observed = _robomimic_resource_permissions(
        kube, namespace=namespace, service_account=service_account, env=env
    )
    assert observed == expected, {
        "unexpected": sorted(observed - expected),
        "missing": sorted(expected - observed),
    }


def _robomimic_delete_path(resource: str, namespace: str) -> str:
    kind, name = resource.split("/", 1)
    encoded_namespace = quote(namespace, safe="")
    encoded_name = quote(name, safe="")
    if kind == "serviceaccount":
        return f"/api/v1/namespaces/{encoded_namespace}/serviceaccounts/{encoded_name}"
    plural = {"role": "roles", "rolebinding": "rolebindings"}[kind]
    return (
        "/apis/rbac.authorization.k8s.io/v1/namespaces/"
        f"{encoded_namespace}/{plural}/{encoded_name}"
    )


@contextmanager
def _robomimic_observer_rbac(
    *, run_id: str, selectors: dict[str, str], env: dict[str, str]
) -> Iterator[str]:
    """Own, verify, and remove the run-scoped Pod-observation identity."""

    namespace = selectors["namespace"]
    service_account = _robomimic_observer_name(run_id)
    owner_token = secrets.token_hex(32)
    kube = _robomimic_kubectl(selectors)
    manifests = _robomimic_observer_manifests(
        run_id=run_id,
        namespace=namespace,
        service_account=service_account,
        owner_token=owner_token,
    )
    created_resources: list[str] = []
    primary_error: BaseException | None = None
    try:
        for manifest in manifests:
            resource = f"{str(manifest['kind']).lower()}/{service_account}"
            prior = subprocess.run(
                [
                    *kube,
                    "get",
                    "--namespace",
                    namespace,
                    resource,
                    "--ignore-not-found=true",
                    "-o",
                    "name",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            assert prior.returncode == 0 and not prior.stdout.strip(), (
                resource,
                prior.stdout,
                prior.stderr,
            )
            # Track after proving no prior object exists but before create, so a
            # server-side create followed by a lost response is still cleaned.
            created_resources.append(resource)
            created = subprocess.run(
                [*kube, "create", "--namespace", namespace, "-f", "-"],
                input=yaml.safe_dump(manifest, sort_keys=False),
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            if created.returncode != 0:
                raise RuntimeError(
                    f"failed to create run-scoped robomimic {resource}: "
                    f"{created.stderr.strip()}"
                )
        permissions = {
            ("get", "pods"): True,
            ("list", "pods"): False,
            ("watch", "pods"): False,
            ("get", "pods/log"): False,
            ("get", "secrets"): False,
        }
        for (verb, resource), expected in permissions.items():
            assert (
                _robomimic_can_i(
                    kube,
                    namespace=namespace,
                    service_account=service_account,
                    verb=verb,
                    resource=resource,
                    env=env,
                )
                is expected
            )
        _assert_robomimic_observer_permissions(
            kube,
            namespace=namespace,
            service_account=service_account,
            env=env,
        )
        yield service_account
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        resources = list(reversed(created_resources))
        cleanup_errors: list[str] = []
        for resource in resources:
            try:
                observed = subprocess.run(
                    [
                        *kube,
                        "get",
                        "--namespace",
                        namespace,
                        resource,
                        "--ignore-not-found=true",
                        "-o",
                        "json",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except OSError as exc:
                cleanup_errors.append(
                    f"inspect {resource} ownership failed: {type(exc).__name__}"
                )
                continue
            if observed.returncode != 0:
                cleanup_errors.append(
                    f"inspect {resource} ownership exited {observed.returncode}: "
                    f"{observed.stderr.strip()}"
                )
                continue
            if not observed.stdout.strip():
                continue
            try:
                observed_payload = json.loads(observed.stdout)
                observed_metadata = observed_payload["metadata"]
                observed_token = observed_metadata["annotations"][
                    "npa.nebius.ai/owner-token"
                ]
                observed_uid = observed_metadata["uid"]
                observed_resource_version = observed_metadata["resourceVersion"]
            except (KeyError, TypeError, json.JSONDecodeError):
                cleanup_errors.append(
                    f"inspect {resource} ownership returned invalid metadata"
                )
                continue
            if observed_token != owner_token:
                cleanup_errors.append(
                    f"refused to delete {resource} owned by another invocation"
                )
                continue
            try:
                delete_options = {
                    "apiVersion": "v1",
                    "kind": "DeleteOptions",
                    "preconditions": {
                        "uid": observed_uid,
                        "resourceVersion": observed_resource_version,
                    },
                }
                deleted = subprocess.run(
                    [
                        *kube,
                        "delete",
                        "--raw",
                        _robomimic_delete_path(resource, namespace),
                        "-f",
                        "-",
                    ],
                    input=json.dumps(delete_options),
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except OSError as exc:
                cleanup_errors.append(f"delete {resource} failed: {type(exc).__name__}")
                continue
            if deleted.returncode != 0:
                cleanup_errors.append(
                    f"delete {resource} exited {deleted.returncode}: "
                    f"{deleted.stderr.strip()}"
                )
        for resource in resources:
            try:
                absent = subprocess.run(
                    [
                        *kube,
                        "get",
                        "--namespace",
                        namespace,
                        resource,
                        "--ignore-not-found=true",
                        "-o",
                        "name",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except OSError as exc:
                cleanup_errors.append(
                    f"verify {resource} absent failed: {type(exc).__name__}"
                )
                continue
            if absent.returncode != 0 or absent.stdout.strip():
                cleanup_errors.append(
                    f"verify {resource} absent failed: exit={absent.returncode}, "
                    f"stdout={absent.stdout.strip()!r}, stderr={absent.stderr.strip()!r}"
                )
        if cleanup_errors:
            message = "robomimic observer RBAC cleanup failed: " + "; ".join(
                cleanup_errors
            )
            if primary_error is not None:
                add_note = getattr(primary_error, "add_note", None)
                if callable(add_note):
                    add_note(message)
                else:  # Python 3.10 compatibility.
                    primary_error.args = (*primary_error.args, message)
            else:
                raise AssertionError(message)


def _robomimic_target_env(
    e2e_project: str, selectors: dict[str, str], cmd: list[str]
) -> dict[str, str]:
    config_path = skypilot_config_for_project(e2e_project)
    if config_path:
        cmd.extend(["--config-path", config_path])
    target = resolve_byof_kubernetes_target(e2e_project)
    assert target.kubeconfig == selectors["kubeconfig"]
    assert target.context == selectors["context"]
    assert target.namespace == selectors["namespace"]
    env = dict(os.environ)
    env["KUBECONFIG"] = target.kubeconfig
    env["NPA_BYOF_KUBECONFIG"] = target.kubeconfig
    env["NPA_BYOF_K8S_CONTEXT"] = target.context
    env["AWS_ENDPOINT_URL"] = selectors["storage_endpoint"]
    env["NEBIUS_S3_ENDPOINT"] = selectors["storage_endpoint"]
    namespace_readback = subprocess.run(
        [
            *_robomimic_kubectl(selectors),
            "config",
            "view",
            "--minify",
            "--output",
            "jsonpath={.contexts[0].context.namespace}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if namespace_readback.returncode != 0:
        raise AssertionError(
            "manager-issued Kubernetes context namespace lookup failed: "
            f"exit={namespace_readback.returncode}, "
            f"stderr={namespace_readback.stderr.strip()!r}"
        )
    observed_namespace = namespace_readback.stdout.strip() or "default"
    assert observed_namespace == selectors["namespace"], (
        "manager-issued Kubernetes context namespace does not match its selector"
    )
    skypilot_bin = resolve_skypilot_bin()
    if skypilot_bin:
        env["PATH"] = f"{Path(skypilot_bin).parent}:{env.get('PATH', '')}"
    return env


def _invoke_robomimic_gate(
    e2e_project: str | None,
) -> tuple[dict[str, object], str, str]:
    context_refused = False
    try:
        selectors = _robomimic_live_selectors(e2e_project)
    except Exception:
        context_refused = True
    if context_refused:
        raise _RobomimicEntitlementRefusal("context-invalid")
    run_id = os.environ.get("NPA_BYOF_ROBOMIMIC_RUN_ID") or (
        f"robomimic-live-{secrets.token_hex(8)}"
    )
    if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", run_id) is None:
        raise _RobomimicEntitlementRefusal("identity-invalid")
    refusal_category = ""
    try:
        entitlement = _preflight_robomimic_runtime_entitlement(
            selectors=selectors, run_id=run_id
        )
    except _RobomimicEntitlementRefusal as exc:
        refusal_category = exc.category
    except Exception:
        refusal_category = "record-invalid"
    if refusal_category:
        raise _RobomimicEntitlementRefusal(refusal_category)
    _activate_nebius_profile()
    config = load_spec(ROBOMIMIC_SPEC).config
    if selectors["registry_visibility"].lower() == "public":
        # Selectors restrict this route to the official namespace and full
        # development SHA; the runner checks the exact published digest.
        registry = selectors["registry"].rstrip("/")
    else:
        registry = resolve_container_registry(e2e_project)
        assert registry == selectors["registry"].rstrip("/")
        assert not is_public_registry(registry)
    bucket = live_bucket(e2e_project)
    assert bucket == selectors["bucket"].removeprefix("s3://").split("/", 1)[0]
    with tempfile.TemporaryDirectory(prefix="npa-robomimic-profile-") as temp_dir:
        service_account = _robomimic_observer_name(run_id)
        profile_yaml = _materialize_robomimic_attested_profile(
            Path(temp_dir) / "robomimic-attested.yaml",
            namespace=selectors["namespace"],
            service_account=service_account,
            runtime_pvc=selectors["runtime_pvc"],
            runtime_inventory_sha256=selectors["runtime_inventory_sha256"],
            runtime_entitlement_file=selectors["runtime_entitlement_file"],
            runtime_entitlement_sha256=entitlement["record_sha256"],
            customer_binding_sha256=entitlement["customer_binding_sha256"],
        )
        cmd = _robomimic_runner_command(
            config,
            registry,
            selectors["project"],
            f"s3://{bucket}/oss-solutions/robomimic",
            run_id,
            profile_yaml,
            selectors["runtime_entitlement_file"],
        )
        env = _robomimic_target_env(selectors["project"], selectors, cmd)
        with _robomimic_observer_rbac(
            run_id=run_id, selectors=selectors, env=env
        ) as observed_service_account:
            assert observed_service_account == service_account
            accepted_image = selectors["image"]
            cmd.extend(["--image", accepted_image, "--skip-build"])
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                env=env,
            )
            assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary.get("status") == "ok", summary
    summary_image = str(summary.get("image", ""))
    assert summary_image.startswith(f"{registry}/") and "@sha256:" in summary_image
    return summary, bucket, run_id


def _robomimic_artifact(
    e2e_project: str | None, bucket: str, run_id: str
) -> dict[str, object]:
    key = f"oss-solutions/robomimic/{run_id}/robomimic-smoke.json"
    body: object | None = None
    artifact: dict[str, object] | None = None
    failed = False
    try:
        client = s3_client_for_project(e2e_project, allow_host_creds=True)
        response = client.get_object(Bucket=bucket, Key=key)
        if not isinstance(response, dict):
            raise ValueError
        body = response.get("Body")
        if body is None:
            raise ValueError
        declared_size = response.get("ContentLength")
        if not _valid_robomimic_smoke_proof_size(declared_size):
            raise ValueError
        artifact = _decode_robomimic_smoke_proof(body, declared_size)
    except Exception:
        failed = True
    finally:
        if body is not None and not _close_robomimic_smoke_proof(body):
            failed = True
    if failed or artifact is None:
        raise RuntimeError(_ROBOMIMIC_SMOKE_PROOF_ERROR) from None
    return artifact


def _valid_robomimic_smoke_proof_size(value: object) -> bool:
    """Accept only bounded, non-negative integer S3 object lengths."""

    return type(value) is int and 0 <= value <= ROBOMIMIC_SMOKE_PROOF_MAX_BYTES


def _decode_robomimic_smoke_proof(
    body: object, declared_size: int
) -> dict[str, object]:
    """Read and decode one bounded robomimic smoke proof."""

    payload = body.read(ROBOMIMIC_SMOKE_PROOF_MAX_BYTES + 1)
    if not isinstance(payload, bytes):
        raise ValueError
    if len(payload) > ROBOMIMIC_SMOKE_PROOF_MAX_BYTES or len(payload) != declared_size:
        raise ValueError
    artifact = json.loads(payload.decode("utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError
    return artifact


def _close_robomimic_smoke_proof(body: object) -> bool:
    """Close one proof body without exposing a cleanup exception."""

    try:
        body.close()
    except Exception:
        return False
    return True


def _assert_robomimic_inputs(artifact: dict[str, object]) -> None:
    source = artifact["source"]
    assert source["repository"] == "ARISE-Initiative/robomimic"
    assert source["revision"] == ROBOMIMIC_SOURCE_REVISION
    assert source["observed_head"] == ROBOMIMIC_SOURCE_REVISION
    assert re.fullmatch(r"[0-9a-f]{64}", source["tree_archive_sha256"])
    dataset = artifact["dataset"]
    assert dataset["repository"] == "robomimic/robomimic_datasets"
    assert dataset["revision"] == ROBOMIMIC_DATASET_REVISION
    assert dataset["path"] == "v1.5/lift/ph/low_dim_v15.hdf5"
    assert dataset["sha256"] == ROBOMIMIC_DATASET_SHA256
    assert dataset["size_bytes"] == 21_084_088
    assert dataset["trajectory_count"] == 200 and dataset["sample_count"] > 0


def _assert_robomimic_split(artifact: dict[str, object]) -> None:
    dataset, split = artifact["dataset"], artifact["split"]
    assert split["overlap_count"] == 0
    assert split["train_trajectory_count"] > 0
    assert split["validation_trajectory_count"] > 0
    assert split["train_trajectory_count"] + split["validation_trajectory_count"] == 200
    assert split["train_sample_count"] > 0 and split["validation_sample_count"] > 0
    assert (
        split["train_sample_count"] + split["validation_sample_count"]
        == dataset["sample_count"]
    )
    assert re.fullmatch(r"[0-9a-f]{64}", split["train_keys_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", split["validation_keys_sha256"])


def _assert_robomimic_training(artifact: dict[str, object]) -> None:
    training = artifact["training"]
    assert training["entrypoint"] == "robomimic/scripts/train.py"
    assert training["algorithm"] == "bc" and training["optimizer"] == "adam"
    assert training["optimizer_step_count"] == 4
    assert training["configured_optimizer_steps"] == 4
    assert training["configured_validation_forward_steps"] == 2
    assert math.isfinite(training["train_loss"])
    assert math.isfinite(training["validation_loss"])
    assert artifact["checkpoint"]["reloaded"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", artifact["checkpoint"]["sha256"])


def _assert_robomimic_action(artifact: dict[str, object]) -> None:
    action, split = artifact["heldout_action"], artifact["split"]
    assert action["demo"] == split["heldout_demo"]
    assert action["shape"] == [7] and action["finite"] is True
    assert action["within_allowed_range"] is True
    assert action["allowed_range"] == [-1.0, 1.0]
    assert -1.0 <= action["observed_min"] <= action["observed_max"] <= 1.0
    assert math.isfinite(action["observed_min"])
    assert math.isfinite(action["observed_max"])


def _assert_robomimic_runtime(
    artifact: dict[str, object], summary_image: str, run_id: str
) -> None:
    external_runtime = artifact["external_runtime"]
    assert external_runtime["prepopulated"] is True
    assert external_runtime["read_only"] is True
    assert external_runtime["runtime_manifest_digest_matched"] is True
    entitlement = artifact["customer_runtime_entitlement"]
    assert entitlement["run_binding_matched"] is True
    assert entitlement["customer_binding_matched"] is True
    assert "field_of_use" not in entitlement
    assert entitlement["redistribution_granted"] is False
    assert external_runtime["atomic_private_snapshot_published"] is True
    assert external_runtime["snapshot_write_bits_absent"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", external_runtime["lock_sha256"])
    assert (
        external_runtime["inventory_sha256"]
        == os.environ["NPA_BYOF_ROBOMIMIC_RUNTIME_INVENTORY_SHA256"]
    )
    hardware = artifact["hardware"]
    assert hardware["accelerator_count"] == 1 and "B200" in hardware["model"].upper()
    assert hardware["architecture"] == "sm_100"
    assert hardware["compute_capability"] == [10, 0]
    assert len(hardware["nvidia_smi_rows"]) == 1
    assert "B200" in hardware["nvidia_smi_rows"][0].upper()
    assert "strict_reserved_capacity_attested" not in hardware
    assert "application_strict_capacity_qualification" in artifact["deferred"]
    pod_image = artifact["pod_image"]
    assert pod_image["runtime_ref"] == summary_image
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", pod_image["digest"])
    assert pod_image["digest"] in pod_image["image_id"]
    assert pod_image["container_name"] == "ray-node"
    assert pod_image["observation_source"] == (
        "Kubernetes Pod status.containerStatuses[].imageID"
    )
    runtime_mount = artifact["runtime_mount"]
    assert runtime_mount["name"] == "robomimic-runtime"
    assert runtime_mount["path"] == "/opt/npa-runtime/robomimic"
    assert runtime_mount["read_only"] is True
    identity = artifact["workload_identity"]
    assert identity["namespace"] == os.environ["NPA_BYOF_K8S_NAMESPACE"]
    assert identity["service_account"] == _robomimic_observer_name(run_id)
    assert identity["observation_source"] == (
        "mounted service-account namespace and Kubernetes Pod spec.serviceAccountName"
    )


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_GPU") != "1"
    or os.environ.get("NPA_BYOF_ROBOMIMIC_LIVE_B200") != "1"
    or os.environ.get("NPA_E2E_MK8S_RESERVED_CAPACITY") != "1",
    reason=(
        "Set NPA_BYOF_LIVE_GPU=1, NPA_BYOF_ROBOMIMIC_LIVE_B200=1, and "
        "NPA_E2E_MK8S_RESERVED_CAPACITY=1 only with the assigned STRICT "
        "one-B200 runtime context to execute the robomimic gate."
    ),
)
def test_live_robomimic_b200_train_reload_gate(e2e_project: str | None) -> None:
    """Use an accepted private candidate and require its complete run artifact."""

    summary, bucket, run_id = _invoke_robomimic_gate(e2e_project)
    summary_image = str(summary.get("image", ""))
    artifact = _robomimic_artifact(e2e_project, bucket, run_id)
    assert artifact["schema"] == "npa.workbench.robomimic.smoke.v1"
    assert artifact["solution"] == "robomimic"
    assert artifact["capability"] == "lift_ph_lowdim_checkpoint_reload_action"
    assert set(artifact["capabilities_exercised"]) == ROBOMIMIC_CAPABILITIES
    _assert_robomimic_inputs(artifact)
    _assert_robomimic_split(artifact)
    _assert_robomimic_training(artifact)
    _assert_robomimic_action(artifact)
    _assert_robomimic_runtime(artifact, summary_image, run_id)
    assert set(artifact["deferred"]) == {
        "application_strict_capacity_qualification",
        "public_image_acceptance",
        "image_policy_sweeps",
        "simulator_rollouts",
        "full_algorithm_matrix",
    }
    assert artifact["exit_status"] == 0
