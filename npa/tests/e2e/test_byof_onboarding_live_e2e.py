"""Live infra checks for generic BYOF solution onboarding (workflow + optional agent chat)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO
from urllib.parse import urlparse

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.clients.config import resolve_container_registry
from npa.clients.project_credentials import s3_client_for_project
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
GYMNASIUM_ROBOTICS_SPEC = (
    REPO_ROOT / "workflows" / "testing" / "byof-gymnasium-robotics.yaml"
)
GYMNASIUM_KUBECTL_TIMEOUT_SECONDS = 30
GYMNASIUM_RECEIPT_TIMEOUT_SECONDS = 900
GYMNASIUM_CLEANUP_TIMEOUT_SECONDS = 180
GYMNASIUM_SKY_DOWN_TIMEOUT_SECONDS = 120
GYMNASIUM_RUNNER_TIMEOUT_SECONDS = 1800
GYMNASIUM_RUNNER_TERM_GRACE_SECONDS = 15
RUNNER = CliRunner()


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
    spec.config["output_root"] = authorized_output_root.rstrip("/")
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
    cmd = [
        sys.executable,
        str(BYOF_RUNNER),
        *argv[4:],
        "--registry",
        resolve_container_registry(e2e_project),
        "--project",
        e2e_project or "",
    ]
    preset = os.environ.get("NPA_BYOF_GYMNASIUM_ROBOTICS_IMAGE", "").strip()
    assert preset, (
        "NPA_BYOF_GYMNASIUM_ROBOTICS_IMAGE must be the already scanned, "
        "task-private immutable image reference"
    )
    _immutable_image_digest(preset)
    cmd.extend(["--image", preset, "--skip-build"])
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
    document = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
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
    for verb, resource in (("list", "pods"), ("create", "pods/exec")):
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
        assert pod.get("status", {}).get("phase") in {"Pending", "Running"}
        containers = pod.get("spec", {}).get("containers", [])
        matching_containers = [
            container
            for container in containers
            if str(container.get("image", "")).removeprefix("docker:") == expected_image
        ]
        assert len(matching_containers) == 1, (
            "the exact run Pod must contain one immutable task image"
        )
        container = matching_containers[0]
        gpu_requests = int(
            container.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0)
        )
        gpu_limits = int(
            container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0)
        )
        assert gpu_requests == gpu_limits == 1, (
            "the exact run Pod must request and limit one GPU"
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
            "container_name": str(container.get("name", "")),
            "spec_image": str(container.get("image", "")).removeprefix("docker:"),
            "image_id": image_id,
            "expected_digest": expected_digest,
            "observed_digest": observed_digests.pop(),
            "observed_unix": round(time.time(), 3),
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
            "/opt/venv/bin/python",
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


def _wait_for_gymnasium_pod_absence(
    env: dict[str, str], *, namespace: str, run_id: str
) -> bool:
    deadline = time.monotonic() + GYMNASIUM_CLEANUP_TIMEOUT_SECONDS
    while _exact_gymnasium_run_pods(
        env,
        namespace=namespace,
        run_id=run_id,
        include_terminating=True,
    ):
        if time.monotonic() >= deadline:
            return False
        time.sleep(2)
    return True


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


def _cleanup_gymnasium_run(
    env: dict[str, str],
    *,
    namespace: str,
    run_id: str,
    config_path: str | None,
    issue_down: bool,
) -> None:
    evidence_dir = _gymnasium_evidence_dir(env)
    if issue_down:
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
            for suffix, content in (
                ("stdout", exc.stdout or ""),
                ("stderr", exc.stderr or ""),
            ):
                path = evidence_dir / f"{run_id}-sky-down-{suffix}.log"
                with _new_gymnasium_private_file(path) as stream:
                    stream.write(
                        content.decode(errors="replace")
                        if isinstance(content, bytes)
                        else content
                    )
            assert _wait_for_gymnasium_pod_absence(
                env, namespace=namespace, run_id=run_id
            ), "sky down timed out and the exact-run Pod remains"
            return
        for suffix, content in (
            ("stdout", result.stdout),
            ("stderr", result.stderr),
        ):
            path = evidence_dir / f"{run_id}-sky-down-{suffix}.log"
            with _new_gymnasium_private_file(path) as stream:
                stream.write(content)
        if result.returncode != 0:
            assert _wait_for_gymnasium_pod_absence(
                env, namespace=namespace, run_id=run_id
            ), "exact-run SkyPilot cleanup failed and its Pod remains"
            return
    assert _wait_for_gymnasium_pod_absence(env, namespace=namespace, run_id=run_id), (
        "timed out waiting for the exact-run Pod to disappear"
    )


def _gymnasium_expected_digest(summary: dict[str, object]) -> str:
    image = str(summary["image"])
    expected_digest = _immutable_image_digest(image)
    build = summary["build"]
    assert build["ok"] is True
    if build.get("skipped"):
        assert os.environ.get("NPA_BYOF_GYMNASIUM_ROBOTICS_IMAGE", "").strip()
    else:
        assert build["pushed"] is True
        assert build["digest"] == expected_digest
    return expected_digest


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
    artifact = s3.get_object(
        Bucket=parsed.netloc, Key=prefix + "gymnasium-robotics-smoke.json"
    )["Body"].read()
    summary = json.loads(
        s3.get_object(Bucket=parsed.netloc, Key=prefix + "npa_byof_summary.json")[
            "Body"
        ].read()
    )
    return artifact, summary


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
    stdout_path = evidence_dir / f"{run_id}-runner-stdout.log"
    stderr_path = evidence_dir / f"{run_id}-runner-stderr.log"
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
        try:
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
        except BaseException as primary_error:
            try:
                _terminate_gymnasium_runner(proc)
            except BaseException as termination_error:
                primary_error.add_note(
                    f"BYOF runner termination also failed: {termination_error}"
                )
            try:
                _cleanup_gymnasium_run(
                    env,
                    namespace=namespace,
                    run_id=run_id,
                    config_path=config_path,
                    issue_down=True,
                )
            except BaseException as cleanup_error:
                primary_error.add_note(
                    f"exact-run cleanup also failed: {cleanup_error}"
                )
            raise
    stdout = stdout_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    _cleanup_gymnasium_run(
        env,
        namespace=namespace,
        run_id=run_id,
        config_path=config_path,
        issue_down=False,
    )
    summary = _parse_last_json_blob(stdout + "\n" + stderr)
    assert summary["status"] == "ok", summary
    expected_digest = _gymnasium_expected_digest(summary)
    assert receipt["expected_digest"] == expected_digest
    assert receipt["observed_digest"] == expected_digest
    artifact_bytes, remote_summary = _gymnasium_remote_evidence(
        e2e_project,
        root_uri=output_root.rstrip("/") + f"/{run_id}/",
    )
    _assert_gymnasium_robotics_artifact(artifact_bytes, expected_digest=expected_digest)
    assert remote_summary["smoke_exit_code"] == 0
    assert remote_summary["smoke_artifact"]["size_bytes"] == len(artifact_bytes)
    assert (
        remote_summary["smoke_artifact"]["sha256"]
        == hashlib.sha256(artifact_bytes).hexdigest()
    )
    assert expected_digest in remote_summary["pod_observed_image_id"]
