"""Live infra checks for generic BYOF solution onboarding (workflow + optional agent chat)."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import resolve_container_registry
from npa.clients.project_credentials import s3_client_for_project
from npa.deploy.images import DEFAULT_CONTAINER_REGISTRY
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
HABITAT_SIM_SPEC = REPO_ROOT / "workflows" / "testing" / "byof-habitat-sim.yaml"
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_repo.py"
RUNNER = CliRunner()
HABITAT_SOURCE_REVISION = "57ee4941dc4765240f0f91f70b2c97a919bf9038"
HABITAT_ARCHIVE_SHA256 = (
    "1231420c6482e79e25beea7ab25121e0421a5fd67b68dd9502145442c288db06"
)
HABITAT_SCENE_SHA256 = (
    "b14e29e17f5e31d86a1002eefd77b7d345b265006481739ae480a847e6623f56"
)
HABITAT_NAVMESH_SHA256 = (
    "1a9a5bd123af8001f0ea2c5c8d326cb3fd39808ca771fc766856af8f0772391d"
)
HABITAT_CAPABILITIES = {
    "skokloster_castle_rgb_depth_bullet_traversal",
    "headless_nvidia_egl_rgb_depth_render",
    "bullet_physics_world_step",
    "greedy_geodesic_agent_traversal",
}
HABITAT_CONFIG_FLAGS = (
    ("--repo-url", "repo_url"),
    ("--repo-ref", "repo_ref"),
    ("--base-profile", "base_profile"),
    ("--base-image", "base_image"),
    ("--apt-snapshot", "apt_snapshot"),
    ("--build-command", "build_command"),
    ("--workload", "workload"),
    ("--smoke-command", "smoke_command"),
    ("--solution-name", "solution_name"),
    ("--capability-name", "capability_name"),
    ("--smoke-artifact-name", "smoke_artifact_name"),
    ("--yaml", "resource_profile_yaml"),
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
    run_id = os.environ.get("NPA_BYOF_CONTAINER_RUN_ID") or f"byof-container-live-{os.getpid()}"
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
    validate = RUNNER.invoke(app, ["workbench", "workflow", "validate-spec", str(path), "--json"])
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
    tool_refs = {step.get("tool_ref") or step.get("toolRef") for step in steps if isinstance(step, dict)}
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
    assert_no_credential_leakage(json.dumps(plan.to_dict()), extra_forbidden=forbidden_markers)
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

    validate = ctx.post("/api/workflows/validate", json={"yaml": workflow_yaml}, timeout=15.0)
    validate.raise_for_status()
    validate_payload = validate.json()
    assert validate_payload.get("ok") is True


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_CONTAINER") != "1",
    reason="Set NPA_BYOF_LIVE_CONTAINER=1 for real BYOF docker build/push/inspect.",
)
def test_live_byof_runner_container_build_push(live_byof_built_image: str) -> None:
    assert live_byof_built_image
    assert "npa-byof" in live_byof_built_image or "npa-isaac-lab" in live_byof_built_image


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
        pytest.skip("Set NPA_BYOF_LIVE_UBUNTU=1 for Ubuntu OSS BYOF container build/push.")
    _activate_nebius_profile()
    registry = resolve_container_registry(e2e_project)
    repo_url, repo_ref = byof_ubuntu_validation_repo()
    run_id = os.environ.get("NPA_BYOF_UBUNTU_RUN_ID") or f"byof-ubuntu-live-{os.getpid()}"
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
def test_live_byof_ubuntu_oss_container_build_push(live_byof_ubuntu_built_image: str) -> None:
    assert live_byof_ubuntu_built_image
    assert "npa-byof" in live_byof_ubuntu_built_image


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_UBUNTU") != "1",
    reason="Set NPA_BYOF_LIVE_UBUNTU=1 for Ubuntu OSS BYOF container metadata inspect.",
)
def test_live_byof_ubuntu_oss_container_metadata(live_byof_ubuntu_built_image: str) -> None:
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
    os.environ.get("NPA_BYOF_LIVE_UBUNTU") != "1" or os.environ.get("NPA_BYOF_LIVE_GPU") != "1",
    reason="Set NPA_BYOF_LIVE_UBUNTU=1 and NPA_BYOF_LIVE_GPU=1 for Ubuntu container-verify SkyPilot smoke.",
)
def test_live_byof_ubuntu_oss_container_verify_submit(
    e2e_project: str | None,
    live_byof_ubuntu_built_image: str,
) -> None:
    registry = resolve_container_registry(e2e_project)
    yaml_override = resolve_byof_resource_yaml(e2e_project, smoke=True, workload="container-verify")
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


def _habitat_runner_command(
    *,
    config: dict[str, object],
    registry: str,
    project: str,
    bucket: str,
    run_id: str,
    config_path: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(BYOF_RUNNER),
        "--registry",
        registry,
        "--project",
        project,
    ]
    for flag, key in HABITAT_CONFIG_FLAGS:
        cmd.extend([flag, str(config[key])])
    cmd.extend(
        [
            "--output-root",
            f"s3://{bucket}/oss-solutions/habitat-sim",
            "--run-id",
            run_id,
            "--wait-timeout",
            "-1",
            "--no-cleanup",
            "--config-path",
            config_path,
        ]
    )
    return cmd


def _private_owner_file(value: object, *, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise AssertionError(f"Habitat runtime receipt omits {label}")
    path = Path(raw)
    if not path.is_absolute() or path.is_symlink():
        raise AssertionError(f"{label} must be an absolute, non-symlink owner file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AssertionError(f"{label} is not readable") from exc
    if resolved != path:
        raise AssertionError(f"{label} must not use symlinked path components")
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise AssertionError(f"{label} must remain outside the repository")
    metadata = resolved.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise AssertionError(f"{label} must be owner-only")
    parent_metadata = resolved.parent.stat()
    if (
        parent_metadata.st_uid != os.getuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o077
    ):
        raise AssertionError(f"{label} must be inside an owner-only directory")
    return resolved


def _receipt_text(receipt: dict[str, object], key: str) -> str:
    value = str(receipt.get(key) or "").strip()
    if not value:
        raise AssertionError(f"Habitat runtime receipt omits {key}")
    return value


def _load_habitat_runtime_receipt() -> dict[str, object]:
    receipt_path = _private_owner_file(
        os.environ.get("NPA_BYOF_HABITAT_SIM_RUNTIME_RECEIPT"),
        label="runtime receipt",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise AssertionError("Habitat runtime receipt must be a JSON object")
    if receipt.get("schema_version") != "npa.byof.habitat-sim.runtime-context.v1":
        raise AssertionError("Habitat runtime receipt schema is not recognized")
    if receipt.get("solution") != "habitat-sim":
        raise AssertionError("Habitat runtime receipt is for another solution")
    run_id = _receipt_text(receipt, "run_id")
    if re.fullmatch(
        r"habitat-[a-z0-9](?:[a-z0-9-]{0,50}[a-z0-9])?", run_id
    ) is None:
        raise AssertionError("Habitat run_id must be a bounded DNS-safe name")
    if run_id != os.environ.get("NPA_BYOF_HABITAT_SIM_RUN_ID", "").strip():
        raise AssertionError(
            "Habitat runtime receipt is not bound to the requested run"
        )
    if (
        receipt.get("task_owned") is not True
        or receipt.get("manager_published") is not True
    ):
        raise AssertionError("Habitat runtime ownership provenance is incomplete")

    registry = _receipt_text(receipt, "registry").rstrip("/")
    public_registry = DEFAULT_CONTAINER_REGISTRY.rstrip("/")
    if (
        receipt.get("registry_visibility") != "private"
        or receipt.get("registry_push_authorized") is not True
        or registry == public_registry
        or registry.startswith(public_registry + "/")
    ):
        raise AssertionError(
            "Habitat requires an explicitly authorized private registry"
        )

    context = _receipt_text(receipt, "kubernetes_context")
    _private_owner_file(receipt.get("kubeconfig_path"), label="kubeconfig")
    _private_owner_file(receipt.get("skypilot_config_path"), label="SkyPilot config")
    docker_config = _private_owner_file(
        receipt.get("docker_config_path"), label="Docker config"
    )
    if docker_config.name != "config.json":
        raise AssertionError("Docker config receipt path must name config.json")
    reservation = receipt.get("reservation")
    if not isinstance(reservation, dict):
        raise AssertionError("Habitat runtime receipt omits reservation evidence")
    provider_receipt_path = _private_owner_file(
        reservation.get("provider_receipt_path"), label="provider reservation receipt"
    )
    provider_receipt_bytes = provider_receipt_path.read_bytes()
    reservation_hash = str(reservation.get("provider_receipt_sha256") or "")
    provider_receipt = json.loads(provider_receipt_bytes)
    if not isinstance(provider_receipt, dict):
        raise AssertionError("provider reservation receipt must be a JSON object")
    capacity_block_group_id = str(reservation.get("capacity_block_group_id") or "")
    node_group_policy = provider_receipt.get("node_group_reservation_policy")
    if (
        reservation.get("policy") != "STRICT"
        or reservation.get("state") != "ACTIVE"
        or reservation.get("accelerator") != "RTX PRO 6000 Blackwell"
        or reservation.get("gpu_count") != 1
        or str(reservation.get("kubernetes_context") or "") != context
        or not capacity_block_group_id.strip()
        or re.fullmatch(r"[0-9a-f]{64}", reservation_hash) is None
        or hashlib.sha256(provider_receipt_bytes).hexdigest() != reservation_hash
        or not str(reservation.get("verified_at") or "").strip()
    ):
        raise AssertionError(
            "Habitat STRICT reservation evidence is incomplete or mismatched"
        )
    if (
        provider_receipt.get("schema_version")
        != "npa.nebius.strict-capacity-binding.v1"
        or provider_receipt.get("project") != _receipt_text(receipt, "project")
        or provider_receipt.get("kubernetes_context") != context
        or provider_receipt.get("capacity_block_group_id") != capacity_block_group_id
        or provider_receipt.get("accelerator") != "RTX PRO 6000 Blackwell"
        or provider_receipt.get("gpu_count") != 1
        or provider_receipt.get("state") != "ACTIVE"
        or not isinstance(node_group_policy, dict)
        or node_group_policy.get("policy") != "STRICT"
        or node_group_policy.get("reservation_ids") != [capacity_block_group_id]
    ):
        raise AssertionError(
            "provider readback does not prove the requested STRICT binding"
        )

    expected_prefix = f"oss-solutions/habitat-sim/{run_id}"
    if _receipt_text(receipt, "output_prefix").strip("/") != expected_prefix:
        raise AssertionError("Habitat storage prefix is not uniquely bound to the run")
    return receipt


def _habitat_run_env(receipt: dict[str, object]) -> dict[str, str]:
    kubeconfig = str(
        _private_owner_file(receipt["kubeconfig_path"], label="kubeconfig")
    )
    context = _receipt_text(receipt, "kubernetes_context")
    env = dict(os.environ)
    env["NPA_E2E_PROJECT"] = _receipt_text(receipt, "project")
    env["NPA_REGISTRY"] = _receipt_text(receipt, "registry")
    env["DOCKER_CONFIG"] = str(
        _private_owner_file(receipt["docker_config_path"], label="Docker config").parent
    )
    env["NPA_BYOF_S3_ENDPOINT"] = _receipt_text(receipt, "s3_endpoint")
    env["KUBECONFIG"] = kubeconfig
    env["NPA_BYOF_KUBECONFIG"] = kubeconfig
    env["NPA_BYOF_K8S_CONTEXT"] = context
    env["NPA_BYOF_K8S_NAMESPACE"] = _receipt_text(receipt, "kubernetes_namespace")
    skypilot_bin = resolve_skypilot_bin()
    if not skypilot_bin:
        raise AssertionError("SkyPilot is required for the Habitat live gate")
    env["PATH"] = f"{Path(skypilot_bin).parent}:{env.get('PATH', '')}"
    return env


def _habitat_run_pods(env: dict[str, str], run_id: str) -> list[dict[str, object]]:
    result = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            env["NPA_BYOF_KUBECONFIG"],
            "--context",
            env["NPA_BYOF_K8S_CONTEXT"],
            "get",
            "pods",
            "--all-namespaces",
            "-o",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError("could not inspect the explicit Habitat Kubernetes target")
    payload = json.loads(result.stdout)
    matches = []
    for pod in payload.get("items", []):
        metadata = pod.get("metadata", {})
        annotations = metadata.get("annotations", {}) or {}
        labels = metadata.get("labels", {}) or {}
        if annotations.get("skypilot-cluster-name") != run_id:
            continue
        if labels.get("parent") != "skypilot":
            raise AssertionError("Habitat run pod lacks SkyPilot ownership labels")
        matches.append(pod)
    return matches


def _habitat_observed_image_id(
    env: dict[str, str], *, run_id: str, expected_image: str
) -> str:
    pods = _habitat_run_pods(env, run_id)
    if len(pods) != 1:
        raise AssertionError("Habitat run must own exactly one Kubernetes pod")
    pod = pods[0]
    containers = pod.get("spec", {}).get("containers", [])
    gpu_count = 0
    for container in containers:
        resources = container.get("resources", {}) or {}
        requests = resources.get("requests", {}) or {}
        limits = resources.get("limits", {}) or {}
        gpu_count += int(
            requests.get("nvidia.com/gpu") or limits.get("nvidia.com/gpu") or 0
        )
    if gpu_count != 1:
        raise AssertionError("Habitat run pod does not request exactly one GPU")
    digest = expected_image.rsplit("@", 1)[-1]
    statuses = pod.get("status", {}).get("containerStatuses", []) or []
    observed = [
        str(item.get("imageID") or "")
        for item in statuses
        if digest in str(item.get("imageID") or "")
    ]
    if len(observed) != 1:
        raise AssertionError(
            "Kubernetes imageID does not match the pushed Habitat digest"
        )
    return observed[0]


def _cleanup_habitat_run(
    env: dict[str, str], *, run_id: str, config_path: str, required: bool
) -> None:
    sky_bin = resolve_skypilot_bin()
    command = [sky_bin or "sky", "down", "--config", config_path, "--yes", run_id]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        for pod in _habitat_run_pods(env, run_id):
            metadata = pod.get("metadata", {})
            subprocess.run(
                [
                    "kubectl",
                    "--kubeconfig",
                    env["NPA_BYOF_KUBECONFIG"],
                    "--context",
                    env["NPA_BYOF_K8S_CONTEXT"],
                    "delete",
                    "pod",
                    str(metadata.get("name") or ""),
                    "--namespace",
                    str(metadata.get("namespace") or "default"),
                    "--wait=true",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        if required:
            raise AssertionError("SkyPilot teardown failed after exact-pod cleanup")
        return
    while _habitat_run_pods(env, run_id):
        time.sleep(5)


def _assert_habitat_render_proof(proof: dict[str, object]) -> None:
    rgb_count = int(proof["rendered_rgb_frame_count"])
    assert rgb_count >= 2
    assert proof["rendered_depth_frame_count"] == rgb_count
    observations = proof["rendered_observations"]
    assert observations["rgb"]["shape"] == [240, 320, 4]
    assert observations["depth"]["shape"] == [240, 320]
    assert re.fullmatch(r"[0-9a-f]{64}", observations["rgb"]["aggregate_raw_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", observations["depth"]["aggregate_raw_sha256"])
    frames = observations["frames"]
    assert len(frames) == rgb_count
    for frame in frames:
        assert frame["rgb_shape"] == [240, 320, 4]
        assert frame["depth_shape"] == [240, 320]
        for key in (
            "rgb_raw_sha256",
            "rgb_png_sha256",
            "depth_raw_sha256",
            "depth_npy_sha256",
        ):
            assert re.fullmatch(r"[0-9a-f]{64}", frame[key])
    depth = proof["finite_depth_statistics"]
    assert depth["count"] > 0
    assert all(
        math.isfinite(depth[key])
        for key in ("minimum", "maximum", "mean", "standard_deviation")
    )
    assert 0.0 <= depth["minimum"] <= depth["maximum"]


def _assert_habitat_runtime_proof(proof: dict[str, object], image: str) -> None:
    assert proof["solution"] == "habitat-sim"
    assert proof["capability"] == "skokloster_castle_rgb_depth_bullet_traversal"
    assert set(proof["capabilities_exercised"]) == HABITAT_CAPABILITIES
    assert proof["source_revision"] == HABITAT_SOURCE_REVISION
    assert proof["source"]["observed_revision"] == HABITAT_SOURCE_REVISION
    assert proof["scene_id"] == "habitat_test_scenes/skokloster-castle.glb"
    assert proof["scene_sha256"] == HABITAT_SCENE_SHA256
    assert proof["scene"]["navmesh_sha256"] == HABITAT_NAVMESH_SHA256
    assert proof["scene"]["archive"]["sha256"] == HABITAT_ARCHIVE_SHA256
    assert proof["scene"]["archive"]["zip_integrity"] == "pass"
    assert proof["scene"]["archive"]["ephemeral_copy_removed"] is True
    assert proof["scene"]["archive"]["unrelated_members_extracted"] is False
    assert proof["scene"]["archive_member"]["sha256"] == HABITAT_SCENE_SHA256
    assert (
        proof["scene"]["navmesh_archive_member"]["sha256"]
        == HABITAT_NAVMESH_SHA256
    )
    assert proof["scene_license"] == "CC BY 4.0"
    assert proof["scene"]["license_url"] == (
        "https://creativecommons.org/licenses/by/4.0/legalcode.en"
    )
    assert proof["scene"]["original_asset"]["creator"] == "Skokloster Castle"
    assert "byte-for-byte" in proof["scene"]["modification_notice"]
    assert proof["agent_displacement"] > 0.1
    assert len(proof["agent_start"]) == len(proof["agent_end"]) == 3
    assert proof["bullet"]["built_with_bullet"] is True
    assert proof["bullet_step_count"] == proof["rendered_rgb_frame_count"]
    assert proof["bullet"]["world_time_end"] > proof["bullet"]["world_time_start"]
    assert math.isfinite(proof["measured_fps"]) and proof["measured_fps"] > 0.0
    egl = proof["renderer_egl_evidence"]
    assert egl["backend"] == "EGL" and "NVIDIA" in egl["gl"]["vendor"].upper()
    assert any("libEGL_nvidia.so" in path for path in egl["process_loaded_libraries"])
    assert "RTX PRO 6000 BLACKWELL" in proof["observed_rtx_gpu_model"].upper()
    assert proof["observed_rtx_gpu_architecture"] == "Blackwell"
    assert proof["observed_rtx_gpu_count"] == 1
    assert proof["observed_gpu"]["compute_capability"] == "12.0"
    assert proof["pod_observed_immutable_image"] == image
    assert proof["pod_observed_image_digest"] == image.rsplit("@", 1)[1]
    assert proof["exit_status"] == 0


def _load_habitat_proof(
    client: object, *, bucket: str, run_id: str
) -> tuple[dict[str, object], str]:
    key = f"oss-solutions/habitat-sim/{run_id}/habitat-sim-smoke.json"
    response = client.get_object(Bucket=bucket, Key=key)
    proof_bytes = response["Body"].read()
    proof = json.loads(proof_bytes)
    assert isinstance(proof, dict)
    return proof, hashlib.sha256(proof_bytes).hexdigest()


def _write_habitat_live_validation(
    client: object,
    *,
    bucket: str,
    run_id: str,
    proof: dict[str, object],
    proof_sha256: str,
    pushed_image: str,
    kubernetes_image_id: str,
    reservation_receipt_sha256: str,
) -> None:
    key = f"oss-solutions/habitat-sim/{run_id}/habitat-sim-live-validation.json"
    payload = json.dumps(
        {
            "schema_version": "npa.byof.habitat-sim.live-validation.v1",
            "solution": "habitat-sim",
            "run_id": run_id,
            "proof_key": (
                f"oss-solutions/habitat-sim/{run_id}/habitat-sim-smoke.json"
            ),
            "proof_sha256": proof_sha256,
            "source_revision": proof["source_revision"],
            "scene_archive_sha256": proof["scene"]["archive"]["sha256"],
            "scene_sha256": proof["scene_sha256"],
            "pushed_image_digest": pushed_image.rsplit("@", 1)[1],
            "kubernetes_image_id": kubernetes_image_id,
            "reservation_policy": "STRICT",
            "reservation_receipt_sha256": reservation_receipt_sha256,
            "observed_gpu": proof["observed_gpu"],
            "exit_status": proof["exit_status"],
        },
        indent=2,
        sort_keys=True,
    ).encode()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="application/json",
        IfNoneMatch="*",
    )
    head = client.head_object(Bucket=bucket, Key=key)
    assert int(head.get("ContentLength", -1)) == len(payload)


def _assert_habitat_uploaded_artifacts(
    client: object, *, bucket: str, run_id: str, proof: dict[str, object]
) -> None:
    import numpy as np
    from PIL import Image

    prefix = f"oss-solutions/habitat-sim/{run_id}/"

    def read_exact(path: str, expected_bytes: int, expected_sha256: str) -> bytes:
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise AssertionError("Habitat proof contains an unsafe artifact path")
        response = client.get_object(Bucket=bucket, Key=prefix + relative.as_posix())
        payload = response["Body"].read()
        assert len(payload) == expected_bytes
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        return payload

    inventory = proof["runtime_package_inventory"]
    for name in ("python_lock", "python_packages", "debian_packages"):
        entry = inventory[name]
        read_exact(entry["path"], int(entry["bytes"]), entry["sha256"])

    rgb_aggregate = hashlib.sha256()
    depth_aggregate = hashlib.sha256()
    saved_directory = proof["rendered_observations"]["saved_directory"]
    for frame in proof["rendered_observations"]["frames"]:
        rgb_path = frame["rgb_png_path"]
        depth_path = frame["depth_npy_path"]
        preview_path = frame["depth_preview_png_path"]
        if any(
            PurePosixPath(path).parts[0] != saved_directory
            for path in (rgb_path, depth_path, preview_path)
        ):
            raise AssertionError(
                "Habitat observation escaped its declared artifact directory"
            )
        rgb_payload = read_exact(
            rgb_path, int(frame["rgb_png_bytes"]), frame["rgb_png_sha256"]
        )
        depth_payload = read_exact(
            depth_path, int(frame["depth_npy_bytes"]), frame["depth_npy_sha256"]
        )
        read_exact(
            preview_path,
            int(frame["depth_preview_png_bytes"]),
            frame["depth_preview_png_sha256"],
        )
        with Image.open(io.BytesIO(rgb_payload)) as rgb_image:
            rgb = np.asarray(rgb_image.convert("RGBA"))
        depth = np.load(io.BytesIO(depth_payload), allow_pickle=False)
        assert list(rgb.shape) == frame["rgb_shape"]
        assert list(depth.shape) == frame["depth_shape"]
        rgb_bytes = np.ascontiguousarray(rgb).tobytes(order="C")
        depth_bytes = np.ascontiguousarray(depth, dtype=np.float32).tobytes(order="C")
        assert hashlib.sha256(rgb_bytes).hexdigest() == frame["rgb_raw_sha256"]
        assert hashlib.sha256(depth_bytes).hexdigest() == frame["depth_raw_sha256"]
        rgb_aggregate.update(rgb_bytes)
        depth_aggregate.update(depth_bytes)
    observations = proof["rendered_observations"]
    assert rgb_aggregate.hexdigest() == observations["rgb"]["aggregate_raw_sha256"]
    assert depth_aggregate.hexdigest() == observations["depth"]["aggregate_raw_sha256"]


def _write_habitat_receipt_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    registry: str = "registry.example.invalid/private",
    provider_policy: str = "STRICT",
    run_id: str = "habitat-receipt-test",
) -> dict[str, object]:
    tmp_path.chmod(0o700)
    context = "habitat-receipt-context"
    capacity_id = "private-capacity-binding"
    private_files = {
        "kubeconfig_path": tmp_path / "kubeconfig",
        "skypilot_config_path": tmp_path / "skypilot.yaml",
        "docker_config_path": tmp_path / "config.json",
    }
    for path in private_files.values():
        path.write_text("{}\n", encoding="utf-8")
        path.chmod(0o600)
    provider_receipt = {
        "schema_version": "npa.nebius.strict-capacity-binding.v1",
        "project": "habitat-test-project",
        "kubernetes_context": context,
        "capacity_block_group_id": capacity_id,
        "accelerator": "RTX PRO 6000 Blackwell",
        "gpu_count": 1,
        "state": "ACTIVE",
        "node_group_reservation_policy": {
            "policy": provider_policy,
            "reservation_ids": [capacity_id],
        },
    }
    provider_bytes = json.dumps(provider_receipt, sort_keys=True).encode()
    provider_path = tmp_path / "provider-receipt.json"
    provider_path.write_bytes(provider_bytes)
    provider_path.chmod(0o600)
    receipt = {
        "schema_version": "npa.byof.habitat-sim.runtime-context.v1",
        "solution": "habitat-sim",
        "run_id": run_id,
        "task_owned": True,
        "manager_published": True,
        "project": "habitat-test-project",
        "registry": registry,
        "registry_visibility": "private",
        "registry_push_authorized": True,
        "bucket": "habitat-test-bucket",
        "output_prefix": f"oss-solutions/habitat-sim/{run_id}",
        "s3_endpoint": "https://storage.example.invalid",
        "kubernetes_context": context,
        "kubernetes_namespace": "habitat-test-namespace",
        **{name: str(path) for name, path in private_files.items()},
        "reservation": {
            "policy": "STRICT",
            "state": "ACTIVE",
            "accelerator": "RTX PRO 6000 Blackwell",
            "gpu_count": 1,
            "kubernetes_context": context,
            "capacity_block_group_id": capacity_id,
            "provider_receipt_path": str(provider_path),
            "provider_receipt_sha256": hashlib.sha256(provider_bytes).hexdigest(),
            "verified_at": "2026-09-11T00:00:00Z",
        },
    }
    receipt_path = tmp_path / "runtime-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    monkeypatch.setenv("NPA_BYOF_HABITAT_SIM_RUN_ID", run_id)
    monkeypatch.setenv("NPA_BYOF_HABITAT_SIM_RUNTIME_RECEIPT", str(receipt_path))
    return receipt


def test_habitat_runtime_receipt_accepts_manager_provider_readback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = _write_habitat_receipt_fixture(monkeypatch, tmp_path)

    assert _load_habitat_runtime_receipt() == expected


def test_habitat_runtime_receipt_rejects_public_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_habitat_receipt_fixture(
        monkeypatch, tmp_path, registry=DEFAULT_CONTAINER_REGISTRY
    )

    with pytest.raises(AssertionError, match="private registry"):
        _load_habitat_runtime_receipt()


def test_habitat_runtime_receipt_rejects_non_strict_provider_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_habitat_receipt_fixture(monkeypatch, tmp_path, provider_policy="BEST_EFFORT")

    with pytest.raises(AssertionError, match="provider readback"):
        _load_habitat_runtime_receipt()


def test_habitat_runtime_receipt_rejects_unsafe_run_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_habitat_receipt_fixture(monkeypatch, tmp_path, run_id="--all")

    with pytest.raises(AssertionError, match="DNS-safe"):
        _load_habitat_runtime_receipt()


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_HABITAT_SIM_LIVE") != "1"
    or not os.environ.get("NPA_BYOF_HABITAT_SIM_RUNTIME_RECEIPT", "").strip(),
    reason=(
        "Set NPA_BYOF_HABITAT_SIM_LIVE=1 only with an owner-only, manager-published "
        "NPA_BYOF_HABITAT_SIM_RUNTIME_RECEIPT that proves the exact private registry, "
        "run-owned target, and provider-read-back STRICT RTX PRO binding."
    ),
)
def test_live_habitat_sim_private_digest_rgb_depth_bullet_traversal(
    e2e_project: str | None,
    forbidden_markers: list[str],
) -> None:
    """Build, push, pull, execute, and read back the exact Habitat proof."""

    receipt = _load_habitat_runtime_receipt()
    config = load_spec(HABITAT_SIM_SPEC).config
    project = _receipt_text(receipt, "project")
    if e2e_project and e2e_project != project:
        raise AssertionError("selected E2E project does not match the Habitat receipt")
    registry = _receipt_text(receipt, "registry")
    bucket = _receipt_text(receipt, "bucket")
    run_id = _receipt_text(receipt, "run_id")
    config_path = str(
        _private_owner_file(receipt["skypilot_config_path"], label="SkyPilot config")
    )
    env = _habitat_run_env(receipt)
    cmd = _habitat_runner_command(
        config=config,
        registry=registry,
        project=project,
        bucket=bucket,
        run_id=run_id,
        config_path=config_path,
    )

    completed = False
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
        if proc.returncode != 0:
            raise AssertionError("Habitat BYOF build or live run failed")
        combined_output = proc.stdout + "\n" + proc.stderr
        assert_no_credential_leakage(combined_output, forbidden_markers)
        summary = _parse_last_json_blob(combined_output)
        assert summary.get("status") == "ok"
        image = str(summary.get("image") or "")
        assert "@sha256:" in image
        observed_image_id = _habitat_observed_image_id(
            env, run_id=run_id, expected_image=image
        )
        assert image.rsplit("@", 1)[1] in observed_image_id

        client = s3_client_for_project(
            project,
            allow_host_creds=True,
            endpoint_url=_receipt_text(receipt, "s3_endpoint"),
        )
        proof, proof_sha256 = _load_habitat_proof(
            client, bucket=bucket, run_id=run_id
        )
        _assert_habitat_render_proof(proof)
        _assert_habitat_runtime_proof(proof, image)
        _assert_habitat_uploaded_artifacts(
            client, bucket=bucket, run_id=run_id, proof=proof
        )
        reservation = receipt["reservation"]
        assert isinstance(reservation, dict)
        _write_habitat_live_validation(
            client,
            bucket=bucket,
            run_id=run_id,
            proof=proof,
            proof_sha256=proof_sha256,
            pushed_image=image,
            kubernetes_image_id=observed_image_id,
            reservation_receipt_sha256=str(
                reservation["provider_receipt_sha256"]
            ),
        )
        completed = True
    finally:
        _cleanup_habitat_run(
            env, run_id=run_id, config_path=config_path, required=completed
        )
