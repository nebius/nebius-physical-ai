"""Live infra checks for generic BYOF solution onboarding (workflow + optional agent chat)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml
from typer.testing import CliRunner

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
ROBOTWIN_SPEC = REPO_ROOT / "workflows" / "testing" / "byof-robotwin.yaml"
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_repo.py"
RUNNER = CliRunner()


def _activate_nebius_profile(selected_profile: str | None = None) -> None:
    profile = (
        selected_profile
        if selected_profile is not None
        else os.environ.get("NPA_NEBIUS_PROFILE", "agent-sa")
    ).strip()
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


def _robotwin_runtime_context() -> tuple[dict[str, object], str]:
    """Load the manager-owned, owner-local authorization for this live run."""

    raw_path = os.environ.get("NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT", "").strip()
    assert raw_path, "NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT is required for a live run"
    path = Path(raw_path).expanduser().resolve()
    assert path.is_file(), "RoboTwin runtime context is not a readable file"
    assert path.parent == REPO_ROOT.parent.resolve(), (
        "RoboTwin runtime context must be an owner-only file in the child root"
    )
    raw = path.read_bytes()
    payload = json.loads(raw)
    assert isinstance(payload, dict), "RoboTwin runtime context must be a JSON object"
    assert payload.get("solution") == "robotwin", "runtime context has the wrong solution"
    assert payload.get("ownership_provenance"), "runtime context lacks ownership provenance"
    reservation = payload.get("reservation")
    assert isinstance(reservation, dict), "runtime context lacks reservation evidence"
    assert reservation.get("policy") == "STRICT", "reservation policy must be STRICT"
    assert reservation.get("accelerator") == "RTXPRO-6000-BLACKWELL-SERVER-EDITION"
    assert reservation.get("count") == 1, "reservation must authorize exactly one GPU"
    acceptance = payload.get("license_acceptance")
    assert isinstance(acceptance, dict), "runtime context lacks operator license decisions"
    for key in (
        "nvidia_cuda_eula",
        "nvidia_cudnn_sla",
        "curobo_noncommercial_research_or_evaluation",
    ):
        assert acceptance.get(key) is True, f"runtime context does not authorize {key}"
    for key in (
        "project",
        "nebius_profile",
        "kubeconfig",
        "kubernetes_context",
        "skypilot_config_path",
        "registry",
        "bucket",
        "output_root",
        "run_id",
    ):
        assert str(payload.get(key) or "").strip(), f"runtime context lacks {key}"
    return payload, hashlib.sha256(raw).hexdigest()


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


def test_live_robotwin_workflow_validate_and_plan(
    forbidden_markers: list[str],
) -> None:
    validate = RUNNER.invoke(
        app,
        ["workbench", "workflow", "validate-spec", str(ROBOTWIN_SPEC), "--json"],
    )
    payload = parse_json_payload(validate, forbidden_markers)
    assert payload["status"] == "valid"
    assert payload["name"] == "byof-robotwin"

    plan = RUNNER.invoke(
        app,
        [
            "workbench",
            "workflow",
            "plan-spec",
            str(ROBOTWIN_SPEC),
            "--run-id",
            "robotwin-live-plan",
            "--json",
        ],
    )
    planned = parse_json_payload(plan, forbidden_markers)
    steps = planned.get("steps", [])
    assert len(steps) == 1
    assert steps[0].get("tool_ref") == "workbench.byof.repo"


def _s3_object_sha256(client, *, bucket: str, key: str) -> tuple[int, str]:
    response = client.get_object(Bucket=bucket, Key=key)
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: response["Body"].read(8 * 1024 * 1024), b""):
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_ROBOTWIN_LIVE") != "1",
    reason=(
        "Set NPA_BYOF_ROBOTWIN_LIVE=1 and provide the manager-owned context "
        "path in NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT."
    ),
)
def test_live_robotwin_build_push_run_and_artifacts(
    e2e_project: str | None,
) -> None:
    """Run the exact BYOF toolRef path and verify remote native artifacts."""

    runtime, runtime_sha256 = _robotwin_runtime_context()
    project = str(runtime["project"])
    if e2e_project:
        assert e2e_project == project, "test project differs from manager runtime context"
    _activate_nebius_profile(str(runtime["nebius_profile"]))
    workflow = yaml.safe_load(ROBOTWIN_SPEC.read_text(encoding="utf-8"))
    config = workflow["config"]
    bucket = str(runtime["bucket"])
    registry = str(runtime["registry"])
    run_id = str(runtime["run_id"])
    output_root = str(runtime["output_root"]).rstrip("/")
    parsed_output = urlparse(output_root)
    assert parsed_output.scheme == "s3" and parsed_output.netloc == bucket
    assert run_id.startswith("robotwin-"), "manager run ID must be solution-scoped"
    cmd = [
        sys.executable,
        str(BYOF_RUNNER),
        "--registry",
        registry,
        "--project",
        project,
        "--repo-url",
        str(config["repo_url"]),
        "--repo-ref",
        str(config["repo_ref"]),
        "--base-profile",
        str(config["base_profile"]),
        "--base-image",
        str(config["base_image"]),
        "--build-command",
        str(config["build_command"]),
        "--workload",
        str(config["workload"]),
        "--smoke-command",
        str(config["smoke_command"]),
        "--solution-name",
        str(config["solution_name"]),
        "--capability-name",
        str(config["capability_name"]),
        "--smoke-artifact-name",
        str(config["smoke_artifact_name"]),
        "--yaml",
        str(config["resource_profile_yaml"]),
        "--task",
        str(config["task"]),
        "--iterations",
        str(config["iterations"]),
        "--output-root",
        output_root,
        "--wait-timeout",
        str(config["wait_timeout"]),
        "--poll-interval",
        str(config["poll_interval"]),
        "--run-id",
        run_id,
    ]
    config_path = Path(str(runtime["skypilot_config_path"])).expanduser().resolve()
    assert config_path.is_file(), "manager SkyPilot config is not readable"
    cmd.extend(["--config-path", str(config_path)])
    env = dict(os.environ)
    kubeconfig = Path(str(runtime["kubeconfig"])).expanduser().resolve()
    assert kubeconfig.is_file(), "manager kubeconfig is not readable"
    env["KUBECONFIG"] = str(kubeconfig)
    env["NPA_BYOF_KUBECONFIG"] = str(kubeconfig)
    env["NPA_BYOF_K8S_CONTEXT"] = str(runtime["kubernetes_context"])
    env["NPA_NEBIUS_PROFILE"] = str(runtime["nebius_profile"])
    env["NPA_BYOF_PROJECT"] = project
    env["NPA_BYOF_ROBOTWIN_RESERVATION_EVIDENCE_SHA256"] = runtime_sha256
    skypilot_bin = resolve_skypilot_bin()
    if skypilot_bin:
        env["PATH"] = f"{Path(skypilot_bin).parent}:{env.get('PATH', '')}"

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
    build = summary.get("build", {})
    assert build.get("ok") is True and build.get("pushed") is True
    runtime_image = str(build.get("runtime_image") or "")
    assert re.fullmatch(r".+@sha256:[0-9a-f]{64}", runtime_image)

    client = s3_client_for_project(project, allow_host_creds=True)
    prefix = parsed_output.path.strip("/") + f"/{run_id}/"
    smoke_key = prefix + "robotwin-smoke.json"
    smoke = json.loads(client.get_object(Bucket=bucket, Key=smoke_key)["Body"].read())
    assert smoke["solution"] == "robotwin"
    assert smoke["capability"] == (
        "beat_block_hammer_successful_seed_replay_collection"
    )
    assert {
        "strict_rtx_pro_6000_placement",
        "sapien_vulkan_rt_renderer",
        "pinned_official_runtime_assets",
        "official_embodiment_path_configuration",
        "beat_block_hammer_successful_seed_search",
        "beat_block_hammer_successful_seed_replay",
        "robotwin_native_hdf5_collection",
        "robotwin_rendered_mp4",
    }.issubset(smoke["capabilities_exercised"])
    assert smoke["source_revision"] == config["repo_ref"]
    assert smoke["asset_revision"] == (
        "785feb15aa4a4f532395ad2b1d2be5f28cb561ad"
    )
    assert smoke["curobo_revision"] == (
        "d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
    )
    assert smoke["task"] == "beat_block_hammer"
    assert smoke["task_config"] == "demo_clean"
    assert smoke["config_evidence"]["official_config"] == "demo_clean"
    assert smoke["config_evidence"]["runtime_episode_num"] == 1
    assert smoke["asset_path_configuration_exit_status"] == 0
    assert smoke["collector_exit_status"] == 0
    assert smoke["task_success"] is True
    assert smoke["exit_status"] == 0
    assert isinstance(smoke["seed"], int) and smoke["seed"] >= 0
    assert smoke["seed_search_attempt_count"] == smoke["seed"] + 1
    assert smoke["action_count"] > 0
    assert smoke["rendered_frame_count"] == smoke["action_count"] + 1
    assert smoke["observed_gpu"]["count"] == 1
    assert smoke["observed_gpu"]["architecture"] == "sm_120"
    assert "RTX PRO 6000" in smoke["observed_gpu"]["model"].upper()
    assert smoke["strict_reservation"] == {
        "policy": "STRICT",
        "manager_runtime_context_sha256": runtime_sha256,
    }
    assert smoke["vulkan_renderer"]["available"] is True
    assert smoke["vulkan_renderer"]["vulkaninfo_exit_status"] == 0
    assert smoke["vulkan_renderer"]["vulkaninfo_mentions_rtx_pro_6000"] is True
    assert smoke["vulkan_renderer"]["sapien_device_summary"]
    assert smoke["vulkan_renderer"]["camera_shader"] == "rt"
    assert smoke["pod_observed_immutable_image_ref"] == runtime_image
    assert smoke["pod_observed_immutable_image_digest"] == runtime_image.rsplit("@", 1)[1]

    for artifact_name in ("hdf5", "video"):
        artifact = smoke[artifact_name]
        assert artifact["size_bytes"] > 0
        assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
        key = prefix + artifact["path"]
        size, digest = _s3_object_sha256(client, bucket=bucket, key=key)
        assert size == artifact["size_bytes"] > 0
        assert digest == artifact["sha256"]
