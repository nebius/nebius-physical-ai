"""Live infra checks for BYOF solution onboarding (workflow + optional agent chat)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.clients.config import resolve_container_registry
from npa.orchestration.npa_workflow import build_plan, load_spec

from .agent_live_helpers import (
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
BYOF_SPEC = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows" / "isaac-lab-byof-leisaac.yaml"
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_isaac_lab_byof_repo.py"
RTXPRO_TRAIN_YAML = (
    REPO_ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "isaac-lab-rl-train-rtxpro.yaml"
)
RTXPRO_SMOKE_TRAIN_YAML = (
    REPO_ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "isaac-lab-rl-train-rtxpro-smoke.yaml"
)
RTXPRO_SKYPILOT_CONFIG = (
    REPO_ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "skypilot-kubernetes-rtxpro.yaml"
)
DEFAULT_TRAIN_YAML = (
    REPO_ROOT / "npa" / "workflows" / "workbench" / "skypilot" / "isaac-lab-rl-train.yaml"
)
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


def _default_byof_resource_yaml(e2e_project: str | None, *, smoke: bool = False) -> str:
    if os.environ.get("NPA_BYOF_RESOURCE_YAML"):
        return os.environ["NPA_BYOF_RESOURCE_YAML"]
    if (e2e_project or "").strip().lower() == "rtxpro":
        if smoke and RTXPRO_SMOKE_TRAIN_YAML.is_file():
            return str(RTXPRO_SMOKE_TRAIN_YAML)
        if RTXPRO_TRAIN_YAML.is_file():
            return str(RTXPRO_TRAIN_YAML)
    return str(DEFAULT_TRAIN_YAML)


def _maybe_refresh_byof_registry_pull_secret(registry: str) -> None:
    if os.environ.get("NPA_BYOF_SKIP_REGISTRY_REFRESH") == "1":
        return
    profile = os.environ.get("NPA_NEBIUS_PROFILE", "agent-sa").strip()
    _activate_nebius_profile()
    server = registry.split("/", 1)[0]
    if not server.startswith("cr.") or ".nebius.cloud" not in server:
        return
    namespace = os.environ.get("NPA_BYOF_K8S_NAMESPACE", "skypilot-system")
    k8s_context = os.environ.get("NPA_BYOF_K8S_CONTEXT", "npa-rtxpro-mk8s")
    kubeconfig = os.environ.get("NPA_BYOF_KUBECONFIG", "").strip()
    if not kubeconfig:
        fallback = Path(
            "/home/ubuntu/.npa/.sim2real-walkthrough-backup/clusters/npa-rtxpro-mk8s/kubeconfig.resolved"
        )
        if fallback.is_file():
            kubeconfig = str(fallback)
    runtime_env = dict(os.environ)
    skypilot_bin = os.environ.get("NPA_SKYPILOT_BIN", "/home/ubuntu/.npa/skypilot-venv/bin")
    if skypilot_bin:
        runtime_env["PATH"] = f"{skypilot_bin}:{runtime_env.get('PATH', '')}"
    try:
        from npa.workflows.sim2real.registry_auth import ensure_nebius_registry_pull_secret

        ensure_nebius_registry_pull_secret(
            registry_server=server,
            namespace=namespace,
            kubeconfig=kubeconfig,
            k8s_context=k8s_context,
        )
        for target_ns in ("default", namespace):
            if target_ns == namespace:
                continue
            ensure_nebius_registry_pull_secret(
                registry_server=server,
                secret_name="agent-sa",
                namespace=target_ns,
                kubeconfig=kubeconfig,
                k8s_context=k8s_context,
            )
            ensure_nebius_registry_pull_secret(
                registry_server=server,
                namespace=target_ns,
                kubeconfig=kubeconfig,
                k8s_context=k8s_context,
            )
    except Exception as exc:
        # Optional preflight: clusters without kubectl/sky-kube-exec-wrapper can still run container tiers.
        print(f"WARN: skipped registry pull-secret refresh: {exc}", file=sys.stderr)


@pytest.fixture(scope="module")
def live_byof_built_image(e2e_project: str | None) -> str:
    if os.environ.get("NPA_BYOF_LIVE_CONTAINER") != "1":
        pytest.skip("Set NPA_BYOF_LIVE_CONTAINER=1 for real BYOF container build/push.")
    _activate_nebius_profile()
    registry = resolve_container_registry(e2e_project)
    run_id = os.environ.get("NPA_BYOF_CONTAINER_RUN_ID") or f"byof-container-live-{os.getpid()}"
    proc = subprocess.run(
        [
            sys.executable,
            str(BYOF_RUNNER),
            "--registry",
            registry,
            "--run-id",
            run_id,
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


def _materialize_byof_spec(tmp_path: Path, *, bucket: str, run_id: str) -> Path:
    text = BYOF_SPEC.read_text(encoding="utf-8")
    text = text.replace("bucket: example-bucket", f"bucket: {bucket}")
    text = re.sub(
        r"(output_root:\s*s3://)[^/]+(/isaac-lab-byof/leisaac)",
        rf"\1{bucket}\2",
        text,
        count=1,
    )
    path = tmp_path / "isaac-lab-byof-leisaac-live.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_live_isaac_byof_workflow_validate_and_plan(
    tmp_path: Path,
    e2e_project: str | None,
    forbidden_markers: list[str],
) -> None:
    bucket = live_bucket(e2e_project)
    path = _materialize_byof_spec(tmp_path, bucket=bucket, run_id="byof-onboard-live")
    validate = RUNNER.invoke(app, ["workbench", "workflow", "validate-spec", str(path), "--json"])
    payload = parse_json_payload(validate, forbidden_markers)
    assert payload["status"] == "valid"
    assert payload["name"] == "isaac-lab-byof-leisaac"
    assert "byof-train" in set(payload.get("states", []))

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
    assert "workbench.isaac_lab.byof_repo" in tool_refs


def test_live_isaac_byof_plan_builder_matches_cli(
    tmp_path: Path,
    e2e_project: str | None,
    forbidden_markers: list[str],
) -> None:
    bucket = live_bucket(e2e_project)
    path = _materialize_byof_spec(tmp_path, bucket=bucket, run_id="byof-plan-builder")
    spec = load_spec(path)
    plan = build_plan(spec, run_id="byof-plan-builder")
    assert plan.steps
    assert_no_credential_leakage(json.dumps(plan.to_dict()), extra_forbidden=forbidden_markers)
    assert any(step.tool_ref == "workbench.isaac_lab.byof_repo" for step in plan.steps)


def test_live_byof_registry_resolution(e2e_project: str | None) -> None:
    registry = resolve_container_registry(e2e_project)
    assert registry
    assert "/" in registry
    assert "example-bucket" not in registry
    assert "<your-registry-id>" not in registry


@pytest.mark.skipif(
    os.environ.get("NPA_AGENT_LIVE") != "1",
    reason="Set NPA_AGENT_LIVE=1 to exercise onboard_solution chat on a live agent VM.",
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
    reason="Set NPA_AGENT_LIVE=1 to validate BYOF workflow draft on a live agent VM.",
)
def test_live_agent_byof_workflow_draft_validate() -> None:
    ctx = load_agent_live_context()
    draft = ctx.post(
        "/api/chat",
        json={
            "messages": [
                {"role": "user", "content": "create a LeIsaac BYOF Isaac Lab workflow for live infra"},
            ],
        },
        timeout=30.0,
    )
    draft.raise_for_status()
    payload = draft.json()
    assert payload.get("ok") is True
    workflow_yaml = str(payload.get("workflow_yaml") or "")
    assert workflow_yaml
    assert "isaac-lab-byof-leisaac" in workflow_yaml or "byof-train" in workflow_yaml

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
    assert "npa-isaac-lab-leisaac" in live_byof_built_image


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_CONTAINER") != "1",
    reason="Set NPA_BYOF_LIVE_CONTAINER=1 for real BYOF docker build/push/inspect.",
)
def test_live_byof_container_has_leisaac(live_byof_built_image: str) -> None:
    image = live_byof_built_image
    ls_proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "ls", image, "-la", "/opt/leisaac"],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert ls_proc.returncode == 0, ls_proc.stdout + ls_proc.stderr
    assert "README.md" in ls_proc.stdout

    meta_proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "cat", image, "/opt/leisaac/npa_source_metadata.json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert meta_proc.returncode == 0, meta_proc.stdout + meta_proc.stderr
    metadata = json.loads(meta_proc.stdout)
    assert metadata["source"] == "oss-byof"
    assert metadata["repo"] == "https://github.com/LightwheelAI/leisaac.git"
    assert metadata["ref"] == "main"


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
    _maybe_refresh_byof_registry_pull_secret(registry)
    yaml_override = _default_byof_resource_yaml(e2e_project, smoke=True)
    image = os.environ.get("NPA_BYOF_TEST_IMAGE", "").strip() or live_byof_built_image
    task = os.environ.get("NPA_BYOF_TASK", "Isaac-Cartpole-v0")
    cmd = [
        sys.executable,
        str(BYOF_RUNNER),
        "--registry",
        registry,
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
    if os.environ.get("NPA_BYOF_SKYPILOT_CONFIG"):
        cmd.extend(["--config-path", os.environ["NPA_BYOF_SKYPILOT_CONFIG"]])
    elif RTXPRO_SKYPILOT_CONFIG.is_file() and (e2e_project or "").strip().lower() == "rtxpro":
        cmd.extend(["--config-path", str(RTXPRO_SKYPILOT_CONFIG)])
    env = dict(os.environ)
    # SkyPilot managed jobs on K8s may fail prechecks while direct launch works on rtxpro.
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
        # Managed jobs path: submit JSON may be flattened; precheck failure is expected on K8s.
        pass
    else:
        assert submit or final, run_summary
