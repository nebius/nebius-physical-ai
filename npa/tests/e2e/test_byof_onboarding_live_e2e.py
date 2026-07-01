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
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    # Fall back to parsing the last JSON object in multi-line output.
    start = text.rfind("{")
    if start >= 0:
        try:
            payload = json.loads(text[start:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no JSON object found in command output:\n{text}")


def _default_byof_resource_yaml(e2e_project: str | None) -> str:
    if os.environ.get("NPA_BYOF_RESOURCE_YAML"):
        return os.environ["NPA_BYOF_RESOURCE_YAML"]
    if (e2e_project or "").strip().lower() == "rtxpro" and RTXPRO_TRAIN_YAML.is_file():
        return str(RTXPRO_TRAIN_YAML)
    return str(DEFAULT_TRAIN_YAML)


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
    assert summary["status"] == "ok"
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
    assert summary["status"] == "ok"
    assert summary["registry"] == registry
    assert registry in summary["image"]


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_GPU") != "1",
    reason="Set NPA_BYOF_LIVE_GPU=1 to submit a real Isaac BYOF SkyPilot smoke (build/push/run).",
)
def test_live_byof_runner_submit_smoke(e2e_project: str | None) -> None:
    registry = resolve_container_registry(e2e_project)
    yaml_override = _default_byof_resource_yaml(e2e_project)
    task = os.environ.get("NPA_BYOF_TASK", "Isaac-Cartpole-v0")
    proc = subprocess.run(
        [
            sys.executable,
            str(BYOF_RUNNER),
            "--registry",
            registry,
            "--yaml",
            yaml_override,
            "--task",
            task,
            "--iterations",
            "1",
            "--run-id",
            f"byof-live-submit-{os.getpid()}",
            "--skip-build",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=int(os.environ.get("NPA_BYOF_LIVE_TIMEOUT", "21600")),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = _parse_last_json_blob(proc.stdout + "\n" + proc.stderr)
    assert summary["status"] == "ok"
    run_summary = summary.get("run", {})
    assert isinstance(run_summary, dict)
    assert run_summary.get("status") in {"submitted", "ok", "running", "succeeded"} or run_summary.get("skipped") is not True
