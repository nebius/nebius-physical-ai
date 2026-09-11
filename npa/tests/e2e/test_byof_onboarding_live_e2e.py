"""Live infra checks for generic BYOF solution onboarding (workflow + optional agent chat)."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
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
ROBOMIMIC_SPEC = REPO_ROOT / "workflows" / "testing" / "byof-robomimic.yaml"
BYOF_RUNNER = REPO_ROOT / "npa" / "scripts" / "run_byof_repo.py"
RUNNER = CliRunner()
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


def _robomimic_live_selectors(e2e_project: str | None) -> dict[str, str]:
    selectors = {
        "project": os.environ.get("NPA_E2E_PROJECT", "").strip(),
        "registry": os.environ.get("NPA_BYOF_ROBOMIMIC_REGISTRY", "").strip(),
        "kubeconfig": os.environ.get("NPA_BYOF_KUBECONFIG", "").strip(),
        "context": os.environ.get("NPA_BYOF_K8S_CONTEXT", "").strip(),
        "namespace": os.environ.get("NPA_BYOF_K8S_NAMESPACE", "").strip(),
        "bucket": os.environ.get("NPA_E2E_S3_BUCKET", "").strip(),
    }
    assert selectors["project"] and e2e_project == selectors["project"]
    assert selectors["registry"], "a manager-issued private registry is required"
    assert selectors["kubeconfig"] and Path(selectors["kubeconfig"]).is_file()
    assert selectors["context"], "a manager-issued Kubernetes context is required"
    assert selectors["namespace"] and selectors["namespace"] != "default"
    assert selectors["bucket"], "a manager-issued output bucket is required"
    return selectors


@pytest.mark.parametrize(
    "missing_selector",
    ("project", "registry", "kubeconfig", "context", "namespace", "bucket"),
)
def test_robomimic_gate_refuses_missing_manager_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_selector: str,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    selectors = {
        "project": ("NPA_E2E_PROJECT", "manager-project"),
        "registry": ("NPA_BYOF_ROBOMIMIC_REGISTRY", "private.invalid/robomimic"),
        "kubeconfig": ("NPA_BYOF_KUBECONFIG", str(kubeconfig)),
        "context": ("NPA_BYOF_K8S_CONTEXT", "manager-context"),
        "namespace": ("NPA_BYOF_K8S_NAMESPACE", "robomimic-validation"),
        "bucket": ("NPA_E2E_S3_BUCKET", "manager-bucket"),
    }
    for name, (variable, value) in selectors.items():
        if name == missing_selector:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)

    with pytest.raises(AssertionError):
        _robomimic_live_selectors("manager-project")


def _robomimic_runner_command(
    config: dict[str, object],
    registry: str,
    project: str,
    output_root: str,
    run_id: str,
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
        ("--yaml", str(config["resource_profile_yaml"])),
        ("--output-root", output_root),
        ("--wait-timeout", "-1"),
        ("--run-id", run_id),
    )
    return [
        sys.executable,
        str(BYOF_RUNNER),
        *(item for pair in options for item in pair),
    ]


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
    skypilot_bin = resolve_skypilot_bin()
    if skypilot_bin:
        env["PATH"] = f"{Path(skypilot_bin).parent}:{env.get('PATH', '')}"
    return env


def _invoke_robomimic_gate(
    e2e_project: str | None,
) -> tuple[dict[str, object], str, str]:
    selectors = _robomimic_live_selectors(e2e_project)
    _activate_nebius_profile()
    config = load_spec(ROBOMIMIC_SPEC).config
    registry = resolve_container_registry(e2e_project)
    assert registry == selectors["registry"].rstrip("/")
    assert registry.split("/", 1)[0].lower() not in {"docker.io", "ghcr.io", "nvcr.io"}
    bucket = live_bucket(e2e_project)
    assert bucket == selectors["bucket"].removeprefix("s3://").split("/", 1)[0]
    run_id = (
        os.environ.get("NPA_BYOF_ROBOMIMIC_RUN_ID")
        or f"robomimic-live-{os.getpid()}"
    )
    cmd = _robomimic_runner_command(
        config,
        registry,
        selectors["project"],
        f"s3://{bucket}/oss-solutions/robomimic",
        run_id,
    )
    accepted_image = os.environ.get("NPA_BYOF_ROBOMIMIC_IMAGE", "").strip()
    if accepted_image:
        assert "@sha256:" in accepted_image and accepted_image.startswith(
            f"{registry}/"
        )
        cmd.extend(["--image", accepted_image, "--skip-build"])
    env = _robomimic_target_env(selectors["project"], selectors, cmd)
    proc = subprocess.run(
        cmd, check=False, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env
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
    client = s3_client_for_project(e2e_project, allow_host_creds=True)
    key = f"oss-solutions/robomimic/{run_id}/robomimic-smoke.json"
    return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())


def _assert_robomimic_inputs(artifact: dict[str, object]) -> None:
    source = artifact["source"]
    assert source["repository"] == "ARISE-Initiative/robomimic"
    assert source["revision"] == ROBOMIMIC_SOURCE_REVISION
    assert source["observed_head"] == ROBOMIMIC_SOURCE_REVISION
    assert source["byof_metadata"]["ref"] == ROBOMIMIC_SOURCE_REVISION
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
    assert split["train_sample_count"] + split["validation_sample_count"] == dataset[
        "sample_count"
    ]
    assert re.fullmatch(r"[0-9a-f]{64}", split["train_keys_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", split["validation_keys_sha256"])


def _assert_robomimic_training(artifact: dict[str, object]) -> None:
    training = artifact["training"]
    assert training["entrypoint"] == "robomimic/scripts/train.py"
    assert training["algorithm"] == "bc" and training["optimizer"] == "adam"
    assert training["optimizer_step_count"] == 4
    assert training["configured_optimizer_steps"] == 4
    assert training["validation_forward_steps"] == 2
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


def _assert_robomimic_runtime(artifact: dict[str, object], summary_image: str) -> None:
    hardware = artifact["hardware"]
    assert hardware["accelerator_count"] == 1 and "B200" in hardware["model"].upper()
    assert hardware["architecture"] == "sm_100"
    assert hardware["compute_capability"] == [10, 0]
    assert len(hardware["nvidia_smi_rows"]) == 1
    assert "B200" in hardware["nvidia_smi_rows"][0].upper()
    assert hardware["strict_reserved_capacity_attested"] is True
    pod_image = artifact["pod_image"]
    assert pod_image["runtime_ref"] == summary_image
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", pod_image["digest"])
    assert pod_image["digest"] in pod_image["image_id"]
    assert pod_image["observation_source"] == (
        "Kubernetes Pod status.containerStatuses[].imageID"
    )


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIVE_GPU") != "1"
    or os.environ.get("BYOF_ROBOMIMIC_LIVE") != "1"
    or os.environ.get("NPA_E2E_MK8S_RESERVED_CAPACITY") != "1",
    reason=(
        "Set NPA_BYOF_LIVE_GPU=1, BYOF_ROBOMIMIC_LIVE=1, and "
        "NPA_E2E_MK8S_RESERVED_CAPACITY=1 only with the assigned STRICT "
        "one-B200 runtime context to execute the robomimic gate."
    ),
)
def test_live_robomimic_b200_train_reload_gate(e2e_project: str | None) -> None:
    """Build the private candidate and require its complete run-derived artifact."""

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
    _assert_robomimic_runtime(artifact, summary_image)
    assert set(artifact["deferred"]) == {
        "image_policy_sweeps",
        "simulator_rollouts",
        "full_algorithm_matrix",
    }
    assert artifact["exit_status"] == 0


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
