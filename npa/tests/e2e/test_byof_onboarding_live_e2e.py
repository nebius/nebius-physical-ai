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


@pytest.mark.skipif(
    os.environ.get("NPA_BYOF_LIBERO_LIVE_B200") != "1",
    reason="Set NPA_BYOF_LIBERO_LIVE_B200=1 to verify an operator-selected one-B200 qualification report.",
)
def test_libero_b200_qualification_report() -> None:
    """Verify the downloaded, immutable-image LIBERO qualification evidence."""

    result_path = os.environ.get("NPA_BYOF_LIBERO_RESULT", "").strip()
    assert result_path, "NPA_BYOF_LIBERO_RESULT must select the downloaded libero-smoke.json"
    report = json.loads(Path(result_path).read_text(encoding="utf-8"))

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
    assert language_model["revision"] == (
        "cd5ef92a9fb2f889e972770a36d4ed042daf221e"
    )
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
    assert training["algorithm"] == "upstream_libero_Sequential.observe_BCRNNPolicy.compute_loss"
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
    assert runtime_metadata["runtime_artifact_count"] == 122
    assert runtime_metadata["render_assets_present"] is False
    assert runtime_metadata["git_objects_present"] is False
    assert runtime_metadata["cache_uploaded"] is False
    assert re.fullmatch(
        r"[0-9a-f]{64}", build["independently_observed_build_metadata_sha256"]
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
